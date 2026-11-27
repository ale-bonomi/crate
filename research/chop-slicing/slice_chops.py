"""Prototype: separate the drums out of song chops, slice the hits, classify them.

Research code, not part of the crate package. See README.md in this folder.

usage: slice_chops.py <chops dir> <crate home with .crate/model.npz> <out dir> <n chops>
"""

from __future__ import annotations

import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model

from crate import db, learn
from crate.analyze.audio import AudioData
from crate.analyze.clap import ClapTagger
from crate.analyze.features import extract, type_guess

SR = 44100
HOP, WIN = 256, 1024
# Round 2 (from listening): let hits ring. A hit lasts at least MIN_LEN for its kind,
# even past the next hit, then until it fades 35 dB below its peak, at most MAX_LEN.
MIN_LEN = {"kick": 0.35, "snare": 0.5, "clap": 0.5, "rim": 0.3, "hat": 0.18, "perc": 0.3}
MAX_LEN = 1.2
FADE_DB = 35.0
KEEP_PER_CLASS = 2
DRUMS = ("kick", "snare", "clap", "hat", "perc")


def load_stereo(path: Path) -> np.ndarray:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    data = data.T[:2]
    if data.shape[0] == 1:
        data = np.vstack([data, data])
    if sr != SR:
        n = int(round(data.shape[1] * SR / sr))
        x = np.linspace(0, 1, data.shape[1], endpoint=False)
        xn = np.linspace(0, 1, n, endpoint=False)
        data = np.vstack([np.interp(xn, x, ch) for ch in data]).astype(np.float32)
    return data[:, : SR * 30]  # the first 30 s is plenty


def flux_onsets(x: np.ndarray) -> list[int]:
    """Spectral-flux onsets (sample positions) on a mono signal."""
    frames = 1 + max(0, len(x) - WIN) // HOP
    win = np.hanning(WIN).astype(np.float32)
    mags = np.stack([np.abs(np.fft.rfft(x[i * HOP : i * HOP + WIN] * win)) for i in range(frames)])
    flux = np.maximum(0, np.diff(np.log1p(mags), axis=0)).sum(1)
    if flux.max() <= 0:
        return []
    med = np.median(flux)
    mad = np.median(np.abs(flux - med)) + 1e-9
    thresh = max(med + 6 * mad, 0.15 * flux.max())
    out, last = [], -10_000
    for i in range(1, len(flux) - 1):
        if flux[i] >= thresh and flux[i] >= flux[i - 1] and flux[i] > flux[i + 1]:
            pos = (i + 1) * HOP
            if pos - last >= int(0.07 * SR):
                out.append(pos)
                last = pos
    return out


def envelope(x: np.ndarray) -> tuple[np.ndarray, int]:
    hop = int(0.005 * SR)
    n = len(x) // hop
    return np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1)), hop


def ring_end(env: np.ndarray, hop: int, start: int, next_on: int | None, kind: str) -> int:
    """Where a hit stops: past MIN_LEN, at the next hit or once it has faded; never past MAX_LEN."""
    i0 = start // hop
    peak_i = i0 + int(np.argmax(env[i0 : i0 + int(0.05 * SR / hop) + 1]))
    floor = env[peak_i] * 10 ** (-FADE_DB / 20)
    min_end = start + int(MIN_LEN.get(kind, 0.3) * SR)
    cap = start + int(MAX_LEN * SR)
    j = peak_i
    while j < len(env) and j * hop < cap and env[j] > floor:
        j += 1
    faded = j * hop
    end = min(faded, next_on) if next_on is not None and next_on > min_end else faded
    return int(min(cap, max(min_end, end), len(env) * hop))


def purity(hit: np.ndarray, kind: str) -> float:
    """1 = only this drum sounds. Kicks lose points for cymbal/hat highs, hats and
    snares for kick lows, in the first 80 ms where the attack is."""
    head = hit[: int(0.08 * SR)]
    spec = np.abs(np.fft.rfft(head * np.hanning(len(head)))) ** 2
    freqs = np.fft.rfftfreq(len(head), 1 / SR)
    total = spec.sum() + 1e-12
    low = spec[freqs < 120].sum() / total
    high = spec[freqs > 6000].sum() / total
    if kind == "kick":
        return float(1 - min(1.0, 4 * high))
    if kind == "hat":
        return float(1 - min(1.0, 3 * low))
    return float(1 - min(1.0, 2 * low))


def cut(x: np.ndarray, start: int, end: int) -> np.ndarray:
    start = max(0, start - int(0.004 * SR))
    hit = x[start:end].copy()
    fi, fo = int(0.002 * SR), min(int(0.06 * SR), len(hit) // 3)
    hit[:fi] *= np.linspace(0, 1, fi)
    if fo:
        hit[-fo:] *= np.linspace(1, 0, fo)
    return hit


def main() -> None:
    src, home, out, n = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
    out.mkdir(parents=True, exist_ok=True)
    (out / "hits").mkdir(exist_ok=True)
    model = get_model("htdemucs").eval()
    drums_idx = model.sources.index("drums")
    clf = learn.load_model(True, home / ".crate/model.npz")
    assert clf is not None
    tagger = ClapTagger({})
    conn = db.connect(out / "proto.db")

    files = sorted(src.glob("*.wav"))
    random.seed(3)
    phrases = [p for p in files if 4 <= sf.info(str(p)).duration <= 20]
    chosen = random.sample(phrases, n)

    report, kept_all = [], []
    t0 = time.time()
    for k, path in enumerate(chosen, 1):
        mix = load_stereo(path)
        with torch.no_grad():
            stems = apply_model(model, torch.from_numpy(mix)[None], device="cpu", shifts=0, split=True)[0]
        drums = stems[drums_idx].mean(0).numpy()
        rest = (stems.sum(0) - stems[drums_idx]).mean(0).numpy()
        mono = mix.mean(0)
        drum_share = float(np.sqrt(np.mean(drums**2)) / (np.sqrt(np.mean(mono**2)) + 1e-9))
        entry = {"chop": path.name, "drum_share": round(drum_share, 2), "hits": 0, "kept": []}
        if drum_share < 0.15:
            entry["note"] = "barely any drums"
            report.append(entry)
            print(f"[{k}/{n}] {path.name[:50]:50} drums {drum_share:.2f} -> skipped", flush=True)
            continue
        onsets = flux_onsets(drums)
        entry["hits"] = len(onsets)
        env, hop = envelope(drums)
        cands = []
        for i, on in enumerate(onsets):
            # Classify on a short window, then cut the real length for the kind it is.
            probe_end = min(onsets[i + 1] if i + 1 < len(onsets) else len(drums), on + int(0.4 * SR))
            if probe_end - on < int(0.05 * SR):
                continue
            hit = cut(drums, on, probe_end)
            bleed = float(np.sqrt(np.mean(rest[on:probe_end] ** 2)) / (np.sqrt(np.mean(hit**2)) + 1e-9))
            audio = AudioData(hit, SR, len(hit) / SR, f"{path.name}:{on}", f"{path.name}:{on}", None)
            emb = tagger.embed(hit, SR)
            if emb is None:
                continue
            # A hit from the drum stem is a drum: choose among the drum types only.
            probs = clf.probabilities(emb, len(hit) / SR)
            drum = [j for j, c in enumerate(clf.classes) if c in DRUMS]
            sub = probs[drum] / probs[drum].sum()
            j = drum[int(sub.argmax())]
            label, prob = clf.classes[j], float(sub.max())
            trusted = label in clf.trusted and prob >= clf.gate
            dsp, _ = type_guess(extract(hit, SR, len(hit) / SR))
            nxt = onsets[i + 1] if i + 1 < len(onsets) else None
            end = ring_end(env, hop, on, nxt, label)
            full = cut(drums, on, end)
            cands.append({"i": i, "on": on, "end": end, "label": label, "prob": prob, "trusted": trusted,
                          "dsp": dsp, "peak": float(np.abs(full).max()), "bleed": bleed,
                          "purity": purity(full, label), "secs": round(len(full) / SR, 2), "hit": full})  # fmt: skip
        by_class: dict[str, list] = {}
        for c in cands:
            if c["label"] in {"kick", "snare", "clap", "hat", "perc"}:
                by_class.setdefault(c["label"], []).append(c)
        for label, cs in by_class.items():
            # cleanest first: confident, little bleed from the rest of the mix, loud
            # Round 2: alone first (no kick under the hat, no hat over the kick), then confident, clean.
            cs.sort(key=lambda c: (-round(c["purity"], 1), -c["prob"], c["bleed"]))
            for c in cs[:KEEP_PER_CLASS]:
                name = f"{re.sub(r'[^A-Za-z0-9]+', '_', path.stem)[:40]}__{c['i']:03d}_{label}.wav"
                peak = c["peak"] or 1.0
                sf.write(str(out / "hits" / name), (c["hit"] / peak * 0.89).astype(np.float32), SR, subtype="PCM_16")
                info = {k2: (round(v, 3) if isinstance(v, float) else v) for k2, v in c.items() if k2 != "hit"}
                entry["kept"].append({"file": name, **info})
                kept_all.append({"chop": path.name, "file": name, **info})
        report.append(entry)
        dist = Counter(c["label"] for c in cands)
        print(f"[{k}/{n}] {path.name[:50]:50} drums {drum_share:.2f} hits {len(cands):3} {dict(dist)} kept {len(entry['kept'])}  ({time.time() - t0:.0f}s)", flush=True)

    (out / "report.json").write_text(json.dumps(report, indent=1))
    (out / "kept.json").write_text(json.dumps(kept_all, indent=1))
    with_drums = [r for r in report if r.get("note") != "barely any drums"]
    print(f"\n{len(report)} chops, {len(with_drums)} with drums, {sum(r['hits'] for r in report)} hits found, "
          f"{len(kept_all)} kept: {dict(Counter(h['label'] for h in kept_all))}")  # fmt: skip
    print("kept and trusted:", sum(h["trusted"] for h in kept_all), " agree with DSP:",
          sum(h["dsp"] == h["label"] for h in kept_all))  # fmt: skip
    print(f"time {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
