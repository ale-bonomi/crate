"""Slice drum hits out of loops and chops: opt-in with `ingest --slice` or `slice.enabled`.

For each audio file at least `min_source_seconds` long:

(only the first `max_source_seconds` of each):

1. Separate the drums from the rest of the mix (Demucs, `uv sync --extra ml --extra slice`).
   A file whose drums are quieter than `min_drum_share` of the mix is left alone.
2. Find the hits (spectral flux on the drum stem).
3. Label each hit as a kick, snare, hat or perc. The `crate train` model does this when
   there is one; otherwise a rough spectral guess, whose labels mostly go to review.
4. Cut each hit from its start to the next hit, so a kick does not carry the hat that
   follows it. A snare keeps at least 0.3 s even if the next hit comes sooner.
   Cuts end once the sound fades 35 dB below its peak, at most 1.2 s, with a short fade.
5. Keep the cleanest `keep_per_kind` hits of each kind per file (0 keeps every hit):
   the drum alone first, then confident, then little of the song underneath.
6. Quick runs of 3+ hits (gaps up to 0.2 s) are also saved whole as drum fills.

The slices are new files. They are planned and moved like any other file, recorded
in the journal, and removed by `crate undo`. The source file is filed as usual.
These choices come from listening tests on record chops (research/chop-slicing).
"""

from __future__ import annotations

import importlib
import importlib.util
import re
import sqlite3
import wave
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .analyze.audio import decode
from .analyze.features import characteristics, type_guess
from .analyze.types import Verdict
from .config import SliceConfig
from .scan import Item

SR = 44100
HOP, WIN = 256, 1024
KINDS = ("kick", "snare", "hat", "perc")
# Claps cut from records were snares (1 real clap in 120 judged hits); rims and toms are
# filed with percussion until the model learns them.
_FOLD = {"clap": "snare", "rim": "perc", "tom": "perc", "cymbal": "hat", "808": "kick"}
MIN_LEN = {"kick": 0.12, "snare": 0.3, "hat": 0.06, "perc": 0.1}
MAX_LEN = 1.2
FADE_DB = 35.0
FILL_GAP = 0.2
FILL_HITS = 3
FILL_MAX = 2.0
# Kick, snare and hat labels held up in listening tests (the drum was in the cut every
# time); perc was hit or miss, so it keeps the model's own confidence.
_RELIABLE = {"kick", "snare", "hat"}
MARK = " > "  # in a slice's `original`: "<source original> > kick 01"

Separator = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]
"""Stereo float32 (2, n) at SR -> (drums mono, everything else mono), both at SR."""
Labeler = Callable[[np.ndarray], tuple[str, float]]
"""A mono hit at SR -> (one of KINDS, probability)."""


@dataclass
class Hit:
    kind: str
    prob: float
    start: int
    audio: np.ndarray
    purity: float = 1.0
    bleed: float = 0.0


@dataclass
class Sliced:
    hits: list[Hit] = field(default_factory=list)
    fills: list[Hit] = field(default_factory=list)
    drum_share: float = 0.0
    skipped: str | None = None


# ------------------------------------------------------------------ separation / labels


def installed() -> bool:
    return importlib.util.find_spec("demucs") is not None


def demucs_separator() -> Separator | None:
    """Demucs htdemucs, or None when it is not installed. Weights download once."""
    if not installed():
        return None
    torch: Any = importlib.import_module("torch")
    pretrained: Any = importlib.import_module("demucs.pretrained")
    apply: Any = importlib.import_module("demucs.apply")
    model = pretrained.get_model("htdemucs")
    model.eval()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    drums_at = list(model.sources).index("drums")

    def separate(stereo: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            mix = torch.from_numpy(np.ascontiguousarray(stereo, dtype=np.float32))[None]
            stems = apply.apply_model(model, mix, device=device, shifts=0, split=True)[0].cpu()
        drums = stems[drums_at].mean(0).numpy()
        rest = (stems.sum(0) - stems[drums_at]).mean(0).numpy()
        return drums.astype(np.float32), rest.astype(np.float32)

    return separate


def model_labeler(model: Any, embed: Callable[[np.ndarray, int], np.ndarray | None]) -> Labeler:
    """The `crate train` model, choosing among drum types only: the hit came from drums."""

    def label(hit: np.ndarray) -> tuple[str, float]:
        vector = embed(hit, SR)
        if vector is None:
            return dsp_label(hit)
        probs = model.probabilities(vector, len(hit) / SR)
        scores: dict[str, float] = {}
        for name, p in zip(model.classes, probs, strict=True):
            kind = _FOLD.get(name, name)
            if kind in KINDS:
                scores[kind] = scores.get(kind, 0.0) + float(p)
        total = sum(scores.values())
        if total <= 0:
            return dsp_label(hit)
        kind = max(scores, key=lambda k: scores[k])
        return kind, scores[kind] / total

    return label


def dsp_label(hit: np.ndarray) -> tuple[str, float]:
    """Fallback without a trained model: Crate's spectral guess, folded onto drum kinds."""
    from .analyze.features import extract

    guess, conf = type_guess(extract(hit, SR, len(hit) / SR))
    kind = _FOLD.get(guess or "", guess or "perc")
    if kind not in KINDS:
        return "perc", 0.4
    return kind, conf


# ------------------------------------------------------------------ cutting


def onsets(drums: np.ndarray) -> list[int]:
    """Spectral-flux onsets (sample positions), at least 70 ms apart."""
    if not len(drums) or float(np.abs(drums).max()) <= 1e-4:
        return []
    # Silence in front, so a hit on the very first sample still rises out of something.
    drums = np.concatenate([np.zeros(WIN, dtype=np.float32), drums.astype(np.float32)])
    frames = 1 + (len(drums) - WIN) // HOP
    window = np.hanning(WIN).astype(np.float32)
    idx = np.arange(WIN)[None, :] + HOP * np.arange(frames)[:, None]
    mags = np.abs(np.fft.rfft(drums[idx] * window, axis=1))
    flux = np.maximum(0, np.diff(np.log1p(mags), axis=0)).sum(1)
    if flux.size == 0 or float(flux.max()) <= 0:
        return []
    med = float(np.median(flux))
    mad = float(np.median(np.abs(flux - med))) + 1e-9
    thresh = max(med + 6 * mad, 0.15 * float(flux.max()))
    found: list[int] = []
    last = -(10**9)
    for i in range(len(flux) - 1):
        before = flux[i - 1] if i else 0.0
        if flux[i] >= thresh and flux[i] >= before and flux[i] > flux[i + 1]:
            # The rise shows once the hit is about mid-window; aim a hop before that.
            pos = max(0, (i + 1) * HOP + WIN // 2 - HOP - WIN)
            if pos - last >= int(0.07 * SR):
                found.append(pos)
                last = pos
    return found


def _envelope(x: np.ndarray) -> tuple[np.ndarray, int]:
    hop = int(0.005 * SR)
    n = max(1, len(x) // hop)
    padded = np.zeros(n * hop, dtype=np.float32)
    padded[: min(len(x), n * hop)] = x[: n * hop]
    return np.sqrt(np.mean(padded.reshape(n, hop) ** 2, axis=1)), hop


def hit_end(env: np.ndarray, hop: int, start: int, next_on: int | None, kind: str) -> int:
    """Stop at the next hit (a snare keeps its minimum), or once faded, at most MAX_LEN."""
    i0 = min(start // hop, len(env) - 1)
    peak = i0 + int(np.argmax(env[i0 : i0 + int(0.05 * SR / hop) + 1]))
    floor = env[peak] * 10 ** (-FADE_DB / 20)
    cap = start + int(MAX_LEN * SR)
    j = peak
    while j < len(env) and j * hop < cap and env[j] > floor:
        j += 1
    end = j * hop
    if next_on is not None:
        end = min(end, next_on)
    end = max(end, start + int(MIN_LEN.get(kind, 0.1) * SR))
    return int(min(end, cap, len(env) * hop))


def _cut(x: np.ndarray, start: int, end: int) -> np.ndarray:
    hit = x[max(0, start - int(0.004 * SR)) : end].astype(np.float32).copy()
    fade_in = min(int(0.002 * SR), len(hit))
    fade_out = min(int(0.03 * SR), len(hit) // 3)
    if fade_in:
        hit[:fade_in] *= np.linspace(0, 1, fade_in, dtype=np.float32)
    if fade_out:
        hit[-fade_out:] *= np.linspace(1, 0, fade_out, dtype=np.float32)
    return hit


def purity(hit: np.ndarray, kind: str) -> float:
    """1 = the drum alone: no hat highs over a kick, no kick lows under a hat or snare."""
    head = hit[: int(0.08 * SR)]
    if len(head) < 64:
        return 1.0
    spec = np.abs(np.fft.rfft(head * np.hanning(len(head)))) ** 2
    freqs = np.fft.rfftfreq(len(head), 1 / SR)
    total = float(spec.sum()) + 1e-12
    low = float(spec[freqs < 120].sum()) / total
    high = float(spec[freqs > 6000].sum()) / total
    if kind == "kick":
        return 1.0 - min(1.0, 4 * high)
    if kind == "hat":
        return 1.0 - min(1.0, 3 * low)
    return 1.0 - min(1.0, 2 * low)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2))) if len(x) else 0.0


def _runs(starts: list[int]) -> list[tuple[int, int]]:
    """Index ranges [i, j] of FILL_HITS+ onsets whose gaps are all <= FILL_GAP."""
    runs, i = [], 0
    while i < len(starts):
        j = i
        while j + 1 < len(starts) and starts[j + 1] - starts[j] <= FILL_GAP * SR:
            j += 1
        if j - i + 1 >= FILL_HITS and starts[j] - starts[i] <= FILL_MAX * SR:
            runs.append((i, j))
        i = j + 1
    return runs


def slice_mix(stereo: np.ndarray, separate: Separator, label: Labeler, cfg: SliceConfig) -> Sliced:
    """Hits and fills from one stereo mix at SR."""
    out = Sliced()
    drums, rest = separate(stereo)
    mono = stereo.mean(0)
    out.drum_share = _rms(drums) / (_rms(mono) + 1e-9)
    if out.drum_share < cfg.min_drum_share:
        out.skipped = "barely any drums"
        return out
    starts = onsets(drums)
    env, hop = _envelope(drums)
    cands: list[Hit] = []
    for i, on in enumerate(starts):
        nxt = starts[i + 1] if i + 1 < len(starts) else None
        probe = _cut(drums, on, min(nxt or len(drums), on + int(0.4 * SR)))
        if len(probe) < int(0.03 * SR):
            continue
        kind, prob = label(probe)
        end = hit_end(env, hop, on, nxt, kind)
        audio = _cut(drums, on, end)
        bleed = _rms(rest[on:end]) / (_rms(audio) + 1e-9)
        cands.append(Hit(kind, prob, on, audio, purity(audio, kind), bleed))
    for kind in KINDS:
        mine = [h for h in cands if h.kind == kind]
        mine.sort(key=lambda h: (-round(h.purity, 1), -h.prob, h.bleed))
        out.hits += mine if cfg.keep_per_kind == 0 else mine[: cfg.keep_per_kind]
    out.hits.sort(key=lambda h: h.start)
    if cfg.fills:
        found = []
        for i, j in _runs(starts):
            nxt = starts[j + 1] if j + 1 < len(starts) else None
            end = hit_end(env, hop, starts[j], nxt, "snare")
            audio = _cut(drums, starts[i], end)
            found.append(Hit("fill", 0.9, starts[i], audio, 1.0, _rms(rest[starts[i] : end])))
        found.sort(key=lambda h: h.bleed)
        out.fills = sorted(found[: max(1, cfg.keep_per_kind or len(found))], key=lambda h: h.start)
    return out


# ------------------------------------------------------------------ files and items


def _to_stereo(samples: np.ndarray, sr: int) -> np.ndarray:
    x = np.asarray(samples, dtype=np.float32)
    if sr != SR and len(x):
        n = max(1, int(round(len(x) * SR / sr)))
        grid = np.linspace(0, 1, len(x), endpoint=False)
        x = np.interp(np.linspace(0, 1, n, endpoint=False), grid, x)
    x = x.astype(np.float32)
    return np.vstack([x, x])


def _write(path: Path, audio: np.ndarray) -> None:
    peak = float(np.abs(audio).max()) or 1.0
    pcm = (np.clip(audio / peak * 0.89, -1, 1) * 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SR)
        handle.writeframes(pcm.tobytes())


def _safe(text: str) -> str:
    return re.sub(r"[^\w\-. ]+", "", text).strip()[:60] or "slice"


@dataclass
class Made:
    item: Item
    kind: str
    prob: float
    source: Item


def slice_items(
    items: list[Item],
    verdicts: dict[int, Verdict],
    out_dir: Path,
    separate: Separator,
    label: Labeler,
    cfg: SliceConfig,
    progress: Callable[[], None] | None = None,
) -> tuple[list[Made], list[str]]:
    """Slice every long enough audio item. Returns the new staged items and notes."""
    made: list[Made] = []
    notes: list[str] = []
    sources = [
        it
        for it in items
        if it.kind == "audio"
        and (v := verdicts.get(id(it))) is not None
        and v.features is not None
        and v.features.duration >= cfg.min_source_seconds
    ]
    for n, src in enumerate(sources):
        if progress is not None:
            progress()
        decoded = decode(src.path)
        if decoded is None:
            continue
        mono, sr = decoded
        mono = mono[: int(sr * cfg.max_source_seconds)]
        result = slice_mix(_to_stereo(mono, sr), separate, label, cfg)
        if result.skipped:
            continue
        counts: dict[str, int] = {}
        for hit in [*result.hits, *result.fills]:
            counts[hit.kind] = counts.get(hit.kind, 0) + 1
            name = f"{_safe(src.path.stem)} {hit.kind} {counts[hit.kind]:02d}.wav"
            path = out_dir / f"s{n}" / name
            _write(path, hit.audio)
            item = Item(
                path=path,
                original=f"{src.original}{MARK}{hit.kind} {counts[hit.kind]:02d}",
                pack=src.pack,
                folders=src.folders,
                kind="audio",
                staged=True,
                root=src.root,
            )
            made.append(Made(item, hit.kind, hit.prob, src))
        hits = len(result.hits) + len(result.fills)
        if hits:
            notes.append(f"sliced {hits} from {src.path.name}")
    return made, notes


def verdict_for(made: Made, base: Verdict) -> Verdict:
    """The slicer's label over the analysis of the new file."""
    fill = made.kind == "fill"
    if fill:
        conf = 0.9
    elif made.kind in _RELIABLE:
        conf = max(made.prob, 0.65)
    else:
        conf = made.prob
    punch = snare = feel = wet = None
    if base.features is not None:
        punch, snare, feel, wet = characteristics(base.features, made.kind, fill)
    return Verdict(
        type=made.kind,
        loop=fill,
        confidence=round(float(conf), 2),
        reason=f"sliced from {made.source.path.name}: {made.kind} {made.prob:.2f}",
        bpm=None,
        key=None,
        punch=punch,
        snare_body=snare,
        feel=feel,
        wet=wet,
        source="slice",
        content_hash=base.content_hash,
        fingerprint=base.fingerprint,
        raw_hash=base.raw_hash,
        features=base.features,
    )


def add_slices(
    items: list[Item],
    verdicts: dict[int, Verdict],
    conn: sqlite3.Connection,
    staging: Path,
    cfg: SliceConfig,
    *,
    separate: Separator | None = None,
    label: Labeler | None = None,
    model: Any = None,
    progress: Callable[[], None] | None = None,
) -> list[str]:
    """Slice `items` in place: new staged items and their verdicts are appended.

    Returns warnings and notes for the plan. Without Demucs, nothing is sliced."""
    from .analyze.clap import ClapTagger
    from .analyze.engine import annotate

    warnings: list[str] = []
    if separate is None:
        separate = demucs_separator()
        if separate is None:
            return ["Slicing needs Demucs. Install it with: uv sync --extra ml --extra slice"]
    if label is None:
        tagger = ClapTagger({})
        if model is not None and tagger.can_embed():
            label = model_labeler(model, tagger.embed)
        else:
            label = dsp_label
            warnings.append(
                "Slices are labelled by a rough guess, so many go to _Review/. "
                "Run `crate train` for better labels."
            )
    made, notes = slice_items(
        items, verdicts, staging / "slices", separate, label, cfg, progress=progress
    )
    if not made:
        return warnings + ["Nothing to slice: no file had enough drums."]
    new_items = [m.item for m in made]
    new_verdicts, more = annotate(new_items, conn, {}, jobs=1, use_clap=False)
    warnings += [w for w in more if "CLAP" not in w]
    for m in made:
        base = new_verdicts.get(id(m.item))
        if base is not None:
            verdicts[id(m.item)] = verdict_for(m, base)
    items.extend(new_items)
    return warnings + notes
