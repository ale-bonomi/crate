"""Slicing drum hits out of loops: cuts, choices, fills, and the trip through the journal."""

from pathlib import Path

import numpy as np
import pytest

from crate import db, mover, planner, slicer
from crate.analyze.engine import annotate
from crate.config import Config, SliceConfig
from crate.scan import scan
from tests.synth import kick, snare, write_wav

SR = slicer.SR


def hat() -> np.ndarray:
    rng = np.random.default_rng(5)
    n = int(SR * 0.05)
    noise = np.diff(rng.uniform(-1, 1, n + 1)).astype(np.float32)  # high-passed noise
    return noise * np.exp(-np.arange(n) / SR * 90) * 0.6


def snare_with_body() -> np.ndarray:
    noise = snare(SR) * 0.7
    t = np.arange(len(noise)) / SR
    return noise + 0.5 * np.sin(2 * np.pi * 200 * t) * np.exp(-t * 25)


SOUNDS = {"kick": kick(SR), "snare": snare_with_body().astype(np.float32), "hat": hat()}
# Eighth notes at 120 BPM: kick hat snare hat, twice, then a quick four-snare fill.
PATTERN = [(0.0, "kick"), (0.25, "hat"), (0.5, "snare"), (0.75, "hat"),
           (1.0, "kick"), (1.25, "hat"), (1.5, "snare"), (1.75, "hat"),
           (2.5, "snare"), (2.62, "snare"), (2.74, "snare"), (2.86, "snare")]  # fmt: skip


def loop() -> np.ndarray:
    out = np.zeros(int(SR * 3.6), dtype=np.float32)
    for at, name in PATTERN:
        s = SOUNDS[name]
        i = int(at * SR)
        out[i : i + len(s)] += s[: len(out) - i]
    return out


def passthrough(stereo: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The whole mix is drums, nothing else."""
    mono = stereo.mean(0)
    return mono, np.zeros_like(mono)


def silent(stereo: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """No drums at all, as for a string chop."""
    mono = stereo.mean(0)
    return np.zeros_like(mono), mono


def by_spectrum(hit: np.ndarray) -> tuple[str, float]:
    """A stand-in for the trained model: deep lows = kick, a body tone = snare, else hat."""
    head = hit[: int(0.05 * SR)]
    spec = np.abs(np.fft.rfft(head)) ** 2
    freqs = np.fft.rfftfreq(len(head), 1 / SR)
    total = float(spec.sum()) + 1e-12
    if float(spec[freqs < 180].sum()) / total > 0.5:
        return "kick", 0.9
    body = float(spec[(freqs > 150) & (freqs < 1000)].sum()) / total
    return ("snare", 0.8) if body > 0.05 else ("hat", 0.8)


def stereo() -> np.ndarray:
    x = loop()
    return np.vstack([x, x])


def test_finds_every_hit_near_its_time():
    found = np.array(slicer.onsets(loop())) / SR
    want = np.array([at for at, _ in PATTERN])
    assert len(found) == len(want)
    assert np.all(np.abs(found - want) < 0.02)


def test_cuts_stop_at_the_next_hit_but_snares_keep_their_body():
    result = slicer.slice_mix(stereo(), passthrough, by_spectrum, SliceConfig(keep_per_kind=0))
    kinds = [h.kind for h in result.hits]
    assert kinds.count("kick") == 2 and kinds.count("hat") == 4
    kicks = [h for h in result.hits if h.kind == "kick"]
    # the hat 0.25 s later is not inside the kick
    assert all(len(h.audio) / SR <= 0.26 for h in kicks)
    snares = [h for h in result.hits if h.kind == "snare"]
    assert all(len(h.audio) / SR >= slicer.MIN_LEN["snare"] - 0.01 for h in snares)
    # fades in and out instead of clicking
    assert all(abs(h.audio[0]) < 1e-3 and abs(h.audio[-1]) < 1e-3 for h in result.hits)


def test_keeps_the_cleanest_few_of_each_kind():
    result = slicer.slice_mix(stereo(), passthrough, by_spectrum, SliceConfig(keep_per_kind=1))
    assert sorted(h.kind for h in result.hits) == ["hat", "kick", "snare"]


def test_a_quick_run_of_hits_is_also_kept_as_a_fill():
    result = slicer.slice_mix(stereo(), passthrough, by_spectrum, SliceConfig())
    assert len(result.fills) == 1
    fill = result.fills[0]
    assert abs(fill.start / SR - 2.5) < 0.02 and len(fill.audio) / SR > 0.36
    off = slicer.slice_mix(stereo(), passthrough, by_spectrum, SliceConfig(fills=False))
    assert off.fills == []


def test_files_without_drums_are_left_alone():
    result = slicer.slice_mix(stereo(), silent, by_spectrum, SliceConfig())
    assert result.skipped == "barely any drums" and result.hits == []


def test_clap_and_rim_from_the_model_become_snare_and_perc():
    class Model:
        classes = ["clap", "kick", "fx"]

        def probabilities(self, vector, duration):
            return np.array([0.5, 0.1, 0.4])

    label = slicer.model_labeler(Model(), lambda hit, sr: np.ones(4))
    kind, prob = label(np.zeros(SR // 4, dtype=np.float32))
    assert kind == "snare" and prob == pytest.approx(0.5 / 0.6)  # fx is not a drum


def test_without_demucs_nothing_is_sliced(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(slicer, "installed", lambda: False)
    assert slicer.demucs_separator() is None
    conn = db.connect(tmp_path / "t.db")
    notes = slicer.add_slices([], {}, conn, tmp_path, SliceConfig())
    assert any("uv sync --extra ml --extra slice" in n for n in notes)


def test_slices_are_planned_filed_and_undone(tmp_path: Path):
    lib = tmp_path / "lib"
    write_wav(tmp_path / "in" / "Chop From A Record.wav", loop(), SR)
    staging = tmp_path / "staging"
    res = scan([tmp_path / "in"], staging)
    conn = db.connect(tmp_path / "t.db")
    verdicts, _ = annotate(res.items, conn, {}, jobs=1, use_clap=False)
    notes = slicer.add_slices(
        res.items, verdicts, conn, staging, SliceConfig(), separate=passthrough, label=by_spectrum
    )
    assert any("sliced" in n for n in notes)
    slices = [it for it in res.items if it.staged]
    assert slices and all(v.source == "slice" for it in slices if (v := verdicts[id(it)]))
    cfg = Config(library_roots=[lib], keep_originals=False)
    plan = planner.build_plan(
        res.items, cfg, planner.sort_router(cfg, verdicts), [], verdicts, conn
    )
    folders = {o.dst.parent.relative_to(lib).parts[:2] for o in plan.ops if o.mode == "extract"}
    assert {("One Shots", "Kicks"), ("One Shots", "Snares"), ("Loops", "Drum Fills")} <= folders
    assert ("One Shots", "Hats & Cymbals") in folders
    names = sorted(o.dst.name for o in plan.ops if o.mode == "extract")
    assert "Chop From A Record kick 01.wav" in names

    batch = mover.execute(conn, plan, "ingest", lib)
    filed = [
        p
        for p in lib.rglob("*.wav")
        if "Chop From A Record " in p.name and p.parent.name != "Drums"
    ]
    assert len(filed) == sum(o.mode == "extract" for o in plan.ops)
    mover.undo(conn, batch, trash=lambda p: Path(p).unlink())
    assert not any(p.name.endswith(" 01.wav") for p in lib.rglob("*.wav"))
    assert (tmp_path / "in" / "Chop From A Record.wav").exists()  # the source is back too
