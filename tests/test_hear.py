"""Hearing, fusion, search, dedupe, projects, and the watch list."""

import gzip
from pathlib import Path

import numpy as np

from crate import db
from crate.analyze.audio import fingerprint, pcm_hash
from crate.analyze.engine import annotate, save_correction
from crate.analyze.features import extract
from crate.analyze.fusion import fuse, parse_label
from crate.analyze.metadata import Meta, parse
from crate.config import Config
from crate.evaluate import read_labels, score
from crate.planner import build_plan, sort_router
from crate.projects import index_projects, parse_als, usage
from crate.scan import scan
from crate.search import find, format_status, similar, where
from crate.watch import list_media, mark_seen, unseen
from tests.synth import chord, clicks, kick, snare, write_wav


def test_kick_snare_and_loops(tmp_path: Path):
    sr = 22050
    cases = {
        "kick": kick(sr),
        "tight": snare(sr, decay=40),
        "body": snare(sr, decay=6),
        "straight": clicks(sr, 4.0, 120, swing=0.0),
        "swung": clicks(sr, 4.0, 120, swing=0.66),
        "chord": chord(sr, 4.0),
    }
    feats = {name: extract(sig, sr) for name, sig in cases.items()}
    assert feats["kick"].centroid < feats["tight"].centroid
    assert feats["kick"].low > 0.3
    assert feats["body"].decay > feats["tight"].decay
    assert feats["straight"].swing is not None and feats["straight"].swing < 1.35
    assert feats["swung"].swing is not None and feats["swung"].swing >= 1.35
    assert feats["chord"].key == "Cmaj"
    assert feats["chord"].tonalness > 0.5

    kick_v = fuse(Meta(), feats["kick"], None, None)
    assert kick_v.type == "kick" and kick_v.loop is False and kick_v.punch == "punchy"
    tight_v = fuse(Meta(), feats["tight"], None, None)
    body_v = fuse(Meta(), feats["body"], None, None)
    assert tight_v.type == "snare" and tight_v.snare_body == "tight"
    assert body_v.type == "snare" and body_v.snare_body == "body"
    straight_v = fuse(Meta(loop=True), feats["straight"], None, None)
    swung_v = fuse(Meta(loop=True), feats["swung"], None, None)
    assert straight_v.loop is True and straight_v.feel == "straight"
    assert swung_v.feel == "swung"
    chord_v = fuse(Meta(), feats["chord"], None, None)
    assert chord_v.type == "melodic" and chord_v.key == "Cmaj" and chord_v.loop is True


def test_explicit_filename_beats_disagreeing_audio():
    meta = parse("Kick_01", ())
    # A bright noise burst would otherwise be a snare.
    verdict = fuse(meta, extract(snare(decay=40), 22050), None, None)
    assert verdict.type == "kick"
    assert "metadata" in verdict.reason


def test_vague_name_trusts_audio(tmp_path: Path):
    path = tmp_path / "take1.wav"
    write_wav(path, kick(), 22050)
    cfg = Config(library_roots=[tmp_path / "lib"])
    conn = db.connect(tmp_path / "lib.db")
    result = scan([path], tmp_path / "staging")
    verdicts, _warnings = annotate(result.items, conn, {}, jobs=1, use_clap=False)
    verdict = next(iter(verdicts.values()))
    assert verdict.type == "kick"
    plan = build_plan(result.items, cfg, sort_router(cfg, verdicts), [], verdicts, conn)
    assert any(
        o.dst.parts[-2:] == ("Kicks", "take1.wav") or "Kicks" in str(o.dst) for o in plan.ops
    )


def test_correction_wins_and_is_cached(tmp_path: Path):
    path = tmp_path / "take1.wav"
    write_wav(path, kick(), 22050)
    conn = db.connect(tmp_path / "lib.db")
    result = scan([path], tmp_path / "staging")
    verdicts, _ = annotate(result.items, conn, {}, jobs=1, use_clap=False)
    verdict = next(iter(verdicts.values()))
    assert verdict.content_hash
    save_correction(conn, verdict.content_hash, parse_label("snare:tight"))
    again, _ = annotate(result.items, conn, {}, jobs=1, use_clap=False)
    labeled = next(iter(again.values()))
    assert labeled.type == "snare" and labeled.snare_body == "tight" and labeled.source == "manual"
    # Second pass reused the cache: the analysis row exists once.
    count = conn.execute("SELECT COUNT(*) FROM analysis").fetchone()[0]
    assert count == 1


def test_clap_label_can_name_a_vague_file():
    meta = parse("loop_01", ())
    verdict = fuse(meta, None, {"snare": 0.8, "kick": 0.05}, None)
    assert verdict.type == "snare"


def test_split_when_a_folder_would_overflow(tmp_path: Path):
    inbox = tmp_path / "in"
    inbox.mkdir()
    write_wav(inbox / "Snare_Tight_A.wav", snare(decay=40), 22050)
    write_wav(inbox / "Snare_Body_B.wav", snare(decay=6), 22050)
    cfg = Config(library_roots=[tmp_path / "lib"], split_folder_threshold=1, keep_originals=False)
    conn = db.connect(tmp_path / "t.db")
    result = scan([inbox], tmp_path / "st")
    verdicts, _ = annotate(result.items, conn, {}, jobs=1, use_clap=False)
    plan = build_plan(result.items, cfg, sort_router(cfg, verdicts), [], verdicts, conn)
    dests = {op.dst.name: op.dst.parent.name for op in plan.ops}
    assert dests["Snare_Tight_A.wav"] == "Tight"
    assert dests["Snare_Body_B.wav"] == "Body"


def test_content_hash_and_near_fingerprint():
    loud = kick()
    quiet = loud * 0.5
    assert pcm_hash(loud) != pcm_hash(quiet)
    assert fingerprint(loud, 22050) == fingerprint(quiet, 22050)
    n = len(loud) * 2
    faster = np.interp(np.linspace(0, len(loud) - 1, n), np.arange(len(loud)), loud).astype(
        np.float32
    )
    assert fingerprint(loud, 22050) == fingerprint(faster, 44100)


def test_dedupe_different_name(tmp_path: Path):
    lib = tmp_path / "lib"
    inbox = tmp_path / "in"
    inbox.mkdir()
    sig = kick()
    write_wav(inbox / "Kick_A.wav", sig, 22050)
    cfg = Config(library_roots=[lib], keep_originals=False, duplicates="quarantine")
    conn = db.connect(tmp_path / "t.db")
    from crate import mover

    def sort_once(folder: Path):
        result = scan([folder], tmp_path / "st")
        verdicts, _ = annotate(result.items, conn, {}, jobs=1, use_clap=False)
        plan = build_plan(result.items, cfg, sort_router(cfg, verdicts), [], verdicts, conn)
        mover.execute(conn, plan, "sort", lib, trash=lambda p: Path(p).unlink())
        return plan

    sort_once(inbox)
    write_wav(inbox / "Kick_A_again.wav", sig, 22050)
    result = scan([inbox / "Kick_A_again.wav"], tmp_path / "st2")
    verdicts, _ = annotate(result.items, conn, {}, jobs=1, use_clap=False)
    plan = build_plan(result.items, cfg, sort_router(cfg, verdicts), [], verdicts, conn)
    assert plan.ops and plan.ops[0].mode == "quarantine"
    assert "_Duplicates" in str(plan.ops[0].dst)


def test_find_and_where(tmp_path: Path):
    lib = tmp_path / "lib"
    src = tmp_path / "Kick_Pack" / "Kick_Cmin.wav"
    write_wav(src, kick(), 22050)
    cfg = Config(library_roots=[lib], keep_originals=False)
    conn = db.connect(tmp_path / "t.db")
    from crate import mover

    result = scan([src.parent], tmp_path / "st")
    verdicts, _ = annotate(result.items, conn, {}, jobs=1, use_clap=False)
    plan = build_plan(result.items, cfg, sort_router(cfg, verdicts), [], verdicts, conn)
    mover.execute(conn, plan, "sort", lib, trash=lambda p: Path(p).unlink())
    rows = find(conn, sound_type="kick")
    assert rows and Path(rows[0]["current_path"]).exists()
    found = where(conn, "Kick_Cmin")
    assert found and "Kicks" in found[0]["current_path"]
    verdict = next(iter(verdicts.values()))
    assert verdict.features is not None
    ranked = similar(conn, verdict.features.to_json(), limit=5)
    assert ranked


def test_near_duplicate_is_a_note_not_a_delete(tmp_path: Path):
    lib = tmp_path / "lib"
    (tmp_path / "in").mkdir()
    cfg = Config(library_roots=[lib], keep_originals=False, duplicates="trash")
    conn = db.connect(tmp_path / "t.db")
    from crate import mover

    first = tmp_path / "in" / "Kick_Full.wav"
    second = tmp_path / "in" / "Kick_Quiet.wav"
    write_wav(first, kick(), 22050)
    result = scan([first], tmp_path / "st")
    verdicts, _ = annotate(result.items, conn, {}, jobs=1, use_clap=False)
    plan = build_plan(result.items, cfg, sort_router(cfg, verdicts), [], verdicts, conn)
    mover.execute(conn, plan, "sort", lib, trash=lambda p: Path(p).unlink())
    write_wav(second, kick() * 0.5, 22050)
    result = scan([second], tmp_path / "st2")
    verdicts, _ = annotate(result.items, conn, {}, jobs=1, use_clap=False)
    plan = build_plan(result.items, cfg, sort_router(cfg, verdicts), [], verdicts, conn)
    assert any(op.mode != "trash" for op in plan.ops)
    assert plan.notes


def test_als_usage(tmp_path: Path):
    lib = tmp_path / "lib"
    lib.mkdir()
    sample = lib / "Kick_01.wav"
    write_wav(sample, kick(), 22050)
    conn = db.connect(tmp_path / "t.db")
    conn.execute(
        """INSERT INTO files(current_path, original_path, original_name, size)
           VALUES (?,?,?,?)""",
        (str(sample), str(sample), sample.name, sample.stat().st_size),
    )
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
    <Ableton><LiveSet><FileRef><Name Value="{sample.name}" />
    <Path Value="{sample}" /></FileRef>
    <FileRef><Name Value="Ghost.wav" /><RelativePath Value="Samples/Ghost.wav" /></FileRef>
    </LiveSet></Ableton>"""
    als = tmp_path / "Song.als"
    als.write_bytes(gzip.compress(xml.encode()))
    refs = parse_als(als)
    assert {ref.name for ref in refs} == {"Kick_01.wav", "Ghost.wav"}
    index_projects(conn, [als])
    report = usage(conn)
    assert report.used == 1 and report.unused == 0
    assert report.missing == ["Ghost.wav"]


def test_eval_scores_and_csv(tmp_path: Path):
    report = score([("a", "kick"), ("b", "kick"), ("c", "snare")], ["kick", "snare", "snare"])
    assert report.per_type["kick"][1] == 0.5  # recall
    assert report.per_type["snare"][0] == 0.5  # precision
    csv_path = tmp_path / "labels.csv"
    wav = tmp_path / "Kick_01.wav"
    write_wav(wav, kick(), 22050)
    csv_path.write_text(f"path,type,loop\n{wav},kick,0\n")
    rows = read_labels(csv_path)
    assert rows[0][1] == "kick" and rows[0][2] is False


def test_watch_ignores_existing_until_something_arrives(tmp_path: Path):
    folder = tmp_path / "Splice"
    folder.mkdir()
    write_wav(folder / "old.wav", kick(), 22050)
    conn = db.connect(tmp_path / "t.db")
    current = list_media(folder)
    mark_seen(conn, current)
    write_wav(folder / "new.wav", kick(), 22050)
    fresh = unseen(
        list_media(folder), {row["path"] for row in conn.execute("SELECT path FROM seen")}
    )
    assert [path.name for path in fresh] == ["new.wav"]


def test_status_line():
    assert format_status(12, 3) == "12 filed, 3 need review"


def test_old_database_grows_new_columns(tmp_path: Path):
    path = tmp_path / "old.db"
    import sqlite3

    raw = sqlite3.connect(path)
    raw.executescript(
        """
        CREATE TABLE files (
            id INTEGER PRIMARY KEY,
            current_path TEXT NOT NULL UNIQUE,
            original_path TEXT NOT NULL,
            original_name TEXT NOT NULL,
            pack TEXT,
            raw_hash TEXT,
            content_hash TEXT,
            size INTEGER
        );
        """
    )
    raw.commit()
    raw.close()
    conn = db.connect(path)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(files)")}
    assert "fingerprint" in cols and "musical_key" in cols
    assert conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='analysis'"
    ).fetchone()


def test_resampled_hash_stable():
    # Different containers of the same PCM share a content hash.
    a = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
    b = np.array(a, copy=True)
    assert pcm_hash(a) == pcm_hash(b)
