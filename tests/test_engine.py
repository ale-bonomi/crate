"""Planner + journal + undo, end to end on a temp tree."""

import hashlib
import zipfile
from pathlib import Path

import pytest

from crate import db, mover, planner
from crate.analyze.engine import annotate
from crate.config import Config
from crate.scan import scan
from tests.synth import kick, write_wav


def snapshot(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    } | {str(p.relative_to(root)) + "/": "" for p in root.rglob("*") if p.is_dir()}


@pytest.fixture
def world(tmp_path: Path):
    lib = tmp_path / "lib"
    inbox = tmp_path / "inbox"
    (inbox / "My Pack" / "Kicks").mkdir(parents=True)
    (inbox / "My Pack" / "Kicks" / "Kick_01.wav").write_bytes(b"kick-one")
    (inbox / "My Pack" / "Kicks" / "Kick_01.wav.asd").write_bytes(b"asd")
    (inbox / "My Pack" / "Snares").mkdir()
    (inbox / "My Pack" / "Snares" / "Snare_Tight.wav").write_bytes(b"snare")
    (inbox / "My Pack" / "mystery_1.wav").write_bytes(b"???")
    (inbox / "My Pack" / ".DS_Store").write_bytes(b"x")
    with zipfile.ZipFile(inbox / "Zipped.zip", "w") as z:
        z.writestr("Hats/Hat_Closed.wav", b"hat")
        z.writestr("__MACOSX/Hats/._Hat_Closed.wav", b"junk")
        z.writestr("../evil.wav", b"nope")
    cfg = Config(library_roots=[lib], keep_originals=False)
    conn = db.connect(tmp_path / "test.db")
    return tmp_path, inbox, lib, cfg, conn


def run(world, apply=True):
    tmp, inbox, lib, cfg, conn = world
    staging = tmp / "staging"
    staging.mkdir(exist_ok=True)
    res = scan([inbox], staging)
    verdicts, _warnings = annotate(res.items, conn, {}, jobs=1, use_clap=False)
    plan = planner.build_plan(
        res.items, cfg, planner.sort_router(cfg, verdicts), [], verdicts, conn
    )
    batch = mover.execute(conn, plan, "sort", lib) if apply else None
    return res, plan, batch


def test_dry_run_changes_nothing(world):
    _tmp, inbox, lib, *_ = world
    before = snapshot(inbox)
    run(world, apply=False)
    assert snapshot(inbox) == before
    assert not lib.exists() or not any(path.is_file() for path in lib.rglob("*"))


def test_sort_then_undo_restores_bytes(world):
    tmp, inbox, lib, cfg, conn = world
    before = snapshot(inbox)
    res, plan, batch = run(world)
    assert (lib / "One Shots/Kicks/Kick_01.wav").exists()
    assert (lib / "One Shots/Kicks/Kick_01.wav.asd").exists()  # sidecar traveled
    assert (lib / "One Shots/Snares/Snare_Tight.wav").exists()
    assert (lib / "One Shots/Hats & Cymbals/Hat_Closed.wav").exists()  # from the zip
    assert (lib / "_Review/mystery_1.wav").exists()
    assert not (tmp / "evil.wav").exists()
    assert any("unsafe" in w for w in res.warnings)
    assert not (inbox / "My Pack" / "Kicks").exists()  # emptied dirs pruned

    assert mover.undo(conn, batch, trash=lambda p: Path(p).unlink()) == []
    assert snapshot(inbox) == before
    assert not lib.exists() or not any(lib.rglob("*.wav"))


def test_idempotent_second_run(world):
    run(world)
    tmp, inbox, lib, cfg, conn = world
    res, plan, _ = run(world, apply=False)
    assert plan.ops == [] or all(o.src.parent.name != "Kicks" for o in plan.ops)


def test_collision_gets_suffix(world):
    tmp, inbox, lib, cfg, conn = world
    (lib / "One Shots/Kicks").mkdir(parents=True)
    (lib / "One Shots/Kicks/Kick_01.wav").write_bytes(b"different")
    _, plan, _ = run(world, apply=False)
    kick = next(o for o in plan.ops if o.src.name == "Kick_01.wav")
    assert kick.dst.name == "Kick_01 (2).wav" and kick.renamed
    assert any(o.dst.name == "Kick_01 (2).wav.asd" for o in plan.ops)


def test_identical_content_is_skipped(world):
    tmp, inbox, lib, cfg, conn = world
    (lib / "One Shots/Kicks").mkdir(parents=True)
    (lib / "One Shots/Kicks/Kick_01.wav").write_bytes(b"kick-one")
    _, plan, _ = run(world, apply=False)
    assert all(o.src.name != "Kick_01.wav" for o in plan.ops)
    assert any("identical" in why for _, why in plan.skipped)


def test_undo_after_crash_mid_batch(world, monkeypatch):
    tmp, inbox, lib, cfg, conn = world
    before = snapshot(inbox)
    calls = {"n": 0}
    real = mover._perform

    def flaky(op, trash=None):
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt
        return real(op, trash)

    monkeypatch.setattr(mover, "_perform", flaky)
    with pytest.raises(KeyboardInterrupt):
        run(world)
    batch = mover.latest_done_batch(conn)
    assert batch is not None
    mover.undo(conn, batch, trash=lambda p: Path(p).unlink())
    assert snapshot(inbox) == before


def test_copy_mode_keeps_originals(world):
    tmp, inbox, lib, cfg, conn = world
    cfg.keep_originals = True
    res = scan([inbox], tmp / "st")
    plan = planner.build_plan(res.items, cfg, planner.sort_router(cfg), [inbox])
    mover.execute(conn, plan, "sort", lib)
    assert (inbox / "My Pack/Kicks/Kick_01.wav").exists()
    assert (lib / "One Shots/Kicks/Kick_01.wav").read_bytes() == b"kick-one"


def _kick_wav(path: Path) -> bytes:
    write_wav(path, kick(), 22050)
    return path.read_bytes()


@pytest.fixture
def dupes(world):
    """The same kick twice in one pack, under two names."""
    tmp, inbox, lib, cfg, conn = world
    kicks = inbox / "My Pack" / "Kicks"
    _kick_wav(kicks / "Kick_01.wav")
    _kick_wav(kicks / "Kick_01 copy.wav")
    return world


def _plan(world):
    return run(world, apply=False)[1]


KICK_NAMES = {"Kick_01.wav", "Kick_01 copy.wav"}


def _pair(plan):
    """(op that files the kept kick, op that resolves its duplicate)."""
    ops = [o for o in plan.ops if o.src.name in KICK_NAMES and not o.sidecar]
    keeper = next(o for o in ops if o.mode == "move")
    dupe = next(o for o in ops if o is not keeper)
    return keeper, dupe


@pytest.mark.parametrize("mode", ["quarantine", "symlink"])
def test_duplicates_in_one_pack_apply_then_undo(dupes, mode):
    tmp, inbox, lib, cfg, conn = dupes
    cfg.duplicates = mode
    before = snapshot(inbox)
    audio = (inbox / "My Pack" / "Kicks" / "Kick_01.wav").read_bytes()
    _, plan, batch = run(dupes)
    keeper, dupe = _pair(plan)
    assert keeper.dst.read_bytes() == audio
    if mode == "symlink":
        assert dupe.src.is_symlink() and dupe.src.resolve() == keeper.dst.resolve()
    else:
        assert dupe.dst.parent == lib / "_Duplicates" and dupe.dst.read_bytes() == audio

    assert mover.undo(conn, batch, trash=lambda p: Path(p).unlink()) == []
    assert snapshot(inbox) == before


def test_duplicate_in_one_pack_is_trashed(dupes):
    tmp, inbox, lib, cfg, conn = dupes
    plan = _plan(dupes)
    keeper, dupe = _pair(plan)
    assert dupe.mode == "trash" and dupe.kept == keeper.dst
    trashed: list[str] = []

    def trash(p: str) -> None:
        trashed.append(p)
        Path(p).unlink()

    batch = mover.execute(conn, plan, "sort", lib, trash=trash)
    assert trashed == [str(dupe.src)]
    assert keeper.dst.exists()
    warnings = mover.undo(conn, batch, trash=lambda p: Path(p).unlink())
    assert any("Trash" in w for w in warnings)  # a trashed file is restored by hand
    assert keeper.src.exists()


def test_duplicate_follows_kept_copy_into_split_folder(dupes):
    tmp, inbox, lib, cfg, conn = dupes
    cfg.duplicates = "symlink"
    cfg.split_folder_threshold = 0  # force Kicks/ to split by punch
    plan = _plan(dupes)
    keeper, dupe = _pair(plan)
    assert keeper.dst.parent.parent == lib / "One Shots/Kicks"  # moved into a subfolder
    assert dupe.dst == dupe.kept == keeper.dst
    mover.execute(conn, plan, "sort", lib)
    assert dupe.src.resolve() == keeper.dst.resolve()


def test_duplicate_of_library_file_is_trashed(dupes):
    tmp, inbox, lib, cfg, conn = dupes
    cfg.duplicates = "keep"
    run(dupes)  # both kicks land in the library and are indexed
    cfg.duplicates = "trash"
    again = inbox / "Again" / "Kick_01.wav"
    _kick_wav(again)
    _, plan, _ = run(dupes, apply=False)
    dupe = next(o for o in plan.ops if o.src == again)
    assert dupe.mode == "trash" and dupe.kept is not None and dupe.kept.exists()
    mover.execute(conn, plan, "sort", lib, trash=lambda p: Path(p).unlink())
    assert not again.exists()


def test_trash_refuses_when_kept_copy_is_gone(dupes):
    tmp, inbox, lib, cfg, conn = dupes
    plan = _plan(dupes)
    keeper, dupe = _pair(plan)
    plan.ops = [o for o in plan.ops if o is not keeper]  # the kept copy never lands
    trashed: list[str] = []
    with pytest.raises(mover.MoveError, match="kept copy is missing"):
        mover.execute(conn, plan, "sort", lib, trash=trashed.append)
    assert trashed == []
    assert dupe.src.exists()
