"""`crate dedupe`: one plan, run through the journal, reversible with undo."""

from pathlib import Path

import pytest

from crate import db, mover, planner
from crate.config import Config
from tests.synth import kick, snare, write_wav


def snapshot(root: Path) -> dict[str, bytes | str]:
    out: dict[str, bytes | str] = {}
    for p in sorted(root.rglob("*")):
        rel = str(p.relative_to(root))
        if p.is_symlink():
            out[rel] = f"-> {p.readlink()}"
        elif p.is_file() and not p.name.startswith("t.db"):
            out[rel] = p.read_bytes()
    return out


@pytest.fixture
def world(tmp_path: Path):
    """A kick in the library, the same kick re-downloaded into Splice, and a lone snare."""
    lib = tmp_path / "lib"
    splice = tmp_path / "splice"
    write_wav(lib / "One Shots/Kicks/Kick_A.wav", kick(), 22050)
    write_wav(splice / "Pack/Kick_A.wav", kick(), 22050)
    write_wav(splice / "Pack/Snare_B.wav", snare(), 22050)
    cfg = Config(library_roots=[lib], splice_folder=splice, keep_originals=False)
    conn = db.connect(tmp_path / "t.db")
    return tmp_path, lib, splice, cfg, conn


def _plan(world, mode=None):
    tmp, lib, splice, cfg, conn = world
    return planner.duplicate_plan(conn, cfg, [lib, splice], mode=mode)


def test_library_copy_is_kept(world):
    tmp, lib, splice, cfg, conn = world
    plan = _plan(world)
    assert len(plan.ops) == 1
    op = plan.ops[0]
    assert op.src == splice / "Pack/Kick_A.wav"
    assert op.kept == lib / "One Shots/Kicks/Kick_A.wav"
    assert op.mode == "trash"


@pytest.mark.parametrize("mode", ["quarantine", "symlink"])
def test_apply_then_undo_restores_everything(world, mode):
    tmp, lib, splice, cfg, conn = world
    before = snapshot(tmp)
    plan = _plan(world, mode)
    batch = mover.execute(conn, plan, "dedupe", cfg.root)
    copy = splice / "Pack/Kick_A.wav"
    if mode == "symlink":
        assert copy.is_symlink() and copy.resolve() == (lib / "One Shots/Kicks/Kick_A.wav")
    else:
        assert not copy.exists() and (lib / "_Duplicates/Kick_A.wav").exists()

    assert mover.undo(conn, batch) == []
    assert snapshot(tmp) == before


@pytest.mark.parametrize("mode", ["quarantine", "symlink"])
def test_second_run_finds_nothing(world, mode):
    mover.execute(world[4], _plan(world, mode), "dedupe", world[3].root)
    assert _plan(world, mode).ops == []


def test_trash_is_journaled(world):
    tmp, lib, splice, cfg, conn = world
    trashed: list[str] = []

    def trash(p: str) -> None:
        trashed.append(p)
        Path(p).unlink()

    batch = mover.execute(conn, _plan(world), "dedupe", cfg.root, trash=trash)
    assert trashed == [str(splice / "Pack/Kick_A.wav")]
    assert (lib / "One Shots/Kicks/Kick_A.wav").exists()
    kinds = [r["kind"] for r in conn.execute("SELECT kind FROM ops WHERE batch_id=?", (batch,))]
    assert kinds == ["trash"]
    warnings = mover.undo(conn, batch)
    assert any("Trash" in w for w in warnings)


def test_keep_reports_without_moving(world):
    plan = _plan(world, "keep")
    assert plan.ops == []
    assert any("is a copy of" in note for note in plan.notes)
