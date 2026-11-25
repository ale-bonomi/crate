"""Journal-first executor and crash-safe undo.

Every operation is written to the `ops` table as `pending` *before* it runs and marked
`done` after. Undo walks a batch in reverse and inspects the real filesystem state, so it
is correct whether the process died before, during or after an operation.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
from collections.abc import Callable
from pathlib import Path

from send2trash import send2trash

from .hashing import file_hash
from .planner import Plan, PlannedOp

Trash = Callable[[str], None]
PART_SUFFIX = ".crate-part"


class MoveError(RuntimeError):
    pass


def _same_volume(src: Path, dst: Path) -> bool:
    d = dst
    while not d.exists():
        d = d.parent
    return src.stat().st_dev == d.stat().st_dev


def copy_verified(src: Path, dst: Path) -> str:
    """Copy via temp name, verify size + hash, then atomically publish. Returns the hash."""
    part = dst.with_name(dst.name + PART_SUFFIX)
    shutil.copy2(src, part)
    if part.stat().st_size != src.stat().st_size:
        part.unlink()
        raise MoveError(f"size mismatch copying {src}")
    h = file_hash(src)
    if file_hash(part) != h:
        part.unlink()
        raise MoveError(f"hash mismatch copying {src}")
    os.replace(part, dst)
    return h


def _log(conn: sqlite3.Connection, batch: int, seq: int, kind: str, src: Path | None,
         dst: Path | None, note: str | None = None) -> int:  # fmt: skip
    cur = conn.execute(
        "INSERT INTO ops(batch_id, seq, kind, src, dst, note) VALUES (?,?,?,?,?,?)",
        (batch, seq, kind, str(src) if src else None, str(dst) if dst else None, note),
    )
    conn.commit()
    assert cur.lastrowid is not None
    return cur.lastrowid


def _done(conn: sqlite3.Connection, op_id: int) -> None:
    conn.execute("UPDATE ops SET status='done' WHERE id=?", (op_id,))
    conn.commit()


def execute(
    conn: sqlite3.Connection,
    plan: Plan,
    command: str,
    library_root: Path,
    trash: Trash = send2trash,
) -> int:
    """Apply a plan. Returns the batch id."""
    cur = conn.execute("INSERT INTO batches(command) VALUES (?)", (command,))
    conn.commit()
    batch = cur.lastrowid
    assert batch is not None
    seq = 0
    made_dirs: set[Path] = set()

    def ensure_dir(d: Path) -> None:
        nonlocal seq
        missing: list[Path] = []
        while not d.exists() and d not in made_dirs:
            missing.append(d)
            d = d.parent
        for m in reversed(missing):
            seq += 1
            oid = _log(conn, batch, seq, "mkdir", None, m)
            m.mkdir(exist_ok=True)
            made_dirs.add(m)
            _done(conn, oid)

    try:
        for op in plan.ops:
            if op.mode in {"trash", "symlink"}:
                # dst is the library copy (or a placeholder), so it is meant to exist.
                if op.kept is not None and not op.kept.exists():
                    raise MoveError(f"kept copy is missing, so {op.src} was left alone: {op.kept}")
            else:
                if op.dst.exists():
                    raise MoveError(f"destination appeared since planning: {op.dst}")
                ensure_dir(op.dst.parent)
            seq += 1
            oid = _log(conn, batch, seq, op.mode, op.src, op.dst, op.original)
            h = _perform(op, trash)
            _done(conn, oid)
            _track(conn, op, h)
        for d in _prunable(plan):
            seq += 1
            oid = _log(conn, batch, seq, "rmdir", d, None)
            d.rmdir()
            _done(conn, oid)
    except BaseException:
        conn.execute("UPDATE batches SET status='failed' WHERE id=?", (batch,))
        conn.commit()
        raise
    conn.execute("UPDATE batches SET status='done' WHERE id=?", (batch,))
    conn.commit()
    return batch


def _perform(op: PlannedOp, trash: Trash) -> str | None:
    if op.mode == "trash":
        if op.src.exists() or op.src.is_symlink():
            trash(str(op.src))
        return op.raw_hash
    if op.mode == "symlink":
        if op.src.is_symlink() or op.src.exists():
            op.src.unlink()
        op.src.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(op.dst, op.src)
        return op.raw_hash
    if op.mode == "copy":
        return copy_verified(op.src, op.dst)
    if _same_volume(op.src, op.dst):
        os.rename(op.src, op.dst)
        return None
    h = copy_verified(op.src, op.dst)  # cross-volume: copy -> verify -> remove source
    op.src.unlink()
    return h


def _loop_flag(value: bool | None) -> int | None:
    if value is None:
        return None
    return int(value)


def _track(conn: sqlite3.Connection, op: PlannedOp, h: str | None) -> None:
    if op.mode in {"trash", "symlink"} or op.sidecar:
        conn.commit()
        return
    size = op.dst.stat().st_size
    raw = h or op.raw_hash
    row = conn.execute("SELECT id FROM files WHERE current_path=?", (str(op.src),)).fetchone()
    fields = (
        op.content_hash,
        op.fingerprint,
        op.sound_type,
        _loop_flag(op.is_loop),
        op.confidence,
        op.bpm,
        op.musical_key,
        op.punch,
        op.snare_body,
        op.feel,
        op.wet,
        op.label_source,
        size,
    )
    if row and op.mode in {"move", "quarantine"}:
        conn.execute(
            """UPDATE files SET current_path=?, raw_hash=COALESCE(?, raw_hash),
               content_hash=COALESCE(?, content_hash), fingerprint=COALESCE(?, fingerprint),
               sound_type=?, is_loop=?, confidence=?, bpm=?, musical_key=?, punch=?,
               snare_body=?, feel=?, wet=?, label_source=?, size=? WHERE id=?""",
            (str(op.dst), raw, *fields, row["id"]),
        )
    else:
        conn.execute(
            """INSERT OR REPLACE INTO files(
                current_path, original_path, original_name, pack, raw_hash, content_hash,
                size, fingerprint, sound_type, is_loop, confidence, bpm, musical_key,
                punch, snare_body, feel, wet, label_source
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                str(op.dst),
                op.original,
                op.src.name,
                op.pack,
                raw,
                op.content_hash,
                size,
                op.fingerprint,
                op.sound_type,
                _loop_flag(op.is_loop),
                op.confidence,
                op.bpm,
                op.musical_key,
                op.punch,
                op.snare_body,
                op.feel,
                op.wet,
                op.label_source,
            ),
        )
    conn.commit()


def _prunable(plan: Plan) -> list[Path]:
    """Empty source directories left behind by moves (deepest first), within input roots."""
    limits = [r.resolve() for r in plan.prune_roots]
    cands: set[Path] = set()
    for op in plan.ops:
        if op.mode != "move":
            continue
        d = op.src.resolve().parent
        while any(d.is_relative_to(lim) for lim in limits):
            cands.add(d)
            d = d.parent
    out: list[Path] = []
    removed: set[Path] = set()
    for d in sorted(cands, key=lambda p: len(p.parts), reverse=True):
        try:
            remaining = [c for c in d.iterdir() if c not in removed]
        except OSError:
            continue
        if not remaining:
            out.append(d)
            removed.add(d)
    return out


# ---------------------------------------------------------------------------- undo


def latest_done_batch(conn: sqlite3.Connection) -> int | None:
    row = conn.execute(
        "SELECT id FROM batches WHERE status IN ('done','failed','undoing') "
        "ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return int(row["id"]) if row else None


def undo(conn: sqlite3.Connection, batch: int, trash: Trash = send2trash) -> list[str]:
    """Reverse a batch. Returns warnings for operations that could not be reversed."""
    warnings: list[str] = []
    conn.execute("UPDATE batches SET status='undoing' WHERE id=?", (batch,))
    conn.commit()
    ops = conn.execute(
        "SELECT * FROM ops WHERE batch_id=? AND status!='undone' ORDER BY seq DESC", (batch,)
    ).fetchall()
    for op in ops:
        src = Path(op["src"]) if op["src"] else None
        dst = Path(op["dst"]) if op["dst"] else None
        ok = True
        kind = op["kind"]
        if kind == "mkdir":
            assert dst is not None
            if dst.is_dir() and not any(dst.iterdir()):
                dst.rmdir()
            elif dst.exists():
                warnings.append(f"left non-empty directory: {dst}")
        elif kind == "rmdir":
            assert src is not None
            src.mkdir(parents=True, exist_ok=True)
        elif kind in ("move", "quarantine"):
            assert src is not None and dst is not None
            ok = _undo_move(src, dst, warnings)
        elif kind == "symlink":
            assert src is not None and dst is not None
            if src.is_symlink():
                src.unlink()
            if not src.exists() and dst.exists():
                copy_verified(dst, src)
        elif kind == "trash":
            assert src is not None
            if not src.exists():
                warnings.append(
                    f"{src} is in the Trash; pull it back from there if you still need it"
                )
        elif kind in ("copy", "extract"):
            assert dst is not None
            part = dst.with_name(dst.name + PART_SUFFIX)
            if part.exists():
                part.unlink()
            if dst.exists():
                trash(str(dst))
            conn.execute("DELETE FROM files WHERE current_path=?", (str(dst),))
        if kind in ("move", "quarantine") and ok:
            assert src is not None and dst is not None
            conn.execute(
                "UPDATE files SET current_path=? WHERE current_path=?", (str(src), str(dst))
            )
        if ok:
            conn.execute("UPDATE ops SET status='undone' WHERE id=?", (op["id"],))
        conn.commit()
    left = conn.execute(
        "SELECT COUNT(*) FROM ops WHERE batch_id=? AND status!='undone'", (batch,)
    ).fetchone()[0]
    conn.execute(
        "UPDATE batches SET status=? WHERE id=?", ("undone" if left == 0 else "undoing", batch)
    )
    conn.commit()
    return warnings


def _undo_move(src: Path, dst: Path, warnings: list[str]) -> bool:
    if src.exists() and not dst.exists():
        return True  # never happened (or already restored)
    if dst.exists() and not src.exists():
        src.parent.mkdir(parents=True, exist_ok=True)
        if _same_volume(dst, src):
            os.rename(dst, src)
        else:
            copy_verified(dst, src)
            dst.unlink()
        return True
    if src.exists() and dst.exists():
        if file_hash(src) == file_hash(dst):  # interrupted cross-volume move
            dst.unlink()
            return True
        warnings.append(f"cannot undo {dst}: {src} exists with different content")
        return False
    warnings.append(f"cannot undo: neither {src} nor {dst} exists")
    return False
