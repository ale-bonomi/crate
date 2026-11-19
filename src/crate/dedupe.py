"""Exact duplicates by PCM hash, and near-duplicates by fingerprint.

Near-duplicates are reported and never deleted.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .analyze.audio import load
from .hashing import file_hash
from .scan import AUDIO_EXT


def library_copy(
    conn: sqlite3.Connection,
    content_hash: str | None,
    raw_hash: str | None,
    src: Path,
) -> Path | None:
    """Path of a different existing library file with the same audio (or the same bytes)."""
    try:
        src_key = str(src.resolve())
    except OSError:
        src_key = str(src)
    for column, value in (("content_hash", content_hash), ("raw_hash", raw_hash)):
        if not value:
            continue
        rows = conn.execute(
            f"SELECT current_path FROM files WHERE {column}=? AND current_path IS NOT NULL",
            (value,),
        )
        for row in rows:
            path = Path(row["current_path"])
            if not path.exists():
                continue
            try:
                if str(path.resolve()) == src_key:
                    continue
            except OSError:
                continue
            return path
    return None


def near_copies(
    conn: sqlite3.Connection, fingerprint: str | None, content_hash: str | None
) -> list[str]:
    if not fingerprint:
        return []
    rows = conn.execute(
        "SELECT current_path, content_hash FROM files WHERE fingerprint=?",
        (fingerprint,),
    )
    found: list[str] = []
    for row in rows:
        if content_hash and row["content_hash"] == content_hash:
            continue
        path = Path(row["current_path"])
        if path.exists():
            found.append(str(path))
    return found


def _cached(conn: sqlite3.Connection, path: Path) -> tuple[str | None, str | None, str | None]:
    row = conn.execute(
        "SELECT content_hash, raw_hash, fingerprint, size FROM files WHERE current_path=?",
        (str(path),),
    ).fetchone()
    if row is None or row["size"] != path.stat().st_size:
        return None, None, None
    return row["content_hash"], row["raw_hash"], row["fingerprint"]


def identify(conn: sqlite3.Connection, path: Path) -> tuple[str | None, str | None, str | None]:
    content, raw, finger = _cached(conn, path)
    if content or finger:
        return content, raw, finger
    audio = load(path)
    if audio is None:
        return None, file_hash(path), None
    return audio.content_hash, audio.raw_hash, audio.fingerprint


def scan_tree(roots: list[Path]) -> list[Path]:
    found: list[Path] = []
    for root in roots:
        root = root.expanduser()
        if root.is_file() and root.suffix.lower() in AUDIO_EXT:
            found.append(root)
        elif root.is_dir():
            for path in sorted(root.rglob("*")):
                if (
                    path.is_file()
                    and path.suffix.lower() in AUDIO_EXT
                    and not path.name.startswith(".")
                ):
                    found.append(path)
    return found
