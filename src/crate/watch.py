"""Notice new files in the Splice folder. Existing files are left alone unless asked."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .scan import AUDIO_EXT, MIDI_EXT


def list_media(root: Path) -> list[Path]:
    root = root.expanduser()
    if not root.exists():
        return []
    if root.is_file():
        return [root]
    found: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() in AUDIO_EXT | MIDI_EXT | {".zip"}:
            found.append(path)
    return found


def unseen(paths: list[Path], seen: set[str]) -> list[Path]:
    return [path for path in paths if str(path) not in seen]


def load_seen(conn: sqlite3.Connection) -> set[str]:
    return {row["path"] for row in conn.execute("SELECT path FROM seen")}


def mark_seen(conn: sqlite3.Connection, paths: list[Path]) -> None:
    conn.executemany(
        "INSERT INTO seen(path, size, mtime) VALUES (?,?,?) "
        "ON CONFLICT(path) DO UPDATE SET size=excluded.size, mtime=excluded.mtime",
        [(str(path), path.stat().st_size, path.stat().st_mtime) for path in paths if path.exists()],
    )
    conn.commit()
