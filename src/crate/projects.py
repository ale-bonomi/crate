"""Read sample references out of Ableton Live sets (.als is gzipped XML)."""

from __future__ import annotations

import gzip
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

_FILE_REF = re.compile(r"<FileRef\b[^>]*>(.*?)</FileRef>", re.DOTALL)
_ATTR = re.compile(r"<(Name|RelativePath|Path)\b[^>]*\bValue=\"([^\"]*)\"")


@dataclass(frozen=True)
class SampleRef:
    name: str | None
    path: str | None


def parse_als(path: Path) -> list[SampleRef]:
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        text = gzip.decompress(raw).decode("utf-8", errors="replace")
    else:
        text = raw.decode("utf-8", errors="replace")
    refs: list[SampleRef] = []
    for block in _FILE_REF.findall(text):
        attrs = dict(_ATTR.findall(block))
        name = attrs.get("Name") or None
        location = attrs.get("Path") or attrs.get("RelativePath") or None
        if name or location:
            refs.append(SampleRef(name, location))
    return refs


def index_projects(
    conn: sqlite3.Connection, paths: list[Path]
) -> list[tuple[Path, list[SampleRef]]]:
    indexed: list[tuple[Path, list[SampleRef]]] = []
    for path in paths:
        if path.is_dir():
            files = sorted(path.rglob("*.als"))
        elif path.suffix.lower() == ".als":
            files = [path]
        else:
            continue
        for als in files:
            refs = parse_als(als)
            conn.execute(
                """DELETE FROM project_samples
                   WHERE project_id IN (SELECT id FROM projects WHERE path=?)""",
                (str(als),),
            )
            conn.execute(
                """INSERT INTO projects(path, name, scanned_at) VALUES (?,?,CURRENT_TIMESTAMP)
                   ON CONFLICT(path) DO UPDATE SET
                     name=excluded.name, scanned_at=CURRENT_TIMESTAMP""",
                (str(als), als.stem),
            )
            project_id = conn.execute(
                "SELECT id FROM projects WHERE path=?", (str(als),)
            ).fetchone()
            assert project_id is not None
            conn.executemany(
                "INSERT INTO project_samples(project_id, name, path) VALUES (?,?,?)",
                [(project_id["id"], ref.name, ref.path) for ref in refs],
            )
            indexed.append((als, refs))
    conn.commit()
    return indexed


@dataclass
class Usage:
    used: int
    unused: int
    missing: list[str]


def usage(conn: sqlite3.Connection) -> Usage:
    files = list(conn.execute("SELECT current_path, original_name FROM files"))
    refs = list(conn.execute("SELECT name, path FROM project_samples"))
    if not refs:
        return Usage(0, len(files), [])
    names = {row["original_name"] for row in files}
    names |= {Path(row["current_path"]).name for row in files}
    paths = {row["current_path"] for row in files}
    used_names: set[str] = set()
    missing: list[str] = []
    seen_missing: set[str] = set()
    for ref in refs:
        name = ref["name"] or ""
        location = ref["path"] or ""
        name_hit = bool(name) and name in names
        path_hit = bool(location) and (location in paths or Path(location).name in names)
        exists = bool(location) and Path(location).expanduser().exists()
        if name_hit or path_hit or exists:
            if name:
                used_names.add(name)
            elif location:
                used_names.add(Path(location).name)
        else:
            label = name or location
            if label and label not in seen_missing:
                seen_missing.add(label)
                missing.append(label)
    used = sum(
        1
        for row in files
        if Path(row["current_path"]).name in used_names or row["original_name"] in used_names
    )
    return Usage(used, len(files) - used, missing)
