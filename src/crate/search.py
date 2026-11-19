"""Look up library files by label, and find sounds that measure alike."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

_SCALES = {
    "duration": 4.0,
    "onset_count": 16.0,
    "crest": 12.0,
    "attack": 0.05,
    "decay": 0.6,
    "centroid": 4000.0,
    "rolloff": 8000.0,
    "low": 1.0,
    "mid": 1.0,
    "high": 1.0,
    "tonalness": 1.0,
    "tempo": 40.0,
    "swing": 0.5,
}


def where(conn: sqlite3.Connection, query: str) -> list[sqlite3.Row]:
    like = f"%{query}%"
    return list(
        conn.execute(
            """SELECT current_path, original_path, original_name, pack, sound_type, musical_key, bpm
               FROM files
               WHERE original_name LIKE ? OR current_path LIKE ? OR original_path LIKE ?
               ORDER BY current_path""",
            (like, like, like),
        )
    )


def find(
    conn: sqlite3.Connection,
    *,
    sound_type: str | None = None,
    musical_key: str | None = None,
    min_bpm: int | None = None,
    max_bpm: int | None = None,
    loop: bool | None = None,
    punch: str | None = None,
    feel: str | None = None,
    pack: str | None = None,
    limit: int = 50,
) -> list[sqlite3.Row]:
    clauses = ["current_path IS NOT NULL"]
    args: list[object] = []
    if sound_type:
        clauses.append("sound_type=?")
        args.append(sound_type)
    if musical_key:
        clauses.append("LOWER(musical_key)=LOWER(?)")
        args.append(musical_key)
    if min_bpm is not None:
        clauses.append("bpm>=?")
        args.append(min_bpm)
    if max_bpm is not None:
        clauses.append("bpm<=?")
        args.append(max_bpm)
    if loop is not None:
        clauses.append("is_loop=?")
        args.append(int(loop))
    if punch:
        clauses.append("punch=?")
        args.append(punch)
    if feel:
        clauses.append("feel=?")
        args.append(feel)
    if pack:
        clauses.append("pack LIKE ?")
        args.append(f"%{pack}%")
    args.append(limit)
    sql = (
        "SELECT current_path, sound_type, is_loop, musical_key, bpm, punch, feel, pack, confidence "
        f"FROM files WHERE {' AND '.join(clauses)} ORDER BY current_path LIMIT ?"
    )
    return list(conn.execute(sql, args))


def _vector(data: dict[str, object]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, scale in _SCALES.items():
        value = data.get(key)
        if isinstance(value, (int, float)) and scale:
            if key == "tempo":
                out[key] = (float(value) - 120.0) / scale
            else:
                out[key] = float(value) / scale
    return out


def similar(
    conn: sqlite3.Connection, features: dict[str, object], limit: int = 8
) -> list[tuple[float, sqlite3.Row]]:
    target = _vector(features)
    if not target:
        return []
    rows = conn.execute(
        """SELECT f.current_path, f.sound_type, f.bpm, f.musical_key, f.pack, a.features_json,
                  a.content_hash
           FROM files f JOIN analysis a ON a.content_hash=f.content_hash
           WHERE f.current_path IS NOT NULL AND a.features_json IS NOT NULL"""
    )
    scored: list[tuple[float, sqlite3.Row]] = []
    for row in rows:
        other = _vector(json.loads(row["features_json"]))
        keys = [k for k in target if k in other]
        if not keys:
            continue
        dist = sum((target[k] - other[k]) ** 2 for k in keys) ** 0.5
        scored.append((dist, row))
    scored.sort(key=lambda pair: pair[0])
    return scored[:limit]


def features_for_hash(conn: sqlite3.Connection, content_hash: str) -> dict[str, object] | None:
    row = conn.execute(
        "SELECT features_json FROM analysis WHERE content_hash=?", (content_hash,)
    ).fetchone()
    if row is None or not row["features_json"]:
        return None
    payload = json.loads(row["features_json"])
    return payload if isinstance(payload, dict) else None


def review_count(conn: sqlite3.Connection, threshold: float) -> int:
    row = conn.execute(
        """SELECT COUNT(*) FROM files
           WHERE confidence IS NOT NULL AND (confidence < ? OR sound_type IS NULL)""",
        (threshold,),
    ).fetchone()
    return int(row[0]) if row else 0


def pending_review(conn: sqlite3.Connection, threshold: float) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            """SELECT current_path, sound_type, confidence, label_source
               FROM files
               WHERE confidence IS NOT NULL AND (confidence < ? OR sound_type IS NULL)
               ORDER BY confidence, current_path""",
            (threshold,),
        )
    )


def filed_in_latest_batch(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        """SELECT id FROM batches WHERE status='done' ORDER BY id DESC LIMIT 1"""
    ).fetchone()
    if row is None:
        return 0
    count = conn.execute(
        """SELECT COUNT(*) FROM ops
           WHERE batch_id=? AND status='done'
             AND kind IN ('move','copy','extract','quarantine')
             AND (dst IS NULL OR dst NOT LIKE '%.asd')""",
        (row["id"],),
    ).fetchone()
    return int(count[0]) if count else 0


def format_status(filed: int, review: int) -> str:
    return f"{filed} filed, {review} need review"


def library_file(conn: sqlite3.Connection, path: Path) -> sqlite3.Row | None:
    row = conn.execute("SELECT * FROM files WHERE current_path=?", (str(path),)).fetchone()
    if not isinstance(row, sqlite3.Row):
        return None
    return row
