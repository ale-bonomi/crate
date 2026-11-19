"""SQLite storage: files, analysis cache, corrections, batches, and the operation journal."""

from __future__ import annotations

import sqlite3
from pathlib import Path

TABLES = """
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY,
    current_path TEXT NOT NULL UNIQUE,
    original_path TEXT NOT NULL,
    original_name TEXT NOT NULL,
    pack TEXT,
    raw_hash TEXT,
    content_hash TEXT,
    size INTEGER,
    fingerprint TEXT,
    sound_type TEXT,
    is_loop INTEGER,
    confidence REAL,
    bpm INTEGER,
    musical_key TEXT,
    punch TEXT,
    snare_body TEXT,
    feel TEXT,
    wet TEXT,
    label_source TEXT
);
CREATE TABLE IF NOT EXISTS batches (
    id INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    command TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'running'
);
CREATE TABLE IF NOT EXISTS ops (
    id INTEGER PRIMARY KEY,
    batch_id INTEGER NOT NULL REFERENCES batches(id),
    seq INTEGER NOT NULL,
    kind TEXT NOT NULL,
    src TEXT,
    dst TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    note TEXT
);
CREATE TABLE IF NOT EXISTS analysis (
    content_hash TEXT PRIMARY KEY,
    raw_hash TEXT,
    duration REAL,
    sample_rate INTEGER,
    onset_count INTEGER,
    crest REAL,
    attack REAL,
    decay REAL,
    centroid REAL,
    rolloff REAL,
    low_energy REAL,
    mid_energy REAL,
    high_energy REAL,
    tonalness REAL,
    tempo REAL,
    key_guess TEXT,
    key_confidence REAL,
    swing REAL,
    features_json TEXT,
    clap_json TEXT,
    analyzed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS corrections (
    content_hash TEXT PRIMARY KEY,
    sound_type TEXT,
    is_loop INTEGER,
    punch TEXT,
    snare_body TEXT,
    feel TEXT,
    wet TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    name TEXT,
    scanned_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS project_samples (
    project_id INTEGER NOT NULL REFERENCES projects(id),
    name TEXT,
    path TEXT
);
CREATE TABLE IF NOT EXISTS seen (
    path TEXT PRIMARY KEY,
    size INTEGER,
    mtime REAL
);
"""

INDEXES = """
CREATE INDEX IF NOT EXISTS idx_files_raw ON files(raw_hash);
CREATE INDEX IF NOT EXISTS idx_files_content ON files(content_hash);
CREATE INDEX IF NOT EXISTS idx_files_fingerprint ON files(fingerprint);
CREATE INDEX IF NOT EXISTS idx_ops_batch ON ops(batch_id, seq);
CREATE INDEX IF NOT EXISTS idx_project_samples ON project_samples(project_id);
"""

_FILE_COLUMNS = {
    "fingerprint": "TEXT",
    "sound_type": "TEXT",
    "is_loop": "INTEGER",
    "confidence": "REAL",
    "bpm": "INTEGER",
    "musical_key": "TEXT",
    "punch": "TEXT",
    "snare_body": "TEXT",
    "feel": "TEXT",
    "wet": "TEXT",
    "label_source": "TEXT",
}


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(TABLES)
    have = _columns(conn, "files")
    for name, decl in _FILE_COLUMNS.items():
        if name not in have:
            conn.execute(f"ALTER TABLE files ADD COLUMN {name} {decl}")
    conn.executescript(INDEXES)
    conn.commit()
    return conn
