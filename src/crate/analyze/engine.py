"""Analyze audio items and cache features by decoded-PCM hash."""

from __future__ import annotations

import json
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..hashing import file_hash
from ..scan import Item
from . import metadata
from .audio import AudioData, load
from .clap import ClapTagger
from .features import extract
from .fusion import fuse
from .types import Correction, Features, Verdict

if TYPE_CHECKING:
    from .. import learn
    from ..learn import Model

_FEATURE_KEYS = (
    "duration",
    "sample_rate",
    "onset_count",
    "crest",
    "attack",
    "decay",
    "centroid",
    "rolloff",
    "low",
    "mid",
    "high",
    "tonalness",
    "tempo",
    "key",
    "key_confidence",
    "swing",
)


@dataclass
class _Job:
    key: int
    audio: AudioData | None
    raw_hash: str
    features: Features | None
    clap: dict[str, float] | None
    cached: bool


def workers(jobs: int | None) -> int:
    if jobs is not None and jobs > 0:
        return jobs
    return max(1, (os.cpu_count() or 2) - 1)


def _features_from_row(row: sqlite3.Row) -> Features | None:
    if row["features_json"] is None:
        return None
    data = json.loads(row["features_json"])
    return Features(**{k: data.get(k) for k in _FEATURE_KEYS})


def _load_cache(
    conn: sqlite3.Connection, content_hash: str
) -> tuple[Features | None, dict[str, float] | None] | None:
    row = conn.execute("SELECT * FROM analysis WHERE content_hash=?", (content_hash,)).fetchone()
    if row is None:
        return None
    clap = json.loads(row["clap_json"]) if row["clap_json"] else None
    return _features_from_row(row), clap


def _save_cache(
    conn: sqlite3.Connection,
    audio: AudioData,
    features: Features | None,
    clap: dict[str, float] | None,
) -> None:
    payload = json.dumps(features.to_json()) if features else None
    clap_payload = json.dumps(clap) if clap else None
    conn.execute(
        """INSERT INTO analysis(
            content_hash, raw_hash, duration, sample_rate, onset_count, crest, attack, decay,
            centroid, rolloff, low_energy, mid_energy, high_energy, tonalness, tempo, key_guess,
            key_confidence, swing, features_json, clap_json
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(content_hash) DO UPDATE SET
            features_json=excluded.features_json,
            clap_json=COALESCE(excluded.clap_json, analysis.clap_json),
            raw_hash=excluded.raw_hash""",
        (
            audio.content_hash,
            audio.raw_hash,
            features.duration if features else audio.duration,
            audio.sample_rate,
            features.onset_count if features else None,
            features.crest if features else None,
            features.attack if features else None,
            features.decay if features else None,
            features.centroid if features else None,
            features.rolloff if features else None,
            features.low if features else None,
            features.mid if features else None,
            features.high if features else None,
            features.tonalness if features else None,
            features.tempo if features else None,
            features.key if features else None,
            features.key_confidence if features else None,
            features.swing if features else None,
            payload,
            clap_payload,
        ),
    )


def load_correction(conn: sqlite3.Connection, identity: str | None) -> Correction | None:
    if not identity:
        return None
    row = conn.execute("SELECT * FROM corrections WHERE content_hash=?", (identity,)).fetchone()
    if row is None:
        return None
    return Correction(
        type=row["sound_type"],
        loop=None if row["is_loop"] is None else bool(row["is_loop"]),
        punch=row["punch"],
        snare_body=row["snare_body"],
        feel=row["feel"],
        wet=row["wet"],
    )


def save_correction(conn: sqlite3.Connection, identity: str, correction: Correction) -> None:
    conn.execute(
        """INSERT INTO corrections(content_hash, sound_type, is_loop, punch, snare_body, feel, wet)
           VALUES (?,?,?,?,?,?,?)
           ON CONFLICT(content_hash) DO UPDATE SET
             sound_type=excluded.sound_type, is_loop=excluded.is_loop, punch=excluded.punch,
             snare_body=excluded.snare_body, feel=excluded.feel, wet=excluded.wet,
             created_at=CURRENT_TIMESTAMP""",
        (
            identity,
            correction.type,
            None if correction.loop is None else int(correction.loop),
            correction.punch,
            correction.snare_body,
            correction.feel,
            correction.wet,
        ),
    )
    conn.commit()


def _read(item_id: int, path: Path) -> tuple[int, AudioData | None, str]:
    try:
        audio = load(path)
    except OSError:
        audio = None
    digest = audio.raw_hash if audio else file_hash(path)
    return item_id, audio, digest


def annotate(
    items: list[Item],
    conn: sqlite3.Connection,
    prompts: dict[str, list[str]],
    *,
    jobs: int | None = None,
    refresh: bool = False,
    use_clap: bool = True,
    model: Model | None = None,
) -> tuple[dict[int, Verdict], list[str]]:
    """Return verdicts keyed by `id(item)`, plus warnings. Writes the feature cache.

    `model` (from `crate train`) names files whose names say nothing, using CLAP
    embeddings cached in the analysis table."""
    from .. import learn

    warnings: list[str] = []
    audio_items = [it for it in items if it.kind == "audio"]
    tagger = ClapTagger(prompts) if use_clap else None
    if tagger is not None and not tagger.available():
        warnings.append(
            "CLAP is not installed, so classification uses filenames and DSP only. "
            "Install it with: uv sync --extra ml"
        )
        tagger = None
    embedder: ClapTagger | None = None
    if model is not None:
        embedder = tagger or ClapTagger({})
        if not embedder.can_embed():
            warnings.append(
                "A trained model exists, but CLAP is not installed to use it. "
                "Install it with: uv sync --extra ml"
            )
            embedder = None

    reads: dict[int, tuple[AudioData | None, str]] = {}
    with ThreadPoolExecutor(max_workers=workers(jobs)) as pool:
        futures = [pool.submit(_read, id(it), it.path) for it in audio_items]
        for fut in futures:
            item_id, audio, digest = fut.result()
            reads[item_id] = (audio, digest)

    pending: list[tuple[Item, AudioData]] = []
    jobs_done: dict[int, _Job] = {}
    for it in audio_items:
        audio, digest = reads[id(it)]
        cached_features: Features | None = None
        cached_clap: dict[str, float] | None = None
        hit = False
        if audio and not refresh:
            cached = _load_cache(conn, audio.content_hash)
            if cached is not None and cached[0] is not None:
                cached_features, cached_clap = cached
                hit = True
        if audio and not hit:
            pending.append((it, audio))
        jobs_done[id(it)] = _Job(id(it), audio, digest, cached_features, cached_clap, hit)

    if pending:
        with ThreadPoolExecutor(max_workers=workers(jobs)) as pool:
            feature_jobs = {
                pool.submit(extract, audio.samples, audio.sample_rate, audio.duration): it
                for it, audio in pending
            }
            for feature_future, it in feature_jobs.items():
                jobs_done[id(it)].features = feature_future.result()

    verdicts: dict[int, Verdict] = {}
    for it in audio_items:
        job = jobs_done[id(it)]
        meta = metadata.parse(it.path.stem, it.folders)
        clap_scores = job.clap
        if (
            tagger is not None
            and job.audio is not None
            and clap_scores is None
            and job.features is not None
        ):
            clap_scores = tagger.score(job.audio.samples, job.audio.sample_rate)
        if job.audio is not None and job.features is not None and (not job.cached or refresh):
            _save_cache(conn, job.audio, job.features, clap_scores)
        elif job.audio is not None and clap_scores and job.cached:
            conn.execute(
                "UPDATE analysis SET clap_json=? WHERE content_hash=?",
                (json.dumps(clap_scores), job.audio.content_hash),
            )
        identity = job.audio.content_hash if job.audio else job.raw_hash
        correction = load_correction(conn, identity)
        learned = None
        if model is not None and embedder is not None and job.audio is not None and not meta.type:
            vector = learn.embedding_for(conn, job.audio, _embed_with(embedder))
            if vector is not None:
                learned = model.predict(vector, job.audio.duration)
        verdict = fuse(meta, job.features, clap_scores, correction, learned)
        verdict.content_hash = job.audio.content_hash if job.audio else None
        verdict.fingerprint = job.audio.fingerprint if job.audio else None
        verdict.raw_hash = job.raw_hash
        verdict.features = job.features
        if job.audio is None:
            verdict.reason = (verdict.reason + " " if verdict.reason else "") + "undecoded"
        verdicts[id(it)] = verdict
    conn.commit()
    if tagger is not None and tagger.error:
        warnings.append(tagger.error + ". Classification used filenames and DSP only.")
    elif embedder is not None and embedder.error:
        warnings.append(embedder.error + ". The trained model was not used.")
    return verdicts, warnings


def _embed_with(tagger: ClapTagger) -> learn.Embedder:
    def embed(audio: AudioData) -> np.ndarray | None:
        return tagger.embed(audio.samples, audio.sample_rate)

    return embed
