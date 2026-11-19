"""Shared analysis records. Raw features stay so rules can change without re-reading audio."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Features:
    duration: float
    sample_rate: int
    onset_count: int
    crest: float
    attack: float
    decay: float
    centroid: float
    rolloff: float
    low: float
    mid: float
    high: float
    tonalness: float
    tempo: float | None
    key: str | None
    key_confidence: float
    swing: float | None  # >1 means alternating long/short onset gaps

    def to_json(self) -> dict[str, float | int | str | None]:
        return asdict(self)


@dataclass
class Correction:
    type: str | None
    loop: bool | None
    punch: str | None = None
    snare_body: str | None = None
    feel: str | None = None
    wet: str | None = None


@dataclass
class Verdict:
    type: str | None
    loop: bool | None
    confidence: float
    reason: str
    bpm: int | None
    key: str | None
    punch: str | None
    snare_body: str | None
    feel: str | None
    wet: str | None
    source: str  # metadata | fusion | manual
    content_hash: str | None = None
    fingerprint: str | None = None
    raw_hash: str | None = None
    features: Features | None = None
