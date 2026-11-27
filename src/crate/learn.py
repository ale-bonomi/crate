"""`crate train`: a small classifier on CLAP audio embeddings, learned from your own library.

Labels come from files whose names say what they are (a filename or folder word at the
confidence Crate would file it by, 0.6) and from every manual correction. Nothing is
labeled by hand for training. The model is then used only for files whose names say
nothing. Types that share a library folder (hat and cymbal, perc and tom) are learned
as one, so each has more examples.

Each sound type is checked on packs the model did not train on. A type is "trusted" only
when it has enough examples from at least two packs and its held-out precision passes
`min_precision`. A guess for an untrusted type, or below `gate`, goes to `_Review/` with
the guess in the reason instead of being filed.

The saved model is plain arrays (`model.npz`), so using it needs numpy only. Training
needs scikit-learn and, to compute embeddings, the CLAP model (`uv sync --extra ml`).
"""

from __future__ import annotations

import importlib
import json
import math
import sqlite3
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .analyze import metadata
from .analyze.audio import AudioData, load
from .analyze.clap import MODEL_ID
from .analyze.engine import load_correction
from .analyze.fusion import SAME_FOLDER
from .config import CRATE_HOME
from .scan import Item

MODEL_PATH = CRATE_HOME / "model.npz"
NAME_CONFIDENCE = 0.6  # the bar Crate files a name by (default review_threshold)
MIN_CLASS = 5  # fewer examples than this and a type is left out of training entirely

Embedder = Callable[[AudioData], "np.ndarray | None"]


class TrainError(RuntimeError):
    pass


@dataclass
class Example:
    content_hash: str
    label: str
    pack: str
    source: str  # "name" | "correction"
    duration: float
    embedding: np.ndarray


@dataclass
class ClassReport:
    label: str
    examples: int
    packs: int
    precision: float | None  # held-out, among guesses at or above the gate
    recall: float | None  # held-out share of this type guessed right at or above the gate
    trusted: bool


@dataclass
class Model:
    classes: list[str]
    coef: np.ndarray
    intercept: np.ndarray
    trusted: set[str]
    gate: float
    embedding_model: str = MODEL_ID
    info: dict[str, object] = field(default_factory=dict)

    def probabilities(self, embedding: np.ndarray, duration: float) -> np.ndarray:
        x = row(embedding, duration)
        z = self.coef @ x + self.intercept
        if self.coef.shape[0] == 1:  # scikit-learn stores one row for two classes
            p1 = 1.0 / (1.0 + math.exp(-float(z[0])))
            return np.array([1.0 - p1, p1])
        z = z - z.max()
        e = np.exp(z)
        result: np.ndarray = e / e.sum()
        return result

    def predict(self, embedding: np.ndarray, duration: float) -> tuple[str, float, bool]:
        """(type, probability, whether Crate may file it without review)."""
        p = self.probabilities(embedding, duration)
        k = int(p.argmax())
        label, prob = self.classes[k], float(p[k])
        return label, prob, label in self.trusted and prob >= self.gate

    def save(self, path: Path = MODEL_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "classes": self.classes,
            "trusted": sorted(self.trusted),
            "gate": self.gate,
            "embedding_model": self.embedding_model,
            "info": self.info,
        }
        with path.open("wb") as fh:
            np.savez(fh, coef=self.coef, intercept=self.intercept, meta=np.array(json.dumps(meta)))
        return path

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> Model:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            return cls(
                classes=list(meta["classes"]),
                coef=data["coef"],
                intercept=data["intercept"],
                trusted=set(meta["trusted"]),
                gate=float(meta["gate"]),
                embedding_model=str(meta.get("embedding_model", MODEL_ID)),
                info=dict(meta.get("info", {})),
            )


def load_model(enabled: bool = True, path: Path = MODEL_PATH) -> Model | None:
    """The trained model, or None when it is switched off, missing, or unreadable."""
    if not enabled or not path.exists():
        return None
    try:
        return Model.load(path)
    except (OSError, ValueError, KeyError):
        return None


def row(embedding: np.ndarray, duration: float) -> np.ndarray:
    """Unit-length embedding plus log duration: length separates hits from loops."""
    v = np.asarray(embedding, dtype=np.float64)
    norm = float(np.linalg.norm(v)) or 1.0
    return np.append(v / norm, math.log1p(max(duration, 0.0)))


def label_for(meta: metadata.Meta, correction_type: str | None) -> tuple[str, str] | None:
    if correction_type:
        return SAME_FOLDER.get(correction_type, correction_type), "correction"
    if meta.type and meta.confidence >= NAME_CONFIDENCE:
        return SAME_FOLDER.get(meta.type, meta.type), "name"
    return None


# ------------------------------------------------------------------ embedding cache


def load_embedding(conn: sqlite3.Connection, content_hash: str) -> np.ndarray | None:
    found = conn.execute(
        "SELECT embedding FROM analysis WHERE content_hash=?", (content_hash,)
    ).fetchone()
    if found is None or found["embedding"] is None:
        return None
    return np.frombuffer(found["embedding"], dtype=np.float32).copy()


def save_embedding(conn: sqlite3.Connection, audio: AudioData, embedding: np.ndarray) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO analysis(content_hash, raw_hash, duration, sample_rate) "
        "VALUES (?,?,?,?)",
        (audio.content_hash, audio.raw_hash, audio.duration, audio.sample_rate),
    )
    conn.execute(
        "UPDATE analysis SET embedding=? WHERE content_hash=?",
        (np.asarray(embedding, dtype=np.float32).tobytes(), audio.content_hash),
    )


def embedding_for(conn: sqlite3.Connection, audio: AudioData, embed: Embedder) -> np.ndarray | None:
    cached = load_embedding(conn, audio.content_hash)
    if cached is not None:
        return cached
    vector = embed(audio)
    if vector is not None:
        save_embedding(conn, audio, vector)
    return vector


# ------------------------------------------------------------------ training


@dataclass
class Collected:
    examples: list[Example] = field(default_factory=list)
    skipped: Counter[str] = field(default_factory=Counter)


def collect(
    items: Iterable[Item],
    conn: sqlite3.Connection,
    embed: Embedder,
    progress: Callable[[], None] | None = None,
) -> Collected:
    """Label every audio item whose name (or a correction) says what it is, and embed it."""
    out = Collected()
    seen: set[str] = set()
    for item in items:
        if progress is not None:
            progress()
        if item.kind != "audio":
            continue
        meta = metadata.parse(item.path.stem, item.folders)
        try:
            audio = load(item.path)
        except OSError:
            audio = None
        if audio is None:
            out.skipped["could not be decoded"] += 1
            continue
        if audio.content_hash in seen:
            out.skipped["duplicate audio"] += 1
            continue
        seen.add(audio.content_hash)
        correction = load_correction(conn, audio.content_hash)
        labeled = label_for(meta, correction.type if correction else None)
        if labeled is None:
            out.skipped["name does not say what it is"] += 1
            continue
        vector = embedding_for(conn, audio, embed)
        if len(seen) % 50 == 0:
            conn.commit()  # an interrupted run keeps what it has heard
        if vector is None:
            out.skipped["could not be embedded"] += 1
            continue
        label, source = labeled
        pack = item.pack or (item.folders[0] if item.folders else "?")
        out.examples.append(
            Example(audio.content_hash, label, pack, source, audio.duration, vector)
        )
    conn.commit()
    return out


def _sklearn(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise TrainError(
            "Training needs scikit-learn. Install it with: uv sync --extra ml"
        ) from exc


def _classifier() -> Any:
    linear = _sklearn("sklearn.linear_model")
    return linear.LogisticRegression(max_iter=4000, C=4.0, class_weight="balanced")


def fit(
    examples: list[Example],
    *,
    gate: float = 0.7,
    min_examples: int = 20,
    min_precision: float = 0.85,
) -> tuple[Model, list[ClassReport]]:
    counts = Counter(e.label for e in examples)
    keep = {label for label, n in counts.items() if n >= MIN_CLASS}
    data = [e for e in examples if e.label in keep]
    if len(keep) < 2:
        raise TrainError(
            f"Need at least two kinds of sound with {MIN_CLASS}+ clearly named files each; "
            f"found {dict(counts) or 'none'}."
        )
    X = np.stack([row(e.embedding, e.duration) for e in data])
    y = np.array([e.label for e in data])
    packs = np.array([e.pack for e in data])

    held_pred: np.ndarray | None = None
    held_prob: np.ndarray | None = None
    if len(set(packs)) >= 2:
        selection = _sklearn("sklearn.model_selection")
        folds = selection.GroupKFold(n_splits=min(5, len(set(packs))))
        held_pred = np.empty(len(y), dtype=object)
        held_prob = np.zeros(len(y))
        for train_idx, test_idx in folds.split(X, y, packs):
            if len(set(y[train_idx])) < 2:
                held_pred[test_idx] = None
                continue
            clf = _classifier()
            clf.fit(X[train_idx], y[train_idx])
            proba = clf.predict_proba(X[test_idx])
            classes = clf.classes_
            held_pred[test_idx] = classes[proba.argmax(1)]
            held_prob[test_idx] = proba.max(1)

    reports: list[ClassReport] = []
    trusted: set[str] = set()
    for label in sorted(keep):
        n = int((y == label).sum())
        n_packs = len(set(packs[y == label]))
        precision = recall = None
        if held_pred is not None and held_prob is not None:
            confident = held_prob >= gate
            guessed = confident & (held_pred == label)
            hits = int((guessed & (y == label)).sum())
            precision = hits / int(guessed.sum()) if guessed.sum() >= 5 else None
            recall = hits / n
        ok = (
            n >= min_examples
            and n_packs >= 2
            and precision is not None
            and precision >= min_precision
        )
        if ok:
            trusted.add(label)
        reports.append(ClassReport(label, n, n_packs, precision, recall, ok))

    final = _classifier()
    final.fit(X, y)
    model = Model(
        classes=[str(c) for c in final.classes_],
        coef=np.asarray(final.coef_, dtype=np.float64),
        intercept=np.asarray(final.intercept_, dtype=np.float64),
        trusted=trusted,
        gate=gate,
        info={
            "examples": len(data),
            "packs": len(set(packs)),
            "from_corrections": sum(e.source == "correction" for e in data),
        },
    )
    return model, reports
