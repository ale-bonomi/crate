"""Score predictions against a private labels CSV. Columns: path, type, loop."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Scores:
    per_type: dict[str, tuple[float, float, int]]  # precision, recall, support
    labeled: int


def read_labels(path: Path) -> list[tuple[str, str, bool | None]]:
    rows: list[tuple[str, str, bool | None]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if (
            reader.fieldnames is None
            or "path" not in reader.fieldnames
            or "type" not in reader.fieldnames
        ):
            raise ValueError("labels CSV needs path and type columns")
        for row in reader:
            loop_raw = (row.get("loop") or "").strip().lower()
            loop: bool | None
            if loop_raw in {"1", "true", "yes", "loop"}:
                loop = True
            elif loop_raw in {"0", "false", "no", "oneshot", "one-shot"}:
                loop = False
            else:
                loop = None
            rows.append((row["path"], row["type"].strip().lower(), loop))
    return rows


def score(
    gold: list[tuple[str, str]],
    predicted: list[str | None],
) -> Scores:
    """gold entries are (id, type). predicted aligns by index."""
    labels = sorted({typ for _, typ in gold if typ})
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)
    support: dict[str, int] = defaultdict(int)
    for (_, truth), guess in zip(gold, predicted, strict=True):
        support[truth] += 1
        if guess == truth:
            tp[truth] += 1
        else:
            fn[truth] += 1
            if guess:
                fp[guess] += 1
    per: dict[str, tuple[float, float, int]] = {}
    for label in labels:
        precision = tp[label] / (tp[label] + fp[label]) if tp[label] + fp[label] else 0.0
        recall = tp[label] / (tp[label] + fn[label]) if tp[label] + fn[label] else 0.0
        per[label] = (precision, recall, support[label])
    return Scores(per, len(gold))
