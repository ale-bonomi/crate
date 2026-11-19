"""Opt-in filename templates. Unknown tokens are left out, not left blank."""

from __future__ import annotations

import re

from .analyze.types import Verdict

_TOKEN = re.compile(r"\{([a-z0-9_]+)\}")
_UNSAFE = re.compile(r"[\\/:\0]")


def character_of(verdict: Verdict | None) -> str | None:
    if verdict is None:
        return None
    return verdict.punch or verdict.snare_body or verdict.feel or verdict.wet


def template_values(verdict: Verdict | None, original_stem: str) -> dict[str, str | None]:
    bpm = None if verdict is None or verdict.bpm is None else str(verdict.bpm)
    return {
        "type": None if verdict is None else verdict.type,
        "character": character_of(verdict),
        "key": None if verdict is None else verdict.key,
        "bpm": bpm,
        "original": original_stem or None,
    }


def filename_from_template(
    template: str, values: dict[str, str | None], suffix: str
) -> str | None:
    matches = list(_TOKEN.finditer(template))
    if not matches:
        return None
    kept: list[tuple[int, int, str]] = []
    for match in matches:
        raw = values.get(match.group(1))
        if not raw:
            continue
        cleaned = _UNSAFE.sub("", raw).strip()
        if cleaned:
            kept.append((match.start(), match.end(), cleaned))
    if not kept:
        return None
    chunks = [kept[0][2]]
    for prev, cur in zip(kept, kept[1:], strict=False):
        raw_sep = template[prev[1] : cur[0]]
        sep = _TOKEN.sub("", raw_sep)
        if not re.search(r"[^\s_-]", sep):
            sep = "_" if "_" in raw_sep else (sep.strip()[:1] or "_")
        chunks.append(sep)
        chunks.append(cur[2])
    stem = "".join(chunks).strip(" _-.")
    if not stem:
        return None
    return stem + suffix
