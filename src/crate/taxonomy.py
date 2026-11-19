"""Map a classified item to a destination folder using the config taxonomy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .analyze.metadata import Meta
from .config import Config

REVIEW_DIR = "_Review"

_LOOP_CATEGORY = {
    "kick": "drums", "snare": "drums", "clap": "drums", "hat": "drums", "cymbal": "drums",
    "tom": "drums", "rim": "drums", "drum": "drums",
    "perc": "perc", "top": "perc",
    "bass": "bass", "808": "bass",
    "melodic": "melodic", "vocal": "vocal", "fx": "fx",
}  # fmt: skip
_ONESHOT_CATEGORY = {"drum": "perc", "top": "perc"}


@dataclass(frozen=True)
class Route:
    folder: str  # relative to the library root, posix style
    label: str
    confidence: float


def route_audio(meta: Meta, cfg: Config) -> Route:
    if meta.type is None:
        return Route(REVIEW_DIR, "unknown", 0.0)
    if meta.loop:
        key = _LOOP_CATEGORY.get(meta.type)
        cat = cfg.taxonomy.loops.get(key) if key else None
        label = f"loop/{key}"
    else:
        key = _ONESHOT_CATEGORY.get(meta.type, meta.type)
        cat = cfg.taxonomy.one_shots.get(key)
        label = f"oneshot/{key}"
    if cat is None or meta.confidence < cfg.review_threshold:
        return Route(REVIEW_DIR, label, meta.confidence)
    folder = PurePosixPath(*PurePosixPath(cat.folder).parts[: cfg.max_depth])
    return Route(str(folder), label, meta.confidence)


def route_midi(cfg: Config) -> Route:
    return Route(cfg.non_audio.midi, "midi", 1.0)


ABLETON_LIBRARY = "<Ableton User Library>"


def expand_folder(folder: str, cfg: Config) -> str:
    """Expand ~ and the Ableton library token. Relative folders stay relative."""
    library = cfg.ableton_user_library
    if library is not None:
        folder = folder.replace(ABLETON_LIBRARY, str(Path(library).expanduser()))
    return str(Path(folder).expanduser())


def route_preset(path: Path, cfg: Config) -> Route | None:
    ext = path.suffix.lower()
    dest = cfg.non_audio.presets.get(ext)
    if dest is None:
        dest = cfg.non_audio.presets.get("unknown")
    if not dest or dest == "leave":
        return None
    return Route(expand_folder(dest, cfg), f"preset/{ext.lstrip('.')}", 1.0)


def route_other(cfg: Config) -> Route | None:
    dest = cfg.non_audio.other
    if not dest or dest == "leave":
        return None
    return Route(expand_folder(dest, cfg), "other", 1.0)
