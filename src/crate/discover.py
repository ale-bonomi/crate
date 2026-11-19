"""Find Ableton's User Library and common plugin preset folders on macOS."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .config import PRESET_CANDIDATES, default_presets

_PROJECT_DIR = re.compile(
    r"<UserLibrary>.*?<ProjectDir\s+Value=\"([^\"]+)\"",
    re.DOTALL,
)
_ANY_PROJECT_DIR = re.compile(r"<ProjectDir\s+Value=\"([^\"]+)\"")


@dataclass(frozen=True)
class Detected:
    user_library: Path | None
    presets: dict[str, str]


def _read_text(path: Path) -> str:
    data = path.read_bytes()
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        return data.decode("utf-16", errors="replace")
    return data.decode("utf-8", errors="replace")


def _version_key(path: Path) -> tuple[int, ...]:
    return tuple(int(n) for n in re.findall(r"\d+", path.parent.name))


def ableton_user_library(preferences: Path | None = None) -> Path | None:
    """Newest Live `Library.cfg` wins. A missing explicit folder does not guess."""
    explicit = preferences is not None
    root = preferences or Path.home() / "Library/Preferences/Ableton"
    if root.is_dir():
        configs = sorted(root.glob("Live */Library.cfg"), key=_version_key)
        for cfg in reversed(configs):
            text = _read_text(cfg)
            match = _PROJECT_DIR.search(text) or _ANY_PROJECT_DIR.search(text)
            if match:
                return Path(match.group(1))
    if explicit:
        return None
    default = Path.home() / "Music/Ableton/User Library"
    if default.is_dir():
        return default
    return None


def _first_dir(candidates: list[Path]) -> Path | None:
    for path in candidates:
        expanded = path.expanduser()
        if expanded.is_dir():
            return expanded
    return None


def preset_map(user_library: Path | None) -> dict[str, str]:
    """Usual install path for each synth, replaced when that folder is already on disk."""
    presets = default_presets()
    for key, candidates in PRESET_CANDIDATES.items():
        found = _first_dir([Path(candidate) for candidate in candidates])
        if found is not None:
            presets[key] = str(found)
    if user_library is not None:
        presets[".adv"] = str(Path(user_library).expanduser() / "Presets")
        presets[".adg"] = str(Path(user_library).expanduser() / "Presets")
    return presets


def detect(preferences: Path | None = None) -> Detected:
    library = ableton_user_library(preferences)
    return Detected(library, preset_map(library))
