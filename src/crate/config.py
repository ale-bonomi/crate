"""Config schema (pydantic) and YAML load/save."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

CRATE_HOME = Path("~/.crate").expanduser()


class RenameConfig(BaseModel):
    enabled: bool = False
    template: str = "{type}_{character}_{key}_{bpm}_{original}"


class Category(BaseModel):
    folder: str
    split_by: str | None = None


def _default_one_shots() -> dict[str, Category]:
    return {
        "kick": Category(folder="One Shots/Kicks", split_by="punch"),
        "snare": Category(folder="One Shots/Snares", split_by="snare_body"),
        "clap": Category(folder="One Shots/Claps"),
        "hat": Category(folder="One Shots/Hats & Cymbals"),
        "cymbal": Category(folder="One Shots/Hats & Cymbals"),
        "perc": Category(folder="One Shots/Percussion"),
        "tom": Category(folder="One Shots/Percussion"),
        "rim": Category(folder="One Shots/Percussion"),
        "bass": Category(folder="One Shots/808 & Bass"),
        "808": Category(folder="One Shots/808 & Bass"),
        "melodic": Category(folder="One Shots/Melodic"),
        "vocal": Category(folder="One Shots/Vocals"),
        "fx": Category(folder="One Shots/FX"),
    }


def _default_loops() -> dict[str, Category]:
    return {
        "drums": Category(folder="Loops/Drums", split_by="feel"),
        "perc": Category(folder="Loops/Tops & Percs", split_by="feel"),
        "bass": Category(folder="Loops/Bass"),
        "melodic": Category(folder="Loops/Melodic"),
        "vocal": Category(folder="Loops/Vocals"),
        "fx": Category(folder="Loops/FX & Textures"),
        "fill": Category(folder="Loops/Drum Fills"),
    }


class Taxonomy(BaseModel):
    one_shots: dict[str, Category] = Field(default_factory=_default_one_shots)
    loops: dict[str, Category] = Field(default_factory=_default_loops)


# First path is the usual install. Later paths are tried only when a folder is already there.
PRESET_CANDIDATES: dict[str, tuple[str, ...]] = {
    ".fxp": (
        "/Library/Audio/Presets/Xfer Records/Serum 2 Presets/Presets/User",
        "~/Library/Audio/Presets/Xfer Records/Serum 2 Presets/Presets/User",
        "/Library/Audio/Presets/Xfer Records/Serum Presets/Presets/User",
        "~/Library/Audio/Presets/Xfer Records/Serum Presets/Presets/User",
    ),
    "serum": (
        "/Library/Audio/Presets/Xfer Records/Serum Presets/Presets/User",
        "~/Library/Audio/Presets/Xfer Records/Serum Presets/Presets/User",
    ),
    ".vital": ("~/Music/Vital/User/Presets",),
    ".omnisphere": (
        "~/Library/Application Support/Spectrasonics/STEAM/"
        "Omnisphere/Settings Library/Patches/User",
        "/Users/Shared/Spectrasonics/STEAM/Omnisphere/Settings Library/Patches/User",
    ),
    ".pig": (
        "/Library/Arturia/Presets/Pigments/User",
        "~/Library/Arturia/Presets/Pigments/User",
    ),
    ".pgtx": (
        "/Library/Arturia/Presets/Pigments/User",
        "~/Library/Arturia/Presets/Pigments/User",
    ),
    ".nmsv": (
        "~/Documents/Native Instruments/Massive/Sounds",
        "~/Library/Application Support/Native Instruments/Massive/Sounds",
    ),
    ".mxpreset": ("~/Documents/Native Instruments/User Content/Massive X/Presets",),
    ".phaseplant": (
        "~/Documents/Kilohearts/presets/Phase Plant/User Presets",
        "~/Library/Audio/Presets/Kilohearts/Phase Plant",
    ),
    ".spf": (
        "~/Library/Application Support/RevealSound/Presets",
        "~/Documents/Reveal Sound/Presets",
    ),
    ".sbf": (
        "~/Library/Application Support/RevealSound/Presets",
        "~/Documents/Reveal Sound/Presets",
    ),
    "sylenth1": ("~/Library/Application Support/LennarDigital/Sylenth1/Presets",),
    "dune": (
        "~/Library/Application Support/Synapse Audio/DUNE 3/Presets",
        "~/Documents/Synapse Audio/DUNE 3/Presets",
    ),
    "diva": (
        "~/Library/Audio/Presets/u-he/Diva",
        "/Library/Audio/Presets/u-he/Diva",
    ),
    "hive": (
        "~/Library/Audio/Presets/u-he/Hive",
        "/Library/Audio/Presets/u-he/Hive",
    ),
    "zebra2": (
        "~/Library/Audio/Presets/u-he/Zebra2",
        "/Library/Audio/Presets/u-he/Zebra2",
    ),
}


def default_presets() -> dict[str, str]:
    presets = {key: paths[0] for key, paths in PRESET_CANDIDATES.items()}
    presets[".adv"] = "<Ableton User Library>/Presets"
    presets[".adg"] = "<Ableton User Library>/Presets"
    presets["unknown"] = "_Presets Unsorted"
    return presets


class NonAudio(BaseModel):
    midi: str = "MIDI"
    presets: dict[str, str] = Field(default_factory=default_presets)
    other: str = "leave"


def _default_prompts() -> dict[str, list[str]]:
    return {
        "kick": ["a punchy electronic kick drum", "a bass drum hit"],
        "snare": ["a snare drum hit"],
        "snare_tight": ["a short tight snare", "a crisp UK garage rimshot"],
        "snare_body": ["a big snare with a long body and low-mid thump"],
        "clap": ["a hand clap"],
        "hat": ["a closed hi-hat", "an open hi-hat"],
        "perc": ["a percussion hit, shaker, conga or cowbell"],
        "808": ["an 808 sub bass hit"],
        "bass": ["a bass note or bass one-shot"],
        "melodic": ["a musical synth, keys, pluck or pad note"],
        "vocal": ["a vocal chop or sung note"],
        "fx": ["a sound effect, riser, impact or texture"],
        "loop_drums": ["a drum loop"],
        "loop_bass": ["a bass loop"],
        "loop_melodic": ["a melodic chord loop"],
        "loop_vocal": ["a vocal loop"],
        "loop_fx": ["an ambient texture loop"],
    }


class SliceConfig(BaseModel):
    """`ingest --slice`: cut drum hits out of loops and chops (see crate.slicer)."""

    enabled: bool = False  # on for every ingest/sort without passing --slice
    keep_per_kind: int = Field(2, ge=0)  # cleanest N of each kind per file; 0 keeps every hit
    fills: bool = True  # also save quick runs of hits whole, to Loops/Drum Fills
    min_drum_share: float = Field(0.15, ge=0, le=1)  # quieter drums than this: not sliced
    min_source_seconds: float = Field(1.0, ge=0)  # shorter files are already one-shots
    max_source_seconds: float = Field(300.0, gt=0)  # only the first 5 minutes of a file


class Config(BaseModel):
    library_roots: list[Path] = Field(default_factory=lambda: [Path("~/Music/Samples Library")])
    splice_folder: Path = Path("~/Splice/Samples")
    external_volumes: bool = True
    review_threshold: float = Field(0.6, ge=0, le=1)
    split_folder_threshold: int = Field(300, ge=1)
    max_depth: int = Field(3, ge=1, le=3)
    rename: RenameConfig = Field(default_factory=RenameConfig)
    duplicates: str = Field("trash", pattern="^(trash|quarantine|keep|symlink)$")
    # True: copy new files and leave originals (keeps Splice's "downloaded" state).
    keep_originals: bool = True
    taxonomy: Taxonomy = Field(default_factory=Taxonomy)
    non_audio: NonAudio = Field(default_factory=NonAudio)
    ableton_user_library: Path | None = None
    clap_prompts: dict[str, list[str]] = Field(default_factory=_default_prompts)
    clap_enabled: bool = True
    # Use the model from `crate train` (~/.crate/model.npz) for files whose names say nothing.
    use_trained_model: bool = True
    slice: SliceConfig = Field(default_factory=SliceConfig)
    project_folders: list[Path] = Field(default_factory=list)

    @property
    def root(self) -> Path:
        return self.library_roots[0].expanduser()


def default_config_path() -> Path:
    return CRATE_HOME / "config.yaml"


def load_config(path: Path | None = None) -> Config:
    path = path or default_config_path()
    if not path.exists():
        return Config()
    data = yaml.safe_load(path.read_text()) or {}
    return Config.model_validate(data)


def save_config(cfg: Config, path: Path | None = None) -> Path:
    path = path or default_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False))
    return path
