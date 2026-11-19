# Crate

Crate files a sample library into a shallow folder tree Ableton Live can browse, and keeps an index so you can find a sound without Splice's search. Filenames stay put, so Live's File Manager can still recover old sets. Every change is a dry run until you pass `--apply`, and `crate undo` reverses a batch.

Classification uses three votes. The filename and parent folders supply type, loop, BPM, and key. DSP measures duration, onsets, crest, attack, decay, spectrum, tempo, key, and swing. If you install the optional model, CLAP scores the audio against prompts in the config. Agreement is high confidence. Disagreement trusts the audio unless the filename is explicit. Below the review threshold (default 0.6) the file goes to `_Review/`. A label you set with `crate correct` wins forever. Features are cached by a hash of the decoded audio, so changing a folder rule does not re-analyze the library.

## Setup

You need [uv](https://docs.astral.sh/uv/). If you don't have it:

```bash
python3 -m pip install --user uv
```

If `uv` isn't found afterwards, add pip's user bin folder to your PATH (the path depends on your Python version):

```bash
export PATH="$HOME/Library/Python/3.10/bin:$PATH"
```

Install the project and its dev tools (uv downloads Python 3.12 if needed):

```bash
uv sync --python 3.12
```

Optional extras:

```bash
uv sync --extra ml    # CLAP zero-shot tags (downloads a model on first use)
uv sync --extra mac   # menu-bar status for crate watch
```

## Usage

Create the config and database. It asks for your library root, Splice folder, and an optional Ableton project folder:

```bash
uv run crate init
```

Preview where files would go. This is a dry run and changes nothing:

```bash
uv run crate sort "/path/to/Some Pack"
```

Apply the plan:

```bash
uv run crate sort "/path/to/Some Pack" --apply
```

Other commands:

```bash
uv run crate ingest "/path/to/Some Pack" --apply   # unpack zips, classify, and sort
uv run crate reorganize --apply                    # re-file the library after a rule change
uv run crate reorganize --section "One Shots/Kicks"
uv run crate analyze "/path/to/Some Pack"          # cache features, move nothing
uv run crate unpack "/path/to/Some Pack"           # flatten folders and extract zips
uv run crate find --type kick --key Amin --min-bpm 90 --max-bpm 140
uv run crate find --similar "/path/to/a/kick.wav"  # nearest files already in the library
uv run crate where Kick_01                         # original name or current name
uv run crate dedupe                                # exact copies; near-duplicates are only reported
uv run crate review                                # low-confidence files
uv run crate correct "/path/to/file.wav" snare:tight --apply
uv run crate projects "/path/to/Sets" --apply      # which samples Live sets use
uv run crate watch                                 # plan new Splice files; --apply files them
uv run crate eval labels.csv                       # precision/recall from a private CSV
uv run crate status                                # "12 filed, 3 need review", plus batches
uv run crate undo                                  # reverse the latest batch
uv run crate undo 3                                # reverse a specific batch
```

Labels look like `kick`, `snare:tight`, `kick:punchy`, `loop:drums`, or `oneshot:hat`.

`watch` leaves files that are already in the Splice folder alone. `--backfill` ingests those too. `--menu` puts `12 filed, 3 need review` in the menu bar when the `mac` extra is installed.

Every command that moves files is a dry run unless you pass `--apply`. Filenames stay as they are unless `rename.enabled` is true in the config. The template is `{type}_{character}_{key}_{bpm}_{original}`, and any token that is unknown is left out. `.asd` files move with their sample, including under a new name. MIDI goes to `MIDI/`.

Presets go to the folder that synth normally uses. `crate init` detects a folder that is already there, otherwise it offers the usual path, and you can type a different one.

| Files | Folder |
| --- | --- |
| `.fxp` | Serum 2 user presets, or Serum 1 if that is the install on disk |
| `.fxp` / `.fxb` named Serum | Serum 1 user presets |
| `.vital` | Vital user presets |
| `.omnisphere` | Omnisphere user patches |
| `.pig`, `.pgtx` | Pigments user presets |
| `.nmsv` | Massive Sounds |
| `.mxpreset` | Massive X user presets |
| `.phaseplant` | Phase Plant user presets |
| `.fxp` / `.fxb` named Sylenth | Sylenth1 presets |
| `.spf`, `.sbf` | Spire user presets |
| `.h2p` named Diva, Hive, or Zebra | that u-he synth's user presets |
| `.fxp` / `.fxb` named Dune | Dune 3 presets |
| `.adv`, `.adg` | Ableton User Library Presets |

Anything else, including a `.fxb` or `.h2p` whose pack name does not say which synth it belongs to, goes to `_Presets Unsorted/`. Nexus and Avenger sell expansions rather than loose preset files, so they are not moved.

`reorganize` moves files already in the library and reuses cached analysis. It leaves `_Duplicates/` alone unless you pass that folder as `--section`.

By default `ingest` and `sort` copy files out of the Splice folder and leave the originals in place (`keep_originals` in the config). A second copy of the same audio follows `duplicates`: `trash` (default), `quarantine`, `keep`, or `symlink`. A file that only *sounds* the same (another sample rate or gain) is reported and kept.

Large folders split into Punchy/Soft, Body/Tight, or Straight/Swung only after about 300 files. There are no BPM folders.

Use a different config file with `--config`:

```bash
uv run crate --config ./my-config.yaml sort "/path/to/Some Pack"
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the module map.

## Development

```bash
uv run pytest        # tests
uv run ruff check .  # lint
uv run mypy          # type check
```

Tests synthesize kicks, snares, click loops, and chords. Don't commit copyrighted samples. `crate eval` is how you score the classifier against your own labels (`path,type,loop`).
