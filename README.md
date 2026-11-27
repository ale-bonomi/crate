# Crate

Crate files a sample library into a shallow folder tree Ableton Live can browse, and keeps an index so you can find a sound without Splice's search. Filenames stay put, so Live's File Manager can still recover old sets. Every change is a dry run until you pass `--apply`, and `crate undo` reverses a batch.

Classification starts from the filename and parent folders, which supply type, loop, BPM, and key. DSP measures duration, onsets, crest, attack, decay, spectrum, tempo, key, and swing. A clear name wins. The audio decides only types it can actually hear, and fills in when the name says nothing. For those unnamed files, a model trained on your own library with `crate train` does much better than the built-in guess (see [Teach Crate your sounds](#teach-crate-your-sounds)). Below the review threshold (default 0.6) the file goes to `_Review/`. A label you set with `crate correct` wins forever. Features are cached by a hash of the decoded audio, so changing a folder rule does not re-analyze the library.

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
uv sync --extra ml    # crate train, and CLAP (downloads a model on first use)
uv sync --extra mac   # menu-bar status for crate watch
uv sync --extra ml --extra slice   # ingest --slice: cut drum hits out of loops and chops
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
uv run crate ingest --slice "/path/to/Chops"       # also cut every drum hit into a one-shot
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
uv run crate train                                 # learn your sounds; needs --extra ml
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

## Slice drum hits out of loops and chops

Cut a section you like out of a record or a YouTube rip, export it, and let Crate cut every drum hit out of it. Kicks land in `One Shots/Kicks`, snares in `One Shots/Snares`, and so on. Quick runs of hits, such as a fill, go whole to `Loops/Drum Fills`. It saves exporting and naming slices one at a time.

Slicing is off unless you ask for it:

```bash
uv sync --python 3.12 --extra ml --extra slice     # once: Demucs, about 80 MB of model weights on first use
uv run crate ingest --slice "/path/to/Chops"       # see the plan
uv run crate ingest --slice "/path/to/Chops" --apply
```

In the Callima window, turn on **Slice drum hits out of loops and chops** on the Organize screen before Go. To slice on every `ingest`, `sort`, and `watch`, set `slice.enabled: true` in `~/.crate/config.yaml`.

How it works:

1. **Separate.** [Demucs](https://github.com/facebookresearch/demucs) pulls the drums out of the mix, so a kick isn't buried under bass and vocals. A file with hardly any drums, such as strings or a vocal, is left alone.
2. **Find and label each hit** as a kick, snare, hat, or perc. The model from `crate train` does this well. Without it, Crate uses a rough guess and sends most slices to `_Review/`, so train first.
3. **Cut.** Each hit stops where the next one starts, so a kick doesn't carry the hat after it. A snare keeps at least 0.3 s so its tail isn't chopped off. Every cut fades in and out instead of clicking.
4. **Keep the best.** For each file Crate keeps the 2 cleanest hits of each kind: the drum alone, confident, with little of the song underneath. Set `slice.keep_per_kind: 0` to keep every hit.

What to expect:

- Slices are new files named after their source, like `Catch The Sun kick 01.wav`. The source file is filed as usual and left as it is.
- Slices are in the plan and the journal like any other file, so `crate undo` removes them.
- Two drums on the same beat (a snare with a hat over it) stay together in one slice. Separating those would need a finer drum-splitting model.
- Claps cut from records almost always turned out to be snares in listening tests, so Crate files them as snares. Rims and toms go to percussion until the model learns them.
- On a laptop CPU it takes about a minute per chop. Apple Silicon is faster. Only the first 5 minutes of a file are sliced (`slice.max_source_seconds`).

Settings, under `slice:` in the config: `enabled`, `keep_per_kind` (2), `fills` (true), `min_drum_share` (0.15), `min_source_seconds` (1), `max_source_seconds` (300).

## Teach Crate your sounds

Most samples are named well enough to file by name alone. `crate train` is for the rest, such as `Sample 12.wav` or `DEMOKIT_8.wav`. It listens to the files in your library whose names *do* say what they are, learns what each kind of sound sounds like, and uses that for files whose names say nothing.

It runs entirely on your Mac. Nothing is uploaded, and it never moves, renames, or changes a file.

**1. Install the extra pieces** (about 2 GB: PyTorch, transformers, and scikit-learn):

```bash
uv sync --python 3.12 --extra ml
```

**2. Train on your library.** With no folders given, it reads your library root and Splice folder from `crate init`:

```bash
uv run crate train
```

Or name the folders to learn from. You can drag them into the terminal, and external drives work:

```bash
uv run crate train ~/Splice/Samples "/Volumes/Samples/Packs" "~/Music/Samples Library"
```

The first run downloads the CLAP audio model (about 600 MB) from Hugging Face, once. It then listens to every file whose name or folder clearly says what it is, such as `Kick_01.wav` or anything in a `Snares/` folder, plus every file you have labeled with `crate correct` or `crate review`. Files with vague names are skipped for training. On a laptop CPU it hears about four files a second, so 10,000 files take roughly 45 minutes the first time. Each file is heard once and cached in `~/.crate/library.db`, so later runs only listen to new files. If you stop it partway, the next run picks up where it left off.

**3. Read the table it prints.** Each sound type gets a row. For example (illustrative numbers):

| Sound | Examples | Packs | Right on unseen packs | Found | Crate files it |
|---|---|---|---|---|---|
| kick | 412 | 37 | 96% | 81% | yes |
| vocal | 18 | 1 | - | - | no, review |

- **Right on unseen packs**: Crate trains on some packs and tests on packs it held back. This is how often its confident guesses were right on sounds it had never seen.
- **Crate files it**: "yes" only when a type has at least 20 examples (`--min-examples`) from two or more packs, and at least 85% of its confident guesses were right. For any other type, the guess goes to `_Review/` with the guess shown, so it is never filed in the wrong folder silently.

The model is saved to `~/.crate/model.npz`, and `sort`, `ingest`, `reorganize`, `analyze`, and the Callima window use it from then on.

**What changes, and what doesn't**

- It only answers for files whose names give no type. A clear filename or folder always wins.
- A label you set with `crate correct` always wins, and it also becomes a training example.
- For a type that isn't trusted yet, the guess is still used when Crate's own audio check independently agrees. When they disagree, the file goes to `_Review/`.
- Guesses below 70% probability (`--gate`) go to `_Review/` even for trusted types.
- With few packs, expect a big `_Review/` pile and very few misfiled files. Trained on just two packs and tested on a third producer's pack with its names hidden, it misfiled 4% of files where Crate alone misfiled 58%, but it sent 90% to review. More packs mean more trusted types.

**Make it better**

- Label what lands in `_Review/` (`uv run crate review`, or `crate correct <file> <label>`), then run `crate train` again.
- Train on packs from many producers. A type seen in only one pack can't be checked on unseen packs, so Crate won't file by it.
- Re-run `crate train` after adding packs. Only new files are listened to.

**Turn it off** by setting `use_trained_model: false` in `~/.crate/config.yaml`, or by deleting `~/.crate/model.npz`.

## Development

```bash
uv run pytest        # tests
uv run ruff check .  # lint
uv run mypy          # type check
```

Tests synthesize kicks, snares, click loops, and chords. Don't commit copyrighted samples. `crate eval` is how you score the classifier against your own labels (`path,type,loop`).
