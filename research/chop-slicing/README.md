# Slicing drum hits out of song chops

Research notes, not a feature yet. The aim is to file chops (phrases cut from records) the way any sample is filed, and to cut usable one-shots (kick, snare, hat…) out of them. Nothing here is wired into `crate`. No audio is committed.

## Pipeline being tested

1. **Separate.** [Demucs](https://github.com/facebookresearch/demucs) `htdemucs` splits each chop into drums, bass, other and vocals. Only the drum stem is used. A chop whose drum stem is under 15% of the mix's level is skipped (strings, vocals and ambient chops).
2. **Find hits.** Spectral-flux onsets on the drum stem, at least 70 ms apart.
3. **Label.** Each hit is embedded with CLAP and labelled by the `crate train` model, restricted to drum classes because the hit came from the drum stem.
4. **Cut.** A hit lasts at least a minimum length for its kind (snare and clap 0.5 s, kick 0.35 s, perc 0.3 s, hat 0.18 s), even past the next hit. It then runs until it falls 35 dB below its peak, capped at 1.2 s, with a 60 ms fade-out.
5. **Keep the best.** In each chop, keep at most 2 hits of each kind. Prefer hits where the drum plays alone (no hat highs over a kick, no kick lows under a hat or snare), then confident ones, then ones with little of the song bleeding through.

`slice_chops.py` runs steps 1–5 on a folder of chops. `listening_page.py` builds a page for judging the hits by ear: each hit is marked with every drum heard in it, plus fill, junk and cut-too-short flags.

```bash
uv run --with demucs python research/chop-slicing/slice_chops.py <chops dir> <crate home with model.npz> <out dir> 40
uv run --with demucs python research/chop-slicing/listening_page.py <out dir> <chops dir> page.html 60
```

## Results so far

The test set is 1,448 chops from one producer's collection. Most are 4–20 s phrases; only 2% are single hits. The same 40 chops were picked at random for both rounds: 25 had drums, and the pipeline found 624 hits and kept about 160. 60 kept hits were judged by ear each round.

**Round 1: cuts stopped at the next hit.**

| Crate said | Usable | Wrong label | Junk |
|---|---|---|---|
| kick | 12 / 12 | 0 | 0 |
| snare | 9 | 2 | 1 |
| hat | 10 | 2 | 0 |
| perc | 6 | 6 | 0 |
| clap | 1 | 11 (snares and rims) | 0 |

Only 1 of 60 was junk, so separating and slicing works. But tails were cut far too short: the median "clap" was 0.09 s and the median snare 0.36 s.

**Round 2: minimum lengths per kind, and a preference for drums playing alone.**

| Crate said | Contains that drum | That drum alone | Fill | Junk | Too short |
|---|---|---|---|---|---|
| kick | 12 / 12 | 3 | 1 | 0 | 0 |
| snare | 12 / 12 | 0 | 5 | 0 | 0 |
| hat | 12 / 12 | 10 | 1 | 0 | 0 |
| perc | 5 / 12 | 5 | 3 | 0 | 0 |
| clap | 0 / 12 | 0 | 8 | 0 | 0 |

No junk, and nothing cut too short. The label is right for kick, snare and hat every time (36 / 36). But only 24 of 60 cuts hold a single drum, because the longer cuts swallow the next hit. Across 120 judged hits there was 1 real clap, 6 toms and 1 rim.

## What to change next

- **Two cuts per hit.** Cut each single hit at the next clear transient, with a short fade so the tail doesn't click. Save a separate fill only when several hits fall close together. Snares can ring a little into the next hit; kicks and hats should stop at it.
- **Hits on the same beat** (8 of 12 snares had a hat on the same beat) can't be fixed by where the cut ends. That needs a model that splits the drum stem into kick, snare, toms, hi-hat and cymbals. Evaluate one before building on it.
- **Stop labelling chop hits as claps** until the model has real clap examples.
- **New drum classes:** rim and tom as their own types. Today `crate` merges both into perc when training (`SAME_FOLDER` in `analyze/fusion.py`).
- **Next requested by the owner:** open vs closed hi-hats, and crash vs ride. Hats are already the most reliable class (10 / 12 clean and right). Length and decay (which the cutter already measures) separate open from closed hats, and a ride's bell and ping should be learnable.
- **Use the listening answers as training data.** Each judged hit is a labelled example for `crate train`, and that is how rim, tom and the hat and cymbal types get enough examples.
- **Then build it in:** `crate ingest --slice`. Write the hits as new files into the One Shots folders, record them in the journal so `crate undo` removes them, and leave the chop itself untouched, filed by its main content.

## Cost

On a CPU-only machine, separation, slicing and labelling took about 12 s per chop. Apple Silicon should be faster but hasn't been measured. The `htdemucs` weights download once from Hugging Face.
