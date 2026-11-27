"""`crate train`: labels from names, held-out trust, the saved model, and how it is used."""

from pathlib import Path

import numpy as np
import pytest

from crate import db, learn
from crate.analyze import clap as clap_mod
from crate.analyze.engine import annotate, save_correction
from crate.analyze.features import extract
from crate.analyze.fusion import fuse
from crate.analyze.metadata import Meta, parse
from crate.analyze.types import Correction
from crate.scan import scan
from tests.synth import kick, snare, write_wav

DIM = 16


def _examples(spec: dict[str, list[str]], per_pack: int = 15, seed: int = 0):
    """spec: label -> packs it appears in. Each label is a tight cluster in embedding space."""
    rng = np.random.default_rng(seed)
    centers = {label: rng.normal(size=DIM) * 4 for label in spec}
    out = []
    for label, packs in spec.items():
        for pack in packs:
            for i in range(per_pack):
                vec = centers[label] + rng.normal(size=DIM) * 0.3
                out.append(learn.Example(f"{label}{pack}{i}", label, pack, "name", 0.5, vec))
    return out, centers


def test_labels_come_from_clear_names_and_corrections():
    assert learn.label_for(parse("Kick_01", ("Kicks",)), None) == ("kick", "name")
    assert learn.label_for(parse("mystery_01", ("Pack",)), None) is None
    weak = parse("Phrase", ("Pack", "Vox", "Old", "Older", "Oldest"))  # far-off folder
    assert weak.type == "vocal" and weak.confidence < 0.6
    assert learn.label_for(weak, None) is None
    assert learn.label_for(parse("Kick_01", ()), "snare") == ("snare", "correction")
    # types that share a library folder are learned as one
    assert learn.label_for(parse("Crash_01", ("Cymbals",)), None) == ("hat", "name")
    assert learn.label_for(parse("Tom 2", ()), None)[0] == "perc"
    assert learn.label_for(parse("808_Glide", ()), "808") == ("bass", "correction")


def test_types_seen_in_several_packs_are_trusted():
    examples, _ = _examples(
        {"kick": ["A", "B", "C"], "snare": ["A", "B", "C"], "vocal": ["A"], "fx": ["B"]}
    )
    examples += _examples({"perc": ["C"]}, per_pack=3, seed=1)[0]  # too few to train on
    model, reports = learn.fit(examples, min_examples=20)
    by_label = {r.label: r for r in reports}
    assert model.trusted == {"kick", "snare"}
    assert by_label["kick"].precision == 1.0 and by_label["kick"].packs == 3
    assert not by_label["vocal"].trusted  # one pack only: can't check it on unseen packs
    assert "perc" not in model.classes  # below MIN_CLASS, left out of training


def test_needs_two_kinds_of_sound():
    examples, _ = _examples({"kick": ["A", "B"]})
    with pytest.raises(learn.TrainError):
        learn.fit(examples)


def test_saved_model_predicts_the_same(tmp_path: Path):
    examples, centers = _examples({"kick": ["A", "B"], "snare": ["A", "B"], "hat": ["A", "B"]})
    model, _ = learn.fit(examples, min_examples=20)
    path = model.save(tmp_path / "model.npz")
    again = learn.Model.load(path)
    probe = centers["hat"]
    assert np.allclose(model.probabilities(probe, 0.5), again.probabilities(probe, 0.5))
    assert again.predict(probe, 0.5)[:1] == ("hat",) and again.trusted == model.trusted
    assert learn.load_model(False, path) is None
    assert learn.load_model(True, tmp_path / "missing.npz") is None
    (tmp_path / "junk.npz").write_bytes(b"not a model")
    assert learn.load_model(True, tmp_path / "junk.npz") is None


def test_two_class_model():
    examples, centers = _examples({"kick": ["A", "B"], "snare": ["A", "B"]})
    model, _ = learn.fit(examples, min_examples=20)
    label, prob, _ = model.predict(centers["snare"], 0.5)
    assert label == "snare" and prob > 0.9


def test_low_probability_is_not_trusted():
    model = learn.Model(["kick", "snare"], np.zeros((1, DIM + 1)), np.zeros(1), {"kick"}, 0.7)
    label, prob, ok = model.predict(np.ones(DIM), 0.5)
    assert prob == pytest.approx(0.5) and not ok


def test_learned_guess_only_fills_a_gap():
    trusted = ("kick", 0.9, True)
    v = fuse(Meta(), None, None, None, trusted)
    assert v.type == "kick" and v.confidence == pytest.approx(0.9) and "learned:kick" in v.reason
    unsure = fuse(Meta(), None, None, None, ("vocal", 0.8, False))
    assert unsure.type == "vocal" and unsure.confidence == 0.0  # review, with the guess
    assert "not trusted" in unsure.reason
    # an untrusted guess the spectrum independently agrees with is filed
    kick_feats = extract(kick(), 22050)
    agreed = fuse(Meta(), kick_feats, None, None, ("kick", 0.6, False))
    assert agreed.type == "kick" and agreed.confidence >= 0.6 and "agrees" in agreed.reason
    disagreed = fuse(Meta(), kick_feats, None, None, ("vocal", 0.6, False))
    assert disagreed.type == "vocal" and disagreed.confidence == 0.0
    named = parse("Snare_01", ("Snares",))
    assert fuse(named, None, None, None, trusted).type == "snare"  # names win
    fixed = fuse(Meta(), None, None, Correction("hat", False), trusted)
    assert fixed.type == "hat"  # a correction wins over everything


def _fake_embedder(calls: list[str]):
    def embed(audio):
        calls.append(audio.content_hash)
        f = extract(audio.samples, audio.sample_rate, audio.duration)
        return np.array([f.centroid / 1000, f.low, f.decay, f.tonalness, f.crest / 10], np.float32)

    return embed


def test_collect_labels_named_files_and_caches_embeddings(tmp_path: Path):
    write_wav(tmp_path / "in" / "Kicks" / "Kick_01.wav", kick(), 22050)
    write_wav(tmp_path / "in" / "Snares" / "Snare_01.wav", snare(decay=40), 22050)
    write_wav(tmp_path / "in" / "mystery.wav", snare(decay=6), 22050)
    items = scan([tmp_path / "in"], tmp_path / "staging").items
    conn = db.connect(tmp_path / "t.db")
    calls: list[str] = []
    got = learn.collect(items, conn, _fake_embedder(calls))
    assert sorted(e.label for e in got.examples) == ["kick", "snare"]
    assert got.skipped["name does not say what it is"] == 1
    assert len(calls) == 2  # the unnamed file is not embedded for training
    learn.collect(items, conn, _fake_embedder(calls))
    assert len(calls) == 2  # second run reads the cache

    # A correction turns the unnamed file into a training example.
    mystery = next(it for it in items if it.path.name == "mystery.wav")
    verdicts, _ = annotate([mystery], conn, {}, jobs=1, use_clap=False)
    save_correction(conn, next(iter(verdicts.values())).content_hash, Correction("snare", False))
    again = learn.collect(items, conn, _fake_embedder(calls))
    assert {e.source for e in again.examples} == {"name", "correction"}


def test_annotate_uses_the_model_for_unnamed_files(tmp_path: Path, monkeypatch):
    write_wav(tmp_path / "in" / "s0001.wav", kick(), 22050)
    write_wav(tmp_path / "in" / "Snare_01.wav", kick(), 22050)  # named: the name wins
    items = scan([tmp_path / "in"], tmp_path / "staging").items
    conn = db.connect(tmp_path / "t.db")
    monkeypatch.setattr(clap_mod.ClapTagger, "can_embed", lambda self: True)
    monkeypatch.setattr(clap_mod.ClapTagger, "embed", lambda self, s, sr: np.ones(4, np.float32))
    # A model that always says "perc" with high probability.
    coef = np.zeros((2, 5))
    model = learn.Model(["kick", "perc"], coef, np.array([-5.0, 5.0]), {"perc"}, 0.7)
    verdicts, warnings = annotate(items, conn, {}, jobs=1, use_clap=False, model=model)
    by_name = {it.path.name: verdicts[id(it)] for it in items}
    assert by_name["s0001.wav"].type == "perc" and "learned:perc" in by_name["s0001.wav"].reason
    assert by_name["Snare_01.wav"].type == "snare"
    assert warnings == []
    assert learn.load_embedding(conn, by_name["s0001.wav"].content_hash) is not None


def test_train_explains_what_to_install(tmp_path: Path, monkeypatch):
    from typer.testing import CliRunner

    from crate.cli import app

    monkeypatch.setattr(clap_mod, "installed", lambda: False)
    (tmp_path / "pack").mkdir()
    result = CliRunner().invoke(app, ["train", str(tmp_path / "pack")])
    assert result.exit_code == 1
    assert "uv sync --extra ml" in result.output
