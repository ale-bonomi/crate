"""CLAP glue: works across transformers versions and never fails silently."""

from pathlib import Path

import numpy as np

from crate import db
from crate.analyze import clap as clap_mod
from crate.analyze import engine
from crate.scan import scan
from tests.synth import kick, write_wav


class NewProcessor:  # transformers 5
    def __call__(self, *, audio=None, audios=None, **kw):
        if audios is not None:
            raise ValueError("You passed keyword argument `audios` which is deprecated.")
        return {"audio": audio, **kw}


class OldProcessor:  # transformers 4.x
    def __call__(self, *, audios=None, **kw):
        return {"audios": audios, **kw}


def test_encode_uses_whichever_keyword_the_processor_takes():
    audio = np.zeros(10, dtype=np.float32)
    assert "audio" in clap_mod._encode(NewProcessor(), ["a kick"], audio, 48000)
    assert "audios" in clap_mod._encode(OldProcessor(), ["a kick"], audio, 48000)


def test_clap_failure_becomes_a_warning(tmp_path: Path, monkeypatch):
    def broken_score(self, samples, sr):
        self.error = "CLAP scoring failed: ValueError: boom"
        return None

    monkeypatch.setattr(clap_mod.ClapTagger, "available", lambda self: True)
    monkeypatch.setattr(clap_mod.ClapTagger, "score", broken_score)
    write_wav(tmp_path / "in" / "Kick_01.wav", kick(), 22050)
    items = scan([tmp_path / "in"], tmp_path / "staging").items
    conn = db.connect(tmp_path / "t.db")
    verdicts, warnings = engine.annotate(items, conn, {"kick": ["a kick drum"]}, jobs=1)
    assert any("CLAP scoring failed" in w for w in warnings)
    assert next(iter(verdicts.values())).type == "kick"  # still classified without CLAP
