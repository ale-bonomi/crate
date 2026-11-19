"""Presets, opt-in rename, and re-filing an existing library."""

from pathlib import Path

from typer.testing import CliRunner

from crate import db, mover, planner
from crate.analyze.types import Verdict
from crate.cli import app
from crate.config import Config, RenameConfig, save_config
from crate.discover import Detected, ableton_user_library, preset_map
from crate.naming import filename_from_template
from crate.scan import Item, scan


def _kick(punch: str | None = "punchy", bpm: int | None = None, key: str | None = None) -> Verdict:
    return Verdict(
        type="kick",
        loop=False,
        confidence=0.9,
        reason="test",
        bpm=bpm,
        key=key,
        punch=punch,
        snare_body=None,
        feel=None,
        wet=None,
        source="metadata",
    )


def _item(path: Path, kind: str) -> Item:
    return Item(path, str(path), path.parent.name, (path.parent.name,), kind, False, path.parent)


def test_template_skips_unknown_tokens():
    name = filename_from_template(
        "{type}_{character}_{key}_{bpm}_{original}",
        {"type": "kick", "character": "punchy", "key": None, "bpm": None, "original": "Kick_01"},
        ".wav",
    )
    assert name == "kick_punchy_Kick_01.wav"


def test_presets_and_midi_leave_audio_names(tmp_path: Path):
    lib = tmp_path / "lib"
    vital = tmp_path / "Vital"
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "bass.vital").write_bytes(b"preset")
    (inbox / "Lead.nmsv").write_bytes(b"massive")
    (inbox / "chord.mid").write_bytes(b"midi")
    (inbox / "notes.txt").write_bytes(b"nope")
    cfg = Config(library_roots=[lib], keep_originals=False)
    cfg.non_audio.presets[".vital"] = str(vital)
    res = scan([inbox], tmp_path / "staging")
    assert all(item.kind != "other" for item in res.items)
    plan = planner.build_plan(res.items, cfg, planner.sort_router(cfg), [], None, None)
    by_name = {op.src.name: op for op in plan.ops}
    assert by_name["bass.vital"].dst == vital / "bass.vital"
    assert by_name["Lead.nmsv"].dst == lib / "_Presets Unsorted" / "Lead.nmsv"
    assert by_name["chord.mid"].dst == lib / "MIDI" / "chord.mid"
    assert "notes.txt" not in by_name


def test_other_folder_when_configured(tmp_path: Path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "readme.txt").write_bytes(b"text")
    cfg = Config(library_roots=[tmp_path / "lib"], keep_originals=False)
    cfg.non_audio.other = "_Other"
    res = scan([inbox], tmp_path / "staging", include_other=True)
    plan = planner.build_plan(res.items, cfg, planner.sort_router(cfg), [], None, None)
    assert plan.ops[0].dst == tmp_path / "lib" / "_Other" / "readme.txt"


def test_rename_moves_sidecar_and_second_pass_is_idle(tmp_path: Path):
    lib = tmp_path / "lib"
    src_dir = tmp_path / "in"
    src_dir.mkdir()
    src = src_dir / "Kick_01.wav"
    src.write_bytes(b"kick-one")
    (src_dir / "Kick_01.wav.asd").write_bytes(b"asd")
    cfg = Config(
        library_roots=[lib],
        keep_originals=False,
        rename=RenameConfig(enabled=True),
    )
    conn = db.connect(tmp_path / "test.db")
    item = _item(src, "audio")
    verdict = _kick(bpm=140)
    verdicts = {id(item): verdict}
    plan = planner.build_plan(
        [item], cfg, planner.sort_router(cfg, verdicts), [], verdicts, conn
    )
    audio = next(op for op in plan.ops if not op.sidecar)
    assert audio.dst.name == "kick_punchy_140_Kick_01.wav"
    assert any(op.dst.name == "kick_punchy_140_Kick_01.wav.asd" for op in plan.ops)
    mover.execute(conn, plan, "sort", lib)

    moved = audio.dst
    again = _item(moved, "audio")
    again_verdicts = {id(again): verdict}
    second = planner.build_plan(
        [again], cfg, planner.sort_router(cfg, again_verdicts), [], again_verdicts, conn
    )
    assert second.ops == []


def test_reorganize_follows_a_new_folder_rule(tmp_path: Path, monkeypatch):
    lib = tmp_path / "lib"
    current = lib / "One Shots" / "Kicks" / "Kick_01.wav"
    current.parent.mkdir(parents=True)
    current.write_bytes(b"kick-one")
    dupes = lib / "_Duplicates"
    dupes.mkdir()
    (dupes / "Kick_02.wav").write_bytes(b"other-kick")
    loose = lib / "Loose"
    loose.mkdir()
    (loose / "Kick_03.wav").write_bytes(b"loose-kick")
    cfg = Config(library_roots=[lib], keep_originals=False)
    cfg.taxonomy.one_shots["kick"].folder = "Drums/Kicks"
    path = save_config(cfg, tmp_path / "config.yaml")
    monkeypatch.setattr("crate.cli.CRATE_HOME", tmp_path / "crate")
    result = CliRunner().invoke(
        app, ["--config", str(path), "--jobs", "1", "reorganize", "--apply"]
    )
    assert result.exit_code == 0, result.output
    assert (lib / "Drums" / "Kicks" / "Kick_03.wav").exists()
    assert (lib / "Drums" / "Kicks" / "Kick_01.wav").exists()
    assert (dupes / "Kick_02.wav").exists()
    assert not any(loose.rglob("*.wav"))


def test_init_saves_confirmed_preset_folders(tmp_path: Path, monkeypatch):
    library = tmp_path / "User Library"
    library.mkdir()
    presets = {
        ".fxp": str(tmp_path / "Serum"),
        ".vital": "~/Music/Vital/User/Presets",
        ".adv": "<Ableton User Library>/Presets",
        ".adg": "<Ableton User Library>/Presets",
        "unknown": "_Presets Unsorted",
    }
    monkeypatch.setattr("crate.cli.detect", lambda: Detected(library, dict(presets)))
    monkeypatch.setattr("crate.cli.CRATE_HOME", tmp_path / "crate")
    cfg_path = tmp_path / "config.yaml"
    result = CliRunner().invoke(
        app,
        ["--config", str(cfg_path), "init"],
        input=f"{tmp_path / 'samples'}\n{tmp_path / 'splice'}\n\n{library}\ny\n",
    )
    assert result.exit_code == 0, result.output
    loaded = Config.model_validate(__import__("yaml").safe_load(cfg_path.read_text()))
    assert loaded.non_audio.presets[".fxp"] == str(tmp_path / "Serum")
    assert loaded.non_audio.presets[".adv"] == str(library / "Presets")
    assert loaded.ableton_user_library == library


def test_ableton_library_prefers_the_newest_live(tmp_path: Path):
    old = tmp_path / "Live 11.3.25"
    new = tmp_path / "Live 12.1.5"
    old.mkdir()
    new.mkdir()
    body = """<Ableton><UserLibrary><ProjectDir Value="{path}"/></UserLibrary></Ableton>"""
    (old / "Library.cfg").write_text(body.format(path="/libraries/old"), encoding="utf-8")
    (new / "Library.cfg").write_bytes(
        body.format(path="/libraries/new").encode("utf-16")
    )
    assert ableton_user_library(tmp_path) == Path("/libraries/new")
    mapped = preset_map(Path("/libraries/new"))
    assert mapped[".adg"] == "/libraries/new/Presets"
    assert mapped["unknown"] == "_Presets Unsorted"
