"""The window can serve its page and plan a flat filing without moving anything."""

from __future__ import annotations

import threading
from pathlib import Path
from urllib.request import urlopen

import pytest

from crate.config import Config, save_config
from crate.gui.server import make_server
from crate.gui.service import Desk, DeskError, describe_flow


def test_flat_plan_keeps_files_put(tmp_path: Path) -> None:
    pack = tmp_path / "pack" / "Kicks"
    pack.mkdir(parents=True)
    (pack / "kick.mid").write_bytes(b"MThd")
    home = tmp_path / "home"
    library = tmp_path / "library"
    library.mkdir()
    desk = Desk(home=home, config_path=home / "config.yaml")
    save_config(
        Config(library_roots=[library], splice_folder=tmp_path / "splice"),
        desk.config_path,
    )
    desk.pack = pack
    desk.action("set-mode", {"mode": "flat"})
    plan, kind = desk._organize_plan(desk.config())
    assert kind == "flat"
    assert len(plan.ops) == 1
    assert plan.ops[0].dst.parent == library
    assert not (library / "kick.mid").exists()


def test_window_serves_the_page(tmp_path: Path) -> None:
    server = make_server(0, home=tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        with urlopen(f"http://127.0.0.1:{port}/") as response:
            page = response.read().decode()
        assert response.status == 200
        assert "Callima" in page
        with urlopen(f"http://127.0.0.1:{port}/api/state") as response:
            assert b"library" in response.read()
    finally:
        server.shutdown()
        server.server_close()


def test_added_folder_and_library_packs_show_up(tmp_path: Path) -> None:
    library = tmp_path / "library"
    (library / "Vengeance Packs").mkdir(parents=True)
    (library / "One Shots").mkdir()
    outside = tmp_path / "Loose"
    (outside / "Kit").mkdir(parents=True)
    home = tmp_path / "home"
    desk = Desk(home=home, config_path=home / "config.yaml")
    save_config(
        Config(library_roots=[library], splice_folder=tmp_path / "splice", clap_enabled=False),
        desk.config_path,
    )
    names = [row["name"] for row in desk.snapshot()["library"]]
    assert names[:4] == [library.name, "splice", "One Shots", "Vengeance Packs"]

    added = desk.action("add-folder", {"path": str(outside) + "/"})
    names = [row["name"] for row in added["library"]]
    assert "Loose" in names
    assert "Kit" in names
    assert added["ends"]["from"]["path"] == str(outside)
    assert names.count(library.name) == 1


def test_every_from_to_pairing_has_a_flow(tmp_path: Path) -> None:
    library = tmp_path / "library"
    library.mkdir()
    splice = tmp_path / "splice"
    splice.mkdir()
    cfg = Config(library_roots=[library], splice_folder=splice, clap_enabled=False)
    pack = tmp_path / "pack"
    (pack / "Kicks").mkdir(parents=True)
    (pack / "kick.mid").write_bytes(b"MThd")
    section = library / "Loops"
    section.mkdir()
    sibling = library / "One Shots"
    sibling.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    moved = cfg.model_copy(update={"keep_originals": False})

    assert describe_flow(None, library, cfg).id == "need-from"
    assert describe_flow(pack, None, cfg).id == "need-to"
    assert describe_flow(tmp_path / "gone", library, cfg).id == "missing-from"
    assert describe_flow(pack / "kick.mid", library, cfg).id == "not-a-folder"
    assert describe_flow(pack, tmp_path / "missing" / "dest", cfg).id == "missing-parent"
    assert describe_flow(pack, pack / "Kicks", cfg).id == "nested"
    assert describe_flow(pack, pack, cfg).id == "in-place"
    assert describe_flow(splice, library, cfg).id == "splice-copy"
    assert describe_flow(splice, outside, cfg).id == "splice-copy-out"
    assert describe_flow(splice, library, moved).id == "splice-move"
    assert describe_flow(section, library, cfg).id == "lift"
    assert describe_flow(section, sibling, cfg).id == "reorganize"
    assert describe_flow(pack, library, cfg).id == "ingest"
    assert describe_flow(section, outside, cfg).id == "export"
    assert describe_flow(pack, outside, cfg).id == "external"
    created = describe_flow(pack, tmp_path / "fresh", cfg)
    assert created.id == "external"
    assert created.creates_dest


def test_to_folder_receives_the_plan_and_splice_is_copied(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "kick.mid").write_bytes(b"MThd")
    splice = tmp_path / "splice"
    splice.mkdir()
    (splice / "hat.mid").write_bytes(b"MThd")
    library = tmp_path / "library"
    library.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    home = tmp_path / "home"
    desk = Desk(home=home, config_path=home / "config.yaml")
    save_config(
        Config(library_roots=[library], splice_folder=splice, clap_enabled=False),
        desk.config_path,
    )
    desk.action("set-from", {"path": str(pack)})
    desk.action("set-to", {"path": str(other)})
    desk.action("set-mode", {"mode": "flat"})
    plan, kind = desk._organize_plan(desk.config())
    assert kind == "flat"
    assert plan.ops[0].dst.parent == other
    assert plan.ops[0].mode == "move"

    desk.action("set-to", {"path": str(pack / "inside")})
    with pytest.raises(DeskError, match="inside From"):
        desk._organize_plan(desk.config())

    desk.action("set-from", {"path": str(splice)})
    desk.action("set-to", {"path": str(library)})
    copied, _kind = desk._organize_plan(desk.config())
    assert copied.ops[0].mode == "copy"
    assert copied.ops[0].dst.parent == library


def test_type_folders_are_created_under_the_to_folder(tmp_path: Path) -> None:
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "kick.mid").write_bytes(b"MThd")
    library = tmp_path / "library"
    library.mkdir()
    shelf = tmp_path / "shelf"
    shelf.mkdir()
    home = tmp_path / "home"
    desk = Desk(home=home, config_path=home / "config.yaml")
    save_config(
        Config(library_roots=[library], splice_folder=tmp_path / "splice", clap_enabled=False),
        desk.config_path,
    )
    desk.action("set-from", {"path": str(pack)})
    desk.action("set-to", {"path": str(shelf)})
    desk.action("set-mode", {"mode": "type"})
    plan, kind = desk._organize_plan(desk.config())
    assert kind == "ingest"
    assert plan.ops[0].dst.parent == shelf / "MIDI"


def test_plan_summary_shows_short_destinations(tmp_path: Path):
    from crate import planner
    from crate.gui.service import _summarize

    root = tmp_path / "Library"
    src = tmp_path / "Pack" / "Kick_01.wav"
    plan = planner.Plan(
        ops=[
            planner.PlannedOp(src, root / "One Shots/Kicks/Kick_01.wav", "move", "kick", 0.9,
                              str(src), "Pack"),  # fmt: skip
            planner.PlannedOp(src, root / "One Shots/Kicks/Kick_01.wav", "trash", "duplicate",
                              1.0, str(src), "Pack"),  # fmt: skip
        ]
    )
    ops = _summarize(plan, root)["ops"]
    assert isinstance(ops, list)
    assert [op["dst"] for op in ops] == ["One Shots/Kicks/", "Trash"]
    assert ops[0]["path"] == str(root / "One Shots/Kicks/Kick_01.wav")
