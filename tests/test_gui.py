"""The window can serve its page and plan a flat filing without moving anything."""

from __future__ import annotations

import threading
from pathlib import Path
from urllib.request import urlopen

from crate.config import Config, save_config
from crate.gui.server import make_server
from crate.gui.service import Desk


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
