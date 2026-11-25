"""Plan output: short paths, and which copy of a duplicate is kept."""

from pathlib import Path

from rich.console import Console

from crate import planner
from crate.render import show_plan, source_base
from crate.scan import ScanResult, scan


def _op(src: str, dst: str, mode: str = "move") -> planner.PlannedOp:
    return planner.PlannedOp(Path(src), Path(dst), mode, "kick", 0.9, src, "Pack")


def test_paths_are_relative_to_pack_and_library():
    root = Path("/lib")
    plan = planner.Plan(
        ops=[
            _op("/in/Pack/Kicks/Kick_01.wav", "/lib/One Shots/Kicks/Kick_01.wav"),
            _op("/in/Pack/Snares/Snare.wav", "/lib/One Shots/Snares/Snare_tight.wav"),
        ]
    )
    assert source_base(plan) == "/in"
    console = Console(record=True, width=200)
    show_plan(console, plan, ScanResult(), apply=False, root=root)
    text = console.export_text()
    assert "Pack/Kicks/Kick_01.wav" in text
    assert "One Shots/Kicks/ " in text  # same name: just the folder
    assert "One Shots/Snares/Snare_tight.wav" in text  # renamed: the full path
    assert "/in/Pack" not in text.split("Plan", 1)[1]  # no absolute paths in the table
    assert "/lib/One Shots" not in text


def test_original_name_sorts_before_its_copies(tmp_path: Path):
    for name in ("Kick copy.wav", "Kick (1).wav", "Kick.wav"):
        (tmp_path / "in" / name).parent.mkdir(exist_ok=True)
        (tmp_path / "in" / name).write_bytes(b"RIFF")
    items = scan([tmp_path / "in"], tmp_path / "staging").items
    assert items[0].path.name == "Kick.wav"
