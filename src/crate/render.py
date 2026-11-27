"""Rich output for plans.

Paths are shown short: sources relative to the folder that was scanned, destinations
relative to the library root, and anything else with `~` for the home folder. The two
base folders are printed once above the table.
"""

from __future__ import annotations

import os
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .planner import Plan, PlannedOp
from .scan import ScanResult


def _note(op: PlannedOp) -> str:
    if op.mode == "extract" and " > " in op.original:  # slicer.MARK
        return "sliced"
    if op.mode in {"trash", "symlink"} and op.kept is not None:
        return f"duplicate of {op.kept.name}"
    if op.renamed:
        return "renamed (collision)"
    original = Path(op.original).name
    if original != op.dst.name:
        return "renamed"
    return ""


def home_short(path: Path | str) -> str:
    text = str(path)
    home = str(Path.home())
    if text == home or text.startswith(home + os.sep):
        return "~" + text[len(home) :]
    return text


def source_base(plan: Plan) -> str | None:
    """The folder that holds everything being filed, so sources can be shown below it."""
    dirs = [os.path.dirname(o.original) for o in plan.ops if not o.sidecar]
    if not dirs:
        return None
    try:
        common = os.path.commonpath(dirs)
    except ValueError:  # mixed absolute and relative, or different drives
        return None
    parent = os.path.dirname(common)
    return parent or None


def short_source(original: str, base: str | None) -> str:
    if base and (original + os.sep).startswith(base + os.sep):
        return os.path.relpath(original, base)
    return home_short(original)


def short_dest(path: Path, root: Path | None) -> str:
    if root is not None:
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            pass
    return home_short(path)


def target_text(op: PlannedOp, root: Path | None) -> str:
    if op.mode == "trash":
        return "Trash"
    if op.mode == "symlink":
        return "link to " + short_dest(op.dst, root)
    if Path(op.original).name == op.dst.name:
        return short_dest(op.dst.parent, root) + "/"  # same name: the folder is enough
    return short_dest(op.dst, root)


def show_plan(
    console: Console,
    plan: Plan,
    scan: ScanResult,
    apply: bool,
    limit: int = 200,
    root: Path | None = None,
) -> None:
    for w in scan.warnings:
        console.print(f"[yellow]warning:[/] {w}")
    base = source_base(plan)
    if base:
        console.print(f"[dim]From[/] {home_short(base)}")
    if root is not None:
        console.print(f"[dim]To[/]   {home_short(root)}")
    table = Table(title="Plan" if apply else "Plan (dry run)", show_lines=False)
    table.add_column("Action")
    table.add_column("From", overflow="fold", ratio=1)
    table.add_column("To", overflow="fold", ratio=1)
    table.add_column("Conf", justify="right")
    table.add_column("Note")
    shown = [o for o in plan.ops if not o.sidecar]
    for o in shown[:limit]:
        table.add_row(
            o.mode, short_source(o.original, base), target_text(o, root), f"{o.confidence:.2f}",
            _note(o),
        )  # fmt: skip
    console.print(table)
    if len(shown) > limit:
        console.print(f"... and {len(shown) - limit} more")

    counts = Table(title="Per destination folder")
    counts.add_column("Folder", overflow="fold")
    counts.add_column("Files", justify="right")
    for folder, n in sorted(plan.per_folder().items()):
        counts.add_row(short_dest(Path(folder), root), str(n))
    console.print(counts)

    for note in plan.notes:
        console.print(f"[yellow]near-duplicate:[/] {note}")

    sidecars = sum(1 for o in plan.ops if o.sidecar)
    console.print(
        f"{len(shown)} files, {sidecars} .asd sidecars, {len(plan.skipped)} skipped"
        f" ({len(scan.skipped)} ignored during scan)"
    )
    if not apply:
        console.print("[bold]Dry run.[/] Re-run with [cyan]--apply[/] to execute.")
