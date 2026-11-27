"""Command-line interface."""

from __future__ import annotations

import shutil
import sqlite3
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from . import db, learn, mover, planner
from .analyze.engine import annotate, save_correction
from .analyze.fusion import apply_correction, parse_label
from .analyze.types import Correction, Verdict
from .config import CRATE_HOME, Config, default_config_path, load_config, save_config
from .discover import detect
from .evaluate import read_labels, score
from .menubar import run as run_menu
from .projects import index_projects, parse_als, usage
from .scan import ScanResult, scan
from .search import (
    features_for_hash,
    filed_in_latest_batch,
    find,
    format_status,
    pending_review,
    review_count,
    similar,
    where,
)
from .taxonomy import ABLETON_LIBRARY
from .watch import list_media, load_seen, mark_seen, unseen

app = typer.Typer(help="Ableton-first audio sample organizer.", no_args_is_help=True)
console = Console()

Apply = Annotated[bool, typer.Option("--apply", help="Execute the plan (default is dry run).")]
Paths = Annotated[list[Path], typer.Argument(help="Files or folders (drag them in).")]
Slice = Annotated[
    bool | None,
    typer.Option(
        "--slice/--no-slice",
        help="Also cut the drum hits out of loops and chops into one-shots. "
        "Default: `slice.enabled` in the config.",
    ),
]


@dataclass
class State:
    config_path: Path | None = None
    verbose: bool = False
    jobs: int | None = None


@app.callback()
def main(
    ctx: typer.Context,
    config: Annotated[Path | None, typer.Option("--config", help="Config file.")] = None,
    verbose: Annotated[bool, typer.Option("--verbose")] = False,
    jobs: Annotated[int | None, typer.Option("--jobs", help="Parallel workers.")] = None,
) -> None:
    ctx.obj = State(config, verbose, jobs)


def _cfg(ctx: typer.Context) -> Config:
    return load_config(ctx.obj.config_path)


def _db_path() -> Path:
    return CRATE_HOME / "library.db"


def clean_path(p: Path) -> Path:
    """Undo shell-escaping/quoting left over from drag-and-drop."""
    s = str(p).strip().strip("'\"").replace("\\ ", " ")
    return Path(s).expanduser()


def _paths(paths: list[Path]) -> list[Path]:
    return [clean_path(p) for p in paths]


@app.command()
def init(ctx: typer.Context) -> None:
    """Create config + DB; ask for the library root and preset folders."""
    path = ctx.obj.config_path or default_config_path()
    cfg = load_config(path) if path.exists() else Config()
    found = detect()
    default = str(cfg.library_roots[0])
    root = typer.prompt("Library root (where sorted samples will live)", default=default)
    splice = typer.prompt("Splice folder", default=str(cfg.splice_folder))
    projects = typer.prompt(
        "Ableton project folder (blank to skip)",
        default=str(cfg.project_folders[0]) if cfg.project_folders else "",
    )
    library_default = str(
        cfg.ableton_user_library or found.user_library or Path("~/Music/Ableton/User Library")
    )
    user_library = typer.prompt("Ableton User Library", default=library_default)
    cfg.library_roots = [Path(root)]
    cfg.splice_folder = Path(splice)
    cfg.project_folders = [Path(projects)] if projects.strip() else []
    cfg.ableton_user_library = Path(user_library) if user_library.strip() else None
    proposed = found.presets
    if cfg.ableton_user_library is not None:
        library = str(cfg.ableton_user_library.expanduser())
        proposed = {
            ext: dest.replace(ABLETON_LIBRARY, library) for ext, dest in proposed.items()
        }
    console.print("Preset destinations:")
    for ext, dest in proposed.items():
        console.print(f"  {ext} -> {dest}")
    if typer.confirm("Use these preset folders?", default=True):
        cfg.non_audio.presets = proposed
    else:
        chosen: dict[str, str] = {}
        for ext, dest in proposed.items():
            chosen[ext] = typer.prompt(f"Folder for {ext}", default=dest)
        cfg.non_audio.presets = chosen
    save_config(cfg, path)
    db.connect(_db_path()).close()
    console.print(f"Config written to {path}; database at {_db_path()}")


def _scan_plan(
    ctx: typer.Context,
    paths: list[Path],
    router_kind: str,
    refresh: bool = False,
    copy_roots: list[Path] | None = None,
    skip_dirs: set[str] | None = None,
    slice_hits: bool = False,
) -> tuple[Config, ScanResult, planner.Plan, Path, sqlite3.Connection]:
    cfg = _cfg(ctx)
    CRATE_HOME.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="staging-", dir=CRATE_HOME))
    with console.status("Scanning..."):
        result = scan(
            _paths(paths),
            staging,
            include_other=router_kind == "sort" and cfg.non_audio.other != "leave",
        )
    if skip_dirs:
        result.items = [
            item for item in result.items if skip_dirs.isdisjoint(item.path.parts)
        ]
    conn = db.connect(_db_path())
    verdicts: dict[int, Verdict] = {}
    if router_kind == "sort":
        model = learn.load_model(cfg.use_trained_model)
        with console.status("Analyzing..."):
            verdicts, warnings = annotate(
                result.items,
                conn,
                cfg.clap_prompts if cfg.clap_enabled else {},
                jobs=ctx.obj.jobs,
                refresh=refresh,
                use_clap=cfg.clap_enabled,
                model=model,
            )
        result.warnings.extend(warnings)
        if slice_hits:
            result.warnings.extend(_slice(result, verdicts, conn, staging, cfg, model))
        router = planner.sort_router(cfg, verdicts)
    else:
        router = planner.flatten_router
    plan = planner.build_plan(
        result.items,
        cfg,
        router,
        [cfg.splice_folder] if copy_roots is None else copy_roots,
        verdicts,
        conn if router_kind == "sort" else None,
    )
    return cfg, result, plan, staging, conn


def _run(
    ctx: typer.Context,
    paths: list[Path],
    apply: bool,
    command: str,
    kind: str,
    copy_roots: list[Path] | None = None,
    skip_dirs: set[str] | None = None,
    slice_hits: bool | None = None,
) -> None:
    from .render import show_plan

    if slice_hits is None:
        slice_hits = _cfg(ctx).slice.enabled
    cfg, result, plan, staging, conn = _scan_plan(
        ctx, paths, kind, copy_roots=copy_roots, skip_dirs=skip_dirs, slice_hits=slice_hits
    )
    try:
        show_plan(console, plan, result, apply, root=cfg.root)
        if not apply or not plan.ops:
            return
        batch = mover.execute(conn, plan, command, cfg.root)
        console.print(f"[green]Done.[/] Batch {batch}. Undo with: crate undo {batch}")
    finally:
        conn.close()
        shutil.rmtree(staging, ignore_errors=True)


def _slice(
    result: ScanResult,
    verdicts: dict[int, Verdict],
    conn: sqlite3.Connection,
    staging: Path,
    cfg: Config,
    model: learn.Model | None,
) -> list[str]:
    """Cut drum hits out of the scanned loops and chops into staged one-shots."""
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, TimeRemainingColumn

    from . import slicer

    long_enough = [
        it
        for it in result.items
        if (v := verdicts.get(id(it))) is not None
        and v.features is not None
        and v.features.duration >= cfg.slice.min_source_seconds
    ]
    if not long_enough:
        return []
    if not slicer.installed():
        return ["Slicing needs Demucs. Install it with: uv sync --extra ml --extra slice"]
    columns = (BarColumn(), MofNCompleteColumn(), TimeRemainingColumn())
    with Progress("Slicing", *columns, console=console) as progress:
        task = progress.add_task("slice", total=len(long_enough))
        return slicer.add_slices(
            result.items,
            verdicts,
            conn,
            staging,
            cfg.slice,
            model=model,
            progress=lambda: progress.advance(task),
        )


@app.command()
def unpack(ctx: typer.Context, paths: Paths, apply: Apply = False) -> None:
    """Flatten nested pack folders and extract zips (no analysis)."""
    _run(ctx, paths, apply, "unpack", "flatten")


@app.command()
def sort(ctx: typer.Context, paths: Paths, apply: Apply = False, slice_: Slice = None) -> None:
    """Plan/apply moves into the library structure."""
    _run(ctx, paths, apply, "sort", "sort", slice_hits=slice_)


@app.command()
def ingest(ctx: typer.Context, paths: Paths, apply: Apply = False, slice_: Slice = None) -> None:
    """Unpack, classify and sort in one shot (the main daily command)."""
    _run(ctx, paths, apply, "ingest", "sort", slice_hits=slice_)


@app.command()
def reorganize(
    ctx: typer.Context,
    section: Annotated[
        str | None,
        typer.Option("--section", help="Library subfolder to re-sort, such as 'One Shots/Kicks'."),
    ] = None,
    apply: Apply = False,
) -> None:
    """Re-sort the library after folder rules change. Cached analysis is reused."""
    cfg = _cfg(ctx)
    if section:
        given = clean_path(Path(section))
        target = given if given.is_absolute() else cfg.root / given
        skip: set[str] = set()
    else:
        target = cfg.root
        skip = {"_Duplicates"}
    if not target.exists():
        console.print(f"{target} does not exist.")
        raise typer.Exit(code=1)
    _run(ctx, [target], apply, "reorganize", "sort", copy_roots=[], skip_dirs=skip)


@app.command()
def analyze(
    ctx: typer.Context,
    paths: Paths,
    refresh: Annotated[bool, typer.Option("--refresh", help="Recompute cached features.")] = False,
) -> None:
    """Analyze and cache results. Does not move files."""
    cfg = _cfg(ctx)
    staging = Path(tempfile.mkdtemp(prefix="staging-", dir=CRATE_HOME))
    CRATE_HOME.mkdir(parents=True, exist_ok=True)
    conn = db.connect(_db_path())
    try:
        result = scan(_paths(paths), staging)
        verdicts, warnings = annotate(
            result.items,
            conn,
            cfg.clap_prompts if cfg.clap_enabled else {},
            jobs=ctx.obj.jobs,
            refresh=refresh,
            use_clap=cfg.clap_enabled,
            model=learn.load_model(cfg.use_trained_model),
        )
        for warning in warnings:
            console.print(f"[yellow]warning:[/] {warning}")
        table = Table(title="Analysis")
        for col in ("File", "Type", "Loop", "Key", "BPM", "Character", "Conf"):
            table.add_column(col)
        for item in result.items:
            verdict = verdicts.get(id(item))
            if verdict is None:
                continue
            character = verdict.punch or verdict.snare_body or verdict.feel or ""
            if verdict.loop is True:
                shape = "loop"
            elif verdict.loop is False:
                shape = "one-shot"
            else:
                shape = "?"
            table.add_row(
                item.path.name,
                verdict.type or "?",
                shape,
                verdict.key or "",
                "" if verdict.bpm is None else str(verdict.bpm),
                character,
                f"{verdict.confidence:.2f}",
            )
        console.print(table)
    finally:
        conn.close()
        shutil.rmtree(staging, ignore_errors=True)


@app.command()
def train(
    ctx: typer.Context,
    paths: Annotated[
        list[Path] | None,
        typer.Argument(help="Folders to learn from. Default: the library and the Splice folder."),
    ] = None,
    min_examples: Annotated[
        int, typer.Option("--min-examples", help="Examples a sound type needs to be trusted.")
    ] = 20,
    gate: Annotated[
        float, typer.Option("--gate", help="Below this probability a guess goes to _Review/.")
    ] = 0.7,
) -> None:
    """Learn your sounds from well-named files and your corrections. Moves nothing."""
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, TimeRemainingColumn

    from .analyze.clap import ClapTagger

    cfg = _cfg(ctx)
    if paths:
        roots = _paths(paths)
    else:
        roots = [p for p in (cfg.root, cfg.splice_folder.expanduser()) if p.exists()]
    missing = [p for p in roots if not p.exists()]
    if missing or not roots:
        console.print(f"Nothing to learn from: {', '.join(map(str, missing)) or 'no folders'}.")
        raise typer.Exit(code=1)
    tagger = ClapTagger({})
    if not tagger.can_embed():
        console.print("[red]Training needs the CLAP model.[/] Install it with: uv sync --extra ml")
        raise typer.Exit(code=1)
    CRATE_HOME.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="staging-", dir=CRATE_HOME))
    conn = db.connect(_db_path())
    try:
        with console.status("Scanning..."):
            items = [item for item in scan(roots, staging).items if item.kind == "audio"]
        columns = (BarColumn(), MofNCompleteColumn(), TimeRemainingColumn())
        with Progress("Listening", *columns, console=console) as progress:
            task = progress.add_task("listen", total=len(items))
            collected = learn.collect(
                items,
                conn,
                lambda audio: tagger.embed(audio.samples, audio.sample_rate),
                progress=lambda: progress.advance(task),
            )
    finally:
        conn.close()
        shutil.rmtree(staging, ignore_errors=True)
    if tagger.error:
        console.print(f"[yellow]warning:[/] {tagger.error}")
    for why, n in collected.skipped.most_common():
        console.print(f"[dim]skipped {n}: {why}[/]")
    try:
        with console.status("Training..."):
            model, reports = learn.fit(collected.examples, gate=gate, min_examples=min_examples)
    except learn.TrainError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from exc

    table = Table(title=f"Learned from {len(collected.examples)} files")
    table.add_column("Sound")
    table.add_column("Examples", justify="right")
    table.add_column("Packs", justify="right")
    table.add_column("Right on unseen packs", justify="right")
    table.add_column("Found", justify="right")
    table.add_column("Crate files it")
    for r in reports:
        precision = "-" if r.precision is None else f"{r.precision:.0%}"
        recall = "-" if r.recall is None else f"{r.recall:.0%}"
        table.add_row(
            r.label, str(r.examples), str(r.packs), precision, recall,
            "[green]yes[/]" if r.trusted else "[yellow]no, review[/]",
        )  # fmt: skip
    console.print(table)
    path = model.save()
    trusted = ", ".join(sorted(model.trusted)) or "none yet"
    console.print(f"Saved {path}. Files whose names say nothing will be filed as: {trusted}.")
    console.print("Other guesses go to _Review/ with the guess shown. Re-run after adding packs.")


@app.command(name="eval")
def evaluate_cmd(
    ctx: typer.Context, labels: Annotated[Path, typer.Argument(help="CSV: path,type,loop")]
) -> None:
    """Precision and recall against a hand-labeled CSV. The CSV stays private."""
    cfg = _cfg(ctx)
    rows = read_labels(clean_path(labels))
    staging = Path(tempfile.mkdtemp(prefix="staging-", dir=CRATE_HOME))
    conn = db.connect(_db_path())
    try:
        paths = [Path(row[0]) for row in rows]
        result = scan(paths, staging)
        by_name = {item.path.resolve(): item for item in result.items}
        verdicts, _warnings = annotate(
            result.items,
            conn,
            cfg.clap_prompts if cfg.clap_enabled else {},
            jobs=ctx.obj.jobs,
            use_clap=cfg.clap_enabled,
            model=learn.load_model(cfg.use_trained_model),
        )
        gold: list[tuple[str, str]] = []
        predicted: list[str | None] = []
        for path_s, typ, _loop in rows:
            item = by_name.get(Path(path_s).expanduser().resolve())
            verdict = verdicts.get(id(item)) if item else None
            gold.append((path_s, typ))
            predicted.append(verdict.type if verdict else None)
        report = score(gold, predicted)
    finally:
        conn.close()
        shutil.rmtree(staging, ignore_errors=True)
    table = Table(title=f"Eval ({report.labeled} files)")
    table.add_column("Type")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("Support", justify="right")
    for label, (precision, recall, support) in sorted(report.per_type.items()):
        table.add_row(label, f"{precision:.2f}", f"{recall:.2f}", str(support))
    console.print(table)


@app.command()
def review(ctx: typer.Context, apply: Apply = False) -> None:
    """Walk low-confidence files. With --apply, play each one and store a label."""
    cfg = _cfg(ctx)
    conn = db.connect(_db_path())
    try:
        rows = pending_review(conn, cfg.review_threshold)
        if not rows:
            console.print("Nothing to review.")
            return
        if not apply or not console.is_terminal:
            table = Table(title="Needs review")
            table.add_column("Path", overflow="fold")
            table.add_column("Guess")
            table.add_column("Conf", justify="right")
            for row in rows:
                table.add_row(
                    row["current_path"], row["sound_type"] or "?", f"{row['confidence']:.2f}"
                )
            console.print(table)
            if apply and not console.is_terminal:
                console.print("Not a terminal, so nothing was assigned. Re-run in a terminal.")
            else:
                console.print("Re-run with [cyan]--apply[/] to listen and assign labels.")
            return
        import subprocess

        for row in rows:
            path = Path(row["current_path"])
            if not path.exists():
                continue
            guess = row["sound_type"] or "?"
            console.print(f"\n[bold]{path.name}[/] guessed {guess} ({row['confidence']:.2f})")
            subprocess.run(["afplay", str(path)], check=False)
            label = typer.prompt("Label (blank skips, q quits)", default="")
            if label.strip().lower() == "q":
                break
            if not label.strip():
                continue
            try:
                correction = parse_label(label)
            except ValueError as exc:
                console.print(f"[yellow]{exc}[/]")
                continue
            _commit_label(ctx, conn, cfg, path, correction)
    finally:
        conn.close()


def _commit_label(
    ctx: typer.Context,
    conn: sqlite3.Connection,
    cfg: Config,
    path: Path,
    correction: Correction,
) -> None:
    staging = Path(tempfile.mkdtemp(prefix="staging-", dir=CRATE_HOME))
    try:
        result = scan([path], staging)
        verdicts, _warnings = annotate(result.items, conn, {}, jobs=1, use_clap=False)
        item = next(iter(result.items), None)
        if item is None:
            console.print(f"[yellow]cannot read {path}[/]")
            return
        verdict = verdicts[id(item)]
        identity = verdict.content_hash or verdict.raw_hash
        if not identity:
            console.print(f"[yellow]cannot identify {path}[/]")
            return
        save_correction(conn, identity, correction)
        verdicts, _warnings = annotate(result.items, conn, {}, jobs=1, use_clap=False)
        plan = planner.build_plan(
            result.items, cfg, planner.sort_router(cfg, verdicts), [], verdicts, conn
        )
        from .render import show_plan

        show_plan(console, plan, result, True, root=cfg.root)
        if plan.ops:
            batch = mover.execute(conn, plan, "correct", cfg.root)
            console.print(f"[green]Labeled.[/] Batch {batch}.")
    finally:
        shutil.rmtree(staging, ignore_errors=True)


@app.command()
def correct(
    ctx: typer.Context,
    path: Annotated[Path, typer.Argument(help="File to relabel.")],
    label: Annotated[str, typer.Argument(help="kick, snare:tight, loop:drums, ...")],
    apply: Apply = False,
) -> None:
    """Store a manual label. It wins over audio and filenames forever."""
    cfg = _cfg(ctx)
    path = clean_path(path)
    try:
        correction = parse_label(label)
    except ValueError as exc:
        console.print(f"[yellow]{exc}[/]")
        raise typer.Exit(code=1) from exc
    conn = db.connect(_db_path())
    staging = Path(tempfile.mkdtemp(prefix="staging-", dir=CRATE_HOME))
    try:
        result = scan([path], staging)
        verdicts, _warnings = annotate(result.items, conn, {}, jobs=ctx.obj.jobs, use_clap=False)
        item = next((it for it in result.items if it.kind == "audio"), None)
        if item is None:
            console.print(f"No audio at {path}")
            raise typer.Exit(code=1)
        current = verdicts[id(item)]
        preview = apply_correction(current, correction)
        preview.content_hash = current.content_hash
        preview.fingerprint = current.fingerprint
        preview.raw_hash = current.raw_hash
        preview.features = current.features
        verdicts[id(item)] = preview
        plan = planner.build_plan(
            result.items, cfg, planner.sort_router(cfg, verdicts), [], verdicts, conn
        )
        from .render import show_plan

        show_plan(console, plan, result, apply, root=cfg.root)
        if not apply:
            return
        identity = current.content_hash or current.raw_hash
        if identity:
            save_correction(conn, identity, correction)
        if plan.ops:
            batch = mover.execute(conn, plan, "correct", cfg.root)
            console.print(f"[green]Labeled.[/] Batch {batch}.")
    finally:
        conn.close()
        shutil.rmtree(staging, ignore_errors=True)


@app.command()
def dedupe(ctx: typer.Context, apply: Apply = False) -> None:
    """Find copies of library audio. Near-duplicates are reported and kept."""
    cfg = _cfg(ctx)
    conn = db.connect(_db_path())
    try:
        plan = planner.duplicate_plan(conn, cfg, [cfg.root, cfg.splice_folder])
        for op in plan.ops:
            console.print(f"duplicate: {op.src}  →  keep {op.kept} ({op.mode})")
        for note in plan.notes:
            console.print(f"[yellow]{note}[/]")
        if not plan.ops:
            console.print("Nothing to resolve.")
        elif not apply:
            console.print(
                "[bold]Dry run.[/] Re-run with [cyan]--apply[/] to resolve exact duplicates."
            )
        else:
            batch = mover.execute(conn, plan, "dedupe", cfg.root)
            console.print(f"[green]Done.[/] Batch {batch}. Undo with: crate undo {batch}")
    finally:
        conn.close()


@app.command(name="where")
def where_cmd(
    ctx: typer.Context,
    query: Annotated[str, typer.Argument(help="Original name, current name, or part of a path.")],
) -> None:
    """Where a file went, from its original name or its name now."""
    conn = db.connect(_db_path())
    try:
        rows = where(conn, query)
    finally:
        conn.close()
    if not rows:
        console.print(f"Nothing matched {query!r}.")
        raise typer.Exit(code=1)
    table = Table(title=f"Matches for {query}")
    for col in ("now", "was", "type", "pack"):
        table.add_column(col, overflow="fold")
    for row in rows:
        table.add_row(
            row["current_path"], row["original_path"], row["sound_type"] or "", row["pack"] or ""
        )
    console.print(table)


@app.command(name="find")
def find_cmd(
    ctx: typer.Context,
    sound_type: Annotated[str | None, typer.Option("--type")] = None,
    musical_key: Annotated[str | None, typer.Option("--key")] = None,
    min_bpm: Annotated[int | None, typer.Option("--min-bpm")] = None,
    max_bpm: Annotated[int | None, typer.Option("--max-bpm")] = None,
    loop: Annotated[
        bool | None, typer.Option("--loop/--oneshot", help="Loops or one-shots.")
    ] = None,
    punch: Annotated[str | None, typer.Option("--punch", help="punchy or soft")] = None,
    feel: Annotated[str | None, typer.Option("--feel", help="straight or swung")] = None,
    pack: Annotated[str | None, typer.Option("--pack")] = None,
    similar_to: Annotated[
        Path | None, typer.Option("--similar", help="Rank library files by this one.")
    ] = None,
    limit: Annotated[int, typer.Option("--limit")] = 40,
) -> None:
    """Search the indexed library. Prints paths you can drag into Live."""
    cfg = _cfg(ctx)
    conn = db.connect(_db_path())
    try:
        if similar_to is not None:
            _print_similar(ctx, conn, cfg, clean_path(similar_to), limit)
            return
        rows = find(
            conn,
            sound_type=sound_type,
            musical_key=musical_key,
            min_bpm=min_bpm,
            max_bpm=max_bpm,
            loop=loop,
            punch=punch,
            feel=feel,
            pack=pack,
            limit=limit,
        )
    finally:
        conn.close()
    if similar_to is not None:
        return
    if not rows:
        console.print("No matches.")
        return
    for row in rows:
        bits = [row["sound_type"] or "?", "loop" if row["is_loop"] else "one-shot"]
        if row["musical_key"]:
            bits.append(row["musical_key"])
        if row["bpm"]:
            bits.append(f"{row['bpm']}bpm")
        if row["punch"]:
            bits.append(row["punch"])
        if row["feel"]:
            bits.append(row["feel"])
        console.print(f"{row['current_path']}  [dim]{' · '.join(bits)}[/]")


def _print_similar(
    ctx: typer.Context, conn: sqlite3.Connection, cfg: Config, path: Path, limit: int
) -> None:
    staging = Path(tempfile.mkdtemp(prefix="staging-", dir=CRATE_HOME))
    try:
        result = scan([path], staging)
        verdicts, _warnings = annotate(result.items, conn, cfg.clap_prompts, jobs=1, use_clap=False)
        item = next((it for it in result.items if it.kind == "audio"), None)
        verdict = verdicts.get(id(item)) if item else None
        payload: dict[str, object] | None = None
        if verdict and verdict.content_hash:
            payload = features_for_hash(conn, verdict.content_hash)
        if payload is None and verdict and verdict.features:
            payload = dict(verdict.features.to_json())
        if not payload:
            console.print(f"Could not hear {path}")
            return
        ranked = similar(conn, payload, limit)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    if not ranked:
        console.print("Nothing comparable in the library yet.")
        return
    for dist, row in ranked:
        if Path(row["current_path"]).resolve() == path.resolve():
            continue
        console.print(f"{row['current_path']}  [dim]{dist:.2f} {row['sound_type'] or ''}[/]")


@app.command()
def projects(
    ctx: typer.Context,
    paths: Annotated[
        list[Path] | None, typer.Argument(help="Live sets or a project folder.")
    ] = None,
    apply: Apply = False,
) -> None:
    """Index .als files and report which library samples they use."""
    cfg = _cfg(ctx)
    targets = _paths(paths or []) or [p.expanduser() for p in cfg.project_folders]
    if not targets:
        console.print("Pass a .als file, or set project_folders in the config.")
        raise typer.Exit(code=1)
    if not apply:
        count = 0
        for target in targets:
            files = sorted(target.rglob("*.als")) if target.is_dir() else [target]
            for als in files:
                if als.suffix.lower() != ".als":
                    continue
                refs = parse_als(als)
                count += 1
                console.print(f"{als}: {len(refs)} sample refs")
        console.print(
            f"[bold]Dry run.[/] {count} sets. Re-run with [cyan]--apply[/] to store them."
        )
        return
    conn = db.connect(_db_path())
    try:
        indexed = index_projects(conn, targets)
        report = usage(conn)
        for als, refs in indexed:
            console.print(f"{als.name}: {len(refs)} sample refs")
        console.print(
            f"Library: {report.used} used, {report.unused} unused, {len(report.missing)} missing"
        )
        for name in report.missing[:20]:
            console.print(f"  missing {name}")
    finally:
        conn.close()


@app.command()
def watch(
    ctx: typer.Context,
    apply: Apply = False,
    backfill: Annotated[
        bool, typer.Option("--backfill", help="Also ingest what is already in Splice.")
    ] = False,
    interval: Annotated[float, typer.Option("--interval", help="Seconds between checks.")] = 5,
    once: Annotated[bool, typer.Option("--once", help="Check once and exit.")] = False,
    menu: Annotated[
        bool | None,
        typer.Option("--menu/--no-menu", help="macOS menu bar when PyObjC is installed."),
    ] = None,
) -> None:
    """Watch the Splice folder. New packs are planned; --apply files them."""
    cfg = _cfg(ctx)
    conn = db.connect(_db_path())
    filed = {"n": 0}

    def poll() -> str:
        current = list_media(cfg.splice_folder)
        known = load_seen(conn)
        if not backfill and not known:
            mark_seen(conn, current)
            console.print(
                f"Watching {cfg.splice_folder}. {len(current)} existing files left as they are."
            )
            return format_status(filed["n"], review_count(conn, cfg.review_threshold))
        fresh = unseen(current, known)
        if fresh:
            _run(ctx, fresh, apply, "watch", "sort")
            if apply:
                filed["n"] += len(fresh)
                mark_seen(conn, fresh)
        return format_status(filed["n"], review_count(conn, cfg.review_threshold))

    try:
        if once:
            console.print(poll())
            return
        use_menu = menu if menu is not None else False
        if use_menu and run_menu(poll, interval):
            return
        if use_menu:
            console.print("Menu bar needs PyObjC. Install with: uv sync --extra mac")
        while True:
            console.print(poll())
            time.sleep(interval)
    except KeyboardInterrupt:
        console.print("Stopped.")
    finally:
        conn.close()


@app.command()
def undo(
    batch_id: Annotated[int | None, typer.Argument()] = None,
) -> None:
    """Reverse a batch (default: the latest)."""
    conn = db.connect(_db_path())
    batch = batch_id if batch_id is not None else mover.latest_done_batch(conn)
    if batch is None:
        console.print("Nothing to undo.")
        raise typer.Exit()
    warnings = mover.undo(conn, batch)
    for w in warnings:
        console.print(f"[yellow]warning:[/] {w}")
    status_row = conn.execute("SELECT status FROM batches WHERE id=?", (batch,)).fetchone()
    console.print(f"Batch {batch}: {status_row['status'] if status_row else 'unknown'}")
    conn.close()


@app.command()
def gui(
    port: Annotated[int, typer.Option("--port", help="Localhost port.")] = 8765,
) -> None:
    """Open the Callima window in the browser."""
    from .gui.server import serve

    serve(port)


@app.command()
def status(ctx: typer.Context) -> None:
    """Library stats, review queue, project usage, and recent batches."""
    cfg = _cfg(ctx)
    conn = db.connect(_db_path())
    n = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    line = format_status(filed_in_latest_batch(conn), review_count(conn, cfg.review_threshold))
    console.print(line)
    console.print(f"Tracked files: {n}")
    report = usage(conn)
    projects_n = conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0]
    if projects_n:
        console.print(
            f"Projects: {projects_n} sets, {report.used} samples used, "
            f"{report.unused} unused, {len(report.missing)} missing"
        )
    table = Table(title="Recent batches")
    for col in ("id", "when", "command", "status", "ops"):
        table.add_column(col)
    for b in conn.execute(
        "SELECT b.*, (SELECT COUNT(*) FROM ops o WHERE o.batch_id=b.id) AS n"
        " FROM batches b ORDER BY id DESC LIMIT 10"
    ):
        table.add_row(str(b["id"]), b["created_at"], b["command"], b["status"], str(b["n"]))
    console.print(table)
    conn.close()
