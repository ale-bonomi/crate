"""Build a plan (pure with respect to the library: only reads) from scanned items."""

from __future__ import annotations

import sqlite3
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .analyze import metadata
from .analyze.types import Verdict
from .config import Category, Config
from .dedupe import library_copy, near_copies
from .hashing import file_hash
from .scan import AUDIO_EXT, Item
from .taxonomy import Route, route_audio, route_midi

_SPLIT_NAMES = {
    "punch": {"punchy": "Punchy", "soft": "Soft"},
    "snare_body": {"body": "Body", "tight": "Tight"},
    "feel": {"straight": "Straight", "swung": "Swung"},
}


@dataclass
class PlannedOp:
    src: Path
    dst: Path
    mode: str  # move | copy | extract | trash | quarantine | symlink
    label: str
    confidence: float
    original: str
    pack: str | None
    renamed: bool = False  # collision suffix added
    sidecar: bool = False
    content_hash: str | None = None
    raw_hash: str | None = None
    fingerprint: str | None = None
    sound_type: str | None = None
    is_loop: bool | None = None
    bpm: int | None = None
    musical_key: str | None = None
    punch: str | None = None
    snare_body: str | None = None
    feel: str | None = None
    wet: str | None = None
    label_source: str | None = None


@dataclass
class Plan:
    ops: list[PlannedOp] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    prune_roots: list[Path] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def per_folder(self) -> Counter[str]:
        return Counter(
            str(o.dst.parent)
            for o in self.ops
            if not o.sidecar and o.mode not in {"trash", "symlink"}
        )


Router = Callable[[Item], tuple[Path, str, float] | None]


def sort_router(cfg: Config, verdicts: dict[int, Verdict] | None = None) -> Router:
    root = cfg.root
    verdicts = verdicts or {}

    def route(item: Item) -> tuple[Path, str, float] | None:
        if item.kind == "audio":
            verdict = verdicts.get(id(item))
            if verdict is not None:
                meta = metadata.Meta(
                    type=verdict.type,
                    loop=verdict.loop,
                    bpm=verdict.bpm,
                    key=verdict.key,
                    confidence=verdict.confidence,
                    reason=verdict.reason,
                )
            else:
                meta = metadata.parse(item.path.stem, item.folders)
            r: Route = route_audio(meta, cfg)
        elif item.kind == "midi":
            r = route_midi(cfg)
        else:
            return None  # presets & others stay where they are
        return root / r.folder, r.label, r.confidence

    return route


def flatten_router(item: Item) -> tuple[Path, str, float] | None:
    return item.root, "flatten", 1.0


def _unique(dst: Path, taken: set[Path]) -> Path:
    n = 2
    while True:
        cand = dst.with_name(f"{dst.stem} ({n}){dst.suffix}")
        if cand not in taken and not cand.exists():
            return cand
        n += 1


def _fill(op: PlannedOp, verdict: Verdict | None) -> None:
    if verdict is None:
        return
    op.content_hash = verdict.content_hash
    op.raw_hash = verdict.raw_hash
    op.fingerprint = verdict.fingerprint
    op.sound_type = verdict.type
    op.is_loop = verdict.loop
    op.bpm = verdict.bpm
    op.musical_key = verdict.key
    op.punch = verdict.punch
    op.snare_body = verdict.snare_body
    op.feel = verdict.feel
    op.wet = verdict.wet
    op.label_source = verdict.source


def build_plan(
    items: list[Item],
    cfg: Config,
    router: Router,
    copy_roots: list[Path],
    verdicts: dict[int, Verdict] | None = None,
    conn: sqlite3.Connection | None = None,
) -> Plan:
    """`copy_roots`: files under these folders are copied (originals left in place)."""
    plan = Plan()
    taken: set[Path] = set()
    copy_roots = [r.expanduser().resolve() for r in copy_roots]
    plan.prune_roots = sorted({it.root for it in items if not it.staged})
    verdicts = verdicts or {}
    seen_hash: dict[str, Path] = {}
    seen_print: dict[str, Path] = {}

    for item in items:
        verdict = verdicts.get(id(item))
        routed = router(item)
        if routed is None:
            plan.skipped.append((item.original, f"{item.kind}: not routed yet"))
            continue
        folder, label, conf = routed
        src = item.path
        dst = folder / src.name
        identity = None
        if verdict is not None:
            identity = verdict.content_hash or verdict.raw_hash

        if dst.exists() and src.resolve() == dst.resolve():
            plan.skipped.append((item.original, "already in place"))
            continue

        hit: Path | None = None
        if identity and identity in seen_hash:
            hit = seen_hash[identity]
        elif conn is not None and verdict is not None:
            hit = library_copy(conn, verdict.content_hash, verdict.raw_hash, src)

        if hit is not None and hit.resolve() == src.resolve():
            plan.skipped.append((item.original, "already in place"))
            continue
        if hit is not None and item.staged:
            plan.skipped.append((item.original, f"already in library at {hit}"))
            continue
        if hit is not None and cfg.duplicates != "keep":
            _add_duplicate(plan, item, verdict, hit, cfg, taken)
            continue

        if verdict is not None and verdict.fingerprint and conn is not None:
            for other in near_copies(conn, verdict.fingerprint, verdict.content_hash):
                plan.notes.append(
                    f"{item.original} sounds like {other} "
                    "(different bit depth or sample rate); left in place"
                )
        if verdict is not None and verdict.fingerprint and verdict.fingerprint in seen_print:
            previous = seen_print[verdict.fingerprint]
            if identity not in seen_hash:
                plan.notes.append(
                    f"{item.original} sounds like {previous} "
                    "(different bit depth or sample rate); left in place"
                )

        renamed = False
        if dst in taken or dst.exists():
            if dst not in taken and dst.exists() and _same_content(src, dst):
                plan.skipped.append((item.original, f"identical file already at {dst}"))
                continue
            dst = _unique(dst, taken)
            renamed = True

        if item.staged:
            mode = "extract"
        elif cfg.keep_originals and _under(src, copy_roots):
            mode = "copy"
        else:
            mode = "move"

        taken.add(dst)
        op = PlannedOp(src, dst, mode, label, conf, item.original, item.pack, renamed)
        _fill(op, verdict)
        plan.ops.append(op)
        if identity and hit is None:
            seen_hash[identity] = dst
        if verdict is not None and verdict.fingerprint and verdict.content_hash:
            seen_print.setdefault(verdict.fingerprint, dst)

        sidecar = src.with_name(src.name + ".asd")
        if sidecar.exists():
            sdst = dst.with_name(dst.name + ".asd")
            if not sdst.exists():
                taken.add(sdst)
                plan.ops.append(
                    PlannedOp(
                        sidecar,
                        sdst,
                        mode,
                        label,
                        conf,
                        item.original + ".asd",
                        item.pack,
                        renamed,
                        sidecar=True,
                    )
                )

    _apply_splits(plan, cfg)
    return plan


def _under(src: Path, roots: list[Path]) -> bool:
    try:
        resolved = src.resolve()
    except OSError:
        return False
    return any(resolved.is_relative_to(root) for root in roots)


def _add_duplicate(
    plan: Plan,
    item: Item,
    verdict: Verdict | None,
    hit: Path,
    cfg: Config,
    taken: set[Path],
) -> None:
    src = item.path
    if cfg.duplicates == "quarantine":
        dst = cfg.root / "_Duplicates" / src.name
        if dst in taken or dst.exists():
            dst = _unique(dst, taken)
        taken.add(dst)
        op = PlannedOp(
            src, dst, "quarantine", "duplicate", 1.0, item.original, item.pack, dst.name != src.name
        )
    elif cfg.duplicates == "symlink":
        op = PlannedOp(src, hit, "symlink", "duplicate", 1.0, item.original, item.pack)
    else:
        op = PlannedOp(src, hit, "trash", "duplicate", 1.0, item.original, item.pack)
    _fill(op, verdict)
    plan.ops.append(op)


def _categories(cfg: Config) -> dict[str, Category]:
    found: dict[str, Category] = {}
    for cat in (*cfg.taxonomy.one_shots.values(), *cfg.taxonomy.loops.values()):
        found[cat.folder] = cat
    return found


def _split_value(op: PlannedOp, axis: str) -> str | None:
    if axis == "punch":
        return op.punch
    if axis == "snare_body":
        return op.snare_body
    if axis == "feel":
        return op.feel
    return None


def _apply_splits(plan: Plan, cfg: Config) -> None:
    root = cfg.root
    cats = _categories(cfg)
    groups: dict[Path, list[PlannedOp]] = defaultdict(list)
    for op in plan.ops:
        if op.sidecar or op.mode in {"trash", "symlink"}:
            continue
        groups[op.dst.parent].append(op)

    for folder, ops in groups.items():
        try:
            rel = folder.relative_to(root)
        except ValueError:
            continue
        cat = cats.get(rel.as_posix())
        if cat is None or not cat.split_by:
            continue
        if len(rel.parts) + 1 > cfg.max_depth:
            continue
        existing = _population(folder)
        if existing + len(ops) <= cfg.split_folder_threshold:
            continue
        names = _SPLIT_NAMES.get(cat.split_by, {})
        for op in ops:
            value = _split_value(op, cat.split_by)
            sub = names.get(value or "")
            if not sub:
                continue
            new_parent = folder / sub
            new_dst = new_parent / op.dst.name
            sidecar = _sidecar_for(plan, op)
            op.dst = new_dst
            if sidecar is not None:
                sidecar.dst = new_parent / sidecar.dst.name
                sidecar.mode = op.mode
        _resolve_collisions(ops, plan)


def _population(folder: Path) -> int:
    if not folder.is_dir():
        return 0
    return sum(
        1
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in AUDIO_EXT and not path.name.endswith(".asd")
    )


def _sidecar_for(plan: Plan, op: PlannedOp) -> PlannedOp | None:
    src = op.src.with_name(op.src.name + ".asd")
    for other in plan.ops:
        if other.sidecar and other.src == src:
            return other
    return None


def _resolve_collisions(ops: list[PlannedOp], plan: Plan) -> None:
    taken: set[Path] = set()
    for op in plan.ops:
        if op not in ops and not op.sidecar:
            taken.add(op.dst)
    for op in ops:
        if op.dst in taken or op.dst.exists():
            op.dst = _unique(op.dst, taken)
            op.renamed = True
            sidecar = _sidecar_for(plan, op)
            if sidecar is not None:
                sidecar.dst = op.dst.with_name(op.dst.name + ".asd")
                sidecar.renamed = True
        taken.add(op.dst)


def _same_content(a: Path, b: Path) -> bool:
    try:
        if a.stat().st_size != b.stat().st_size:
            return False
        return file_hash(a) == file_hash(b)
    except OSError:
        return False
