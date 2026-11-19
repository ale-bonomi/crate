"""Desk state and the actions behind the Callima window.

File moves stay a plan until Apply. The window reads and writes the same config
and database as the command line.
"""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import cast

from .. import db, mover, planner
from ..analyze.engine import annotate, save_correction
from ..analyze.fusion import apply_correction, parse_label
from ..analyze.types import Verdict
from ..config import CRATE_HOME, Config, load_config, save_config
from ..dedupe import identify, scan_tree
from ..evaluate import read_labels, score
from ..projects import index_projects, usage
from ..scan import Item, scan
from ..search import (
    features_for_hash,
    filed_in_latest_batch,
    format_status,
    pending_review,
    review_count,
    similar,
    where,
)
from ..watch import list_media, load_seen, unseen

_UNSAFE = re.compile(r"[\\/:\0]+")
_MODES = ("type", "genre", "flat", "original", "random", "custom")
_STYLES = ("keep", "lowercase", "tagged")
_BINS = ("Bin A", "Bin B", "Bin C", "Bin D")
_LIBRARY_FOLDERS = {
    "One Shots",
    "Loops",
    "MIDI",
    "_Review",
    "_Duplicates",
    "_Presets Unsorted",
}
_PRESET_LABELS = {
    ".fxp": "Serum",
    "serum": "Serum 1",
    ".vital": "Vital",
    ".omnisphere": "Omnisphere",
    ".pig": "Pigments",
    ".pgtx": "Pigments",
    ".nmsv": "Massive",
    ".mxpreset": "Massive X",
    ".phaseplant": "Phase Plant",
    ".spf": "Spire",
    ".sbf": "Spire",
    "sylenth1": "Sylenth1",
    "dune": "Dune",
    "diva": "Diva",
    "hive": "Hive",
    "zebra2": "Zebra",
    ".adv": "Ableton racks",
    ".adg": "Ableton devices",
    "unknown": "Unsorted presets",
    "kontakt": "Kontakt",
}


class DeskError(Exception):
    pass


def _unquote_path(path: Path) -> Path:
    text = str(path).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return Path(text[1:-1].strip())
    return path


def _strip_quotes(cfg: Config) -> bool:
    changed = False
    roots = [_unquote_path(path) for path in cfg.library_roots]
    if roots != list(cfg.library_roots):
        cfg.library_roots = roots
        changed = True
    splice = _unquote_path(cfg.splice_folder)
    if splice != cfg.splice_folder:
        cfg.splice_folder = splice
        changed = True
    projects = [_unquote_path(path) for path in cfg.project_folders]
    if projects != list(cfg.project_folders):
        cfg.project_folders = projects
        changed = True
    if cfg.ableton_user_library is not None:
        library = _unquote_path(cfg.ableton_user_library)
        if library != cfg.ableton_user_library:
            cfg.ableton_user_library = library
            changed = True
    presets: dict[str, str] = {}
    for key, dest in cfg.non_audio.presets.items():
        cleaned = str(_unquote_path(Path(dest))) if dest else dest
        presets[key] = cleaned
        if cleaned != dest:
            changed = True
    if changed:
        cfg.non_audio.presets = presets
    return changed


def _slug(text: str) -> str:
    cleaned = _UNSAFE.sub(" ", text).strip()
    return cleaned or "Untitled"


def _default_state() -> dict[str, object]:
    return {
        "selected": "root",
        "structure": "library",
        "structures": [
            {"id": "library", "name": "My library", "mode": "type"},
            {"id": "splice", "name": "Splice packs", "mode": "original"},
        ],
        "meta": {"genre": "", "mood": ""},
        "title": {"genre": "", "mood": ""},
        "cleanup": {"asd": True, "duplicates": True, "least": True},
        "extras": [],
        "watch": False,
        "pinned": [],
        "tag_style": "keep",
    }


def _load_state(path: Path) -> dict[str, object]:
    base = _default_state()
    if not path.exists():
        return base
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return base
    if not isinstance(raw, dict):
        return base
    base.update(raw)
    return base


class Desk:
    def __init__(self, home: Path | None = None, config_path: Path | None = None) -> None:
        self.home = home or CRATE_HOME
        self.config_path = config_path or (self.home / "config.yaml")
        self.state_path = self.home / "gui.json"
        self._lock = threading.Lock()
        self._state = _load_state(self.state_path)
        self.pack: Path | None = None
        self.search = ""
        self.notice = ""
        self.message = ""
        self._plan: planner.Plan | None = None
        self._plan_kind = ""
        self.phase = "idle"  # idle | running | ready | error
        self._summary: dict[str, object] = {}

    def config(self) -> Config:
        cfg = load_config(self.config_path if self.config_path.exists() else None)
        if self.config_path.exists():
            return cfg
        return cfg

    def connect(self) -> sqlite3.Connection:
        return db.connect(self.home / "library.db")

    def _save(self) -> None:
        self.home.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self._state, indent=2))

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            cfg = self._fresh_config()
            return self._public(cfg)

    def pick(self, prompt: str) -> dict[str, object]:
        safe = prompt.replace('"', "")
        try:
            out = subprocess.run(
                ["osascript", "-e", f'POSIX path of (choose folder with prompt "{safe}")'],
                capture_output=True,
                text=True,
                timeout=300,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise DeskError("The folder dialog did not open.") from exc
        path = out.stdout.strip()
        if out.returncode != 0 or not path:
            return {"path": None}
        return {"path": path}

    def action(self, name: str, payload: dict[str, object]) -> dict[str, object]:
        handler = {
            "select-folder": self._select_folder,
            "add-folder": self._add_folder,
            "set-pack": self._set_pack,
            "clear-pack": self._clear_pack,
            "set-tags": self._set_tags,
            "set-structure": self._set_structure,
            "set-mode": self._set_mode,
            "new-structure": self._new_structure,
            "set-cleanup": self._set_cleanup,
            "set-style": self._set_style,
            "search": self._search,
            "pin": self._pin,
            "save-settings": self._save_settings,
            "reset-settings": self._reset_settings,
            "go": self._go,
            "apply": self._apply,
            "cancel": self._cancel,
            "undo": self._undo,
            "correct": self._correct,
            "projects": self._projects,
            "analyze": self._analyze,
            "eval": self._eval,
            "watch": self._watch,
            "similar": self._similar,
        }.get(name)
        if handler is None:
            raise DeskError(f"Unknown action {name}.")
        handler(payload)
        return self.snapshot()

    def watch_tick(self) -> None:
        with self._lock:
            if not self._state.get("watch"):
                return
            cfg = self._fresh_config()
        conn = self.connect()
        try:
            fresh = unseen(list_media(cfg.splice_folder), load_seen(conn))
        finally:
            conn.close()
        with self._lock:
            self.notice = f"{len(fresh)} new in Splice" if fresh else ""

    def _fresh_config(self) -> Config:
        if not self.config_path.exists():
            return Config()
        cfg = load_config(self.config_path)
        changed = _strip_quotes(cfg)
        if changed:
            save_config(cfg, self.config_path)
        return cfg

    def _public(self, cfg: Config) -> dict[str, object]:
        rows = self._folders(cfg)
        selected = str(self._state.get("selected") or "root")
        if not any(row["id"] == selected for row in rows):
            selected = str(rows[0]["id"]) if rows else "root"
        structures = self._structures()
        active_id = str(self._state.get("structure") or "library")
        active = next((item for item in structures if item["id"] == active_id), structures[0])
        style = str(self._state.get("tag_style") or "keep")
        dest = self._destination(cfg, rows, selected, str(active["mode"]))
        return {
            "library": [{**row, "selected": row["id"] == selected} for row in rows],
            "destination": {"name": dest.name or "Library", "path": str(dest)},
            "structure": active,
            "structures": structures,
            "meta": self._pair("meta"),
            "title": self._pair("title"),
            "tagStyle": style,
            "cleanup": self._cleanup(),
            "files": self._files(cfg),
            "search": self.search,
            "pack": None
            if self.pack is None
            else {"name": self.pack.name, "path": str(self.pack)},
            "notice": self.notice,
            "message": self.message,
            "phase": self.phase,
            "summary": self._summary,
            "status": self._status(cfg),
            "review": self._review(cfg),
            "settings": self._settings(cfg),
            "tools": self._tools(cfg),
        }

    def _status(self, cfg: Config) -> str:
        conn = self.connect()
        try:
            return format_status(
                filed_in_latest_batch(conn), review_count(conn, cfg.review_threshold)
            )
        finally:
            conn.close()

    def _review(self, cfg: Config) -> list[dict[str, object]]:
        conn = self.connect()
        try:
            rows = pending_review(conn, cfg.review_threshold)
        finally:
            conn.close()
        return [
            {
                "path": row["current_path"],
                "name": Path(row["current_path"]).name,
                "guess": row["sound_type"] or "?",
                "confidence": row["confidence"],
            }
            for row in rows[:40]
        ]

    def _settings(self, cfg: Config) -> dict[str, object]:
        synths: list[dict[str, str]] = []
        for key, dest in cfg.non_audio.presets.items():
            synths.append(
                {"key": key, "label": _PRESET_LABELS.get(key, key), "path": dest, "group": "synth"}
            )
        if "kontakt" not in cfg.non_audio.presets:
            synths.append({"key": "kontakt", "label": "Kontakt", "path": "", "group": "synth"})
        library = [
            {
                "key": "library",
                "label": "Library",
                "path": str(cfg.library_roots[0]),
                "group": "library",
            },
            {
                "key": "splice",
                "label": "Splice",
                "path": str(cfg.splice_folder),
                "group": "library",
            },
            {
                "key": "projects",
                "label": "Ableton projects",
                "path": str(cfg.project_folders[0]) if cfg.project_folders else "",
                "group": "library",
            },
        ]
        daws = [
            {
                "key": "ableton",
                "label": "Ableton",
                "path": "" if cfg.ableton_user_library is None else str(cfg.ableton_user_library),
                "group": "daw",
            },
            {
                "key": "cubase",
                "label": "Cubase",
                "path": str(cfg.non_audio.presets.get("cubase", "")),
                "group": "daw",
            },
        ]
        return {"library": library, "synths": synths, "daws": daws}

    def _tools(self, cfg: Config) -> dict[str, object]:
        conn = self.connect()
        try:
            report = usage(conn)
            batches = list(
                conn.execute(
                    "SELECT id, command, status FROM batches ORDER BY id DESC LIMIT 6"
                )
            )
        finally:
            conn.close()
        return {
            "used": report.used,
            "unused": report.unused,
            "missing": report.missing[:8],
            "batches": [
                {"id": row["id"], "command": row["command"], "status": row["status"]}
                for row in batches
            ],
            "projects": [str(path) for path in cfg.project_folders],
        }

    def _pair(self, key: str) -> dict[str, str]:
        raw = self._state.get(key)
        if not isinstance(raw, dict):
            return {"genre": "", "mood": ""}
        return {"genre": str(raw.get("genre") or ""), "mood": str(raw.get("mood") or "")}

    def _cleanup(self) -> dict[str, bool]:
        raw = self._state.get("cleanup")
        if not isinstance(raw, dict):
            return {"asd": True, "duplicates": True, "least": True}
        return {
            "asd": bool(raw.get("asd", True)),
            "duplicates": bool(raw.get("duplicates", True)),
            "least": bool(raw.get("least", True)),
        }

    def _structures(self) -> list[dict[str, str]]:
        raw = self._state.get("structures")
        items: list[dict[str, str]] = []
        if isinstance(raw, list):
            for item in raw:
                if not isinstance(item, dict):
                    continue
                mode = str(item.get("mode") or "type")
                if mode not in _MODES:
                    mode = "type"
                items.append(
                    {
                        "id": str(item.get("id") or "library"),
                        "name": str(item.get("name") or "Structure"),
                        "mode": mode,
                    }
                )
        return items or cast(list[dict[str, str]], _default_state()["structures"])

    def _folders(self, cfg: Config) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = [
            {
                "id": "splice",
                "name": "Splice",
                "path": str(cfg.splice_folder.expanduser()),
                "kind": "source",
            }
        ]
        root = cfg.root
        children: list[Path] = []
        if root.exists():
            children = sorted(
                path
                for path in root.iterdir()
                if path.is_dir() and not path.name.startswith(".") and path.name in _LIBRARY_FOLDERS
            )
        if children:
            rows.extend(
                {
                    "id": f"child:{path.name}",
                    "name": path.name,
                    "path": str(path),
                    "kind": "library",
                }
                for path in children
            )
        rows.insert(
            0,
            {
                "id": "root",
                "name": root.name or "Library",
                "path": str(root),
                "kind": "library",
            },
        )
        extras = self._state.get("extras")
        if isinstance(extras, list):
            for item in extras:
                if not isinstance(item, str) or not item:
                    continue
                path = Path(item)
                rows.append(
                    {
                        "id": f"extra:{path}",
                        "name": path.name or str(path),
                        "path": str(path),
                        "kind": "library",
                    }
                )
        return rows

    def _destination(
        self, cfg: Config, rows: list[dict[str, str]], selected: str, mode: str
    ) -> Path:
        if mode == "type":
            return cfg.root
        row = next(
            (item for item in rows if item["id"] == selected and item["kind"] == "library"),
            None,
        )
        if row is None:
            return cfg.root
        return Path(row["path"])

    def _files(self, cfg: Config) -> list[dict[str, str]]:
        pinned = self._state.get("pinned")
        found: list[dict[str, str]] = []
        if isinstance(pinned, list):
            for item in pinned:
                if isinstance(item, str) and item:
                    found.append({"name": Path(item).name, "path": item, "note": "added"})
        query = self.search.strip()
        conn = self.connect()
        try:
            if query:
                for row in where(conn, query)[:40]:
                    found.append(
                        {
                            "name": Path(row["current_path"]).name,
                            "path": row["current_path"],
                            "note": row["sound_type"] or "",
                        }
                    )
            elif self._cleanup()["least"]:
                if conn.execute("SELECT COUNT(*) FROM project_samples").fetchone()[0] == 0:
                    return found
                names: set[str] = set()
                for row in conn.execute("SELECT name, path FROM project_samples"):
                    if row["name"]:
                        names.add(row["name"])
                    if row["path"]:
                        names.add(Path(row["path"]).name)
                for row in conn.execute(
                    "SELECT current_path, original_name FROM files ORDER BY current_path"
                ):
                    current = Path(row["current_path"]).name
                    if current in names or row["original_name"] in names:
                        continue
                    if current.endswith(".asd"):
                        continue
                    found.append(
                        {"name": current, "path": row["current_path"], "note": "least used"}
                    )
                    if len(found) >= 40:
                        break
            else:
                for row in conn.execute(
                    "SELECT current_path, sound_type FROM files ORDER BY current_path LIMIT 40"
                ):
                    found.append(
                        {
                            "name": Path(row["current_path"]).name,
                            "path": row["current_path"],
                            "note": row["sound_type"] or "",
                        }
                    )
        finally:
            conn.close()
        return found

    def _select_folder(self, payload: dict[str, object]) -> None:
        with self._lock:
            self._state["selected"] = str(payload.get("id") or "root")
            self._clear_plan()
            self._save()

    def _add_folder(self, payload: dict[str, object]) -> None:
        path = str(payload.get("path") or "").strip()
        if not path:
            raise DeskError("Choose a folder first.")
        with self._lock:
            extras = self._state.get("extras")
            items = extras if isinstance(extras, list) else []
            items = [item for item in items if isinstance(item, str)]
            if path not in items:
                items.append(path)
            self._state["extras"] = items
            self._state["selected"] = f"extra:{Path(path)}"
            self._save()

    def _set_pack(self, payload: dict[str, object]) -> None:
        raw = str(payload.get("path") or "").strip()
        path = Path(raw).expanduser()
        if not raw or not path.exists():
            raise DeskError("That folder is not on disk.")
        with self._lock:
            self.pack = path
            self._clear_plan()
            self.message = path.name

    def _clear_pack(self, payload: dict[str, object]) -> None:
        del payload
        with self._lock:
            self.pack = None
            self._clear_plan()

    def _set_tags(self, payload: dict[str, object]) -> None:
        which = "title" if payload.get("which") == "title" else "meta"
        with self._lock:
            self._state[which] = {
                "genre": str(payload.get("genre") or ""),
                "mood": str(payload.get("mood") or ""),
            }
            self._save()

    def _set_structure(self, payload: dict[str, object]) -> None:
        chosen = str(payload.get("id") or "")
        with self._lock:
            if any(item["id"] == chosen for item in self._structures()):
                self._state["structure"] = chosen
                self._clear_plan()
                self._save()

    def _set_mode(self, payload: dict[str, object]) -> None:
        mode = str(payload.get("mode") or "")
        if mode not in _MODES:
            raise DeskError("Unknown structure.")
        with self._lock:
            items = self._structures()
            active = str(self._state.get("structure") or "library")
            for item in items:
                if item["id"] == active:
                    item["mode"] = mode
            self._state["structures"] = items
            self._clear_plan()
            self._save()

    def _new_structure(self, payload: dict[str, object]) -> None:
        name = _slug(str(payload.get("name") or "Custom"))
        with self._lock:
            items = self._structures()
            ident = f"custom-{len(items) + 1}"
            items.append({"id": ident, "name": name, "mode": "custom"})
            self._state["structures"] = items
            self._state["structure"] = ident
            self._save()

    def _set_cleanup(self, payload: dict[str, object]) -> None:
        current = self._cleanup()
        for key in ("asd", "duplicates", "least"):
            if key in payload:
                current[key] = bool(payload[key])
        with self._lock:
            self._state["cleanup"] = current
            self._clear_plan()
            self._save()

    def _set_style(self, payload: dict[str, object]) -> None:
        style = str(payload.get("style") or "keep")
        if style not in _STYLES:
            raise DeskError("Unknown file tag style.")
        with self._lock:
            self._state["tag_style"] = style
            self._save()

    def _search(self, payload: dict[str, object]) -> None:
        with self._lock:
            self.search = str(payload.get("q") or "")

    def _pin(self, payload: dict[str, object]) -> None:
        path = str(payload.get("path") or "").strip()
        if not path:
            raise DeskError("Choose a file or folder.")
        with self._lock:
            pinned = self._state.get("pinned")
            items = pinned if isinstance(pinned, list) else []
            items = [item for item in items if isinstance(item, str)]
            if path not in items:
                items.append(path)
            self._state["pinned"] = items
            self._save()

    def _save_settings(self, payload: dict[str, object]) -> None:
        cfg = self._fresh_config()
        paths = payload.get("paths")
        if isinstance(paths, dict):
            library = paths.get("library")
            splice = paths.get("splice")
            projects = paths.get("projects")
            ableton = paths.get("ableton")
            if isinstance(library, str) and library.strip():
                cfg.library_roots = [Path(library.strip())]
            if isinstance(splice, str) and splice.strip():
                cfg.splice_folder = Path(splice.strip())
            if isinstance(projects, str):
                cfg.project_folders = [Path(projects.strip())] if projects.strip() else []
            if isinstance(ableton, str):
                cfg.ableton_user_library = Path(ableton.strip()) if ableton.strip() else None
            presets = dict(cfg.non_audio.presets)
            for key, value in paths.items():
                if key in {"library", "splice", "projects", "ableton"}:
                    continue
                if isinstance(value, str):
                    presets[str(key)] = value.strip()
            cfg.non_audio.presets = presets
        style = str(self._state.get("tag_style") or "keep")
        cfg.rename.enabled = style != "keep"
        if style == "tagged":
            cfg.rename.template = "{type}_{character}_{key}_{bpm}_{original}"
        elif style == "lowercase":
            cfg.rename.template = "{original}"
        save_config(cfg, self.config_path)
        with self._lock:
            self.message = "Preferences saved."

    def _reset_settings(self, payload: dict[str, object]) -> None:
        del payload
        with self._lock:
            self._state["tag_style"] = "keep"
            self._save()
            self.message = "File tag style set back to keep. Paths stay as saved."

    def _go(self, payload: dict[str, object]) -> None:
        screen = str(payload.get("screen") or "organize")
        with self._lock:
            if self.phase == "running":
                raise DeskError("A job is already running.")
            self.phase = "running"
            self.message = "Reading the folder…"
            self._plan = None
            self._summary = {}
        thread = threading.Thread(target=self._plan_job, args=(screen,), daemon=True)
        thread.start()

    def _apply(self, payload: dict[str, object]) -> None:
        del payload
        with self._lock:
            if self.phase != "ready" or self._plan is None:
                raise DeskError("Make a plan with Go first.")
            self.phase = "running"
            self.message = "Filing…"
        thread = threading.Thread(target=self._apply_job, daemon=True)
        thread.start()

    def _cancel(self, payload: dict[str, object]) -> None:
        del payload
        with self._lock:
            if self.phase == "running":
                raise DeskError("Wait for the current job to finish.")
            self._clear_plan()
            self.message = ""

    def _plan_job(self, screen: str) -> None:
        try:
            cfg = self._fresh_config()
            cfg = cfg.model_copy(deep=True)
            cfg.rename.enabled = False
            if screen == "cleanup":
                plan, kind = self._cleanup_plan(cfg)
            else:
                plan, kind = self._organize_plan(cfg)
            self._stamp_titles(plan, cfg)
            summary = _summarize(plan)
            with self._lock:
                self._plan = plan
                self._plan_kind = kind
                self._summary = summary
                self.phase = "ready"
                count = summary["count"]
                self.message = f"{count} files in the plan. Apply files them. Nothing has moved."
        except Exception as exc:
            with self._lock:
                self.phase = "error"
                self.message = str(exc) or "The plan failed."

    def _apply_job(self) -> None:
        try:
            with self._lock:
                plan = self._plan
                kind = self._plan_kind
            if plan is None:
                raise DeskError("The plan expired.")
            cfg = self._fresh_config()
            conn = self.connect()
            try:
                if plan.ops:
                    batch = mover.execute(conn, plan, kind, cfg.root)
                    for op in plan.ops:
                        if op.mode == "trash":
                            conn.execute(
                                "DELETE FROM files WHERE current_path=?", (str(op.src),)
                            )
                    conn.commit()
                    message = f"Batch {batch} filed. Undo puts moves back. Trash stays in Trash."
                else:
                    message = "Nothing to file."
            finally:
                conn.close()
            with self._lock:
                self._clear_plan()
                self.phase = "idle"
                self.message = message
                self.pack = None
        except Exception as exc:
            with self._lock:
                self.phase = "error"
                self.message = str(exc) or "Filing failed. The journal has the batch."

    def _organize_plan(self, cfg: Config) -> tuple[planner.Plan, str]:
        with self._lock:
            source = self.pack
            selected = str(self._state.get("selected") or "")
            mode = self._active_mode()
            tags = self._pair("meta")
            rows = self._folders(cfg)
        if source is None:
            row = next((item for item in rows if item["id"] == selected), None)
            if row is None:
                raise DeskError("Choose a pack, or a folder in the library.")
            source = Path(row["path"])
        if not source.exists():
            raise DeskError(f"{source} is not on disk.")
        dest = self._destination(cfg, rows, selected, mode)
        staging = Path(tempfile.mkdtemp(prefix="staging-", dir=self._staging_root()))
        conn = self.connect()
        try:
            result = scan([source], staging)
            verdicts: dict[int, Verdict] = {}
            if mode == "type":
                prompts = cfg.clap_prompts if cfg.clap_enabled else {}
                verdicts, warnings = annotate(
                    result.items, conn, prompts, use_clap=cfg.clap_enabled
                )
                result.warnings.extend(warnings)
                router: planner.Router = planner.sort_router(cfg, verdicts)
                kind = "ingest"
            else:
                tags = self._pair("meta")
                router = _structure_router(
                    mode, dest, source, tags["genre"], tags["mood"]
                )
                kind = mode
            copy_roots = [cfg.splice_folder] if cfg.keep_originals else []
            plan = planner.build_plan(result.items, cfg, router, copy_roots, verdicts, conn)
            plan.notes.extend(result.warnings)
            return plan, kind
        finally:
            conn.close()
            shutil.rmtree(staging, ignore_errors=True)

    def _cleanup_plan(self, cfg: Config) -> tuple[planner.Plan, str]:
        flags = self._cleanup()
        plan = planner.Plan()
        scope = self._scope(cfg)
        if flags["duplicates"]:
            self._duplicate_ops(plan, cfg, scope)
        if flags["asd"]:
            self._asd_ops(plan, scope)
        if flags["least"] and not flags["duplicates"] and not flags["asd"]:
            plan.notes.append(
                "Least used stays in the list. Turn on a delete switch to remove files."
            )
        return plan, "cleanup"

    def _scope(self, cfg: Config) -> Path:
        with self._lock:
            selected = str(self._state.get("selected") or "")
            rows = self._folders(cfg)
        row = next((item for item in rows if item["id"] == selected), None)
        if row is None:
            return cfg.root
        return Path(row["path"])

    def _duplicate_ops(self, plan: planner.Plan, cfg: Config, scope: Path) -> None:
        conn = self.connect()
        try:
            groups: dict[str, list[Path]] = {}
            prints: dict[str, list[tuple[Path, str | None]]] = {}
            root = scope if scope.exists() else cfg.root
            for path in scan_tree([root]):
                content, raw, finger = identify(conn, path)
                key = content or raw
                if key:
                    groups.setdefault(key, []).append(path)
                if finger:
                    prints.setdefault(finger, []).append((path, content))
        finally:
            conn.close()
        for paths in groups.values():
            if len(paths) < 2:
                continue
            kept = _canonical(paths, cfg.root)
            for extra in paths:
                if extra == kept:
                    continue
                fake = extra.with_name(extra.name + ".crate-trash")
                plan.ops.append(
                    planner.PlannedOp(
                        extra, fake, "trash", "duplicate", 1.0, extra.name, None
                    )
                )
        for entries in prints.values():
            hashes = {content for _, content in entries if content}
            if len(hashes) < 2:
                continue
            shown = ", ".join(path.name for path, _content in entries)
            plan.notes.append(f"Near-duplicate, left in place: {shown}")

    def _asd_ops(self, plan: planner.Plan, scope: Path) -> None:
        if not scope.exists():
            return
        files = [scope] if scope.is_file() else sorted(scope.rglob("*.asd"))
        for path in files:
            if path.suffix.lower() != ".asd" or not path.is_file():
                continue
            plan.ops.append(
                planner.PlannedOp(
                    path,
                    path.with_name(path.name + ".crate-trash"),
                    "trash",
                    "analysis",
                    1.0,
                    path.name,
                    None,
                )
            )
        if files:
            plan.notes.append("Analysis files go to the Trash. Live will rebuild warp data.")

    def _stamp_titles(self, plan: planner.Plan, cfg: Config) -> None:
        del cfg
        style = str(self._state.get("tag_style") or "keep")
        if style == "keep":
            return
        title = self._pair("title")
        taken: set[Path] = set()
        audio: dict[Path, Path] = {}
        for op in plan.ops:
            if op.sidecar or op.mode in {"trash", "symlink"}:
                continue
            renamed = _titled_name(op.dst.name, title["genre"], title["mood"], style, op.label)
            dst = op.dst.with_name(renamed)
            while dst in taken or (dst.exists() and dst != op.dst):
                dst = dst.with_name(f"{dst.stem} (2){dst.suffix}")
            taken.add(dst)
            audio[op.src] = dst
            op.dst = dst
            op.renamed = dst.name != Path(renamed).name or style != "keep"
        for op in plan.ops:
            if not op.sidecar or not str(op.src).endswith(".asd"):
                continue
            sample = Path(str(op.src)[:-4])
            if sample in audio:
                op.dst = Path(str(audio[sample]) + ".asd")

    def _active_mode(self) -> str:
        active = str(self._state.get("structure") or "library")
        for item in self._structures():
            if item["id"] == active:
                return item["mode"]
        return "type"

    def _staging_root(self) -> Path:
        root = self.home / "staging"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _clear_plan(self) -> None:
        self._plan = None
        self._plan_kind = ""
        self._summary = {}
        if self.phase != "running":
            self.phase = "idle"

    def _undo(self, payload: dict[str, object]) -> None:
        del payload
        conn = self.connect()
        try:
            batch = mover.latest_done_batch(conn)
            if batch is None:
                raise DeskError("Nothing to undo.")
            warnings = mover.undo(conn, batch)
        finally:
            conn.close()
        with self._lock:
            extra = f" {warnings[0]}" if warnings else ""
            self.message = f"Batch {batch} reversed.{extra}"

    def _correct(self, payload: dict[str, object]) -> None:
        path = Path(str(payload.get("path") or "")).expanduser()
        label = str(payload.get("label") or "").strip()
        if not path.exists():
            raise DeskError("That file is not on disk.")
        try:
            correction = parse_label(label)
        except ValueError as exc:
            raise DeskError(str(exc)) from exc
        cfg = self._fresh_config()
        cfg = cfg.model_copy(deep=True)
        cfg.rename.enabled = False
        staging = Path(tempfile.mkdtemp(prefix="staging-", dir=self._staging_root()))
        conn = self.connect()
        try:
            result = scan([path], staging)
            verdicts, _warnings = annotate(result.items, conn, {}, use_clap=False)
            item = next((it for it in result.items if it.kind == "audio"), None)
            if item is None:
                raise DeskError("No audio in that file.")
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
            identity = current.content_hash or current.raw_hash
            if identity:
                save_correction(conn, identity, correction)
            batch = mover.execute(conn, plan, "correct", cfg.root) if plan.ops else None
        finally:
            conn.close()
            shutil.rmtree(staging, ignore_errors=True)
        with self._lock:
            self.message = "Label saved." if batch is None else f"Label saved. Batch {batch}."

    def _projects(self, payload: dict[str, object]) -> None:
        raw = str(payload.get("path") or "").strip()
        if not raw:
            raise DeskError("Choose the Ableton project folder.")
        path = Path(raw).expanduser()
        conn = self.connect()
        try:
            indexed = index_projects(conn, [path])
        finally:
            conn.close()
        cfg = self._fresh_config()
        if path not in cfg.project_folders:
            cfg.project_folders = [*cfg.project_folders, path]
            save_config(cfg, self.config_path)
        with self._lock:
            self.message = f"Indexed {len(indexed)} Live sets."

    def _analyze(self, payload: dict[str, object]) -> None:
        del payload
        source = self.pack
        if source is None:
            raise DeskError("Choose a pack first.")
        cfg = self._fresh_config()
        staging = Path(tempfile.mkdtemp(prefix="staging-", dir=self._staging_root()))
        conn = self.connect()
        try:
            result = scan([source], staging)
            prompts = cfg.clap_prompts if cfg.clap_enabled else {}
            verdicts, warnings = annotate(
                result.items, conn, prompts, use_clap=cfg.clap_enabled
            )
        finally:
            conn.close()
            shutil.rmtree(staging, ignore_errors=True)
        lines = [
            f"{item.path.name}: {verdicts[id(item)].type or '?'}"
            for item in result.items
            if id(item) in verdicts
        ]
        with self._lock:
            self.message = "; ".join(lines[:8]) or "Nothing to analyze."
            if warnings:
                self.message += " " + warnings[0]

    def _eval(self, payload: dict[str, object]) -> None:
        raw = str(payload.get("path") or "").strip()
        path = Path(raw).expanduser()
        if not path.is_file():
            raise DeskError("Choose a labels CSV.")
        rows = read_labels(path)
        cfg = self._fresh_config()
        staging = Path(tempfile.mkdtemp(prefix="staging-", dir=self._staging_root()))
        conn = self.connect()
        try:
            result = scan([Path(row[0]) for row in rows], staging)
            by_name = {item.path.resolve(): item for item in result.items}
            prompts = cfg.clap_prompts if cfg.clap_enabled else {}
            verdicts, _warnings = annotate(
                result.items, conn, prompts, use_clap=cfg.clap_enabled
            )
        finally:
            conn.close()
            shutil.rmtree(staging, ignore_errors=True)
        gold: list[tuple[str, str]] = []
        predicted: list[str | None] = []
        for path_s, typ, _loop in rows:
            item = by_name.get(Path(path_s).expanduser().resolve())
            verdict = verdicts.get(id(item)) if item else None
            gold.append((path_s, typ))
            predicted.append(verdict.type if verdict else None)
        report = score(gold, predicted)
        parts = [
            f"{label} {precision:.2f}/{recall:.2f}"
            for label, (precision, recall, _support) in sorted(report.per_type.items())
        ]
        with self._lock:
            self.message = f"Eval {report.labeled} files. " + ", ".join(parts[:6])

    def _watch(self, payload: dict[str, object]) -> None:
        del payload
        with self._lock:
            self._state["watch"] = not bool(self._state.get("watch"))
            self._save()
            self.notice = "Watching Splice." if self._state["watch"] else ""

    def _similar(self, payload: dict[str, object]) -> None:
        path = str(payload.get("path") or "")
        conn = self.connect()
        try:
            row = conn.execute(
                "SELECT content_hash FROM files WHERE current_path=?", (path,)
            ).fetchone()
            features = None if row is None or not row["content_hash"] else features_for_hash(
                conn, row["content_hash"]
            )
            matches = [] if features is None else similar(conn, features, limit=5)
        finally:
            conn.close()
        if not matches:
            raise DeskError("Analyze that file before looking for neighbors.")
        names = [Path(match[1]["current_path"]).name for match in matches]
        with self._lock:
            self.message = "Sounds like " + ", ".join(names)


def _structure_router(
    mode: str, dest: Path, source: Path, genre: str, mood: str
) -> planner.Router:
    def route(item: Item) -> tuple[Path, str, float] | None:
        parent = _relative_parent(item, source)
        if mode == "flat":
            folder = dest
        elif mode == "original":
            folder = dest / parent
        elif mode == "genre":
            pack = genre.strip() or item.pack or "Untagged"
            folder = dest / _slug(pack)
            if mood.strip():
                folder = folder / _slug(mood)
            folder = folder / parent
        elif mode == "random":
            folder = dest / _BINS[sum(item.path.name.encode()) % len(_BINS)]
        else:
            folder = dest / _slug(genre or "Custom") / parent
        return folder, mode, 1.0

    return route


def _relative_parent(item: Item, source: Path) -> Path:
    try:
        return item.path.resolve().relative_to(source.resolve()).parent
    except ValueError:
        return Path(*item.folders)


def _canonical(paths: list[Path], root: Path) -> Path:
    root = root.expanduser().resolve()
    inside = [path for path in paths if _is_relative(path, root)]
    return inside[0] if inside else paths[0]


def _is_relative(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root)
    except ValueError:
        return False
    return True


def _titled_name(filename: str, genre: str, mood: str, style: str, label: str) -> str:
    path = Path(filename)
    parts: list[str] = []
    if style == "tagged":
        parts.extend(piece for piece in (genre, mood, label, path.stem) if piece)
    else:
        parts.extend(piece for piece in (genre, mood, path.stem) if piece)
    stem = "_".join(_slug(piece) for piece in parts).lower()
    stem = stem.strip(" _-.") or path.stem.lower()
    return stem + path.suffix.lower()


def _summarize(plan: planner.Plan) -> dict[str, object]:
    shown = [
        {"src": op.src.name, "dst": str(op.dst), "mode": op.mode, "label": op.label}
        for op in plan.ops
        if not op.sidecar
    ]
    return {
        "count": len(shown),
        "sidecars": sum(1 for op in plan.ops if op.sidecar),
        "skipped": len(plan.skipped),
        "notes": plan.notes[:8],
        "ops": shown[:60],
    }
