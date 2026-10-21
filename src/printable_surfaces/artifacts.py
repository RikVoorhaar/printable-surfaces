"""Artifact registry: keeps ``manifest.json`` (the web dashboard's data) in sync.

Artifacts are grouped into **folders** (collapsible on the dashboard), ordered newest
first. Each phase calls :func:`register` when it writes a file the dashboard should
surface. ``stage`` is kept as a small tag (``lattice``, ``slicer``, …); ``folder`` is the
batch/run the artifact belongs to (newest folder on top).
"""

from __future__ import annotations

import json
from pathlib import Path

STAGES = ("surface", "triangulation", "lattice", "slicer")


def load(manifest_path: str | Path = "manifest.json") -> dict:
    """Read the manifest, returning a fresh default if it doesn't exist."""
    path = Path(manifest_path)
    if path.exists():
        data = json.loads(path.read_text())
    else:
        data = {}
    data.setdefault("folders", [])
    return data


def register(
    stage: str,
    path: str,
    kind: str,
    label: str,
    folder: str | None = None,
    manifest_path: str | Path = "manifest.json",
) -> None:
    """Add or update an artifact in ``folder`` (idempotent by ``path``).

    ``kind`` is one of ``"stl"``, ``"image"``, ``"gcode"``, or ``"other"``. If ``folder``
    is omitted it defaults to ``stage``. New folders are inserted at the front (newest).
    """
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
    folder = folder or stage

    data = load(manifest_path)
    folders = data["folders"]
    target = next((f for f in folders if f.get("name") == folder), None)
    if target is None:
        target = {"name": folder, "artifacts": []}
        folders.insert(0, target)  # newest first

    artifacts = target.setdefault("artifacts", [])
    entry = next((a for a in artifacts if a.get("path") == path), None)
    if entry is None:
        entry = {"path": path}
        artifacts.append(entry)
    entry.update(kind=kind, label=label, stage=stage)

    Path(manifest_path).write_text(json.dumps(data, indent=2) + "\n")


def remove_artifact(path: str, manifest_path: str | Path = "manifest.json") -> None:
    """Remove any artifact entry matching ``path`` from the manifest."""
    data = load(manifest_path)
    for folder in data["folders"]:
        folder["artifacts"] = [a for a in folder.get("artifacts", []) if a.get("path") != path]
    # drop empty folders
    data["folders"] = [f for f in data["folders"] if f.get("artifacts")]
    Path(manifest_path).write_text(json.dumps(data, indent=2) + "\n")
