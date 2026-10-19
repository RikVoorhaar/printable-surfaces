"""Artifact registry: keeps ``manifest.json`` (the web dashboard's data) in sync.

Every phase calls :func:`register` when it writes a file the dashboard should surface,
so ``index.html`` automatically lists it under the right pipeline stage.

Stages are the four fixed pipeline stops: ``surface`` → ``triangulation`` → ``lattice``
→ ``slicer``.
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
    data.setdefault("stages", [])
    return data


def register(
    stage: str,
    path: str,
    kind: str,
    label: str,
    manifest_path: str | Path = "manifest.json",
) -> None:
    """Add or update an artifact entry under ``stage`` (idempotent by ``path``).

    ``kind`` is one of ``"stl"``, ``"image"``, ``"gcode"``, or ``"other"`` — the dashboard
    decides how to render each.
    """
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")

    data = load(manifest_path)
    stages = data["stages"]
    target = next((s for s in stages if s.get("id") == stage), None)
    if target is None:
        target = {"id": stage, "label": stage.title(), "description": "", "artifacts": []}
        stages.append(target)

    artifacts = target.setdefault("artifacts", [])
    entry = next((a for a in artifacts if a.get("path") == path), None)
    if entry is None:
        entry = {"path": path}
        artifacts.append(entry)
    entry.update(kind=kind, label=label)

    Path(manifest_path).write_text(json.dumps(data, indent=2) + "\n")


def remove_artifact(path: str, manifest_path: str | Path = "manifest.json") -> None:
    """Remove any artifact entry matching ``path`` from the manifest."""
    data = load(manifest_path)
    for stage in data["stages"]:
        stage["artifacts"] = [a for a in stage.get("artifacts", []) if a.get("path") != path]
    Path(manifest_path).write_text(json.dumps(data, indent=2) + "\n")
