"""Phase 3 — slice the full cages with PrusaSlicer and record the results.

Slices each full-cage STL, captures the slicer's printability warnings, and registers the
G-code + a human-readable report on the dashboard under the ``slicer`` stage.

Run:  uv run python -m printable_surfaces.phase3
"""

from __future__ import annotations

import os

from .artifacts import register
from .slicer import slice_stl

TARGETS = [
    ("13%", "output/lattice/sphere_full_cage_13pct.stl"),
    ("21%", "output/lattice/sphere_full_cage_21pct.stl"),
]

_WARNINGS = [
    ("overhang", "Collapsing overhang"),
    ("low bed adhesion", "Low bed adhesion"),
    ("thin part", "Thin fragile part"),
    ("loose extrusions", "Loose extrusions"),
]


def main() -> None:
    for label, stl in TARGETS:
        tag = label.replace("%", "pct")
        gcode = f"output/slicer/sphere_full_cage_{tag}.gcode"
        os.makedirs(os.path.dirname(gcode), exist_ok=True)

        try:
            proc = slice_stl(stl, gcode)
            text = (proc.stdout or "") + "\n" + (proc.stderr or "")
            ok = proc.returncode == 0 and os.path.exists(gcode) and os.path.getsize(gcode) > 0
        except FileNotFoundError as e:
            ok = False
            text = f"prusa-slicer binary not found: {e}"
            proc = None

        warnings = [name for name, needle in _WARNINGS if needle in text]
        status = ("sliced OK" + (f" · warns: {', '.join(warnings)}" if warnings else "")) if ok else "FAILED"

        report = f"output/slicer/sphere_full_cage_{tag}.txt"
        with open(report, "w") as f:
            f.write(f"target: {stl}\n")
            f.write(f"returncode: {proc.returncode if proc else 'n/a'}\n")
            f.write(f"status: {status}\n\n--- slicer output ---\n{text}\n")

        register("slicer", gcode, "gcode", f"Full cage {label} · {status}") if ok else None
        register("slicer", report, "other", f"Full cage {label} · report")
        print(f"{label}: {status}")


if __name__ == "__main__":
    main()
