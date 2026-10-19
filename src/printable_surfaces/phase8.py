"""Phase 8 — unified cage: surface + interior sharing nodes.

Builds ONE coherent lattice (diamond interior + a surface frame derived from the
diamond lattice's boundary nodes), thickens it, emits STLs, and slices them.

Run:  uv run python -m printable_surfaces.phase8
"""

from __future__ import annotations

import os

import numpy as np

from .artifacts import register
from .lattice import build_frame_mesh, unified_cage
from .slicer import slice_stl

OBJECT_MM = 100.0
ROD_MM = 1.2
JOINT_MM = 1.8
CELL_SIZES = [34.0, 30.0, 26.0]  # sparser / ~13% / denser


def main() -> None:
    radius = OBJECT_MM / 2.0

    for a in CELL_SIZES:
        tag = f"a{int(a)}"
        nodes, diamond_edges, surface_edges, bmask = unified_cage(radius, a)
        all_edges = np.vstack([diamond_edges, surface_edges])
        all_edges = np.unique(np.sort(all_edges, axis=1), axis=0)
        n_boundary = int(bmask.sum())

        mesh = build_frame_mesh(nodes, all_edges, rod_radius=ROD_MM, joint_radius=JOINT_MM)
        path = f"output/lattice/sphere_unified_{tag}.stl"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        mesh.export(path)
        register(
            "lattice", path, "stl",
            f"Unified cage · a={a:.0f} ({len(nodes)} nodes / {len(all_edges)} rods)",
        )

        print(
            f"a={a:.0f}: {len(nodes)} nodes ({n_boundary} surface) · "
            f"{len(diamond_edges)} diamond + {len(surface_edges)} surface = {len(all_edges)} rods · "
            f"{len(mesh.faces):,} tris"
        )

        # Slice it.
        gcode = f"output/slicer/sphere_unified_{tag}.gcode"
        try:
            proc = slice_stl(path, gcode)
            ok = proc.returncode == 0 and os.path.exists(gcode) and os.path.getsize(gcode) > 0
            text = (proc.stdout or "") + (proc.stderr or "")
        except FileNotFoundError:
            ok, text = False, "prusa-slicer binary not found"

        warnings = [w for w in ("overhang", "adhesion", "thin", "loose") if w in text]
        status = ("sliced OK" + (f" · warns: {', '.join(warnings)}" if warnings else "")) if ok else "FAILED"
        if ok:
            register("slicer", gcode, "gcode", f"Unified a={a:.0f} · {status}")
        print(f"    slice: {status}")


if __name__ == "__main__":
    main()
