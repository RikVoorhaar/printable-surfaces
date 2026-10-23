"""Phase 14 — support-free printable lattice (Perplexity method).

Relax with a support term, generate printable candidates, select rods with a mixed-integer
program. Emits a colored PLY + STL. Run: uv run python -m printable_surfaces.phase14
"""

from __future__ import annotations

import os

import numpy as np

from .artifacts import register
from .lattice import build_frame_mesh
from .platonic import cube_rhombohedron
from .printable import printable_lattice
from .slicer import slice_stl

ROD_MM = 1.2
JOINT_MM = 1.8


def main() -> None:
    verts, edges, faces = cube_rhombohedron(50.0, 1.69)
    nodes, rods = printable_lattice(verts, edges, faces, spacing=15.0, n_face=24, n_interior=105,
                                    seed=1, iters=900, lam=1.0)

    d = nodes[rods[:, 1]] - nodes[rods[:, 0]]
    L = np.linalg.norm(d, axis=1)
    sin_elev = np.abs(d[:, 2]) / L
    unprintable = (sin_elev < np.sin(np.deg2rad(45))) & ~((sin_elev <= np.sin(np.deg2rad(5))) & (L <= 10))

    mesh = build_frame_mesh(nodes, rods, rod_radius=ROD_MM, joint_radius=JOINT_MM, color_by_direction=True)
    ply = "output/lattice/rhombohedron_printable.ply"
    stl = "output/lattice/rhombohedron_printable.stl"
    os.makedirs(os.path.dirname(ply), exist_ok=True)
    mesh.export(ply)
    mesh.export(stl)
    register("lattice", ply, "ply",
             f"Rhombohedron · printable · {len(nodes)} nodes / {len(rods)} rods / {unprintable.sum()} unprintable",
             folder="Cube · rhombohedron")

    ok = False
    try:
        proc = slice_stl(stl, "output/slicer/rhombohedron_printable.gcode")
        ok = proc.returncode == 0 and os.path.exists("output/slicer/rhombohedron_printable.gcode")
    except FileNotFoundError:
        pass

    print(f"nodes={len(nodes)} rods={len(rods)} unprintable={unprintable.sum()} -> {'sliced OK' if ok else 'slice n/a'}")


if __name__ == "__main__":
    main()
