"""Phase 16 — version matrix: all five Platonic solids, sparse + slim + scaled.

Generates the support-free lattice for each of the five (stretched) Platonic solids, then
exports the chosen "T4×S2" version (Ø1.2 rod, 0.8 scale, sparse base) plus a material table.

Run: uv run python -m printable_surfaces.phase16
"""

from __future__ import annotations

import os

import numpy as np

from .artifacts import register
from .lattice import build_frame_mesh
from .platonic import cube_rhombohedron, dodecahedron, icosahedron, octahedron, tetrahedron
from .printable import printable_lattice

PLA_DENSITY = 1.24  # g/cm^3 (PrusaSlicer nominal PLA; Wikipedia lists 1.21–1.43)

# chosen "slim + smaller" version: rod radius 0.6 (Ø1.2), joint 0.9 (Ø1.8), scale 0.8
ROD_R = 0.6
JOINT_R = 0.9
SCALE = 0.8

# (name, geometry, lattice params) — sparse base (spacing 17, ~25% fewer rods than the dense cut)
SOLIDS = [
    ("cube",        cube_rhombohedron(50.0, 1.69), dict(spacing=17, n_face=36, n_interior=80)),
    ("tetrahedron", tetrahedron(50.0),             dict(spacing=17, n_face=12, n_interior=40)),
    ("octahedron",  octahedron(50.0),              dict(spacing=17, n_face=15, n_interior=60)),
    ("icosahedron", icosahedron(50.0),             dict(spacing=17, n_face=12, n_interior=60)),
    ("dodecahedron", dodecahedron(50.0),           dict(spacing=17, n_face=10, n_interior=70)),
]


def material_volume_mm3(nodes, rods, rod_r, joint_r):
    L = np.linalg.norm(nodes[rods[:, 1]] - nodes[rods[:, 0]], axis=1)
    rod_vol = float(np.sum(np.pi * rod_r**2 * np.maximum(0.0, L - 2.0 * joint_r)))
    node_vol = float((4.0 / 3.0) * np.pi * joint_r**3 * len(nodes))
    return rod_vol + node_vol


def main() -> None:
    out_dir = "output/lattice/solids"
    os.makedirs(out_dir, exist_ok=True)

    print("| Solid | W×D×H (mm) | nodes | rods | cm³ | PLA (g) |")
    print("|---|---|---|---|---|---|")
    rows = []
    for name, (verts, edges, faces), params in SOLIDS:
        nodes, rods = printable_lattice(verts, edges, faces, seed=1, iters=900, lam=1.0, **params)
        sc = nodes * SCALE
        w, d, h = np.ptp(verts, axis=0) * SCALE
        vol_cm3 = material_volume_mm3(sc, rods, ROD_R, JOINT_R) / 1000.0
        grams = vol_cm3 * PLA_DENSITY
        rows.append((name, f"{w:.0f}×{d:.0f}×{h:.0f}", len(nodes), len(rods), vol_cm3, grams))

        mesh = build_frame_mesh(sc, rods, rod_radius=ROD_R, joint_radius=JOINT_R)
        stl = f"{out_dir}/{name}_T4_S2.stl"
        mesh.export(stl)
        register("lattice", stl, "stl",
                 f"{name} T4×S2 · rod Ø1.2 · {w:.0f}×{h:.0f} mm · {grams:.1f} g",
                 folder="Platonic solids · T4×S2")

    for name, dims, nn, nr, vol, grams in rows:
        print(f"| {name} | {dims} | {nn} | {nr} | {vol:.2f} | {grams:.1f} |")

    print(f"\nSaved {len(rows)} STLs in {out_dir}")


if __name__ == "__main__":
    main()
