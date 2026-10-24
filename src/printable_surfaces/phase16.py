"""Phase 16b — minimal-stretch versions of all five Platonic solids, normalized by volume.

Each solid is oriented with its *least-stretch* axis vertical (face-down where that's easier,
vertex-down where face-down would need more stretch), with explicit horizontal-face handling:
the bottom face gets a base lattice on the build plate (horizontal rods, exempt), the top face
is left open (its edges become bridges). All five are scaled to the same *shape* volume
(200 cm³). Emits the slim T4 version (Ø1.2 rod) plus a material table.

Run: uv run python -m printable_surfaces.phase16
"""

from __future__ import annotations

import os

import numpy as np
from scipy.spatial import ConvexHull

from .artifacts import register
from .lattice import build_frame_mesh
from .platonic import cube_face_down, dodecahedron, icosahedron, octahedron_face_down, tetrahedron
from .printable import printable_lattice

PLA_DENSITY = 1.24  # g/cm^3 (PrusaSlicer nominal PLA)

ROD_R = 0.6       # Ø1.2 rod (T4)
JOINT_R = 0.9     # Ø1.8 joint
TARGET_VOLUME = 200_000.0  # mm³ (200 cm³) — normalize to equal *shape* volume

# (name, geometry, lattice params) — spacing scaled to each solid's size so small solids
# aren't empty and big ones stay sparse.
SOLIDS = [
    ("cube",        cube_face_down(50.0),     dict(spacing=14, n_face=16, n_interior=120)),
    ("tetrahedron", tetrahedron(50.0),        dict(spacing=12, n_face=18, n_interior=80)),
    ("octahedron",  octahedron_face_down(50.0), dict(spacing=14, n_face=14, n_interior=120)),
    ("icosahedron", icosahedron(50.0),        dict(spacing=17, n_face=12, n_interior=60)),
    ("dodecahedron", dodecahedron(50.0),      dict(spacing=17, n_face=12, n_interior=120)),
]


def material_volume_mm3(nodes, rods, rod_r, joint_r):
    L = np.linalg.norm(nodes[rods[:, 1]] - nodes[rods[:, 0]], axis=1)
    rod_vol = float(np.sum(np.pi * rod_r**2 * np.maximum(0.0, L - 2.0 * joint_r)))
    node_vol = float((4.0 / 3.0) * np.pi * joint_r**3 * len(nodes))
    return rod_vol + node_vol


def main() -> None:
    out_dir = "output/lattice/minstretch"
    os.makedirs(out_dir, exist_ok=True)
    folder = "Platonic solids · min-stretch"

    print("| Solid | W×D×H (mm) | nodes | rods | cm³ | PLA (g) |")
    print("|---|---|---|---|---|---|")
    rows = []
    for name, (verts, edges, faces), params in SOLIDS:
        nodes, rods = printable_lattice(verts, edges, faces, seed=1, iters=900, lam=1.0, **params)
        shape_vol = ConvexHull(verts).volume
        scale = (TARGET_VOLUME / shape_vol) ** (1.0 / 3.0)
        sc = nodes * scale
        w, d, h = np.ptp(verts, axis=0) * scale
        vol_cm3 = material_volume_mm3(sc, rods, ROD_R, JOINT_R) / 1000.0
        grams = vol_cm3 * PLA_DENSITY
        rows.append((name, f"{w:.0f}×{d:.0f}×{h:.0f}", len(nodes), len(rods), vol_cm3, grams))

        mesh = build_frame_mesh(sc, rods, rod_radius=ROD_R, joint_radius=JOINT_R)
        stl = f"{out_dir}/{name}_T4_S2.stl"
        mesh.export(stl)
        register("lattice", stl, "stl",
                 f"{name} · rod Ø1.2 · {w:.0f}×{d:.0f}×{h:.0f} mm · {grams:.1f} g",
                 folder=folder)

    for name, dims, nn, nr, vol, grams in rows:
        print(f"| {name} | {dims} | {nn} | {nr} | {vol:.2f} | {grams:.1f} |")

    print(f"\nSaved {len(rows)} STLs in {out_dir}")


if __name__ == "__main__":
    main()
