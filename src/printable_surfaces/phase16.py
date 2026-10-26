"""Phase 16b — minimal-stretch versions of all five Platonic solids, normalized by volume.

Each solid is oriented with its *least-stretch* axis vertical (face-down where that's easier,
vertex-down where face-down would need more stretch), with explicit horizontal-face handling:
the bottom face gets a base lattice on the build plate (horizontal rods, exempt), the top face
is left open (its edges become bridges). All five are scaled to the same *shape* volume
(200 cm³). Emits the T5 version (Ø1.8 rod, 50 % thicker than T4) with the mesh clipped flush
to the polytope (bottom rods/nodes sit on the plate), plus a material table.

Run: uv run python -m printable_surfaces.phase16
"""

from __future__ import annotations

import os

import numpy as np
from scipy.spatial import ConvexHull

from .artifacts import register
from .lattice import build_frame_mesh
from .platonic import (cube_face_down, dodecahedron, face_planes, icosahedron,
                       octahedron_face_down, tetrahedron)
from .printable import printable_lattice

PLA_DENSITY = 1.24  # g/cm^3 (PrusaSlicer nominal PLA)

ROD_R = 0.9       # Ø1.8 rod (T5: 50 % thicker than T4's Ø1.2)
JOINT_R = 1.35    # Ø2.7 joint (keeps the 1.5 rod:joint ratio, so joints still read as nodes)
BASE_CUT = 2.0    # mm — flat base cut for vertex-down solids (a point base can't print)
TARGET_VOLUME = 200_000.0  # mm³ (200 cm³) — normalize to equal *shape* volume

# (name, geometry, lattice params) — spacing scaled to each solid's size so small solids
# aren't empty and big ones stay sparse.
SOLIDS = [
    ("cube",        cube_face_down(50.0),     dict(spacing=14, n_face=16, n_interior=160)),
    ("tetrahedron", tetrahedron(50.0),        dict(spacing=12, n_face=18, n_interior=80)),
    ("octahedron",  octahedron_face_down(50.0), dict(spacing=14, n_face=14, n_interior=160)),
    ("icosahedron", icosahedron(50.0),        dict(spacing=17, n_face=12, n_interior=60)),
    ("dodecahedron", dodecahedron(50.0),      dict(spacing=17, n_face=12, n_interior=160)),
]


def material_volume_mm3(nodes, rods, rod_r, joint_r):
    L = np.linalg.norm(nodes[rods[:, 1]] - nodes[rods[:, 0]], axis=1)
    rod_vol = float(np.sum(np.pi * rod_r**2 * np.maximum(0.0, L - 2.0 * joint_r)))
    node_vol = float((4.0 / 3.0) * np.pi * joint_r**3 * len(nodes))
    return rod_vol + node_vol


def main() -> None:
    out_dir = "output/lattice/minstretch_v2"
    os.makedirs(out_dir, exist_ok=True)
    folder = "Platonic solids · min-stretch v2 (Ø1.8)"

    print("| Solid | W×D×H (mm) | nodes | rods | cm³ | PLA (g) | base |")
    print("|---|---|---|---|---|---|---|")
    rows = []
    for name, (verts, edges, faces), params in SOLIDS:
        nodes, rods = printable_lattice(verts, edges, faces, seed=1, iters=900, lam=1.0, **params)
        shape_vol = ConvexHull(verts).volume
        scale = (TARGET_VOLUME / shape_vol) ** (1.0 / 3.0)
        sc = nodes * scale
        poly = np.asarray(verts, dtype=float) * scale
        w, d, h = np.ptp(verts, axis=0) * scale
        vol_cm3 = material_volume_mm3(sc, rods, ROD_R, JOINT_R) / 1000.0
        grams = vol_cm3 * PLA_DENSITY

        # Clip the *joints* at each face plane offset outward by half a rod diameter. The
        # face rods are tangent to that plane, so they stay whole solid cylinders; only the
        # joint spheres (which bulge past the rods) get their protruding caps cut off. The
        # lowest surface is then the rods' own outer surface, so the bottom lattice really
        # touches the plate.
        normals, centers = face_planes(poly, faces)
        offset_centers = centers + normals * ROD_R
        has_floor = bool(np.any((np.abs(normals[:, 2]) > 0.99) & (normals[:, 2] < 0.0)))
        base_cut = None if has_floor else float(poly[:, 2].min()) + BASE_CUT
        mesh = build_frame_mesh(sc, rods, rod_radius=ROD_R, joint_radius=JOINT_R,
                                color_by_direction=True,
                                clip_planes=(normals, offset_centers), clip_base_z=base_cut)
        mesh.apply_translation([0.0, 0.0, -float(mesh.bounds[0][2])])  # sit on z = 0

        base_kind = "face on plate" if has_floor else f"flat cut {BASE_CUT:.0f} mm"
        rows.append((name, f"{w:.0f}×{d:.0f}×{h:.0f}", len(nodes), len(rods), vol_cm3, grams, base_kind))

        stl = f"{out_dir}/{name}_T5_S2.stl"
        ply = f"{out_dir}/{name}_T5_S2.ply"
        mesh.export(stl)
        mesh.export(ply)
        register("lattice", ply, "ply",
                 f"{name} · rod angle · Ø1.8 · {grams:.1f} g",
                 folder=folder)
        register("lattice", stl, "stl",
                 f"{name} · rod Ø1.8 · {w:.0f}×{d:.0f}×{h:.0f} mm · {grams:.1f} g",
                 folder=folder)

    for name, dims, nn, nr, vol, grams, base in rows:
        print(f"| {name} | {dims} | {nn} | {nr} | {vol:.2f} | {grams:.1f} | {base} |")

    print(f"\nSaved {len(rows)} STLs + PLYs in {out_dir}")


if __name__ == "__main__":
    main()
