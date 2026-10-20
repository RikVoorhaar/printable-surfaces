"""Phase 11 — Coulomb-gas lattice (Idea A: relax first, wire later).

Relaxes a point cloud to a uniform ball with Coulomb repulsion + a harmonic trap (no
surface/interior distinction), then wires it with k-nearest neighbours. Emits STLs,
slices them, and registers them.

Run:  uv run python -m printable_surfaces.phase11
"""

from __future__ import annotations

import os

import numpy as np

from .artifacts import register
from .lattice import build_frame_mesh
from .relax import coulomb_lattice
from .slicer import slice_stl

OBJECT_MM = 100.0
ROD_MM = 1.2
JOINT_MM = 1.8
# (name, label, n_points, k)
VARIANTS = [
    ("coulomb_n120_k4", "Coulomb · 120 pts · k=4", 120, 4),
    ("coulomb_n160_k4", "Coulomb · 160 pts · k=4", 160, 4),
    ("coulomb_n200_k4", "Coulomb · 200 pts · k=4", 200, 4),
    ("coulomb_n160_k3", "Coulomb · 160 pts · k=3", 160, 3),
    ("coulomb_n160_k5", "Coulomb · 160 pts · k=5", 160, 5),
]


def build_and_register(radius: float, name: str, label: str, n: int, k: int) -> None:
    nodes, edges = coulomb_lattice(radius, n, k=k, seed=1)
    mesh = build_frame_mesh(nodes, edges, rod_radius=ROD_MM, joint_radius=JOINT_MM)

    edge_len = np.linalg.norm(nodes[edges[:, 0]] - nodes[edges[:, 1]], axis=1)

    path = f"output/lattice/sphere_{name}.stl"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mesh.export(path)
    register("lattice", path, "stl", f"{label} · {len(nodes)} nodes / {len(edges)} rods")

    gcode = f"output/slicer/sphere_{name}.gcode"
    try:
        proc = slice_stl(path, gcode)
        ok = proc.returncode == 0 and os.path.exists(gcode) and os.path.getsize(gcode) > 0
    except FileNotFoundError:
        ok = False
    status = "sliced OK" if ok else "FAILED"
    if ok:
        register("slicer", gcode, "gcode", f"{label} · {status}")

    print(
        f"{label:22s} {len(nodes):3d} nodes  {len(edges):4d} rods  "
        f"rod len {edge_len.mean():.1f}±{edge_len.std():.1f}  -> {status}"
    )


def main() -> None:
    radius = OBJECT_MM / 2.0
    for name, label, n, k in VARIANTS:
        build_and_register(radius, name, label, n, k)


if __name__ == "__main__":
    main()
