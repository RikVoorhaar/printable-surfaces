"""Phase 10 — force-directed relaxed lattice.

Seeds vertices randomly, wires them (k-NN), then relaxes the layout with springs +
repulsion (see ``relax.py``) so all rods converge to a near-uniform length. Emits STLs,
slices them, and registers them on the dashboard.

Run:  uv run python -m printable_surfaces.phase10
"""

from __future__ import annotations

import os

import numpy as np

from .artifacts import register
from .lattice import build_frame_mesh, complete_surface
from .relax import relaxed_lattice
from .slicer import slice_stl

OBJECT_MM = 100.0
ROD_MM = 1.2
JOINT_MM = 1.8
# (name, label, target rod length, seed)
VARIANTS = [
    ("relaxed_L18_s1", "Relaxed · L=18 · seed 1", 18.0, 1),
    ("relaxed_L18_s2", "Relaxed · L=18 · seed 2", 18.0, 2),
    ("relaxed_L16_s1", "Relaxed · L=16 · seed 1", 16.0, 1),
    ("relaxed_L20_s1", "Relaxed · L=20 · seed 1", 20.0, 1),
]


def build_and_register(radius: float, name: str, label: str, L: float, seed: int) -> None:
    nodes, edges = relaxed_lattice(radius, L, k=4, seed=seed)
    nodes, edges, surface_edges, b = complete_surface(radius, nodes, edges)
    all_edges = np.unique(np.sort(np.vstack([edges, surface_edges]), axis=1), axis=0)
    mesh = build_frame_mesh(nodes, all_edges, rod_radius=ROD_MM, joint_radius=JOINT_MM)

    # Report rod-length uniformity (interior vs surface).
    spring_len = np.linalg.norm(nodes[edges[:, 0]] - nodes[edges[:, 1]], axis=1)
    hull_len = np.linalg.norm(nodes[surface_edges[:, 0]] - nodes[surface_edges[:, 1]], axis=1)

    path = f"output/lattice/sphere_{name}.stl"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mesh.export(path)
    register("lattice", path, "stl", f"{label} · {len(nodes)} nodes / {len(all_edges)} rods")

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
        f"{label:22s} {len(nodes):4d} nodes ({b.sum():3d} surf)  {len(all_edges):4d} rods  "
        f"spring {spring_len.mean():.1f}±{spring_len.std():.1f}  hull {hull_len.mean():.1f}±{hull_len.std():.1f}  -> {status}"
    )


def main() -> None:
    radius = OBJECT_MM / 2.0
    for name, label, L, seed in VARIANTS:
        build_and_register(radius, name, label, L, seed)


if __name__ == "__main__":
    main()
