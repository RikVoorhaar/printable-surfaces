"""Phase 9 — interior lattice experiments (sparser + randomized).

Builds several unified cages (shared surface/interior nodes) with different interior
lattices — sparser diamond, jittered diamond, and randomized Delaunay webs — and slices
each. Run:  uv run python -m printable_surfaces.phase9
"""

from __future__ import annotations

import os

import numpy as np

from .artifacts import register
from .lattice import (
    build_frame_mesh,
    complete_surface,
    diamond_lattice,
    jittered_diamond,
    random_lattice,
)
from .slicer import slice_stl

OBJECT_MM = 100.0
ROD_MM = 1.2
JOINT_MM = 1.8


def build_and_register(radius: float, name: str, label: str, nodes: np.ndarray, edges: np.ndarray) -> None:
    nodes, edges, surface_edges, b = complete_surface(radius, nodes, edges)
    all_edges = np.unique(np.sort(np.vstack([edges, surface_edges]), axis=1), axis=0)
    mesh = build_frame_mesh(nodes, all_edges, rod_radius=ROD_MM, joint_radius=JOINT_MM)

    path = f"output/lattice/sphere_{name}.stl"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mesh.export(path)
    n_boundary = int(b.sum())
    register("lattice", path, "stl", f"{label} · {len(nodes)} nodes / {len(all_edges)} rods")

    gcode = f"output/slicer/sphere_{name}.gcode"
    try:
        proc = slice_stl(path, gcode)
        ok = proc.returncode == 0 and os.path.exists(gcode) and os.path.getsize(gcode) > 0
        text = (proc.stdout or "") + (proc.stderr or "")
    except FileNotFoundError:
        ok, text = False, "prusa-slicer binary not found"
    status = "sliced OK" if ok else "FAILED"
    if ok:
        register("slicer", gcode, "gcode", f"{label} · {status}")

    print(f"{label:24s} {len(nodes):4d} nodes ({n_boundary:3d} surf)  {len(all_edges):4d} rods  {len(mesh.faces):,} tris  -> {status}")


def main() -> None:
    radius = OBJECT_MM / 2.0

    nodes, edges = diamond_lattice(radius, 40.0)
    build_and_register(radius, "diamond_sparse_a40", "Diamond · sparse (a=40)", nodes, edges)

    nodes, edges = jittered_diamond(radius, 38.0, jitter=9.0, seed=1)
    build_and_register(radius, "jittered_diamond", "Jittered diamond", nodes, edges)

    nodes, edges = random_lattice(radius, n_surface=70, n_interior=90, k=4, seed=2)
    build_and_register(radius, "random_knn_1", "Random · k-NN #1", nodes, edges)

    nodes, edges = random_lattice(radius, n_surface=60, n_interior=80, k=4, seed=3)
    build_and_register(radius, "random_knn_2", "Random · k-NN #2", nodes, edges)


if __name__ == "__main__":
    main()
