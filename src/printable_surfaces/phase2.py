"""Phase 2 — surface-frame lattice → STL (walking skeleton), selected densities.

Triangulates the cube-sphere, extracts its 1-skeleton (edges → rods, vertices → nodes),
thickens it into a printable rod+node mesh, and emits STLs — both the thin triangulation
wireframe and the printable lattice — for the two selected densities.

Run:  uv run python -m printable_surfaces.phase2
"""

from __future__ import annotations

import glob
import os

from .artifacts import register, remove_artifact
from .lattice import build_frame_mesh, edges_from_faces, merge_vertices
from .parametrizations import cube_sphere_grid

OBJECT_MM = 100.0     # target bounding box (scaled up: 60 → 100 mm)
ROD_MM = 1.2          # printable rod radius (kept ≥1.2 mm for strength)
JOINT_MM = 1.8        # node joint radius
WIRE_ROD_MM = 0.4     # thin wireframe (triangulation stage)
WIRE_JOINT_MM = 0.6
BASELINE_N = 12       # the original (too-dense) reference
DENSITY_N = [5, 6]    # the two selected densities (~13%, ~21% of n=12)


def _frame(n: int):
    radius = OBJECT_MM / 2.0
    pts, faces = cube_sphere_grid(n=n, radius=radius)
    nodes, inverse = merge_vertices(pts)
    edges = edges_from_faces(nodes, inverse[faces])
    return nodes, edges


def _baseline_edges() -> int:
    pts, faces = cube_sphere_grid(n=BASELINE_N, radius=1.0)
    nodes, inverse = merge_vertices(pts)
    return len(edges_from_faces(nodes, inverse[faces]))


def _clear(pattern: str) -> None:
    for path in glob.glob(pattern):
        remove_artifact(path)
        os.remove(path)


def main() -> None:
    base = _baseline_edges()
    _clear("output/lattice/sphere_surface_frame*.stl")
    _clear("output/triangulation/sphere_wireframe*.stl")

    for n in DENSITY_N:
        nodes, edges = _frame(n)
        pct = len(edges) / base * 100.0

        lattice = build_frame_mesh(nodes, edges, rod_radius=ROD_MM, joint_radius=JOINT_MM)
        lattice_path = f"output/lattice/sphere_surface_frame_n{n:02d}.stl"
        os.makedirs(os.path.dirname(lattice_path), exist_ok=True)
        lattice.export(lattice_path)
        register("lattice", lattice_path, "stl", f"Surface frame · {pct:.0f}% ({len(edges)} rods)")

        wire = build_frame_mesh(nodes, edges, rod_radius=WIRE_ROD_MM, joint_radius=WIRE_JOINT_MM)
        wire_path = f"output/triangulation/sphere_wireframe_n{n:02d}.stl"
        os.makedirs(os.path.dirname(wire_path), exist_ok=True)
        wire.export(wire_path)
        register("triangulation", wire_path, "stl", f"Triangulation · {pct:.0f}% ({len(nodes)} nodes)")

        print(f"n={n:2d}  {pct:5.1f}%  {len(nodes):4d} nodes  {len(edges):4d} rods")


if __name__ == "__main__":
    main()
