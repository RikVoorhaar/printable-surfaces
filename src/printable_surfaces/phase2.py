"""Phase 2 — surface-frame lattice → STL (walking skeleton), density sweep.

Triangulates the cube-sphere, extracts its 1-skeleton (edges → rods, vertices → nodes),
thickens it into a printable rod+node mesh, and emits STLs at several densities — both the
thin triangulation wireframe and the printable lattice — registering them on the dashboard.

Run:  uv run python -m printable_surfaces.phase2
"""

from __future__ import annotations

import os

from .artifacts import register, remove_artifact
from .lattice import build_frame_mesh, edges_from_faces, merge_vertices
from .parametrizations import cube_sphere_grid

OBJECT_MM = 60.0      # target bounding box
ROD_MM = 1.2          # printable rod radius
JOINT_MM = 1.8        # node joint radius
WIRE_ROD_MM = 0.4     # thin wireframe (triangulation stage)
WIRE_JOINT_MM = 0.6
BASELINE_N = 12       # the original (too-dense) reference
DENSITY_N = [5, 6, 7, 8, 9]  # ~13%, 21%, 30%, 40%, 53% of n=12


def _frame(n: int):
    """Return (nodes, edges) for the cube-sphere surface frame at resolution ``n``."""
    radius = OBJECT_MM / 2.0
    pts, faces = cube_sphere_grid(n=n, radius=radius)
    nodes, inverse = merge_vertices(pts)
    edges = edges_from_faces(nodes, inverse[faces])
    return nodes, edges


def _baseline_edges() -> int:
    pts, faces = cube_sphere_grid(n=BASELINE_N, radius=1.0)
    nodes, inverse = merge_vertices(pts)
    return len(edges_from_faces(nodes, inverse[faces]))


def main() -> None:
    base = _baseline_edges()

    # Remove the original (too-dense) single versions.
    for old in ("output/lattice/sphere_surface_frame.stl", "output/triangulation/sphere_wireframe.stl"):
        remove_artifact(old)
        if os.path.exists(old):
            os.remove(old)

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

        print(f"n={n:2d}  {pct:5.1f}% density  {len(nodes):4d} nodes  {len(edges):4d} rods")


if __name__ == "__main__":
    main()
