"""Phase 2 — surface-frame lattice → STL (walking skeleton).

Triangulates the cube-sphere, extracts its 1-skeleton (edges → rods, vertices → nodes),
thickens it into a printable rod+node mesh, and emits STLs — both the thin triangulation
wireframe and the printable lattice — registering them on the dashboard.

Run:  uv run python -m printable_surfaces.phase2
"""

from __future__ import annotations

import os

from .artifacts import register
from .lattice import build_frame_mesh, edges_from_faces, merge_vertices
from .parametrizations import cube_sphere_grid

OBJECT_MM = 60.0      # target bounding box
ROD_MM = 1.2          # printable rod radius
JOINT_MM = 1.8        # node joint radius
WIRE_ROD_MM = 0.4     # thin wireframe (triangulation stage)
WIRE_JOINT_MM = 0.6
N = 12                # grid resolution per chart


def main() -> None:
    radius = OBJECT_MM / 2.0  # 30 mm → 60 mm diameter

    pts, faces = cube_sphere_grid(n=N, radius=radius)
    nodes, inverse = merge_vertices(pts)
    faces_merged = inverse[faces]
    edges = edges_from_faces(nodes, faces_merged)

    print(f"triangulation: {len(nodes)} nodes, {len(faces_merged)} faces, {len(edges)} unique edges")

    # Printable lattice (thick rods + joints).
    lattice = build_frame_mesh(nodes, edges, rod_radius=ROD_MM, joint_radius=JOINT_MM)
    lattice_path = "output/lattice/sphere_surface_frame.stl"
    os.makedirs(os.path.dirname(lattice_path), exist_ok=True)
    lattice.export(lattice_path)
    register("lattice", lattice_path, "stl", f"Sphere surface frame — {len(edges)} rods")

    # Triangulation wireframe (thin, raw 1-skeleton).
    wire = build_frame_mesh(nodes, edges, rod_radius=WIRE_ROD_MM, joint_radius=WIRE_JOINT_MM)
    wire_path = "output/triangulation/sphere_wireframe.stl"
    os.makedirs(os.path.dirname(wire_path), exist_ok=True)
    wire.export(wire_path)
    register("triangulation", wire_path, "stl", f"Sphere triangulation — {len(nodes)} nodes")

    print(f"wrote {lattice_path} ({len(lattice.faces):,} tris)")
    print(f"wrote {wire_path} ({len(wire.faces):,} tris)")


if __name__ == "__main__":
    main()
