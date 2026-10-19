"""Phase 4 — interior diamond lattice + full cage, for the selected densities.

Generates a 4-regular diamond lattice clipped to the sphere (the "inside"), combines it
with the surface frame into a full printable cage, and emits STLs + registers them.

Run:  uv run python -m printable_surfaces.phase4
"""

from __future__ import annotations

import os

import trimesh

from .artifacts import register
from .lattice import build_frame_mesh, diamond_lattice, edges_from_faces, merge_vertices
from .parametrizations import cube_sphere_grid

OBJECT_MM = 100.0
ROD_MM = 1.2
JOINT_MM = 1.8
# (label, surface resolution n, diamond unit-cell size a in mm)
VARIANTS = [
    ("13%", 5, 30.0),
    ("21%", 6, 24.0),
]


def main() -> None:
    radius = OBJECT_MM / 2.0

    for label, n, a in VARIANTS:
        tag = label.replace("%", "pct")

        # Surface frame (outer shell).
        pts, faces = cube_sphere_grid(n=n, radius=radius)
        nodes, inverse = merge_vertices(pts)
        surface_edges = edges_from_faces(nodes, inverse[faces])
        surface = build_frame_mesh(nodes, surface_edges, rod_radius=ROD_MM, joint_radius=JOINT_MM)

        # Interior diamond lattice (the "inside"), clipped to the sphere.
        inodes, iedges = diamond_lattice(radius, a)
        interior = build_frame_mesh(inodes, iedges, rod_radius=ROD_MM, joint_radius=JOINT_MM)

        combined = trimesh.util.concatenate([surface, interior])

        interior_path = f"output/lattice/sphere_interior_diamond_{tag}.stl"
        os.makedirs(os.path.dirname(interior_path), exist_ok=True)
        interior.export(interior_path)
        register("lattice", interior_path, "stl", f"Interior (diamond) · {label} ({len(iedges)} rods)")

        combined_path = f"output/lattice/sphere_full_cage_{tag}.stl"
        os.makedirs(os.path.dirname(combined_path), exist_ok=True)
        combined.export(combined_path)
        register("lattice", combined_path, "stl", f"Full cage · {label} ({len(surface_edges)} surf + {len(iedges)} int rods)")

        print(
            f"{label}: surface {len(nodes)}n/{len(surface_edges)}e | "
            f"interior {len(inodes)}n/{len(iedges)}e | combined {len(combined.faces):,} tris"
        )


if __name__ == "__main__":
    main()
