"""Figure lattices: a sphere's 1-skeleton, coloured by how steep each rod is.

This is the mesh behind the §1 figure — the claim that a printed surface cage fails for an
*angular* reason, not a meshing one. Every rod is coloured by its own angle above
horizontal, so coarsening or refining the mesh changes the colour pattern's resolution but
never gets rid of it.

The rods follow the printed set (Ø1.8 mm rod, Ø2.7 mm joint); the sphere is 100 mm across.
Run::

    uv run python -m printable_surfaces.surface_figure
"""

from __future__ import annotations

import os

import numpy as np

from .lattice import build_frame_mesh, edges_from_faces, merge_vertices
from .parametrizations import cube_sphere_grid

OBJECT_MM = 100.0  # sphere diameter
ROD_R = 0.9  # Ø1.8 mm, the T5 printed set
JOINT_R = 1.35  # Ø2.7 mm, keeps the 1.5 rod:joint ratio
SIN48 = float(np.sin(np.radians(48.0)))
SIN5 = float(np.sin(np.radians(5.0)))
BRIDGE_MAX = 10.0


def sphere_frame(n: int, diameter: float = OBJECT_MM) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and rods of the cube-sphere's 1-skeleton at ``n`` subdivisions per face."""
    points, faces = cube_sphere_grid(n=n, radius=diameter / 2.0)
    nodes, inverse = merge_vertices(points)
    return nodes, edges_from_faces(nodes, inverse[faces])


def rod_elevations(nodes: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Each rod's angle above horizontal, in degrees."""
    d = np.asarray(nodes)[edges[:, 1]] - np.asarray(nodes)[edges[:, 0]]
    length = np.linalg.norm(d, axis=1)
    return np.degrees(np.arcsin(np.clip(np.abs(d[:, 2]) / length, 0.0, 1.0)))


def printability(nodes: np.ndarray, edges: np.ndarray) -> dict[str, float]:
    """Share of rods the printer could actually hold, using the project's rule.

    Printable = steeper than 48°, or a horizontal bridge no longer than 10 mm, or lying on
    the build plate. A fine mesh creates short horizontal rods that qualify as bridges,
    which is why the unprintable share drops from ~78 % at n=5–6 to ~63 % at n≥12 — even
    though the *bands* of shallow rods are just as much there.
    """
    d = np.asarray(nodes)[edges[:, 1]] - np.asarray(nodes)[edges[:, 0]]
    length = np.linalg.norm(d, axis=1)
    slope = np.abs(d[:, 2]) / length
    z_min = float(np.asarray(nodes)[:, 2].min())
    on_plate = (np.asarray(nodes)[edges[:, 0], 2] < z_min + 0.5) & (
        np.asarray(nodes)[edges[:, 1], 2] < z_min + 0.5
    )
    printable = (slope >= SIN48) | ((slope < SIN5) & (length <= BRIDGE_MAX)) | on_plate
    return {
        "rods": int(len(edges)),
        "unprintable": float(100.0 * (~printable).mean()),
        "mean_length": float(length.mean()),
    }


def build(
    n: int,
    *,
    colors: bool = True,
    rod_sections: int = 8,
    joint_count: tuple[int, int] = (6, 6),
):
    """The thickened, coloured frame mesh for ``n``.

    Eight-sided rods keep the GLB near 370 kB (the Platonic solids are ~1 MB); at 1.8 mm
    across, the facets are below what the viewer resolves.
    """
    nodes, edges = sphere_frame(n)
    mesh = build_frame_mesh(
        nodes,
        edges,
        rod_radius=ROD_R,
        joint_radius=JOINT_R,
        rod_sections=rod_sections,
        joint_count=joint_count,
        color_by="elevation" if colors else None,
    )
    return nodes, edges, mesh


def main() -> None:
    os.makedirs("output/lattice", exist_ok=True)
    for n in (5, 6, 12, 16, 24):
        nodes, edges = sphere_frame(n)
        stats = printability(nodes, edges)
        print(
            f"n={n:2d}  nodes={len(nodes):5d}  rods={stats['rods']:6d}  "
            f"unprintable={stats['unprintable']:5.1f}%  mean rod={stats['mean_length']:5.2f} mm"
        )

    nodes, edges, mesh = build(6)
    # PLY keeps the per-vertex colours; STL does not, so the colour source is the PLY.
    for suffix in ("ply", "stl"):
        path = f"output/lattice/sphere_surface_frame_n06.{suffix}"
        mesh.export(path)
        print(f"wrote {path} ({os.path.getsize(path) / 1e6:.2f} MB, {len(mesh.faces)} triangles)")


if __name__ == "__main__":
    main()
