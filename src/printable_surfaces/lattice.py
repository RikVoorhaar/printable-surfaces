"""Rod + node lattice construction.

Turns a triangulated surface (or any node/edge frame) into a printable mesh by
thickening edges into cylinders and nodes into spheres, and generates interior
crystal lattices (diamond) clipped to a shape.
"""

from __future__ import annotations

import numpy as np
import trimesh


def merge_vertices(vertices: np.ndarray, decimals: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Merge near-coincident vertices.

    Returns ``(merged_vertices, inverse)`` where ``inverse`` maps each original vertex
    index to its merged index (used to remap faces). Chart boundaries in the cube-sphere
    share exact sample points, so this closes the "cracks" between charts.
    """
    _, idx, inverse = np.unique(
        np.round(np.asarray(vertices, dtype=float), decimals),
        axis=0,
        return_index=True,
        return_inverse=True,
    )
    return np.asarray(vertices)[idx], inverse


def edges_from_faces(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Unique undirected edges ``(E, 2)`` of a triangle mesh (the 1-skeleton)."""
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return np.asarray(mesh.edges_unique, dtype=int)


def build_frame_mesh(
    nodes: np.ndarray,
    edges: np.ndarray,
    rod_radius: float,
    joint_radius: float,
    rod_sections: int = 12,
    joint_count: tuple[int, int] = (6, 6),
) -> trimesh.Trimesh:
    """Thicken a node+edge frame into a single mesh.

    Rods become cylinders, nodes become spheres, and the parts are concatenated into one
    triangle soup. (This is intentionally *not* boolean-unioned; overlapping joints are
    fine for viewing and slicing, and a true watertight union is a later refinement.)
    """
    parts: list[trimesh.Trimesh] = []
    for a, b in edges:
        seg = np.asarray([nodes[a], nodes[b]], dtype=float)
        if np.linalg.norm(seg[1] - seg[0]) < 1e-9:
            continue  # skip degenerate (zero-length) edges
        parts.append(trimesh.creation.cylinder(radius=rod_radius, segment=seg, sections=rod_sections))
    for p in np.asarray(nodes, dtype=float):
        s = trimesh.creation.uv_sphere(radius=joint_radius, count=joint_count)
        s.apply_translation(p)
        parts.append(s)
    return trimesh.util.concatenate(parts)


def surface_frame_mesh(radius: float, n: int, rod_radius: float, joint_radius: float) -> trimesh.Trimesh:
    """The surface frame of a cube-sphere: triangulate, take the 1-skeleton, thicken."""
    from .parametrizations import cube_sphere_grid

    pts, faces = cube_sphere_grid(n=n, radius=radius)
    nodes, inverse = merge_vertices(pts)
    edges = edges_from_faces(nodes, inverse[faces])
    return build_frame_mesh(nodes, edges, rod_radius=rod_radius, joint_radius=joint_radius)


def clip_segment_to_sphere(
    p0: np.ndarray, p1: np.ndarray, center: np.ndarray, radius: float
) -> tuple[np.ndarray, np.ndarray] | None:
    """Clip the segment ``p0→p1`` to the ball ``|x - center| ≤ radius``.

    Returns the clipped endpoints ``(q0, q1)`` (the portion inside the ball), or ``None``
    if the segment does not intersect the ball's interior.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    center = np.asarray(center, dtype=float)
    d = p1 - p0
    f = p0 - center
    a = float(d @ d)
    if a == 0.0:
        return None
    b = 2.0 * float(f @ d)
    c = float(f @ f) - radius * radius
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return None
    s = np.sqrt(disc)
    t0 = (-b - s) / (2.0 * a)
    t1 = (-b + s) / (2.0 * a)
    lo = max(0.0, t0)
    hi = min(1.0, t1)
    if lo > hi:
        return None
    q0 = p0 + lo * d
    q1 = p0 + hi * d
    if np.allclose(q0, q1, atol=1e-9):
        return None
    return q0, q1


def diamond_lattice(
    radius: float,
    a: float,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a **diamond-cubic** lattice clipped to a sphere.

    The diamond lattice is 4-regular (each interior atom bonds to 4 neighbours at
    tetrahedral angles ~109.5°), matching the "degree-4 interior" hint. Rods crossing the
    sphere boundary are truncated at the surface, so the cage fills the volume and its
    rods terminate on the shell.

    ``a`` is the cubic unit-cell edge length in the same units as ``radius`` (smaller
    ``a`` = denser). Returns ``(nodes, edges)``.
    """
    center = np.asarray(center, dtype=float)

    # FCC sublattice A basis + the B sublattice offset, and the 4 tetrahedral bond
    # directions (each of length a·√3/4).
    A_offsets = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]]) * a
    bond_dirs = np.array(
        [[0.25, 0.25, 0.25], [0.25, -0.25, -0.25], [-0.25, 0.25, -0.25], [-0.25, -0.25, 0.25]]
    ) * a

    n = int(np.ceil((radius + a) / a)) + 1

    node_map: dict[tuple[float, float, float], int] = {}
    nodes: list[np.ndarray] = []
    edges: list[tuple[int, int]] = []

    def node_index(p: np.ndarray) -> int:
        key = (round(float(p[0]), 6), round(float(p[1]), 6), round(float(p[2]), 6))
        if key not in node_map:
            node_map[key] = len(nodes)
            nodes.append(p)
        return node_map[key]

    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            for k in range(-n, n + 1):
                base = np.array([i, j, k], dtype=float) * a + center
                for A in A_offsets:
                    p0 = base + A
                    for d in bond_dirs:
                        seg = clip_segment_to_sphere(p0, p0 + d, center, radius)
                        if seg is None:
                            continue
                        ia = node_index(seg[0])
                        ib = node_index(seg[1])
                        if ia != ib:
                            edges.append((ia, ib))

    if not nodes:
        return np.zeros((0, 3)), np.zeros((0, 2), dtype=int)
    return np.asarray(nodes, dtype=float), np.asarray(edges, dtype=int)
