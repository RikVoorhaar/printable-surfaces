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


def _rod_color(d: np.ndarray) -> np.ndarray:
    """RGBA (uint8) colour for a rod, by its angle from horizontal.

    green = steep (≥45°, printable) · orange = horizontal (bridge) · red = shallow.
    """
    length = float(np.linalg.norm(d))
    if length < 1e-9:
        return np.array([154, 163, 178, 255], dtype=np.uint8)  # joint grey
    s = abs(float(d[2])) / length  # sin(elevation): 0 = horizontal, 1 = vertical
    if s < 0.05:
        return np.array([232, 138, 46, 255], dtype=np.uint8)   # horizontal bridge
    if s < 0.7071:
        return np.array([224, 90, 90, 255], dtype=np.uint8)    # shallow (unprintable)
    return np.array([76, 184, 106, 255], dtype=np.uint8)       # steep (printable)


def build_frame_mesh(
    nodes: np.ndarray,
    edges: np.ndarray,
    rod_radius: float,
    joint_radius: float,
    rod_sections: int = 12,
    joint_count: tuple[int, int] = (6, 6),
    color_by_direction: bool = False,
) -> trimesh.Trimesh:
    """Thicken a node+edge frame into a single mesh.

    Rods become cylinders, nodes become spheres, and the parts are concatenated into one
    triangle soup. (This is intentionally *not* boolean-unioned; overlapping joints are
    fine for viewing and slicing, and a true watertight union is a later refinement.)

    If ``color_by_direction`` is True, rods are vertex-coloured by their angle from
    horizontal (green=steep, orange=horizontal, red=shallow) and joints grey — bake this
    into a PLY export for the viewer.
    """
    joint_color = np.array([154, 163, 178, 255], dtype=np.uint8)

    def _paint(mesh: trimesh.Trimesh, color: np.ndarray) -> trimesh.Trimesh:
        if color_by_direction:
            mesh.visual.vertex_colors = np.tile(color, (len(mesh.vertices), 1))
        return mesh

    parts: list[trimesh.Trimesh] = []
    for a, b in edges:
        seg = np.asarray([nodes[a], nodes[b]], dtype=float)
        if np.linalg.norm(seg[1] - seg[0]) < 1e-9:
            continue  # skip degenerate (zero-length) edges
        cyl = trimesh.creation.cylinder(radius=rod_radius, segment=seg, sections=rod_sections)
        parts.append(_paint(cyl, _rod_color(seg[1] - seg[0])))
    for p in np.asarray(nodes, dtype=float):
        s = trimesh.creation.uv_sphere(radius=joint_radius, count=joint_count)
        s.apply_translation(p)
        parts.append(_paint(s, joint_color))
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


def boundary_mask(nodes: np.ndarray, radius: float, tol: float = 1e-3) -> np.ndarray:
    """Boolean mask: True for nodes lying on (or within ``tol`` of) the sphere surface."""
    return np.linalg.norm(np.asarray(nodes, dtype=float), axis=1) >= radius - tol


def surface_edges(boundary_points: np.ndarray) -> np.ndarray:
    """Triangulate co-spherical boundary points into surface edges via convex hull.

    All points lie on the sphere, so the convex hull is the sphere's triangulation; its
    edges become the surface frame's rods. Returns ``(E, 2)`` indices into ``boundary_points``.
    """
    from scipy.spatial import ConvexHull

    if len(boundary_points) < 4:
        return np.zeros((0, 2), dtype=int)
    hull = ConvexHull(boundary_points)
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for tri in hull.simplices:
        for e in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = (min(e), max(e))
            if key not in seen:
                seen.add(key)
                out.append(key)
    return np.asarray(sorted(out), dtype=int)


def complete_surface(
    radius: float,
    nodes: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Add a surface frame that shares nodes with an interior lattice.

    Given interior ``nodes``/``edges`` (already clipped to the sphere, with boundary nodes
    on the surface), triangulate the boundary nodes (convex hull) into surface edges.
    Returns ``(nodes, edges, surface_edges, boundary_mask)``.
    """
    nodes = np.asarray(nodes, dtype=float)
    b = boundary_mask(nodes, radius)
    b_idx = np.where(b)[0]
    if len(b_idx) >= 4:
        surface_edges_global = b_idx[surface_edges(nodes[b_idx])]
    else:
        surface_edges_global = np.zeros((0, 2), dtype=int)
    return nodes, np.asarray(edges, dtype=int), surface_edges_global, b


def unified_cage(
    radius: float,
    a: float,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Diamond interior + a surface frame sharing its boundary nodes (see :func:`complete_surface`)."""
    nodes, edges = diamond_lattice(radius, a, center=center)
    return complete_surface(radius, nodes, edges)


def jittered_diamond(
    radius: float,
    a: float,
    jitter: float,
    seed: int = 0,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """A diamond lattice with interior atoms randomly displaced (breaks cubic regularity).

    Boundary nodes stay on the sphere (so the surface frame still lines up); interior
    atoms are jittered by up to ``jitter`` (clamped to stay inside). Keeps degree-4
    topology but gives an organic, non-cubic look.
    """
    nodes, edges = diamond_lattice(radius, a, center=center)
    rng = np.random.default_rng(seed)
    norms = np.linalg.norm(nodes - np.asarray(center), axis=1)
    interior = norms < radius - 1e-3

    # Random direction, random magnitude, clamped so jittered atoms stay inside.
    directions = rng.normal(size=(len(nodes), 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    max_shift = np.minimum(jitter, np.maximum(0.0, 0.9 * (radius - norms)))
    shifts = directions * (rng.uniform(0.0, 1.0, size=(len(nodes), 1)) * max_shift[:, None])
    shifts[~interior] = 0.0
    return nodes + shifts, edges


def _sample_ball(rng: np.random.Generator, n: int, radius: float) -> np.ndarray:
    """``n`` points uniformly distributed in a ball of radius ``radius`` (up to 98% of it)."""
    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    r = radius * 0.98 * np.cbrt(rng.uniform(0.0, 1.0, size=n))
    return directions * r[:, None]


def random_lattice(
    radius: float,
    n_surface: int,
    n_interior: int,
    k: int = 4,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """An organic random lattice: a sparse k-nearest-neighbour web of scattered points.

    Scatters points uniformly on the sphere surface (these become the shared boundary
    nodes) and in the interior, then connects each point to its ``k`` nearest neighbours.
    Degree stays near ``k``, so the result is sparse and non-cubic. Returns ``(nodes, edges)``.
    """
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)

    surf = rng.normal(size=(n_surface, 3))
    surf = surf / np.linalg.norm(surf, axis=1, keepdims=True) * radius
    intr = _sample_ball(rng, n_interior, radius)

    pts = np.vstack([surf, intr])
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=k + 1)  # k+1: the closest neighbour is the point itself

    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for i in range(len(pts)):
        for j in idx[i, 1:]:
            key = (min(i, j), max(i, j))
            if key not in seen:
                seen.add(key)
                out.append(key)
    return pts, np.asarray(out, dtype=int)
