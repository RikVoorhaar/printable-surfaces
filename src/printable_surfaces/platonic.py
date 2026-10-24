"""Platonic-solid geometry, optionally stretched into printable polytopes.

Per the Perplexity report, a cube with its body diagonal vertical, stretched along Z,
becomes a rhombohedron whose edges are all at a printable elevation (≥45°). The
"edges-suffice" theorem means getting the edges right is enough.

The other four solids (from docs/Support-Free Rod Lattices for the Platonic Solids…):
  - tetrahedron:   face down, no stretch
  - octahedron:    vertex down, vertical stretch ×1.19
  - icosahedron:   vertex down, tip pyramids lifted (tip edges → 50°)
  - dodecahedron:  face down, vertical stretch ×1.93 (relaxed, flat bridged top)
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

PHI = (1.0 + np.sqrt(5.0)) / 2.0


def _convex_faces(vertices: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    """Edges and (merged, cyclic) polygon faces of a convex polytope from its vertices.

    ConvexHull gives triangles; coplanar triangles are merged into polygons (the
    dodecahedron's pentagons), and each face's vertices are ordered cyclically. Returns
    ``(edges (E,2), faces: list of int arrays)``.
    """
    verts = np.asarray(vertices, dtype=float)
    hull = ConvexHull(verts)
    tris = hull.simplices
    centroid = verts.mean(axis=0)

    normals = np.zeros((len(tris), 3))
    offsets = np.zeros(len(tris))
    for i, tri in enumerate(tris):
        p0, p1, p2 = verts[tri]
        n = np.cross(p1 - p0, p2 - p0)
        n = n / np.linalg.norm(n)
        if n @ (p0 - centroid) < 0.0:
            n = -n
        normals[i] = n
        offsets[i] = float(n @ p0)

    faces: list[np.ndarray] = []
    used = np.zeros(len(tris), dtype=bool)
    for i in range(len(tris)):
        if used[i]:
            continue
        grp = np.where((normals @ normals[i] > 1.0 - 1e-6) & (np.abs(offsets - offsets[i]) < 1e-6))[0]
        used[grp] = True
        vidx = sorted(set(tris[grp].ravel()))
        fpts = verts[vidx]
        c = fpts.mean(axis=0)
        u = fpts[0] - c
        u = u / np.linalg.norm(u)
        v = np.cross(normals[i], u)
        v = v / np.linalg.norm(v)
        ang = np.arctan2([(p - c) @ v for p in fpts], [(p - c) @ u for p in fpts])
        faces.append(np.asarray(vidx, dtype=int)[np.argsort(ang)])

    eset = set()
    for f in faces:
        m = len(f)
        for k in range(m):
            a, b = f[k], f[(k + 1) % m]
            eset.add((min(a, b), max(a, b)))
    edges = np.asarray(sorted(eset), dtype=int)
    return edges, faces


def tetrahedron(circumradius: float = 50.0):
    """Regular tetrahedron, face down (apex up), no stretch. 3 steep edges at 54.7°."""
    v = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)
    v *= circumradius / np.sqrt(3.0)
    rot = Rotation.align_vectors([[0.0, 0.0, 1.0]], [[1.0, 1.0, 1.0]])[0].as_matrix()
    v = v @ rot.T
    edges, faces = _convex_faces(v)
    return v, edges, faces


def octahedron(circumradius: float = 50.0):
    """Regular octahedron, vertex down, vertical stretch ×1.19 (8 steep edges → 50°)."""
    v = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)
    v *= circumradius
    v[:, 2] *= 1.19
    edges, faces = _convex_faces(v)
    return v, edges, faces


def icosahedron(circumradius: float = 50.0):
    """Regular icosahedron, vertex down, tip pyramids lifted so tip edges reach 50°."""
    v = np.array([
        [0, 1, PHI], [0, -1, PHI], [0, 1, -PHI], [0, -1, -PHI],
        [1, PHI, 0], [1, -PHI, 0], [-1, PHI, 0], [-1, -PHI, 0],
        [PHI, 0, 1], [PHI, 0, -1], [-PHI, 0, 1], [-PHI, 0, -1],
    ], dtype=float)
    v *= circumradius / np.sqrt(1.0 + PHI * PHI)
    rot = Rotation.align_vectors([[0.0, 0.0, 1.0]], [[0.0, 1.0, PHI]])[0].as_matrix()
    v = v @ rot.T

    # lift the top tip (and lower the bottom) so the 5 tip edges hit 50°
    itop = int(np.argmax(v[:, 2]))
    ibot = int(np.argmin(v[:, 2]))
    dist = np.linalg.norm(v - v[itop], axis=1)
    nb = np.argsort(dist)[1:6]                       # the 5 upper-ring neighbours
    h_u = float(v[nb, 2].mean())
    r_u = float(np.linalg.norm(v[nb, :2], axis=1).mean())
    z_t = h_u + r_u * np.tan(np.radians(50.0))
    v[itop, 2] = z_t
    v[ibot, 2] = -z_t
    edges, faces = _convex_faces(v)
    return v, edges, faces


def dodecahedron(circumradius: float = 50.0):
    """Regular dodecahedron, face down, vertical stretch ×1.93 (relaxed: flat bridged top)."""
    iphi = 1.0 / PHI
    v = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, iphi, PHI], [0, iphi, -PHI], [0, -iphi, PHI], [0, -iphi, -PHI],
        [iphi, PHI, 0], [iphi, -PHI, 0], [-iphi, PHI, 0], [-iphi, -PHI, 0],
        [PHI, 0, iphi], [PHI, 0, -iphi], [-PHI, 0, iphi], [-PHI, 0, -iphi],
    ], dtype=float)
    v *= circumradius / np.sqrt(3.0)
    edges, faces = _convex_faces(v)

    # face down: outward normal of face 0 -> -Z
    p = v[faces[0]]
    n = np.cross(p[1] - p[0], p[2] - p[0])
    n = n / np.linalg.norm(n)
    if n @ (p[0] - v.mean(axis=0)) < 0.0:
        n = -n
    rot = Rotation.align_vectors([[0.0, 0.0, -1.0]], [n])[0].as_matrix()
    v = v @ rot.T
    v[:, 2] *= 1.93
    return v, edges, faces


def cube_rotation() -> np.ndarray:
    """3×3 rotation mapping the body diagonal (1,1,1) onto +Z."""
    return Rotation.align_vectors([[0.0, 0.0, 1.0]], [[1.0, 1.0, 1.0] / np.sqrt(3.0)])[0].as_matrix()


def cube_rhombohedron(circumradius: float = 50.0, stretch: float = 1.69):
    """Cube (circumradius ``circumradius``) with a body diagonal vertical, z×``stretch``.

    Returns ``(vertices (8,3), edges (12,2), faces (6,4))``. Vertices are ordered by
    ``(x, y, z)`` loops; edges connect pairs differing in exactly one coordinate; each
    face is the 4 vertices sharing one coordinate value.
    """
    half = circumradius / np.sqrt(3.0)
    verts = np.array([[x, y, z] for x in (-half, half) for y in (-half, half) for z in (-half, half)], dtype=float)

    edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            if int(np.sum(verts[i] != verts[j])) == 1:
                edges.append((i, j))
    edges = np.asarray(edges, dtype=int)

    faces = []
    for axis in range(3):
        for sign in (-half, half):
            faces.append([i for i in range(8) if verts[i][axis] == sign])
    faces = np.asarray(faces, dtype=int)

    # Body diagonal (1,1,1) -> vertical, then stretch z.
    rot = cube_rotation()
    verts = rot @ verts.T
    verts = verts.T
    verts[:, 2] *= stretch

    return verts, edges, faces


def face_parallelogram(vertices: np.ndarray, face: np.ndarray):
    """Return ``(corner, e1, e2)`` — a corner and two edge vectors of a rhombus face."""
    pts = vertices[face]
    v0 = pts[0]
    vecs = pts[1:] - v0
    order = np.argsort(np.linalg.norm(vecs, axis=1))
    return v0, vecs[order[0]], vecs[order[1]]


def face_planes(vertices: np.ndarray, faces: np.ndarray):
    """Outward normals and face centroids for each face (convex polytope)."""
    centroid = vertices.mean(axis=0)
    normals = []
    centers = []
    for face in faces:
        corner, e1, e2 = face_parallelogram(vertices, face)
        n = np.cross(e1, e2)
        n /= np.linalg.norm(n)
        c = vertices[face].mean(axis=0)
        if np.dot(n, c - centroid) < 0:  # orient outward
            n = -n
        normals.append(n)
        centers.append(c)
    return np.asarray(normals), np.asarray(centers)


def inside_mask(points: np.ndarray, normals: np.ndarray, centers: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Boolean: True for points on the inner side of every face (i.e. inside the polytope)."""
    inside = np.ones(len(points), dtype=bool)
    for n, c in zip(normals, centers):
        inside &= (points - c) @ n <= eps
    return inside


# ----------------------------------------------------------- polygon face geometry
# Used by the lattice face-filler so it works for *any* convex face (triangle, rhombus,
# pentagon), not just the cube's parallelograms.
def face_geometry(vertices: np.ndarray, face: np.ndarray) -> dict:
    """2-D parametrization of a convex polygon face.

    Returns ``{origin, u, v, normal, poly}`` where ``poly`` is the face's vertices in
    (u, v) plane coordinates (CCW). ``origin`` is a point on the face plane; ``u``/``v``
    are orthonormal in-plane axes. Input vertices need not be in cyclic order.
    """
    pts = np.asarray(vertices[face], dtype=float)
    c = pts.mean(axis=0)
    n = np.cross(pts[1] - pts[0], pts[2] - pts[0])
    n = n / np.linalg.norm(n)
    u = pts[1] - pts[0]
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)
    v = v / np.linalg.norm(v)
    # cyclic sort around the centroid (robust to non-cyclic input)
    ang = np.arctan2([(p - c) @ v for p in pts], [(p - c) @ u for p in pts])
    pts = pts[np.argsort(ang)]
    origin = pts[0].copy()
    u = pts[1] - origin
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)
    v = v / np.linalg.norm(v)
    poly = np.array([[(p - origin) @ u, (p - origin) @ v] for p in pts])
    if _signed_area(poly) < 0.0:
        poly = poly[::-1]  # CCW
    return {"origin": origin, "u": u, "v": v, "normal": n, "poly": poly}


def _signed_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def shrink_polygon(poly: np.ndarray, d: float) -> np.ndarray:
    """Inward-offset a CCW convex polygon by distance ``d`` (half-plane intersection)."""
    poly = np.asarray(poly, dtype=float)
    n = len(poly)
    if d <= 0.0:
        return poly
    normals = np.zeros((n, 2))
    offsets = np.zeros(n)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        e = b - a
        elen = np.linalg.norm(e)
        nrm = np.array([-e[1], e[0]]) / elen  # CCW inward normal
        normals[i] = nrm
        offsets[i] = float(nrm @ a) + d
    out = np.zeros((n, 2))
    for i in range(n):
        A = np.array([normals[i - 1], normals[i]])
        b = np.array([offsets[i - 1], offsets[i]])
        out[i] = np.linalg.solve(A, b)
    return out


def sample_in_polygon(poly: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Uniform random point inside a convex polygon (fan triangulation, area-weighted)."""
    poly = np.asarray(poly, dtype=float)
    o = poly[0]
    m = len(poly)
    areas = []
    for i in range(1, m - 1):
        a = poly[i] - o
        b = poly[i + 1] - o
        areas.append(0.5 * abs(a[0] * b[1] - a[1] * b[0]))
    areas = np.asarray(areas)
    if areas.sum() <= 0.0:
        return o.copy()
    r = rng.uniform(0.0, areas.sum())
    idx = min(int(np.searchsorted(np.cumsum(areas), r)), m - 3)
    a = poly[idx + 1] - o
    b = poly[idx + 2] - o
    s = rng.uniform()
    t = rng.uniform()
    if s + t > 1.0:
        s, t = 1.0 - s, 1.0 - t
    return o + s * a + t * b


def clamp_to_polygon(poly: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Clamp a 2-D point to a convex polygon (nearest boundary point if outside)."""
    poly = np.asarray(poly, dtype=float)
    p = np.asarray(p, dtype=float)
    n = len(poly)
    inside = True
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        e = b - a
        nrm = np.array([-e[1], e[0]])  # CCW inward normal
        if float(nrm @ (p - a)) < 0.0:
            inside = False
            break
    if inside:
        return p
    best = None
    bestd = np.inf
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        e = b - a
        denom = float(e @ e)
        t = 0.0 if denom < 1e-12 else float(np.clip((p - a) @ e / denom, 0.0, 1.0))
        q = a + t * e
        dd = float(np.linalg.norm(p - q))
        if dd < bestd:
            bestd = dd
            best = q
    return best
