"""Platonic-solid geometry, optionally stretched into printable polytopes.

Per the Perplexity report, a cube with its body diagonal vertical, stretched along Z,
becomes a rhombohedron whose edges are all at a printable elevation (≥45°). The
"edges-suffice" theorem means getting the edges right is enough.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


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
