"""Tests for Platonic-solid geometry (Phase 13)."""

import numpy as np

from printable_surfaces.platonic import cube_rhombohedron, face_parallelogram, inside_mask


def test_cube_rhombohedron_shape():
    verts, edges, faces = cube_rhombohedron(50.0, 1.69)
    assert verts.shape == (8, 3)
    assert edges.shape == (12, 2)
    assert faces.shape == (6, 4)


def test_cube_rhombohedron_edges_steep():
    verts, edges, _ = cube_rhombohedron(50.0, 1.69)
    d = verts[edges[:, 1]] - verts[edges[:, 0]]
    sin_ang = np.abs(d[:, 2]) / np.linalg.norm(d, axis=1)
    # every edge is >= 45° from horizontal (self-supporting)
    assert (sin_ang >= np.sin(np.deg2rad(45.0))).all()


def test_face_parallelogram_spans_face():
    verts, _, faces = cube_rhombohedron(50.0, 1.69)
    for face in faces:
        corner, e1, e2 = face_parallelogram(verts, face)
        # corner + e1 + e2 reproduces three distinct face vertices
        pts = {tuple(np.round(verts[i], 6)) for i in face}
        assert tuple(np.round(corner, 6)) in pts
        assert tuple(np.round(corner + e1, 6)) in pts
        assert tuple(np.round(corner + e2, 6)) in pts
