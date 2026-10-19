"""Tests for lattice construction (Phase 2)."""

import numpy as np

from printable_surfaces.lattice import build_frame_mesh, edges_from_faces, merge_vertices


def test_merge_vertices_dedupes_and_remaps():
    pts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    merged, inverse = merge_vertices(pts)
    assert merged.shape == (2, 3)
    assert inverse.tolist() == [0, 0, 1]


def test_edges_from_faces_are_unique_and_undirected():
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    faces = np.array([[0, 1, 2], [0, 1, 3]])
    edges = edges_from_faces(vertices, faces)
    # 5 unique edges total (tetrahedron missing edge 2-3)
    assert edges.shape == (5, 2)
    # undirected: (0,1) present, not (1,0) as a separate edge
    pairs = {tuple(sorted(e)) for e in edges}
    assert (0, 1) in pairs


def test_build_frame_mesh_is_finite_and_nonempty():
    nodes = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    edges = np.array([[0, 1]])
    mesh = build_frame_mesh(nodes, edges, rod_radius=1.0, joint_radius=1.5)
    assert len(mesh.faces) > 0
    assert np.isfinite(mesh.vertices).all()
