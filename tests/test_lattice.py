"""Tests for lattice construction (Phase 2)."""

import numpy as np

from printable_surfaces.lattice import (
    build_frame_mesh,
    clip_segment_to_sphere,
    diamond_lattice,
    edges_from_faces,
    merge_vertices,
)


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


def test_clip_segment_inside_returns_full_segment():
    c = np.array([0.0, 0.0, 0.0])
    q0, q1 = clip_segment_to_sphere(np.array([-5.0, 0.0, 0.0]), np.array([5.0, 0.0, 0.0]), c, 10.0)
    assert np.allclose(q0, [-5.0, 0.0, 0.0])
    assert np.allclose(q1, [5.0, 0.0, 0.0])


def test_clip_segment_crossing_sphere():
    c = np.array([0.0, 0.0, 0.0])
    q0, q1 = clip_segment_to_sphere(np.array([-20.0, 0.0, 0.0]), np.array([20.0, 0.0, 0.0]), c, 10.0)
    assert np.allclose(q0, [-10.0, 0.0, 0.0])
    assert np.allclose(q1, [10.0, 0.0, 0.0])


def test_clip_segment_fully_outside_is_none():
    c = np.array([0.0, 0.0, 0.0])
    assert clip_segment_to_sphere(np.array([20.0, 0.0, 0.0]), np.array([30.0, 0.0, 0.0]), c, 10.0) is None


def test_diamond_lattice_nodes_within_sphere():
    nodes, edges = diamond_lattice(radius=30.0, a=20.0)
    assert len(nodes) > 0 and len(edges) > 0
    assert (np.linalg.norm(nodes, axis=1) <= 30.0 + 1e-6).all()
    assert edges.max() < len(nodes)
