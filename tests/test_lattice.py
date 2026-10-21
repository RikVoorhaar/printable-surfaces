"""Tests for lattice construction (Phase 2)."""

import numpy as np

from printable_surfaces.lattice import (
    boundary_mask,
    build_frame_mesh,
    clip_segment_to_sphere,
    diamond_lattice,
    edges_from_faces,
    jittered_diamond,
    merge_vertices,
    random_lattice,
    unified_cage,
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


def test_build_frame_mesh_color_by_direction():
    # horizontal rod (should be orange) + vertical rod (should be green) + joints (grey)
    nodes = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 0.0, 10.0]])
    edges = np.array([[0, 1], [0, 2]])
    mesh = build_frame_mesh(nodes, edges, rod_radius=1.0, joint_radius=1.5, color_by_direction=True)
    vc = mesh.visual.vertex_colors
    assert vc is not None
    assert vc.shape == (len(mesh.vertices), 4)
    colors = {tuple(c) for c in vc[:, :3]}
    # green (steep) and orange (horizontal) must both be present
    assert (76, 184, 106) in colors
    assert (232, 138, 46) in colors


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


def test_unified_cage_shares_boundary_nodes():
    nodes, diamond_edges, surface_edges, bmask = unified_cage(radius=30.0, a=20.0)
    b_idx = np.where(bmask)[0]
    assert len(b_idx) > 0

    def incident(edges, i):
        return any(i == a or i == b for a, b in edges)

    for i in b_idx:
        # Every surface node must be the endpoint of both an interior (diamond) rod
        # and a surface rod — i.e. interior and surface share nodes.
        assert incident(diamond_edges, i), f"boundary node {i} missing interior rod"
        assert incident(surface_edges, i), f"boundary node {i} missing surface rod"


def test_jittered_diamond_keeps_boundary_on_sphere():
    nodes, edges = jittered_diamond(radius=30.0, a=20.0, jitter=6.0, seed=1)
    b = boundary_mask(nodes, 30.0)
    # boundary nodes stay exactly on the sphere (they are not jittered) ...
    assert np.allclose(np.linalg.norm(nodes[b], axis=1), 30.0, atol=1e-3)
    # ... while interior nodes are displaced but remain inside.
    assert (np.linalg.norm(nodes[~b], axis=1) < 30.0).all()
    assert len(edges) > 0


def test_random_lattice_is_deterministic_and_in_sphere():
    pts1, e1 = random_lattice(30.0, 40, 60, k=4, seed=7)
    pts2, e2 = random_lattice(30.0, 40, 60, k=4, seed=7)
    assert np.allclose(pts1, pts2)
    assert np.array_equal(e1, e2)
    assert (np.linalg.norm(pts1, axis=1) <= 30.0 + 1e-6).all()
    assert e1.max() < len(pts1)
    assert len(e1) > 0
