"""Tests for force-directed relaxation (Phase 10)."""

import numpy as np

from printable_surfaces.relax import coulomb_lattice, knn_edges, relaxed_lattice, seed_vertices


def test_relaxed_lattice_is_deterministic():
    n1, e1 = relaxed_lattice(radius=40.0, target_length=14.0, k=4, seed=3, iterations=300)
    n2, e2 = relaxed_lattice(radius=40.0, target_length=14.0, k=4, seed=3, iterations=300)
    assert np.allclose(n1, n2)
    assert np.array_equal(e1, e2)


def test_relaxed_lattice_nodes_in_sphere_and_edges_converge():
    nodes, edges = relaxed_lattice(radius=50.0, target_length=18.0, k=4, seed=1, iterations=800)
    assert (np.linalg.norm(nodes, axis=1) <= 50.0 + 1e-3).all()
    lengths = np.linalg.norm(nodes[edges[:, 0]] - nodes[edges[:, 1]], axis=1)
    # Edges should cluster near the target length (roughly, since layout is noisy).
    assert abs(lengths.mean() - 18.0) < 3.0
    assert lengths.std() < 6.0


def test_seed_vertices_surface_first_and_on_sphere():
    pts = seed_vertices(radius=30.0, n_surface=50, n_interior=30, seed=0)
    assert np.allclose(np.linalg.norm(pts[:50], axis=1), 30.0)
    assert (np.linalg.norm(pts[50:], axis=1) <= 30.0).all()


def test_knn_edges_within_bounds_and_symmetric():
    pts = np.random.default_rng(0).normal(size=(30, 3))
    edges = knn_edges(pts, k=4)
    assert edges.max() < len(pts)
    assert (edges[:, 0] < edges[:, 1]).all()  # canonical order


def test_coulomb_lattice_uniform_and_in_sphere():
    from scipy.spatial import cKDTree

    nodes, edges = coulomb_lattice(radius=40.0, n_points=80, k=4, seed=1, iterations=1200)
    r = np.linalg.norm(nodes, axis=1)
    assert r.max() <= 40.0 + 2.0  # soft edge, roughly within the ball
    tree = cKDTree(nodes)
    d, _ = tree.query(nodes, k=2)
    # Nearest-neighbour spacing is fairly uniform.
    assert d[:, 1].std() / d[:, 1].mean() < 0.2
    assert edges.max() < len(nodes)
