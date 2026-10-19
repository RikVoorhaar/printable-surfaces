"""Tests for the sphere parametrization comparison (Phase 1)."""

import numpy as np

from printable_surfaces.parametrizations import (
    area_stats,
    cube_sphere_grid,
    icosphere,
    latlong_grid,
    naive_cube_sphere_grid,
)


def _areas(points, faces):
    p = np.asarray(points)[np.asarray(faces)]
    return np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=-1) / 2.0


def test_cube_sphere_grid_has_no_degenerate_triangles():
    # This would have caught the face-offset bug (faces indexing into the wrong chart).
    pts, faces = cube_sphere_grid(n=16)
    areas = _areas(pts, faces)
    assert areas.min() > 0.0
    assert faces.max() < len(pts)


def test_gnomonic_is_more_uniform_than_naive():
    p1, f1 = cube_sphere_grid(n=16)  # gnomonic
    p2, f2 = naive_cube_sphere_grid(n=16)  # naive
    cv_gnomonic = area_stats(_areas(p1, f1))["cv"]
    cv_naive = area_stats(_areas(p2, f2))["cv"]
    assert cv_gnomonic < cv_naive


def test_latlong_has_degenerate_poles():
    pts, faces = latlong_grid(n_theta=64, n_phi=32)
    areas = _areas(pts, faces)
    assert (areas < 1e-12 * areas.mean()).sum() > 0


def test_icosphere_is_highly_uniform():
    pts, faces = icosphere(subdivisions=4)
    stats = area_stats(_areas(pts, faces))
    assert stats["cv"] < 0.1
