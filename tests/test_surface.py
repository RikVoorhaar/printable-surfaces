"""Tests for the surface model (Phase 1)."""

import numpy as np

from printable_surfaces.surface import Sphere


def test_sphere_charts_lie_on_unit_sphere():
    s = Sphere(radius=1.0)
    uv = np.linspace(-1.0, 1.0, 9)
    u, v = np.meshgrid(uv, uv, indexing="ij")
    for chart in s.charts():
        pts = chart.evaluate(u, v)
        assert np.allclose(np.linalg.norm(pts, axis=-1), 1.0, atol=1e-12)


def test_sphere_inside():
    s = Sphere(radius=2.0)
    pts = np.array([[0.0, 0.0, 0.0], [1.9, 0.0, 0.0], [2.1, 0.0, 0.0], [0.0, 0.0, 2.0]])
    assert s.inside(pts).tolist() == [True, True, False, True]


def test_sphere_bounding_box():
    s = Sphere(radius=2.0, center=(1.0, 2.0, 3.0))
    assert np.allclose(s.bounding_box, [[-1.0, 0.0, 1.0], [3.0, 4.0, 5.0]])
