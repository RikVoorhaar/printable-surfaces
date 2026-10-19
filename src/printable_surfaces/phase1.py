"""Phase 1 — shape definition (sphere).

Builds the cube-sphere ``Surface``, compares it against the lat/long parametrization and a
reference icosphere on triangle-uniformity, and renders the parametrizations so we can
*see* that the cube-sphere is pole-free and near-uniform.

Run:  uv run python -m printable_surfaces.phase1
"""

from __future__ import annotations

import os

import numpy as np

from .parametrizations import (
    area_stats,
    cube_sphere_grid,
    icosphere,
    latlong_grid,
    naive_cube_sphere_grid,
)
from .surface import Sphere
from . import viz


def _compare() -> dict[str, dict[str, float]]:
    """Compare triangle-area uniformity across the candidate parametrizations."""
    results: dict[str, dict[str, float]] = {}

    pts, faces = naive_cube_sphere_grid(n=16)
    results["cube-sphere (naive)"] = area_stats(_areas(pts, faces))

    pts, faces = cube_sphere_grid(n=16)
    results["cube-sphere (gnomonic)"] = area_stats(_areas(pts, faces))

    pts, faces = latlong_grid(n_theta=64, n_phi=32)
    results["lat/long (theta, phi)"] = area_stats(_areas(pts, faces))

    pts, faces = icosphere(subdivisions=4)
    results["icosphere (reference)"] = area_stats(_areas(pts, faces))

    return results


def _areas(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    p = np.asarray(points)[np.asarray(faces)]
    return np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=-1) / 2.0


def main() -> None:
    out_dir = os.path.join("output", "phase1")
    os.makedirs(out_dir, exist_ok=True)

    sphere = Sphere(radius=1.0)
    print("Sphere bounding box:\n", sphere.bounding_box)

    # Inside/outside smoke test.
    probes = np.array([[0.0, 0.0, 0.0], [0.9, 0.9, 0.9], [2.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    inside = sphere.inside(probes)
    print("\nInside/outside test (|p| <= 1):")
    for p, isin in zip(probes, inside):
        print(f"  {p.tolist()}  ->  inside={bool(isin)}")

    # Chart coverage sanity: each chart's corners should land on the sphere.
    print("\nChart corner samples (should all have |p| == 1):")
    uv = np.array([-1.0, 1.0])
    for chart in sphere.charts():
        u, v = np.meshgrid(uv, uv, indexing="ij")
        pts = chart.evaluate(u, v)
        radii = np.linalg.norm(pts, axis=-1)
        print(f"  {chart.name}: radii={np.round(radii.ravel(), 6).tolist()}")

    # Visualization: cube-sphere grid lines (all 6 charts).
    n = 16
    uv = np.linspace(-1.0, 1.0, n)
    u, v = np.meshgrid(uv, uv, indexing="ij")
    chart_grids = [c.evaluate(u, v) for c in sphere.charts()]
    fig, _ = viz.plot_grid_lines(uv, chart_grids, labels=[c.name for c in sphere.charts()], title="Cube-sphere (6 charts)")
    viz.save(fig, os.path.join(out_dir, "cube_sphere.png"))

    # Visualization: lat/long wireframe (to show the degenerate poles).
    import matplotlib.pyplot as plt

    pts, faces = latlong_grid(n_theta=64, n_phi=32)
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    # Draw only the triangle edges as a wireframe.
    for tri in faces:
        p = pts[list(tri) + [tri[0]]]
        ax.plot(p[:, 0], p[:, 1], p[:, 2], color="tab:blue", lw=0.3)
    ax.set_title("Lat/long sphere (note dense poles)")
    viz._equalize(ax, pts)
    viz.save(fig, os.path.join(out_dir, "latlong.png"))

    # Uniformity comparison.
    print("\nTriangle-area uniformity (lower CV and higher min/mean = more uniform):")
    print(f"{'parametrization':28s} {'faces':>6s} {'CV':>7s} {'min/mean':>9s} {'degenerate':>11s}")
    for name, stats in _compare().items():
        print(
            f"{name:28s} {int(stats['n_faces']):6d} {stats['cv']:7.3f} "
            f"{stats['min_over_mean']:9.3f} {int(stats['n_degenerate']):11d}"
        )

    print(f"\nFigures written to {out_dir}/")


if __name__ == "__main__":
    main()
