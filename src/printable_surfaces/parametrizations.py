"""Alternative sphere parametrizations, kept only for comparison against the cube-sphere.

These exist so Phase 1 can *demonstrate* why the cube-sphere is the right choice (pole-free,
near-uniform), rather than assert it.
"""

from __future__ import annotations

import numpy as np
import trimesh


def triangle_areas(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Areas of triangles given by ``faces`` (F, 3) indices into ``points`` (V, 3)."""
    p = np.asarray(points)[np.asarray(faces)]  # (F, 3, 3)
    cross = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    return np.linalg.norm(cross, axis=-1) / 2.0


def grid_faces(m: int, n: int, periodic_cols: bool = True) -> np.ndarray:
    """Triangulate an ``(m, n)`` grid of points into a ``(faces, 3)`` index array.

    Column indices wrap around (periodic) by default, as on a sphere's longitude; rows
    are not wrapped so pole rows stay as explicit (degenerate) rings.
    """
    faces = []
    for i in range(m):
        i1 = (i + 1) % m if periodic_cols else i + 1
        if periodic_cols is False and i1 >= m:
            break
        for j in range(n - 1):
            a = i * n + j
            b = i1 * n + j
            c = i * n + j + 1
            d = i1 * n + j + 1
            faces.append((a, b, c))
            faces.append((b, d, c))
    return np.asarray(faces, dtype=int)


def latlong_grid(n_theta: int, n_phi: int, radius: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Classic lat/long sphere: ``points`` (n_theta*n_phi, 3) and ``faces``.

    Note: the ``phi=0`` and ``phi=pi`` rows collapse to the poles, so the triangles there
    are degenerate — exactly the defect the cube-sphere avoids.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    phi = np.linspace(0.0, np.pi, n_phi)
    Theta, Phi = np.meshgrid(theta, phi, indexing="ij")
    x = radius * np.sin(Phi) * np.cos(Theta)
    y = radius * np.sin(Phi) * np.sin(Theta)
    z = radius * np.cos(Phi)
    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    faces = grid_faces(n_theta, n_phi, periodic_cols=True)
    return points, faces


def cube_sphere_grid(n: int, radius: float = 1.0, center=(0.0, 0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """Cube-sphere sampled on an ``n``-by-``n`` grid per chart: ``points`` and ``faces``."""
    from .surface import Sphere

    sphere = Sphere(radius=radius, center=center)
    uv = np.linspace(-1.0, 1.0, n)
    u, v = np.meshgrid(uv, uv, indexing="ij")
    all_points = []
    all_faces = []
    offset = 0
    for chart in sphere.charts():
        pts = chart.evaluate(u, v).reshape(-1, 3)
        faces = grid_faces(n, n, periodic_cols=False)
        all_faces.append(faces + offset)
        all_points.append(pts)
        offset += len(pts)
    return np.concatenate(all_points, axis=0), np.concatenate(all_faces, axis=0)


def naive_cube_sphere_grid(n: int, radius: float = 1.0, center=(0.0, 0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    """The *rejected* naive cube-sphere (normalize raw cube coordinate).

    Kept only so the Phase 1 comparison can demonstrate *why* the equiangular (gnomonic)
    variant in :class:`~printable_surfaces.surface.Sphere` was chosen: this one has ~2×
    corner distortion (CV ≈ 0.38 vs 0.09).
    """
    center = np.asarray(center, dtype=float)
    uv = np.linspace(-1.0, 1.0, n)
    u, v = np.meshgrid(uv, uv, indexing="ij")
    all_points = []
    all_faces = []
    offset = 0
    for axis, sign in [(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)]:
        ones = np.ones_like(u)
        if axis == 0:
            q = np.stack([sign * ones, u, v], axis=-1)
        elif axis == 1:
            q = np.stack([u, sign * ones, v], axis=-1)
        else:
            q = np.stack([u, v, sign * ones], axis=-1)
        pts = (center + radius * q / np.linalg.norm(q, axis=-1, keepdims=True)).reshape(-1, 3)
        all_faces.append(grid_faces(n, n, periodic_cols=False) + offset)
        all_points.append(pts)
        offset += len(pts)
    return np.concatenate(all_points, axis=0), np.concatenate(all_faces, axis=0)


def icosphere(subdivisions: int = 3, radius: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Reference icosphere (geodesic): ``points`` and ``faces`` via trimesh."""
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    return np.asarray(mesh.vertices, dtype=float), np.asarray(mesh.faces, dtype=int)


def area_stats(areas: np.ndarray) -> dict[str, float]:
    """Uniformity summary for a set of triangle areas (scale-free: CV and min/mean)."""
    areas = np.asarray(areas, dtype=float)
    return {
        "n_faces": float(len(areas)),
        "mean": float(areas.mean()),
        "std": float(areas.std()),
        "cv": float(areas.std() / areas.mean()) if areas.mean() > 0 else float("inf"),
        "min_over_mean": float(areas.min() / areas.mean()) if areas.mean() > 0 else 0.0,
        "n_degenerate": float((areas < 1e-12 * areas.mean()).sum()),
    }
