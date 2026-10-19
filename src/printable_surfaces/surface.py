"""Surface model: a closed 2-manifold embedded in R^3.

A ``Surface`` knows how to answer "is this point inside the shape?" — the predicate
later phases use to clip interior lattices to the shape's volume.

A ``ParametricSurface`` is described by one or more rectangular ``Chart`` objects.
Keeping every chart a *rectangle* in (u, v) is what lets later phases trivially overlay
a uniform grid -> triangles -> lattice on any surface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Chart:
    """A single coordinate chart: a rectangle in (u, v) mapped onto the surface.

    Parameters
    ----------
    map_fn:
        ``(u, v) -> points`` where ``u``/``v`` are broadcast-compatible float arrays and
        the result has shape ``(*broadcast_shape, 3)``.
    """

    name: str
    u_min: float
    u_max: float
    v_min: float
    v_max: float
    map_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]

    def evaluate(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Map chart coordinates to points on the surface: ``(..., 3)`` array."""
        return np.asarray(self.map_fn(np.asarray(u, dtype=float), np.asarray(v, dtype=float)), dtype=float)


class Surface(ABC):
    """A closed 2-manifold surface embedded in R^3."""

    @abstractmethod
    def inside(self, points: np.ndarray) -> np.ndarray:
        """Return a boolean mask ``(N,)`` indicating which of the ``(N, 3)`` points are inside."""

    @property
    @abstractmethod
    def bounding_box(self) -> np.ndarray:
        """Axis-aligned bounding box as a ``(2, 3)`` array of ``[[min], [max]]``."""


class ParametricSurface(Surface):
    """A surface described by one or more rectangular charts."""

    @abstractmethod
    def charts(self) -> list[Chart]:
        """Return the charts that cover the surface."""


class Sphere(ParametricSurface):
    """A sphere, parametrized as an **equiangular (gnomonic) cube-sphere** (6 square charts).

    Why this parametrization:

    * **Pole-free** — unlike lat/long ``(theta, phi)``, no chart collapses to a point.
    * **Near-uniform** — mapping ``(u, v)`` through ``tan(angle)`` before normalizing gives
      triangle areas with CV ≈ 0.09, essentially matching an icosphere, whereas the naive
      "normalize the raw cube coordinate" cube-sphere has CV ≈ 0.38 (≈ 2× corner distortion).
    * **Rectangular charts** — each chart is a plain rectangle, so later phases can overlay a
      uniform grid -> triangles -> lattice with no special casing.

    (See ``parametrizations.cube_sphere_grid`` and the Phase 1 comparison for the numbers.)
    """

    def __init__(
        self,
        radius: float = 1.0,
        center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self.radius = float(radius)
        self.center = np.asarray(center, dtype=float)

    def inside(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        return np.linalg.norm(points - self.center, axis=-1) <= self.radius

    @property
    def bounding_box(self) -> np.ndarray:
        r = self.radius
        return np.asarray([self.center - r, self.center + r])

    def charts(self) -> list[Chart]:
        """Six square charts, one per cube face, each normalized onto the sphere."""
        r = self.radius
        c = self.center

        def _face(axis: int, sign: float) -> Chart:
            def fn(u: np.ndarray, v: np.ndarray) -> np.ndarray:
                u, v = np.broadcast_arrays(u, v)
                # Equiangular (gnomonic) mapping: the chart coordinate is an *angle*,
                # so we take tan() to get the tangent-plane coordinate before normalizing.
                # This spreads samples evenly by angle and removes the corner distortion
                # of the naive radial cube-sphere.
                tu = np.tan(u * (np.pi / 4.0))
                tv = np.tan(v * (np.pi / 4.0))
                ones = np.ones_like(tu)
                if axis == 0:
                    q = np.stack([sign * ones, tu, tv], axis=-1)
                elif axis == 1:
                    q = np.stack([tu, sign * ones, tv], axis=-1)
                else:
                    q = np.stack([tu, tv, sign * ones], axis=-1)
                return c + r * q / np.linalg.norm(q, axis=-1, keepdims=True)

            return Chart(name=f"{'+' if sign > 0 else '-'}{'XYZ'[axis]}", u_min=-1.0, u_max=1.0, v_min=-1.0, v_max=1.0, map_fn=fn)

        return [
            _face(0, +1.0),
            _face(0, -1.0),
            _face(1, +1.0),
            _face(1, -1.0),
            _face(2, +1.0),
            _face(2, -1.0),
        ]
