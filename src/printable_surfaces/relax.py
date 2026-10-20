"""Force-directed spring relaxation — 3D graph embedding with boundary constraints.

Seeds vertices randomly, wires them with k-nearest neighbours, then relaxes the layout
with springs (Hooke's law, target rest length) + finite-range repulsion, under boundary
conditions: surface vertices are projected onto the sphere, interior vertices are held
inside by a soft wall. This yields a lattice with near-uniform rod lengths everywhere.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def seed_vertices(radius: float, n_surface: int, n_interior: int, seed: int = 0) -> np.ndarray:
    """Random seed: ``n_surface`` points on the sphere, ``n_interior`` uniformly in the ball.

    Surface vertices come first (indices ``0..n_surface-1``).
    """
    rng = np.random.default_rng(seed)
    surf = rng.normal(size=(n_surface, 3))
    surf = surf / np.linalg.norm(surf, axis=1, keepdims=True) * radius
    directions = rng.normal(size=(n_interior, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    r = radius * 0.95 * np.cbrt(rng.uniform(0.0, 1.0, size=n_interior))
    interior = directions * r[:, None]
    return np.vstack([surf, interior])


def knn_edges(points: np.ndarray, k: int = 4) -> np.ndarray:
    """Undirected k-nearest-neighbour edges (symmetrized)."""
    tree = cKDTree(points)
    _, idx = tree.query(points, k=k + 1)
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for i in range(len(points)):
        for j in idx[i, 1:]:
            key = (min(i, j), max(i, j))
            if key not in seen:
                seen.add(key)
                out.append(key)
    return np.asarray(out, dtype=int)


def relax(
    nodes: np.ndarray,
    edges: np.ndarray,
    radius: float,
    n_surface: int,
    rest_length: float,
    iterations: int = 2000,
    cooling: float = 0.995,
) -> np.ndarray:
    """Relax vertex positions.

    Forces:
      * Hooke springs along edges: ``F = (dist − L)·û`` (attractive past ``L``, repulsive
        below it), so every rod converges to ``rest_length``.
      * Finite-range repulsion between nearby pairs (cutoff ≈ 1.3·L), to spread vertices
        uniformly without the long-range repulsion that would pack them at the boundary.
      * Soft wall: interior vertices past ``radius − L`` are pushed back inward.

    ``nodes[0:n_surface]`` are projected onto the sphere each step. Returns positions.
    """
    pos = np.array(nodes, dtype=float)
    L = rest_length
    temperature = radius / 2.0
    cutoff = 1.3 * L
    wall_r = radius - L

    for _ in range(iterations):
        force = np.zeros_like(pos)

        # Hooke springs.
        if len(edges):
            a = pos[edges[:, 0]]
            b = pos[edges[:, 1]]
            d = b - a
            dist = np.maximum(np.linalg.norm(d, axis=1), 1e-6)
            unit = d / dist[:, None]
            f = (dist - L)[:, None] * unit
            np.add.at(force, edges[:, 0], f)
            np.add.at(force, edges[:, 1], -f)

        # Finite-range repulsion.
        delta = pos[:, None, :] - pos[None, :, :]
        dd = np.linalg.norm(delta, axis=-1)
        np.fill_diagonal(dd, np.inf)
        unit = delta / np.maximum(dd, 1e-6)[:, :, None]
        rep = np.where(dd < cutoff, cutoff - dd, 0.0)
        force += (unit * rep[:, :, None]).sum(axis=1)

        # Soft wall keeping interior vertices ~L inside the surface.
        interior = pos[n_surface:]
        ir = np.linalg.norm(interior, axis=1)
        over = ir > wall_r
        if over.any():
            inward = -3.0 * (ir[over] - wall_r)[:, None] * (interior[over] / ir[over, None])
            force[n_surface:][over] += inward

        # Cap each vertex's step at ``temperature`` and integrate.
        dnorm = np.linalg.norm(force, axis=1, keepdims=True)
        scale = np.minimum(1.0, temperature / np.maximum(dnorm, 1e-12))
        pos += force * scale

        # Boundary conditions.
        pos[:n_surface] = (
            pos[:n_surface] / np.linalg.norm(pos[:n_surface], axis=1, keepdims=True) * radius
        )
        interior = pos[n_surface:]
        inorm = np.linalg.norm(interior, axis=1)
        out = inorm > radius
        if out.any():
            interior[out] = interior[out] / inorm[out, None] * (radius - 1e-3)
        pos[n_surface:] = interior

        temperature *= cooling

    return pos


def _derive_counts(radius: float, target_length: float) -> tuple[int, int]:
    """Surface and interior vertex counts consistent with ``target_length`` spacing."""
    n_surface = int(round(4.0 * np.pi * radius * radius / (target_length * target_length)))
    interior_volume = (4.0 / 3.0) * np.pi * (radius - target_length) ** 3
    n_interior = int(round(interior_volume / (target_length ** 3)))
    return max(n_surface, 4), max(n_interior, 1)


def relaxed_lattice(
    radius: float,
    target_length: float,
    k: int = 4,
    iterations: int = 2000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Seed → wire (k-NN) → relax, for a target rod length.

    Vertex counts are derived from ``target_length`` so the lattice fills the ball at that
    spacing. Returns ``(nodes, edges)`` with surface vertices first.
    """
    n_surface, n_interior = _derive_counts(radius, target_length)
    pts = seed_vertices(radius, n_surface, n_interior, seed)
    edges = knn_edges(pts, k)
    pos = relax(pts, edges, radius, n_surface, target_length, iterations=iterations)
    return pos, edges
