"""A coarse Klein-bottle surface lattice, relaxed out of rod-rod collisions.

The figure-8 immersion crosses itself along a circle, so a grid of rods laid on the surface
puts rods from the two sheets in the same place near that circle. A plain tensor grid is
unsalvageable for exactly that reason: both sheets' crossing points sit at the same ``u``.

The fix here is not a special construction but a **relaxation in parameter space**. Nodes
start on a regular ``(u, v)`` grid, are embedded on the surface, and then nudge themselves
apart — every pair of rods closer than the clearance pushes its four endpoints along the
shortest separating direction, the push is projected back into ``(u, v)`` through the local
Jacobian, and the next iteration re-embeds. The surface never changes shape; only where the
nodes sit on it.

The seam is the Klein identification ``(u + 2π, v) = (u, -v)``: the last u-column is derived
from the first, reversed, so the mesh closes into a Klein bottle rather than a cylinder.

Target density is deliberately close to the sphere figure: 16 x 14 cells gives 448 rods and
224 nodes (the sphere is 450/152).
"""

from __future__ import annotations

import numpy as np

from .klein import (
    JOINT_R,
    ROD_R,
    SIZE_MM,
    immersed_bbox,
    printability,
    rod_elevations,
)

TWO_PI = 2.0 * np.pi
M_DEFAULT = 16  # u-cells
N_DEFAULT = 14  # v-cells  -> 448 rods, 224 nodes
CLEARANCE_MM = 2.4  # Ø1.8 rod + 0.6 mm gap
RELAX_TARGET_MM = 3.4  # aim higher than the requirement so the check has margin
_LOWER, _UPPER = immersed_bbox()
_SCALE = SIZE_MM / float((_UPPER - _LOWER).max())
_ORIGIN = np.array([0.5 * (_LOWER[0] + _UPPER[0]), 0.5 * (_LOWER[1] + _UPPER[1]), _LOWER[2]])


# ---------------------------------------------------------------------------------------
# Grid, seam, embedding
# ---------------------------------------------------------------------------------------
def initial_grid(m: int = M_DEFAULT, n: int = N_DEFAULT, *, shear: float = 1.0, offset: float = 0.5):
    """Warm start: a tensor grid nudged off the crossing, shape ``(m + 1, n)`` each.

    Two cheap choices do most of the work before any force is applied:

    * rows sit at ``v = (j + ½)·2π/n``, so no run of rods lies *along* the double curve
      (``v = 0, π``) where it would overlap the other sheet for its whole length;
    * a ``(1 − cos v)`` shear in ``u`` staggers the two sheets by half a cell at ``v = π``,
      so their crossing rods do not meet at the same point.

    The relaxation then only has to clean up the leftovers.
    """
    i = np.arange(m + 1)[:, None]
    j = np.arange(n)[None, :]
    u = np.broadcast_to(TWO_PI * i / m, (m + 1, n)).astype(float).copy()
    v = np.broadcast_to(TWO_PI * (j + offset) / n, (m + 1, n)).astype(float).copy()
    u += shear * (np.pi / m) * (1.0 - np.cos(v))
    return u, v


def apply_seam(u: np.ndarray, v: np.ndarray) -> None:
    """Tie the last u-column to the first, reversed: ``(2π, v) = (0, -v)``, in place."""
    m, n = u.shape[0] - 1, u.shape[1]
    k = (n - np.arange(n)) % n
    u[m, :] = u[0, k] + TWO_PI
    v[m, :] = -v[0, k]


def embed(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Points on the immersed surface, in mm, with the lowest point at ``z = 0``."""
    from .klein import figure8

    p = (figure8(u, v) - _ORIGIN) * _SCALE
    return p


def canonical_index(m: int, n: int) -> np.ndarray:
    """Map every ``(m + 1) × n`` grid slot onto its physical node (``m × n`` of them)."""
    out = np.empty((m + 1) * n, dtype=int)
    for i in range(m + 1):
        for j in range(n):
            out[i * n + j] = i * n + j if i < m else (n - j) % n
    return out


def grid_edges(m: int, n: int) -> np.ndarray:
    """Quad-mesh edges: ``m * n`` in u, ``m * n`` in v (after the seam collapses a column)."""
    def idx(i: int, j: int) -> int:
        return i * n + (j % n)

    edges = [(idx(i, j), idx(i + 1, j)) for i in range(m) for j in range(n)]
    edges += [(idx(i, j), idx(i, j + 1)) for i in range(m + 1) for j in range(n)]
    canonical = canonical_index(m, n)
    e = canonical[np.asarray(edges)]
    e = np.sort(e, axis=1)
    e = e[e[:, 0] != e[:, 1]]
    return np.unique(e, axis=0)


# ---------------------------------------------------------------------------------------
# Relaxation
# ---------------------------------------------------------------------------------------
def _closest_points(p0, p1, q0, q1):
    """Closest points between paired segments; returns ``(a, b, dist)`` point arrays."""
    d1, d2 = p1 - p0, q1 - q0
    r = p0 - q0
    a = np.einsum("ij,ij->i", d1, d1)
    e = np.einsum("ij,ij->i", d2, d2)
    b = np.einsum("ij,ij->i", d1, d2)
    c = np.einsum("ij,ij->i", d1, r)
    f = np.einsum("ij,ij->i", d2, r)
    den = a * e - b * b
    safe = np.where(np.abs(den) > 1e-12, den, 1.0)
    s = np.clip((b * f - c * e) / safe, 0.0, 1.0)
    t = np.clip((a * f - b * c) / safe, 0.0, 1.0)
    # Re-project each parameter onto the other segment a couple of times: cheap, and exact
    # enough for a repulsion force.
    for _ in range(2):
        t = np.clip((b * s + f) / np.where(e > 1e-12, e, 1.0), 0.0, 1.0)
        s = np.clip((b * t - c) / np.where(a > 1e-12, a, 1.0), 0.0, 1.0)
    pa = p0 + s[:, None] * d1
    qb = q0 + t[:, None] * d2
    return pa, qb, np.linalg.norm(pa - qb, axis=1), s, t


def _jacobian(u: np.ndarray, v: np.ndarray, h: float = 1e-6):
    """``∂P/∂u`` and ``∂P/∂v`` at every grid point, by central differences."""
    pu = (embed(u + h, v) - embed(u - h, v)) / (2.0 * h)
    pv = (embed(u, v + h) - embed(u, v - h)) / (2.0 * h)
    return pu, pv


def _to_uv(force: np.ndarray, pu: np.ndarray, pv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project 3-D displacements onto the surface's ``(u, v)`` parameters (least squares)."""
    a = np.einsum("ij,ij->i", pu, pu)
    b = np.einsum("ij,ij->i", pu, pv)
    c = np.einsum("ij,ij->i", pv, pv)
    fu = np.einsum("ij,ij->i", force, pu)
    fv = np.einsum("ij,ij->i", force, pv)
    det = a * c - b * b
    safe = np.where(np.abs(det) > 1e-12, det, 1.0)
    du = (c * fu - b * fv) / safe
    dv = (a * fv - b * fu) / safe
    return du, dv


def relax(
    m: int = M_DEFAULT,
    n: int = N_DEFAULT,
    *,
    iterations: int = 400,
    gain: float = 0.55,
    target: float = RELAX_TARGET_MM,
    max_step: float = 0.05,
    spring: float = 0.02,
    verbose: bool = False,
):
    """Push nodes around in ``(u, v)`` until no two non-adjacent rods are too close."""
    u, v = initial_grid(m, n)
    u0, v0 = u.copy(), v.copy()
    edges = grid_edges(m, n)
    physical = np.arange(m * n)  # one slot per physical node, ordered (i, j) with i < m

    from scipy.spatial import cKDTree

    for it in range(iterations):
        apply_seam(u, v)
        x = embed(u, v).reshape(-1, 3)[physical]
        p0, p1 = x[edges[:, 0]], x[edges[:, 1]]
        mid = 0.5 * (p0 + p1)
        span = float(np.linalg.norm(p1 - p0, axis=1).max()) + target
        cand = cKDTree(mid).query_pairs(span, output_type="ndarray")
        if len(cand):
            shared = (
                (edges[cand[:, 0], 0] == edges[cand[:, 1], 0])
                | (edges[cand[:, 0], 0] == edges[cand[:, 1], 1])
                | (edges[cand[:, 0], 1] == edges[cand[:, 1], 0])
                | (edges[cand[:, 0], 1] == edges[cand[:, 1], 1])
            )
            cand = cand[~shared]
        if len(cand) == 0:
            break

        pa, qb, dist, s_par, t_par = _closest_points(
            p0[cand[:, 0]], p1[cand[:, 0]], p0[cand[:, 1]], p1[cand[:, 1]]
        )
        close = dist < target
        cand = cand[close]
        pa, qb = pa[close], qb[close]
        s_par, t_par = s_par[close], t_par[close]
        dist = np.maximum(dist[close], 1e-6)
        if len(cand) == 0:
            break

        direction = (qb - pa) / dist[:, None]
        push = 0.5 * gain * (target - dist)[:, None] * direction

        force = np.zeros_like(x)
        # Spread each push over the two endpoints of each rod, by where the gap is.
        a0, a1 = edges[cand[:, 0], 0], edges[cand[:, 0], 1]
        b0, b1 = edges[cand[:, 1], 0], edges[cand[:, 1], 1]
        # `push` points from rod a toward rod b, so a moves away from it and b toward it.
        np.add.at(force, a0, -push * (1.0 - s_par)[:, None])
        np.add.at(force, a1, -push * s_par[:, None])
        np.add.at(force, b0, push * (1.0 - t_par)[:, None])
        np.add.at(force, b1, push * t_par[:, None])

        # Physical node -> grid slot (the first m rows; row m is derived from row 0).
        pu, pv = _jacobian(u, v)
        pu, pv = pu.reshape(-1, 3), pv.reshape(-1, 3)
        du, dv = _to_uv(force, pu[physical], pv[physical])

        du = np.clip(du, -max_step, max_step).reshape(m, n)
        dv = np.clip(dv, -max_step, max_step).reshape(m, n)
        # Weak spring home: the collisions get fixed, but the grid does not wander into
        # crumpled cells on the way there.
        u[:m] += du - spring * (u[:m] - u0[:m])
        v[:m] += dv - spring * (v[:m] - v0[:m])
        apply_seam(u, v)

        if verbose and (it % 25 == 0 or len(cand) == 0):
            print(f"  iter {it:3d}  close pairs {len(cand):4d}  min {dist.min():.3f} mm")

    apply_seam(u, v)
    return u, v, edges


# ---------------------------------------------------------------------------------------
# Building the figure
# ---------------------------------------------------------------------------------------
def build(m: int = M_DEFAULT, n: int = N_DEFAULT, *, iterations: int = 400):
    """Relaxed nodes and edges, in mm."""
    u, v, edges = relax(m, n, iterations=iterations)
    x = embed(u, v).reshape(-1, 3)[canonical_index(m, n)[: m * n]]
    return x, edges


def main() -> None:
    import os

    from .klein import clearance_pairs
    from .lattice import build_frame_mesh

    nodes, edges = build()
    stats = clearance_pairs(nodes, edges)
    lengths = np.linalg.norm(nodes[edges[:, 1]] - nodes[edges[:, 0]], axis=1)
    print(
        f"{M_DEFAULT} x {N_DEFAULT}: {len(nodes)} nodes, {len(edges)} rods, "
        f"min clearance {stats['min_distance']:.2f} mm "
        f"({stats['violations']} under {CLEARANCE_MM} mm), "
        f"rods {lengths.min():.1f}-{lengths.max():.1f} mm"
    )
    print("printability:", printability(nodes, edges))

    mesh = build_frame_mesh(
        nodes,
        edges,
        rod_radius=ROD_R,
        joint_radius=JOINT_R,
        rod_sections=8,
        joint_count=(6, 6),
        color_by="elevation",
    )
    os.makedirs("output/lattice", exist_ok=True)
    for suffix in ("ply", "stl"):
        path = f"output/lattice/klein_relaxed.{suffix}"
        mesh.export(path)
        print(f"wrote {path} ({os.path.getsize(path) / 1e6:.2f} MB, {len(mesh.faces)} triangles)")


if __name__ == "__main__":
    main()
