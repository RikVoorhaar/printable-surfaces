"""The figure-8 immersion of the Klein bottle, and the measurements taken on it.

The immersion (``a = 2``)::

    r = a + cos(u/2) sin v − sin(u/2) sin 2v
    x = r cos u,  y = r sin u,  z = sin(u/2) sin v + cos(u/2) sin 2v

with ``u, v ∈ [0, 2π)``. It is invariant under ``v → v + 2π`` and, because
``cos((u+2π)/2) = −cos(u/2)`` and ``sin((u+2π)/2) = −sin(u/2)``, under the Klein
identification ``(u + 2π, v) = (u, −v)``. Gluing a parameter grid with that flip gives a
genuine Klein bottle rather than a cylinder.

The image self-intersects along the circle ``(a cos u, a sin u, 0)``. Its preimage is exactly
the two parameter lines ``v = 0`` and ``v = π``, and the sheets cross at a constant angle
``2·atan(½) ≈ 53.13°``. That is why a lattice laid on the surface puts rods from the two
sheets in the same place: a plain tensor grid has both sheets crossing at the same ``u``.

This module is the geometry and the scoring only. The lattice itself — the relaxed grid
that gets the rods out of each other's way — lives in :mod:`printable_surfaces.klein_lattice`.
"""

from __future__ import annotations

import numpy as np

A = 2.0
SIZE_MM = 100.0
ROD_R = 0.9  # Ø1.8 mm, the T5 printed set
JOINT_R = 1.35  # Ø2.7 mm
BRIDGE_MAX = 10.0


def figure8(u, v, a: float = A) -> np.ndarray:
    """The immersion, shape ``(..., 3)``. ``u`` is not 2π-periodic; see the module docs."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    cu, su = np.cos(u / 2.0), np.sin(u / 2.0)
    sv, s2v = np.sin(v), np.sin(2.0 * v)
    r = a + cu * sv - su * s2v
    z = su * sv + cu * s2v
    return np.stack([r * np.cos(u), r * np.sin(u), z], axis=-1)


def immersed_bbox(a: float = A, samples: int = 1081) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounding box of the image (a fine sample; no analytic shortcut)."""
    t = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    uu, vv = np.meshgrid(t, t, indexing="ij")
    p = figure8(uu.ravel(), vv.ravel(), a=a)
    return p.min(axis=0), p.max(axis=0)


# ---------------------------------------------------------------------------------------
# Scoring a lattice
# ---------------------------------------------------------------------------------------
def rod_elevations(nodes: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Each rod's angle above horizontal, in degrees (``asin(|dz| / length)``)."""
    d = np.asarray(nodes)[edges[:, 1]] - np.asarray(nodes)[edges[:, 0]]
    return np.degrees(np.arcsin(np.clip(np.abs(d[:, 2]) / np.linalg.norm(d, axis=1), 0, 1)))


def _point_segment_distance(x: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Distance from points ``x`` (P,3) to segments ``p → q`` (P,3)."""
    d = q - p
    dd = np.einsum("ij,ij->i", d, d)
    t = np.einsum("ij,ij->i", x - p, d) / np.where(dd > 0.0, dd, 1.0)
    t = np.clip(t, 0.0, 1.0)
    return np.linalg.norm(x - (p + t[:, None] * d), axis=1)


def segment_distances(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Minimum distance between paired segments ``(p1,q1)`` vs ``(p2,q2)``, arrays (P,3).

    The squared distance is a quadratic over the unit square, so its minimum is either the
    unconstrained stationary point (if inside) or on the boundary — and every boundary
    minimum is a point-to-segment distance from one of the four endpoints.
    """
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    aa = np.einsum("ij,ij->i", d1, d1)
    ee = np.einsum("ij,ij->i", d2, d2)
    bb = np.einsum("ij,ij->i", d1, d2)
    cc = np.einsum("ij,ij->i", d1, r)
    ff = np.einsum("ij,ij->i", d2, r)
    den = aa * ee - bb * bb
    safe = np.where(np.abs(den) > 1e-12, den, 1.0)
    s = (bb * ff - cc * ee) / safe
    t = (aa * ff - bb * cc) / safe
    best = np.full(len(p1), np.inf)
    inside = (np.abs(den) > 1e-12) & (s >= 0.0) & (s <= 1.0) & (t >= 0.0) & (t <= 1.0)
    if inside.any():
        i = np.where(inside)[0]
        gap = r[i] + s[i, None] * d1[i] - t[i, None] * d2[i]
        best[i] = np.linalg.norm(gap, axis=1)
    for x, p, q in ((p1, p2, q2), (q1, p2, q2), (p2, p1, q1), (q2, p1, q1)):
        best = np.minimum(best, _point_segment_distance(x, p, q))
    return best


def clearance_pairs(
    nodes: np.ndarray,
    edges: np.ndarray,
    clearance: float = 2.4,
    *,
    adjacent: bool = False,
    chunk: int = 200_000,
) -> dict:
    """Minimum rod-rod distance over pairs that do (not) share a node.

    Broadphase: a KD-tree on rod midpoints with radius ``max_length + clearance`` (any pair
    closer than ``clearance`` must have midpoints within that). Then the exact
    segment-segment distance for every surviving pair.
    """
    from scipy.spatial import cKDTree  # ty: ignore[import-not-found]

    e = np.asarray(edges, dtype=int)
    p1, p2 = np.asarray(nodes, float)[e[:, 0]], np.asarray(nodes, float)[e[:, 1]]
    mids = 0.5 * (p1 + p2)
    length = np.linalg.norm(p2 - p1, axis=1)
    cand = cKDTree(mids).query_pairs(float(length.max()) + clearance, output_type="ndarray")
    if len(cand) == 0:
        return {"pairs": 0, "violations": 0, "min_distance": float("inf"), "pair": None}
    shared = (
        (e[cand[:, 0], 0] == e[cand[:, 1], 0])
        | (e[cand[:, 0], 0] == e[cand[:, 1], 1])
        | (e[cand[:, 0], 1] == e[cand[:, 1], 0])
        | (e[cand[:, 0], 1] == e[cand[:, 1], 1])
    )
    cand = cand[shared] if adjacent else cand[~shared]
    dist = np.empty(len(cand))
    for k in range(0, len(cand), chunk):
        c = cand[k : k + chunk]
        dist[k : k + chunk] = segment_distances(p1[c[:, 0]], p2[c[:, 0]], p1[c[:, 1]], p2[c[:, 1]])
    if len(dist) == 0:
        return {"pairs": 0, "violations": 0, "min_distance": float("inf"), "pair": None}
    worst = int(np.argmin(dist))
    return {
        "pairs": int(len(cand)),
        "violations": int((dist < clearance).sum()),
        "min_distance": float(dist.min()),
        "pair": (int(cand[worst, 0]), int(cand[worst, 1])),
    }


def printability(nodes: np.ndarray, edges: np.ndarray) -> dict:
    """Share of rods the printer can hold, using the project's rule.

    Printable = elevation ≥ 48°, or a horizontal (≤5°) bridge no longer than 10 mm, or a rod
    lying on the build plate.
    """
    nodes = np.asarray(nodes, float)
    e = np.asarray(edges, int)
    d = nodes[e[:, 1]] - nodes[e[:, 0]]
    length = np.linalg.norm(d, axis=1)
    elev = np.degrees(np.arcsin(np.clip(np.abs(d[:, 2]) / length, 0.0, 1.0)))
    z_min = float(nodes[:, 2].min())
    on_plate = (nodes[e[:, 0], 2] < z_min + 0.5) & (nodes[e[:, 1], 2] < z_min + 0.5)
    steep = elev >= 48.0
    bridge = (elev < 5.0) & (length <= BRIDGE_MAX)
    hist = np.histogram(elev, bins=np.arange(0.0, 91.0, 10.0))[0]
    return {
        "rods": int(len(e)),
        "below_48_pct": float(100.0 * (~(steep | bridge | on_plate)).mean()),
        "raw_below_48_pct": float(100.0 * (~steep).mean()),
        "steep_pct": float(100.0 * steep.mean()),
        "bridge_pct": float(100.0 * bridge.mean()),
        "on_plate": int(on_plate.sum()),
        "hist": {f"{lo:.0f}-{lo + 10:.0f}": int(c) for lo, c in zip(np.arange(0, 90, 10), hist, strict=False)},
        "median_elev": float(np.median(elev)),
        "mean_length": float(length.mean()),
        "min_length": float(length.min()),
        "max_length": float(length.max()),
    }
