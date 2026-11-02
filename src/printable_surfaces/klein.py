"""Figure-8 immersion of the Klein bottle, as a printable rod lattice.

The immersion (the standard figure-eight Klein bottle, ``a = 2``)::

    r = a + cos(u/2) sin v - sin(u/2) sin 2v
    x = r cos u,  y = r sin u,  z = sin(u/2) sin v + cos(u/2) sin 2v

with ``u, v ∈ [0, 2π)``. It is invariant under ``v → v + 2π`` and — because
``cos((u+2π)/2) = -cos(u/2)`` and ``sin((u+2π)/2) = -sin(u/2)`` — under the Klein
identification ``(u + 2π, v) = (u, -v)``. A grid over the square glued with that flip is a
genuine Klein bottle: ``m·n`` nodes, no duplicates, the last ``u``-cell joined to the first
row with the columns reversed (``j → n-1-j``).

The image is a self-intersecting torus-like tube: ``r ∈ [0.75, 3.25]``, ``z ∈ [-1.25, 1.25]``
for ``a = 2``, so at 100 mm the largest bbox dimension (``y``, 6.49 model units) sets the
scale, ``15.408``.

**Double curve.** The image intersects itself along the circle ``(a cos u, a sin u, 0)``
(radius 30.8 mm at the 100 mm scale). Its preimage is exactly the two parameter lines
``v = 0`` and ``v = π``; both map onto the whole circle, for every ``u`` — verified
numerically to 1e-15. The two sheets cross transversally at a constant angle
``2·atan(1/2) ≈ 53.13°``.

**Why a plain grid cannot work.** Every rod whose cell straddles ``v = 0`` crosses the
curve at ``(u_i, 0)``; the ``v = π`` rod of the other sheet crosses at ``(u_i, π)`` — the
same 3-D point, because both sheets use the same ``u`` lines. Measured on a plain
``36 × 36`` grid (``klein_grid(36, 36, shear=0, cross_bias=0, diagonals="all")``):
minimum rod-rod distance ``2.2e-16 mm``, 3 525 of 196 513 candidate pairs
*coincident* — the two sheets are welded together along the circle. Refining does not help
(``48 × 40``: 10 081 coincident pairs), and the crossing-point sets of the two sheets are
identical for *any* tensor grid (each sheet contributes one crossing per ``u``-line and one
per diagonal, and both use the same ``u`` values). The three fixes used here are:

* **shear** — ``u → u + (h_u/4)(1 − cos v)``, so the ``v = 0`` rods cross at ``u_i`` and the
  ``v = π`` rods at ``u_i + h_u/2``: the maximum separation available, half a ``u``-cell.
* **drop/brace diagonals by measurement** — a diagonal crosses at ``u_i + h_u/2``, i.e.
  exactly where the *other* sheet's diagonal or ``v``-edge crosses. Those, and any other
  diagonal that cannot hold the clearance, are rejected by :func:`brace_diagonals`.
* **arc-length + biased ``v`` spacing** — ``|dP/dv| = √(cos²v + 4cos²2v)`` varies 3× with
  ``v`` (0.70 at 45°, 2.24 at 0°), so ``v`` rows are spaced by arc length, and the two cells
  that straddle the double curve are widened, because their ``u``-edges sit only ``h_v/2``
  from the curve and the other sheet's crossing rod passes ``0.8·√5·h_v/2`` away.

Rods follow the printed set (Ø1.8 mm rod, Ø2.7 mm joint) on a 100 mm object. Run::

    uv run python -m printable_surfaces.klein
"""

from __future__ import annotations

import numpy as np

A = 2.0
SIZE_MM = 100.0
ROD_R = 0.9  # Ø1.8 mm, the T5 printed set
JOINT_R = 1.35  # Ø2.7 mm
CLEARANCE_MM = 2.4  # rod Ø1.8 + 0.6 mm gap
BRACE_MARGIN_MM = 0.2  # a diagonal must clear by this much beyond the requirement
SIN48 = float(np.sin(np.radians(48.0)))
SIN5 = float(np.sin(np.radians(5.0)))
BRIDGE_MAX = 10.0

M_DEFAULT = 30  # u-cells — limited by the tight waist and by the crossing separation
N_DEFAULT = 24  # v-rows
CROSS_BIAS = 0.6  # widening of the two v-cells that straddle the double curve
SHEAR = 0.5  # u-phase slip between the two sheets, in u-cells

TWO_PI = 2.0 * np.pi


# --------------------------------------------------------------------------------------
# The immersion
# --------------------------------------------------------------------------------------
def figure8(u, v, a: float = A) -> np.ndarray:
    """The figure-8 immersion of the Klein bottle, shape ``(..., 3)``.

    Accepts scalars or arrays (broadcast together). ``u`` is *not* 2π-periodic in this
    chart; the Klein identification reverses ``v`` (see :func:`klein_grid`).
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    cu, su = np.cos(u / 2.0), np.sin(u / 2.0)
    sv, s2v = np.sin(v), np.sin(2.0 * v)
    r = a + cu * sv - su * s2v
    z = su * sv + cu * s2v
    return np.stack([r * np.cos(u), r * np.sin(u), z], axis=-1)


def verify_symmetries(n: int = 64, tol: float = 1e-12) -> dict:
    """Check ``P(u, v+2π) = P(u, v)`` and ``P(u+2π, -v) = P(u, v)`` numerically.

    Also reports how badly ``P(u+2π, v) = P(u, v)`` fails, to show ``u`` really is not
    periodic and the flip is needed.
    """
    rng = np.random.default_rng(0)
    u = rng.uniform(0.0, TWO_PI, n)
    v = rng.uniform(0.0, TWO_PI, n)
    p = figure8(u, v)
    err_v = float(np.abs(figure8(u, v + TWO_PI) - p).max())
    err_klein = float(np.abs(figure8(u + TWO_PI, -v) - p).max())
    err_u = float(np.abs(figure8(u + TWO_PI, v) - p).max())
    return {
        "v_periodic_err": err_v,
        "klein_flip_err": err_klein,
        "u_without_flip_err": err_u,
        "tol": tol,
        "ok": bool(err_v <= tol and err_klein <= tol),
    }


def immersed_bbox(a: float = A, samples: int = 1081) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounding box of the *image* (a fine sample; no analytic shortcut)."""
    t = np.linspace(0.0, TWO_PI, samples, endpoint=False)
    uu, vv = np.meshgrid(t, t, indexing="ij")
    p = figure8(uu.ravel(), vv.ravel(), a=a)
    return p.min(axis=0), p.max(axis=0)


def scale_to_size(a: float = A, size: float = SIZE_MM) -> float:
    """Uniform scale that makes the largest bounding-box dimension ``size``."""
    lo, hi = immersed_bbox(a=a)
    return float(size / (hi - lo).max())


def _to_mm(p: np.ndarray, a: float = A, size: float = SIZE_MM) -> tuple[np.ndarray, float]:
    """Scale about the bbox centre and drop so ``z_min = 0``; returns ``(points, scale)``."""
    lo, hi = immersed_bbox(a=a)
    s = size / float((hi - lo).max())
    q = (np.asarray(p, dtype=float) - 0.5 * (lo + hi)) * s
    q[..., 2] -= q[..., 2].min()
    return q, s


# --------------------------------------------------------------------------------------
# Double curve
# --------------------------------------------------------------------------------------
def double_curve_preimage() -> dict[str, object]:
    """Where the self-intersection lives in ``(u, v)``: the two lines ``v = 0`` and ``v = π``.

    Analytic: a double point needs equal ``(x, y)``, which (``r ≥ a - 1.25 > 0``, so the
    image never reaches the axis) forces ``u' ≡ u``; equal ``r`` and equal ``z`` then give a
    2×2 *rotation* system in ``(sin v, sin 2v)``, whose only solutions are ``v' = v`` or
    ``v' = π - v`` with ``sin 2v = 0`` — i.e. ``v ∈ {0, π}`` for all ``u``.
    """
    return {
        "lines": [("v", 0.0), ("v", float(np.pi))],
        "u_range": (0.0, TWO_PI),
        "image": "circle (a cos u, a sin u, 0)",
        "crossing_angle_deg": float(np.degrees(2.0 * np.arctan(0.5))),
    }


def find_double_points(samples: int = 600, a: float = A, tol_rel: float = 1e-3) -> dict:
    """Numerically rediscover the double-point pairs and compare with the analytic answer.

    Samples the square densely, finds 3-D-coincident pairs within ``tol_rel·size`` that are
    far apart in parameter space, and reports the parameter gaps of those pairs.
    """
    from scipy.spatial import cKDTree

    lo, hi = immersed_bbox(a=a, samples=511)
    size = float((hi - lo).max())
    tol = tol_rel * size

    t = np.linspace(0.0, TWO_PI, samples, endpoint=False)
    uu, vv = np.meshgrid(t, t, indexing="ij")
    U, V = uu.ravel(), vv.ravel()
    p = figure8(U, V, a=a)
    pairs = cKDTree(p).query_pairs(tol, output_type="ndarray")
    step = TWO_PI / samples

    dv = np.abs(V[pairs[:, 0]] - V[pairs[:, 1]])
    dv = np.minimum(dv, TWO_PI - dv)
    du = np.abs(U[pairs[:, 0]] - U[pairs[:, 1]])
    du = np.minimum(du, TWO_PI - du)
    far = dv > 3.0 * step                      # a real double point, not a sample neighbour
    sel = pairs[far]
    if len(sel) == 0:
        return {"candidate_pairs": int(len(pairs)), "double_pairs": 0, "ok": False}

    va, vb = V[sel[:, 0]], V[sel[:, 1]]
    dist = np.linalg.norm(p[sel[:, 0]] - p[sel[:, 1]], axis=1)

    def line_dev(v: np.ndarray) -> np.ndarray:      # distance to {0, π} in v
        return np.abs((v + np.pi / 2.0) % np.pi - np.pi / 2.0)

    return {
        "samples": int(samples * samples),
        "tol_model": float(tol),
        "candidate_pairs": int(len(pairs)),
        "double_pairs": int(len(sel)),
        "max_pair_distance": float(dist.max()),
        "max_u_gap": float(du[far].max()),
        "max_v_dev_from_mod_pi": float(max(line_dev(va).max(), line_dev(vb).max())),
        "distinct_u_values": int(len(np.unique(np.round(U[sel[:, 0]], 6)))),
        "ok": bool(len(sel) > 0 and dist.max() <= tol),
    }


# --------------------------------------------------------------------------------------
# Grid  (m cells in u, n rows in v, periodic in v, welded across the u seam with v → −v)
# --------------------------------------------------------------------------------------
def v_speed(v: np.ndarray) -> np.ndarray:
    """``|dP/dv| = √(cos²v + 4cos²2v)`` — a function of ``v`` only.

    (``dr/dv`` and ``dz/dv`` are a rotation of ``(cos v, 2cos 2v)``, so the ``u`` dependence
    cancels.) It swings from 0.70 at ``v = 45°`` to 2.24 at ``v = 0``: a plain ``v``-grid
    squeezes the rows by 3×.
    """
    v = np.asarray(v, dtype=float)
    return np.sqrt(np.cos(v) ** 2 + 4.0 * np.cos(2.0 * v) ** 2)


def klein_v_rows(
    n: int, v_offset: float = 0.5, cross_bias: float = CROSS_BIAS, samples: int = 20_001
) -> np.ndarray:
    """``n`` rows along ``v``: equal arc length in the bulk, **wider near the double curve**.

    ``v_offset = 0.5`` puts ``v = 0`` and ``v = π`` in the middle of a ``v``-cell, so the
    curve crosses cell interiors and no node or rod lies on it; it is also the only offset
    (besides 0) for which the Klein flip ``v → -v`` maps columns onto columns, ``j → n-1-j``.

    Rows are equally spaced in ``N`` with ``dN/dv = |dP/dv| / (1 + cross_bias·cos²v)``, so
    the two straddling cells are ``(1 + cross_bias)`` times wider than the bulk. ``cos²v``
    is even and symmetric about ``π``; the second half is mirrored exactly
    (``v[n-1-j] = 2π - v[j]``) so the flip is exact rather than interpolation-accurate.
    """
    if n % 2:
        raise ValueError("n must be even (the v → -v flip must map columns onto columns)")
    t = np.linspace(0.0, TWO_PI, samples)
    dens = v_speed(t) / (1.0 + cross_bias * np.cos(t) ** 2)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(t))])
    v = np.interp((np.arange(n) + v_offset) * (cum[-1] / n), cum, t)
    v[n // 2 :] = TWO_PI - v[n // 2 - 1 :: -1]
    return v


def klein_uv(
    m: int,
    n: int,
    u_offset: float = 0.0,
    v_offset: float = 0.5,
    shear: float = SHEAR,
    cross_bias: float = CROSS_BIAS,
) -> np.ndarray:
    """Parameter points ``(m*n, 2)`` of the grid, row-major in ``(i, j)``.

    ``shear`` is the ``u``-phase slip between the two sheets, in ``u``-cells:
    ``u → u + (shear/2)·h_u·(1 − cos v)``, i.e. 0 at ``v = 0`` and ``shear·h_u`` at
    ``v = π``. ``0.5`` staggers the two sheets' crossing rods by half a ``u``-cell;
    ``0`` reproduces the plain tensor grid, whose rods intersect at the double curve.
    ``1 − cos v`` is even in ``v``, so the Klein flip still maps nodes to nodes.
    """
    hu = TWO_PI / m
    v = klein_v_rows(n, v_offset=v_offset, cross_bias=cross_bias)
    u = (np.arange(m) + u_offset) * hu
    slip = 0.5 * shear * hu * (1.0 - np.cos(v))
    uu, vv = np.meshgrid(u, v, indexing="ij")
    return np.stack([uu + slip[None, :], vv], axis=-1).reshape(-1, 2)


def klein_faces(m: int, n: int) -> np.ndarray:
    """Triangulation of the grid, gluing row ``m-1`` to row 0 with ``v → -v``.

    For the last cell in ``u`` the far side is the ``u = 0`` row with the columns reversed
    (``j → n-1-j``), which is exactly ``P(u+2π, v) = P(u, -v)``. Every grid node is used
    once — the surface is closed with ``m*n`` nodes and no welding.
    """
    faces = []
    for i in range(m):
        wrap = i == m - 1
        i1 = 0 if wrap else i + 1

        def idx(k: int, j: int) -> int:
            jj = (n - 1 - j) % n if wrap else j % n
            return k * n + jj

        for j in range(n):
            j2 = (j + 1) % n
            a = i * n + j
            b = idx(i1, j)
            c = i * n + j2
            d = idx(i1, j2)
            faces.append((a, b, c))
            faces.append((b, d, c))
    return np.asarray(faces, dtype=int)


def crossing_strip_cells(m: int, n: int) -> list[int]:
    """The ``v``-cells that straddle the double-curve preimage ``v = 0`` and ``v = π``."""
    if n % 2:
        raise ValueError("n must be even so that both v = 0 and v = π sit at cell centres")
    return [n - 1, n // 2 - 1]


def candidate_diagonals(m: int, n: int) -> list[tuple[int, int]]:
    """The ``b–c`` diagonal of every cell (the ones a full triangulation would add)."""
    out: list[tuple[int, int]] = []
    for i in range(m):
        wrap = i == m - 1
        i1 = 0 if wrap else i + 1
        for j in range(n):
            j2 = (j + 1) % n
            far = (n - 1 - j) % n if wrap else j % n
            out.append((i1 * n + far, i * n + j2))
    return out


def klein_edges(m: int, n: int, diagonals: str = "greedy") -> np.ndarray:
    """Rod graph: the grid's 1-skeleton, optionally without some diagonals.

    The ``u``- and ``v``-edges alone already form the (Klein-bottle) grid; the diagonals only
    brace the quads. ``diagonals="greedy"`` (default) offers every diagonal to
    :func:`brace_diagonals` and keeps the ones that clear; ``"crossing"`` drops the ``2m``
    diagonals of the two straddling strips only; ``"none"`` is the pure quad lattice;
    ``"all"`` keeps the full 1-skeleton (which is *not* collision-free).
    """
    if diagonals not in ("all", "crossing", "greedy", "none"):
        raise ValueError("diagonals must be 'all', 'crossing', 'greedy' or 'none'")
    strip = set(crossing_strip_cells(m, n)) if diagonals == "crossing" else set()
    if diagonals == "none":
        strip = set(range(n))
    seen: set[tuple[int, int]] = set()

    def add(a: int, b: int) -> None:
        seen.add((a, b) if a < b else (b, a))

    for i in range(m):
        wrap = i == m - 1
        i1 = 0 if wrap else i + 1
        for j in range(n):
            j2 = (j + 1) % n

            def far(jj: int) -> int:
                return (n - 1 - jj) % n if wrap else jj % n

            a, c = i * n + j, i * n + j2
            b, d = i1 * n + far(j), i1 * n + far(j2)
            add(a, b)  # v-edge: the family that crosses the double curve
            add(c, d)
            add(a, c)  # u-edge: runs parallel to the double curve
            add(b, d)
            if j not in strip and diagonals != "greedy":
                add(b, c)  # diagonal
    return np.asarray(sorted(seen), dtype=int)


def brace_diagonals(
    nodes: np.ndarray,
    edges: np.ndarray,
    candidates: list[tuple[int, int]],
    clearance: float = CLEARANCE_MM + BRACE_MARGIN_MM,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Add each candidate diagonal **only if it keeps the whole frame clear**.

    The quad lattice alone already clears; the diagonals are what collide. A diagonal spans
    a cell, so at the tight waist (``r`` falls to 0.75, a ``u``-cell there is ~3 mm) it
    passes within a millimetre of a neighbouring cell's diagonal, and in the two straddling
    strips it passes through the other sheet's rod. Greedy in cell order: keep a diagonal
    when its minimum distance to every already-present non-adjacent rod is at least
    ``clearance`` (default ``2.4 + 0.2 mm``, so the finished frame keeps a margin). This is
    the "coarsen locally" step — bracing survives wherever it fits.

    Returns ``(all_edges, kept)``.
    """
    e = np.asarray(edges, dtype=int)
    kept: list[tuple[int, int]] = []
    for cand in candidates:
        a, b = cand
        if len(e):
            d = segment_distances(
                np.repeat(nodes[a][None, :], len(e), axis=0),
                np.repeat(nodes[b][None, :], len(e), axis=0),
                nodes[e[:, 0]],
                nodes[e[:, 1]],
            )
            adjacent = (e[:, 0] == a) | (e[:, 1] == a) | (e[:, 0] == b) | (e[:, 1] == b)
            if np.where(adjacent, np.inf, d).min() < clearance:
                continue
        e = np.vstack([e, [[a, b]]])
        kept.append(cand)
    return e, kept


def klein_grid(
    m: int,
    n: int,
    *,
    a: float = A,
    size: float = SIZE_MM,
    u_offset: float = 0.0,
    v_offset: float = 0.5,
    shear: float = SHEAR,
    cross_bias: float = CROSS_BIAS,
    diagonals: str = "greedy",
    brace_clearance: float = CLEARANCE_MM + BRACE_MARGIN_MM,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Nodes (mm, on the plate), edges (rods), faces and the ``(u, v)`` chart.

    The immersed surface is scaled uniformly so its largest bounding-box dimension is
    ``size``, centred in ``x, y`` and dropped so ``z_min = 0``. ``faces`` is the full
    reference triangulation (its 1-skeleton is the rod graph when ``diagonals="all"``).
    """
    uv = klein_uv(m, n, u_offset=u_offset, v_offset=v_offset, shear=shear,
                  cross_bias=cross_bias)
    nodes, _ = _to_mm(figure8(uv[:, 0], uv[:, 1], a=a), a=a, size=size)
    faces = klein_faces(m, n)
    edges = klein_edges(m, n, diagonals=diagonals)
    if diagonals == "greedy":
        edges, _ = brace_diagonals(nodes, edges, candidate_diagonals(m, n),
                                   clearance=brace_clearance)
        edges = np.asarray(sorted({tuple(sorted(e)) for e in edges.tolist()}), dtype=int)
    return nodes, faces, edges, uv


# --------------------------------------------------------------------------------------
# Double curve vs the grid
# --------------------------------------------------------------------------------------
def double_curve_check(
    m: int,
    n: int,
    *,
    a: float = A,
    size: float = SIZE_MM,
    v_offset: float = 0.5,
    cross_bias: float = CROSS_BIAS,
    shear: float = SHEAR,
    samples: int = 8001,
) -> dict:
    """How the double curve sits relative to the grid, in parameter and in 3-D space."""
    from scipy.spatial import cKDTree

    nodes, faces, edges, uv = klein_grid(
        m, n, a=a, size=size, u_offset=0.0, v_offset=v_offset, shear=shear,
        cross_bias=cross_bias,
    )
    # reproduce the same transform the nodes got, so the curve lives in their frame
    lo, hi = immersed_bbox(a=a)
    s = size / float((hi - lo).max())
    mid = 0.5 * (lo + hi)
    z_drop = ((figure8(uv[:, 0], uv[:, 1], a=a) - mid) * s)[:, 2].min()

    u = np.linspace(0.0, TWO_PI, samples, endpoint=False)
    curve = (figure8(u, np.zeros_like(u), a=a) - mid) * s
    curve[:, 2] -= z_drop
    tree = cKDTree(curve)

    node_d, _ = tree.query(nodes)
    t = np.linspace(0.0, 1.0, 33)
    p1, p2 = nodes[edges[:, 0]], nodes[edges[:, 1]]
    pts = p1[:, None, :] + t[None, :, None] * (p2 - p1)[:, None, :]
    rod_pts_d = tree.query(pts.reshape(-1, 3))[0].reshape(len(edges), -1)
    rod_d = rod_pts_d.min(axis=1)

    # rods that cross the curve: how long do they stay near it, and at what angle?
    crossing = np.where(rod_d < 0.05)[0]
    near_samples = int(rod_pts_d[crossing].min(axis=1).__le__(0.1).sum()) if len(crossing) else 0
    worst_near = int((rod_pts_d[crossing] < 0.1).sum(axis=1).max()) if len(crossing) else 0
    tangent = np.gradient(curve, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    angles = []
    for k in crossing:
        _, idx = tree.query(0.5 * (p1[k] + p2[k]))
        d = (p2[k] - p1[k]) / np.linalg.norm(p2[k] - p1[k])
        angles.append(float(np.degrees(np.arccos(abs(float(d @ tangent[idx]))))))
    ang_min = float(min(angles)) if angles else float("nan")

    v = uv[:, 1]
    v_dev = np.abs((v + np.pi / 2.0) % np.pi - np.pi / 2.0)  # parameter distance to {0, π}

    return {
        "nodes": int(len(nodes)),
        "rods": int(len(edges)),
        "min_node_to_curve_mm": float(node_d.min()),
        "rods_crossing_curve": int(len(crossing)),
        "samples_near_curve_total": near_samples,
        "max_samples_per_rod_within_0p1mm": worst_near,
        "min_crossing_angle_deg": ang_min,
        "min_v_dev_deg": float(np.degrees(v_dev.min())),
        "v_cell_deg": float(np.degrees(2.0 * klein_v_rows(n, cross_bias=cross_bias)[0])),
        "curve_circle_radius_mm": float(a * s),
        "curve_z_mm": float(curve[0, 2]),
        "edge_lying_on_curve": int(worst_near >= 8),
    }


def crossing_separation_mm(
    m: int, n: int, *, a: float = A, size: float = SIZE_MM, cross_bias: float = CROSS_BIAS
) -> dict:
    """Distance along the double circle between the two sheets' crossing rods.

    The shear puts the ``v = 0`` rods at ``u_i`` and the ``v = π`` rods at
    ``u_i + (h_u/4)(1 + cos v_0)``, so the separation is ``a·(h_u/2)·cos(v_0)·scale``.
    """
    s = scale_to_size(a, size)
    v = klein_v_rows(n, cross_bias=cross_bias)
    hu = TWO_PI / m
    return {
        "u_cell_mm": float(a * hu * s),
        "v0_deg": float(np.degrees(v[0])),
        "separation_mm": float(a * 0.5 * hu * np.cos(v[0]) * s),
    }


# --------------------------------------------------------------------------------------
# Clearance
# --------------------------------------------------------------------------------------
def _point_segment_distance(x: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Distance from points ``x`` (P,3) to segments ``p → q`` (P,3)."""
    d = q - p
    dd = np.einsum("ij,ij->i", d, d)
    t = np.einsum("ij,ij->i", x - p, d) / np.where(dd > 0.0, dd, 1.0)
    t = np.clip(t, 0.0, 1.0)
    return np.linalg.norm(x - (p + t[:, None] * d), axis=1)


def segment_distances(
    p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray
) -> np.ndarray:
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


def rod_lengths(nodes: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Length of each rod, in mm."""
    return np.linalg.norm(np.asarray(nodes)[edges[:, 1]] - np.asarray(nodes)[edges[:, 0]], axis=1)


def clearance_pairs(
    nodes: np.ndarray,
    edges: np.ndarray,
    clearance: float = CLEARANCE_MM,
    *,
    adjacent: bool = False,
    chunk: int = 200_000,
) -> dict:
    """Minimum rod-rod distance over pairs that do (not) share a node.

    Broadphase: a KD-tree on rod midpoints with radius ``max_length + clearance`` (any pair
    closer than ``clearance`` must have midpoints within that). Then the exact
    segment-segment distance for every surviving pair.
    """
    from scipy.spatial import cKDTree

    e = np.asarray(edges, dtype=int)
    p1, p2 = np.asarray(nodes, float)[e[:, 0]], np.asarray(nodes, float)[e[:, 1]]
    mids = 0.5 * (p1 + p2)
    cand = cKDTree(mids).query_pairs(float(rod_lengths(nodes, edges).max()) + clearance,
                                     output_type="ndarray")
    if len(cand) == 0:
        return {"pairs": 0, "violations": 0, "min_distance": float("inf"), "pair": None}
    i, j = cand[:, 0], cand[:, 1]
    shared = (
        (e[i, 0] == e[j, 0]) | (e[i, 0] == e[j, 1]) | (e[i, 1] == e[j, 0]) | (e[i, 1] == e[j, 1])
    )
    cand = cand[shared] if adjacent else cand[~shared]
    dist = np.empty(len(cand))
    for k in range(0, len(cand), chunk):
        c = cand[k : k + chunk]
        dist[k : k + chunk] = segment_distances(p1[c[:, 0]], p2[c[:, 0]],
                                                p1[c[:, 1]], p2[c[:, 1]])
    if len(dist) == 0:
        return {"pairs": 0, "violations": 0, "min_distance": float("inf"), "pair": None}
    worst = int(np.argmin(dist))
    return {
        "pairs": int(len(cand)),
        "violations": int((dist < clearance).sum()),
        "min_distance": float(dist.min()),
        "pair": (int(cand[worst, 0]), int(cand[worst, 1])),
        "distances": dist,
        "pairs_idx": cand,
    }


def adjacent_angles(nodes: np.ndarray, edges: np.ndarray) -> dict:
    """Angles between rods that share a node, measured **outward from that node**.

    ``0°`` would mean two rods lying on top of each other; ``180°`` means the same grid line
    continuing straight through the joint (the normal case in a lattice).
    """
    e = np.asarray(edges, dtype=int)
    d = np.asarray(nodes, float)[e[:, 1]] - np.asarray(nodes, float)[e[:, 0]]
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    by_node: dict[int, list[int]] = {}
    for k, (a, b) in enumerate(e):
        by_node.setdefault(int(a), []).append(k)
        by_node.setdefault(int(b), []).append(k)
    ips = []
    for node, rods in by_node.items():
        out = [d[k] if e[k, 0] == node else -d[k] for k in rods]
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                ips.append(float(np.degrees(np.arccos(np.clip(out[i] @ out[j], -1, 1)))))
    a = np.asarray(ips)
    return {
        "pairs": int(len(a)),
        "min_deg": float(a.min()),
        "median_deg": float(np.median(a)),
        "under_15_deg": int((a < 15.0).sum()),
        "under_30_deg": int((a < 30.0).sum()),
        "over_165_deg": int((a > 165.0).sum()),
    }


def adjacent_far_end_clearance(nodes: np.ndarray, edges: np.ndarray) -> dict:
    """For rods sharing a node: distance from each rod's far end to the other rod.

    The shared node itself gives a distance of 0 (the joint sphere covers both), so this is
    the honest measure of how close two *non-overlapping* neighbours are.
    """
    e = np.asarray(edges, dtype=int)
    p = np.asarray(nodes, float)
    by_node: dict[int, list[int]] = {}
    for k, (a, b) in enumerate(e):
        by_node.setdefault(int(a), []).append(k)
        by_node.setdefault(int(b), []).append(k)
    vals = []
    for node, rods in by_node.items():
        for i in range(len(rods)):
            for j in range(i + 1, len(rods)):
                ri, rj = rods[i], rods[j]
                fi = p[e[ri, 0]] if e[ri, 1] == node else p[e[ri, 1]]
                fj = p[e[rj, 0]] if e[rj, 1] == node else p[e[rj, 1]]
                d1 = _point_segment_distance(fi[None, :], p[e[rj, 0]][None, :],
                                             p[e[rj, 1]][None, :])[0]
                d2 = _point_segment_distance(fj[None, :], p[e[ri, 0]][None, :],
                                             p[e[ri, 1]][None, :])[0]
                vals.append(float(min(d1, d2)))
    v = np.asarray(vals)
    return {
        "pairs": int(len(v)),
        "min_mm": float(v.min()),
        "below_2p4": int((v < CLEARANCE_MM).sum()),
        "below_1p8": int((v < 1.8).sum()),
        "median_mm": float(np.median(v)),
    }


def connected_components(nodes: np.ndarray, edges: np.ndarray) -> int:
    """Number of connected components of the rod graph (1 = printable as one piece)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components as cc

    e = np.asarray(edges, dtype=int)
    n = len(nodes)
    g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    return int(cc(g, directed=False)[0])


def node_degrees(nodes: np.ndarray, edges: np.ndarray) -> dict[int, int]:
    """How many rods meet at each node (histogram ``{degree: count}``)."""
    deg = np.bincount(np.asarray(edges, dtype=int).ravel(), minlength=len(nodes))
    vals, counts = np.unique(deg, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts)}


# --------------------------------------------------------------------------------------
# Printability
# --------------------------------------------------------------------------------------
def rod_elevations(nodes: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Each rod's angle above horizontal, in degrees (``asin(|dz| / length)``)."""
    d = np.asarray(nodes)[edges[:, 1]] - np.asarray(nodes)[edges[:, 0]]
    return np.degrees(np.arcsin(np.clip(np.abs(d[:, 2]) / np.linalg.norm(d, axis=1), 0, 1)))


def printability(nodes: np.ndarray, edges: np.ndarray) -> dict:
    """Share of rods the printer can hold, using the project's rule.

    Printable = elevation ≥ 48°, or a horizontal (≤5°) bridge no longer than 10 mm, or a rod
    lying on the build plate. Reports the overall share, the raw below-48° share, and a 10°
    histogram of elevations.
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
        "hist": {f"{lo:.0f}-{lo + 10:.0f}": int(c)
                 for lo, c in zip(np.arange(0, 90, 10), hist)},
        "median_elev": float(np.median(elev)),
        "mean_length": float(length.mean()),
        "min_length": float(length.min()),
        "max_length": float(length.max()),
    }


# --------------------------------------------------------------------------------------
# Building
# --------------------------------------------------------------------------------------
def build(
    m: int = M_DEFAULT,
    n: int = N_DEFAULT,
    *,
    a: float = A,
    size: float = SIZE_MM,
    rod_radius: float = ROD_R,
    joint_radius: float = JOINT_R,
    u_offset: float = 0.0,
    v_offset: float = 0.5,
    shear: float = SHEAR,
    cross_bias: float = CROSS_BIAS,
    diagonals: str = "greedy",
    rod_sections: int = 12,
    joint_count: tuple[int, int] = (6, 6),
):
    """Thicken the rod graph into the printable mesh (lifted so the joints touch the plate)."""
    from .lattice import build_frame_mesh

    nodes, faces, edges, uv = klein_grid(
        m, n, a=a, size=size, u_offset=u_offset, v_offset=v_offset, shear=shear,
        cross_bias=cross_bias, diagonals=diagonals,
    )
    lift = nodes.copy()
    lift[:, 2] += joint_radius  # the joint spheres, not the node centres, touch z = 0
    mesh = build_frame_mesh(
        lift, edges, rod_radius=rod_radius, joint_radius=joint_radius,
        rod_sections=rod_sections, joint_count=joint_count,
    )
    return nodes, edges, uv, mesh


def report(m: int = M_DEFAULT, n: int = N_DEFAULT, *, verbose: bool = True) -> dict:
    """All the numbers for the finished lattice."""
    nodes, faces, edges, uv = klein_grid(m, n)
    cc = clearance_pairs(nodes, edges)
    ad = clearance_pairs(nodes, edges, adjacent=True)
    ang = adjacent_angles(nodes, edges)
    pr = printability(nodes, edges)
    dcc = double_curve_check(m, n)
    cs = crossing_separation_mm(m, n)
    L = rod_lengths(nodes, edges)
    out = {
        "a": A,
        "scale": scale_to_size(),
        "grid": (m, n),
        "nodes": int(len(nodes)),
        "rods": int(len(edges)),
        "braces": int(len(edges) - 2 * m * n),
        "brace_candidates": int(m * n),
        "clearance": cc,
        "adjacent": ad,
        "adjacent_angles": ang,
        "adjacent_far_end": adjacent_far_end_clearance(nodes, edges),
        "components": connected_components(nodes, edges),
        "node_degrees": node_degrees(nodes, edges),
        "printability": pr,
        "double_curve": dcc,
        "crossing": cs,
        "rod_length": (float(L.min()), float(np.median(L)), float(L.max())),
    }
    if verbose:
        print(f"a={A}  scale={out['scale']:.4f}  grid={m}x{n}")
        print(f"nodes={out['nodes']}  rods={out['rods']}  "
              f"braces kept={out['braces']}/{out['brace_candidates']}")
        print(f"rod length mm: min {out['rod_length'][0]:.2f}  "
              f"median {out['rod_length'][1]:.2f}  max {out['rod_length'][2]:.2f}")
        print(f"node degrees: {out['node_degrees']}  components={out['components']}")
        print(f"clearance: pairs={cc['pairs']}  violations={cc['violations']}  "
              f"min={cc['min_distance']:.3f} mm")
        af = out["adjacent_far_end"]
        print(f"adjacent pairs: pairs={ad['pairs']} min={ad['min_distance']:.3f} "
              f"(0 at the shared node, by construction)   outward angles: "
              f"min={ang['min_deg']:.1f} median={ang['median_deg']:.1f} "
              f"<15deg={ang['under_15_deg']} >165deg(straight)={ang['over_165_deg']}")
        print(f"adjacent far-end to the other rod: min={af['min_mm']:.2f} mm "
              f"median={af['median_mm']:.2f} mm, {af['below_2p4']} pairs < 2.4 mm "
              f"(all within one joint sphere)")
        print(f"double curve: min node dist {dcc['min_node_to_curve_mm']:.2f} mm; "
              f"{dcc['rods_crossing_curve']} rods cross it, at most "
              f"{dcc['max_samples_per_rod_within_0p1mm']}/33 samples within 0.1 mm "
              f"(crossing angle {dcc['min_crossing_angle_deg']:.2f} deg to the tangent); "
              f"min v-deviation {dcc['min_v_dev_deg']:.2f} deg of a {dcc['v_cell_deg']:.2f} deg cell")
        print(f"crossing: u-cell {cs['u_cell_mm']:.2f} mm, sheet separation "
              f"{cs['separation_mm']:.2f} mm")
        print(f"printability: raw below 48 deg = {pr['raw_below_48_pct']:.1f}%  "
              f"(project rule: {pr['below_48_pct']:.1f}% unprintable; "
              f"{pr['steep_pct']:.1f}% steep, {pr['bridge_pct']:.1f}% bridge, "
              f"{pr['on_plate']} on plate)")
        print(f"elevation histogram (deg: count): {pr['hist']}")
    return out


def main() -> None:
    import os

    out = report(M_DEFAULT, N_DEFAULT)
    nodes, edges, uv, mesh = build(M_DEFAULT, N_DEFAULT)
    os.makedirs("output/lattice", exist_ok=True)
    path = "output/lattice/klein_surface_frame.stl"
    mesh.export(path)
    print(f"wrote {path} ({os.path.getsize(path) / 1e6:.2f} MB, {len(mesh.faces)} triangles, "
          f"{len(mesh.vertices)} vertices, watertight={mesh.is_watertight})")
    print(f"mesh bounds (mm): {np.round(mesh.bounds, 2).tolist()}")
    return out


if __name__ == "__main__":
    main()
