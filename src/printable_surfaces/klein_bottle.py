"""The bottle-shaped immersion of the Klein bottle, and a rod lattice laid on it.

This is the *conventional* picture of a Klein bottle — the one everybody draws: a bulbous
body with a neck that bends over, dives through the body wall and opens into the base from
the inside. It is a different immersion from the figure-8 one in
:mod:`printable_surfaces.klein` (which has no such piercing neck), so it gets its own
module.

The parametrisation (``u, v ∈ [0, 2π]``, ``H = 100 mm`` target height)::

    r(u) = 4 − 2 cos u
    c    = (5 − 3√3)/2                     # the u where the body first touches z = 0
    b    = (20 − 2c) √(1 − c²)
    H0   = 16 + b
    s    = H / H0

    x = s [ 6 + 6 cos u (1 + sin u) + r cos u cos v ]     0 ≤ u ≤ π
    x = s [ 6 + 6 cos u (1 + sin u) − r cos v       ]     π < u ≤ 2π
    y = s r sin v
    z = s [ b − (16 + r cos v) sin u ]                    0 ≤ u ≤ π
    z = s [ b − 16 sin u             ]                    π < u ≤ 2π

Why ``c`` is what it is: on the first sheet ``z = s[b − (16 + r cos v) sin u]``, so the
lowest point is ``s[b − max(16 + r(u)) sin u]``. Maximising ``(20 − 2 cos u) sin u`` gives
``cos u = c`` and the maximum ``(20 − 2c)√(1 − c²) = b``. Hence the body reaches ``z = 0``
exactly and the neck top reaches ``s[b + 16] = H`` at ``u = 3π/2``.

``UPSIDE_DOWN`` then reflects ``z`` about the mid-plane, so the *neck* is what rests on the
plate. The reflection is an isometry: the lattice, its rod clearances and every rod's angle
above horizontal are identical either way round, only the orientation changes.

The seam. ``P(0, v) = (12 + 2cos v, 2 sin v, b)`` and ``P(2π, v) = (12 − 2cos v, 2 sin v, b)``
(on the second sheet), so the two ends of the hose are glued by the **glide reflection**

    φ(v) = π − v,        (2π, v) ≡ (0, φ(v)),

which is orientation-reversing and an involution: that is precisely what makes the object a
Klein bottle instead of a cylinder. ``v`` itself is ``2π``-periodic (``y`` and the two ``cos v``
terms only see ``v`` through ``sin``/``cos``).

The surface is only ``C⁰`` at ``u = π``: both sheets give the same horizontal circle there,
but the frame of the tube keeps rotating on the first sheet and is frozen on the second, so
``∂P/∂u`` jumps (by 4.7° median, 10.4° worst). ``verify()`` reports that jump rather than
hiding it.

Where it crosses itself
-----------------------
On the second sheet the surface is a family of **horizontal circles** — ``z = s[b − 16 sin u]``
has no ``v`` in it — so ``u`` and ``3π − u`` share a height, with ``A(3π − u) = 12 − A(u)`` and
``r(3π − u) = 8 − r(u)``. The two circles then cross wherever
``|r₁ − r₂| < |A₁ − A₂| < r₁ + r₂``. That gives

* ``u ∈ [3.447270, 3.871228]`` and its mirror ``3π − u ∈ [5.553550, 5.977508]``,
* a **closed loop** on the upper right shoulder at ``z = 69.0 … 85.2 mm``,
  ``√(x²+y²) = 21.9 … 27.7 mm``, ``y = ±6.3 mm`` — the hole through which the neck dives
  into the body,
* a preimage of **two** closed circles in the ``(u, v)`` torus, both mapping one-to-one onto
  that same 3-D curve.

All of it is inside the second sheet (``u ↔ 3π − u``); the two *sheets* never meet. The ends
of each arc are the tangencies ``d = r₁ + r₂`` and ``d = |r₁ − r₂|`` (i.e. ``sin u = −2/3``).

Why the lattice is a 22 × 10 grid
---------------------------------
``u`` is very unevenly paced: ``|∂P/∂u|`` runs from 5.5 mm/rad at ``u = 3π/2`` (the mouth at
the top of the neck, a focus of the coordinate) up to 47 mm/rad. ``v`` is much better
behaved, but the *thinnest* ring — ``r = 2s = 5.54 mm`` at the seam — only leaves room for
``n ≤ 14`` nodes at 2.4 mm clearance at all (``2·5.54·sin(π/14) = 2.45 mm``). In practice
``n = 12`` and ``14`` leave none: the two lobes genuinely interpenetrate at the double curve
and the relaxer stalls with 10–70 violations. ``n = 10`` with ``m = 22`` reaches
**440 rods / 220 nodes, minimum non-adjacent rod-rod distance 2.78 mm, zero violations**,
with rods 2.9–17.4 mm long.

The ``v`` offset is constrained by the seam: ``φ`` maps grid row ``j`` to row
``n/2 − 2·offset − j``, so only ``offset ∈ {0, ½}`` keep the glue on the grid. ``½`` is the
figure-8 recipe and works (0 violations), but only reaches 2.42 mm and leaves 23.6 mm rods;
``offset = 0`` is used here. Neither offset puts a grid line *along* the double curve — the
preimage is not a constant-``v`` line (``v`` sweeps 0.01 … 6.27 rad along it).

Printability is the one real compromise: **36 % of rods are below 48° and not short
bridges** (30 % before relaxation, 43 % if the rods are made coarser). The 10–48° band is
the bottle's shoulder, where the surface itself slopes too shallowly for a self-supporting
rod and the rods are too long to count as bridges; pushing cells below 10 mm there would
need well over a thousand rods.

Nothing in here touches the figure-8 bottle: that lives in :mod:`printable_surfaces.klein`
and :mod:`printable_surfaces.klein_lattice` and is left alone.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------------------
SIZE_MM = 100.0
ROD_R = 0.9  # Ø1.8 mm, the T5 printed set
JOINT_R = 1.35  # Ø2.7 mm
BRIDGE_MAX = 10.0

TWO_PI = 2.0 * np.pi

C_CUSP = (5.0 - 3.0 * np.sqrt(3.0)) / 2.0  # ≈ −0.0980762
B_LIFT = (20.0 - 2.0 * C_CUSP) * np.sqrt(1.0 - C_CUSP**2)  # ≈ 20.0987849
H0 = 16.0 + B_LIFT  # ≈ 36.0987849
SCALE = SIZE_MM / H0  # ≈ 2.77017635
U_CUSP = float(np.arccos(C_CUSP))  # ≈ 1.6690305 rad, where the body cusp sits

# The parametrisation below is written with the *body* on the plate and the neck on top.
# The figure we want is the other way up — neck down, body up — so z is reflected through
# the mid-plane. This is an isometry, so the lattice, its clearances and every rod angle are
# unchanged; only which end touches the table moves.
UPSIDE_DOWN = True


# ---------------------------------------------------------------------------------------
# The immersion
# ---------------------------------------------------------------------------------------
def radius(u):
    """Tube radius ``r(u) = 4 − 2 cos u`` (before scaling)."""
    return 4.0 - 2.0 * np.cos(u)


def bottle(u, v):
    """The immersion, shape ``(..., 3)``, in mm, with ``0 ≤ z ≤ 100``.

    ``u`` is taken literally: only ``u ∈ [0, 2π]`` is the fundamental domain. For values
    outside it (which the relaxer can produce on derived rows) use :func:`extended`.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    u, v = np.broadcast_arrays(u, v)

    r = radius(u)
    cu, su, cv = np.cos(u), np.sin(u), np.cos(v)
    first = u <= np.pi  # 0 ≤ u ≤ π; the second sheet is π < u ≤ 2π

    x_off = np.where(first, r * cu * cv, -r * cv)
    z_off = np.where(first, -(16.0 + r * cv) * su, -16.0 * su)
    y = r * np.sin(v)

    x = 6.0 + 6.0 * cu * (1.0 + su) + x_off
    z = B_LIFT + z_off
    if UPSIDE_DOWN:
        z = H0 - z
    return SCALE * np.stack([x, y, z], axis=-1)


def seam_v(v):
    """The seam identification, ``φ(v) = π − v`` (mod 2π): ``(0, v) ≡ (2π, φ(v))``."""
    return np.mod(np.pi - np.asarray(v, dtype=float), TWO_PI)


def extended(u, v):
    """The immersion continued past the fundamental domain, consistently with the seam.

    ``P(u + 2π, v) = P(u, φ(v))``, and ``φ`` is an involution, so the continuation is just a
    flip of ``v`` for every ``2π`` shift of ``u``. This is what lets the derived seam column
    be written as ``u = 2π`` with the *original* row-0 parameters, exactly as the figure-8
    lattice does, without special-casing the branch at ``u > 2π``.
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    q = np.floor(u / TWO_PI)
    u_wrapped = u - q * TWO_PI
    v_wrapped = np.where(np.mod(q, 2.0) == 0.0, v, np.pi - v)
    return bottle(u_wrapped, v_wrapped)


def uv_grid(nu: int = 721, nv: int = 361) -> tuple[np.ndarray, np.ndarray]:
    """A sample grid with ``nv`` even, so it is closed under the seam flip ``v ↦ π − v``."""
    if nv % 2:
        raise ValueError("nv must be even so that the v grid is closed under v ↦ π − v")
    return np.meshgrid(
        np.linspace(0.0, TWO_PI, nu), np.linspace(0.0, TWO_PI, nv), indexing="ij"
    )


def bbox(nu: int = 721, nv: int = 361) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounding box of the image, from a fine sample."""
    uu, vv = uv_grid(nu, nv)
    p = bottle(uu.ravel(), vv.ravel())
    return p.min(axis=0), p.max(axis=0)


# ---------------------------------------------------------------------------------------
# Verification of the embedding
# ---------------------------------------------------------------------------------------
def verify(nu: int = 601, nv: int = 302) -> dict:
    """Numerical checks of the seam, the periodicity, the crease and the height range."""
    uu, vv = uv_grid(nu, nv)
    allp = bottle(uu.ravel(), vv.ravel())
    lo, hi = allp.min(axis=0), allp.max(axis=0)

    # v-periodicity: P(u, v) == P(u, v + 2π) for the whole v grid.
    v_shift = extended(uu.ravel(), vv.ravel() + TWO_PI)
    v_err = float(np.linalg.norm(v_shift - allp, axis=-1).max())

    # Seam: (0, v) and (2π, φ(v)) are the same point.
    v = np.linspace(0.0, TWO_PI, 4001)
    left = bottle(np.zeros_like(v), v)
    right = bottle(np.full_like(v, TWO_PI), seam_v(v))
    seam_err = float(np.linalg.norm(left - right, axis=-1).max())

    # ... and that this is the only seam: a wrong φ should fail loudly.
    wrong = bottle(np.full_like(v, TWO_PI), v)
    seam_wrong = float(np.linalg.norm(left - wrong, axis=-1).max())

    # z ≥ 0 and the extremes sit where the algebra says.
    zmin, zmax = float(lo[2]), float(hi[2])
    p_cusp = bottle(U_CUSP, 0.0)
    p_top = bottle(1.5 * np.pi, 0.0)

    # Continuity (C⁰) across u = π, and the size of the tangent jump (C¹) there.
    v2 = np.linspace(0.0, TWO_PI, 721)
    c0 = {}
    for eps in (1e-3, 1e-5):
        c0[eps] = float(
            np.linalg.norm(bottle(np.pi - eps, v2) - bottle(np.pi + eps, v2), axis=-1).max()
        )
    du_l = (bottle(np.pi, v2) - bottle(np.pi - 1e-4, v2)) / 1e-4
    du_r = (bottle(np.pi + 1e-4, v2) - bottle(np.pi, v2)) / 1e-4
    c1 = np.degrees(
        np.arccos(
            np.clip(
                np.einsum("ij,ij->i", du_l, du_r)
                / (np.linalg.norm(du_l, axis=1) * np.linalg.norm(du_r, axis=1)),
                -1.0,
                1.0,
            )
        )
    )

    return {
        "b": B_LIFT,
        "H0": H0,
        "s": SCALE,
        "c": C_CUSP,
        "u_cusp": U_CUSP,
        "bbox_min": lo,
        "bbox_max": hi,
        "size": hi - lo,
        "z_min": zmin,
        "z_max": zmax,
        "z_at_cusp_point": float(p_cusp[2]),
        "z_at_top_point": float(p_top[2]),
        "v_periodicity_error": v_err,
        "seam_phi_max_error": seam_err,
        "seam_wrong_phi_error": seam_wrong,
        "crease_c0_error_1e-3": c0[1e-3],
        "crease_c0_error_1e-5": c0[1e-5],
        "crease_c1_max_deg": float(c1.max()),
        "crease_c1_median_deg": float(np.median(c1)),
    }


# ---------------------------------------------------------------------------------------
# The surface mesh
# ---------------------------------------------------------------------------------------
def surface_mesh(nu: int = 480, nv: int = 240):
    """A dense triangle mesh of the immersion, welded into a closed surface.

    The ``u = 2π`` column is identified with the ``u = 0`` column through ``v ↦ π − v``, which
    is why ``nv`` must be even: the uniform ``v`` grid is then closed under the flip, so the
    two columns have exactly the same vertices and the weld is combinatorial rather than
    numerical.
    """
    import trimesh

    if nv % 2:
        raise ValueError("nv must be even")
    uu, vv = np.meshgrid(
        np.linspace(0.0, TWO_PI, nu + 1), np.linspace(0.0, TWO_PI, nv + 1), indexing="ij"
    )
    verts = bottle(uu[:nu, :nv].ravel(), vv[:nu, :nv].ravel())  # (2π, v) is welded to (0, π−v)

    def vid(i, j):
        i = np.asarray(i)
        j = np.mod(np.asarray(j), nv)
        return np.where(i == nu, (nv // 2 - j) % nv, i * nv + j)

    ii, jj = np.meshgrid(np.arange(nu), np.arange(nv), indexing="ij")
    a = vid(ii, jj).ravel()
    b = vid(ii + 1, jj).ravel()
    c = vid(ii + 1, jj + 1).ravel()
    d = vid(ii, jj + 1).ravel()
    faces = np.concatenate(
        [np.stack([a, b, c], axis=1), np.stack([a, c, d], axis=1)], axis=0
    )
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    # No `fix_normals()`: a Klein bottle is non-orientable, so a globally consistent winding
    # cannot exist. The parameterisation's own winding is consistent except across the seam.
    return mesh


# ---------------------------------------------------------------------------------------
# Lattice: grid, seam, embedding
# ---------------------------------------------------------------------------------------
M_DEFAULT = 22  # u-cells  -> 22 * 10 = 440 rods, 220 nodes  (the shipped figure)
N_DEFAULT = 10  # v-cells
CLEARANCE_MM = 2.4  # Ø1.8 rod + 0.6 mm gap
RELAX_TARGET_MM = 2.9  # push for this; the requirement is CLEARANCE_MM, this leaves margin


def seam_index(n: int, offset: float = 0.0) -> np.ndarray:
    """Combinatorial seam: row ``j`` of the last u-column is row ``k(j)`` of the first.

    The flip ``φ(v) = π − v`` maps grid row ``j`` onto grid row ``k`` only when
    ``v_k = π − v_j``, i.e. ``k = n/2 − 2·offset − j``. That is an integer for the two
    seam-compatible offsets ``0`` and ``½``, so the glue never needs interpolation — but it
    does force ``n`` even.
    """
    j = np.arange(n)
    return (n // 2 - int(round(2.0 * offset)) - j) % n


def u_speed(samples: int = 4001, v_samples: int = 48):
    """Mean speed ``⟨|∂P/∂u|⟩_v`` along ``u`` — the metric the equal-arclength grid uses."""
    t = np.linspace(0.0, TWO_PI, samples)
    v = np.linspace(0.0, TWO_PI, v_samples, endpoint=False)
    h = 1e-6
    sp = np.zeros(samples)
    for vv in v:
        sp += np.linalg.norm((bottle(t + h, vv) - bottle(t - h, vv)) / (2.0 * h), axis=1)
    return t, sp / v_samples


def arclength_u(m: int, samples: int = 4001) -> np.ndarray:
    """``m + 1`` values of ``u`` spacing the loop by equal mean arc length.

    The raw ``u`` parameter is very unevenly paced: ``|∂P/∂u|`` runs from 5.5 mm/rad at
    ``u = 3π/2`` (the mouth at the top of the neck, where the coordinate has a focus) to
    47 mm/rad elsewhere. A uniform ``u`` grid therefore has 1.5 mm rods at the top and 17 mm
    rods at the seam. Equal arclength evens that out, which is what keeps the shortest rods
    above the joint diameter and the longest rods from cutting chords through the surface.
    """
    t, sp = u_speed(samples)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (sp[1:] + sp[:-1]) * np.diff(t))])
    targets = cum[-1] * np.arange(m + 1) / m
    u = np.interp(targets, cum, t)
    u[0], u[-1] = 0.0, TWO_PI
    return u


def initial_grid(
    m: int = M_DEFAULT,
    n: int = N_DEFAULT,
    *,
    offset: float = 0.0,
    shear: float = 0.0,
    equal_arclength: bool = False,
):
    """Warm start: a tensor grid, rows at ``v = 2π(j + offset)/n``.

    ``offset`` must keep the seam ``φ(v) = π − v`` on the grid, which allows only ``0`` or
    ``½`` (see :func:`seam_index`). The figure-8 recipe uses ``½`` so that no run of rods lies
    along the double-curve preimage; here the preimage is not a constant-``v`` line at all, so
    ``0`` is used: it is both seam-compatible and measurably roomier (2.78 mm against 2.42 mm
    minimum clearance at this density).

    ``shear`` adds a ``cos v`` lean in ``u`` — it staggers the two sheets where they cross so
    their rods do not meet node-to-node. ``0`` is the plain grid.
    """
    if n % 2:
        raise ValueError("n must be even for the seam φ(v) = π − v to stay on the grid")
    i = np.arange(m + 1)[:, None]
    j = np.arange(n)[None, :]
    if equal_arclength:
        u = np.broadcast_to(arclength_u(m)[:, None], (m + 1, n)).astype(float).copy()
    else:
        u = np.broadcast_to(TWO_PI * i / m, (m + 1, n)).astype(float).copy()
    v = np.broadcast_to(TWO_PI * (j + offset) / n, (m + 1, n)).astype(float).copy()
    if shear:
        u += shear * (np.pi / m) * np.cos(v)
    return u, v


def apply_seam(u: np.ndarray, v: np.ndarray, offset: float = 0.0) -> None:
    """Tie the last u-column to the first through the flip, in place.

    ``u[m, j] = u[0, k(j)] + 2π`` and ``v[m, j] = π − v[0, k(j)]``, so that
    ``P(u[m, j], v[m, j]) = P(u[0, k(j)], v[0, k(j)])`` exactly (see :func:`extended`).
    """
    m, n = u.shape[0] - 1, u.shape[1]
    k = seam_index(n, offset)
    u[m, :] = u[0, k] + TWO_PI
    v[m, :] = np.pi - v[0, k]


def canonical_index(m: int, n: int, offset: float = 0.0) -> np.ndarray:
    """Map every ``(m + 1) × n`` grid slot onto its physical node (``m × n`` of them)."""
    out = np.empty((m + 1) * n, dtype=int)
    k = seam_index(n, offset)
    for i in range(m + 1):
        for j in range(n):
            out[i * n + j] = i * n + j if i < m else k[j]
    return out


def grid_edges(m: int, n: int, offset: float = 0.0) -> np.ndarray:
    """Quad-mesh edges after the seam folds one column away: ``2 m n`` rods."""
    def idx(i: int, j: int) -> int:
        return i * n + (j % n)

    edges = [(idx(i, j), idx(i + 1, j)) for i in range(m) for j in range(n)]
    edges += [(idx(i, j), idx(i, j + 1)) for i in range(m + 1) for j in range(n)]
    e = canonical_index(m, n, offset)[np.asarray(edges)]
    e = np.sort(e, axis=1)
    e = e[e[:, 0] != e[:, 1]]
    return np.unique(e, axis=0)


# ---------------------------------------------------------------------------------------
# Lattice: relaxation
# ---------------------------------------------------------------------------------------
def _closest_points(p0, p1, q0, q1):
    """Closest points between paired segments; returns ``(a, b, dist, s, t)``."""
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
    for _ in range(2):
        t = np.clip((b * s + f) / np.where(e > 1e-12, e, 1.0), 0.0, 1.0)
        s = np.clip((b * t - c) / np.where(a > 1e-12, a, 1.0), 0.0, 1.0)
    pa = p0 + s[:, None] * d1
    qb = q0 + t[:, None] * d2
    return pa, qb, np.linalg.norm(pa - qb, axis=1), s, t


def _jacobian(u: np.ndarray, v: np.ndarray, h: float = 1e-6):
    """``∂P/∂u`` and ``∂P/∂v`` at every slot, by central differences of the continuation."""
    pu = (extended(u + h, v) - extended(u - h, v)) / (2.0 * h)
    pv = (extended(u, v + h) - extended(u, v - h)) / (2.0 * h)
    return pu, pv


def _to_uv(force: np.ndarray, pu: np.ndarray, pv: np.ndarray):
    """Project 3-D displacements onto the surface's ``(u, v)`` parameters (least squares)."""
    a = np.einsum("ij,ij->i", pu, pu)
    b = np.einsum("ij,ij->i", pu, pv)
    c = np.einsum("ij,ij->i", pv, pv)
    fu = np.einsum("ij,ij->i", force, pu)
    fv = np.einsum("ij,ij->i", force, pv)
    det = a * c - b * b
    safe = np.where(np.abs(det) > 1e-12, det, 1.0)
    return (c * fu - b * fv) / safe, (a * fv - b * fu) / safe


def relax(
    m: int = M_DEFAULT,
    n: int = N_DEFAULT,
    *,
    iterations: int = 6000,
    gain: float = 0.9,
    target: float = RELAX_TARGET_MM,
    max_step: float = 0.05,
    spring: float = 0.005,
    shear: float = 0.0,
    offset: float = 0.0,
    equal_arclength: bool = False,
    verbose: bool = False,
):
    """Push nodes around in ``(u, v)`` until no two non-adjacent rods are too close.

    Identical in spirit to :func:`printable_surfaces.klein_lattice.relax` — repulsion in 3-D,
    projected back through the local Jacobian — with two differences forced by this
    parametrisation: the continuation :func:`extended` (so the derived ``u = 2π`` column can
    carry row-0's parameters) and the flip ``π − v`` instead of ``−v`` in the seam. Forces
    landing on the derived column are folded back onto their row-0 alias, so the seam does not
    swallow half of every push it receives.
    """
    u, v = initial_grid(m, n, offset=offset, shear=shear, equal_arclength=equal_arclength)
    u0, v0 = u.copy(), v.copy()
    edges = grid_edges(m, n, offset)
    canonical = canonical_index(m, n, offset)
    physical = np.arange(m * n)
    alias = canonical[m * n :]  # row m -> row 0 slot

    for it in range(iterations):
        apply_seam(u, v, offset)
        x = extended(u, v).reshape(-1, 3)
        xp = x[physical]
        p0, p1 = xp[edges[:, 0]], xp[edges[:, 1]]
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
            if verbose:
                print(f"  iter {it:3d}  clean")
            break

        pa, qb, dist, s_par, t_par = _closest_points(
            p0[cand[:, 0]], p1[cand[:, 0]], p0[cand[:, 1]], p1[cand[:, 1]]
        )
        close = dist < target
        cand, pa, qb = cand[close], pa[close], qb[close]
        s_par, t_par = s_par[close], t_par[close]
        dist = np.maximum(dist[close], 1e-6)
        if len(cand) == 0:
            if verbose:
                print(f"  iter {it:3d}  clean")
            break

        # `push` points from rod a toward rod b, so a recedes along −push and b along +push.
        direction = (qb - pa) / dist[:, None]
        push = 0.5 * gain * (target - dist)[:, None] * direction

        force = np.zeros_like(x)
        a0, a1 = edges[cand[:, 0], 0], edges[cand[:, 0], 1]
        b0, b1 = edges[cand[:, 1], 0], edges[cand[:, 1], 1]
        np.add.at(force, a0, -push * (1.0 - s_par)[:, None])
        np.add.at(force, a1, -push * s_par[:, None])
        np.add.at(force, b0, push * (1.0 - t_par)[:, None])
        np.add.at(force, b1, push * t_par[:, None])
        # Fold the derived column's share onto its row-0 alias.
        np.add.at(force, alias, force[m * n :])
        force = force[: m * n]

        pu, pv = _jacobian(u, v)
        pu, pv = pu.reshape(-1, 3), pv.reshape(-1, 3)
        du, dv = _to_uv(force, pu[physical], pv[physical])
        du = np.clip(du, -max_step, max_step).reshape(m, n)
        dv = np.clip(dv, -max_step, max_step).reshape(m, n)

        u[:m] += du - spring * (u[:m] - u0[:m])
        v[:m] += dv - spring * (v[:m] - v0[:m])
        apply_seam(u, v, offset)

        if verbose and (it % 25 == 0 or len(cand) == 0):
            print(f"  iter {it:3d}  close pairs {len(cand):4d}  min {dist.min():.3f} mm")

    apply_seam(u, v, offset)
    return u, v, edges


def build(m: int = M_DEFAULT, n: int = N_DEFAULT, **kwargs):
    """Relaxed nodes and edges, in mm (before the joint-sphere lift)."""
    u, v, edges = relax(m, n, **kwargs)
    x = extended(u, v).reshape(-1, 3)[: m * n]
    return x, edges, (u, v)


def degrees(nodes_or_count, edges: np.ndarray) -> np.ndarray:
    """Degree of every node."""
    n = int(nodes_or_count) if np.isscalar(nodes_or_count) else len(nodes_or_count)
    return np.bincount(np.asarray(edges, dtype=int).ravel(), minlength=n)


def components(n_nodes: int, edges: np.ndarray) -> int:
    """Number of connected components of the rod graph."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    e = np.asarray(edges, dtype=int)
    g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n_nodes, n_nodes))
    return int(connected_components(g, directed=False)[0])


# ---------------------------------------------------------------------------------------
# Distance of the rods to the double curve
# ---------------------------------------------------------------------------------------
def rods_near_curve(nodes: np.ndarray, edges: np.ndarray, curve_points: np.ndarray):
    """For every rod, the distance from its segment to the double-curve polyline.

    A rod that lies *on* the double curve scores ~0; that is the configuration that must be
    avoided, because there the two sheets coincide and a rod has a twin under its feet.
    """
    from scipy.spatial import cKDTree

    e = np.asarray(edges, dtype=int)
    p0, p1 = nodes[e[:, 0]], nodes[e[:, 1]]
    d = p1 - p0
    dd = np.einsum("ij,ij->i", d, d)
    thresh = float(np.sqrt(dd).max()) + 0.0
    tree = cKDTree(curve_points)
    idx = tree.query_ball_point(0.5 * (p0 + p1), thresh, return_sorted=False)
    best = np.full(len(e), np.inf)
    for k, cand in enumerate(idx):
        if not cand:
            continue
        q = curve_points[np.asarray(cand)]
        t = np.clip(((q - p0[k]) @ d[k]) / max(dd[k], 1e-12), 0.0, 1.0)
        best[k] = np.linalg.norm(q - (p0[k] + t[:, None] * d[k]), axis=1).min()
    return best


def rods_along_curve(nodes: np.ndarray, edges: np.ndarray, curve_points: np.ndarray, tol: float = 0.5):
    """Longest stretch of each rod that stays within ``tol`` mm of the double curve.

    Crossing the curve is unavoidable — it lies on the surface, so every rod that spans it
    does. *Running along* it is the failure mode: such a rod sits on top of its own image on
    the other sheet. This returns, per rod, the length of its longest near-touching run.
    """
    from scipy.spatial import cKDTree

    e = np.asarray(edges, dtype=int)
    p0, p1 = nodes[e[:, 0]], nodes[e[:, 1]]
    length = np.linalg.norm(p1 - p0, axis=1)
    k = 24
    ts = np.linspace(0.0, 1.0, k + 1)
    tree = cKDTree(curve_points)
    samples = p0[:, None, :] + ts[None, :, None] * (p1 - p0)[:, None, :]
    d = tree.query(samples.reshape(-1, 3))[0].reshape(len(e), len(ts))
    near = d < tol
    run = np.zeros(len(e))
    for i in range(len(e)):
        best = cur = 0
        for flag in near[i]:
            cur = cur + 1 if flag else 0
            best = max(best, cur)
        run[i] = (best - 1) / k * length[i] if best > 1 else 0.0
    return run


# ---------------------------------------------------------------------------------------
# The double curve
# ---------------------------------------------------------------------------------------
def double_curve(nu: int = 241, nv: int = 120, *, coarse: float = 2.0, verbose: bool = False):
    """Locate the preimage of the double curve, numerically.

    The whole surface is sampled on a ``(u, v)`` grid, and every pair of samples closer than
    ``coarse`` mm is a candidate unless it is a parameter-space neighbour or related by the
    seam ``(0, v) ≡ (2π, π − v)``. Each candidate is refined by least squares on all four
    parameters so that ``P(u₁, v₁) = P(u₂, v₂)`` to machine precision.

    Note there is no assumption that the two sheets meet: the search finds whatever the
    surface does. (It turns out the crossing is *within* the second sheet — see the module
    docs and :func:`double_curve_branch2`.)

    Returns a dict with the refined parameter pairs, the 3-D image, and the two traces.
    """
    from scipy.optimize import least_squares

    if nv % 2:
        raise ValueError("nv must be even")
    uu, vv = uv_grid(nu, nv)
    pts = bottle(uu.ravel(), vv.ravel())
    par = np.stack([uu.ravel(), vv.ravel()], axis=1)

    tree = cKDTree(pts)
    d, idx = tree.query(pts, k=24)

    def _cyc(a, b, period=TWO_PI):
        e = np.abs(a - b) % period
        return np.minimum(e, period - e)

    du = _cyc(par[:, None, 0], par[idx, 0])
    dv = _cyc(par[:, None, 1], par[idx, 1])
    dv_seam = _cyc(par[idx, 1], np.pi - par[:, None, 1])
    neighbour = (du < 0.35) & ((dv < 0.35) | (dv_seam < 0.35))
    cand_d = np.where(neighbour, np.inf, d)
    best = cand_d.min(axis=1)
    partner = idx[np.arange(len(idx)), np.argmin(cand_d, axis=1)]
    tight = np.where(np.isfinite(best) & (best < coarse))[0]
    if verbose:
        print(f"  coarse candidates: {len(tight)} of {len(pts)} samples")

    def resid(x):
        return bottle(x[0], x[1]) - bottle(x[2], x[3])

    lb = np.zeros(4)
    ub = np.full(4, TWO_PI)
    solved = []
    seen = set()
    tried = set()
    for k in tight:
        # Many samples straddle the same piece of the curve; one solve per coarse cell.
        cue = (round(par[k, 0] * 6), round(par[k, 1] * 6))
        if cue in tried:
            continue
        tried.add(cue)
        x0 = np.array([par[k, 0], par[k, 1], par[partner[k], 0], par[partner[k], 1]])
        sol = least_squares(
            resid, x0, bounds=(lb, ub), xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=400
        )
        if float(np.linalg.norm(sol.fun)) > 1e-6:
            continue
        x = sol.x
        if abs(x[0] - x[2]) < 0.05 and _cyc(x[1], x[3]) < 0.05:
            continue
        seam = (
            min(x[0], TWO_PI - x[0]) < 1e-3
            and min(x[2], TWO_PI - x[2]) < 1e-3
            and _cyc(x[1], np.pi - x[3]) < 0.05
        )
        if seam:
            continue
        key = (round(x[0], 2), round(x[1], 2), round(x[2], 2), round(x[3], 2))
        if key in seen:
            continue
        seen.add(key)
        solved.append(x)

    if not solved:
        return {"found": False, "pairs": np.zeros((0, 4))}

    x = np.asarray(solved)
    # Canonical order inside each pair: lower u first.
    swap = x[:, 0] > x[:, 2]
    x[swap] = x[swap][:, [2, 3, 0, 1]]
    pts = bottle(x[:, 0], x[:, 1])
    return {
        "found": True,
        "pairs": x,
        "points": pts,
        "radius": np.hypot(pts[:, 0], pts[:, 1]),
        "height": pts[:, 2],
        "u1": x[:, 0],
        "v1": np.mod(x[:, 1], TWO_PI),
        "u2": x[:, 2],
        "v2": np.mod(x[:, 3], TWO_PI),
        "u1_on_second_sheet": bool((x[:, 0] > np.pi).all()),
    }


def double_curve_branch2(nu: int = 721):
    """The double curve in closed form, from the structure of the second sheet.

    On ``π < u ≤ 2π`` the surface is a family of **horizontal** circles::

        P(u, v) = s ( A(u) − r(u) cos v,  r(u) sin v,  b − 16 sin u ),
        A(u) = 6 + 6 cos u (1 + sin u),   r(u) = 4 − 2 cos u.

    ``z`` sees only ``sin u``, so two values ``u`` and ``3π − u`` sit at the same height, with

        A(3π − u) = 12 − A(u),   r(3π − u) = 8 − r(u).

    The two circles lie in the same horizontal plane, so they meet in two points whenever
    ``|r₁ − r₂| < |A₁ − A₂| < r₁ + r₂`` — that is the double curve. Writing ``t = r₁ sin v₁``
    for the shared ``y``, the pair of equations reduces to a single scalar equation::

        w = 2A(u) − 12,      ±√(r₂² − t²) ∓ √(r₁² − t²) = w,

    with ``cos v = −(sign)·√(r² − t²)/r``. Scanning ``u`` and solving that by bisection
    returns the whole preimage curve, which is the exact curve the numeric search in
    :func:`double_curve` lands on.

    Returns arrays ``u1, v1, u2, v2`` and the 3-D image.
    """
    from scipy.optimize import brentq

    rows = []
    for u in np.linspace(np.pi, TWO_PI, nu):
        a = 6.0 + 6.0 * np.cos(u) * (1.0 + np.sin(u))
        r1 = 4.0 - 2.0 * np.cos(u)
        r2 = 8.0 - r1
        w_target = 2.0 * a - 12.0
        if abs(w_target) >= r1 + r2:
            continue
        # Circles at (A1, 0) and (A2, 0) in x, radii r1, r2. The common points have
        # |y| ≤ min(r1, r2); solve for y directly by circle-circle intersection.
        x1, x2 = a, 12.0 - a
        dx = x2 - x1
        if dx == 0.0:
            continue
        # x of the radical line: x = (d² + r1² − r2²)/(2d) measured from centre 1.
        d = abs(dx)
        if d >= r1 + r2 or d <= abs(r1 - r2):
            continue
        xr = (d * d + r1 * r1 - r2 * r2) / (2.0 * d)  # along the axis from centre 1
        h2 = r1 * r1 - xr * xr
        if h2 <= 0.0:
            continue
        h = np.sqrt(h2)
        sign = np.sign(dx)
        X = x1 + sign * xr
        for y in (h, -h):
            v1 = np.arctan2(y / r1, -(X - x1) / r1)
            v2 = np.arctan2(y / r2, -(X - x2) / r2)
            rows.append((u, np.mod(v1, TWO_PI), 3.0 * np.pi - u, np.mod(v2, TWO_PI)))
        _ = w_target  # kept for the derivation above; the circle intersection is equivalent
    if not rows:
        return {"found": False, "pairs": np.zeros((0, 4))}
    x = np.asarray(rows)
    pts = bottle(x[:, 0], x[:, 1])
    return {
        "found": True,
        "pairs": x,
        "points": pts,
        "radius": np.hypot(pts[:, 0], pts[:, 1]),
        "height": pts[:, 2],
        "u1": x[:, 0],
        "v1": x[:, 1],
        "u2": x[:, 2],
        "v2": x[:, 3],
    }
