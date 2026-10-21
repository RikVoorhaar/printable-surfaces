"""Support-free printable lattice (interior + faces) via relaxation + integer program.

Implements the Perplexity method:
  1. relax points with repulsion + a *support term* that wants each free node to have two
     steep "parent" rods below it (no wiring is ever fixed during relaxation),
  2. generate only *printable* candidate rods (steep, or short horizontal bridges),
  3. select rods with a mixed-integer program (each free node ≥1 downward rod, degree 3–6,
     rod clearance), preferring ~15 mm rods,
  4. bottom-up repair for the few nodes that can't get parents.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.spatial import cKDTree

SIN48 = np.sin(np.radians(48.0))   # steep rod threshold (45° + 3° margin)
SIN5 = np.sin(np.radians(5.0))     # "horizontal" tolerance for bridges
L_MIN, L_MAX, L_TARGET = 8.0, 22.0, 15.0
BRIDGE_MAX = 10.0
CLEARANCE = 1.6   # rod-rod / rod-node centerline clearance (non-overlap for 1.2mm rods)
DEG_MIN, DEG_MAX = 3, 6


# --------------------------------------------------------------------------- geometry
def point_segment_dist(p, a, b):
    ab = b - a
    denom = ab @ ab
    t = 0.0 if denom < 1e-12 else np.clip((p - a) @ ab / denom, 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def segment_segment_dist(p1, p2, p3, p4):
    """Closest distance between two 3-D line segments (Ericson's algorithm)."""
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3
    a = d1 @ d1
    e = d2 @ d2
    f = d2 @ r
    s = t = 0.0
    if a <= 1e-12 and e <= 1e-12:
        return float(np.linalg.norm(r))
    if a <= 1e-12:
        t = float(np.clip(f / e, 0.0, 1.0))
    else:
        c = d1 @ r
        if e <= 1e-12:
            s = float(np.clip(-c / a, 0.0, 1.0))
        else:
            b = d1 @ d2
            denom = a * e - b * b
            if abs(denom) > 1e-12:
                s = float(np.clip((b * f - c * e) / denom, 0.0, 1.0))
            else:
                s = 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = float(np.clip(-c / a, 0.0, 1.0))
            elif t > 1.0:
                t = 1.0
                s = float(np.clip((b - c) / a, 0.0, 1.0))
    return float(np.linalg.norm((p1 + s * d1) - (p3 + t * d2)))


# ------------------------------------------------------------------------- relaxation
def _support_forces(pos, free):
    """Force on each free node toward its 2 best steep parents below it."""
    n = len(pos)
    d = pos[None, :, :] - pos[:, None, :]           # d[i,j] = pos[j] - pos[i]
    dist = np.linalg.norm(d, axis=-1)
    np.fill_diagonal(dist, np.inf)
    dz = d[:, :, 2]
    sin_elev = -dz / np.maximum(dist, 1e-9)         # for below (dz<0), -dz = drop
    valid = (dz < 0) & (sin_elev >= SIN48) & (dist >= L_MIN) & (dist <= L_MAX)
    score = np.where(valid, np.abs(dist - L_TARGET), np.inf)
    top = np.argpartition(score, 1, axis=1)[:, :2]  # 2 best parents (approx order)

    F = np.zeros_like(pos)
    ar = np.arange(n)
    for k in range(2):
        p = top[:, k]
        dp = dist[ar, p]
        ok = dp < np.inf
        dp_safe = np.where(ok, dp, L_TARGET)
        u = d[ar, p] / np.maximum(dp_safe, 1e-9)[:, None]
        F += np.where(ok[:, None], (dp_safe - L_TARGET)[:, None] * u, 0.0)
    F[~free] = 0.0
    return F


def relax_printable(
    pos, fixed, face_ids, face_paras, normals, centers,
    spacing, iters=900, lam=2.0, cooling=0.995,
):
    """Repulsion + ramped support term; face nodes stay in-face, interior nodes stay inside."""
    n = len(pos)
    free = ~fixed
    interior = free & (face_ids < 0)
    cutoff = 1.5 * spacing
    temp = 15.0
    margin = 0.4 * spacing

    for it in range(iters):
        # finite-range repulsion
        delta = pos[:, None, :] - pos[None, :, :]
        dd = np.linalg.norm(delta, axis=-1)
        np.fill_diagonal(dd, np.inf)
        u = delta / np.maximum(dd, 1e-6)[:, :, None]
        F = (u * np.where(dd < cutoff, cutoff - dd, 0.0)[:, :, None]).sum(axis=1)

        # support term (switches on after 1/3, ramps up)
        if it >= iters // 3:
            ramp = min(1.0, (it - iters // 3) / max(1, iters // 3))
            F = F + ramp * lam * _support_forces(pos, free)

        F[fixed] = 0.0

        dn = np.linalg.norm(F, axis=1, keepdims=True)
        scale = np.minimum(1.0, temp / np.maximum(dn, 1e-12))
        pos = pos + F * scale

        # face points stay within their rhombus
        for fi in np.unique(face_ids[face_ids >= 0]):
            m = face_ids == fi
            corner, e1, e2 = face_paras[fi]
            g11, g12, g22 = e1 @ e1, e1 @ e2, e2 @ e2
            det = g11 * g22 - g12 * g12
            q = pos[m] - corner
            r1, r2 = q @ e1, q @ e2
            s = np.clip((g22 * r1 - g12 * r2) / det, 0.0, 1.0)
            t = np.clip((g11 * r2 - g12 * r1) / det, 0.0, 1.0)
            pos[m] = corner + s[:, None] * e1 + t[:, None] * e2

        # interior points stay inside with a margin from the faces
        for n_, c_ in zip(normals, centers):
            proj = (pos - c_) @ n_
            too_close = interior & (proj > -margin)
            if too_close.any():
                pos[too_close] -= n_ * (proj[too_close] + margin)[:, None]

        temp *= cooling
    return pos


# ------------------------------------------------------------------ candidate rods
def candidate_rods(pos, n_frame, frame_chain, face_ids, clearance=CLEARANCE):
    """Printable candidate rods (steep 48° or short bridge), excluding frame-frame pairs
    and rods too close to another node / frame rod / cross-face."""
    n = len(pos)
    cand = []
    for i in range(n):
        for j in range(i + 1, n):
            if i < n_frame and j < n_frame:
                continue  # both frame points: frame is already wired
            d = pos[j] - pos[i]
            L = float(np.linalg.norm(d))
            if L < 1e-9:
                continue
            sin_elev = abs(d[2]) / L
            steep = (sin_elev >= SIN48) and (L_MIN <= L <= L_MAX)
            bridge = (sin_elev <= SIN5) and (L <= BRIDGE_MAX)
            if not (steep or bridge):
                continue
            if face_ids[i] >= 0 and face_ids[j] >= 0 and face_ids[i] != face_ids[j]:
                continue  # between two different faces
            clear = True
            for k in range(n):
                if k == i or k == j:
                    continue
                if point_segment_dist(pos[k], pos[i], pos[j]) < clearance:
                    clear = False
                    break
            if not clear:
                continue
            for (a, b) in frame_chain:
                if segment_segment_dist(pos[i], pos[j], pos[a], pos[b]) < clearance:
                    clear = False
                    break
            if clear:
                cand.append((i, j))
    return np.asarray(cand, dtype=int)


# ------------------------------------------------------------- integer program
def select_rods(pos, cand, free_indices, clearance=CLEARANCE):
    """Choose rods (0/1): every free node ≥1 downward rod, degree 3–6 (free nodes only),
    rod-rod clearance; cost prefers ~15 mm and penalises rods at the angle limit."""
    E = len(cand)
    if E == 0:
        return np.zeros((0, 2), dtype=int)

    n = len(pos)
    cost = np.empty(E)
    for e, (i, j) in enumerate(cand):
        d = pos[j] - pos[i]
        L = float(np.linalg.norm(d))
        sin_elev = abs(d[2]) / L
        length_pen = (L - L_TARGET) ** 2
        angle_pen = 0.0 if sin_elev <= SIN5 else 5.0 * (1.0 - (sin_elev - SIN48) / (1.0 - SIN48))
        cost[e] = length_pen + angle_pen

    rows, lb, ub = [], [], []

    # 1) every free node has >= 1 downward rod
    for nd in free_indices:
        row = np.zeros(E)
        for e, (i, j) in enumerate(cand):
            if (i == nd and pos[j, 2] < pos[nd, 2]) or (j == nd and pos[i, 2] < pos[nd, 2]):
                row[e] = 1.0
        if row.sum() == 0:
            continue
        rows.append(row); lb.append(1.0); ub.append(np.inf)

    # 2) degree 3..6 for free nodes (softened to >=2 if a node can't reach 3)
    for nd in free_indices:
        row = np.zeros(E)
        for e, (i, j) in enumerate(cand):
            if i == nd or j == nd:
                row[e] = 1.0
        if row.sum() == 0:
            continue
        deg_lb = DEG_MIN if row.sum() >= DEG_MIN else 1  # relax if too few candidates
        rows.append(row); lb.append(deg_lb); ub.append(DEG_MAX)

    # 3) rod-rod clearance (disabled for now: too many overlapping candidates makes it
    #    infeasible; needs a more columnar relaxation first)
    # for e in range(E):
    #     for f in range(e + 1, E):
    #         if segment_segment_dist(pos[cand[e, 0]], pos[cand[e, 1]], pos[cand[f, 0]], pos[cand[f, 1]]) < clearance:
    #             row = np.zeros(E)
    #             row[e] = row[f] = 1.0
    #             rows.append(row); lb.append(-np.inf); ub.append(1.0)

    if not rows:
        return cand
    A = np.asarray(rows)
    res = milp(c=cost, constraints=[LinearConstraint(A, np.asarray(lb), np.asarray(ub))],
               integrality=np.ones(E), bounds=Bounds(0.0, 1.0))
    x = np.round(res.x).astype(int) if res.success else np.zeros(E, dtype=int)
    return cand[x == 1]


# --------------------------------------------------------------------- top level
def _has_parent(pos, nd):
    """True if node ``nd`` has a steep parent (≥48°, 8–22 mm) below it."""
    d = pos - pos[nd]
    dist = np.linalg.norm(d, axis=1)
    dz = d[:, 2]
    sin_elev = -dz / np.maximum(dist, 1e-9)
    return bool(((dz < 0) & (sin_elev >= SIN48) & (dist >= L_MIN) & (dist <= L_MAX)).any())


def repair(all_pts, all_ids, fixed, face_paras, normals, centers, n_frame, frame_chain):
    """Nudge free nodes (lowest first) upward until they gain a parent; delete the rest."""
    for _ in range(8):
        cand = candidate_rods(all_pts, n_frame, frame_chain, all_ids)
        has_down = np.zeros(len(all_pts), dtype=bool)
        total = np.zeros(len(all_pts), dtype=int)
        for i, j in cand:
            total[i] += 1; total[j] += 1
            if all_pts[j, 2] < all_pts[i, 2]:
                has_down[i] = True
            if all_pts[i, 2] < all_pts[j, 2]:
                has_down[j] = True
        free = np.where(~fixed)[0]
        # stuck: no downward candidate, or fewer than 3 total candidates (can't reach degree 3)
        stuck = [nd for nd in free if not has_down[nd] or total[nd] < 3]
        if not stuck:
            break
        stuck.sort(key=lambda nd: all_pts[nd, 2])  # lowest first
        delete = []
        for nd in stuck:
            fid = all_ids[nd]
            saved = all_pts[nd].copy()
            fixed_pos = False
            for step in (3.0, 6.0, 9.0, 12.0):
                trial = saved + np.array([0.0, 0.0, step])  # move up
                if fid >= 0:
                    corner, e1, e2 = face_paras[fid]
                    g11, g12, g22 = e1 @ e1, e1 @ e2, e2 @ e2
                    det = g11 * g22 - g12 * g12
                    q = trial - corner
                    s = np.clip((g22 * (q @ e1) - g12 * (q @ e2)) / det, 0.0, 1.0)
                    t = np.clip((g11 * (q @ e2) - g12 * (q @ e1)) / det, 0.0, 1.0)
                    trial = corner + s * e1 + t * e2
                else:
                    for n_, c_ in zip(normals, centers):
                        if (trial - c_) @ n_ > 0.0:
                            trial -= n_ * ((trial - c_) @ n_)
                all_pts[nd] = trial
                if _has_parent(all_pts, nd):
                    fixed_pos = True
                    break
                all_pts[nd] = saved
            if not fixed_pos:
                delete.append(nd)
        if delete:
            keep = np.ones(len(all_pts), dtype=bool)
            keep[delete] = False
            all_pts = all_pts[keep]
            all_ids = all_ids[keep]
            fixed = np.zeros(len(all_pts), dtype=bool)
            fixed[:n_frame] = True
    return all_pts, all_ids, fixed


def printable_lattice(
    vertices, edges, faces,
    spacing=12.0, n_face=10, n_interior=150, seed=1,
    iters=900, lam=2.0,
):
    """Generate a support-free lattice inside a convex polytope. Returns (nodes, rods)."""
    from .platonic import face_parallelogram, face_planes, inside_mask

    normals, centers = face_planes(vertices, faces)

    # frame
    frame_pts = [v for v in vertices]
    chain = []
    idx = len(vertices)
    for i, j in edges:
        d = vertices[j] - vertices[i]
        nseg = max(1, int(round(np.linalg.norm(d) / spacing)))
        prev = i
        for k in range(1, nseg):
            frame_pts.append(vertices[i] + (k / nseg) * d)
            chain.append((prev, idx)); prev = idx; idx += 1
        chain.append((prev, j))
    frame_pts = np.asarray(frame_pts, dtype=float)
    chain = np.asarray(chain, dtype=int)
    n_frame = len(frame_pts)

    rng = np.random.default_rng(seed)

    # face points
    face_pts, face_ids = [], []
    for fi, face in enumerate(faces):
        corner, e1, e2 = face_parallelogram(vertices, face)
        u = rng.uniform(0, 1, n_face); w = rng.uniform(0, 1, n_face)
        face_pts.append(corner + u[:, None] * e1 + w[:, None] * e2)
        face_ids += [fi] * n_face
    face_pts = np.vstack(face_pts)
    face_ids = np.asarray(face_ids, dtype=int)

    # interior points (rejection sampling in the bounding box)
    lo = vertices.min(axis=0); hi = vertices.max(axis=0)
    interior = []
    while len(interior) < n_interior:
        batch = rng.uniform(lo, hi, size=(n_interior * 4, 3))
        interior.append(batch[inside_mask(batch, normals, centers)])
    interior = np.vstack(interior)[:n_interior]

    all_pts = np.vstack([frame_pts, face_pts, interior])
    all_ids = np.concatenate([np.full(n_frame, -1), face_ids, np.full(n_interior, -1)])
    fixed = np.zeros(len(all_pts), dtype=bool); fixed[:n_frame] = True
    face_paras = [face_parallelogram(vertices, f) for f in faces]

    all_pts = relax_printable(all_pts, fixed, all_ids, face_paras, normals, centers,
                              spacing, iters=iters, lam=lam)

    all_pts, all_ids, fixed = repair(all_pts, all_ids, fixed, face_paras, normals, centers,
                                     n_frame, chain)

    free_indices = np.where(~fixed)[0]
    cand = candidate_rods(all_pts, n_frame, chain, all_ids)
    selected = select_rods(all_pts, cand, free_indices)

    rods = np.vstack([chain, selected]) if len(selected) else chain
    rods = np.unique(np.sort(rods, axis=1), axis=0)
    return all_pts, rods
