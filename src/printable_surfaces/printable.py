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
from scipy.spatial import Delaunay, cKDTree

from .platonic import clamp_to_polygon, face_geometry, sample_in_polygon, shrink_polygon

SIN48 = np.sin(np.radians(48.0))   # steep rod threshold (45° + 3° margin)
SIN5 = np.sin(np.radians(5.0))     # "horizontal" tolerance for bridges
L_MIN, L_MAX, L_TARGET = 8.0, 28.0, 15.0
BRIDGE_MAX = 10.0
CLEARANCE = 2.6   # rod-rod axis-to-axis minimum (gap of 1.4 mm between 1.2 mm rods)
NODE_CLEARANCE = 2.0  # rod-axis-to-node-center minimum (0.6 rod + 0.9 node + 0.5 gap)
DEG_MIN, DEG_MAX = 3, 6

ALPHA_DEG = 48.0      # steep rod threshold in degrees
BETA = 0.6            # per-rod reward in the MIP cost (raises degree)
GAMMA = 0.3           # bonus for rods lying in a face plane
H = 15.0              # target node spacing


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
def _allowed_parents(face_ids, normals, centers, n_frame, frame_on_face):
    """Boolean (n,n): j is an *allowed* parent of i (Section 4). Constant during relaxation."""
    n = len(face_ids)
    is_frame = np.zeros(n, dtype=bool); is_frame[:n_frame] = True
    is_face = face_ids >= 0
    is_interior = ~is_frame & ~is_face
    overhang = np.zeros(n, dtype=bool)
    for i in np.where(is_face)[0]:
        overhang[i] = normals[face_ids[i], 2] < 0.0

    allowed = np.zeros((n, n), dtype=bool)
    allowed[is_interior] = True                      # I -> any node
    for i in np.where(is_face)[0]:
        fi = face_ids[i]
        same = is_face & (face_ids == fi)
        on_frame = np.zeros(n, dtype=bool)
        on_frame[:n_frame] = frame_on_face[:, fi]
        allowed[i] = same | on_frame
        if not overhang[i]:
            allowed[i] |= is_interior                # F↑ may also use I
    np.fill_diagonal(allowed, False)
    return allowed, overhang, is_frame, is_face, is_interior


def _psi_grad(pos):
    """Pairwise ψ (Section 5.2) and its gradient pieces.

    ψ(p,q) = (sinα·L − Δz)₊² + (L−Lmax)₊² + (Lmin−L)₊² with p the child, q the parent.
    Returns (psi, L, d, deltaz)."""
    n = len(pos)
    d = pos[None, :, :] - pos[:, None, :]      # d[i,j] = pos[j] - pos[i]
    L = np.linalg.norm(d, axis=-1)
    np.fill_diagonal(L, np.inf)
    deltaz = -d[:, :, 2]                        # z_i - z_j (>0 => parent below)
    c = np.clip(SIN48 * L - deltaz, 0.0, None)
    psi = c * c + np.clip(L - L_MAX, 0.0, None) ** 2 + np.clip(L_MIN - L, 0.0, None) ** 2
    return psi, L, d, deltaz


def _support_forces(pos, allowed, is_frame, lam_t):
    """ψ-based symmetric support force (Section 5.2) over restricted parent sets."""
    n = len(pos)
    psi, L, d, deltaz = _psi_grad(pos)
    psi = np.where(allowed & (deltaz > 0), psi, np.inf)

    top = np.argpartition(psi, 1, axis=1)[:, :2]
    ar = np.arange(n)
    zhat = np.array([0.0, 0.0, 1.0])

    F = np.zeros_like(pos)
    for k in range(2):
        p = top[:, k]
        Lk = L[ar, p]
        ok = np.isfinite(Lk) & (Lk > 0.0)
        Ls = np.where(ok, Lk, L_TARGET)
        u = np.where(ok[:, None], (pos[ar] - pos[p]) / np.maximum(Ls, 1e-9)[:, None], 0.0)
        ck = np.clip(SIN48 * Ls - deltaz[ar, p], 0.0, None)
        g = (2.0 * ck[:, None] * (SIN48 * u - zhat)
             + 2.0 * np.clip(Ls - L_MAX, 0.0, None)[:, None] * u
             - 2.0 * np.clip(L_MIN - Ls, 0.0, None)[:, None] * u)
        g = np.where(ok[:, None], g, 0.0)
        F += -g                           # child force = −∂ψ/∂p
        np.add.at(F, p, g)                # parent force = +∂ψ/∂p (symmetric)
    F[is_frame] = 0.0
    return lam_t * F


def _anchor_forces(pos, allowed, overhang, is_frame, is_face, is_interior, normals, face_ids, lam_t):
    """Anchor (Section 5.3): pull one interior node under each F↑ non-vertical face node."""
    n = len(pos)
    up_face = is_face & ~overhang
    non_vertical = np.zeros(n, dtype=bool)
    for i in np.where(up_face)[0]:
        non_vertical[i] = abs(normals[face_ids[i], 2]) < 0.99
    targets = np.where(up_face & non_vertical)[0]
    if len(targets) == 0:
        return np.zeros_like(pos)

    psi, L, d, deltaz = _psi_grad(pos)
    zhat = np.array([0.0, 0.0, 1.0])
    F = np.zeros_like(pos)
    iq = np.where(is_interior)[0]
    for i in targets:
        if len(iq) == 0:
            continue
        q = iq[int(np.argmin(psi[i, iq]))]
        Lk = L[i, q]
        if not np.isfinite(Lk) or Lk <= 0.0:
            continue
        u = (pos[i] - pos[q]) / Lk
        ck = np.clip(SIN48 * Lk - deltaz[i, q], 0.0, None)
        g = (2.0 * ck * (SIN48 * u - zhat)
             + 2.0 * np.clip(Lk - L_MAX, 0.0, None) * u
             - 2.0 * np.clip(L_MIN - Lk, 0.0, None) * u)
        F[i] += -g
        F[q] += g
    return lam_t * F


def _parent_matrix(pos, allowed):
    """Boolean (n,n): j is a printable parent of i (steep/bridge + allowed)."""
    n = len(pos)
    d = pos[None, :, :] - pos[:, None, :]
    dist = np.linalg.norm(d, axis=-1)
    np.fill_diagonal(dist, np.inf)
    dz = d[:, :, 2]
    sin_elev = -dz / np.maximum(dist, 1e-9)
    below = dz < 0
    steep = (sin_elev >= SIN48) & (dist >= L_MIN) & (dist <= L_MAX)
    bridge = (sin_elev <= SIN5) & (dist <= BRIDGE_MAX)
    return (below & (steep | bridge)) & allowed


def _count_parents(pos, i, allowed_row):
    """Number of printable parents of node i (O(n)), given its allowed-parent row."""
    d = pos - pos[i]
    dist = np.linalg.norm(d, axis=1)
    dz = d[:, 2]
    sin_elev = -dz / np.maximum(dist, 1e-9)
    below = dz < 0
    steep = (sin_elev >= SIN48) & (dist >= L_MIN) & (dist <= L_MAX)
    bridge = (sin_elev <= SIN5) & (dist <= BRIDGE_MAX)
    return int(((below & (steep | bridge)) & allowed_row).sum())


def _face_project(geom: dict, pts3d: np.ndarray) -> np.ndarray:
    """Project 3-D points onto a face's plane and clamp them to its inset polygon."""
    pts3d = np.asarray(pts3d, dtype=float)
    basis = np.stack([geom["u"], geom["v"]], axis=1)  # (3, 2)
    p2d = (pts3d - geom["origin"]) @ basis
    out = np.empty_like(pts3d)
    for k in range(len(pts3d)):
        q = clamp_to_polygon(geom["inset"], p2d[k])
        out[k] = geom["origin"] + q[0] * geom["u"] + q[1] * geom["v"]
    return out


def _project(pos, fixed, face_ids, face_geoms, normals, centers, margin):
    """Project nodes back onto their class region (face plane inset 0.35h / interior inset)."""
    pos = pos.copy()
    is_interior = (~fixed) & (face_ids < 0)
    for fi in np.unique(face_ids[face_ids >= 0]):
        m = face_ids == fi
        pos[m] = _face_project(face_geoms[fi], pos[m])
    for n_, c_ in zip(normals, centers):
        proj = (pos - c_) @ n_
        too_close = is_interior & (proj > -margin)
        if too_close.any():
            pos[too_close] -= n_ * (proj[too_close] + margin)[:, None]
    return pos


def relax_printable(
    pos, fixed, face_ids, face_geoms, normals, centers, n_frame, frame_on_face,
    spacing, iters=900, lam=1.0, kappa=1.25, seed=None, anchor_scale=1.0, r_c_factor=1.25,
    same_face_boost=1.0, rep_power=2.0,
):
    """Sliding-class relaxation (Sections 5–6): quadratic repulsion (+ κ anisotropy on I
    pairs), ψ support, anchor, and a hard final phase that locks in ≥2 parents."""
    n = len(pos)
    free = ~fixed
    is_face = face_ids >= 0
    is_interior = free & ~is_face
    r_c = r_c_factor * spacing
    step_clip = 0.08 * spacing
    jitter0 = 0.03 * spacing
    margin = 0.4 * spacing
    rng = np.random.default_rng(seed if seed is not None else 0)

    allowed, overhang, is_frame, _, _ = _allowed_parents(
        face_ids, normals, centers, n_frame, frame_on_face)

    for it in range(iters):
        lam_t = 0.0
        if it >= iters // 3:
            lam_t = lam * 2.0 * min(1.0, (it - iters // 3) / max(1, iters // 3))

        # repulsion: quadratic, vertical anisotropy κ on pairs involving an interior node
        delta = pos[:, None, :] - pos[None, :, :]
        dd = np.linalg.norm(delta, axis=-1)
        np.fill_diagonal(dd, np.inf)
        mix = is_interior[:, None] | is_interior[None, :]
        dz = delta[:, :, 2]
        d_eff = np.where(mix, np.sqrt(delta[:, :, 0] ** 2 + delta[:, :, 1] ** 2 + (dz / kappa) ** 2), dd)
        r = np.clip(r_c - d_eff, 0.0, None)
        # boost same-face repulsion so face nodes spread uniformly across their face
        same_face_pair = is_face[:, None] & (face_ids[None, :] == face_ids[:, None])
        r = r * np.where(same_face_pair, same_face_boost, 1.0)
        # force magnitude = rep_power * r^(rep_power-1)  (potential is r^rep_power)
        mag = rep_power * np.power(np.maximum(r, 0.0), rep_power - 1.0)
        denom = np.maximum(d_eff, 1e-9)
        F = np.zeros_like(pos)
        F[:, 0] += (mag * delta[:, :, 0] / denom).sum(axis=1)
        F[:, 1] += (mag * delta[:, :, 1] / denom).sum(axis=1)
        F[:, 2] += (mag * np.where(mix, dz / kappa ** 2, dz) / denom).sum(axis=1)

        # support + anchor
        if it >= iters // 3:
            F += _support_forces(pos, allowed, is_frame, lam_t)
            F += anchor_scale * _anchor_forces(pos, allowed, overhang, is_frame, is_face, is_interior, normals, face_ids, lam_t)

        F[fixed] = 0.0

        step = 0.5 * F
        dn = np.linalg.norm(step, axis=1, keepdims=True)
        step = step * np.minimum(1.0, step_clip / np.maximum(dn, 1e-12))
        step = step + jitter0 * (1.0 - it / iters) * rng.standard_normal(pos.shape)
        step[fixed] = 0.0

        if it >= 2 * iters // 3:
            # hard phase: a move may not drop a node (that already has ≥2 parents) below 2
            for i in range(n):
                if fixed[i]:
                    continue
                d_i = step[i]
                old = _count_parents(pos, i, allowed[i])
                placed = False
                for _attempt in range(5):
                    trial = _project(pos[i:i + 1] + d_i[None, :], np.array([False]),
                                     face_ids[i:i + 1], face_geoms, normals, centers, margin)[0]
                    trial_pos = pos.copy(); trial_pos[i] = trial
                    if old < 2 or _count_parents(trial_pos, i, allowed[i]) >= 2:
                        pos[i] = trial
                        placed = True
                        break
                    d_i = d_i * 0.5
                if not placed:
                    pos[i] = _project(pos[i:i + 1], np.array([False]),
                                      face_ids[i:i + 1], face_geoms, normals, centers, margin)[0]
        else:
            pos = pos + step
            pos = _project(pos, fixed, face_ids, face_geoms, normals, centers, margin)

    return pos


# ------------------------------------------------------------------ candidate rods
def _node_classes(n, n_frame, face_ids, normals):
    """Classify nodes: frame (fixed), F↓ (overhanging face), F↑ (other face), I (interior)."""
    is_frame = np.zeros(n, dtype=bool); is_frame[:n_frame] = True
    is_face = face_ids >= 0
    is_interior = ~is_frame & ~is_face
    overhang = np.zeros(n, dtype=bool)
    for i in np.where(is_face)[0]:
        overhang[i] = normals[face_ids[i], 2] < 0.0
    return is_frame, is_face, is_interior, overhang


def _frame_on_faces(frame_pts, normals, centers, tol=1.0):
    """Boolean (n_frame, n_faces): frame node k lies on face fi (on its boundary plane)."""
    n_faces = len(normals)
    mask = np.zeros((len(frame_pts), n_faces), dtype=bool)
    for fi in range(n_faces):
        mask[:, fi] = np.abs((frame_pts - centers[fi]) @ normals[fi]) < tol
    return mask


def candidate_rods(pos, n_frame, frame_chain, face_ids, normals, centers, clearance=CLEARANCE):
    """Printable candidate rods (steep 48° or short bridge), excluding frame-frame pairs,
    cross-face pairs, rods too close to another node/frame rod, and — for F↓ (overhanging
    face) nodes — interior/off-face parents (Section 4 of the method)."""
    n = len(pos)
    _, is_face, _, overhang = _node_classes(n, n_frame, face_ids, normals)
    frame_on_face = _frame_on_faces(pos[:n_frame], normals, centers)
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
            if is_face[i] and is_face[j] and face_ids[i] != face_ids[j]:
                continue  # between two different faces
            # F↓ nodes: parents must be same-face F or frame nodes on the same face (not I)
            if overhang[i]:
                j_ok = (face_ids[j] == face_ids[i]) or (j < n_frame and frame_on_face[j, face_ids[i]])
                if not j_ok:
                    continue
            if overhang[j]:
                i_ok = (face_ids[i] == face_ids[j]) or (i < n_frame and frame_on_face[i, face_ids[j]])
                if not i_ok:
                    continue
            clear = True
            for k in range(n):
                if k == i or k == j:
                    continue
                if point_segment_dist(pos[k], pos[i], pos[j]) < NODE_CLEARANCE:
                    clear = False
                    break
            if not clear:
                continue
            for (a, b) in frame_chain:
                if a == i or a == j or b == i or b == j:
                    continue  # shares an endpoint: not a clearance conflict
                if segment_segment_dist(pos[i], pos[j], pos[a], pos[b]) < clearance:
                    clear = False
                    break
            if clear:
                cand.append((i, j))
    return np.asarray(cand, dtype=int)


# ------------------------------------------------------------- integer program
def select_rods(pos, cand, frame_chain, face_ids, normals, centers, n_frame, clearance=CLEARANCE):
    """Choose rods with an integer program (Section 7 of the method).

    Cost carries a per-rod reward β so rods are added until degree/clearance stop them;
    every lower bound has a slack so the LP is never infeasible; rod-rod clearance pairs
    that *share a node* are skipped (they are not conflicts). The sf/sa constraints make
    the face lattice visible (≥2 in-face parents per F node, ≥1 interior parent per F↑)."""
    E = len(cand)
    n = len(pos)
    empty = {"s1": np.zeros(n), "s2": np.zeros(n), "sd": np.zeros(n), "sf": np.zeros(n), "sa": np.zeros(n)}
    if E == 0:
        return np.zeros((0, 2), dtype=int), empty

    is_frame = np.zeros(n, dtype=bool); is_frame[:n_frame] = True
    is_face = face_ids >= 0
    is_interior = ~is_frame & ~is_face
    overhang = np.zeros(n, dtype=bool)
    for i in np.where(is_face)[0]:
        overhang[i] = normals[face_ids[i], 2] < 0.0
    up_nonvertical = np.zeros(n, dtype=bool)
    for i in np.where(is_face & ~overhang)[0]:
        up_nonvertical[i] = abs(normals[face_ids[i], 2]) < 0.99
    frame_on_face = _frame_on_faces(pos[:n_frame], normals, centers)

    # rod properties
    Lc = np.linalg.norm(pos[cand[:, 1]] - pos[cand[:, 0]], axis=1)
    dzc = np.abs(pos[cand[:, 1], 2] - pos[cand[:, 0], 2])
    sin_elev = dzc / np.maximum(Lc, 1e-9)
    theta_deg = np.degrees(np.arcsin(np.clip(sin_elev, 0.0, 1.0)))
    same_face = np.zeros(E, dtype=bool)
    for e, (i, j) in enumerate(cand):
        fi, fj = face_ids[i], face_ids[j]
        same_face[e] = ((fi >= 0 and fi == fj)
                        or (fi >= 0 and j < n_frame and frame_on_face[j, fi])
                        or (fj >= 0 and i < n_frame and frame_on_face[i, fj]))

    # angle term only for steep rods (θ ≥ α); for a bridge (θ ≈ 0°) it would blow up to
    # ~48k and wrongly forbid every horizontal rod.
    angle_pen = np.where(sin_elev >= SIN48, 0.3 * np.exp(-(theta_deg - ALPHA_DEG) / 4.0), 0.0)
    cost = ((Lc - H) / H) ** 2 - BETA + angle_pen - GAMMA * same_face

    # 5 slack blocks: s1 (≥1 down), s2 (≥2 down), sd (degree≥3), sf (≥2 in-face), sa (≥1 to-I)
    nslack = 5 * n
    total = E + nslack
    C = np.zeros(total)
    C[:E] = cost
    C[E:E + n] = 1000.0        # s1
    C[E + n:E + 2 * n] = 8.0   # s2
    C[E + 2 * n:E + 3 * n] = 5.0   # sd
    C[E + 3 * n:E + 4 * n] = 50.0  # sf
    C[E + 4 * n:E + 5 * n] = 20.0  # sa
    integrality = np.zeros(total)
    integrality[:E] = 1

    # frame degree (chain rods already count toward degree)
    frame_deg = np.zeros(n)
    for a, b in frame_chain:
        frame_deg[a] += 1
        frame_deg[b] += 1

    # per-candidate masks
    down_mask = np.zeros((n, E), dtype=bool)
    inc_mask = np.zeros((n, E), dtype=bool)
    inface_down = np.zeros((n, E), dtype=bool)
    toI_down = np.zeros((n, E), dtype=bool)
    for e, (i, j) in enumerate(cand):
        if pos[j, 2] < pos[i, 2]:
            down_mask[i, e] = True
            if is_face[i]:
                if (face_ids[j] == face_ids[i]) or (j < n_frame and frame_on_face[j, face_ids[i]]):
                    inface_down[i, e] = True
                if is_interior[j]:
                    toI_down[i, e] = True
        if pos[i, 2] < pos[j, 2]:
            down_mask[j, e] = True
            if is_face[j]:
                if (face_ids[i] == face_ids[j]) or (i < n_frame and frame_on_face[i, face_ids[j]]):
                    inface_down[j, e] = True
                if is_interior[i]:
                    toI_down[j, e] = True
        inc_mask[i, e] = inc_mask[j, e] = True

    rows, lb, ub = [], [], []

    def add(nd, slack_off, mask, rhs):
        row = np.zeros(total)
        row[:E] = mask[nd]
        row[E + slack_off + nd] = 1.0
        rows.append(row); lb.append(rhs); ub.append(np.inf)

    for nd in range(n):
        if not is_frame[nd]:
            add(nd, 0, down_mask, 1.0)      # s1: ≥ 1 downward rod
            add(nd, n, down_mask, 2.0)      # s2: ≥ 2 downward rods
        if is_face[nd]:
            add(nd, 3 * n, inface_down, 2.0)  # sf: ≥ 2 in-face rods
        if up_nonvertical[nd]:
            add(nd, 4 * n, toI_down, 1.0)   # sa: ≥ 1 rod to interior
        need = max(0.0, DEG_MIN - frame_deg[nd])
        if need > 0:
            row = np.zeros(total)
            row[:E] = inc_mask[nd]
            row[E + 2 * n + nd] = 1.0
            rows.append(row); lb.append(need); ub.append(np.inf)
        if frame_deg[nd] < DEG_MAX:
            row = np.zeros(total)
            row[:E] = inc_mask[nd]
            rows.append(row); lb.append(-np.inf); ub.append(float(DEG_MAX - frame_deg[nd]))

    # rod-rod clearance: pairs that do NOT share a node and whose axes are < clearance apart
    mids = (pos[cand[:, 0]] + pos[cand[:, 1]]) / 2.0
    tree = cKDTree(mids)
    for e, f in tree.query_pairs(r=L_MAX + clearance):
        i, j = cand[e]
        k, l = cand[f]
        if i == k or i == l or j == k or j == l:
            continue  # shared node: not a conflict
        if segment_segment_dist(pos[i], pos[j], pos[k], pos[l]) < clearance:
            row = np.zeros(total)
            row[e] = row[f] = 1.0
            rows.append(row); lb.append(-np.inf); ub.append(1.0)

    A = np.asarray(rows)
    lb_lo = np.zeros(total); lb_lo[:E] = 0.0
    lb_hi = np.full(total, np.inf); lb_hi[:E] = 1.0
    res = milp(c=C, constraints=[LinearConstraint(A, np.asarray(lb), np.asarray(ub))],
               integrality=integrality, bounds=Bounds(lb_lo, lb_hi))
    x = res.x if res.success else np.zeros(total)
    y = np.round(x[:E]).astype(int)
    slack = {"s1": x[E:E + n], "s2": x[E + n:E + 2 * n], "sd": x[E + 2 * n:E + 3 * n],
             "sf": x[E + 3 * n:E + 4 * n], "sa": x[E + 4 * n:E + 5 * n]}
    return cand[y == 1], slack


def _node_is_frame(nd, n_frame):
    return nd < n_frame


# --------------------------------------------------------------------- top level
def _has_parent(pos, nd):
    """True if node ``nd`` has a steep parent (≥48°, 8–22 mm) below it."""
    d = pos - pos[nd]
    dist = np.linalg.norm(d, axis=1)
    dz = d[:, 2]
    sin_elev = -dz / np.maximum(dist, 1e-9)
    return bool(((dz < 0) & (sin_elev >= SIN48) & (dist >= L_MIN) & (dist <= L_MAX)).any())


def repair(all_pts, all_ids, fixed, face_paras, normals, centers, n_frame, frame_chain):
    """Nudge stuck nodes upward, then delete any still without a downward candidate."""
    for _ in range(12):
        cand = candidate_rods(all_pts, n_frame, frame_chain, all_ids, normals, centers)
        has_down = np.zeros(len(all_pts), dtype=bool)
        for i, j in cand:
            if all_pts[j, 2] < all_pts[i, 2]:
                has_down[i] = True
            if all_pts[i, 2] < all_pts[j, 2]:
                has_down[j] = True
        free = np.where(~fixed)[0]
        stuck = [nd for nd in free if not has_down[nd]]
        if not stuck:
            break
        # nudge all stuck nodes up a small step (within their face / inside)
        for nd in stuck:
            all_pts[nd, 2] += 3.0
            fid = all_ids[nd]
            if fid >= 0:
                corner, e1, e2 = face_paras[fid]
                g11, g12, g22 = e1 @ e1, e1 @ e2, e2 @ e2
                det = g11 * g22 - g12 * g12
                q = all_pts[nd] - corner
                s = np.clip((g22 * (q @ e1) - g12 * (q @ e2)) / det, 0.0, 1.0)
                t = np.clip((g11 * (q @ e2) - g12 * (q @ e1)) / det, 0.0, 1.0)
                all_pts[nd] = corner + s * e1 + t * e2
            else:
                for n_, c_ in zip(normals, centers):
                    if (all_pts[nd] - c_) @ n_ > 0.0:
                        all_pts[nd] -= n_ * ((all_pts[nd] - c_) @ n_)
        # re-check and delete any still stuck
        cand = candidate_rods(all_pts, n_frame, frame_chain, all_ids, normals, centers)
        has_down = np.zeros(len(all_pts), dtype=bool)
        for i, j in cand:
            if all_pts[j, 2] < all_pts[i, 2]:
                has_down[i] = True
            if all_pts[i, 2] < all_pts[j, 2]:
                has_down[j] = True
        still = [nd for nd in np.where(~fixed)[0] if not has_down[nd]]
        if still:
            keep = np.ones(len(all_pts), dtype=bool)
            keep[still] = False
            all_pts = all_pts[keep]
            all_ids = all_ids[keep]
            fixed = np.zeros(len(all_pts), dtype=bool)
            fixed[:n_frame] = True
    return all_pts, all_ids, fixed


def _constructive_face(start_positions, geom, n_target, r, rng):
    """Bottom-up sampler for an overhanging face (Section 6.1): each new node gets ≥2
    in-face parents among already-placed nodes, keeping min distance ≥ r."""
    placed = [np.asarray(p, dtype=float) for p in start_positions]
    result = []
    for _ in range(n_target):
        best = None
        best_score = np.inf
        for _t in range(400):
            p2d = sample_in_polygon(geom["inset"], rng)
            p = geom["origin"] + p2d[0] * geom["u"] + p2d[1] * geom["v"]
            parents = 0
            mind = np.inf
            for q in placed:
                dq = p - q
                Lq = float(np.linalg.norm(dq))
                if Lq < mind:
                    mind = Lq
                dz = p[2] - q[2]
                if dz > 0 and L_MIN <= Lq <= L_MAX and dz / Lq >= SIN48:
                    parents += 1
            if parents >= 2 and mind >= r:
                score = p[2] - 0.5 * mind
                if score < best_score:
                    best_score = score
                    best = p
        if best is None:
            break
        placed.append(best)
        result.append(best)
    return result


def _repair(pos, fixed, face_ids, face_geoms, normals, centers, n_frame, frame_on_face,
            chain, spacing, seed=None):
    """Section 8 repair: move unsupported nodes (by MIP slack) and re-solve, then a
    clearance-aware fallback rod for the last holdouts."""
    allowed, _, _, _, _ = _allowed_parents(face_ids, normals, centers, n_frame, frame_on_face)
    margin = 0.4 * spacing
    rng = np.random.default_rng(seed if seed is not None else 0)

    for _round in range(3):
        cand = candidate_rods(pos, n_frame, chain, face_ids, normals, centers)
        _, slack = select_rods(pos, cand, chain, face_ids, normals, centers, n_frame)
        bad = np.where(((slack['s1'] > 0.5) | (slack['s2'] > 0.5)) & ~fixed)[0]
        if len(bad) == 0:
            break
        bad = bad[np.argsort(pos[bad, 2])]
        moved = False
        for i in bad:
            best, best_cnt = None, -1
            for _t in range(400):
                trial = _project((pos[i] + rng.uniform(-1.0, 1.0, 3) * 0.6 * spacing)[None, :],
                                 np.array([False]), face_ids[i:i + 1], face_geoms,
                                 normals, centers, margin)[0]
                dd = np.linalg.norm(pos - trial, axis=1); dd[i] = np.inf
                if dd.min() < 0.65 * spacing:
                    continue
                trial_pos = pos.copy(); trial_pos[i] = trial
                cnt = _count_parents(trial_pos, i, allowed[i])
                if cnt > best_cnt:
                    best_cnt = cnt
                    best = trial
            if best is not None and best_cnt >= 1:
                pos[i] = best
                moved = True
        if not moved:
            break

    cand = candidate_rods(pos, n_frame, chain, face_ids, normals, centers)
    selected, slack = select_rods(pos, cand, chain, face_ids, normals, centers, n_frame)

    # delete interior nodes with no downward rod (Section 8: "delete any I node with s1 > 0"),
    # then re-solve once
    bad_i = np.where((slack['s1'] > 0.5) & (face_ids < 0) & ~fixed)[0]
    if len(bad_i):
        keep = np.ones(len(pos), dtype=bool)
        keep[bad_i] = False
        pos = pos[keep]
        face_ids = face_ids[keep]
        fixed = fixed[keep]

    cand = candidate_rods(pos, n_frame, chain, face_ids, normals, centers)
    selected, _ = select_rods(pos, cand, chain, face_ids, normals, centers, n_frame)
    rods = [(min(a, b), max(a, b)) for a, b in chain] + [tuple(map(int, e)) for e in selected]

    # clearance-aware fallback for any remaining unsupported free node
    deg = np.zeros(len(pos), dtype=int)
    for a, b in rods:
        deg[a] += 1; deg[b] += 1
    for nd in range(len(pos)):
        if fixed[nd]:
            continue
        if any((a == nd and pos[b, 2] < pos[nd, 2]) or (b == nd and pos[a, 2] < pos[nd, 2]) for a, b in rods):
            continue
        best, best_d = None, np.inf
        for p in range(len(pos)):
            if p == nd or not (pos[p, 2] < pos[nd, 2]):
                continue
            dd = float(np.linalg.norm(pos[p] - pos[nd]))
            if not (L_MIN <= dd <= L_MAX and (pos[nd, 2] - pos[p, 2]) / dd >= SIN48):
                continue
            if deg[nd] >= DEG_MAX or deg[p] >= DEG_MAX:
                continue
            ok = True
            for a, b in rods:
                if a == nd or b == nd or a == p or b == p:
                    continue
                if segment_segment_dist(pos[nd], pos[p], pos[a], pos[b]) < CLEARANCE:
                    ok = False; break
            if not ok:
                continue
            for k in range(len(pos)):
                if k == nd or k == p:
                    continue
                if point_segment_dist(pos[k], pos[nd], pos[p]) < NODE_CLEARANCE:
                    ok = False; break
            if ok and dd < best_d:
                best = p; best_d = dd
        if best is not None:
            rods.append((min(nd, best), max(nd, best)))
            deg[nd] += 1; deg[best] += 1

    rods = np.unique(np.sort(np.asarray(rods, dtype=int), axis=1), axis=0)
    return pos, rods, fixed, face_ids


def _spread_faces(pos, fixed, face_ids, face_geoms, normals, centers, n_frame, frame_on_face,
                  spacing, iters=300, seed=None):
    """2-D Coulomb spreading of face nodes across their own face (organic-uniform), keeping the
    0.35h inset. Runs after the 3-D relaxation so the interior is untouched."""
    rng = np.random.default_rng(seed if seed is not None else 0)
    is_face = face_ids >= 0
    r_c = 1.25 * spacing
    margin = 0.4 * spacing
    step_clip = 0.08 * spacing
    jitter0 = 0.02 * spacing
    faces_present = np.unique(face_ids[face_ids >= 0])
    for it in range(iters):
        F = np.zeros_like(pos)
        for fi in faces_present:
            m = face_ids == fi
            sub = pos[m]
            delta = sub[:, None, :] - sub[None, :, :]
            dd = np.linalg.norm(delta, axis=-1)
            np.fill_diagonal(dd, np.inf)
            r = np.clip(r_c - dd, 0.0, None)
            u = delta / np.maximum(dd, 1e-9)[:, :, None]
            F[m] += (2.0 * r[..., None] * u).sum(axis=1)
            # repulsion from the frame nodes on this face's boundary
            fr = np.where(frame_on_face[:, fi])[0]
            if len(fr) == 0:
                continue
            delta = pos[m][:, None, :] - pos[fr][None, :, :]
            dd = np.linalg.norm(delta, axis=-1)
            r = np.clip(r_c - dd, 0.0, None)
            u = delta / np.maximum(dd, 1e-9)[:, :, None]
            F[m] += (2.0 * r[..., None] * u).sum(axis=1)
        step = 0.5 * F
        dn = np.linalg.norm(step, axis=1, keepdims=True)
        step = step * np.minimum(1.0, step_clip / np.maximum(dn, 1e-12))
        step = step + jitter0 * (1.0 - it / iters) * rng.standard_normal(pos.shape)
        step[~is_face] = 0.0
        pos = pos + step
        pos = _project(pos, fixed, face_ids, face_geoms, normals, centers, margin)
    return pos


def _add_bridge_posts(pos, rods, bridge_nodes, face_ids=None):
    """Give each intermediate bridge node (on a horizontal frame edge) a steep post from
    below, so the ≤10 mm bridge spans rest on something already printed."""
    if len(bridge_nodes) == 0:
        return rods
    rods = [tuple(map(int, e)) for e in rods]
    deg = np.zeros(len(pos), dtype=int)
    for a, b in rods:
        deg[a] += 1
        deg[b] += 1

    def _has_steep_parent(nd):
        for a, b in rods:
            if (a == nd and pos[b, 2] < pos[nd, 2]) or (b == nd and pos[a, 2] < pos[nd, 2]):
                q = b if a == nd else a
                dd = float(np.linalg.norm(pos[q] - pos[nd]))
                if dd > 0 and (pos[nd, 2] - pos[q, 2]) / dd >= SIN48:
                    return True
        return False

    for nd in bridge_nodes:
        if _has_steep_parent(nd):
            continue
        best, best_d = None, np.inf
        for p in range(len(pos)):
            if p == nd or pos[p, 2] >= pos[nd, 2]:
                continue
            dd = float(np.linalg.norm(pos[p] - pos[nd]))
            if not (L_MIN <= dd <= L_MAX and (pos[nd, 2] - pos[p, 2]) / dd >= SIN48):
                continue
            if deg[nd] >= DEG_MAX or deg[p] >= DEG_MAX:
                continue
            ok = True
            for a, b in rods:
                if a == nd or b == nd or a == p or b == p:
                    continue
                if segment_segment_dist(pos[nd], pos[p], pos[a], pos[b]) < CLEARANCE:
                    ok = False
                    break
            if not ok:
                continue
            for k in range(len(pos)):
                if k == nd or k == p:
                    continue
                if point_segment_dist(pos[k], pos[nd], pos[p]) < NODE_CLEARANCE:
                    ok = False
                    break
            if ok and dd < best_d:
                best = p
                best_d = dd
        if best is not None:
            rods.append((min(nd, best), max(nd, best)))
            deg[nd] += 1
            deg[best] += 1
    return np.unique(np.sort(np.asarray(rods, dtype=int), axis=1), axis=0)


def _has_steep_parent(pos, rods, nd):
    for a, b in rods:
        if a == nd and pos[b, 2] < pos[nd, 2]:
            dd = float(np.linalg.norm(pos[b] - pos[nd]))
            if dd >= L_MIN and (pos[nd, 2] - pos[b, 2]) / dd >= SIN48:
                return True
        if b == nd and pos[a, 2] < pos[nd, 2]:
            dd = float(np.linalg.norm(pos[a] - pos[nd]))
            if dd >= L_MIN and (pos[nd, 2] - pos[a, 2]) / dd >= SIN48:
                return True
    return False


def _delete_unsupported(pos, rods, fixed):
    """Delete non-frame nodes with no steep parent (stranded too close to the raft)."""
    pos = np.asarray(pos, dtype=float)
    rods = np.asarray(rods, dtype=int)
    fixed = np.asarray(fixed, dtype=bool)
    for _ in range(4):
        keep = np.ones(len(pos), dtype=bool)
        for nd in range(len(pos)):
            if fixed[nd]:
                continue
            if not _has_steep_parent(pos, rods, nd):
                keep[nd] = False
        if keep.all():
            break
        new_idx = np.cumsum(keep) - 1
        valid = keep[rods[:, 0]] & keep[rods[:, 1]]
        rods = np.stack([new_idx[rods[valid, 0]], new_idx[rods[valid, 1]]], axis=1)
        pos = pos[keep]
        fixed = fixed[keep]
    return pos, rods, fixed


def printable_lattice(
    vertices, edges, faces,
    spacing=12.0, n_face=10, n_interior=150, seed=1,
    iters=900, lam=1.0, kappa=1.25, anchor_scale=1.0, r_c_factor=1.25, same_face_boost=1.0,
    rep_power=2.0, face_spread_iters=0,
):
    """Generate a support-free lattice inside a convex polytope. Returns (nodes, rods)."""
    from .platonic import face_planes, inside_mask

    normals, centers = face_planes(vertices, faces)

    # classify faces: horizontal (bottom on plate / top ceiling) vs side (steep)
    horizontal_face = np.abs(normals[:, 2]) > 0.99
    bottom_face = horizontal_face & (normals[:, 2] < 0.0)   # on the build plate
    top_face = horizontal_face & (normals[:, 2] > 0.0)      # horizontal ceiling (left open)
    zmin = float(vertices[:, 2].min())

    # frame: subdivide each edge; horizontal edges become ≤10 mm bridge spans (unless on raft)
    frame_pts = [v for v in vertices]
    chain = []
    bridge_nodes = []
    idx = len(vertices)
    for i, j in edges:
        d = vertices[j] - vertices[i]
        L = float(np.linalg.norm(d))
        sin_elev = abs(d[2]) / L if L > 1e-9 else 0.0
        horizontal = sin_elev < SIN5
        on_raft = horizontal and vertices[i, 2] <= zmin + 1e-6 and vertices[j, 2] <= zmin + 1e-6
        if horizontal and not on_raft:
            nseg = max(1, int(np.ceil(L / BRIDGE_MAX)))   # bridges in the air
        else:
            # steep edge, OR a horizontal edge on the raft: subdivide at ~spacing so its
            # intermediate nodes can serve as parents for the face nodes above it.
            nseg = max(1, int(round(L / spacing)))
        prev = i
        for k in range(1, nseg):
            frame_pts.append(vertices[i] + (k / nseg) * d)
            chain.append((prev, idx))
            if horizontal and not on_raft:
                bridge_nodes.append(idx)
            prev = idx
            idx += 1
        chain.append((prev, j))
    frame_pts = np.asarray(frame_pts, dtype=float)
    chain = np.asarray(chain, dtype=int)
    bridge_nodes = np.asarray(bridge_nodes, dtype=int)
    n_frame = len(frame_pts)
    frame_on_face = _frame_on_faces(frame_pts, normals, centers)

    rng = np.random.default_rng(seed)

    # face geometry (any convex polygon) + inset
    inset = 0.35 * spacing
    face_geoms = []
    for f in faces:
        g = face_geometry(vertices, f)
        g["inset"] = shrink_polygon(g["poly"], inset)
        face_geoms.append(g)

    # base lattice on the bottom (plate) faces: scatter nodes + Delaunay-triangulate into
    # horizontal rods lying on the build plate (exempt from the steep/bridge rule). These
    # nodes also give the interior and side faces something to rest on.
    n_frame_orig = n_frame
    if bottom_face.any():
        base_pts: list[np.ndarray] = []
        base_chain: list[tuple[int, int]] = []
        base_idx = n_frame
        for fi in np.where(bottom_face)[0]:
            geom = face_geoms[fi]
            bidx = np.where(frame_on_face[:, fi])[0]
            b2d = (frame_pts[bidx] - geom["origin"]) @ np.stack([geom["u"], geom["v"]], axis=1)
            pts2d = np.array([sample_in_polygon(geom["inset"], rng) for _ in range(n_face)])
            all2d = np.vstack([b2d, pts2d]) if len(b2d) else pts2d
            if len(all2d) < 3:
                continue
            tri = Delaunay(all2d)
            local_map = np.concatenate([bidx, np.arange(base_idx, base_idx + len(pts2d))])
            base_idx += len(pts2d)
            for p2d in pts2d:
                base_pts.append(geom["origin"] + p2d[0] * geom["u"] + p2d[1] * geom["v"])
            for s in tri.simplices:
                for k in range(3):
                    a = int(local_map[s[k]])
                    b = int(local_map[s[(k + 1) % 3]])
                    if a == b:
                        continue
                    if a < n_frame_orig and b < n_frame_orig:
                        continue  # already a frame edge
                    base_chain.append((min(a, b), max(a, b)))
        if base_pts:
            frame_pts = np.vstack([frame_pts, np.asarray(base_pts, dtype=float)])
            chain = np.vstack([chain, np.asarray(base_chain, dtype=int)])
            n_frame = len(frame_pts)
            frame_on_face = _frame_on_faces(frame_pts, normals, centers)

    # every face gets a constructive bottom-up start (Section 6.1) so the face lattice
    # has ~2 in-face parents per node by construction
    r_construct = 0.73 * spacing

    face_pts, face_ids = [], []
    for fi, face in enumerate(faces):
        if bottom_face[fi] or top_face[fi]:
            continue  # bottom face is on the plate; top face is a horizontal ceiling left open
        geom = face_geoms[fi]
        start = frame_pts[frame_on_face[:, fi]]
        pts = _constructive_face(start, geom, n_face, r_construct, rng)
        pts_arr = np.asarray(pts, dtype=float).reshape(-1, 3)
        if len(pts) < n_face:
            n_extra = n_face - len(pts)
            extra = []
            for _ in range(n_extra):
                p2d = sample_in_polygon(geom["inset"], rng)
                extra.append(geom["origin"] + p2d[0] * geom["u"] + p2d[1] * geom["v"])
            pts_arr = np.vstack([pts_arr, np.asarray(extra)])
        face_pts.append(pts_arr)
        face_ids += [fi] * n_face
    if face_pts:
        face_pts = np.vstack(face_pts)
    else:
        face_pts = np.zeros((0, 3))
    face_ids = np.asarray(face_ids, dtype=int)

    # interior points (rejection sampling in the bounding box), kept ≥ L_MIN above the raft
    lo = vertices.min(axis=0); hi = vertices.max(axis=0)
    lo[2] += L_MIN  # bottom margin: nodes too close to the plate can't find a parent
    interior = []
    while len(interior) < n_interior:
        batch = rng.uniform(lo, hi, size=(n_interior * 4, 3))
        interior.append(batch[inside_mask(batch, normals, centers)])
    interior = np.vstack(interior)[:n_interior]

    all_pts = np.vstack([frame_pts, face_pts, interior])
    all_ids = np.concatenate([np.full(n_frame, -1), face_ids, np.full(n_interior, -1)])
    fixed = np.zeros(len(all_pts), dtype=bool); fixed[:n_frame] = True

    all_pts = relax_printable(all_pts, fixed, all_ids, face_geoms, normals, centers,
                              n_frame, frame_on_face, spacing, iters=iters, lam=lam,
                              kappa=kappa, seed=seed, anchor_scale=anchor_scale,
                              r_c_factor=r_c_factor, same_face_boost=same_face_boost,
                              rep_power=rep_power)

    if face_spread_iters > 0:
        all_pts = _spread_faces(all_pts, fixed, all_ids, face_geoms, normals, centers,
                                n_frame, frame_on_face, spacing, iters=face_spread_iters,
                                seed=seed)

    all_pts, rods, fixed, all_ids = _repair(all_pts, fixed, all_ids, face_geoms, normals, centers,
                                            n_frame, frame_on_face, chain, spacing, seed=seed)
    rods = _add_bridge_posts(all_pts, rods, bridge_nodes, all_ids)
    all_pts, rods, fixed = _delete_unsupported(all_pts, rods, fixed)
    return all_pts, rods
