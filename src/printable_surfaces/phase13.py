"""Phase 13 — cube → rhombohedron lattice (fixed frame + Coulomb fill).

Per the Perplexity report: orient the cube body-diagonal-vertical, stretch it into a
rhombohedron (all edges printable), fix the frame (vertices + edge subdivisions), then
seed and Coulomb-relax face + interior points with the frame held fixed.

Run:  uv run python -m printable_surfaces.phase13
"""

from __future__ import annotations

import os

import numpy as np
from scipy.spatial import cKDTree

from .artifacts import register
from .lattice import build_frame_mesh
from .platonic import cube_rhombohedron, cube_rotation, face_parallelogram, face_planes
from .slicer import slice_stl

CIRCUMRADIUS = 50.0
ROD_MM = 1.2
JOINT_MM = 1.8
SPACING = 15.0          # target rod spacing (mm)
N_FACE = 8              # face points per face
N_INTERIOR = 120        # interior points
# (name, stretch)
VARIANTS = [("cube_s140", 1.40), ("cube_s169", 1.69), ("cube_s190", 1.90)]


def build_frame(vertices, edges, spacing):
    """Fixed frame: vertices + per-edge subdivisions; returns (points, chain_edges)."""
    pts = list(vertices)
    chain = []
    idx = len(vertices)
    for i, j in edges:
        d = vertices[j] - vertices[i]
        nseg = max(1, int(round(np.linalg.norm(d) / spacing)))
        prev = i
        for k in range(1, nseg):
            pts.append(vertices[i] + (k / nseg) * d)
            chain.append((prev, idx))
            prev = idx
            idx += 1
        chain.append((prev, j))
    return np.asarray(pts, dtype=float), np.asarray(chain, dtype=int)


def _clamp_parallelogram(p, corner, e1, e2):
    """Project ``p`` onto the plane span(e1,e2) and clamp to the parallelogram [0,1]²."""
    g11, g12, g22 = e1 @ e1, e1 @ e2, e2 @ e2
    det = g11 * g22 - g12 * g12
    q = p - corner
    r1, r2 = q @ e1, q @ e2
    s = (g22 * r1 - g12 * r2) / det
    t = (g11 * r2 - g12 * r1) / det
    return corner + np.clip(s, 0.0, 1.0)[:, None] * e1 + np.clip(t, 0.0, 1.0)[:, None] * e2


def relax(all_pts, n_frame, face_ids, face_paras, normals, centers, spacing, iterations=1200, cooling=0.995):
    """Coulomb-ish relaxation: frame fixed, face points clamped to faces, interior clamped inside."""
    n = len(all_pts)
    cutoff = 1.5 * spacing
    temp = 15.0
    fixed = np.zeros(n, dtype=bool)
    fixed[:n_frame] = True
    movable = ~fixed

    for _ in range(iterations):
        delta = all_pts[:, None, :] - all_pts[None, :, :]
        dd = np.linalg.norm(delta, axis=-1)
        np.fill_diagonal(dd, np.inf)
        u = delta / np.maximum(dd, 1e-6)[:, :, None]
        rep = np.where(dd < cutoff, cutoff - dd, 0.0)
        F = (u * rep[:, :, None]).sum(axis=1)
        F[fixed] = 0.0

        dn = np.linalg.norm(F, axis=1, keepdims=True)
        scale = np.minimum(1.0, temp / np.maximum(dn, 1e-12))
        all_pts += F * scale

        # face points stay within their rhombus
        for fi in np.unique(face_ids[face_ids >= 0]):
            m = face_ids == fi
            corner, e1, e2 = face_paras[fi]
            all_pts[m] = _clamp_parallelogram(all_pts[m], corner, e1, e2)

        # all movable points stay inside the polytope (safety)
        for n_, c_ in zip(normals, centers):
            out = movable & ((all_pts - c_) @ n_ > 0.0)
            if out.any():
                all_pts[out] -= n_ * ((all_pts[out] - c_) @ n_)[:, None]

        temp *= cooling

    return all_pts


def wire(all_pts, chain_edges, n_frame, k=4):
    """Frame chains + k-NN for the non-frame points (to all points)."""
    edges = {tuple(e) for e in chain_edges}
    tree = cKDTree(all_pts)
    non_frame = np.arange(n_frame, len(all_pts))
    _, idx = tree.query(all_pts[non_frame], k=k + 1)
    for local_i, gi in enumerate(non_frame):
        for j in idx[local_i, 1:]:
            edges.add((min(gi, int(j)), max(gi, int(j))))
    return np.asarray(sorted(edges), dtype=int)


def build_and_register(name: str, stretch: float) -> None:
    vertices, edges, faces = cube_rhombohedron(CIRCUMRADIUS, stretch)
    normals, centers = face_planes(vertices, faces)

    frame_pts, chain = build_frame(vertices, edges, SPACING)
    n_frame = len(frame_pts)

    rng = np.random.default_rng(1)

    # face points
    face_pts = []
    face_ids = []
    for fi, face in enumerate(faces):
        corner, e1, e2 = face_parallelogram(vertices, face)
        u = rng.uniform(0.0, 1.0, N_FACE)
        w = rng.uniform(0.0, 1.0, N_FACE)
        face_pts.append(corner + u[:, None] * e1 + w[:, None] * e2)
        face_ids += [fi] * N_FACE
    face_pts = np.vstack(face_pts)
    face_ids = np.asarray(face_ids, dtype=int)

    # interior points (uniform in the cube, mapped to the rhombohedron)
    rot = cube_rotation()
    half = CIRCUMRADIUS / np.sqrt(3.0)
    interior = rng.uniform(-half, half, size=(N_INTERIOR, 3))
    interior = (rot @ interior.T).T
    interior[:, 2] *= stretch

    all_pts = np.vstack([frame_pts, face_pts, interior])
    all_ids = np.concatenate([np.full(n_frame, -1), face_ids, np.full(N_INTERIOR, -1)])

    face_paras = [face_parallelogram(vertices, face) for face in faces]
    all_pts = relax(all_pts, n_frame, all_ids, face_paras, normals, centers, SPACING)
    all_edges = wire(all_pts, chain, n_frame, k=4)

    mesh = build_frame_mesh(all_pts, all_edges, rod_radius=ROD_MM, joint_radius=JOINT_MM)
    path = f"output/lattice/rhombohedron_{name}.stl"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mesh.export(path)
    register("lattice", path, "stl", f"Rhombohedron · stretch ×{stretch:.2f} · {len(all_pts)} nodes / {len(all_edges)} rods", folder="Cube · rhombohedron")

    gcode = f"output/slicer/rhombohedron_{name}.gcode"
    try:
        proc = slice_stl(path, gcode)
        ok = proc.returncode == 0 and os.path.exists(gcode) and os.path.getsize(gcode) > 0
    except FileNotFoundError:
        ok = False
    if ok:
        register("slicer", gcode, "gcode", f"Rhombohedron ×{stretch:.2f} · sliced OK", folder="Cube · rhombohedron")

    print(f"stretch ×{stretch:.2f}: {len(all_pts)} nodes ({n_frame} frame)  {len(all_edges)} rods  -> {'sliced OK' if ok else 'FAILED'}")


def main() -> None:
    for name, stretch in VARIANTS:
        build_and_register(name, stretch)


if __name__ == "__main__":
    main()
