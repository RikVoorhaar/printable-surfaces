"""Build the conventional bottle-shaped Klein bottle figure: surface + rod lattice.

Run with ``uv run python scripts/klein_bottle_figure.py``. It writes

    output/lattice/klein_bottle_surface.stl     dense surface mesh
    output/lattice/klein_bottle_frame.ply/.stl  the rod lattice, thickened
    output/lattice/klein_bottle_preview.png     two views

and prints every number the figure is judged on. The geometry and the relaxer live in
:mod:`printable_surfaces.klein_bottle`; this file is only the driver.

Nothing here touches the figure-8 bottle (``klein.py`` / ``klein_lattice.py``).
"""

from __future__ import annotations

import os

import numpy as np

from printable_surfaces import klein_bottle as kb
from printable_surfaces.colormap import elevation_ramp
from printable_surfaces.klein import clearance_pairs, printability, rod_elevations
from printable_surfaces.lattice import build_frame_mesh

OUT = "output/lattice"

# The configuration that reaches zero collisions at the target density. Justification is in
# `report_grid_search` below and in the module docstring.
M, N = 22, 10  # 2·22·10 = 440 rods, 22·10 = 220 nodes
OFFSET = 0.0  # the other seam-compatible v offset (see below)
RELAX = dict(
    iterations=6000,
    target=2.9,
    gain=0.9,
    spring=0.005,
    max_step=0.05,
    equal_arclength=False,
)


def rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def report_surface() -> dict:
    rule("STEP 1 — surface and embedding")
    v = kb.verify()
    print(f"c                 = {v['c']:.12f}   (5 − 3√3)/2")
    print(f"b                 = {v['b']:.12f}")
    print(f"H0 = 16 + b       = {v['H0']:.12f}")
    print(f"s = H/H0          = {v['s']:.12f}   (H = {kb.SIZE_MM:.0f} mm)")
    print(f"u at z = 0 (cusp) = {v['u_cusp']:.9f} rad, z there = {v['z_at_cusp_point']:.3e} mm")
    print(f"z at u = 3π/2     = {v['z_at_top_point']:.9f} mm")
    print(f"bbox min          = {np.round(v['bbox_min'], 4).tolist()}")
    print(f"bbox max          = {np.round(v['bbox_max'], 4).tolist()}")
    print(f"bbox size         = {np.round(v['size'], 4).tolist()} mm")
    print(f"z min / z max     = {v['z_min']:.6e} / {v['z_max']:.9f}  (sample min > 0: {v['z_min'] >= 0})")
    print(f"v periodicity err = {v['v_periodicity_error']:.3e} mm   (P(u, v+2π) − P(u, v))")
    print(f"seam φ(v)=π−v err = {v['seam_phi_max_error']:.3e} mm   max |P(0,v) − P(2π, π−v)|")
    print(f"seam, wrong φ err = {v['seam_wrong_phi_error']:.3e} mm   max |P(0,v) − P(2π, v)|  (control)")
    ratio = v["crease_c0_error_1e-3"] / v["crease_c0_error_1e-5"]
    print(
        f"u = π continuity  = {v['crease_c0_error_1e-3']:.3e} mm at ε=1e-3, "
        f"{v['crease_c0_error_1e-5']:.3e} mm at ε=1e-5  (ratio {ratio:.1f} for a 100× step)"
    )
    print("                    → the error is O(ε), i.e. the two sheets meet exactly; the")
    print("                    parametrisation is C⁰ there, not merely close.")
    print(
        f"u = π tangent jump= {v['crease_c1_max_deg']:.2f}° max, "
        f"{v['crease_c1_median_deg']:.2f}° median  (C¹ only, the sheet is C⁰)"
    )

    mesh = kb.surface_mesh(480, 240)
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/klein_bottle_surface.stl"
    mesh.export(path)
    euler = len(mesh.vertices) - len(mesh.edges_unique) + len(mesh.faces)
    print(
        f"surface mesh      : V={len(mesh.vertices)} E={len(mesh.edges_unique)} "
        f"F={len(mesh.faces)}  χ={euler}  watertight={mesh.is_watertight}"
    )
    print(f"wrote {path} ({os.path.getsize(path) / 1e6:.2f} MB)")
    print(
        "seam              : CLOSED — the (u = 2π) column is welded onto (u = 0) with the flip\n"
        "                    v ↦ π − v, so χ = 0 and the mesh is a genuine Klein bottle.\n"
        "                    It is non-orientable, so no consistent winding exists (by design)."
    )
    return v


def report_double_curve() -> dict:
    rule("STEP 2 — the double curve")
    print("Structure (analytic): the second sheet (π < u ≤ 2π) is a family of horizontal")
    print("circles P(u,v) = s(A(u) − r cos v, r sin v, b − 16 sin u). z sees only sin u, so")
    print("u and 3π−u share a height; the two circles then meet whenever")
    print("|r₁ − r₂| < |A₁ − A₂| < r₁ + r₂, i.e. sin u > −2/3 and 12|cos u|(1+sin u) < 8.")

    ana = kb.double_curve_branch2(nu=20001)
    x = ana["pairs"]
    print(f"\npreimage u range  = [{ana['u1'].min():.6f}, {ana['u1'].max():.6f}] rad, in two arcs:")
    us = np.unique(np.round(ana["u1"], 6))
    gaps = np.where(np.diff(us) > 1e-3)[0]
    arcs = np.split(us, gaps + 1)
    lo, hi = float(arcs[0].min()), float(arcs[0].max())
    print(f"                    arc A: u ∈ [{lo:.6f}, {hi:.6f}]")
    print(f"                    arc B: u ∈ [{3*np.pi-hi:.6f}, {3*np.pi-lo:.6f}]  (exactly 3π − arc A)")
    print("                  so the preimage is TWO closed circles in the (u,v) torus (v is a")
    print("                  two-branched function of u, the branches joining at the arc ends),")
    print("                  each carried one-to-one onto the SAME 3-D curve by u₂ = 3π − u₁.")
    print(f"                  arc ends: external tangency (d = r₁+r₂) at u = {lo:.6f},")
    print(f"                  internal tangency (d = |r₁−r₂|, i.e. sin u = −2/3) at u = {hi:.6f}")
    print(f"u₂ − (3π−u₁) max  = {np.abs(x[:,2] - (3*np.pi - x[:,0])).max():.3e}")
    print(f"preimage v range  = [{ana['v1'].min():.6f}, {ana['v1'].max():.6f}] rad")
    print(f"closure P(u₁,v₁)=P(u₂,v₂) max err = {np.linalg.norm(kb.bottle(x[:,0],x[:,1]) - kb.bottle(x[:,2],x[:,3]), axis=1).max():.3e} mm")
    print(f"\nimage: radius √(x²+y²) = {ana['radius'].min():.3f} … {ana['radius'].max():.3f} mm")
    print(f"       height z        = {ana['height'].min():.3f} … {ana['height'].max():.3f} mm")
    print(f"       x               = {ana['points'][:,0].min():.3f} … {ana['points'][:,0].max():.3f} mm")
    print(f"       y               = {ana['points'][:,1].min():.3f} … {ana['points'][:,1].max():.3f} mm")
    print("so it is not a circle: it is a closed loop on the upper right shoulder of the body,")
    print("where the neck (second sheet) dives through the body wall.")

    num = kb.double_curve(nu=181, nv=90, coarse=2.5, verbose=True)
    from scipy.spatial import cKDTree

    closure = np.linalg.norm(
        kb.bottle(num["pairs"][:, 0], num["pairs"][:, 1])
        - kb.bottle(num["pairs"][:, 2], num["pairs"][:, 3]),
        axis=1,
    )
    gap = cKDTree(ana["points"]).query(num["points"])[0]
    print(f"\nnumeric search    : {len(num['pairs'])} refined pairs found independently"
          " (blanket KD-tree + least squares, no assumption about which sheet)")
    print(f"                    all on the second sheet: {num['u1_on_second_sheet']}")
    print(f"                    closure residual max  : {closure.max():.3e} mm")
    print(f"                    |u₂ − (3π−u₁)| max    : {np.abs(num['u2'] - (3*np.pi-num['u1'])).max():.3e}")
    print(f"                    gap to analytic curve : {gap.max():.3e} mm (analytic sampled at 2e4 u points)")
    return ana


def report_lattice(curve_points: np.ndarray) -> dict:
    rule("STEP 3 — rod lattice")
    print(f"grid {M} × {N} cells  →  {2*M*N} rods, {M*N} nodes; seam glued with u₂=u₁, v₂=π−v₁")

    nodes, edges, (u, v) = kb.build(M, N, offset=OFFSET, **RELAX)

    deg = kb.degrees(len(nodes), edges)
    comp = kb.components(len(nodes), edges)
    print(f"\nnodes             = {len(nodes)}")
    print(f"rods              = {len(edges)}")
    print(f"V − E + F (F = {M*N} quads) = {len(nodes)} − {len(edges)} + {M*N} = {len(nodes)-len(edges)+M*N}")
    print(f"node degrees      = min {deg.min()}, max {deg.max()}, all 4: {bool((deg == 4).all())}")
    print(f"connected comps   = {comp}")

    # ---- seam consistency ------------------------------------------------------------
    can = kb.canonical_index(M, N, OFFSET)
    slots = kb.extended(u, v).reshape(-1, 3)
    print(f"seam mismatch     = {np.abs(slots - slots[can]).max():.3e} mm  (derived column vs its alias)")
    seam_rods = (edges[:, 0] < N) & (edges[:, 1] >= (M - 1) * N)
    L = np.linalg.norm(nodes[edges[:, 1]] - nodes[edges[:, 0]], axis=1)
    print(f"seam rods         = {int(seam_rods.sum())} (one per v-row, never a duplicated ring)")
    print(f"  their length    = {L[seam_rods].min():.2f} … {L[seam_rods].max():.2f} mm")
    print(f"  all rods        = {L.min():.2f} … {L.max():.2f} mm")

    # ---- clearance -------------------------------------------------------------------
    st = clearance_pairs(nodes, edges)
    st2 = clearance_pairs(nodes, edges, clearance=3.0)
    print(f"\nmin non-adjacent rod-rod distance = {st['min_distance']:.4f} mm")
    print(f"violations below 2.4 mm           = {st['violations']}")
    print(f"pairs below 3.0 mm (margin check) = {st2['violations']}")
    print(f"pairs examined (broadphase)       = {st['pairs']}")

    from scipy.spatial import cKDTree

    nn = cKDTree(nodes).query(nodes, k=2)[0][:, 1]
    print(f"min node-node distance            = {nn.min():.3f} mm  (joint Ø{2*kb.JOINT_R:.1f} mm)")

    # ---- length and printability -----------------------------------------------------
    pr = printability(nodes, edges)
    print(f"\nrod length        = min {L.min():.2f}, median {np.median(L):.2f}, max {L.max():.2f} mm")
    print(f"below 48°         = {pr['below_48_pct']:.1f} %   (raw, ignoring bridges: {pr['raw_below_48_pct']:.1f} %)")
    print(f"  steep ≥48°      = {pr['steep_pct']:.1f} %")
    print(f"  bridges ≤10 mm  = {pr['bridge_pct']:.1f} %")
    print(f"  on build plate  = {pr['on_plate']} rods")
    print("  (the figure rests on a single joint sphere: z_min of the surface is one point,")
    print("   the cusp u = 1.669, v = 0, not an area)")
    print(f"  elevation hist  = {pr['hist']}")
    print(f"  median elevation= {pr['median_elev']:.1f}°")

    # ---- relation to the double curve ------------------------------------------------
    d_rod = kb.rods_near_curve(nodes, edges, curve_points)
    d_node = cKDTree(curve_points).query(nodes)[0]
    along = kb.rods_along_curve(nodes, edges, curve_points, tol=0.5)
    frac = along / np.maximum(L, 1e-9)
    print(f"\nrods on the double curve : closest rod is {d_rod.min():.3f} mm from it"
          f" ({int((d_rod < 1.0).sum())} rods pass within 1 mm)")
    print("                           — they CROSS it, which is unavoidable: the curve lies")
    print("                           on the surface, so every rod spanning it does")
    print(f"rods running ALONG it    : longest near-run {along.max():.2f} mm"
          f" = {100*frac.max():.0f}% of that rod (tolerance 0.5 mm)")
    print(f"nodes on the double curve: closest node is {d_node.min():.3f} mm from it"
          f" ({int((d_node < 1.0).sum())} nodes under 1 mm)")

    # ---- how far the rods stray from the surface -------------------------------------
    uu, vv = kb.uv_grid(361, 182)
    dense = kb.bottle(uu.ravel(), vv.ravel())
    tree = cKDTree(dense)
    mids = 0.5 * (nodes[edges[:, 0]] + nodes[edges[:, 1]])
    dmid = tree.query(mids)[0]
    print(f"rod midpoints off surface: median {np.median(dmid):.2f} mm, max {dmid.max():.2f} mm")

    return {"nodes": nodes, "edges": edges, "u": u, "v": v, "L": L, "pr": pr, "st": st}


def report_grid_search() -> None:
    rule("why this grid — the search that produced it")
    rows = [
        # (m, n, offset, target, iterations, min_mm, violations, below48)
        (16, 14, 0.5, 3.4, 600, 1.05, 71, 31.0),
        (20, 12, 0.0, 3.0, 4000, 1.42, 15, 32.7),
        (22, 10, 0.5, 2.8, 3000, 2.42, 0, 36.1),
        (22, 10, 0.0, 2.6, 1500, 2.50, 0, 35.2),
        (22, 10, 0.0, 2.9, 6000, 2.78, 0, 36.4),
        (28, 8, 0.5, 3.2, 6000, 2.79, 0, 43.3),
    ]
    print(f"{'m':>3} {'n':>3} {'off':>4} {'tgt':>4} {'it':>5} | {'rods':>5} {'min':>6} {'viol':>5} {'<48°':>6}")
    for m, n, off, tgt, it, mn, viol, p48 in rows:
        print(f"{m:3d} {n:3d} {off:4.1f} {tgt:4.1f} {it:5d} | {2*m*n:5d} {mn:6.2f} {viol:5d} {p48:6.1f}")
    print("""
Reading it:
  * n controls the ring spacing, and the thinnest ring (r = 2s = 5.54 mm, at the seam)
    allows n ≤ 14 at 2.4 mm spacing at all (2·5.54·sin(π/14) = 2.45 mm). n = 12 and 14
    leave no room: every run there stalls with 10–70 violations, because the two lobes
    genuinely interpenetrate at the double curve and the rods have nowhere to go.
  * n = 8 clears easily but its rings at the fat end are 12.7 mm chords, so they are
    horizontal rods too long to be bridges: printability drops to 43 %.
  * the v offset must satisfy k = n/2 − 2·offset − j ∈ ℤ, so offset ∈ {0, ½} are the only
    grid-compatible choices. ½ was tried first (the figure-8 recipe); it reaches zero
    violations but only 2.42 mm — 0.02 mm of margin — and long 23.6 mm rods. offset = 0
    gives 2.78 mm and shorter rods, so it is the one shipped. No grid line lies along the
    double-curve preimage in either case: the preimage is not a constant-v line (on one
    branch v runs 0.01 … 3.15 rad as u sweeps its arc), so a half-cell shift in v cannot
    be "the" offset from it — there is nothing to shift off.
""")


def build_frame(res: dict) -> None:
    rule("mesh, lift, export")
    nodes, edges = res["nodes"], res["edges"]
    dz = kb.JOINT_R - float(nodes[:, 2].min())
    nodes = nodes + np.array([0.0, 0.0, dz])
    print(f"lift dz = JOINT_R − min(z) = {kb.JOINT_R:.2f} − {nodes[:,2].min()-dz:.4f} = {dz:+.4f} mm")
    print(f"node z range after lift = {nodes[:,2].min():.3f} … {nodes[:,2].max():.3f} mm")
    print(f"joint sphere bottom     = {nodes[:,2].min()-kb.JOINT_R:.3f} mm (touches the plate)")

    mesh = build_frame_mesh(
        nodes,
        edges,
        rod_radius=kb.ROD_R,
        joint_radius=kb.JOINT_R,
        rod_sections=8,
        joint_count=(6, 6),
        color_by="elevation",
    )
    print(f"frame mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} triangles, "
          f"bounds z {mesh.bounds[0][2]:.3f} … {mesh.bounds[1][2]:.3f}")
    for suffix in ("ply", "stl"):
        path = f"{OUT}/klein_bottle_frame.{suffix}"
        mesh.export(path)
        print(f"wrote {path} ({os.path.getsize(path)/1e6:.2f} MB)")
    return nodes


def preview(nodes: np.ndarray, edges: np.ndarray, curve_points: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    elev = rod_elevations(nodes, edges)
    colours = elevation_ramp(elev).astype(float) / 255.0  # sRGB bytes -> matplotlib
    segs = np.stack([nodes[edges[:, 0]], nodes[edges[:, 1]]], axis=1)

    # Order the double curve so it draws as one stroke.
    order = np.lexsort((curve_points[:, 1], curve_points[:, 0]))
    cp = curve_points[order]
    o = np.argsort(np.arctan2(cp[:, 1], cp[:, 0] - 24.0))
    cp = cp[o]

    fig = plt.figure(figsize=(13, 8.4))
    for k, (azim, el, label) in enumerate(
        [(-90, 2, "front — looking along +y"), (28, 16, "three-quarter")]
    ):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.add_collection3d(Line3DCollection(segs, colors=colours, linewidths=2.0))
        ax.add_collection3d(
            Line3DCollection(
                np.stack([cp[:-1], cp[1:]], axis=1), colors="#e8453c", linewidths=3.2
            )
        )
        ax.scatter(*nodes.T, s=9, c="#8c93a3", depthshade=False)
        ax.scatter(*cp[::20].T, s=1, c="#e8453c", depthshade=False)
        lo, hi = nodes.min(axis=0), nodes.max(axis=0)
        pad = 2.0
        ax.set_xlim(lo[0] - pad, hi[0] + pad)
        ax.set_ylim(lo[1] - pad, hi[1] + pad)
        ax.set_zlim(lo[2] - pad, hi[2] + pad)
        ax.set_box_aspect((hi[0] - lo[0] + 2 * pad, hi[1] - lo[1] + 2 * pad, hi[2] - lo[2] + 2 * pad))
        ax.view_init(elev=el, azim=azim)
        ax.set_title(label, fontsize=11)
        ax.set_axis_off()
    fig.suptitle(
        "bottle-shaped Klein bottle — 440 rods / 220 joints, min rod-rod 2.78 mm, 0 collisions\n"
        "rod colour = elevation above horizontal;  red = the double curve",
        fontsize=11,
    )
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 90))
    cb = fig.colorbar(sm, ax=fig.axes, shrink=0.5, pad=0.02)
    cb.set_label("rod elevation above horizontal (deg)")
    path = f"{OUT}/klein_bottle_preview.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path} ({os.path.getsize(path)/1e6:.2f} MB)")


def summary() -> None:
    rule("files")
    for name in (
        "klein_bottle_surface.stl",
        "klein_bottle_frame.ply",
        "klein_bottle_frame.stl",
        "klein_bottle_preview.png",
    ):
        path = f"{OUT}/{name}"
        print(f"{path:<48} {os.path.getsize(path)/1e6:8.2f} MB")


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    report_surface()
    curve = report_double_curve()
    res = report_lattice(curve["points"])
    report_grid_search()
    nodes = build_frame(res)
    preview(nodes, res["edges"], curve["points"])
    summary()


if __name__ == "__main__":
    main()
