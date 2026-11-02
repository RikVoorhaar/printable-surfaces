"""Evidence for the figure-8 Klein-bottle rod lattice: the checks behind the design.

Prints, in order: the two immersion symmetries; the numeric rediscovery of the double curve;
the plain tensor grid's *intersecting* rods (the reason the grid is sheared); a sweep over
grid sizes for the finished construction (shear + arc-length/biased v rows + measured
bracing); the double curve's relation to the nodes and rods; the printability bins; and a
two-view PNG of the rods in ``output/lattice/``.

Run::

    uv run python scripts/klein_probe.py

The full lattice report and the STL export live in ``python -m printable_surfaces.klein``.
"""

from __future__ import annotations

import numpy as np

from printable_surfaces.klein import (
    A,
    CLEARANCE_MM,
    clearance_pairs,
    crossing_separation_mm,
    double_curve_check,
    double_curve_preimage,
    find_double_points,
    immersed_bbox,
    klein_grid,
    printability,
    rod_lengths,
    scale_to_size,
    verify_symmetries,
)

HDR = f"{'m':>3} {'n':>3} {'nodes':>5} {'rods':>5} {'braces':>6} {'pairs':>7} {'viol':>5} {'minD':>7}"


def preview(m: int, n: int, path: str = "output/lattice/klein_surface_frame_views.png") -> None:
    """Two-view PNG of the rods, coloured by elevation (steep = green, shallow = red)."""
    import os

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection  # noqa: F401  (documented below)
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    from printable_surfaces.klein import rod_elevations

    nodes, _, edges, _ = klein_grid(m, n)
    elev = rod_elevations(nodes, edges)
    seg = np.stack([nodes[edges[:, 0]], nodes[edges[:, 1]]], axis=1)
    center = 0.5 * (nodes.min(0) + nodes.max(0))
    radius = 0.5 * float((nodes.max(0) - nodes.min(0)).max()) * 1.05

    fig = plt.figure(figsize=(13, 6), dpi=120)
    for k, (el, az) in enumerate([(55, -60), (12, -75), (89, 0)]):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        lc = Line3DCollection(seg, linewidths=0.7, cmap="RdYlGn")
        lc.set_array(elev)
        lc.set_clim(0.0, 90.0)
        ax.add_collection3d(lc)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(max(0.0, center[2] - radius), center[2] + radius)
        ax.view_init(elev=el, azim=az)
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()
        ax.set_title(f"elev {el} / azim {az}", fontsize=8)
    fig.suptitle(f"figure-8 Klein bottle rods — {m} x {n}, coloured by elevation", fontsize=9)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    lo, hi = immersed_bbox(a=A)
    print("== immersion ==")
    print(f"a = {A}   bbox model units lo={np.round(lo, 4)} hi={np.round(hi, 4)}")
    print(f"extent = {np.round(hi - lo, 4)}   largest = {(hi - lo).max():.4f}")
    print(f"scale to 100 mm = {scale_to_size():.6f}")
    print(f"symmetries: {verify_symmetries()}")
    print(f"double curve: {double_curve_preimage()}")
    print()

    print("== double points, found numerically (no analytic input) ==")
    for s in (200, 400, 600):
        r = find_double_points(samples=s)
        print(f"  sample {s:3d}x{s:3d}: pairs matching within tol = {r['candidate_pairs']:5d}, "
              f"far-apart pairs = {r['double_pairs']:5d}, max 3-D gap = "
              f"{r['max_pair_distance']:.2e}, max |v| deviation from k*pi = "
              f"{r['max_v_dev_from_mod_pi']:.2e}, distinct u = {r['distinct_u_values']}")
    print()

    print("== plain tensor grid: the two sheets' rods intersect exactly ==")
    print(f"{'m':>3} {'n':>3} {'rods':>5} {'pairs':>7} {'viol':>6} {'minD mm':>9}")
    for m, n in [(20, 20), (28, 24), (36, 36), (48, 40)]:
        nodes, _, edges, _ = klein_grid(m, n, shear=0.0, cross_bias=0.0, diagonals="all")
        r = clearance_pairs(nodes, edges)
        print(f"{m:3d} {n:3d} {len(edges):5d} {r['pairs']:7d} {r['violations']:6d} "
              f"{r['min_distance']:9.2e}")
    print("  (minD == 0: a v-edge of the v=0 sheet and one of the v=pi sheet are the same"
          " 3-D segment)")
    print()

    print("== finished construction: shear + biased v rows + measured bracing ==")
    print(HDR)
    for m, n in [(24, 20), (24, 22), (28, 22), (30, 22), (30, 24), (32, 24), (34, 24), (36, 30)]:
        nodes, _, edges, _ = klein_grid(m, n)
        r = clearance_pairs(nodes, edges)
        print(f"{m:3d} {n:3d} {len(nodes):5d} {len(edges):5d} {len(edges) - 2 * m * n:6d} "
              f"{r['pairs']:7d} {r['violations']:5d} {r['min_distance']:7.3f}")
    print(f"  (requirement {CLEARANCE_MM} mm; 34x24 is where the tight waist starts to fail)")
    print()

    print("== double curve vs the grid (30 x 24) ==")
    d = double_curve_check(30, 24)
    for k, v in d.items():
        print(f"  {k}: {v}")
    print(f"  crossing: {crossing_separation_mm(30, 24)}")
    print()

    print("== printability (30 x 24) ==")
    nodes, _, edges, _ = klein_grid(30, 24)
    pr = printability(nodes, edges)
    L = rod_lengths(nodes, edges)
    print(f"  rods {pr['rods']}  length min {L.min():.2f} median {np.median(L):.2f} "
          f"max {L.max():.2f} mm  median elevation {pr['median_elev']:.1f} deg")
    print(f"  below 48 deg, raw: {pr['raw_below_48_pct']:.1f}%")
    print(f"  project rule (>=48 deg, or <=5 deg bridge <=10 mm, or on plate): "
          f"{pr['below_48_pct']:.1f}% unprintable  "
          f"(steep {pr['steep_pct']:.1f}%, bridge {pr['bridge_pct']:.1f}%, "
          f"on plate {pr['on_plate']})")
    print(f"  10-degree bins: {pr['hist']}")
    print()
    preview(30, 24)


if __name__ == "__main__":
    main()
