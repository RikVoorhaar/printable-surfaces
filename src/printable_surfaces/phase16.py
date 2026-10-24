"""Phase 16 — version matrix: thickness × scale, with dimensions, material, and STLs.

Generates the support-free rhombohedron lattice once, then tabulates 4 rod-thickness
settings × 4 scale settings. Material is computed analytically (union of node spheres
plus rods cut at the joint spheres), not from the (non-watertight) triangle soup.

Each of the 16 combinations is also exported as an STL and registered on the dashboard.

Run: uv run python -m printable_surfaces.phase16
"""

from __future__ import annotations

import os

import numpy as np

from .artifacts import register
from .lattice import build_frame_mesh
from .platonic import cube_rhombohedron
from .printable import printable_lattice

PLA_DENSITY = 1.24  # g/cm^3 (PrusaSlicer nominal PLA; Wikipedia lists 1.21–1.43)

# (label, rod radius mm, joint radius mm) — joint kept at 1.5× rod, as now.
THICKNESS = [
    ("T1", 1.2, 1.8),  # rod Ø2.4 / joint Ø3.6 (current)
    ("T2", 1.0, 1.5),  # rod Ø2.0 / joint Ø3.0
    ("T3", 0.8, 1.2),  # rod Ø1.6 / joint Ø2.4
    ("T4", 0.6, 0.9),  # rod Ø1.2 / joint Ø1.8
]

# (label, scale factor)
SCALES = [
    ("S1", 1.00),  # original
    ("S2", 0.80),
    ("S3", 0.60),
    ("S4", 0.50),
]


def material_volume_mm3(nodes: np.ndarray, rods: np.ndarray, rod_r: float, joint_r: float) -> float:
    """Union volume of the ball-and-stick lattice (mm^3).

    Every rod is a cylinder of radius ``rod_r``; at each end it is swallowed by the
    endpoint's joint sphere of radius ``joint_r`` up to the sphere surface (length
    ``joint_r`` along the axis), so the visible rod length is ``L - 2·joint_r``.
    """
    L = np.linalg.norm(nodes[rods[:, 1]] - nodes[rods[:, 0]], axis=1)
    rod_vol = float(np.sum(np.pi * rod_r**2 * np.maximum(0.0, L - 2.0 * joint_r)))
    node_vol = float((4.0 / 3.0) * np.pi * joint_r**3 * len(nodes))
    return rod_vol + node_vol


def main() -> None:
    verts, edges, faces = cube_rhombohedron(50.0, 1.69)
    nodes, rods = printable_lattice(verts, edges, faces, spacing=15.0, n_face=48, n_interior=105,
                                    seed=1, iters=900, lam=1.0)

    base_bbox = np.ptp(verts, axis=0)  # (W, D, H) of the polytope at scale 1
    out_dir = "output/lattice/versions"
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for tlabel, rod_r, joint_r in THICKNESS:
        for slabel, s in SCALES:
            sc_nodes = nodes * s
            vol_cm3 = material_volume_mm3(sc_nodes, rods, rod_r, joint_r) / 1000.0
            grams = vol_cm3 * PLA_DENSITY
            w, d, h = base_bbox * s
            rows.append((tlabel, rod_r, joint_r, slabel, f"{w:.0f}×{d:.0f}×{h:.0f}", vol_cm3, grams))

            # Export a sliceable STL for this version.
            mesh = build_frame_mesh(sc_nodes, rods, rod_radius=rod_r, joint_radius=joint_r,
                                    color_by_direction=False)
            stl = f"{out_dir}/rhombohedron_{tlabel}_{slabel}.stl"
            mesh.export(stl)
            register("lattice", stl, "stl",
                     f"Rhombohedron {tlabel}×{slabel} · rod Ø{2*rod_r:.1f} · {w:.0f}×{h:.0f} mm · {grams:.1f} g",
                     folder="Cube · rhombohedron · versions")

    # Markdown table
    print("| Thickness | rod Ø | joint Ø | Scale | W×D×H (mm) | Volume (cm³) | PLA (g) |")
    print("|---|---|---|---|---|---|---|")
    for tlabel, rod_r, joint_r, slabel, dims, vol, grams in rows:
        print(f"| {tlabel} | Ø{2*rod_r:.1f} | Ø{2*joint_r:.1f} | {slabel} | {dims} | {vol:.2f} | {grams:.1f} |")

    # CSV for the dashboard
    csv = "output/lattice/version_matrix.csv"
    with open(csv, "w") as f:
        f.write("thickness,rod_dia_mm,joint_dia_mm,scale,width_mm,depth_mm,height_mm,volume_cm3,mass_g\n")
        for tlabel, rod_r, joint_r, slabel, dims, vol, grams in rows:
            w, d, h = dims.split("×")
            f.write(f"{tlabel},{2*rod_r:.1f},{2*joint_r:.1f},{slabel[1:]},"
                    f"{w},{d},{h},{vol:.3f},{grams:.2f}\n")

    print(f"\nSaved {csv} + {len(rows)} STLs in {out_dir}")
    print(f"base polytope bbox (W×D×H) = {base_bbox[0]:.0f}×{base_bbox[1]:.0f}×{base_bbox[2]:.0f} mm; "
          f"{len(nodes)} nodes / {len(rods)} rods")


if __name__ == "__main__":
    main()
