"""Phase 12 — variable-degree surface wiring (fills surface voids).

Same Coulomb-gas point cloud as Phase 11, but wired with a radius-dependent degree:
higher degree near the surface (to fill voids), sparser in the interior.

Run:  uv run python -m printable_surfaces.phase12
"""

from __future__ import annotations

import os

import numpy as np

from .artifacts import register
from .lattice import build_frame_mesh
from .relax import coulomb_points, wire_surface_reinforced
from .slicer import slice_stl

OBJECT_MM = 100.0
ROD_MM = 1.2
JOINT_MM = 1.8
# (name, label, n_points, k_base, k_surface)
VARIANTS = [
    ("coulomb_ki4_ks6", "Coulomb · base 4 + surf 6", 160, 4, 6),
    ("coulomb_ki4_ks7", "Coulomb · base 4 + surf 7", 160, 4, 7),
    ("coulomb_ki3_ks6", "Coulomb · base 3 + surf 6", 160, 3, 6),
    ("coulomb_ki4_ks5", "Coulomb · base 4 + surf 5", 160, 4, 5),
]


def build_and_register(radius: float, name: str, label: str, n: int, kb: int, ks: int) -> None:
    nodes = coulomb_points(radius, n, seed=1)
    edges = wire_surface_reinforced(nodes, radius, k_base=kb, k_surface=ks)
    mesh = build_frame_mesh(nodes, edges, rod_radius=ROD_MM, joint_radius=JOINT_MM)

    # Degree stats (interior vs surface band).
    deg = np.zeros(len(nodes), dtype=int)
    np.add.at(deg, edges[:, 0], 1)
    np.add.at(deg, edges[:, 1], 1)
    r = np.linalg.norm(nodes, axis=1)
    band = float(np.linalg.norm(nodes[edges[:, 0]] - nodes[edges[:, 1]], axis=1).mean()) * 1.2
    boundary = r > radius - band
    deg_in = deg[~boundary]
    deg_surf = deg[boundary]

    edge_len = np.linalg.norm(nodes[edges[:, 0]] - nodes[edges[:, 1]], axis=1)

    path = f"output/lattice/sphere_{name}.stl"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mesh.export(path)
    register("lattice", path, "stl", f"{label} · {len(nodes)} nodes / {len(edges)} rods")

    gcode = f"output/slicer/sphere_{name}.gcode"
    try:
        proc = slice_stl(path, gcode)
        ok = proc.returncode == 0 and os.path.exists(gcode) and os.path.getsize(gcode) > 0
    except FileNotFoundError:
        ok = False
    status = "sliced OK" if ok else "FAILED"
    if ok:
        register("slicer", gcode, "gcode", f"{label} · {status}")

    print(
        f"{label:26s} {len(nodes):3d} nodes  {len(edges):4d} rods  "
        f"deg interior={deg_in.mean():.1f} surface={deg_surf.mean():.1f}  "
        f"rod {edge_len.mean():.1f}±{edge_len.std():.1f}  -> {status}"
    )


def main() -> None:
    radius = OBJECT_MM / 2.0
    for name, label, n, kb, ks in VARIANTS:
        build_and_register(radius, name, label, n, kb, ks)


if __name__ == "__main__":
    main()
