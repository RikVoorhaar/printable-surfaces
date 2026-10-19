"""Rod + node lattice construction.

Turns a triangulated surface (or any node/edge frame) into a printable mesh by
thickening edges into cylinders and nodes into spheres.
"""

from __future__ import annotations

import numpy as np
import trimesh


def merge_vertices(vertices: np.ndarray, decimals: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Merge near-coincident vertices.

    Returns ``(merged_vertices, inverse)`` where ``inverse`` maps each original vertex
    index to its merged index (used to remap faces). Chart boundaries in the cube-sphere
    share exact sample points, so this closes the "cracks" between charts.
    """
    _, idx, inverse = np.unique(
        np.round(np.asarray(vertices, dtype=float), decimals),
        axis=0,
        return_index=True,
        return_inverse=True,
    )
    return np.asarray(vertices)[idx], inverse


def edges_from_faces(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Unique undirected edges ``(E, 2)`` of a triangle mesh (the 1-skeleton)."""
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return np.asarray(mesh.edges_unique, dtype=int)


def build_frame_mesh(
    nodes: np.ndarray,
    edges: np.ndarray,
    rod_radius: float,
    joint_radius: float,
    rod_sections: int = 12,
    joint_count: tuple[int, int] = (6, 6),
) -> trimesh.Trimesh:
    """Thicken a node+edge frame into a single mesh.

    Rods become cylinders, nodes become spheres, and the parts are concatenated into one
    triangle soup. (This is intentionally *not* boolean-unioned; overlapping joints are
    fine for viewing and slicing, and a true watertight union is a later refinement.)
    """
    parts: list[trimesh.Trimesh] = []
    for a, b in edges:
        seg = np.asarray([nodes[a], nodes[b]], dtype=float)
        if np.linalg.norm(seg[1] - seg[0]) < 1e-9:
            continue  # skip degenerate (zero-length) edges
        parts.append(trimesh.creation.cylinder(radius=rod_radius, segment=seg, sections=rod_sections))
    for p in np.asarray(nodes, dtype=float):
        s = trimesh.creation.uv_sphere(radius=joint_radius, count=joint_count)
        s.apply_translation(p)
        parts.append(s)
    return trimesh.util.concatenate(parts)
