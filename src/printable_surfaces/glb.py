"""Export a lattice mesh to a compact, quantised binary glTF (``.glb``) for the web.

The browser viewer needs indexed geometry with small transfers, not the raw binary STL
that the slicer eats. A 4 MB STL of ~83k triangles lands at a few hundred kB here, with
no visible loss on 1.8 mm rods:

* **welded + indexed** — STL is a triangle soup; coincident vertices are merged so the
  GPU can reuse them (and smooth normals come out of it for free);
* **Y-up** — glTF's convention. The project builds in Z-up, so the mesh is rotated a
  fixed −90° about X. This is a convention, not a fudge: the face that was on the build
  plate still rests on the table.
* **placed on the origin** — centred in X/Z, lowest point at Y = 0, so the viewer can put
  it straight onto a surface without guessing;
* **quantised** — positions as normalised int16 and normals as normalised int8 under
  ``KHR_mesh_quantization``, which Three.js decodes natively (no decoder to ship).

Usage::

    uv run python -m printable_surfaces.glb output/lattice/minstretch_v2/cube_T5_S2.stl out.glb
    uv run python -m printable_surfaces.glb --help
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import trimesh

# glTF component types
_BYTE = 5120
_UBYTE = 5121
_SHORT = 5122
_UINT = 5125
_USHORT = 5123

# glTF accessor types
_SCALAR, _VEC3, _VEC4 = "SCALAR", "VEC3", "VEC4"


def _pad(data: bytes, fill: bytes = b"\x00") -> bytes:
    """Pad ``data`` to a 4-byte boundary (glTF requires aligned chunks)."""
    remainder = len(data) % 4
    return data if remainder == 0 else data + fill * (4 - remainder)


def load_mesh(path: str | Path, *, weld: bool = True) -> trimesh.Trimesh:
    """Load an STL/PLY as a single indexed trimesh, optionally welding duplicate vertices."""
    mesh = trimesh.load(str(path), force="mesh")
    if isinstance(mesh, trimesh.Scene):  # pragma: no cover - defensive
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):  # pragma: no cover - defensive
        raise TypeError(f"expected a single mesh, got {type(mesh).__name__}")

    if getattr(mesh.visual, "kind", None) == "vertex":
        # Already an indexed mesh carrying per-vertex data (a coloured PLY, say). Rebuilding
        # or welding it here would drop or scramble the colour attribute, and the export is
        # welded already, so take it as it is.
        return trimesh.Trimesh(
            vertices=mesh.vertices, faces=mesh.faces, process=False, visual=mesh.visual
        )

    mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=True)
    if weld:
        mesh.merge_vertices()
    return mesh


def to_y_up(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Rotate the mesh from the project's Z-up build frame into glTF's Y-up frame."""
    out = mesh.copy()
    out.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1.0, 0.0, 0.0]))
    return out


def place_on_origin(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Centre in X/Z and drop the lowest point to Y = 0 (a pure translation)."""
    out = mesh.copy()
    lo, hi = out.bounds
    centre = (lo + hi) / 2.0
    out.apply_translation([-centre[0], -lo[1], -centre[2]])
    return out


def _quantize_positions(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Map positions into normalised int16, returning ``(q, translation, scale)``."""
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    centre = (lo + hi) / 2.0
    half = float(np.max(hi - lo)) / 2.0 or 1.0
    q = np.round((vertices - centre) / half * 32767.0)
    return np.clip(q, -32767, 32767).astype(np.int16), centre, half


def _quantize_normals(normals: np.ndarray) -> np.ndarray:
    """Map unit normals into normalised int8."""
    q = np.round(normals * 127.0)
    return np.clip(q, -127, 127).astype(np.int8)


def _linear_colors(colors: np.ndarray) -> np.ndarray:
    """sRGB RGBA bytes → linear RGBA bytes.

    glTF defines ``COLOR_0`` as linear. trimesh stores what you gave it (sRGB hex, in
    practice), so writing it straight through would wash every colour out in the viewer.
    """
    srgb = colors[:, :3].astype(np.float64) / 255.0
    linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    rgba = np.concatenate([np.round(linear * 255.0), colors[:, 3:4].astype(np.float64)], axis=1)
    return np.clip(rgba, 0, 255).astype(np.uint8)


def _mesh_colors(mesh: trimesh.Trimesh, vertex_count: int) -> np.ndarray | None:
    """The mesh's per-vertex colours, or ``None`` if it does not carry any."""
    visual = getattr(mesh, "visual", None)
    if visual is None or getattr(visual, "kind", None) != "vertex":
        return None
    colors = np.asarray(visual.vertex_colors)
    if len(colors) != vertex_count or len(colors) == 0:
        return None
    if colors.dtype != np.uint8:
        scale = 255.0 if colors.max() <= 1.0 else 1.0
        colors = np.clip(colors * scale, 0, 255).astype(np.uint8)
    # A single flat colour is the material's job, not an attribute's.
    if float(np.ptp(colors[:, :3].astype(np.int16))) == 0.0:
        return None
    return colors[:, :4]


def _buffer_view(blob: bytearray, payload: bytes, target: int | None = None) -> dict:
    """Append ``payload`` to ``blob`` and return the matching bufferView dict."""
    offset = len(blob)
    blob.extend(_pad(payload))
    view = {"buffer": 0, "byteOffset": offset, "byteLength": len(payload)}
    if target is not None:
        view["target"] = target
    return view


def glb_bytes(
    mesh: trimesh.Trimesh,
    *,
    quantize: bool = True,
    name: str = "lattice",
    color: tuple[float, float, float, float] = (1.0, 0.42, 0.05, 1.0),
) -> bytes:
    """Serialise ``mesh`` (already Y-up, already placed) as a single-primitive GLB."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)

    blob = bytearray()
    views: list[dict] = []
    accessors: list[dict] = []
    extensions_used: list[str] = []
    node: dict = {"mesh": 0, "name": name}

    if quantize:
        positions, translation, scale = _quantize_positions(vertices)
        extensions_used.append("KHR_mesh_quantization")
        node["translation"] = [float(v) for v in translation]
        node["scale"] = [scale, scale, scale]
        pos_type, pos_index = _SHORT, "position"
        normal_data, normal_type = _quantize_normals(normals), _BYTE
    else:
        positions = vertices.astype(np.float32)
        pos_type, pos_index = 5126, "position"
        normal_data, normal_type = normals.astype(np.float32), 5126

    views.append(_buffer_view(blob, positions.tobytes(), target=34962))
    accessors.append(
        {
            "bufferView": len(views) - 1,
            "componentType": pos_type,
            "count": int(len(positions)),
            "type": _VEC3,
            "normalized": bool(quantize),
            "min": [float(v) for v in positions.min(axis=0)],
            "max": [float(v) for v in positions.max(axis=0)],
        }
    )

    views.append(_buffer_view(blob, normal_data.tobytes(), target=34962))
    accessors.append(
        {
            "bufferView": len(views) - 1,
            "componentType": normal_type,
            "count": int(len(normal_data)),
            "type": _VEC3,
            "normalized": bool(quantize),
        }
    )

    colors = _mesh_colors(mesh, len(vertices))
    color_accessor = None
    if colors is not None:
        views.append(_buffer_view(blob, _linear_colors(colors).tobytes(), target=34962))
        color_accessor = len(accessors)
        accessors.append(
            {
                "bufferView": len(views) - 1,
                "componentType": _UBYTE,
                "count": int(len(colors)),
                "type": _VEC4,
                "normalized": True,
            }
        )

    index_type = _USHORT if len(vertices) < 65536 else _UINT
    index_dtype = np.uint16 if index_type == _USHORT else np.uint32
    indices = faces.astype(index_dtype).ravel()
    views.append(_buffer_view(blob, indices.tobytes(), target=34963))
    index_accessor = len(accessors)
    accessors.append(
        {
            "bufferView": len(views) - 1,
            "componentType": index_type,
            "count": int(len(indices)),
            "type": _SCALAR,
        }
    )

    attributes: dict = {pos_index.upper(): 0, "NORMAL": 1}
    if color_accessor is not None:
        attributes["COLOR_0"] = color_accessor

    tree: dict = {
        "asset": {"version": "2.0", "generator": "printable-surfaces glb.py"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [node],
        "meshes": [
            {
                "name": name,
                "primitives": [
                    {
                        "attributes": attributes,
                        "indices": index_accessor,
                        "material": 0,
                        "mode": 4,
                    }
                ],
            }
        ],
        "materials": [
            {
                "name": "PLA",
                "doubleSided": True,
                "pbrMetallicRoughness": {
                    # Vertex colours multiply this, so a coloured mesh gets white.
                    "baseColorFactor": [1.0, 1.0, 1.0, 1.0] if colors is not None else list(color),
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.62,
                },
            }
        ],
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": len(blob)}],
    }
    if extensions_used:
        tree["extensionsUsed"] = extensions_used

    json_chunk = _pad(json.dumps(tree, separators=(",", ":")).encode("utf-8"), b" ")
    bin_chunk = _pad(bytes(blob))
    total = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    header = struct.pack("<III", 0x46546C67, 2, total)
    json_header = struct.pack("<II", len(json_chunk), 0x4E4F534A)
    bin_header = struct.pack("<II", len(bin_chunk), 0x004E4942)
    return b"".join([header, json_header, json_chunk, bin_header, bin_chunk])


def export_glb(
    src: str | Path,
    dst: str | Path,
    *,
    quantize: bool = True,
    weld: bool = True,
    y_up: bool = True,
) -> dict:
    """Convert ``src`` (STL/PLY) to a web-ready GLB at ``dst``. Returns conversion stats."""
    mesh = load_mesh(src, weld=weld)
    if y_up:
        mesh = to_y_up(mesh)
    mesh = place_on_origin(mesh)
    data = glb_bytes(mesh, quantize=quantize, name=Path(src).stem)

    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(data)

    return {
        "src": str(src),
        "dst": str(dst),
        "triangles": int(len(mesh.faces)),
        "vertices": int(len(mesh.vertices)),
        "src_bytes": Path(src).stat().st_size,
        "dst_bytes": len(data),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m printable_surfaces.glb",
        description="Convert lattice STL/PLY meshes to compact quantised GLB for the web viewer.",
    )
    parser.add_argument("src", nargs="+", help="source .stl/.ply file(s)")
    parser.add_argument(
        "-o",
        "--out",
        help="output .glb path, or output directory when several sources are given",
    )
    parser.add_argument(
        "--no-quantize", action="store_true", help="store float32 positions/normals instead"
    )
    parser.add_argument("--no-weld", action="store_true", help="keep the triangle soup unindexed")
    parser.add_argument(
        "--no-y-up",
        action="store_true",
        help="keep the project's Z-up frame instead of converting to glTF's Y-up",
    )
    args = parser.parse_args(argv)

    sources = [Path(p) for p in args.src]
    if args.out is None:
        raise SystemExit("error: -o/--out is required")
    out = Path(args.out)
    if len(sources) > 1 and out.suffix.lower() == ".glb":
        raise SystemExit("error: with several sources, -o must be a directory")

    for src in sources:
        dst = out if len(sources) == 1 and out.suffix else out / f"{src.stem}.glb"
        stats = export_glb(
            src,
            dst,
            quantize=not args.no_quantize,
            weld=not args.no_weld,
            y_up=not args.no_y_up,
        )
        ratio = stats["dst_bytes"] / max(stats["src_bytes"], 1)
        print(
            f"{stats['dst']}  {stats['triangles']:,} tris  {stats['vertices']:,} verts  "
            f"{stats['src_bytes'] / 1e6:.1f} MB -> {stats['dst_bytes'] / 1e3:.0f} kB "
            f"({ratio:.0%} of source)"
        )


if __name__ == "__main__":
    main()
