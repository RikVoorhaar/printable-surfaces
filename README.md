# printable-surfaces

Turn mathematical shapes into **printable rod-and-node lattices**: cages of cylindrical
rods and spherical joints that hold themselves up on an FDM printer, with no supports.

The main result is the five Platonic solids as support-free lattices. The story, including
the failures, is in the blog post
[An algorithm for organic 3D printable polytopes](https://rikvoorhaar.com/blog/3d_print_platonic_solids).

## The models

Five STLs, ready to slice. All are Ø1.8 mm rods with Ø2.7 mm joints, scaled to the same
200 cm³ shape volume, in millimetres, Z-up, and sitting on z = 0.

| Solid | Bounding box | PLA (est.) | File |
| --- | --- | --- | --- |
| Tetrahedron | 115 × 115 × 97 mm | 28.5 g | [tetrahedron.stl](models/platonic/tetrahedron.stl) |
| Cube | 58 × 58 × 58 mm | 42.2 g | [cube.stl](models/platonic/cube.stl) |
| Octahedron | 84 × 84 × 61 mm | 39.7 g | [octahedron.stl](models/platonic/octahedron.stl) |
| Icosahedron | 68 × 71 × 120 mm | 30.3 g | [icosahedron.stl](models/platonic/icosahedron.stl) |
| Dodecahedron | 66 × 62 × 102 mm | 37.6 g | [dodecahedron.stl](models/platonic/dodecahedron.stl) |

The dodecahedron and icosahedron are stretched vertically so their faces are steep enough
to print. They are no longer regular, but they still look like themselves.

## What "printable" means here

Every rod has to be at least about 48° above horizontal (45° plus a margin), or be a
horizontal bridge no longer than 10 mm, or lie on the build plate. Every node gets two
parents below it, at least 48° up, so nothing hangs in the air. Rod length is kept between
8 and 28 mm for looks, and node degree between 3 and 6. If a repair has to break a length
or degree limit, the angle rule still wins.

## How it works

1. Seed nodes on the surface and in the interior of the shape.
2. Relax them with pairwise repulsion, plus terms that pull nodes under overhanging faces
   and line up good parents, until the spacing is even and the parents are steep enough.
3. Pick a readable set of rods with a mixed-integer program, subject to the printability
   rule, the degree limits and rod-to-rod clearance.
4. Thicken the rods and joints into one watertight mesh and export an STL.

## Install

Python 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Regenerate the models

```bash
uv run python -m printable_surfaces.phase16
```

This writes the five STLs and coloured PLYs to `output/lattice/minstretch_v2/`, and
updates `manifest.json`. It takes a few minutes. The other `phase*.py` modules are earlier
experiments (sphere lattices, relaxations, slicer checks) kept for the record.

## Tests

```bash
uv run pytest
```

## Viewer

```bash
python3 serve.py
```

- Dashboard: <http://localhost:8000/>
- STL/PLY viewer: <http://localhost:8000/web/viewer.html>

Both read from `output/`, which is generated locally and not in the repository. Generate
something first if the dashboard looks empty. `models/platonic/` is checked in, so the
finished STLs are available without running anything.

## License

MIT. The code and the STLs, do what you like with them.
