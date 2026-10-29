# Project status (compact)

**Goal:** fill a convex polytope with a rod-and-node lattice that FDM-prints with zero support.

**Printability rule:** a rod is printable iff its elevation is ≥ 48° (`SIN48`, any length), or it is
a horizontal bridge ≤ 10 mm (`SIN5`); rods on the build plate are exempt. **The angle rule is never
negotiable** — post-optimization may break length/degree limits, but never the 48° rule.

## Current version — T5×S2

Rods Ø1.8 mm (50 % thicker than T4's Ø1.2, which was fragile in practice), joints Ø2.7 mm (keeps
the 1.5 rod:joint ratio). Normalized to 200 cm³ shape volume. Output in
`output/lattice/minstretch_v2/`, dashboard folder `Platonic solids · min-stretch v2 (Ø1.8)`
(STL for slicing, PLY for per-rod angle colouring). Older versions kept.

| Solid | Orientation | W×D×H (mm) | nodes | rods | base |
|---|---|---|---|---|---|
| cube | face down | 60×60×60 | 305 | 986 | face on plate |
| tetrahedron | face down | 118×118×100 | 165 | 475 | face on plate |
| octahedron | face down | 86×86×63 | 306 | 903 | face on plate |
| icosahedron | vertex down (tip-lift) | 70×73×120 | 373 | 845 | 2 mm flat cut |
| dodecahedron | face down ×1.93 | 68×64×104 | 407 | 1102 | face on plate |

## Mesh

- Frames are closed solid shells: each convex primitive is clipped with shared (1e-6-snapped)
  vertices and an angle-ordered fan cap — no cracks, no hollow rods.
- Only **joint caps** are clipped, at face planes offset outward by half a rod diameter; face rods
  are tangent to that plane and stay whole. The bottom lattice therefore touches the plate
  (first-layer footprint 539–1532 mm² for the face-down solids; icosahedron 19 mm² from its flat
  base).
- Every mesh sits at `z = 0`.

## Post-optimization rules (`printable_lattice`)

Order: repair → delete unsupported → enforce min degree 2 (delete) → force min degree 3 (add) →
snap horizontals (iterated) → force upward links.

- `_snap_horizontal`: every rod with `|dz|/L ≤ sin 5°` becomes exactly horizontal (component mean;
  components spanning > 2 mm are skipped).
- `_force_min_degree(3)`: adds a rod to any node below degree 3; the length window may be broken.
- `_force_upward_links`: every node below the top plane gets a neighbour above it; the degree cap
  may be broken. The rod must still be printable (≥ 48°, or a legal ≤ 10 mm bridge), otherwise the
  node is skipped.
- Every forced rod is printable by construction: length/degree may be broken, the angle never.

## Verification (final)

| Solid | band rods 5°–48° | min deg | deg≤2 | horiz dz≠0 | dup/self | nodes w/o printable up-link |
|---|---|---|---|---|---|---|
| cube | 0 | 3 | 0 | 0 | 0 | 14 |
| tetrahedron | 0 | 3 | 0 | 0 | 0 | 0 |
| octahedron | 0 | 3 | 0 | 0 | 0 | 4 |
| icosahedron | 0 | 3 | 0 | 0 | 0 | 0 |
| dodecahedron | 0 | 3 | 0 | 0 | 0 | 0 |

“Band rods” are rods with `sin 5° < slope < sin 48°` — the unprintable angled rods that the forced
passes used to create. Now zero everywhere.

## Known limitations (geometric, not code bugs)

1. **Ceiling gap.** The topmost interior layer sits ~5.6 mm below a horizontal ceiling, so for
   14 cube / 4 octahedron nodes the maximum possible upward slope is ≤ 45° and no printable upward
   link exists. The fix belongs in the relaxation/interior sampling (place an interior node under
   each ceiling node), not in the forced pass.
2. **Top-face hatching rods > 10 mm** (cube 57, octahedron 32, dodecahedron 35): Delaunay hatching
   on the ceiling produces flat horizontal spans longer than the bridge limit. Subdividing them adds
   ceiling nodes and therefore more converging posts — the open ceiling problem.

## Method

Sliding-class relaxation (`docs/sliding-class-relaxation-method.md`); orientation and stretch from
`docs/Support-Free Rod Lattices for the Platonic Solids - Theory and Algorithm.md`. Node classes
V/E/F↓/F↑/I; ψ support + anchor + hard phase; constructive face start; MIP with β reward, slacks
and clearance; repair. Top-face/ring posts are MIP constraints (folded in), not post-hoc.

## Files

- `src/printable_surfaces/printable.py` — lattice pipeline + post-optimization passes.
- `src/printable_surfaces/platonic.py` — solid geometry, face polygons.
- `src/printable_surfaces/phase16.py` — version driver, volume normalization, clipping, export.
- `src/printable_surfaces/lattice.py` — thickening, clipping, per-rod colours.
- `plan/plan.html` — living plan (phases 0–21 + history).
- `AGENTS.md` — subagent policy (when to delegate, model routing, foreground rule).
