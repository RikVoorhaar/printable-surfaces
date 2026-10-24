# Project status (compact)

**Goal:** fill a convex polytope with a rod-and-node lattice that FDM-prints with zero support.
Rods 1.2 mm, nodes 1.8 mm (chosen "slim" Ø), build direction +Z.

**Printability rule:** rod printable iff within 48° of vertical (α=48°), or a horizontal bridge
≤10 mm.

## Solids

All five Platonic solids, oriented + stretched so their frame edges are printable (per
`docs/Support-Free Rod Lattices for the Platonic Solids…`):

| Solid | Orientation | Deformation | Horizontal edges |
|---|---|---|---|
| cube | vertex down (body diagonal vertical) | ×1.69 | none |
| tetrahedron | face down | none | base (raft) |
| octahedron | vertex down | ×1.19 | 4 belt, 70.7 mm |
| icosahedron | vertex down | tip-lift ×1.51 | 10 (two pentagon rings) |
| dodecahedron | face down | ×1.93 (relaxed) | 5 top pentagon |

Geometry in `platonic.py`; lattice in `printable.py` (generalized: polygon faces, horizontal
edges → ≤10 mm bridges + steep posts, stranded-node cleanup).

## Chosen version — T4×S2 (Ø1.2 rod, 0.8 scale, spacing 17)

| Solid | W×D×H (mm) | nodes | rods | PLA (g) |
|---|---|---|---|---|
| cube | 73×73×135 | 337 | 851 | 15.2 |
| tetrahedron | 63×63×53 | 86 | 180 | 2.9 |
| octahedron | 80×80×95 | 245 | 612 | 10.3 |
| icosahedron | 68×72×121 | 380 | 823 | 13.5 |
| dodecahedron | 79×75×123 | 280 | 645 | 13.3 |

All 0 unprintable rods. STLs in `output/lattice/solids/`, registered on the dashboard under
`Platonic solids · T4×S2`.

## Method (sliding-class relaxation, `docs/sliding-class-relaxation-method.md`)

Node classes V/E/F↓/F↑/I; ψ support + anchor + hard phase; constructive bottom-up start for
faces; MIP (β reward + slacks + clearance); repair; bridge posts. Key bugs fixed: shared-node
clearance, fixed-node jitter, face inset, frame-clearance, bridge angle-penalty.

## Known limitation

Icosahedron's upper pentagon ring: ~11 bridge nodes have no post (their posts would run along
the near-vertical band face and collide with its face lattice). Fix = fold posts into the MIP
as fixed parent requirements (follow-up).

## Next steps

1. Fold icosahedron ring posts into the MIP (or accept 11 bridge nodes).
2. Slice a chosen solid (T4×S2) for real print-time + filament estimate.
3. Orient / brim / stabilizer-strut decisions (vertex-down solids stand on a point).
