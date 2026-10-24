# Project status (compact)

**Goal:** fill a convex polytope with a rod-and-node lattice that FDM-prints with zero support.
Rods 1.2 mm, nodes 1.8 mm (chosen "slim" Ø), build direction +Z.

**Printability rule:** rod printable iff within 48° of vertical (α=48°), or a horizontal bridge
≤10 mm; rods lying on the build plate are exempt.

## Solids (minimal-stretch orientation)

Each solid is oriented with its **least-stretch** axis vertical: face-down where that's easier,
vertex-down where face-down would need more stretch.

| Solid | Orientation | Stretch | Horizontal faces |
|---|---|---|---|
| cube | face down | none | bottom (plate) + top (open) |
| tetrahedron | face down | none | bottom (plate) |
| octahedron | face down | none | bottom + top |
| icosahedron | vertex down | tip-lift | 2 pentagon rings (bridged) |
| dodecahedron | face down | ×1.93 | bottom + top |

**Horizontal-face exceptions** (in `printable.py`): the bottom face is on the plate (its edges
subdivide at ~spacing to provide parents, no posts); the top face is a horizontal ceiling left
open (no face nodes, its edges become ≤10 mm bridges + posts).

## Chosen version — T4×S2 (Ø1.2 rod, 0.8 scale)

| Solid | W×D×H (mm) | nodes | rods | PLA (g) |
|---|---|---|---|---|
| cube | 46×46×46 | 174 | 438 | 7.7 |
| tetrahedron | 63×63×53 | 107 | 226 | 3.4 |
| octahedron | 63×63×46 | 197 | 490 | 8.0 |
| icosahedron | 68×72×121 | 380 | 810 | 13.5 |
| dodecahedron | 79×75×123 | 280 | 632 | 13.0 |

All 0 unprintable. STLs in `output/lattice/minstretch/`, dashboard folder `Platonic solids ·
min-stretch` (earlier `… · T4×S2` versions left in place).

## Method (sliding-class relaxation, `docs/sliding-class-relaxation-method.md`)

Node classes V/E/F↓/F↑/I; ψ support + anchor + hard phase; constructive bottom-up start for
faces; MIP (β reward + slacks + clearance); repair; bridge posts. Generalized to polygon faces
(triangle/rhombus/pentagon). Key bugs fixed: shared-node clearance, fixed-node jitter, face
inset, frame-clearance, bridge angle-penalty.

## Known limitation

Icosahedron's upper pentagon ring: ~16 bridge nodes have no post (their posts would run along
the near-vertical band face and collide with its face lattice). Fix = fold posts into the MIP as
fixed parent requirements (follow-up).

## Next steps

1. Fold icosahedron ring posts into the MIP (or accept ~16 bridge nodes).
2. Slice a chosen solid for a real print-time + filament estimate.
3. Scale the face-down solids (cube/octa are ~46 mm at 0.8 — quite small) if a consistent size
   across solids is wanted.
