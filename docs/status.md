# Project status (compact)

**Goal:** fill a convex polytope (currently a rhombohedron = stretched cube, ~94×169 mm) with a
rod-and-node lattice that FDM-prints with zero support. Rods 1.2 mm, nodes 1.8 mm.

**Printability rule:** rod printable iff within 48° of vertical (α=48°), or a horizontal bridge
≤10 mm.

**Repos & docs:** `plan/plan.html` (living plan), `docs/` (problem statements + external method
`sliding-class-relaxation-method.md`), `web/` (three.js viewer, served on Tailscale), `src/printable_surfaces/`.

**Code modules:**
- `surface.py`, `parametrizations.py` — cube-sphere surface (Phase 1, done).
- `lattice.py` — `build_frame_mesh` (rods→cylinders, nodes→spheres, optional rod-direction colouring).
- `platonic.py` — `cube_rhombohedron`, face planes/parallelograms.
- `relax.py` — Coulomb gas (Phase 11), jittered/random lattices (Phase 9/12 experiments).
- `printable.py` — the support-free pipeline (Phase 14): relax → candidates → MIP → thicken.
- `phase*.py` — runnable phases; `artifacts.py` — dashboard manifest; `slicer.py` — PrusaSlicer CLI.

**Current printable pipeline** (`printable_lattice`):
1. frame (12 fixed edges, subdivided) → seed face + interior points → relaxation (finite-range
   repulsion + anisotropic face repulsion + symmetric support spring + face/interior constraints)
   → printable candidates (steep 48°/8–22mm or bridge) → MIP (≥1 downward rod, degree 3–6, cost
   (length−15)²) → post-fix (rod to nearest node below).

**Current metrics (rhombohedron, 3 seeds):** 0 unprintable rods; ~4–7 unsupported nodes (downward
faces); degree ~3.3 (target 4.7); rod-rod clearance disabled (was infeasible); nodes distributed
after symmetric support.

**Plan:** implement the sliding-class relaxation method (`docs/sliding-class-relaxation-method.md`):
node classes V/E/F↓/F↑/I with sliding projections, restricted parent sets, ψ-based support energy +
anchor, 3-phase schedule with hard support, constructive start for F↓, and an MIP with β reward +
slacks + clearance fix (skip shared-node pairs).
