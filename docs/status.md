# Project status (compact)

**Goal:** fill a convex polytope (currently a rhombohedron = stretched cube, ~94×169 mm) with a
rod-and-node lattice that FDM-prints with zero support. Rods 1.2 mm, nodes 1.8 mm.

**Printability rule:** rod printable iff within 48° of vertical (α=48°), or a horizontal bridge
≤10 mm.

## Method (implemented)

The sliding-class relaxation method (`docs/sliding-class-relaxation-method.md`) is implemented in
`src/printable_surfaces/printable.py`:

1. **Node classes** V (vertices, fixed) / E (edge, fixed for now) / F↓ (overhanging face) / F↑
   (upward face) / I (interior), with **restricted parent sets** (Section 4): F↓ may only use
   same-face F + frame-on-face (not I); F↑ may also use I; I uses any.
2. **Relaxation** (Sections 5–6): quadratic repulsion `(r_c−d)²` with κ vertical anisotropy on
   I-pairs, ψ-based symmetric support `(sinα·L−Δz)₊² + (L−Lmax)₊² + (Lmin−L)₊²`, anchor term
   (F↑→I), and a hard final phase that forbids any move dropping a node below 2 parents.
3. **Constructive start** (Section 6.1) for F↓ faces: bottom-up sampler, each node placed where
   ≥2 in-face parents already exist.
4. **MIP rod selection** (Section 7): β=0.6 per-rod reward, slacks on every lower bound (never
   infeasible), rod-rod clearance as pairwise exclusions (skipping shared-node pairs — the fix
   for the old 3398 spurious conflicts).
5. **Repair** (Section 8): slack-driven random moves, delete I nodes with no downward rod,
   clearance-aware fallback rod.

## Current metrics (rhombohedron, 3 seeds, spacing=15 / n_face=17 / n_interior=105)

| Metric | Value | Target |
|---|---|---|
| Unprintable rods | **0** | 0 |
| Rod-rod clearance violations | **0** | 0 |
| Mean degree | ~3.7 | 4.7 |
| Nodes with no downward rod | 5–7 (all F↑ near top) | ~0 |
| Slices (PrusaSlicer) | OK | OK |

## Remaining gaps

- **Degree ~3.7 vs 4.7**: limited by candidate count — each free node has ~2.4 allowed steep
  parents → ~3.9 incident candidates after clearance. The β reward works (degree ~3.2 without
  it) but there aren't enough printable candidates to reach 4.7.
- **5–7 F↓…F↑ unsupported nodes**: the F↓ overhanging-face nodes are now handled by the
  constructive start. The remaining unsupported nodes are **F↑ nodes near the top vertex**, which
  have 3–4 raw steep parents but whose candidate rods are blocked by **rod-rod clearance** (the
  shape tapers to a point at the top, so rods cross in that narrow region). These would print as
  dangling nodes.

## Next steps

1. F↑ top-vertex nodes: reduce crowding near the top (e.g. interior inset grows toward the tip,
   or taper the interior fill), or accept a slightly-relaxed rod-rod clearance for the last few.
2. Degree: raise the steep-neighbour share without pushing vertical spacing past L_MAX (κ trades
   these off); try larger repulsion cutoff or different interior count.
3. Then run tetrahedron/octahedron/icosahedron/dodecahedron through the pipeline.
