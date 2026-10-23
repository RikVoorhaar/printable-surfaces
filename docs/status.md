# Project status (compact)

**Goal:** fill a convex polytope (currently a rhombohedron = stretched cube, ~94×169 mm) with a
rod-and-node lattice that FDM-prints with zero support. Rods 1.2 mm, nodes 1.8 mm.

**Printability rule:** rod printable iff within 48° of vertical (α=48°), or a horizontal bridge
≤10 mm.

## Method (implemented in `src/printable_surfaces/printable.py`)

The sliding-class relaxation method (`docs/sliding-class-relaxation-method.md`):

1. **Node classes** V/E/F↓/F↑/I with restricted parent sets (Section 4); face nodes inset 0.35h
   from edges, interior inset 0.4h from faces.
2. **Relaxation** (Sections 5–6): quadratic repulsion `(r_c−d)²` with κ vertical anisotropy,
   ψ-based symmetric support, anchor term, and a hard final phase forbidding a move that drops a
   node below 2 parents.
3. **Constructive start** (Section 6.1) for every face: bottom-up sampler, each node placed where
   ≥2 in-face parents already exist.
4. **MIP** (Section 7): β=0.6 per-rod reward; slacks s1/s2/sd/sf/sa on every lower bound; rod-rod
   clearance as pairwise exclusions (skipping shared-node pairs).
5. **Repair** (Section 8): slack-driven moves, single-pass I-node deletion, clearance-aware
   fallback rod.

## Current metrics (rhombohedron, 3 seeds, spacing=15 / n_face=17 / n_interior=105)

| Metric | Value | Target |
|---|---|---|
| Unprintable rods | **0** | 0 |
| Rod-rod clearance violations | **0** | 0 |
| Mean degree (overall) | ~4.65 | 4.7 |
| Frame / face / interior degree | 4.2 / 4.2 / 5.4 | — |
| Unsupported free nodes | **0** (bottom vertex sits on raft) | ~0 |
| Slices (PrusaSlicer) | OK | OK |

## Key bug fixed this session

`candidate_rods` deleted every face→frame rod: the frame-clearance check tested candidate rods
against frame rods **sharing an endpoint** (distance 0), which is exactly what the method warns
against ("a frame rod it does *not* share an endpoint with"). Fixing it raised frame degree
2.3 → 4.2 and face degree 3.5 → 4.2 — the "faces not connected to the edges" problem.

## Next steps

1. Run tetrahedron / octahedron / icosahedron / dodecahedron through the pipeline.
2. Optional: flatten the rod-elevation histogram (rods cluster just above 48°).
