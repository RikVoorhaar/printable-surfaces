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
3. **Constructive start** (Section 6.1) for every face: bottom-up sampler.
4. **MIP** (Section 7): β=0.6 per-rod reward; slacks s1/s2/sd/sf/sa; rod-rod clearance as pairwise
   exclusions (skipping shared-node pairs).
5. **Repair** (Section 8): slack-driven moves, single-pass I-node deletion, clearance-aware fallback.

## Current metrics (rhombohedron, spacing=15 / n_face=48 / n_interior=105)

| Metric | Value | Target |
|---|---|---|
| Unprintable rods | **0** | 0 |
| Rod-rod clearance violations | **0** | 0 |
| Mean degree | ~4.7–5.3 | 4.7 |
| Unsupported free nodes | **0** (bottom vertex sits on raft) | ~0 |
| Slices (PrusaSlicer) | OK | OK |

## Key bugs fixed (this session)

1. **Clearance — skip shared-node pairs**: was counting rods sharing a node as conflicts (distance
   0) → ~3,400 spurious exclusions (why clearance was infeasible in Phase 14).
2. **Jitter moved fixed frame nodes** off their edges → 42 unprintable rods.
3. **Missing face inset (0.35h)**: face nodes sat on the edges so rods crossed the frame.
4. **Frame-clearance — skip shared endpoints**: every face→frame rod was deleted (distance 0 vs the
   chain). This was the "faces not connected to edge points" bug; fix took frame degree 2.3→5.2.
5. **Bridge angle-penalty**: `0.3·exp(−(θ−48°)/4°)` ≈ 48,000 for a 0° bridge, silently forbidding
   horizontal rods.

## Known limitation — cross-slope face voids

The face lattice is inherently **columnar**: rods are steep (within ±36° of downhill), so the
perpendicular ("cross-slope") direction sits at ~22° elevation — too shallow for steep, too steep
for a bridge. This leaves elongated voids between the downhill columns. Density is the only lever
(max distance to nearest rod: ~6.7 mm at n_face=24, ~4.2 mm at n_face=64). Open: densify vs accept
columnar vs longer bridges (10→13 mm).

## Next steps

1. Decide the face-density / bridge trade-off (see above).
2. Then run tetrahedron / octahedron / icosahedron / dodecahedron through the pipeline.
