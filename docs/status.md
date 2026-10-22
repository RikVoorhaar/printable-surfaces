# Project status (compact)

**Goal:** fill a convex polytope (currently a rhombohedron = stretched cube, ~94×169 mm) with a
rod-and-node lattice that FDM-prints with zero support. Rods 1.2 mm, nodes 1.8 mm.

**Printability rule:** rod printable iff within 48° of vertical (α=48°), or a horizontal bridge
≤10 mm.

## Method (implemented in `src/printable_surfaces/printable.py`)

The sliding-class relaxation method (`docs/sliding-class-relaxation-method.md`):

1. **Node classes** V/E/F↓/F↑/I with restricted parent sets (Section 4); face nodes inset 0.35h
   from edges, interior inset 0.4h from faces.
2. **Relaxation** (Sections 5–6): quadratic repulsion `(r_c−d)²` with κ vertical anisotropy on
   I-pairs, ψ-based symmetric support `(sinα·L−Δz)₊² + (L−Lmax)₊² + (Lmin−L)₊²`, anchor term
   (F↑→I), and a hard final phase that forbids dropping a node below 2 parents.
3. **Constructive start** (Section 6.1) for F↓ faces: bottom-up sampler.
4. **MIP** (Section 7): β=0.6 per-rod reward, slacks on every lower bound, rod-rod clearance as
   pairwise exclusions (skipping shared-node pairs — fixes the old 3398 spurious conflicts).
5. **Repair** (Section 8): slack-driven moves, single-pass I-node deletion, clearance-aware
   fallback rod.

## Current metrics (rhombohedron, 3 seeds, spacing=15 / n_face=17 / n_interior=105)

| Metric | Value | Target |
|---|---|---|
| Unprintable rods | **0** | 0 |
| Rod-rod clearance violations | **0** | 0 |
| Mean degree | ~3.85 | 4.7 |
| Nodes with no downward rod | 2–3 (interior hubs) | ~0 |
| Slices (PrusaSlicer) | OK | OK |

## Remaining gaps

- **Degree ~3.85 vs 4.7**: each free node has only ~2.4 allowed steep parents → ~3.9 incident
  candidates after clearance. The β reward works, but there aren't enough printable candidates.
  Root cause: the 48° steep cone + 8–22 mm length window limits neighbours; κ trades steep share
  against vertical spacing.
- **2–3 unsupported interior nodes**: hubs that support a cluster above but have no steep parent
  below (their only parent was deleted). Single-pass deletion leaves a few; iterating cascades.

## Next steps

1. Raise steep-neighbour count (stronger vertical packing, or a smaller α margin), then re-check
   degree and unsupported together.
2. Then run tetrahedron/octahedron/icosahedron/dodecahedron through the pipeline.
