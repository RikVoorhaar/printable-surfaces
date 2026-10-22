# Support-Free Printable Lattice — Current Algorithm Status

*(Self-contained status report for external review.)*

## Goal

Fill a convex polytope — a **rhombohedron** ≈ 91 mm wide × 169 mm tall (a cube with its
body diagonal vertical, stretched ×1.69 so all 12 frame edges sit at 50°) — with a
**rod-and-node lattice** that 3-D prints with **zero support material**. Rods are 1.2 mm
cylinders, nodes 1.8 mm spheres. Output STL is sliced with PrusaSlicer.

## Printability rule

Build direction is **+Z**. A rod is printable iff it is **within 45° of vertical**
(self-supporting), **or** nearly horizontal and **≤ 10 mm** (a bridge). The algorithm uses
**48°** (3° margin).

## Current algorithm

1. **Frame** — the 12 polytope edges (subdivided to ~15 mm) are *fixed* nodes + rods; all
   at 50°, printable.
2. **Seed** — face points (uniform on each rhombus) + interior points (uniform in the
   volume), ~210 free points total.
3. **Relaxation** (900 iterations):
   - finite-range repulsion (cutoff 1.5×spacing);
   - **anisotropic** repulsion for face nodes — weaker along the face's downhill direction
     (weight 0.7), so face nodes stack into downhill chains;
   - **support term** — a *symmetric* spring toward each free node's 2 best "parents"
     (steep nodes 8–22 mm below, rest length 15 mm), ramped up after iteration 300; face
     nodes additionally accept downhill same-face parents;
   - face nodes are projected back onto their rhombus each step; interior nodes clamped
     inside (0.4×spacing margin).
4. **Candidate rods** — only printable pairs (steep ≥48° & 8–22 mm, or horizontal bridge
   ≤10 mm); exclude frame–frame, cross-face, and any rod within 1.6 mm of another node or
   a frame rod.
5. **Integer program** (SciPy `milp`/HiGHS): choose rods (0/1 per candidate) so every free
   node has ≥1 downward rod (≥2 if it has 2 candidates), degree 3–6, minimising
   `(length − 15)²` + an angle penalty.
6. **Post-fix** — any node still lacking a downward rod gets a steep rod to its nearest
   node below.
7. Thicken → STL/PLY → slice.

## What works

- **0 unprintable rods** across seeds — every emitted rod is steep or a bridge.
- **Distributed fill** — after making the support term symmetric (equal-and-opposite force
  on the parent), nodes span the full height (z ≈ −61…+61 in a −84…+84 shape) instead of
  piling into the bottom half.
- Slices cleanly in PrusaSlicer.

## What's wrong

1. **~4–7 unsupported nodes** (no downward rod). They are all face nodes on the three
   *downward*-facing faces, where the steep "cone" below is mostly outside the shape. The
   in-face downhill rule was added but doesn't fully fix it — the face nodes still aren't
   aligned downhill enough.
2. **Degree ~3.3 vs target ~4.7** — the lattice is sparser than intended (and has
   seed-to-seed variance; one seed gave 2.0).
3. **Rod–rod clearance is disabled** — enforcing "no two rods closer than 2.6 mm" produced
   ~3400 exclusion pairs and made the MIP infeasible, so it's dropped.
4. **Trade-off** — the symmetric support fixes the bottom-heavy distribution but raises
   unsupported nodes slightly (the asymmetric version gave ~1 unsupported but was
   bottom-heavy). We can't yet get *both* distributed *and* ~0 unsupported.

## Questions

1. How to eliminate the unsupported nodes on downward faces (the in-face downhill
   alignment) without breaking the distribution?
2. How to raise the degree toward ~4.7 (denser lattice) while keeping every rod printable?
3. How to enforce the 2.6 mm rod–rod clearance without making the integer program
   infeasible?
