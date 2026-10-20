# 3D-Printable Lattice Generation — Problem & Current State

*(Written as a self-contained briefing for an external reviewer.)*

## Goal

Generate a **lattice** — a rod-and-node space frame — that fills a 3-D shape (starting
with a sphere) and can be 3-D printed **with zero support material**. The lattice should
also *look good*: slightly organic/random, but with a sense of uniformity.

Concrete current target: a **sphere of radius 50 mm** (100 mm diameter), filled with ~160
vertices connected by ~500 rods. Rods are thickened to **1.2 mm** cylinders; vertices to
**1.8 mm** spheres. Output is a binary STL, verified by slicing headlessly with PrusaSlicer.

## The printability constraint (the hard part)

Build direction is **+Z** (vertical, "up"). A rod (an edge between two vertices) is
printable **without support** if and only if one of:

1. It is **within 45° of vertical** — i.e. its angle from the horizontal plane is ≥ 45°
   (overhang ≤ 45°, self-supporting), *or*
2. It is **nearly horizontal and short enough to bridge** — length ≤ **10 mm** for PLA.

Any other rod (a shallow slope, i.e. between 0° and 45° from horizontal, that is *not* a
short bridge) will sag and needs support material, which we want to avoid entirely.

## Pipeline

1. Place N vertices inside the sphere.
2. Wire them into a graph (edges = rods).
3. Thicken edges into cylinders + vertices into spheres → one mesh.
4. Export STL, slice with PrusaSlicer to verify.

## What we tried (and learned)

1. **Separate surface + interior lattices** → they have to share nodes; we moved to a
   unified lattice.
2. **Diamond (cubic) crystal lattice** clipped to the sphere → too regular (rejected).
3. **Jittered diamond / random k-NN** → organic, but not uniform enough.
4. **Force-directed spring relaxation** (Hooke springs on a *fixed* k-NN graph + repulsion
   + a soft wall):
   - Fruchterman–Reingold's always-attractive spring pulled interior vertices onto the
     surface (switched to Hooke's law).
   - Long-range 1/r² repulsion packed everything at the boundary — the **shell theorem**
     (a uniform surface shell exerts zero net force on interior points), so nothing
     confined them (switched to finite-range repulsion + a soft wall).
   - Result: uniform rod lengths, but the **interior was not uniform** — the *fixed graph
     topology* locked vertices into awkward positions.
5. **Coulomb gas (current, and the good one)**:
   - Treat the N vertices as point charges that repel (force ∝ 1/r²) inside a **harmonic
     trap** pulling toward the center (F = −k·r).
   - This is the *confined Coulomb gas / jellium* model, which has a **provably uniform
     density** in a ball — a genuine energy minimum.
   - No interior/exterior distinction: the boundary just emerges.
   - Wire with k-NN (k ≈ 4), plus extra surface↔surface links to fill surface voids.
   - Result: uniform vertex spacing (nearest-neighbor distance std ≈ 1.2 mm), rod length
     ≈ 15 mm ± 1.2 mm. **This is the aesthetic we want to keep.**

## The problem

The Coulomb gas is **isotropic** — it produces rod directions uniformly distributed over
all orientations. But printability is **directional** (depends on the build axis Z) and
**edge-based** (a property of each rod, a pair of vertices).

Measured on the current best lattice (511 rods):

| Category | Count | % |
|---|---|---|
| Steep (≥45° from horizontal, printable) | 146 | 28.6% |
| Short bridges (≤10 mm) | 0 | 0% |
| **Unprintable (shallow, not a bridge)** | **365** | **71.4%** |

(The histogram of rod angles is flat — every orientation is equally likely, exactly what
an isotropic cloud produces. And there are zero valid bridges because the ~15 mm spacing
already exceeds the 10 mm bridge limit.)

So the isotropic cloud is fundamentally at odds with the directional printability rule.
The rule effectively demands a lattice with a **vertical grain** (columns + steep diagonals
+ short horizontal ties) — the "cool" must come from the pattern, not isotropic randomness.

## Proposed direction (needs your opinion)

A **co-optimization / evolutionary** approach that searches over *both* vertex positions
*and* edge topology, with an energy:

- repulsion (uniform spacing),
- spring (uniform rod length),
- a **soft printability penalty** on rod angle (penalize rods in the forbidden 0–45° band
  that are not short bridges),
- boundary (vertices inside the sphere),
- plus **periodic re-wiring** (reconnect neighbors, with some randomness) — i.e. alternate
  "relax positions" and "rewire topology" for several rounds.

## The questions for you

1. How to **encode the printability rule** (steep *or* short-bridge) as a smooth,
   differentiable energy term suitable for gradient-based relaxation?
2. How to **structure the evolution** — what are the right "move" and "rewire" operators,
   selection/fitness, and annealing schedule?
3. How to **balance the two goals** (uniformity vs. printability) — they conflict
   (isotropy vs. a directional grain).
4. Whether a **fundamentally different approach** would be better than "Coulomb gas +
   evolutionary printability repair" — e.g. designing a printable lattice from the start,
   anisotropic point distributions, or something else.

**In short:** what's a good algorithm/energy function to generate a *support-free*
3-D-printable lattice that still retains the uniform-ish, slightly-organic Coulomb-gas
aesthetic?
