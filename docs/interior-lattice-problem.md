# Support-Free Interior Lattice for a Convex Polytope — Problem Statement

*(Self-contained briefing; no solutions proposed below on purpose.)*

## Goal

Fill the interior and faces of a convex polytope with a **rod-and-node lattice** that can
be 3-D printed **with zero support material**, while looking uniform and organic.

## Setup

- Build direction is **+Z** (vertical).
- A rod is printable iff it is **within 45° of vertical** (self-supporting), **or** nearly
  horizontal and **≤ 10 mm** (a bridge).
- The shape is a **rhombohedron** ≈ 91 mm wide × 169 mm tall: a cube (circumradius 50 mm)
  with its body diagonal vertical, stretched ×1.69 along Z so all 12 frame edges sit at
  50° elevation. Its faces are rhombi at ≈ 67° to horizontal.

## Current approach (and what already works)

1. **Frame** — the 12 polytope edges (subdivided to ~15 mm) are fixed nodes + rods. These
   are all printable (50°).
2. **Point placement** — seed points on the faces and inside the volume, then relax them
   with Coulomb repulsion (uniform spacing) while holding the frame points fixed. This
   yields a uniform point cloud.
3. **Wiring** — connect each face/interior point to its 4 nearest neighbours (k-NN) to
   form the rods.
4. Thicken rods into cylinders (1.2 mm) + node spheres (1.8 mm) → STL → slice.

## The problem

The frame rods are all printable, but the k-NN wiring of the face/interior points produces
rods at **random orientations**, so roughly **40% of those rods are shallower than 45°**
and unprintable.

The root cause: the relaxation optimizes point **positions** (uniform spacing), but rod
**directions** are only determined afterwards by an unconstrained nearest-neighbour step.
Printability is never actually enforced.

## The question

What is a good, concretely implementable algorithm to generate the face/interior rods so
that **every rod is printable** (within 45° of vertical, or a short horizontal bridge),
while keeping the point distribution roughly uniform and organic (the current Coulomb
look)?

We would appreciate a specific method — an energy function, a constraint, a wiring rule,
or an optimization — with enough detail to implement directly, rather than high-level
guidance.
