# printable-surfaces

Turn shapes (starting with the sphere) into *printable rod-and-node lattices* — cool
geometric cages you can actually 3-D print, support-free.

Pipeline idea:

1. **Define** the shape (parametric surface + inside/outside test).
2. **Generate a lattice** — a space-frame of rods and nodes (surface frame, diamond,
   honeycomb, Voronoi, …). This is the aesthetic core.
3. **Thicken** rods into cylinders and nodes into joints → one watertight mesh.
4. **Emit an STL** (millimetres, binary).
5. **Verify support-free** with the PrusaSlicer CLI (configurable max overhang angle,
   default 45°).

The work is tracked in the living planning document `plan/plan.html` (append new phases,
don't rewrite old ones).
