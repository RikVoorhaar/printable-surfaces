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

## Viewing

- **Dashboard** (browse all artifacts by pipeline stage) — http://100.81.11.110:8000/
- **Plan** — http://100.81.11.110:8000/plan/plan.html
- **STL viewer** (in-browser, overhang-colored) — http://100.81.11.110:8000/web/viewer.html

All served over Tailscale by `python3 -m http.server 8000` from the repo root.
Artifacts are tracked in `manifest.json` and listed by stage (surface → triangulation →
lattice → slicer) on the dashboard.

