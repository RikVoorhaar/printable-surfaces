# printable-surfaces

Turn smooth surfaces (sphere, Klein bottle, torus, fruit, …) into *printable* 3-D
objects.

Pipeline idea:

1. **Parametrize** a surface (explicit/parametric form).
2. **Triangulate** it into a mesh.
3. **Fix / orient** the triangulation and emit an **STL**.
4. **Build a graph** (adjacency of faces) that encodes printability constraints —
   e.g. maximum overhang, full support of every feature.
5. **Verify** with the **PrusaSlicer CLI**.

The work is tracked in the living planning document `plan/plan.html` (add new
phases, don't rewrite old ones).
