# Support-Free Rod Lattices for the Platonic Solids: Theory and Algorithm

Version 3. The requirements changed again: stretching the solid is now allowed, and the goal is every face and every edge printable in its own right.

## Requirements (updated)

- **Process.** A raft is always allowed, and so are outside supports. Rods are printable if their elevation is at least \(\alpha = 45^\circ\), or if they are horizontal bridges of at most 10 mm between nodes that are already printed. A long horizontal edge is therefore a chain of ≤ 10 mm spans, with a steep post under each intermediate node.
- **Shape.** Start from a Platonic solid. Deforming it is allowed (it then stops being Platonic), as long as it stays a convex polytope with the same combinatorics.
- **Faces.** Every face except the bed must be printable in its own right: slope at least 45°. In terms of normals, the face normal is **at least** 45° away from vertical, i.e. within 45° of horizontal. ("Less than 45° from vertical" would describe a nearly flat face, which is the case to avoid.) A shallow face that can be reached, like the old icosahedron top, would look bad, so it's excluded.
- **Edges.** Every polytope edge is a printable straight rod: steep, horizontal and bridged, or on the raft.
- A vertex as the lowest point is acceptable.

## Answer

All five solids work with the right orientation and a modest vertical deformation. Design target 50° (5° margin); values at exactly 45° in brackets. Shapes are normalized to circumradius 50 mm before deformation; scale the result to taste.

| Solid | Orientation | Deformation | Height/width | Edges | Horizontal edges | Faces (slope, in-face rod range) |
|---|---|---|---|---|---|---|
| **Tetrahedron** | face down | **none** | 0.71 | 3 at 54.7° | none above the raft | 3 at 70.5°, ±41° |
| **Octahedron** | vertex down | vertical stretch ×1.19 [×1.00] | 1.19 | 8 at 50° | 4 (the "belt", 70.7 mm, 8 spans each) | 8 at 59.3°, ±35° |
| **Cube** | vertex down (body diagonal vertical) | vertical stretch ×1.69 [×1.41] | 1.79 | 12 at 50° | **none** | 6 rhombi at 67.2°, ±40° |
| **Icosahedron** | vertex down | **lift the two tip pyramids only**: height ×1.51 [×1.34] (a uniform stretch would need ×1.93) | 1.69 | 10 at 50°, 10 at 58.3° | 10 (two pentagon rings, 52.6 mm, 6 spans each) | 10 at 55.8° (±31°), 10 at 79.2° (±44°) |
| **Dodecahedron** (strict) | vertex down | vertical stretch ×3.12 [×2.62] | 3.31 | 24 at 50–83° | 6 (35.7 mm, 4 spans) | 6 at 67.2°, 6 at 86.5° |
| **Dodecahedron** (flat top allowed) | face down | vertical stretch ×1.93 [×1.62] | 1.56 | 20 at 50–72° | 5 (top pentagon, 4 spans) | 10 at 75.5°; top face horizontal |

Notes:

- **The resulting shapes have names.**
  - The stretched cube is a **rhombohedron** (trigonal trapezohedron): 6 congruent rhombi, every edge at the same angle, and no bridges anywhere.
  - The stretched octahedron is an elongated square bipyramid.
  - The icosahedron becomes a pentagonal antiprism with taller pyramids on each end. Its band of 10 equilateral triangles is untouched.
- **The dodecahedron is the outlier.** Keeping its pentagons flat forces a uniform stretch (shown below). With a strict "no horizontal faces" rule it ends up about 3.3 times as tall as it is wide. If a flat, bridged top pentagon is acceptable (it can be left open as just its 5 bridged edges), ×1.93 is enough and keeps 5-fold symmetry. I'd recommend that variant.

---

## Theory

### 1. Checking the edges is enough

> **Theorem (edges suffice).** Let \(K\) be a convex polytope whose lowest feature \(B\) (vertex, edge or face) sits on the raft. If every edge not in \(B\) is steep (≥ α) or horizontal, then:
> (a) every face that isn't horizontal has slope ≥ α;
> (b) \(K \subseteq B \oplus C_\alpha\), so every point of \(K\) can be reached from the raft by steep rods.

**(a)** A face that isn't horizontal contains at least one edge that isn't horizontal. A face's slope is at least the elevation of any line lying in it.

**(b)** Use the normal-cone picture from version 2: the cap of outward normals within \(90^\circ-\alpha\) of straight down may only meet normals of features in \(B\). Take any feature \(F\) outside \(B\) with a normal \(n\) inside the open cap. Three cases:

- **\(F\) is a face.** Then it faces downward with slope < α. That contradicts (a), unless the face is horizontal; but a horizontal downward face is the bottom of the solid, so it is \(B\).
- **\(F\) is an edge.** Then \(n \perp F\), which forces the edge's elevation below α, so the edge is horizontal. Its arc of normals lies on the vertical great circle perpendicular to the edge. Neither end of the arc (the normals of the two adjacent faces) lies in the cap, by the face case. So the arc can only enter the cap by passing through \(-\hat z\). Then \(F\) is the lowest feature, i.e. part of \(B\).
- **\(F\) is a vertex.** If none of its edges' arcs enters the cap, its normal region contains the whole cap, including \(-\hat z\), so the vertex is lowest. Otherwise we're back in the edge case.

**Consequence.** Your face requirement reduces to an edge requirement plus "no horizontal faces except the bed." Reachability, which was the whole problem with the sphere, follows automatically. Everything below is about edges.

### 2. Vertical stretching only makes things steeper

Scale \(z\) by \(s\). An edge at elevation \(e\) moves to elevation \(e'\) with \(\tan e' = s\tan e\), and horizontal edges stay horizontal. The stretch also preserves:

- convexity;
- flat faces;
- parallel edges;
- every symmetry about the vertical axis.

So for any orientation the minimal uniform stretch is
\[
s^* = \frac{\tan\alpha}{\tan e_\text{min}}, \qquad e_\text{min} = \text{smallest elevation among edges that aren't horizontal.}
\]
The orthogonal-triple obstruction from version 2 disappears: it only applies to rigid orientations.

### 3. Choosing the orientation

- **Horizontal faces** occur only when the vertical axis passes through a face. The cube, octahedron, dodecahedron and icosahedron are centrally symmetric, so face-down leaves a flat top face. The strict rule therefore excludes face-down for those four. The tetrahedron has no opposite face, so face-down works.
- **Avoiding horizontal edges entirely costs a lot.** Optimizing over generic tilts, the best minimum edge elevation gives \(s^*\):

  | Solid | Minimal stretch with no horizontal edges |
  |---|---|
  | Tetrahedron | ×3.0 |
  | Octahedron | ×3.0 |
  | Icosahedron | ×7.5 |
  | Dodecahedron | ×7.5 |
  | Cube | ×1.41 (the body-diagonal axis already has no horizontal edges) |

  Generic tilts also destroy the symmetry.
- **Symmetric axes are the sweet spot.** Vertices come in horizontal rings, so the ring edges are horizontal (bridged), and every other edge is fairly steep. I ran an exhaustive search over all orientations where at least one edge direction is horizontal: every great circle of such orientations, plus every intersection of two. The best strict orientation is the vertex axis for the octahedron, cube, icosahedron and dodecahedron, and the face axis for the tetrahedron.

### 4. Least deformation: a linear program over vertex heights

Uniform stretching is wasteful when only a few edges are shallow. Keep each vertex's horizontal position \((x_v, y_v)\) fixed and treat its height \(z_v\) as a variable. Every requirement is then **linear** in the heights:

- **Steep edge:** \(\sigma_e (z_j - z_i) \ge \tan\alpha\,\lVert x_j - x_i\rVert_{xy}\), keeping the original up/down order \(\sigma_e\).
- **Horizontal edge:** \(z_i = z_j\).
- **Flat faces:** with \(xy\) fixed, a face's plane \(z = ax+by+c\) has coefficients linear in three reference heights. Each further vertex of the face must satisfy that linear equation.
- **Convexity:** every other vertex lies on the inner side of every face plane, with a margin \(\varepsilon\) (0.5 mm) so the combinatorics can't change. That's linear for the same reason.
- **Raft:** the bottom feature sits at \(z = 0\).

Objective: minimize the total change in the vertical extent of the edges, \(\sum_e \lvert\Delta z_e - \Delta z_e^0\rvert\). This favors sparse, local changes. It is solved with HiGHS in milliseconds.

Results:

- **Icosahedron.** The LP lifts only the two tip vertices. Their pyramid edges go from 31.7° to 50°, and the antiprism band, whose edges are already at 58.3°, is untouched. Height ×1.51 instead of ×1.93.
- **Cube, octahedron, dodecahedron.** The LP returns exactly the uniform stretch. For the cube, each rhombus face spans three height levels and must stay a flat parallelogram, which forces the levels to be equally spaced. For the dodecahedron, each pentagon spans several levels, and the flatness equations again force a uniform stretch.

  A tetrahedron, octahedron or icosahedron has only triangular faces, so any vertex heights keep the faces flat. That freedom is what the icosahedron uses.

### 5. General affine maps don't help

The vertical axis could also come with horizontal shears or uneven stretching. I minimized the condition number of a general 3×3 linear map under the same edge constraints, starting from the best orientations. Shape distortion here means the condition number: how unequally the map stretches different directions. It never beat the pure vertical stretch.

For axes with at least 3-fold symmetry there's a short reason. For a fixed vertical direction, the feasible set of metric tensors is convex and invariant under the symmetry group. Averaging a feasible map over the group keeps it feasible, and gives a tensor of the form \(aI + b\,\hat z\hat z^T\): a vertical stretch. The averaging doesn't make the distortion worse.

---

## Design notes per solid

### Tetrahedron (face down, as is)

Already ideal: no stretch, no horizontal edges above the raft, 3 faces at 70.5° (in-face rods within ±41° of the downhill direction). It's the only solid that is "Platonic and printable" with no changes.

### Octahedron (vertex down, ×1.19)

The 4 belt edges at the widest point are horizontal, 70.7 mm long, so they need 7 posts each. The posts come up from the lower pyramid's interior, and they always exist by the theorem. All faces are congruent at 59.3°.

At exactly 45° no stretch is needed, but every edge would then be right at the limit.

### Cube (vertex down, ×1.69): the cleanest result

This is a rhombohedron: every edge at 50°, no bridges anywhere, 6 congruent rhombi at 67.2° with a wide in-face range (±40°). The interior funnels to the bottom vertex. Of all the results, this one best meets "every edge printable on its own."

### Icosahedron (vertex down, tip pyramids lifted)

There are two horizontal pentagon rings (10 edges of 52.6 mm, 5 posts each). The steep faces come in two classes: 10 band faces at 79.2° (±44°) and 10 tip faces at 55.8° (±31°). Five-fold symmetry is kept, and the lift is localized and reads as deliberate.

If you'd prefer the uniform look, the uniform ×1.93 stretch also works.

### Dodecahedron

- **Strict (vertex down, ×3.12):** 3-fold symmetry, very tall (height/width 3.3). Six horizontal edges in two rings of 3, with 3 posts each.
- **Relaxed (face down, ×1.93):** 5-fold symmetry, height/width 1.56, all 10 side faces at 75.5°. The top pentagon is horizontal. Leave it open (5 bridged edges, 3 posts each) or fill it with horizontal hatching on posts.

---

## Practical issues that remain

1. **A single-point base.** With vertex-down, \(B\) is a point. By the cone-hull theorem, every steep chain in the lattice then funnels into that one node, so the whole print stands on one 1.8 mm joint. That's a stiffness and wobble risk for parts 120–170 mm tall. Since outside supports are allowed, add a few **stabilizer struts** from the raft to nodes low on the downward faces.

   These aren't needed for printability; they work like slicer "tree supports" for stability, and are snapped off afterwards. The alternative is a small flat cut at the bottom vertex (a tiny triangle, square or pentagon resting on the raft), at the cost of a slight change in shape.
2. **Long horizontal edges.** Each intermediate bridge node needs a post in the downward cone below it, and posts at regular spacing will be visible. Fold them into the interior Coulomb lattice as fixed parent requirements, rather than adding them afterwards.
3. **Edges at exactly the limit.** Hinge penalties and these LP solutions put many edges at exactly α. Designing to 50° rather than 45° gives slack for rod thickness and joint geometry.
4. **Size.** The heights are for circumradius 50 mm before deformation. Scale the deformed shape to fit your build volume and target rod spacing.

## Updated algorithm

1. **Choose the orientation.** For each symmetry axis (vertex, edge or face, as both "up" and "down"), and optionally for orientations where one edge direction is horizontal:
   - reject the orientation if a non-bed face would be horizontal (strict mode);
   - compute \(e_\text{min}\) and the uniform \(s^*\);
   - solve the height LP.

   Rank by height ratio (distortion), then symmetry order, then number and length of horizontal edges.
2. **Deform.** Apply the LP heights (or the uniform stretch). The theorem then guarantees that every face is steep and every point is reachable.
3. **Frame.** Steep edges become rods, subdivided at the target spacing. Horizontal edges become chains of ≤ 10 mm bridge spans, and each intermediate node is marked as needing a parent rod below it.
4. **Face lattices.** Run a 2D Coulomb gas on each face, wired within that face's allowed range \(\pm\arccos(\sin\alpha/\sin\beta)\) of the downhill direction. After deformation every face allows at least ±31°, so the face lattices look reasonably uniform.
5. **Interior lattice.** Run a 3D Coulomb gas with the surface nodes fixed, plus the support energy and the top-down growth/repair sweep. It can't get stuck, by the theorem. Choose the rods with the integer program: at least m parents per node, rod clearance, and a target distribution of rod angles.
6. **Stability.** If the bed is a vertex, add stabilizer struts from the raft, or make a small flat cut.
7. **Verification.** Graph checks, then STL, then PrusaSlicer.

## Earlier results still in force

These carry over from version 2; they explain why stretching is needed at all.

- **Cone-hull theorem.** A convex \(K\) can be printed support-free from its bed contact \(B\) iff \(K \subseteq B \oplus C_\alpha\). For polytopes this is the vertex-shadow test: each vertex's horizontal distance outside the bed polygon must be at most its height times \(\cot\alpha\). The edges-suffice theorem above is a sufficient condition for it.
- **Orthogonal-triple lemma.** For three mutually perpendicular edge directions, \(\sum\sin^2 e_i = 1\). This is why no rigid orientation of the icosahedron or dodecahedron works, and why the cube can't stand on a vertex unstretched. A vertical stretch breaks the orthogonality, which is why it gets around the lemma.
- **In-face rule.** In a face of slope β, rods must stay within \(\pm\arccos(\sin\alpha/\sin\beta)\) of the downhill direction, or be horizontal bridges.

Literature context: the no-local-minimum condition of [Martínez et al. (2018)](https://inria.hal.science/hal-01697103/document), bridges resting on pillars as in [Dumas et al. (2014)](https://members.loria.fr/JDumas/publications/scaffoldings/), and the layer-wise printability filter of [Langelaar (2017)](https://www.mendeley.com/catalogue/72920aca-b7af-39b3-bc2f-7cd602a8ca29/).

Analysis scripts:

- `poly/stretch.py` and `stretch2.py`: orientations and uniform stretch.
- `poly/heightlp.py`: the height LP.
- `poly/affine.py`: the general affine check.
- Earlier scripts for the version 2 numbers.
