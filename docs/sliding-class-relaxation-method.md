# Support-Free Rod Lattice in a Convex Polytope: Sliding-Class Relaxation Method

*(External method reference — from Perplexity, 2026-10-22.)*

This document describes a complete algorithm for filling a convex polytope with a rod-and-node lattice. The lattice prints on an FDM printer with no support material, and its node layout looks evenly spread but irregular (the "Coulomb" look).

The method works for any convex polytope whose edges are all printable. Numbers to try are given for the current target, a stretched cube (Section 9).

---

## 1. Printability rule

Build direction is +z. A raft is always allowed, and outside supports are allowed. The lattice itself must not need support.

| Symbol | Meaning | Value |
|---|---|---|
| `α` | Minimum rod elevation (angle above horizontal), 45° plus a 3° margin | 48° |
| `Lmin`, `Lmax` | Allowed length of a steep rod | 8 mm, 22 mm |
| `Lb`, `dzb` | Bridge: maximum length, maximum height difference | 10 mm, 0.4 mm |
| `h` | Target node spacing | 15 mm |
| `clear` | Minimum axis-to-axis distance between non-adjacent rods | 2.6 mm |

A rod from node `p` down to node `q` (`z_q < z_p`) is printable if either:

- Steep: `z_p − z_q ≥ sin α · |p − q|` and `Lmin ≤ |p − q| ≤ Lmax`, or
- Bridge: `|z_p − z_q| ≤ dzb` and `|p − q| ≤ Lb` (rarely used; it can be left out).

A node `q` that satisfies this for `p` is called a **parent** of `p`. **Goal:** every non-fixed node has at least 1 parent, and preferably 2.

### Assumptions about the polytope

- It is convex.
- Every face is steep: its slope from horizontal `σ_F` satisfies `σ_F ≥ α`. The bottom face, if there is one, sits on the raft and is exempt.
- Every edge is printable: its elevation is at least `α`. The exceptions are horizontal edges lying on the raft, and horizontal edges no longer than `Lb`. A longer horizontal edge in the air would need its intermediate nodes supported from below — treat those as face nodes of an adjacent face (Section 3) rather than as edge nodes.

---

## 2. Two geometric facts the method relies on

### 2.1 How far an overhanging face reaches into the solid

A face is **overhanging** if its outward normal points downward (`n_z < 0`).

From a node on such a face, the cone of printable parent directions lies mostly outside the solid. Inside the solid, a parent at distance `L` can be at most this deep below the face:

```
depth_max = L · sin(σ_F − α)
```

For a face slope of 67.2° and α = 48°, this is 0.33·L: 4.9 mm at L = 15 mm and 7.2 mm at L = 22 mm.

Interior nodes are normally kept 0.3–0.4h inside the faces, so they are practically never reachable. **Nodes on an overhanging face must therefore be supported within their own face, or by the frame nodes on that face's edges.**

Faces whose normal points up or sideways have no such limitation: their printable cone points into the solid.

### 2.2 The printable wedge inside a face

Take a direction lying in face F, at angle `φ` from the face's downhill line. Its elevation is `asin(sin σ_F · cos φ)`. It is printable when:

```
|φ| ≤ φ_max(F) = acos( sin α / sin σ_F )
```

So within each face, parents of a node lie in a wedge of half-angle `φ_max` around the downhill direction below it.

If the face's two lower edges both lie inside that wedge (they do whenever every edge is printable and the face is a parallelogram), the in-face rule simplifies. In edge coordinates `p = V0 + a·e1 + b·e2`, where V0 is the face's lowest vertex and `e1`, `e2` are its two lower edges:

```
q is an in-face parent of p  if  a_q ≤ a_p,  b_q ≤ b_p,  Lmin ≤ |p − q| ≤ Lmax
```

For general polygonal faces, use the explicit printability test from Section 1.

---

## 3. Node classes

Every node belongs to exactly one class. After each optimisation step it is projected back onto its own part of the polytope.

| Class | Lives on | Projection after each step | Initial count |
|---|---|---|---|
| **V** | Polytope vertices | Fixed | Number of vertices |
| **E** | Edges | Slides along its edge: sort by edge parameter `t`, then clamp each `t` into `[t_prev + Lmin/L_e, t_next − Lmin/L_e]` | `n_e = round(L_e / h) − 1` per edge |
| **F** | Faces | Moves in its face plane, clamped to the face polygon inset by `0.35h` from the edges | `≈ A_F / (0.866 h²) − P_F / (2h)` |
| **I** | Interior | Free, but pushed back to at least `m_I = 0.4h` from every face plane | Start with `≈ 0.75 · V / (h³/√2)`; adjust until the mean nearest-neighbour distance is about 0.8h |

Counts are fixed up front and never change during relaxation. A face therefore cannot end up empty.

Face classes are split further:
- **F↓**: nodes on overhanging faces (`n_z < 0`).
- **F↑**: nodes on all other faces, including vertical ones.

Projection details:
- **Face (parallelogram):** clamp the coordinates `(a, b)` to `[ε, 1−ε]²` with `ε = 0.35h / edge length`.
- **Face (general polygon):** project onto the plane, then repeatedly project onto each inset edge half-plane that is violated (2–3 passes are enough).
- **Interior:** for each face plane that a node is closer than `m_I` to, move the node back along that face's normal.

### Frame rods

The frame rods are the segments between consecutive V and E nodes along each edge. They are rebuilt from the current edge order at every step. They are always printable, because they lie along a printable edge and the order clamp keeps their length at least `Lmin`. Also check that the gap stays at most `Lmax`, which `n_e` guarantees.

---

## 4. Allowed parents per class

Used both in the support energy and when building candidate rods:

| Child class | Allowed parents |
|---|---|
| E | The next V/E node below it on the same edge. Only the length condition matters. |
| F↓ | F nodes on the same face, plus V/E nodes on that face's edges. Not I (Section 2.1). |
| F↑ | F nodes on the same face, V/E nodes on that face's edges, and I nodes |
| I | Any node |

F nodes on different faces are never paired directly. Faces connect through shared edge nodes and the interior.

---

## 5. Energy

```
E = E_rep + λ(t) · E_sup + λ(t) · E_anchor
```

### 5.1 Repulsion

Repulsion acts over all pairs of every class:

```
E_rep = Σ_{i<j} (r_c − d_ij)₊²        r_c = 1.25 h
```

Fixed nodes (V, and frozen nodes if used) push other nodes away but are never moved themselves.

Optional vertical anisotropy, applied to pairs that include an I node:

```
d_ij = sqrt( Δx² + Δy² + (Δz / κ)² )        κ = 1.2 – 1.3
```

What κ does:
- A neighbour direction at elevation θ' in this metric has true elevation θ with `tan θ = κ tan θ'`.
- So the share of neighbour directions steeper than α grows from `1 − sin α` (26% at κ = 1) to 35% at κ = 1.3, and to 45% at κ = 1.69.
- The cost: steep neighbours become longer, by up to κ times.

Do not use repulsion that is weakened along the downhill direction on faces; the support term does that job in a targeted way.

### 5.2 Support

For each non-fixed node `i`, let `P2(i)` be the two allowed parents (Section 4) with the lowest penalty ψ. Recompute `P2(i)` every iteration.

```
E_sup = Σ_i Σ_{q ∈ P2(i)} ψ(x_i, x_q)

ψ(p, q) = (sin α · L − Δz)₊² + (L − Lmax)₊² + (Lmin − L)₊²
          L = |p − q|,   Δz = z_p − z_q
```

Gradient with respect to `p`, where `u = (p − q) / L`, `c = (sin α · L − Δz)₊` and `ẑ` is the vertical unit vector:

```
∂ψ/∂p = 2c (sin α · u − ẑ) + 2 (L − Lmax)₊ u − 2 (Lmin − L)₊ u
∂ψ/∂q = −∂ψ/∂p
```

The force acts on both ends (symmetric) unless an end is fixed. If only the child moved, nodes would pile up at the bottom; the symmetric force prevents that.

For E nodes, only the length terms are non-zero. Those are already enforced by the order clamp, so E nodes can be left out of `E_sup`.

### 5.3 Anchor

This term connects the upper faces to the interior:

```
E_anchor = Σ_{f ∈ F↑, face not vertical} min_{q ∈ I} ψ(x_f, x_q)
```

Without it, nodes on an upward-facing face are only attached to their own face: nothing above them is inside the solid, so no interior node can use them as a parent. The anchor pulls one interior node into the printable cone under each such face node. As a side effect, it pulls interior nodes upward, toward the top of the shape.

---

## 6. Relaxation schedule

```
seed:
    E: evenly spaced along each edge, jitter ±0.15h
    F: uniform random in each face (inset); for F↓ optionally use the constructive start (6.1)
    I: uniform random in the solid (inset m_I)

for it in 0 .. N_it−1                         (N_it = 900)
    λ = 0                           if it < N_it/3
        2 · (it − N_it/3)/(N_it/3)  if N_it/3 ≤ it < 2N_it/3
        2                           otherwise
    T = 0.03h · (1 − it/N_it)                  # jitter amplitude
    g = ∇E_rep + λ (∇E_sup + ∇E_anchor)
    Δx_i = −η g_i, clipped to length 0.08h,  + T · N(0, I)
    if it ≥ 2N_it/3:                           # phase 3: hard support
        for each node i not in V:
            k = number of parents i has now (allowed set)
            propose x_i + Δx_i, project to its class
            if i had ≥ 2 parents and now has < 2, or any child of i drops below 2:
                halve Δx_i and retry (up to 4 times), else keep x_i
    project every node to its class; rebuild frame rods
```

Choose the step size `η` so that the median step is about 0.02h in phase 1. With clipping, `η = 0.5` works for the energy scaling above.

What each phase does:
- **Phase 1** spreads the nodes evenly.
- **Phase 2** shapes support.
- **Phase 3** locks support in: once a node has two parents, no move may take them away.

Phase 3 is necessary because the soft support term alone does not guarantee support on overhanging faces. In a test on one overhanging face of the stretched cube, Coulomb plus soft support left 1–8 single-parent nodes per seed.

### 6.1 Optional constructive start for overhanging faces

If phase 3 still ends with nodes on an F↓ face that have fewer than 2 parents, seed that face with this sampler instead of random points, then relax as usual:

```
placed = V/E nodes on the face's edges
loop:
    C = 400 uniform random points in the face (inset 0.35h)
    keep c if (c has ≥ 2 in-face parents in placed) and min_dist(c, placed) ≥ r
    if none kept: break
    add the kept c minimising  z(c) − 0.5 · min_dist(c, placed)
```

`r = 0.73h` gives about the same density as Coulomb relaxation. The sampler places nodes bottom-up, and each one only where two parents already exist, so support holds by construction. Phase 3 then keeps it.

Test on one overhanging face of the stretched cube, 6 seeds, `r = 11 mm`: every node had 2 parents in every seed; 15–18 nodes per face; nearest-neighbour distance 12.8 ± 1.0 mm (CV 0.08), the same as plain Coulomb relaxation.

---

## 7. Wiring: choosing rods with an integer program

### 7.1 Candidate rods

Include every pair `(i, j)` where `j` is an allowed parent of `i` (Section 4) and the rod is printable (Section 1). Then remove:

- pairs where both ends are V/E nodes (the frame already covers those);
- any candidate whose axis passes within `clear` of a node other than its endpoints, or of a frame rod it does not share an endpoint with.

### 7.2 Variables

- `y_e ∈ {0, 1}` for each candidate rod: 1 if the rod is used.
- Continuous slacks `≥ 0` per node: `s1_i`, `s2_i`, `sd_i`, `sf_i`, `sa_i`.

### 7.3 Objective

```
min  Σ_e c_e y_e + Σ_i (1000 s1_i + 8 s2_i + 5 sd_i + 50 sf_i + 20 sa_i)

c_e = ((L_e − h)/h)² − β + 0.3 · exp(−(θ_e − α)/4°) − γ · [both ends on the same face, or face + its edge]
β = 0.6,  γ = 0.3,  θ_e = elevation of rod e
```

- **β (per-rod reward):** without it, every rod has positive cost. The solver then picks only the required parents, giving mean degree ≈ 3.2. With β = 0.6, every rod in the length window has negative cost, so rods are added until the degree cap or clearance stops them.
- **Angle term:** mildly avoids rods right at the limit angle α.
- **γ (face bonus):** prefers rods lying in a face plane, which make the faces visible.

### 7.4 Constraints

`down(i)` = candidate rods from `i` to its allowed parents; `deg(i)` = number of chosen rods (including frame rods) at `i`.

| Constraint | Applies to |
|---|---|
| `Σ_{down(i)} y_e + s1_i ≥ 1` | all non-fixed, non-E nodes |
| `Σ_{down(i)} y_e + s2_i ≥ 2` | same |
| `Σ_{down(i) in same face} y_e + sf_i ≥ 2` | F nodes |
| `Σ_{down(i) to I} y_e + sa_i ≥ 1` | F↑ nodes on non-vertical faces |
| `deg(i) + sd_i ≥ 3` | all nodes |
| `deg(i) ≤ 6` | all nodes (hard) |
| `y_a + y_b ≤ 1` | pairs of candidates that do not share a node and whose axes are < `clear` apart (hard) |

- **Optional:** also add `y_a + y_b ≤ 1` for pairs sharing a node and meeting at an angle under 25° (two 1.2 mm rods at that angle stay fused ~6 mm from the node).
- **Why never infeasible:** every lower bound has a slack; only upper bounds and pairwise exclusions are hard. `y = 0` with full slack is always feasible. A shortage shows up as positive slack on a specific node, telling you which nodes need repair.
- **Clearance list size:** do NOT count pairs that share a node (distance 0, thousands of spurious conflicts). On the stretched cube, ~600 candidates gave ~13 real conflicts.

Find conflict pairs with a KD-tree on rod midpoints (radius `Lmax + clear`) and an exact segment-to-segment test. If the list grows, add constraints lazily: solve without them, find clashes among chosen rods, add those, re-solve (2–3 rounds).

Solver: `scipy.optimize.milp` (HiGHS), under a second. Use the result only if `res.status == 0`.

---

## 8. Repair and final checks

```
for round in 1..4:
    bad = nodes with s1 > 0 or s2 > 0, sorted by z ascending
    if bad is empty: break
    for i in bad:
        try 400 random moves of length ≤ 0.6h (projected to i's class)
        keep only moves that stay ≥ 0.65h from all other nodes
          and do not reduce the parent count of any child of i
        take the move with the largest number of candidate parents
    rebuild candidates, re-solve Section 7
delete any I node with s1 > 0; re-solve once
```

Only I nodes are deleted. E and F nodes are covered by phase 3 or the constructive start. If one still lacks a parent, move it within its class, or give it a rod to the nearest frame node (if printable and passes clearance).

Do NOT use a "rod to the nearest node below" post-fix — it ignores clearance and degree limits.

### Checks before writing the STL

1. 0 rods outside the printable rule.
2. 0 non-fixed nodes without a downward rod. Nodes at minimum z sit on the raft and are exempt.
3. 0 pairs of non-adjacent rods closer than `clear`.
4. All frame segments within `[Lmin, Lmax]`.

Then thicken: rods as 1.2 mm cylinders, nodes as 1.8 mm spheres.

---

## 9. Numbers for the stretched cube

Geometry: circumradius 50 mm, body diagonal vertical, stretched by `s = tan 50° / tan 35.26° = 1.6855` (all 12 edges at exactly 50°).

| Quantity | Value |
|---|---|
| Width, height | 94.3 mm, 168.5 mm (z ∈ [−84.25, 84.25]) |
| Edge length | 73.3 mm, all at 50° elevation |
| Faces | 6 congruent rhombi, area ≈ 4967 mm², slope σ = 67.24°, rhombus angle 67.7° |
| Overhanging faces F↓ | the 3 faces meeting at the bottom vertex |
| Upward faces F↑ | the 3 faces meeting at the top vertex |
| In-face wedge φ_max | 36.3° (39.9° at the true 45° limit) |
| Edges vs downhill, in-face | ±33.8°, inside the wedge |
| Overhang reach sin(σ − α) | 0.33: 4.9 mm at L=15, 7.2 mm at L=22 |
| Elevation of an in-face rod at ±30° from downhill | 53° |

Starting parameters: `h`=15, E nodes=4/edge (48 total), F=17/face (102 total), I=105, total ≈263 with 8 vertices; `r_c`=18.75 (1.25h), κ=1.25 (interior only), face inset 5.25 (0.35h), `m_I`=6 (0.4h), iters 900 (λ ramps 0→2 over 300–600), step clip 1.2 (0.08h), jitter T=0.45 decaying to 0, constructive-start r=11, β=0.6, γ=0.3, slack costs s1/s2/sd/sf/sa = 1000/8/5/50/20, degree soft ≥3 hard ≤6, clear=2.6.

Expected outcome (earlier tested run, 4 seeds): 0 unprintable rods, 0 nodes without a downward rod (after deleting 2–7 nodes), 0 clearance violations, 2–7 single-parent nodes, mean degree 4.7, rod length 16±3 mm, rod elevations 48–89° (median ~58°), nearest-neighbour spacing 11.8±0.7 mm (CV 0.06).

The sliding-class relaxation with hard phase 3 is designed to remove the deletions and the single-parent nodes on overhanging faces (not yet run end-to-end). The constructive start was tested on one overhanging face.

---

## 10. Which earlier failures this addresses

| Earlier failure | Cause | Fix |
|---|---|---|
| Up to 40% of rods unprintable (k-NN) | Nearest-neighbour wiring ignores direction | Only printable candidates are ever wired (7.1) |
| Unsupported nodes on overhanging faces | Interior parents out of reach (2.1); soft support not enough | Restricted parent sets (4), sliding edge nodes, hard phase 3 (6), constructive start (6.1) |
| Top faces with no nodes inside | Face nodes could vanish / were never kept | Fixed counts per class (3); anchor term (5.3); soft interior-parent constraint (7.4) |
| Nodes piling at the bottom | Support force acting only on the child | Symmetric support (5.2), plus the anchor pulling interior nodes up |
| Mean degree around 3.3 | Every rod has positive cost | Per-rod reward β (7.3) |
| Clearance makes the IP infeasible | Shared-node pairs counted as conflicts; hard lower bounds | Skip shared-node pairs; slack on every lower bound (7.4) |
| Face planes hard to see | Few rods lying in a face | In-face parents (4), face-parent constraint sf, face bonus γ (7.3, 7.4) |

---

## 11. Known limitations

- Rod elevations cluster just above α (the solver favours short rods, many near the limit angle). Flatten with a target-histogram penalty.
- The interior shows a vertical grain (every rod within 42° of vertical — inherent to the support-free rule). Larger κ makes the grain stronger.
- A shape on a vertex has a single-point base: needs the raft and ideally a brim/outside supports.
- Phase 3 costs one parent recount per moved node (~260 nodes) — cheap with a KD-tree.
- Assumes a convex polytope; non-convex shapes would also need line-of-sight checks.
