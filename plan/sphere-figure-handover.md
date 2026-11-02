# Handover — §1 sphere figure (interactive, rods colored by slope)

Separate work session. The prose for §1 is being drafted elsewhere; this is only the figure.
Read `plan/blog-plan.html` first (voice notes, facts table, §1 sketch) and `AGENTS.md`.

## What the figure has to say

§1 claims the sphere fails for an angular reason, not a meshing reason. The figure is the
evidence: rotate the sphere and you see that rods at 10–40° cover most of it, in bands, and
that refining the mesh only makes the same bands finer.

So: **one sphere lattice, every rod colored by the absolute elevation of its own axis**, on a
continuous color scale. Not pass/fail. No red/green.

## What already exists (do not rebuild)

| Piece | Where |
|---|---|
| Cube-sphere parametrization | `src/printable_surfaces/parametrizations.py` → `cube_sphere_grid(n, radius)` |
| 1-skeleton (nodes = vertices, rods = edges) | `phase2.py` → `_frame()` (`merge_vertices` + `edges_from_faces`) |
| Rods as a solid mesh | `lattice.build_frame_mesh(nodes, edges, rod_radius=1.2, joint_radius=1.8)` |
| STL → quantised Y-up GLB | `uv run python -m printable_surfaces.glb in.stl out.glb` |
| Web viewer | `blog/src/components/LatticeViewer.astro` + `blog/src/lib/lattice/{engine,presets,types}.ts` |
| Usage example | `blog/src/posts/lattice_viewer_test.mdx` |

Post uses Ø1.8 mm rods; node/joint radius should follow whatever the printed solids use in
`src/printable_surfaces/platonic.py` (check before generating).

## The actual work

1. **Per-rod color.** The viewer currently builds one flat-orange `MeshStandardMaterial`
   (`engine.ts` ~line 139). Options, in order of preference:
   - bake a `COLOR_0` vertex attribute in the exporter (uint8 normalized) and set
     `material.vertexColors = true` in the engine, keeping the existing orange path untouched
     when no colors are present;
   - or emit one primitive per color band — no, that breaks the continuous gradient.
   The GLB must stay a single mesh; other posts use the same component and must not change.
2. **Legend.** An HTML overlay is fine and easier than three.js: a vertical colorbar 0–90°
   with a tick at **48°** and a quiet band at **0–5°** for bridges (don't explain it; §2 does).
   Draw the 48° tick from a shared constant, not from the data, so §2 can reuse it.
3. **Static fallback.** Same camera, same colors, rendered to PNG in
   `blog/src/assets/blog/3d_print_platonic_solids/`. This is what runs if JS fails.
4. **Preset.** A `sphere` preset in `presets.ts`: 100 mm sphere, camera pulled back enough to
   frame it, wood backdrop, origin at the lowest point so it sits on the table.

## Numbers that must match the post

Sphere diameter 100 mm, plate exemption included, rods = edges of the cube-sphere:

| n | nodes | rods | unprintable |
|---|---|---|---|
| 5 | 98 | 288 | 78% |
| 6 | 152 | 450 | 78% |
| 12 | 728 | 2,178 | 65% |
| 16 | 1,352 | 4,050 | 63% |
| 24 | 3,176 | 9,522 | 64% |

**Use n=6** unless it looks too chunky: 450 rods, 78% below 48°, still readable as individual
rods. n=5 is the fallback. Anything ≥12 turns into grey mush on screen.

If the post ends up quoting the plateau (~63% at every resolution), consider a second small
static figure with n=6 and n=24 side by side rather than a second viewer.

## Acceptance

- Colors match the rule: a rod running up the side near the equator reads ~85–90° (bright end
  of the scale); a rod running east–west reads ~0–5° (dark end).
- Renders on desktop and mobile, orbit + zoom work, no layout shift, no console errors.
- JS disabled → fallback image, no empty canvas.
- GLB lands in `blog/src/assets/models/...` and is committed (keep it under ~300 kB); the STL
  it came from stays in gitignored `output/`.
- The existing posts and `lattice_viewer_test.mdx` still render after the engine change.

## Second figure: the Klein bottle

The first paragraph of §1 names a Klein bottle, so the reader should see one — and see that it
fails the same way the sphere does. This one is aspirational: if it turns out to need weeks, the
sphere carries §1 on its own and the Klein bottle can be cut without touching the prose.

- **Geometry.** Figure-8 immersion, the standard parametrization over `u, v ∈ [0, 2π)`:
  `x = (a + cos(u/2) sin v − sin(u/2) sin 2v) cos u`,
  `y = (a + cos(u/2) sin v − sin(u/2) sin 2v) sin u`,
  `z = sin(u/2) sin v + cos(u/2) sin 2v`.
- **No rod-rod collisions.** The surface self-intersects; the lattice must not. Triangulate the
  `(u, v)` grid so the preimage of the intersection curve falls **inside triangles, not along
  edges**, then check clearance numerically (min distance between non-adjacent rods ≥ rod
  diameter + gap) rather than by eye. If it fails near the crossing, coarsen locally instead of
  nudging vertices — nudging changes the shape.
- **Same pipeline as the sphere.** 1-skeleton → `build_frame_mesh` → GLB, same continuous
  colormap, same 48° tick, same legend.
- **Composition.** One interactive with a sphere/Klein-bottle toggle, or two stacked viewers if
  the toggle is fiddly. The colorbar is shared between them.
- **Report the numbers** (nodes, rods, % below 48°) for the Klein bottle the same way as the
  sphere. If the bands look different, say why — that difference may be worth a sentence in §1.

**Not in this session:** actually running the relaxation/selection pipeline on the Klein bottle
to see whether it can be *made* printable. That is a research question, and it belongs to the
post only if it produces something.

## Coordination

`blog/src/lib/lattice/*.ts` was being edited when this handover was written. One writer at a
time: check `git status` in `blog/` before touching it, and commit the viewer change on its own
so the prose commits stay clean.
