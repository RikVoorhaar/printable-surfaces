# Handover — solid switcher (end of the post)

Separate work session. Spec from Rik, 2026-11-08. The prose around it is already written;
this is only the widget.

## What it is

One viewer at the end of the "lattice algorithm" section, just before the conclusion. Five
labelled buttons swap between the five finished solids.

## Spec

- **Contents:** the five Platonic solids only. No sphere, no Klein bottle.
- **Buttons:** five, labelled with the solid's name. Same styling as the existing
  Orange / Angles buttons (`lattice-viewer__mode`).
- **One mesh at a time.** The page must not download all five GLBs. Load the default solid
  on mount, and lazily fetch a GLB the first time its button is pressed.
- **Colour modes:** keep the existing Orange / Angles toggle, same as the Klein bottle and
  sphere viewers (`modes`, `preset.colorMode`).
- **Default:** tetrahedron.
- **Label:** just the name. No dimensions, no mass, no "stretched ×1.93".
- **Fallback:** one static WebP of the default solid when JavaScript is off.
- **Placement:** after the battle-scars paragraph, before the conclusion. There is a
  placeholder comment in the post at that spot.

## What already exists

| Piece | Path |
|---|---|
| GLBs | `blog/src/assets/models/platonic/{cube,tetrahedron,octahedron,icosahedron,dodecahedron}_T5_S2.glb` |
| Presets | `blog/src/lib/lattice/presets.ts` (all five already have camera/light/backdrop) |
| Component | `blog/src/components/LatticeViewer.astro` |
| Working example | `blog/src/posts/lattice_viewer_test.mdx` |

## Implementation notes

- `LatticeViewer` currently takes a single `src` and mounts one engine. Three options:
  extend it with a list + buttons, add a small wrapper that swaps `src` and re-mounts, or
  render five instances but only mount the selected one. Pick whichever keeps
  `engine.ts` / `presets.ts` untouched.
- Three.js must stay out of `Layout.astro` and any global entry point (see `blog/AGENTS.md`).
- Dispose the old geometry/material on switch; don't leak a WebGL context per click.
- Decide whether switching keeps the orbit position or resets to the preset camera. Resetting
  to the preset is probably better, since each solid has its own framing.
- Buttons must wrap on narrow screens, not scroll.

## Acceptance

- On first load only the tetrahedron GLB is fetched; each other GLB is fetched on first click.
- Switching works both ways, repeatedly, with no console errors and no memory growth.
- Orange / Angles works for every solid.
- No-JS shows the static fallback, not an empty box.
- The existing posts and `lattice_viewer_test.mdx` still render unchanged.

## Coordination

`blog/src/lib/lattice/*.ts` may be edited by another session. One writer at a time: check
`git status` in `blog/` before touching it, and commit the widget separately from the prose.
