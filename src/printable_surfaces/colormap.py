"""Continuous colour ramps for meshes that carry data rather than a material.

The web viewer can shade a mesh by a per-vertex colour, so anything that is a *number* per
rod (its angle above horizontal, say) can be shown as a continuous scale instead of a
pass/fail colour. Keeping the ramp here means the figure, the legend and any exported image
all read from the same numbers.

Colours are stored as sRGB bytes, which is what trimesh's ``vertex_colors`` expects. The
GLB writer converts to linear before it puts them in a ``COLOR_0`` attribute, because that
is what the glTF spec requires.
"""

from __future__ import annotations

import numpy as np

# Viridis, sampled at 12 stops. Truncated at t=0.22: the real ramp starts at a near-black
# purple (#440154) which disappears against the dark wooden tabletop the viewer renders on.
# The first stop here is a mid blue that still reads as "low" without vanishing.
VIRIDIS = np.array(
    [
        (62, 73, 137),
        (54, 93, 141),
        (46, 111, 142),
        (39, 128, 142),
        (33, 145, 140),
        (31, 162, 135),
        (46, 179, 124),
        (76, 194, 108),
        (115, 208, 86),
        (160, 218, 57),
        (208, 225, 28),
        (253, 231, 37),
    ],
    dtype=np.uint8,
)


def viridis(t: float | np.ndarray) -> np.ndarray:
    """Sample the ramp at ``t`` in [0, 1]; returns sRGB bytes, shape ``(..., 3)``."""
    t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
    xp = np.linspace(0.0, 1.0, len(VIRIDIS))
    channels = [np.interp(t, xp, VIRIDIS[:, i]) for i in range(3)]
    return np.stack(channels, axis=-1).round().astype(np.uint8)


def elevation_ramp(elevation_deg: float | np.ndarray) -> np.ndarray:
    """Map an angle above horizontal (0–90°) onto the ramp: blue at 0°, yellow at 90°."""
    return viridis(np.asarray(elevation_deg, dtype=float) / 90.0)
