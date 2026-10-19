"""Matplotlib 3-D helpers for quick visual sanity checks of surfaces and lattices."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def _new_3d_axis(fig=None, title: str = ""):
    if fig is None:
        fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    return fig, ax


def _equalize(ax, points: np.ndarray):
    """Set equal aspect using a bounding box around the given points."""
    points = np.asarray(points, dtype=float)
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    center = (lo + hi) / 2
    radius = float((hi - lo).max()) / 2 * 1.05
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def plot_grid_lines(uv: np.ndarray, chart_points: list[np.ndarray], labels: list[str] | None = None, title: str = ""):
    """Plot grid lines of one or more charts.

    ``uv``: 1-D array of sample values along each axis (same for u and v).
    ``chart_points``: list of ``(n, n, 3)`` grids, one per chart.
    """
    fig, ax = _new_3d_axis(title=title)
    all_pts = np.concatenate([g.reshape(-1, 3) for g in chart_points], axis=0)
    for k, g in enumerate(chart_points):
        color = plt.cm.tab10(k % 10)
        label = labels[k] if labels else None
        for i in range(g.shape[0]):
            ax.plot(g[i, :, 0], g[i, :, 1], g[i, :, 2], color=color, lw=0.6, label=label if i == 0 else None)
        for j in range(g.shape[1]):
            ax.plot(g[:, j, 0], g[:, j, 1], g[:, j, 2], color=color, lw=0.6)
    _equalize(ax, all_pts)
    if labels:
        ax.legend(fontsize=7, loc="upper left")
    return fig, ax


def save(fig, path: str) -> None:
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
