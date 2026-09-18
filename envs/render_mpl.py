"""Top-down PNG of a World — the cheap renderer.

Two jobs, and neither of them is to look good. It is the audit view (does
this scenario's graph match what the reward is measured against?) and the
frame dumper for a rollout video. The 3D view is `envs/render3d.py`; both
draw the same `World`, so a disagreement between them is a bug in one of
them rather than two pictures of two worlds.

    python -m envs.render_mpl                 # all scenarios -> scenarios.png
    python -m envs.render_mpl roundabout_yield
"""

from __future__ import annotations

import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

from envs import road_network
from envs.world import World

ASPHALT = "#2a2b30"
MARKING = "#e6e6da"
GROUND = "#e8e4dc"
BUILDING = "#8d8579"
TREE = "#4a7c3f"
ISLAND = "#4d7546"
VEHICLE = "#d9822b"
EGO = "#2b7fd9"
GOAL = "#c43b3b"


def _rect_patch(cx, cy, hl, hw, yaw, **kw):
    cos, sin = math.cos(yaw), math.sin(yaw)
    corners = [(hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)]
    pts = [(cx + cos * x - sin * y, cy + sin * x + cos * y) for x, y in corners]
    return Polygon(pts, closed=True, **kw)


def draw(world: World, vehicles=None, goals=None, path: str | None = None,
         ax=None, title: str | None = None, dpi: int = 110):
    """Draw one world. Returns an (H, W, 3) uint8 array when `path` is None
    (so it can be a gym `rgb_array` frame), else writes the PNG and returns
    the path."""
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(8, 8), dpi=dpi)
    else:
        fig = ax.get_figure()

    net = world.net
    ax.set_facecolor(GROUND)

    # Road surface, then junction pads over it, then markings — the pads
    # exist to mask lane lines inside a junction, so order matters.
    for piece in net.pieces:
        poly = piece.polyline(0.0, piece.length)
        _ribbon(ax, poly, piece.half_width, ASPHALT, zorder=1)
    for (x, y), pad in zip(net.nodes, net.node_pads):
        if pad > 0.0:
            ax.add_patch(Circle((x, y), pad, color=ASPHALT, zorder=2))
    for i, piece in enumerate(net.pieces):
        poly = piece.polyline(0.0, piece.length)
        inset = max(piece.half_width - 0.28, 0.05)
        for side in (inset, -inset):
            _ribbon(ax, _offset(poly, side), 0.07, MARKING, zorder=3)
        # One dashed line per lane divider. A single centre dash was only
        # ever right for a two-lane road; on a three-lane arterial it drew a
        # road the lane graph does not agree with.
        for offset in world.lanes.dividers(i):
            s = 2.25
            while s < piece.length:
                end = min(s + 3.0, piece.length)
                _ribbon(ax, _offset(piece.polyline(s, end), -offset), 0.08,
                        MARKING, zorder=3)
                s = end + 4.5
    for cx, cy, r in net.islands:
        ax.add_patch(Circle((cx, cy), r, color=ISLAND, zorder=4))

    for cx, cy, hl, hw, yaw, _h in world.scenery.boxes:
        ax.add_patch(_rect_patch(cx, cy, hl, hw, yaw, facecolor=BUILDING,
                                 edgecolor="#6f6960", lw=0.5, zorder=5))
    for cx, cy, r, _h in world.scenery.trees:
        ax.add_patch(Circle((cx, cy), r, color=TREE, zorder=5))

    if goals is not None:
        for gx, gy in np.asarray(goals).reshape(-1, 2):
            ax.add_patch(Circle((gx, gy), 3.0, facecolor="none",
                                edgecolor=GOAL, lw=1.6, zorder=6))
    if vehicles is not None:
        for k, (cx, cy, hl, hw, yaw) in enumerate(np.asarray(vehicles).reshape(-1, 5)):
            ax.add_patch(_rect_patch(cx, cy, hl, hw, yaw,
                                     facecolor=EGO if k == 0 else VEHICLE,
                                     edgecolor="black", lw=0.5, zorder=7))

    x0, y0, x1, y1 = world.bounds()
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)

    if not own_fig:
        return ax
    if path:
        fig.savefig(path, bbox_inches="tight", facecolor=GROUND)
        plt.close(fig)
        return path
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return frame


def _offset(poly, d: float) -> np.ndarray:
    """A polyline shifted `d` metres to the right of travel."""
    poly = np.asarray(poly, dtype=float)
    if len(poly) < 2:
        return poly
    seg = np.gradient(poly, axis=0)
    n = np.hypot(seg[:, 0], seg[:, 1])
    n[n < 1e-9] = 1.0
    return np.stack([poly[:, 0] + seg[:, 1] / n * d,
                     poly[:, 1] - seg[:, 0] / n * d], axis=1)


def _ribbon(ax, poly, half_width: float, color: str, zorder: int = 1) -> None:
    """A constant-width strip along a centreline, as one filled polygon."""
    poly = np.asarray(poly, dtype=float)
    if len(poly) < 2:
        return
    left = _offset(poly, -half_width)
    right = _offset(poly, half_width)
    pts = np.vstack([left, right[::-1]])
    ax.add_patch(Polygon(pts, closed=True, color=color, lw=0, zorder=zorder))


def contact_sheet(path: str = "scenarios.png", kinds=None,
                  scenery_density: float = 1.0, seed: int = 0):
    """Every scenario on one sheet — the picture to look at after changing a
    builder, a scenery rule, or the widths."""
    kinds = list(kinds or road_network.SCENARIO_KINDS)
    cols = min(4, len(kinds))
    rows = math.ceil(len(kinds) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), dpi=100)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(kinds):]:
        ax.axis("off")
    for ax, kind in zip(axes, kinds):
        world = World.build(kind, rng=np.random.default_rng(seed),
                            scenery_density=scenery_density)
        draw(world, ax=ax,
             title=f"{kind}\n{len(world.net.pieces)} pieces · "
                   f"{len(world.scenery.boxes)} buildings · "
                   f"{len(world.scenery.trees)} trees")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor=GROUND)
    plt.close(fig)
    return path


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        kind = sys.argv[1]
        out = draw(World.build(kind), path=f"{kind}.png")
    else:
        out = contact_sheet()
    print(f"wrote {out}")
