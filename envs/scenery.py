"""Scenery — buildings and trees along the street, not scattered beside it.

Nothing here is authored per scenario. The road graph already says where the
roads are and which way they run, so the scenery is derived from it: every
carriageway gets a **frontage line** on each side, and buildings are set out
along it facing the street.

This replaced a distance-band sampler that dropped buildings anywhere
between 12 m and 55 m of a road at a random spacing. That produced an even
scatter of boxes in open ground — the road graph was a diagram with clutter
around it rather than a place. Buildings in a row, at one setback, square to
the street they face, is what makes a block read as a block.

    kerb  ->  pavement  ->  front gap  ->  building frontage
              2.5 m         3.0 m          8-20 m wide, 9-18 m deep

Trees go in the verge between kerb and building line, which is where street
trees are, and are skipped near junctions so corners stay open — both
because that is how junctions are built and because a tree on a corner
blinds the lidar exactly where a policy most needs to see.

Two footprint shapes, and the split is a sensor decision rather than an
aesthetic one. A tree is a circle: rotationally symmetric, one distance
test. A building is a rotated rectangle, because a row of buildings along a
street is a flat wall to a lidar and modelling it as circles would leave the
policy sight-lines through gaps that do not exist in the picture. Both are
what the renderer draws AND what the raycast hits, so what the agent sees
and what a human sees can never drift apart.
"""

from __future__ import annotations

import math

import numpy as np

# The cross-section out from the kerb.
PAVEMENT = 2.5           # footway, drawn but not an obstacle
FRONT_GAP = 3.0          # between pavement and the building line
BUILDING_DEPTH = (9.0, 18.0)    # perpendicular to the street
BUILDING_FRONTAGE = (8.0, 20.0)  # along it
BUILDING_HEIGHT = (6.0, 24.0)
PLOT_GAP = (1.5, 5.0)    # between neighbouring frontages

# Street trees live between kerb and building line.
TREE_SETBACK = 1.6       # out from the kerb
TREE_SPACING = (9.0, 15.0)
TREE_RADIUS = (1.0, 2.0)
TREE_HEIGHT = (4.0, 9.0)

# Keep clear of junctions and of episode endpoints.
JUNCTION_CLEAR = 14.0
ENDPOINT_CLEAR = 9.0
# Only real carriageways get a frontage; a one-lane bay or aisle does not
# have buildings fronting onto it.
MIN_LANES = 2


class Scenery:
    """Static world objects, as two arrays the sensors and both renderers read.

    `boxes`  (N, 6) — cx, cy, half_length, half_width, yaw_rad, height
    `trees`  (M, 4) — cx, cy, radius, height

    Empty scenery is an ordinary Scenery with zero rows, not None, so callers
    never branch on it.
    """

    def __init__(self, boxes: np.ndarray, trees: np.ndarray):
        self.boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 6)
        self.trees = np.asarray(trees, dtype=np.float32).reshape(-1, 4)

    def __len__(self) -> int:
        return len(self.boxes) + len(self.trees)

    def __repr__(self) -> str:
        return f"Scenery(buildings={len(self.boxes)}, trees={len(self.trees)})"

    @classmethod
    def empty(cls) -> "Scenery":
        return cls(np.zeros((0, 6)), np.zeros((0, 4)))


def _blocked(net, x, y, clearance):
    """True when a point is on, or within `clearance` of, any road."""
    return -net.road_margin(float(x), float(y)) < clearance


def _near_node(net, x, y, extra=0.0):
    """True near a junction, where nothing should be built."""
    for (nx, ny), pad in zip(net.nodes, net.node_pads):
        if math.hypot(x - nx, y - ny) < pad + JUNCTION_CLEAR + extra:
            return True
    return False


def _near_endpoint(net, x, y):
    ends = [(x0, y0) for x0, y0, _ in net.sources] + list(net.goals)
    return any(math.hypot(x - ex, y - ey) < ENDPOINT_CLEAR for ex, ey in ends)


def _fits(placed, x, y, radius):
    for px, py, pr in placed:
        if math.hypot(x - px, y - py) < radius + pr:
            return False
    return True


def generate(net, rng: np.random.Generator, density: float = 1.0,
             margin: float = 0.0) -> Scenery:
    """Buildings and street trees for one road network.

    `density` is the fraction of frontage plots actually built on — 1.0 is a
    continuous terrace, lower leaves gaps. `margin` is accepted and ignored;
    frontage placement has no use for it.
    """
    if density <= 0.0:
        return Scenery.empty()

    placed: list[tuple[float, float, float]] = []
    boxes, trees = [], []

    for index, piece in enumerate(net.pieces):
        if piece.lanes < MIN_LANES or piece.length < 2 * JUNCTION_CLEAR:
            continue

        for sign in (1.0, -1.0):
            # -- buildings, stepping along the frontage ------------------
            s = JUNCTION_CLEAR
            while s < piece.length - JUNCTION_CLEAR:
                frontage = float(rng.uniform(*BUILDING_FRONTAGE))
                depth = float(rng.uniform(*BUILDING_DEPTH))
                s += frontage / 2.0
                if s > piece.length - JUNCTION_CLEAR:
                    break

                cx, cy = piece.point(s)
                tx, ty = piece.tangent(s)
                nx, ny = ty * sign, -tx * sign
                out = piece.half_width + PAVEMENT + FRONT_GAP + depth / 2.0
                bx, by = cx + nx * out, cy + ny * out
                radius = math.hypot(frontage, depth) / 2.0

                ok = (rng.random() < density
                      and not _near_node(net, bx, by)
                      and not _near_endpoint(net, bx, by)
                      and not _blocked(net, bx, by, radius + 1.0)
                      and _fits(placed, bx, by, radius))
                if ok:
                    placed.append((bx, by, radius))
                    boxes.append((bx, by, frontage / 2.0, depth / 2.0,
                                  math.atan2(ty, tx),
                                  float(rng.uniform(*BUILDING_HEIGHT))))
                s += frontage / 2.0 + float(rng.uniform(*PLOT_GAP))

            # -- street trees in the verge -------------------------------
            s = JUNCTION_CLEAR
            while s < piece.length - JUNCTION_CLEAR:
                cx, cy = piece.point(s)
                tx, ty = piece.tangent(s)
                nx, ny = ty * sign, -tx * sign
                radius = float(rng.uniform(*TREE_RADIUS))
                out = piece.half_width + PAVEMENT + TREE_SETBACK
                px, py = cx + nx * out, cy + ny * out

                if (rng.random() < density
                        and not _near_node(net, px, py)
                        and not _near_endpoint(net, px, py)
                        and not _blocked(net, px, py, radius + 0.5)
                        and _fits(placed, px, py, radius)):
                    placed.append((px, py, radius))
                    trees.append((px, py, radius,
                                  float(rng.uniform(*TREE_HEIGHT))))
                s += float(rng.uniform(*TREE_SPACING))

    return Scenery(np.array(boxes or np.zeros((0, 6))),
                   np.array(trees or np.zeros((0, 4))))
