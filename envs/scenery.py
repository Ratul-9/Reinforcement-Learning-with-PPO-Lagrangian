"""Scenery — buildings and trees placed procedurally from the road graph.

Nothing here is authored per scenario. A layout is a graph of centrelines
(`road_network.py`) and that graph already says where the road is NOT, so the
scenery is derived from it: candidates are drawn on a jittered grid over the
network's bounds and kept or rejected by how far they are from the nearest
road edge.

    distance from road edge            what goes there
    --------------------------------   ---------------------------
    < verge_min                         nothing (kerb, too close)
    verge_min .. verge_max              trees (street planting)
    building_setback .. building_max    buildings (block interiors)
    > building_max                      nothing (open country)

Both bands are closed at the top, and the building one being closed is what
makes the result read as a town rather than as a field of boxes: a built-up
frontage exists BECAUSE it faces a street, so a candidate 200 m from the
nearest road is not a building site. It also keeps the object count
proportional to the network instead of to the square of the map margin.

A roundabout's central island is the one place that is neither road nor
building land — it is off-road by the margin test, sits well inside the
network, and would otherwise be the most attractive building plot on the
map. It is excluded explicitly.

Two footprint shapes, and the split is a sensor decision rather than an
aesthetic one. A tree is a circle: rotationally symmetric, one distance test.
A building is a rotated rectangle, because a row of buildings along a street
is a flat wall to a lidar and modelling it as circles would leave the policy
sight-lines through gaps that do not exist in the picture. Both are what the
renderer draws AND what the raycast hits, so what the agent sees and what a
human sees can never drift apart.

Buildings are yawed to the nearest centreline's tangent so they face the
street they stand on, which costs one `heading_at` call per building and is
the whole difference between "a town" and "boxes on a lawn".
"""

from __future__ import annotations

import math

import numpy as np

# Placement bands, in metres from the nearest road EDGE (not centreline).
VERGE_MIN = 2.5          # kerb clearance: nothing closer than this
VERGE_MAX = 7.0          # trees live in the strip between the two
BUILDING_SETBACK = 12.0  # buildings start here, well back off the carriageway
BUILDING_MAX = 55.0      # and stop here: past this is open country, not a plot

# Candidate grid. Step is the spacing between candidate cells; each candidate
# is jittered inside its own cell so the result does not read as a lattice.
TREE_STEP = 11.0
BUILDING_STEP = 26.0
JITTER = 0.40            # fraction of a cell, each way

# Sizes. Buildings are drawn from a range rather than fixed so a block is not
# a row of identical cubes; the ranges are half-extents in metres.
TREE_RADIUS = (1.0, 2.2)
TREE_HEIGHT = (4.0, 9.0)
BUILDING_HALF_L = (6.0, 14.0)
BUILDING_HALF_W = (5.0, 10.0)
BUILDING_HEIGHT = (6.0, 24.0)

# Nothing is placed within this distance of a declared source or goal, so a
# scenario's spawn bay never opens onto a tree trunk.
ENDPOINT_CLEAR = 8.0


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


def _candidates(bounds, step: float, rng: np.random.Generator) -> np.ndarray:
    """Jittered grid points covering `bounds` — an (N, 2) array."""
    x0, y0, x1, y1 = bounds
    xs = np.arange(x0, x1 + step, step)
    ys = np.arange(y0, y1 + step, step)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    pts += rng.uniform(-JITTER, JITTER, size=pts.shape) * step
    return pts


def _clearances(net, pts: np.ndarray) -> np.ndarray:
    """Distance from each point to the nearest road edge, positive when OFF
    the road — i.e. `-road_margin`. Looped because `road_margin` walks every
    piece; this runs once per episode at reset, not per step."""
    return np.array([-net.road_margin(float(x), float(y)) for x, y in pts])


def _endpoint_mask(net, pts: np.ndarray) -> np.ndarray:
    """True where a point is far enough from every declared source and goal."""
    ends = [(x, y) for x, y, _ in net.sources] + list(net.goals)
    if not ends:
        return np.ones(len(pts), dtype=bool)
    ends = np.asarray(ends, dtype=float)
    d = np.hypot(pts[:, None, 0] - ends[None, :, 0],
                 pts[:, None, 1] - ends[None, :, 1])
    return d.min(axis=1) > ENDPOINT_CLEAR


def _island_mask(net, pts: np.ndarray, clearance: float = 0.0) -> np.ndarray:
    """True where a point is outside every roundabout island."""
    if not net.islands:
        return np.ones(len(pts), dtype=bool)
    ok = np.ones(len(pts), dtype=bool)
    for cx, cy, r in net.islands:
        ok &= np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) > r + clearance
    return ok


def _no_overlap(placed: list, x: float, y: float, r: float) -> bool:
    """Reject a candidate whose circumscribed circle touches one already
    placed. O(n^2) over a few hundred objects at reset — fine."""
    for px, py, pr in placed:
        if math.hypot(x - px, y - py) < r + pr:
            return False
    return True


def generate(net, rng: np.random.Generator, density: float = 1.0,
             margin: float = 60.0) -> Scenery:
    """Buildings and trees for one road network.

    `density` scales how many of the accepted candidates are actually kept
    (1.0 = all of them, 0.0 = bare network); `margin` is how far past the
    network's own bounds scenery may spread, which is what stops a layout
    from ending in a hard edge of empty ground.
    """
    if density <= 0.0:
        return Scenery.empty()

    x0, y0, x1, y1 = net.bounds()
    bounds = (x0 - margin, y0 - margin, x1 + margin, y1 + margin)
    placed: list[tuple[float, float, float]] = []

    # -- buildings first: they are larger, so they get first refusal on space.
    boxes = []
    pts = _candidates(bounds, BUILDING_STEP, rng)
    clear = _clearances(net, pts)
    keep = (_endpoint_mask(net, pts) & _island_mask(net, pts, BUILDING_SETBACK)
            & (clear > BUILDING_SETBACK) & (clear < BUILDING_MAX))
    pts = pts[keep]
    for x, y in pts[rng.random(len(pts)) < density]:
        hl = rng.uniform(*BUILDING_HALF_L)
        hw = rng.uniform(*BUILDING_HALF_W)
        r = math.hypot(hl, hw)
        if not _no_overlap(placed, x, y, r):
            continue
        # A building's own footprint must clear the road too, not just its
        # centre: a 14 m-long block dropped 12 m off the kerb would otherwise
        # have one corner standing in the outside lane.
        if -net.road_margin(float(x), float(y)) < r + VERGE_MIN:
            continue
        placed.append((float(x), float(y), r))
        boxes.append((x, y, hl, hw, net.heading_at(float(x), float(y)),
                      rng.uniform(*BUILDING_HEIGHT)))

    # -- trees: the verge strip, plus whatever gaps the buildings left.
    trees = []
    pts = _candidates(bounds, TREE_STEP, rng)
    clear = _clearances(net, pts)
    keep = (_endpoint_mask(net, pts) & _island_mask(net, pts, VERGE_MIN)
            & (clear > VERGE_MIN) & (clear < VERGE_MAX))
    pts = pts[keep]
    for x, y in pts[rng.random(len(pts)) < density]:
        r = rng.uniform(*TREE_RADIUS)
        if not _no_overlap(placed, x, y, r):
            continue
        placed.append((float(x), float(y), r))
        trees.append((x, y, r, rng.uniform(*TREE_HEIGHT)))

    return Scenery(np.array(boxes or np.zeros((0, 6))),
                   np.array(trees or np.zeros((0, 4))))
