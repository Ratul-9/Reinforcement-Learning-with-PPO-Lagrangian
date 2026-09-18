"""Parking bays — the canonical task is bay to bay, on every scenario.

The episode we actually want is "leave a parking space, drive the scenario,
park in another parking space". Only `parking_lot` came with bays; the other
seven start and finish at a point on a road. This module attaches bays to any
built network, so the task definition is the same across the whole
curriculum instead of being one thing in the parking lot and another thing
everywhere else.

## Why the bays go where the scenario's endpoints already were

Bolting bays on at random would quietly delete the scenario: a roundabout
whose source and destination are two bays on the same side of the map is not
a roundabout task any more, it is a short drive with a roundabout visible in
the distance. So every declared source and goal gets a bay attached to the
road **at that point** and then hands its role over to it. The route still
enters the merge from the ramp, still crosses the unsignalised junction —
the episode just now begins and ends stationary in a bay, which is also the
hardest part of the manoeuvre and the part a road-to-road task never
exercises.

Extra bays are then scattered along the network for the other 18-28 vehicles,
because a scenario declares two endpoints and the traffic needs twenty.

## What attaching one costs

A bay is a one-lane stub running perpendicular out of the carriageway, so
the road it leaves has to gain a junction node where the stub meets it. That
means **splitting a piece** — the stub's node has to be a real node of the
graph or nothing can route through it. Only straight pieces are split:
splitting an arc means recovering its centre and sweep for both halves, and
nobody parks on a roundabout ring.
"""

from __future__ import annotations

import math

import numpy as np

from envs.road_network import RoadNetwork, _Straight

BAY_DEPTH = 5.5          # how far the bay sticks out past the road edge
BAY_LANES = 1            # a bay is one lane wide, by definition
MIN_SPLIT_GAP = 8.0      # never split a piece within this of either end
EXTRA_BAYS = 18          # beyond the ones inheriting a declared endpoint
# Centre-to-centre spacing between two bays' entrances. A bay is one lane
# wide, so anything under about a lane and a half puts two parked cars in
# the same space — and `_nearest_straight` clamps a candidate away from a
# piece's ends, which makes every random point near an end land on the SAME
# split position unless something rejects it.
BAY_SPACING = 7.5


def attach(net: RoadNetwork, rng: np.random.Generator,
           extra: int = EXTRA_BAYS, depth: float = BAY_DEPTH) -> RoadNetwork:
    """Add bays to `net` and make them its sources and goals.

    Returns the same network, re-finalised. Call before generating scenery,
    so the scenery placement sees the bays as road and keeps clear of them.
    """
    targets = [(x, y) for x, y, _ in net.sources] + list(net.goals)
    n_declared = len(targets)

    bays: list[tuple[float, float, float]] = []      # x, y, heading_deg
    entrances: list[tuple[float, float]] = []        # where each meets the road
    for x, y in targets:
        bay = _attach_one(net, x, y, depth, entrances)
        if bay is not None:
            bays.append(bay)

    # Scatter the rest. Sampling by piece length rather than by piece keeps
    # a long arterial from getting the same number of bays as a short stub.
    attempts = 0
    while len(bays) - len(targets) < extra and attempts < extra * 8:
        attempts += 1
        x, y, _h = net.sample_pose(rng, lane_bias=False)
        bay = _attach_one(net, x, y, depth, entrances)
        if bay is not None:
            bays.append(bay)

    if not bays:
        return net.finalize()

    # Bays take over as the episode endpoints. Declared endpoints go first in
    # `bays`, so the scenario's own two survive even if the scatter pass adds
    # nothing — a layout whose geometry refuses every extra bay still runs
    # its intended task.
    net.sources = [(x, y, h) for x, y, h in bays]
    net.goals = [(x, y) for x, y, _ in bays]
    # NOT "bays": parking_lot's own spec already uses that key for
    # bays-per-aisle, and quietly overwriting it would change its layout.
    net.spec = dict(net.spec)
    net.spec["parking_bays"] = len(bays)
    net.spec["parking_bays_declared"] = n_declared
    return net.finalize()


def _attach_one(net: RoadNetwork, x: float, y: float, depth: float,
                entrances: list):
    """Split the straight nearest (x, y) and run a bay out of it.

    Returns the bay's pose `(x, y, heading_deg)` facing OUT toward the road,
    or None when there is nowhere here to put one. `entrances` is the list of
    road-side points of the bays placed so far, and is appended to on
    success.
    """
    piece_index, s = _nearest_straight(net, x, y)
    if piece_index is None:
        return None

    piece = net.pieces[piece_index]
    px, py = piece.point(s)
    if any(math.hypot(px - ex, py - ey) < BAY_SPACING for ex, ey in entrances):
        return None
    tx, ty = piece.tangent(s)

    # Try both sides; take the first whose far end is clear of every road.
    # A bay driven out into the middle of a parallel street would be
    # drivable, connected, and wrong.
    reach = piece.half_width + depth
    for sign in (1.0, -1.0):
        nx, ny = ty * sign, -tx * sign               # right of travel, then left
        ex, ey = px + nx * reach, py + ny * reach
        if net.road_margin(ex, ey) > -BAY_LANES * net.lane_width:
            continue

        node = _split(net, piece_index, s)
        if node is None:
            return None
        end = net.add_node(ex, ey, "end")
        net.add_straight(node, end, BAY_LANES)
        entrances.append((px, py))
        # Facing out of the bay is facing back toward the road, i.e. along
        # -(nx, ny). H is the Panda angle road_network stores headings in.
        return (ex, ey, math.degrees(math.atan2(nx, -ny)))
    return None


def _nearest_straight(net: RoadNetwork, x: float, y: float,
                      gap: float = MIN_SPLIT_GAP):
    """(piece index, s) of the nearest straight with room to be split, or
    (None, None).

    `gap` is how close to a piece's ends the split may fall. It is generous
    when placing bays — a junction needs room around it — and small when
    RECONNECTING one, because by then the road has already been chopped into
    short segments by the bays either side of it and a generous gap would
    reject every piece on the map.
    """
    best = (None, None, float("inf"))
    for i, piece in enumerate(net.pieces):
        if not isinstance(piece, _Straight) or piece.length < 2 * gap:
            continue
        s, _lateral, d = piece.closest(x, y)
        s = min(max(s, gap), piece.length - gap)
        if d < best[2]:
            best = (i, s, d)
    return best[0], best[1]


def _split(net: RoadNetwork, piece_index: int, s: float,
           gap: float = MIN_SPLIT_GAP):
    """Cut a straight in two at `s`, returning the new node between them.

    The piece is replaced rather than added to: leaving the original in place
    would give the graph two routes over the same tarmac, one of which does
    not pass through the bay's junction, and `route_distance` would take the
    one that skips it.
    """
    piece = net.pieces[piece_index]
    if not isinstance(piece, _Straight):
        return None
    if s < gap or s > piece.length - gap:
        return None

    mid = net.add_node(*piece.point(s), kind="tee")
    a, b = piece.node_a, piece.node_b
    lanes = piece.lanes
    net.pieces.pop(piece_index)
    net.add_straight(a, mid, lanes)
    net.add_straight(mid, b, lanes)
    return mid
