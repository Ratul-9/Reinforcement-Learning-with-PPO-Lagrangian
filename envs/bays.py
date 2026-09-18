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

## A bay is a lay-by, not a stub

Bays used to be one-lane stubs running perpendicular out of the
carriageway. Twenty-two of those down a street render as a comb of teeth,
and they are not how on-street parking works anywhere.

A bay is now a **lay-by**: a pocket alongside the kerb, parallel to the
road, entered by a short taper and left by another. Three pieces —
taper in, the bay itself, taper out — with the bay parallel to the traffic
it sits beside, so a vehicle parked in one is oriented the way a parked car
actually is.

That also makes the manoeuvre the right one. Reversing into a perpendicular
stub is a different (and easier) problem from parallel parking, which needs
the vehicle to judge a gap alongside moving traffic.

The road gains two junction nodes per bay, so a piece is **split twice**.
Only straight pieces are split: splitting an arc means recovering its
centre and sweep for both halves, and nobody parks on a roundabout ring.
"""

from __future__ import annotations

import math

import numpy as np

from envs.road_network import RoadNetwork, _Straight

BAY_LANES = 1            # a bay is one lane wide, by definition
BAY_LENGTH = 7.0         # the parking space itself, along the kerb
BAY_TAPER = 4.0          # the angled entry and exit at each end
MIN_SPLIT_GAP = 8.0      # never split a piece within this of either end
EXTRA_BAYS = 12          # beyond the ones inheriting a declared endpoint
# Centre-to-centre spacing between two bays' entrances. A bay is one lane
# wide, so anything under about a lane and a half puts two parked cars in
# the same space — and `_nearest_straight` clamps a candidate away from a
# piece's ends, which makes every random point near an end land on the SAME
# split position unless something rejects it.
# A lay-by consumes BAY_LENGTH + 2 * BAY_TAPER of kerb, so two of them any
# closer than that would share tarmac.
BAY_SPACING = BAY_LENGTH + 2 * BAY_TAPER + 6.0


def attach(net: RoadNetwork, rng: np.random.Generator,
           extra: int = EXTRA_BAYS) -> RoadNetwork:
    """Add bays to `net` and make them its sources and goals.

    Returns the same network, re-finalised. Call before generating scenery,
    so the scenery placement sees the bays as road and keeps clear of them.
    """
    targets = [(x, y) for x, y, _ in net.sources] + list(net.goals)
    n_declared = len(targets)

    bays: list[tuple[float, float, float]] = []      # x, y, heading_deg
    entrances: list[tuple[float, float]] = []        # where each meets the road
    for x, y in targets:
        bay = _attach_one(net, x, y, entrances)
        if bay is not None:
            bays.append(bay)

    # Scatter the rest. Sampling by piece length rather than by piece keeps
    # a long arterial from getting the same number of bays as a short stub.
    attempts = 0
    while len(bays) - len(targets) < extra and attempts < extra * 8:
        attempts += 1
        x, y, _h = net.sample_pose(rng, lane_bias=False)
        bay = _attach_one(net, x, y, entrances)
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


def _attach_one(net: RoadNetwork, x: float, y: float, entrances: list):
    """Cut a lay-by into the kerb nearest (x, y).

    Returns the bay's pose `(x, y, heading_deg)`, parallel to the road, or
    None when there is nowhere here to put one. `entrances` holds the
    road-side midpoints of the bays placed so far and is appended to on
    success.
    """
    piece_index, s = _nearest_straight(net, x, y)
    if piece_index is None:
        return None

    piece = net.pieces[piece_index]
    half = BAY_LENGTH / 2.0 + BAY_TAPER
    if s - half < MIN_SPLIT_GAP or s + half > piece.length - MIN_SPLIT_GAP:
        return None

    mx, my = piece.point(s)
    if any(math.hypot(mx - ex, my - ey) < BAY_SPACING for ex, ey in entrances):
        return None
    tx, ty = piece.tangent(s)
    offset = piece.half_width + BAY_LANES * net.lane_width / 2.0

    for sign in (1.0, -1.0):
        nx, ny = ty * sign, -tx * sign          # kerb side: right, then left
        # The four corners of the lay-by: two taper feet on the centreline,
        # two bay ends offset out to the kerb.
        foot_a = piece.point(s - half)
        foot_b = piece.point(s + half)
        end_a = (mx - tx * BAY_LENGTH / 2.0 + nx * offset,
                 my - ty * BAY_LENGTH / 2.0 + ny * offset)
        end_b = (mx + tx * BAY_LENGTH / 2.0 + nx * offset,
                 my + ty * BAY_LENGTH / 2.0 + ny * offset)

        # Both ends must be clear of every other road, or the lay-by is cut
        # into the middle of a parallel street.
        clearance = -BAY_LANES * net.lane_width / 2.0
        if (net.road_margin(*end_a) > clearance
                or net.road_margin(*end_b) > clearance):
            continue

        node_a = _split_at_point(net, foot_a)
        node_b = _split_at_point(net, foot_b)
        if node_a is None or node_b is None:
            return None

        bay_a = net.add_node(*end_a, kind="end")
        bay_b = net.add_node(*end_b, kind="end")
        net.add_straight(node_a, bay_a, BAY_LANES)   # taper in
        net.add_straight(bay_a, bay_b, BAY_LANES)    # the space itself
        net.add_straight(bay_b, node_b, BAY_LANES)   # taper out
        entrances.append((mx, my))

        # Parked parallel to the traffic beside it, facing the way the
        # taper-out leads, so pulling away is a forward move.
        centre = ((end_a[0] + end_b[0]) / 2.0, (end_a[1] + end_b[1]) / 2.0)
        return (centre[0], centre[1], math.degrees(math.atan2(-tx, ty)))
    return None


def _split_at_point(net: RoadNetwork, point):
    """Split whichever straight passes through `point`, returning the node."""
    index, s = _nearest_straight(net, point[0], point[1], gap=2.0)
    if index is None:
        return None
    return _split(net, index, s, gap=2.0)


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

    # Pad sized to the STUB, not auto-sized to the arterial.
    #
    # `finalize` gives any non-"end" node a pad of 1.45x the half-width of
    # the widest road meeting it, which is right for a crossroads and very
    # wrong here: a bay attachment is a one-lane stub joining a
    # carriageway, and auto-sizing gave it a 5-7 m disc on a road 5.2 m
    # wide. Twenty-two of those down one street read as scalloped blobs
    # across the tarmac — and because `road_margin` counts a pad as
    # drivable, they also inflated the drivable area past the kerb, so the
    # off-road cost was wrong at every bay.
    #
    # The stub's own half-width sits inside the arterial's ribbon, so it
    # adds no drivable area at all while still rounding the turn-in.
    stub_pad = BAY_LANES * net.lane_width / 2.0
    mid = net.add_node(*piece.point(s), kind="tee", pad=stub_pad)
    a, b = piece.node_a, piece.node_b
    lanes = piece.lanes
    net.pieces.pop(piece_index)
    net.add_straight(a, mid, lanes)
    net.add_straight(mid, b, lanes)
    return mid
