"""Perimeter ring — give every dangling arm somewhere to go.

The scenario builders produce conflict geometry: a crossroads, a merge, a
roundabout. What they do not produce is a *place*. Every layout ended with
four to six arms that simply stop in open ground — a vehicle driving to the
end of one meets no junction, no turning head, nothing. It is the single
biggest reason the maps did not read as complete environments.

    scenario          dead ends before
    intersection_x           6
    merge_ramp               5
    weave_roundabout         5
    roundabout_yield         4
    manhattan                4
    left_turn                4
    lane_closure             4

This module runs a ring road around the outside and extends each loose arm
out to meet it. The conflict geometry is untouched — the crossings, the
merge taper, the roundabout are exactly as their builder made them — and
the map becomes closed and fully connected, so a route exists between any
two points and no road terminates in a field.

## Why a ring rather than joining arms to each other

Joining arm ends pairwise would run new roads straight across the middle of
the layout, through the blocks and often through the junction the scenario
is *about*. A ring stays outside the network's own bounds, so it cannot
intersect anything the builder placed, and it is also what a real town has:
arterials running out to a bypass.

Arms are extended along their OWN direction rather than to the nearest ring
point, so the extension is a continuation of the road rather than a kink.
"""

from __future__ import annotations

import math

import numpy as np

from envs.road_network import RoadNetwork, _Straight

RING_MARGIN = 32.0        # how far outside the network's bounds the ring runs
RING_LANES = 2
CORNER_PAD = 1.45         # matches road_network's JUNCTION_PAD
MIN_JOIN_SPACING = 14.0   # two arms may not meet the ring at the same spot


def close_network(net: RoadNetwork, margin: float = RING_MARGIN,
                  lanes: int = RING_LANES) -> RoadNetwork:
    """Add a perimeter ring and connect every dangling arm to it.

    Returns the same network, re-finalised. Call BEFORE attaching bays: a
    bay is a legitimate dead end and must not be extended to the ring, and
    running this first means the bay pass simply never sees one.
    """
    # A layout may declare itself open. A motorway ends at the edge of the
    # map because the real road continues beyond it, and ringing it would
    # produce a bypass loop around a motorway, which exists nowhere.
    if not net.spec.get("closed", True):
        return net.finalize()

    dead = _dangling(net)
    if not dead:
        return net.finalize()

    x0, y0, x1, y1 = net.bounds()
    rect = (x0 - margin, y0 - margin, x1 + margin, y1 + margin)
    ring_pieces = _add_ring(net, rect, lanes)

    joined: list[tuple[float, float]] = []
    for node, direction in dead:
        hit = _ray_to_rect(net.nodes[node], direction, rect)
        if hit is None:
            continue
        if any(math.hypot(hit[0] - jx, hit[1] - jy) < MIN_JOIN_SPACING
               for jx, jy in joined):
            continue
        target = _split_ring(net, ring_pieces, hit, lanes)
        if target is None:
            continue
        # The arm keeps its own width out to the ring: a three-lane
        # arterial that narrows to two at the boundary is a merge nobody
        # asked for, in a scenario that may already contain one.
        net.add_straight(node, target, net.pieces[_piece_at(net, node)].lanes)
        joined.append(hit)

    net.spec = dict(net.spec)
    net.spec["perimeter"] = True
    return net.finalize()


def _dangling(net: RoadNetwork) -> list:
    """Degree-one nodes on real roads, with the direction the road points.

    One-lane pieces are skipped: those are parking bays and aisles, and a
    bay is *supposed* to be a dead end.
    """
    degree: dict[int, int] = {}
    for piece in net.pieces:
        degree[piece.node_a] = degree.get(piece.node_a, 0) + 1
        degree[piece.node_b] = degree.get(piece.node_b, 0) + 1

    out = []
    for index, piece in enumerate(net.pieces):
        if piece.lanes < 2:
            continue
        for node, s_end in ((piece.node_a, 0.0), (piece.node_b, piece.length)):
            if degree.get(node, 0) != 1:
                continue
            tx, ty = piece.tangent(s_end)
            # Point outward: at node_a the piece runs away from us.
            if s_end == 0.0:
                tx, ty = -tx, -ty
            out.append((node, (tx, ty)))
    return out


def _piece_at(net: RoadNetwork, node: int) -> int:
    for index, piece in enumerate(net.pieces):
        if piece.node_a == node or piece.node_b == node:
            return index
    return 0


def _add_ring(net: RoadNetwork, rect, lanes: int) -> list:
    """Four straights around `rect`. Returns their piece indices."""
    x0, y0, x1, y1 = rect
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    nodes = [net.add_node(cx, cy, "tee") for cx, cy in corners]
    first = len(net.pieces)
    for a, b in zip(nodes, nodes[1:] + nodes[:1]):
        net.add_straight(a, b, lanes)
    return list(range(first, len(net.pieces)))


def _ray_to_rect(origin, direction, rect):
    """Where a ray from inside `rect` leaves it, or None."""
    ox, oy = float(origin[0]), float(origin[1])
    dx, dy = float(direction[0]), float(direction[1])
    x0, y0, x1, y1 = rect

    best = None
    for value, delta, origin_axis in ((x0, dx, ox), (x1, dx, ox),
                                      (y0, dy, oy), (y1, dy, oy)):
        if abs(delta) < 1e-9:
            continue
        t = (value - origin_axis) / delta
        if t <= 1e-6:
            continue
        hx, hy = ox + dx * t, oy + dy * t
        # On the rectangle, not merely on the infinite line through an edge.
        if not (x0 - 1e-6 <= hx <= x1 + 1e-6 and y0 - 1e-6 <= hy <= y1 + 1e-6):
            continue
        if best is None or t < best[0]:
            best = (t, (hx, hy))
    return None if best is None else best[1]


def _split_ring(net: RoadNetwork, ring_pieces: list, point, lanes: int):
    """Put a node on whichever ring edge contains `point`, splitting it.

    `ring_pieces` is updated in place, because each split replaces one edge
    with two and every later arm has to be able to find them.
    """
    target = None
    for slot, index in enumerate(ring_pieces):
        if index >= len(net.pieces):
            continue
        piece = net.pieces[index]
        if not isinstance(piece, _Straight):
            continue
        s, _lateral, distance = piece.closest(point[0], point[1])
        if distance > 1.0 or s < 2.0 or s > piece.length - 2.0:
            continue
        target = (slot, index, s, piece)
        break
    if target is None:
        return None

    slot, index, s, piece = target
    node = net.add_node(*piece.point(s), kind="tee")
    a, b = piece.node_a, piece.node_b
    net.pieces.pop(index)
    # Indices after the removed one shift down by one.
    ring_pieces[:] = [i - 1 if i > index else i
                      for i in ring_pieces if i != index]
    net.add_straight(a, node, lanes)
    net.add_straight(node, b, lanes)
    ring_pieces.extend([len(net.pieces) - 2, len(net.pieces) - 1])
    return node
