"""Lane graph — discrete lanes derived from the centreline graph.

`road_network.py` models a road as ONE centreline carrying a lane *count* and
a half-width. That is enough to ask "am I on the road" and "how far off the
centre am I", and it is not enough to ask the two questions the cost
functions actually need:

    which lane am I in, and is it going my way?
    how far am I from the middle of THAT lane?

So this module derives, per piece, the set of lane centrelines the count
implies — and derives rather than stores them, so `road_network.py` stays the
verbatim port it is and a network loaded from a PNG or built by any scenario
gets lanes for free.

## The convention

Offsets are signed lateral distances from the piece's own centreline,
**positive to the LEFT of the piece's tangent** — the same sign
`_Piece.closest` returns, so nothing has to be flipped at the boundary.

**Which side traffic drives on is a parameter**, not a constant. Under
right-hand traffic a lane on the right half of the carriageway (negative
offset) carries traffic along +tangent; under left-hand traffic the left
half does. Everything downstream — spawns, the wrong-way cost, the lane a
route waypoint is shifted into, which lane a vehicle belongs in — reads the
lane graph, so one flag flips all of it consistently.

A ONE-WAY piece ignores the side rule entirely: every lane runs +tangent,
across the full width. A dual carriageway, a slip road and a gyratory are
one-way, and modelling a three-lane motorway carriageway as two-way would
give it one and a half lanes in each direction.

Under two-way traffic:

    lanes = 1   one bidirectional lane on the centreline. A parking bay or a
                single-track aisle: there is no wrong side of a road this
                narrow, and pretending otherwise would charge every vehicle
                in a parking lot for existing.
    lanes = 2n  n lanes each way.
    lanes = 2n+1 (n >= 1)
                n each way plus a CENTRE lane, direction 0. A shared turning
                lane: drivable, and occupying it is never a wrong-way
                violation, because the whole point of one is that traffic
                from both directions waits in it to turn.

That last case is a modelling choice, not a fact — three-lane urban roads are
sometimes two-plus-one instead. It is here because the alternative (handing
the odd lane to one direction) makes a road that is asymmetric for no reason
anyone painted.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np


class Lane(NamedTuple):
    """One lane of one piece.

    `index` runs left to right across the carriageway (0 is the leftmost lane
    looking along +tangent), `offset` is its centre's signed lateral distance,
    and `direction` is +1 for travel along the piece's tangent, -1 against it,
    0 for a shared centre lane.
    """

    index: int
    offset: float
    direction: int
    width: float


class LaneFix(NamedTuple):
    """Where a pose sits in the lane graph — what `LaneGraph.locate` answers."""

    piece: int
    s: float
    lane: Lane
    offset: float        # signed distance from THIS lane's centre, left-positive
    wrong_way: bool      # travelling against the lane's direction
    forward: bool        # travelling along the piece's +tangent
    on_road: bool


def lanes_of(piece, drive_side: str = "right") -> list[Lane]:
    """The lanes of one piece, left to right across the carriageway."""
    n = max(1, int(piece.lanes))
    width = 2.0 * piece.half_width / n

    if getattr(piece, "oneway", False):
        # Every lane runs the piece's own way; no wrong side exists.
        return [Lane(k, piece.half_width - (k + 0.5) * width, 1, width)
                for k in range(n)]

    if n == 1:
        return [Lane(0, 0.0, 0, width)]

    # Under right-hand traffic the nearside is the right of travel, which is
    # the NEGATIVE offset half; under left-hand traffic it is the positive
    # half. One sign carries the whole difference.
    keep_left = drive_side == "left"
    out = []
    for k in range(n):
        offset = piece.half_width - (k + 0.5) * width
        if n % 2 == 1 and k == n // 2:
            direction = 0                     # the shared centre lane
        elif keep_left:
            direction = 1 if offset > 0.0 else -1
        else:
            direction = -1 if offset > 0.0 else 1
        out.append(Lane(k, offset, direction, width))
    return out


class LaneGraph:
    """Lanes for a whole network. Built once per World, read every step."""

    def __init__(self, net, drive_side: str = "right"):
        if drive_side not in ("left", "right"):
            raise ValueError(f"drive_side must be 'left' or 'right', "
                             f"got {drive_side!r}")
        self.net = net
        self.drive_side = drive_side
        self._lanes = [lanes_of(p, drive_side) for p in net.pieces]

    def __len__(self) -> int:
        return sum(len(l) for l in self._lanes)

    def __repr__(self) -> str:
        return f"LaneGraph(pieces={len(self._lanes)}, lanes={len(self)})"

    def lanes(self, piece_index: int) -> list[Lane]:
        return self._lanes[piece_index]

    def centreline(self, piece_index: int, lane_index: int) -> np.ndarray:
        """The lane's own centreline as a polyline — what the renderers draw
        and what a lane-following controller would track."""
        piece = self.net.pieces[piece_index]
        lane = self._lanes[piece_index][lane_index]
        poly = piece.polyline(0.0, piece.length)
        return _offset_polyline(poly, lane.offset)

    def dividers(self, piece_index: int) -> list[float]:
        """Lateral offsets of the lines painted BETWEEN lanes — one fewer
        than there are lanes, and not the same thing as the lane centres."""
        lanes = self._lanes[piece_index]
        if len(lanes) < 2:
            return []
        return [(a.offset + b.offset) / 2.0 for a, b in zip(lanes, lanes[1:])]

    def locate(self, x: float, y: float, heading: float) -> LaneFix:
        """Which lane a pose is in, and whether it is facing the right way.

        The piece is chosen the way `RoadNetwork._project_full` chooses it —
        by which road's EDGE is nearest, not whose centreline is — so a
        vehicle in the outside lane of an arterial is not suddenly reported
        as being in a parking stub whose centreline happens to be closer.
        """
        i, s, lateral, _d, margin = self.net._project_full(x, y)
        piece = self.net.pieces[i]
        lanes = self._lanes[i]

        lane = min(lanes, key=lambda ln: abs(lateral - ln.offset))
        tx, ty = piece.tangent(s)
        forward = math.cos(heading - math.atan2(ty, tx)) >= 0.0
        travelling = 1 if forward else -1
        return LaneFix(
            piece=i, s=s, lane=lane, offset=lateral - lane.offset,
            wrong_way=lane.direction != 0 and lane.direction != travelling,
            forward=forward, on_road=margin >= 0.0,
        )

    def lane_for_travel(self, piece_index: int, forward: bool) -> Lane:
        """The lane a vehicle travelling this way down this piece belongs in —
        the rightmost one going its direction, which is where a vehicle that
        is not overtaking should be. Falls back to the shared centre lane, and
        then to the single lane, for pieces too narrow to have a choice."""
        lanes = self._lanes[piece_index]
        want = 1 if forward else -1
        mine = [ln for ln in lanes if ln.direction == want]
        if mine:
            # The NEARSIDE lane — the one a vehicle not overtaking belongs
            # in. Under right-hand traffic that is the most negative offset
            # relative to travel; under left-hand traffic the most positive.
            sign = want if self.drive_side == "right" else -want
            return min(mine, key=lambda ln: ln.offset * sign)
        return next((ln for ln in lanes if ln.direction == 0), lanes[0])


def _offset_polyline(poly: np.ndarray, d: float) -> np.ndarray:
    """A polyline shifted `d` metres to the LEFT of travel (matching the
    lateral sign convention)."""
    poly = np.asarray(poly, dtype=float)
    if len(poly) < 2 or abs(d) < 1e-9:
        return poly
    seg = np.gradient(poly, axis=0)
    n = np.hypot(seg[:, 0], seg[:, 1])
    n[n < 1e-9] = 1.0
    return np.stack([poly[:, 0] - seg[:, 1] / n * d,
                     poly[:, 1] + seg[:, 0] / n * d], axis=1)
