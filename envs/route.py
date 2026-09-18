"""RouteTracker — one fixed path per episode, walked monotonically.

`RoadNetwork.route_distance` re-solves the shortest path from wherever the
vehicle is projected, every step. That is correct as a graph query and wrong
as a reward signal, because the answer can jump: at a junction, near a bay
entrance, or anywhere two pieces overlap, the projection switches piece and
the shortest path switches with it.

Measured on `manhattan`, 16 agents, 900 steps, with the scripted driver:
**257 steps where the route distance moved further than the vehicle did**,
the largest forward jump +255 m and the largest backward jump -416 m. The
progress reward is the change in that number, so a single +255 m jump on a
150 m route pays 1.7 — more than arriving is worth. A policy would have
found that long before a human noticed it, by loitering exactly where the
projection flickers.

Three things break, not one:

* **Reward.** Free progress, farmable, and noise on every other step.
* **Observation.** `navigation[0]` is that distance, so the policy is fed a
  discontinuous input it cannot explain from its own motion.
* **Termination.** The goal test is `route <= GOAL_RADIUS`, so a downward
  jump is a phantom arrival.

The fix is to stop asking the question repeatedly. The route is solved
**once**, when the destination is assigned, and the vehicle is then tracked
*along that fixed polyline*. Progress becomes arc length travelled, which is
continuous by construction and cannot exceed the distance driven.

This is also cheaper than what it replaces: one graph solve per episode
instead of several per step.

## Why the projection is windowed

A route can pass near itself — a loop, a block circled twice, a roundabout.
Projecting onto the whole polyline would let a vehicle halfway round jump to
the geometrically-nearer earlier segment and lose its progress. The search
is therefore restricted to a window ahead of where the vehicle already was,
which is both faster and the only way the answer stays monotone.

A vehicle that leaves the route entirely — shoved off by a collision,
detouring round a stalled lorry — is handled by letting the window slide
forward but never back: it reattaches wherever it rejoins, ahead of where it
left, rather than teleporting its progress backwards.
"""

from __future__ import annotations

import math

import numpy as np

# How far along the polyline the projection may look, in segments, from the
# vehicle's last known position. Wide enough that a fast vehicle cannot
# outrun it in one control step: at 25 m/s and a 0.1 s step that is 2.5 m,
# and route polylines are sampled far finer than that.
_WINDOW = 24


class RouteTracker:
    """The fixed path for one episode, and where along it the vehicle is."""

    def __init__(self, net, start_xy, goal_xy):
        poly = net.route_polyline(start_xy, goal_xy)
        poly = np.asarray(poly, dtype=float).reshape(-1, 2)
        if len(poly) < 2:
            poly = np.array([start_xy, goal_xy], dtype=float)

        # Drop repeated points: a zero-length segment has no direction and
        # would divide by zero in the projection.
        keep = np.ones(len(poly), dtype=bool)
        keep[1:] = np.hypot(*np.diff(poly, axis=0).T) > 1e-9
        self.poly = poly[keep] if keep.sum() >= 2 else poly

        seg = np.diff(self.poly, axis=0)
        self.seg = seg
        self.seg_len = np.hypot(seg[:, 0], seg[:, 1])
        self.cum = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.total = float(self.cum[-1])

        self._i = 0            # index of the segment the vehicle is on
        self.s = 0.0           # arc length travelled along the route

    def __len__(self) -> int:
        return len(self.poly)

    # -- tracking ---------------------------------------------------------

    def advance(self, x: float, y: float) -> float:
        """Update the position along the route; return remaining distance.

        Monotone in the index, not in `s`: a vehicle that reverses genuinely
        loses ground, and should. What it cannot do is jump to a different
        part of the route because that part happens to be nearer.
        """
        lo = self._i
        hi = min(lo + _WINDOW, len(self.seg))
        if hi <= lo:
            self.s = self.total
            return 0.0

        seg = self.seg[lo:hi]
        rel = np.array([x, y]) - self.poly[lo:hi]
        length2 = np.maximum((seg * seg).sum(axis=1), 1e-12)
        t = np.clip((rel * seg).sum(axis=1) / length2, 0.0, 1.0)
        closest = self.poly[lo:hi] + seg * t[:, None]
        d = np.hypot(closest[:, 0] - x, closest[:, 1] - y)
        k = int(d.argmin())

        self._i = lo + k
        self.s = float(self.cum[self._i] + t[k] * self.seg_len[self._i])
        return max(self.total - self.s, 0.0)

    # -- queries ----------------------------------------------------------

    def remaining(self) -> float:
        return max(self.total - self.s, 0.0)

    def waypoints(self, lookaheads) -> list:
        """Points `lookahead` metres further along the route, clamped to its
        end. Walking a polyline the tracker already holds, rather than
        re-solving the graph for each one."""
        out = []
        last = (float(self.poly[-1][0]), float(self.poly[-1][1]))
        for lookahead in lookaheads:
            target = self.s + float(lookahead)
            if target >= self.total:
                out.append(last)
                continue
            j = int(np.searchsorted(self.cum, target, side="right") - 1)
            j = min(max(j, 0), len(self.seg) - 1)
            frac = (target - self.cum[j]) / max(self.seg_len[j], 1e-9)
            point = self.poly[j] + self.seg[j] * frac
            out.append((float(point[0]), float(point[1])))
        return out

    def heading_at(self, lookahead: float = 0.0) -> float:
        """Direction the route runs at the vehicle's position (or ahead of
        it) — what a waypoint should be offset perpendicular to."""
        target = min(self.s + lookahead, self.total)
        j = int(np.searchsorted(self.cum, target, side="right") - 1)
        j = min(max(j, 0), len(self.seg) - 1)
        return math.atan2(self.seg[j][1], self.seg[j][0])
