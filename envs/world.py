"""World — a road network plus the scenery standing beside it.

This is the thing an episode happens in, and the single place the rest of the
code asks about geometry. It owns three jobs that would otherwise be spread
across the env, the sensors and the renderer, and would disagree the first
time one of them was changed:

1. **Heading convention.** `road_network.py` is a verbatim port from the
   earlier Panda3D project and speaks that engine's H angle — degrees, zero
   along +Y, increasing anticlockwise. `vehicle.Sedan` speaks the ordinary
   maths convention — radians, zero along +X. Every crossing between the two
   goes through `h_to_rad` / `rad_to_h` here and nowhere else.

2. **Static collision geometry**, as the two arrays the sensors want:
   `static_boxes` (N, 5) and `static_circles` (M, 3).

3. **Overlap tests** — is this vehicle's footprint touching a building, a
   tree, or another vehicle. Separating-axis for rectangles, closest-point
   for circles, both vectorised over all candidates at once.
"""

from __future__ import annotations

import math

import numpy as np

from envs import road_network, scenery
from envs.lanes import LaneGraph


def h_to_rad(h_deg: float) -> float:
    """Panda H degrees -> maths radians (0 = +X)."""
    return math.radians(float(h_deg)) + math.pi / 2.0


def rad_to_h(rad: float) -> float:
    """Maths radians -> Panda H degrees."""
    return math.degrees(float(rad) - math.pi / 2.0)


def wrap_pi(a):
    """Angle(s) folded into [-pi, pi]. Scalar in, scalar out."""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


class World:
    """One built scenario: `net` (geometry + routing) and `scenery` (static
    objects). Rebuilt per scenario, not per episode — the road graph and the
    buildings are the environment, and re-rolling them every reset would mean
    the policy never sees the same junction twice."""

    def __init__(self, net, objects: scenery.Scenery, spec: dict):
        self.net = net
        self.scenery = objects
        self.spec = dict(spec)
        # Derived, not stored: the lane count on each piece already implies
        # the lanes, so a network from any source gets them without the
        # builders (or the PNG loader) knowing lanes exist.
        self.lanes = LaneGraph(net)

        # Buildings as (N, 5) for the raycast: cx, cy, hl, hw, yaw.
        self.static_boxes = objects.boxes[:, :5].astype(np.float32).copy()
        # Trees as (M, 3): cx, cy, r.
        self.static_circles = objects.trees[:, :3].astype(np.float32).copy()

    # -- construction -----------------------------------------------------

    @classmethod
    def build(cls, spec="cross", rng: np.random.Generator | None = None,
              scenery_density: float = 1.0, bays: bool = True) -> "World":
        """Build from a road spec (a kind name like `"roundabout_yield"`, or a
        full spec dict). Scenery placement is seeded, so the same spec and the
        same seed give the same town every time.

        `bays=True` attaches parking bays and makes them the episode
        endpoints, so every scenario runs the same bay-to-bay task. Bays go on
        BEFORE the scenery, so buildings and trees treat them as road and
        keep clear. `parking_lot` already is bays, and is left alone.
        """
        if rng is None:
            rng = np.random.default_rng(0)
        net = road_network.build(spec)
        if bays and net.spec.get("kind") != "parking_lot":
            from envs import bays as bays_mod
            net = bays_mod.attach(net, rng)
        objects = scenery.generate(net, rng, density=scenery_density)
        full = dict(net.spec)
        full["scenery_density"] = scenery_density
        return cls(net, objects, full)

    def __repr__(self) -> str:
        return (f"World(kind={self.spec.get('kind')!r}, "
                f"pieces={len(self.net.pieces)}, {self.scenery!r})")

    # -- episode endpoints ------------------------------------------------

    def sample_start(self, rng: np.random.Generator,
                     declared: bool = True) -> tuple[float, float, float]:
        """(x, y, heading_rad) for an episode start.

        `declared=True` prefers the scenario's own source points — "enter this
        roundabout from the southern approach" rather than "start somewhere on
        a roundabout", which is the whole difference between a scenario and a
        layout. A scenario declares a handful of them, though, and a run with
        20-30 vehicles needs more starts than that, so callers filling the
        rest of the traffic pass `declared=False` and get a pose anywhere on
        the network, in the right-hand lane, facing the way it runs.
        """
        if declared:
            x, y, h = self.net.sample_source(rng)
        else:
            x, y, h = self.net.sample_pose(rng)
        return float(x), float(y), h_to_rad(h)

    def sample_goal(self, rng: np.random.Generator, start_xy,
                    min_route: float = 40.0):
        """(x, y, route distance) for a destination at least `min_route`
        metres away ALONG THE ROADS."""
        return self.net.sample_goal(rng, start_xy, min_route)

    # -- overlap tests ----------------------------------------------------

    @staticmethod
    def _corners(rect: np.ndarray) -> np.ndarray:
        """(..., 4, 2) corners of (..., 5) cx, cy, hl, hw, yaw rectangles."""
        rect = np.asarray(rect, dtype=float).reshape(-1, 5)
        cx, cy, hl, hw, yaw = rect.T
        cos, sin = np.cos(yaw), np.sin(yaw)
        sx = np.array([1.0, 1.0, -1.0, -1.0])
        sy = np.array([1.0, -1.0, -1.0, 1.0])
        lx = hl[:, None] * sx[None, :]
        ly = hw[:, None] * sy[None, :]
        return np.stack([cx[:, None] + cos[:, None] * lx - sin[:, None] * ly,
                         cy[:, None] + sin[:, None] * lx + cos[:, None] * ly],
                        axis=2)

    def rect_hits_rects(self, rect, others) -> np.ndarray:
        """Separating-axis overlap of one rectangle against (N, 5) others.

        Two convex polygons miss each other exactly when some edge normal of
        one separates them, so four axes (two per rectangle, the other two
        being parallel) settle it. Returns a boolean per row of `others`.
        """
        others = np.asarray(others, dtype=float).reshape(-1, 5)
        out = np.zeros(len(others), dtype=bool)
        if len(others) == 0:
            return out

        # Broad phase. SAT costs O(candidates^2) in memory because every
        # candidate contributes two axes to the shared axis set, so against a
        # town's worth of buildings it is the whole cost of the step. A
        # bounding-circle test throws out all but a handful first.
        rect_arr = np.asarray(rect, dtype=float).reshape(5)
        ra = math.hypot(rect_arr[2], rect_arr[3])
        rb = np.hypot(others[:, 2], others[:, 3])
        near = np.hypot(others[:, 0] - rect_arr[0],
                        others[:, 1] - rect_arr[1]) <= ra + rb
        if not near.any():
            return out
        idx_near = np.flatnonzero(near)
        others = others[near]

        a = self._corners(rect)[0]                       # (4, 2)
        b = self._corners(others)                        # (N, 4, 2)

        yaw_a = float(rect_arr[4])
        yaws = np.concatenate([[yaw_a, yaw_a + np.pi / 2.0],
                               others[:, 4], others[:, 4] + np.pi / 2.0])
        axes = np.stack([np.cos(yaws), np.sin(yaws)], axis=1)   # (2N+2, 2)

        pa = a @ axes.T                                  # (4, K)
        pb = np.einsum("nij,kj->nik", b, axes)           # (N, 4, K)
        a_lo, a_hi = pa.min(axis=0), pa.max(axis=0)      # (K,)
        b_lo, b_hi = pb.min(axis=1), pb.max(axis=1)      # (N, K)

        # A shared axis set is used for every pair, which only over-tests: an
        # axis belonging to a different box can never report a false overlap,
        # it can only fail to separate, and the pair's own two axes are always
        # in the set.
        own = np.zeros((len(others), len(axes)), dtype=bool)
        own[:, :2] = True
        idx = np.arange(len(others))
        own[idx, 2 + idx] = True
        own[idx, 2 + len(others) + idx] = True

        gap = (a_hi[None, :] < b_lo) | (b_hi < a_lo[None, :])
        out[idx_near] = ~(gap & own).any(axis=1)
        return out

    def rect_hits_circles(self, rect, circles) -> np.ndarray:
        """Overlap of one rectangle against (M, 3) cx, cy, r circles, by
        clamping each centre into the rectangle's own frame."""
        circles = np.asarray(circles, dtype=float).reshape(-1, 3)
        if len(circles) == 0:
            return np.zeros(0, dtype=bool)
        cx, cy, hl, hw, yaw = np.asarray(rect, dtype=float).reshape(5)
        cos, sin = math.cos(-yaw), math.sin(-yaw)
        dx = circles[:, 0] - cx
        dy = circles[:, 1] - cy
        lx = cos * dx - sin * dy
        ly = sin * dx + cos * dy
        qx = np.clip(lx, -hl, hl)
        qy = np.clip(ly, -hw, hw)
        return np.hypot(lx - qx, ly - qy) <= circles[:, 2]

    def hits_scenery(self, rect) -> bool:
        """True when a vehicle footprint is inside a building or a tree."""
        return bool(self.rect_hits_rects(rect, self.static_boxes).any()
                    or self.rect_hits_circles(rect, self.static_circles).any())

    # -- road queries (thin pass-throughs, so callers hold one object) -----

    def road_state(self, x: float, y: float):
        """(margin to road edge, signed lateral offset, centreline tangent
        in radians) — one projection, three answers."""
        return self.net.road_state(float(x), float(y))

    def route_probe(self, a_xy, b_xy, lookaheads):
        return self.net.route_probe(a_xy, b_xy, lookaheads)

    def locate_lane(self, x: float, y: float, heading: float):
        """Which lane this pose is in and whether it faces the right way."""
        return self.lanes.locate(float(x), float(y), float(heading))

    def bounds(self):
        """Network bounds widened to cover the scenery that was placed around
        it — what a renderer frames and what a stray vehicle leaves."""
        x0, y0, x1, y1 = self.net.bounds()
        pts = []
        if len(self.scenery.boxes):
            r = np.hypot(self.scenery.boxes[:, 2], self.scenery.boxes[:, 3])
            pts.append(np.stack([self.scenery.boxes[:, 0] - r,
                                 self.scenery.boxes[:, 1] - r], axis=1))
            pts.append(np.stack([self.scenery.boxes[:, 0] + r,
                                 self.scenery.boxes[:, 1] + r], axis=1))
        if len(self.scenery.trees):
            r = self.scenery.trees[:, 2]
            pts.append(np.stack([self.scenery.trees[:, 0] - r,
                                 self.scenery.trees[:, 1] - r], axis=1))
            pts.append(np.stack([self.scenery.trees[:, 0] + r,
                                 self.scenery.trees[:, 1] + r], axis=1))
        if pts:
            arr = np.concatenate(pts, axis=0)
            x0 = min(x0, float(arr[:, 0].min()))
            y0 = min(y0, float(arr[:, 1].min()))
            x1 = max(x1, float(arr[:, 0].max()))
            y1 = max(y1, float(arr[:, 1].max()))
        return x0, y0, x1, y1
