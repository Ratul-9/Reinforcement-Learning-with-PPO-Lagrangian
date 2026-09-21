"""Road networks — the drivable graph the Environment tab authors and
`rl/road_nav_env.py` trains on. Qt-free and render-free: this module is pure
geometry + graph queries, exactly like sim/world/sim_world.py is pure
physics. The mesh that draws a network lives in sim/visuals/roads.py, and the
UI that edits one in gui/tabs/environment/road_config.py.

A network is a **graph of centrelines**, not a mesh:

    nodes   junction points (a plain end, a 3-way tee, a 4-way cross, or a
            roundabout ring node), each with a `pad` radius that makes the
            junction itself drivable
    pieces  centreline segments between two nodes — a straight or a circular
            arc — each carrying the road's half-width

Everything the simulator and the RL task need is a query against that graph,
and phrasing them against centrelines rather than against a triangle soup is
what makes them cheap enough to run at 20 Hz inside a training loop:

* `is_on_road(x, y)` — |lateral offset| <= half_width, or inside a junction
  pad. This is what the off-road penalty is measured with. Note it is a
  *geometric* test, not a physics one: roads are painted flush onto the ground
  and have no collision bodies of their own, so nothing about driving off one
  changes the vehicle dynamics. The reward is the only thing that notices.
* `route_distance(a, b)` — distance ALONG THE ROADS, not through the scenery.
  This is the one that matters most for learning. With a Euclidean distance
  the progress term pays a vehicle for cutting the corner of an intersection
  (or driving straight across the middle of a roundabout), which is precisely
  the behaviour the off-road penalty is trying to stop — the two terms would
  be fighting each other. Route distance makes "shorter" and "on the road"
  agree.
* `route_guidance(a, b, lookahead)` — a waypoint some metres ahead along that
  same route. A goal 300 m away round two corners is invisible to a 30 m
  lidar and encoded only as a straight-line bearing through a building, so the
  policy would have no way to know which arm of an intersection to take. The
  waypoint is what turns "get to the goal" into a sequence of locally
  answerable questions.

Shortest paths are precomputed once per network with Floyd-Warshall (node
counts here are tens, not thousands, so the O(n^3) is microseconds at build
time and every later query is two projections plus four table lookups).

Coordinates are metres in the same world frame as everything else (1 Panda3D
unit = 1 m), and headings are Panda's: H = 0 faces +Y, and the forward vector
for a heading H is (-sin H, cos H).
"""

from __future__ import annotations

import math

import numpy as np

# ── Defaults every builder shares ────────────────────────────────────────
DEFAULT_LANES = 2               # total lanes across the carriageway
DEFAULT_LANE_WIDTH = 3.5        # metres per lane — standard urban lane
# A junction's drivable disc, as a multiple of the road's half-width. Without
# it the corner a turning vehicle cuts falls outside both crossing corridors
# and reads as "off the road" for a step or two mid-turn, which would punish
# the one manoeuvre an intersection exists to teach.
JUNCTION_PAD = 1.45
DEFAULT_SPEED_LIMIT = 13.9      # m/s, 50 km/h — an urban street
_ARC_STEP_DEG = 5.0             # arc polyline resolution (also the mesh's)
_EPS = 1e-9

KINDS = (
    # Generic layouts — the original set, still what the Environment tab
    # opens on and what a quick "can it drive at all" test wants.
    "cross", "tee", "grid", "roundabout", "loop", "town",
    # Scenario layouts — the eight collision-prone environments the
    # constrained-RL work trains on, in curriculum order (see
    # agent-files/paper-idea.md). Each one declares its own source and
    # destination points and mixes carriageway widths across its pieces.
    "parking_lot", "intersection_x", "merge_ramp", "roundabout_yield",
    "manhattan", "left_turn", "lane_closure", "weave_roundabout",
)

# The eight in order, for anything that wants to iterate the curriculum
# (the render check, a future training sweep) without hard-coding the names
# in two places.
SCENARIO_KINDS = KINDS[6:]


# ── Pieces ───────────────────────────────────────────────────────────────

class _Piece:
    """One centreline between two nodes. Subclasses implement the geometry;
    everything else in this module only ever talks to this interface.

    A piece carries its OWN `lanes`/`half_width` rather than reading the
    network's. A real junction is where roads of different sizes meet — a
    two-lane side street joining a three-lane arterial, a one-lane parking bay
    off a two-lane aisle — and a single network-wide width cannot express that.
    Everything width-dependent (the on-road test, the lane-centre spawn offset,
    the mesh ribbon, the junction pad radius) therefore reads the width of the
    piece it is standing on, not the network's nominal one.
    """

    __slots__ = ("node_a", "node_b", "length", "lanes", "half_width", "_poly",
                 "oneway", "z0", "z1", "speed_limit")

    def __init__(self, node_a: int, node_b: int):
        self.node_a = node_a
        self.node_b = node_b
        self.length = 0.0
        self.lanes = DEFAULT_LANES
        self.half_width = DEFAULT_LANES * DEFAULT_LANE_WIDTH / 2.0
        self._poly: np.ndarray = np.zeros((0, 2))
        # A ONE-WAY piece carries all of its lanes in the +tangent
        # direction. A two-way piece splits them by side of the centreline.
        # Dual carriageways, slip roads and gyratories are one-way; without
        # this a three-lane motorway carriageway would be modelled as one
        # and a half lanes each way, which is not a road.
        self.oneway = False
        # Elevation at each end, linear in between. Flat everywhere except a
        # grade separation, where it is the whole point: a bridge deck and
        # the road beneath it occupy the same x/y and must not see or
        # collide with each other.
        self.z0 = 0.0
        self.z1 = 0.0
        # Per road, not per world. One global limit made a motorway a 50
        # km/h road, so the `speeding` cost charged correct motorway driving
        # and the only way to satisfy it was to crawl.
        self.speed_limit = DEFAULT_SPEED_LIMIT

    # -- geometry ---------------------------------------------------------

    def point(self, s: float) -> tuple[float, float]:
        raise NotImplementedError

    def tangent(self, s: float) -> tuple[float, float]:
        raise NotImplementedError

    def closest(self, x: float, y: float) -> tuple[float, float, float]:
        """(s along the piece, SIGNED lateral offset, distance to the piece).

        The sign of the lateral offset is the 2D cross product of the tangent
        with the offset vector — positive when the point is to the LEFT of the
        direction of travel — so a lane-keeping term can tell which side of the
        centreline it is on, not merely how far off it is. `distance` is the
        true distance to the segment (clamped at the ends), which is what the
        on-road test needs; |lateral| would wrongly report a point beyond the
        end of a piece as being right next to it."""
        raise NotImplementedError

    def elevation(self, s: float) -> float:
        """Height above datum at `s`, linear between the ends."""
        if self.z0 == self.z1:
            return self.z0
        t = min(max(s / max(self.length, _EPS), 0.0), 1.0)
        return self.z0 + (self.z1 - self.z0) * t

    def grade(self) -> float:
        """Slope as a fraction — 0.05 is a 5% gradient."""
        return (self.z1 - self.z0) / max(self.length, _EPS)

    # -- cached polyline --------------------------------------------------

    def _build_poly(self, step: float) -> None:
        n = max(2, int(math.ceil(self.length / max(step, 0.5))) + 1)
        self._poly = np.array([self.point(self.length * i / (n - 1))
                               for i in range(n)], dtype=float)

    def polyline(self, s0: float, s1: float) -> np.ndarray:
        """The cached polyline clipped to [s0, s1] (either order), with exact
        endpoints. Used to build route polylines without re-evaluating any
        trigonometry per step."""
        a, b = (s0, s1) if s0 <= s1 else (s1, s0)
        n = len(self._poly)
        if n < 2 or self.length <= _EPS:
            return np.array([self.point(s0), self.point(s1)], dtype=float)
        span = self.length / (n - 1)
        i0 = int(math.ceil(a / span - 1e-6))
        i1 = int(math.floor(b / span + 1e-6))
        mid = self._poly[max(i0, 0):min(i1 + 1, n)]
        pts = [self.point(a)]
        pts.extend(mid.tolist())
        pts.append(self.point(b))
        out = np.array(pts, dtype=float)
        return out[::-1] if s0 > s1 else out


class _Straight(_Piece):
    __slots__ = ("x0", "y0", "ux", "uy")

    def __init__(self, node_a: int, node_b: int, p0, p1):
        super().__init__(node_a, node_b)
        self.x0, self.y0 = float(p0[0]), float(p0[1])
        dx, dy = float(p1[0]) - self.x0, float(p1[1]) - self.y0
        self.length = math.hypot(dx, dy)
        inv = 1.0 / max(self.length, _EPS)
        self.ux, self.uy = dx * inv, dy * inv
        self._build_poly(1e9)   # a straight needs only its two endpoints

    def point(self, s):
        return (self.x0 + self.ux * s, self.y0 + self.uy * s)

    def tangent(self, s):
        return (self.ux, self.uy)

    def closest(self, x, y):
        vx, vy = x - self.x0, y - self.y0
        s = vx * self.ux + vy * self.uy
        lateral = self.ux * vy - self.uy * vx
        if s < 0.0:
            return 0.0, lateral, math.hypot(vx, vy)
        if s > self.length:
            ex, ey = x - (self.x0 + self.ux * self.length), y - (self.y0 + self.uy * self.length)
            return self.length, lateral, math.hypot(ex, ey)
        return s, lateral, abs(lateral)


class _Arc(_Piece):
    """A circular arc from angle `a0`, sweeping `sweep` radians (signed —
    negative is clockwise) about (cx, cy) at radius r."""

    __slots__ = ("cx", "cy", "r", "a0", "sweep", "_dir")

    def __init__(self, node_a: int, node_b: int, center, r: float,
                 a0: float, sweep: float):
        super().__init__(node_a, node_b)
        self.cx, self.cy = float(center[0]), float(center[1])
        self.r = float(r)
        self.a0 = float(a0)
        self.sweep = float(sweep)
        self._dir = 1.0 if sweep >= 0.0 else -1.0
        self.length = abs(sweep) * self.r
        self._build_poly(self.r * math.radians(_ARC_STEP_DEG))

    def _angle_at(self, s: float) -> float:
        return self.a0 + self._dir * (s / max(self.r, _EPS))

    def point(self, s):
        a = self._angle_at(s)
        return (self.cx + self.r * math.cos(a), self.cy + self.r * math.sin(a))

    def tangent(self, s):
        a = self._angle_at(s)
        # d/ds of the point above: the unit tangent, flipped for a clockwise arc.
        return (-math.sin(a) * self._dir, math.cos(a) * self._dir)

    def closest(self, x, y):
        vx, vy = x - self.cx, y - self.cy
        rho = math.hypot(vx, vy)
        a = math.atan2(vy, vx)
        # How far round the arc the point's bearing is, measured in the arc's
        # own sweep direction and wrapped into [0, 2pi) so the comparison
        # against the sweep is a plain range test.
        delta = (a - self.a0) * self._dir
        delta = delta % (2.0 * math.pi)
        span = abs(self.sweep)
        if delta <= span:
            s = delta * self.r
            # Outward radial offset, re-signed into "left of travel": on a
            # counter-clockwise arc the centre is to the left, so a point
            # further out than the radius is to the RIGHT.
            lateral = -(rho - self.r) * self._dir
            return s, lateral, abs(rho - self.r)
        # Past an end: the nearer endpoint decides, and the lateral sign is
        # taken from the tangent there so it stays meaningful for a vehicle
        # that has overshot slightly.
        s = 0.0 if delta > (span + 2.0 * math.pi) / 2.0 else self.length
        px, py = self.point(s)
        tx, ty = self.tangent(s)
        ox, oy = x - px, y - py
        return s, tx * oy - ty * ox, math.hypot(ox, oy)


# ── Network ──────────────────────────────────────────────────────────────

class RoadNetwork:
    """A built network: nodes, pieces, and the queries the sim/RL side needs.

    Built by `build(spec)` rather than assembled by hand — the builders below
    are what guarantee the graph is connected and that every junction's pad is
    sized to its roads.
    """

    def __init__(self, half_width: float, lanes: int, lane_width: float,
                 spec: dict | None = None):
        self.half_width = float(half_width)
        self.lanes = int(lanes)
        self.lane_width = float(lane_width)
        self.spec = dict(spec or {})
        self.nodes: list[tuple[float, float]] = []
        self.node_kinds: list[str] = []
        self.node_pads: list[float] = []
        self._pad_auto: list[bool] = []
        self.pieces: list[_Piece] = []
        # The widest piece in the network. `half_width` above stays the
        # NOMINAL width (lanes x lane_width / 2) because that is what the spec
        # says and what normalisation constants are scaled by; this is the one
        # the ground has to be big enough for.
        self.max_half_width = float(half_width)
        # Episode endpoints, when the layout defines them rather than leaving
        # them to be sampled anywhere on the network. A scenario like "leave
        # this parking bay and reach that one" or "merge from this ramp and
        # exit there" is only that scenario if the endpoints are where the
        # scenario says. Each source is a full pose (x, y, heading_deg) since
        # which way the vehicle faces at a ramp entry or in a parking bay is
        # part of the problem; each goal is a point.
        self.sources: list[tuple[float, float, float]] = []
        self.goals: list[tuple[float, float]] = []
        # Roundabout centres, as (cx, cy, inner_radius) — the non-drivable
        # island in the middle. Kept separately because it is the one part of a
        # network that is deliberately NOT road, and both the mesh and the
        # placement sampler have to know about it.
        self.islands: list[tuple[float, float, float]] = []
        self._dist: np.ndarray = np.zeros((0, 0))
        self._next: np.ndarray = np.zeros((0, 0), dtype=int)
        self._link: dict[tuple[int, int], int] = {}
        self._cum_len: np.ndarray = np.zeros(0)
        # Broad-phase index for `_project_full`; built in `finalize`.
        self._bb: np.ndarray = np.zeros((0, 4))
        self._bb_hw: np.ndarray = np.zeros(0)

    # -- construction -----------------------------------------------------

    def add_node(self, x: float, y: float, kind: str = "end",
                 pad: float | None = None) -> int:
        """A junction. `pad` defaults to JUNCTION_PAD x the half-width of the
        WIDEST road meeting it (resolved in `finalize`, once the pieces exist)
        and to ZERO for a plain dead end — an end node has only one road
        meeting it, so there is no corner to cut and a pad there would both
        draw a bulb hanging off the end of the road and quietly extend the
        drivable area past where the road stops."""
        self.nodes.append((float(x), float(y)))
        self.node_kinds.append(kind)
        self.node_pads.append(0.0 if pad is None else float(pad))
        self._pad_auto.append(pad is None and kind != "end")
        return len(self.nodes) - 1

    def _size(self, piece: _Piece, lanes: int | None, oneway: bool = False,
              z=(0.0, 0.0), speed_limit: float | None = None) -> _Piece:
        piece.lanes = self.lanes if lanes is None else max(1, int(lanes))
        piece.half_width = piece.lanes * self.lane_width / 2.0
        piece.oneway = bool(oneway)
        piece.z0, piece.z1 = float(z[0]), float(z[1])
        if speed_limit is not None:
            piece.speed_limit = float(speed_limit)
        self.max_half_width = max(self.max_half_width, piece.half_width)
        return piece

    def add_straight(self, a: int, b: int, lanes: int | None = None,
                     oneway: bool = False, z=(0.0, 0.0),
                     speed_limit: float | None = None) -> None:
        self.pieces.append(self._size(
            _Straight(a, b, self.nodes[a], self.nodes[b]), lanes, oneway, z,
            speed_limit))

    def add_arc(self, a: int, b: int, center, radius: float,
                a0: float, sweep: float, lanes: int | None = None,
                oneway: bool = False, z=(0.0, 0.0),
                speed_limit: float | None = None) -> None:
        self.pieces.append(self._size(
            _Arc(a, b, center, radius, a0, sweep), lanes, oneway, z,
            speed_limit))

    def add_source(self, x: float, y: float, heading_deg: float) -> None:
        self.sources.append((float(x), float(y), float(heading_deg)))

    def add_goal(self, x: float, y: float) -> None:
        self.goals.append((float(x), float(y)))

    def finalize(self) -> "RoadNetwork":
        """Precompute everything a query needs: the all-pairs shortest path
        table over the junction graph, the piece used for each hop, and the
        cumulative piece lengths the uniform-by-length sampler draws from."""
        # Every piece must actually START at node_a and END at node_b. A
        # builder that gets an arc's centre or sweep wrong still produces a
        # plausible-looking network, but its graph distances quietly stop
        # describing its geometry — routes come out shorter than the straight
        # line and the progress reward starts paying for the impossible. Caught
        # here, at build time, where the message can name the piece.
        for idx, piece in enumerate(self.pieces):
            for s, node in ((0.0, piece.node_a), (piece.length, piece.node_b)):
                px, py = piece.point(s)
                nx, ny = self.nodes[node]
                if math.hypot(px - nx, py - ny) > 1e-6:
                    raise ValueError(
                        f"piece {idx} ({type(piece).__name__}) does not meet node "
                        f"{node} at s={s:g}: piece is at ({px:.3f}, {py:.3f}), "
                        f"node is at ({nx:.3f}, {ny:.3f})")

        # Auto pads, now that the pieces exist: a junction's disc has to cover
        # the corner cut by the WIDEST road meeting it, or a lorry turning out
        # of a three-lane arterial into a two-lane side street clips off the
        # pad mid-turn and reads as off-road for a step.
        widest: dict[int, float] = {}
        for piece in self.pieces:
            for node in (piece.node_a, piece.node_b):
                widest[node] = max(widest.get(node, 0.0), piece.half_width)
        for i, auto in enumerate(self._pad_auto):
            if auto:
                self.node_pads[i] = widest.get(i, self.half_width) * JUNCTION_PAD

        n = len(self.nodes)
        inf = float("inf")
        dist = np.full((n, n), inf)
        nxt = np.full((n, n), -1, dtype=int)
        np.fill_diagonal(dist, 0.0)
        for i in range(n):
            nxt[i, i] = i

        for idx, piece in enumerate(self.pieces):
            a, b = piece.node_a, piece.node_b
            if piece.length < dist[a, b]:
                dist[a, b] = dist[b, a] = piece.length
                nxt[a, b], nxt[b, a] = b, a
                self._link[(a, b)] = idx
                self._link[(b, a)] = idx

        for k in range(n):
            # Vectorised Floyd-Warshall inner double loop: for every (i, j),
            # is going through k shorter than what we have?
            through = dist[:, k, None] + dist[None, k, :]
            better = through < dist
            if better.any():
                dist = np.where(better, through, dist)
                nxt = np.where(better, nxt[:, k, None], nxt)

        self._dist = dist
        self._next = nxt
        lengths = np.array([p.length for p in self.pieces], dtype=float)
        self._cum_len = np.cumsum(lengths) if len(lengths) else np.zeros(0)
        self._build_index()
        return self

    # -- basic geometry ---------------------------------------------------

    @property
    def total_length(self) -> float:
        return float(self._cum_len[-1]) if len(self._cum_len) else 0.0

    def bounds(self) -> tuple[float, float, float, float]:
        """(min_x, min_y, max_x, max_y) of the drivable surface, half-width
        included — what the ground has to cover for the network to sit on it."""
        if not self.nodes:
            return (0.0, 0.0, 0.0, 0.0)
        pts = [p for piece in self.pieces for p in piece._poly] or list(self.nodes)
        arr = np.asarray(pts, dtype=float)
        m = self.max_half_width
        return (float(arr[:, 0].min()) - m, float(arr[:, 1].min()) - m,
                float(arr[:, 0].max()) + m, float(arr[:, 1].max()) + m)

    def extent(self) -> float:
        """Half-side of the smallest origin-centred square containing the
        network — the number a square ground's `radius` wants."""
        x0, y0, x1, y1 = self.bounds()
        return max(abs(x0), abs(y0), abs(x1), abs(y1))

    # -- projection / on-road ---------------------------------------------

    def project(self, x: float, y: float) -> tuple[int, float, float, float]:
        """Nearest point on the network: (piece index, s, signed lateral,
        distance). Distance is to the centreline, so subtract that piece's
        `half_width` for the distance to the road's edge."""
        return self._project_full(x, y)[:4]

    def _build_index(self) -> None:
        """Bounding box per piece, for the broad phase in `_project_full`.

        `_project_full` is the single hottest call in a training run — a
        profile of one step of twenty agents shows it walking every piece of
        the network six times per agent, which on a network with bays is
        over a million `closest` calls per 150 steps. The boxes let almost
        all of those be skipped without changing the answer.

        Padded by the arc sampling sagitta, because a piece's cached
        polyline is a chord approximation that sits INSIDE the true arc:
        an unpadded box could exclude the very piece the point is nearest.
        """
        if not self.pieces:
            self._bb = np.zeros((0, 4))
            self._bb_hw = np.zeros(0)
            return
        pad = []
        boxes = []
        for piece in self.pieces:
            poly = piece._poly if len(piece._poly) else np.array(
                [piece.point(0.0), piece.point(piece.length)])
            boxes.append([poly[:, 0].min(), poly[:, 1].min(),
                          poly[:, 0].max(), poly[:, 1].max()])
            if isinstance(piece, _Arc):
                half = math.radians(_ARC_STEP_DEG) / 2.0
                pad.append(piece.r * (1.0 - math.cos(half)))
            else:
                pad.append(0.0)
        pad = np.asarray(pad)[:, None]
        self._bb = np.asarray(boxes, dtype=float) + np.hstack([-pad, -pad, pad, pad])
        self._bb_hw = np.array([p.half_width for p in self.pieces], dtype=float)

    def _project_full(self, x: float, y: float):
        """(piece index, s, signed lateral, distance, margin), where the piece
        chosen is the one whose EDGE is nearest — not whose centreline is.

        The distinction only appears once pieces have different widths, and
        then it matters: standing in the outside lane of a three-lane arterial,
        the centreline of a one-lane parking stub on the verge can easily be
        the closer of the two, and a nearest-centreline choice would answer
        "off road" for a vehicle sitting squarely on the arterial. Ranking by
        margin asks the question the callers actually mean — which road am I
        on — and reduces to the old behaviour when every piece is the same
        width.

        Broad phase first. The distance from the point to a piece's bounding
        box is a LOWER bound on its distance to that piece, so
        `half_width - box_distance` is an UPPER bound on the margin the piece
        could offer. Visiting pieces in descending order of that bound and
        stopping once it falls below the best margin actually measured is
        exact — it cannot skip a piece that would have won — while
        evaluating a handful of pieces instead of all of them.
        """
        best = (0, 0.0, 0.0, float("inf"), -float("inf"))
        if self._bb.shape[0] != len(self.pieces):
            self._build_index()

        dx = np.maximum(self._bb[:, 0] - x, 0.0) + np.maximum(x - self._bb[:, 2], 0.0)
        dy = np.maximum(self._bb[:, 1] - y, 0.0) + np.maximum(y - self._bb[:, 3], 0.0)
        bound = self._bb_hw - np.hypot(dx, dy)

        for i in np.argsort(-bound):
            if bound[i] <= best[4]:
                break
            s, lateral, d = self.pieces[i].closest(x, y)
            margin = self.pieces[i].half_width - d
            if margin > best[4]:
                best = (int(i), s, lateral, d, margin)
        return best

    def is_on_road(self, x: float, y: float) -> bool:
        return self.road_margin(x, y) >= 0.0

    def road_margin(self, x: float, y: float) -> float:
        """Metres from the point to the nearest road EDGE, positive when on the
        road. The single quantity the off-road penalty and the road-margin
        observation channel are both read off, so they can never disagree."""
        _, _, _, _, margin = self._project_full(x, y)
        for i, (nx, ny) in enumerate(self.nodes):
            pad = self.node_pads[i]
            m = pad - math.hypot(x - nx, y - ny)
            if m > margin:
                margin = m
        # A roundabout's central island is a hole punched through everything
        # above: the ring arc's own corridor and the ring nodes' pads both
        # cover ground that is not drivable at all.
        for cx, cy, r in self.islands:
            inside = r - math.hypot(x - cx, y - cy)
            if inside > 0.0:
                margin = min(margin, -inside)
        return margin

    def road_state(self, x: float, y: float) -> tuple[float, float, float]:
        """(margin to the road edge, signed lateral offset from the nearest
        centreline, centreline tangent in radians) in ONE projection.

        The env reads all three every step and would otherwise project three
        times per step per vehicle for the same answer — the single most
        repeated call in a training run, so it gets its own entry point rather
        than being composed out of the individual queries above."""
        i, s, lateral, _d, margin = self._project_full(x, y)
        for k, (nx, ny) in enumerate(self.nodes):
            m = self.node_pads[k] - math.hypot(x - nx, y - ny)
            if m > margin:
                margin = m
        for cx, cy, r in self.islands:
            inside = r - math.hypot(x - cx, y - cy)
            if inside > 0.0:
                margin = min(margin, -inside)
        tx, ty = self.pieces[i].tangent(s)
        return margin, lateral, math.atan2(ty, tx)

    def speed_limit_at(self, x: float, y: float) -> float:
        """The limit on the road under a point."""
        i, _s, _lateral, _d = self.project(x, y)
        return self.pieces[i].speed_limit

    def elevation_at(self, x: float, y: float) -> float:
        """Height of the road surface under a point. Zero on a flat map."""
        i, s, _lateral, _d = self.project(x, y)
        return self.pieces[i].elevation(s)

    def heading_at(self, x: float, y: float) -> float:
        """Compass-free tangent of the nearest centreline, in radians as
        atan2(dy, dx). Direction is the piece's own, not the travel
        direction — callers that need "toward the goal" use route_guidance."""
        i, s, _, _ = self.project(x, y)
        tx, ty = self.pieces[i].tangent(s)
        return math.atan2(ty, tx)

    # -- routing ----------------------------------------------------------

    def _endpoints(self, piece_index: int, s: float):
        p = self.pieces[piece_index]
        return ((p.node_a, s), (p.node_b, p.length - s))

    def route_distance(self, a_xy, b_xy) -> float:
        """Shortest distance from a to b travelling only on roads. Falls back
        to the straight line if the two ends land in disconnected components,
        which the shipped builders never produce but a hand-built network
        could."""
        d, _ = self._route(a_xy, b_xy)
        if math.isinf(d):
            return math.hypot(b_xy[0] - a_xy[0], b_xy[1] - a_xy[1])
        return d

    def route_probe(self, a_xy, b_xy, lookaheads):
        """Several route waypoints from ONE shortest-path solve.

        Returns `[(x, y), ...]`, one per entry in `lookaheads` and in the order
        asked for (which need not be sorted), each clamped to the end of the
        route, plus the remaining route distance.

        This exists because `route_guidance` re-solves the route on every call
        and the RL task wants a handful of lookaheads per step — a steering
        target, a lane tangent, and two probes to measure the bend ahead. The
        graph query is the expensive part; walking a polyline it already has is
        nearly free, so this is the difference between one shortest-path lookup
        per control step and four."""
        d, poly = self._route(a_xy, b_xy)
        if poly is None or len(poly) < 2:
            if math.isinf(d):
                d = math.hypot(b_xy[0] - a_xy[0], b_xy[1] - a_xy[1])
            end = (float(b_xy[0]), float(b_xy[1]))
            return [end for _ in lookaheads], d

        seg = np.diff(poly, axis=0)
        step = np.hypot(seg[:, 0], seg[:, 1])
        walked = np.cumsum(step)
        last = (float(poly[-1][0]), float(poly[-1][1]))
        points = []
        for lookahead in lookaheads:
            idx = int(np.searchsorted(walked, lookahead))
            if idx >= len(step):
                points.append(last)
                continue
            before = walked[idx - 1] if idx else 0.0
            frac = (lookahead - before) / max(step[idx], _EPS)
            p = poly[idx] + seg[idx] * frac
            points.append((float(p[0]), float(p[1])))
        return points, d

    def route_guidance(self, a_xy, b_xy, lookahead: float) -> tuple[float, float, float]:
        """(waypoint x, waypoint y, remaining route distance) — the point
        `lookahead` metres along the route from a toward b, or b itself when
        the route is shorter than that."""
        (point,), d = self.route_probe(a_xy, b_xy, (lookahead,))
        return point[0], point[1], d

    def route_polyline(self, a_xy, b_xy) -> np.ndarray:
        """The shortest on-road path from a to b as an (N, 2) polyline. Used by
        the Environment tab to draw the route between a previewed spawn and
        destination — the same path the RL task's waypoints are taken from, so
        what the user sees is what the policy is steered along."""
        _, poly = self._route(a_xy, b_xy)
        if poly is None:
            return np.array([a_xy, b_xy], dtype=float)
        return poly

    def _route(self, a_xy, b_xy):
        """(distance, polyline) for the shortest on-road path. The polyline
        starts at a's projection and ends at b's, so a caller can walk it
        directly without re-projecting."""
        ia, sa, _, _ = self.project(a_xy[0], a_xy[1])
        ib, sb, _, _ = self.project(b_xy[0], b_xy[1])

        best_d = float("inf")
        best = None

        if ia == ib:
            # Same piece: travelling along it beats any detour through the
            # graph, except in the (legal) case of a one-piece loop.
            best_d = abs(sa - sb)
            best = ("direct", None)

        for (na, da) in self._endpoints(ia, sa):
            for (nb, db) in self._endpoints(ib, sb):
                total = da + self._dist[na, nb] + db
                if total < best_d:
                    best_d = total
                    best = (na, nb)

        if best is None or math.isinf(best_d):
            return float("inf"), None
        if best[0] == "direct":
            return best_d, self.pieces[ia].polyline(sa, sb)

        na, nb = best
        parts = [self.pieces[ia].polyline(sa, 0.0 if na == self.pieces[ia].node_a
                                          else self.pieces[ia].length)]
        for u, v in self._hops(na, nb):
            piece = self.pieces[self._link[(u, v)]]
            fwd = piece.node_a == u
            parts.append(piece.polyline(0.0 if fwd else piece.length,
                                        piece.length if fwd else 0.0))
        parts.append(self.pieces[ib].polyline(0.0 if nb == self.pieces[ib].node_a
                                              else self.pieces[ib].length, sb))
        poly = np.vstack([p for p in parts if len(p)])
        return best_d, poly

    def _hops(self, a: int, b: int):
        """The node pairs along the shortest a->b path, from the next-hop
        table Floyd-Warshall filled in."""
        if a == b or self._next[a, b] < 0:
            return
        u = a
        guard = 0
        while u != b and guard <= len(self.nodes):
            v = int(self._next[u, b])
            if v < 0:
                return
            yield u, v
            u = v
            guard += 1

    # -- sampling ---------------------------------------------------------

    def sample_pose(self, rng: np.random.Generator, lane_bias: bool = True):
        """A random pose ON the network: (x, y, heading_deg).

        Uniform BY LENGTH (a 200 m arterial is ten times likelier than a 20 m
        stub), placed in the centre of the right-hand lane for a randomly
        chosen direction of travel, and facing that direction. Spawning in the
        correct lane facing the correct way is deliberate: the task is
        "navigate this road network", not "recover from being dropped
        sideways across a carriageway", and starting every episode in a
        recovery state would spend most of training on the wrong problem."""
        s_global = float(rng.uniform(0.0, self.total_length))
        i = int(np.searchsorted(self._cum_len, s_global, side="right"))
        i = min(i, len(self.pieces) - 1)
        base = self._cum_len[i - 1] if i else 0.0
        piece = self.pieces[i]
        s = min(max(s_global - base, 0.0), piece.length)

        x, y = piece.point(s)
        tx, ty = piece.tangent(s)
        if rng.random() < 0.5:
            tx, ty = -tx, -ty          # travel the other way down this piece
        if lane_bias and piece.lanes > 1:
            # Right of travel is the tangent rotated -90 degrees, half this
            # PIECE's width out — a one-lane parking bay has no right-hand lane
            # to sit in and a three-lane arterial's is further out than a
            # two-lane street's.
            off = piece.half_width / 2.0
            x += ty * off
            y += -tx * off
        return x, y, math.degrees(math.atan2(-tx, ty))

    def sample_source(self, rng: np.random.Generator):
        """Where an episode starts: (x, y, heading_deg).

        A layout that declared source points gets one of those; anything else
        falls back to `sample_pose`, which is what every network did before
        scenarios existed. The distinction is the whole difference between
        "drive around this roundabout" and "enter this roundabout from the
        southern approach", and only the second is a scenario.

        The pose is on the CENTRELINE — callers shift it into the right-hand
        lane of whichever direction the route to their chosen destination
        runs, which is not knowable here."""
        if self.sources:
            return self.sources[int(rng.integers(len(self.sources)))]
        return self.sample_pose(rng, lane_bias=False)

    def sample_goal(self, rng: np.random.Generator, spawn_xy,
                    min_route: float, attempts: int = 64):
        """A goal on the network at least `min_route` metres away ALONG THE
        ROADS. Measuring the separation as a route distance rather than a
        straight line is what stops "20 m away" from meaning "on the parallel
        street you cannot reach without driving 180 m".

        Declared goals are drawn from first, and the farthest one is taken if
        none of them clears `min_route` — a scenario's destinations are the
        destinations even when the vehicle happens to spawn near one."""
        best = None
        best_d = -1.0
        candidates = None
        if self.goals:
            order = rng.permutation(len(self.goals))
            candidates = [self.goals[int(k)] for k in order]
        for k in range(attempts):
            if candidates is not None:
                if k >= len(candidates):
                    break
                gx, gy = candidates[k]
            else:
                gx, gy, _ = self.sample_pose(rng, lane_bias=False)
            d = self.route_distance(spawn_xy, (gx, gy))
            if d >= min_route:
                return gx, gy, d
            if d > best_d:
                best, best_d = (gx, gy), d
        gx, gy = best if best is not None else spawn_xy
        return gx, gy, best_d


# ── Builders ─────────────────────────────────────────────────────────────

def default_spec(kind: str = "cross") -> dict:
    """A complete, sane spec for one network kind — what the UI's type combo
    populates its fields from and what `build` fills any gaps with."""
    base = {"kind": kind, "lanes": DEFAULT_LANES, "lane_width": DEFAULT_LANE_WIDTH}
    extra = {
        "cross":      {"arm": 70.0},
        "tee":        {"arm": 70.0},
        "grid":       {"rows": 3, "cols": 3, "block": 80.0},
        "roundabout": {"radius": 18.0, "arms": 4, "arm": 70.0},
        "loop":       {"radius": 45.0, "straight": 90.0},
        "town":       {"rows": 3, "cols": 3, "block": 90.0, "radius": 18.0},
        # -- scenarios --------------------------------------------------
        # `lanes` is the MAIN carriageway and `minor` the side streets, so
        # every scenario ships with two widths in play by default. Both are
        # ordinary spec fields: a sweep can widen or narrow either without
        # touching the builder.
        "parking_lot":      {"lanes": 3, "aisles": 3, "bays": 6, "block": 60.0,
                             "bay": 6.0, "minor": 2},
        "intersection_x":   {"lanes": 3, "arm": 110.0, "block": 80.0, "minor": 2},
        "merge_ramp":       {"lanes": 3, "arm": 150.0, "ramp": 90.0, "minor": 2},
        "roundabout_yield": {"lanes": 3, "radius": 20.0, "arms": 4, "arm": 80.0,
                             "minor": 2},
        "manhattan":        {"lanes": 3, "rows": 3, "cols": 3, "block": 95.0,
                             "minor": 2},
        "left_turn":        {"lanes": 3, "arm": 130.0, "block": 75.0, "minor": 2},
        "lane_closure":     {"lanes": 3, "arm": 150.0, "closure": 45.0, "minor": 2},
        "weave_roundabout": {"lanes": 3, "radius": 26.0, "arms": 5, "arm": 85.0,
                             "minor": 2},
    }[kind]
    base.update(extra)
    return base


# Kind -> (menu label, [(spec key, caption, min, max, step, decimals), ...]).
#
# Lives here, next to `default_spec`, rather than in either of the two UIs
# that need it (the Environment tab's ROADS section and the studio's road
# editor). It is a property of the layout — which parameters it has and what
# they may be — not of a particular panel, and while it was duplicated in both
# panels, adding a layout meant editing three files and forgetting one of them
# raised a KeyError at startup. `lanes` and `lane_width` are deliberately NOT
# here: every kind has them, so both panels render them as fixed controls.
KIND_FIELDS: dict[str, tuple[str, list]] = {
    "cross":      ("Crossroads (4-way)", [("arm", "Arm", 20.0, 500.0, 5.0, 0)]),
    "tee":        ("T-junction (3-way)", [("arm", "Arm", 20.0, 500.0, 5.0, 0)]),
    "grid":       ("Grid of streets", [("rows", "Rows", 2, 8, 1, 0),
                                       ("cols", "Cols", 2, 8, 1, 0),
                                       ("block", "Block", 30.0, 300.0, 5.0, 0)]),
    "roundabout": ("Roundabout", [("radius", "Radius", 8.0, 80.0, 1.0, 1),
                                  ("arms", "Arms", 3, 8, 1, 0),
                                  ("arm", "Arm", 20.0, 500.0, 5.0, 0)]),
    "loop":       ("Loop circuit", [("radius", "Half-width", 10.0, 300.0, 5.0, 0),
                                    ("straight", "Straight", 10.0, 500.0, 5.0, 0)]),
    "town":       ("Town (grid + roundabout)", [("rows", "Rows", 3, 8, 1, 0),
                                                ("cols", "Cols", 3, 8, 1, 0),
                                                ("block", "Block", 50.0, 300.0, 5.0, 0),
                                                ("radius", "Ring", 8.0, 60.0, 1.0, 1)]),
    # -- the eight scenarios, in curriculum order ------------------------
    "parking_lot": ("1 · Parking lot", [("aisles", "Aisles", 1, 6, 1, 0),
                                        ("bays", "Bays/aisle", 2, 12, 1, 0),
                                        ("block", "Lot size", 30.0, 200.0, 5.0, 0),
                                        ("bay", "Bay depth", 3.0, 12.0, 0.5, 1),
                                        ("minor", "Aisle lanes", 1, 4, 1, 0)]),
    "intersection_x": ("2 · Unsignalised crossings",
                       [("arm", "Arterial", 40.0, 400.0, 5.0, 0),
                        ("block", "Block", 30.0, 200.0, 5.0, 0),
                        ("minor", "Street lanes", 1, 4, 1, 0)]),
    "merge_ramp": ("3 · On-ramp merge", [("arm", "Carriageway", 60.0, 500.0, 10.0, 0),
                                         ("ramp", "Ramp lead-in", 20.0, 300.0, 5.0, 0),
                                         ("minor", "Ramp lanes", 1, 4, 1, 0)]),
    "roundabout_yield": ("4 · Roundabout (yield)",
                         [("radius", "Ring radius", 10.0, 80.0, 1.0, 1),
                          ("arms", "Approaches", 3, 8, 1, 0),
                          ("arm", "Approach", 30.0, 300.0, 5.0, 0),
                          ("minor", "Ring lanes", 1, 4, 1, 0)]),
    "manhattan": ("5 · City grid", [("rows", "Rows", 2, 8, 1, 0),
                                    ("cols", "Cols", 2, 8, 1, 0),
                                    ("block", "Block", 40.0, 250.0, 5.0, 0),
                                    ("minor", "Street lanes", 1, 4, 1, 0)]),
    "left_turn": ("6 · Unprotected left turn",
                  [("arm", "Arterial", 40.0, 400.0, 5.0, 0),
                   ("block", "Block", 30.0, 200.0, 5.0, 0),
                   ("minor", "Side lanes", 1, 4, 1, 0)]),
    "lane_closure": ("7 · Lane closure", [("arm", "Road", 60.0, 500.0, 10.0, 0),
                                          ("closure", "Closure", 10.0, 150.0, 5.0, 0),
                                          ("minor", "Detour lanes", 1, 4, 1, 0)]),
    "weave_roundabout": ("8 · Weaving roundabout",
                         [("radius", "Ring radius", 12.0, 90.0, 1.0, 1),
                          ("arms", "Approaches", 4, 8, 1, 0),
                          ("arm", "Approach", 40.0, 300.0, 5.0, 0),
                          ("minor", "Outer lanes", 1, 4, 1, 0)]),
}


def build(spec: dict | str | None = None) -> RoadNetwork:
    """Build a network from a spec dict (or a bare kind name).

    Dispatches to `envs/scenarios.py` for the seven purpose-built training
    layouts, and to the builders below for the original set.
    """
    from envs import scenarios

    kind = (spec if isinstance(spec, str)
            else (spec or {}).get("kind", "cross"))
    if kind in scenarios.BUILDERS:
        merged = dict(scenarios.DEFAULTS[kind], kind=kind)
        if isinstance(spec, dict):
            merged.update({k: v for k, v in spec.items()})
        net = RoadNetwork(
            half_width=merged["lanes"] * DEFAULT_LANE_WIDTH / 2.0,
            lanes=merged["lanes"], lane_width=DEFAULT_LANE_WIDTH,
            spec=merged)
        scenarios.BUILDERS[kind](net, merged)
        return net.finalize()
    return _build_legacy(spec)


def _build_legacy(spec: dict | str | None = None) -> RoadNetwork:
    """Build a network from a spec dict (or a bare kind name). Unknown keys are
    ignored rather than rejected so an older saved road still loads after a
    builder gains a parameter."""
    if isinstance(spec, str):
        spec = {"kind": spec}
    spec = dict(spec or {})
    kind = spec.get("kind", "cross")
    if kind not in KINDS:
        raise ValueError(f"unknown road network kind {kind!r}. "
                         f"Valid kinds: {', '.join(KINDS)}")
    merged = default_spec(kind)
    merged.update({k: v for k, v in spec.items() if k in merged or k == "kind"})

    lanes = max(1, int(merged["lanes"]))
    lane_width = max(1.5, float(merged["lane_width"]))
    net = RoadNetwork(lanes * lane_width / 2.0, lanes, lane_width, merged)
    _BUILDERS[kind](net, merged)
    return net.finalize()


def _build_cross(net: RoadNetwork, spec: dict) -> None:
    arm = float(spec["arm"])
    c = net.add_node(0.0, 0.0, "cross")
    for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        end = net.add_node(dx * arm, dy * arm, "end")
        net.add_straight(c, end)


def _build_tee(net: RoadNetwork, spec: dict) -> None:
    arm = float(spec["arm"])
    c = net.add_node(0.0, 0.0, "tee")
    for dx, dy in ((0, 1), (1, 0), (-1, 0)):
        end = net.add_node(dx * arm, dy * arm, "end")
        net.add_straight(c, end)


def _grid_nodes(net: RoadNetwork, rows: int, cols: int, block: float,
                skip: tuple[int, int] | None = None) -> dict:
    """Junction nodes for an origin-centred rows x cols lattice, keyed by
    (row, col). `skip` leaves one cell empty — that is how the `town` layout
    makes room for a roundabout where a plain crossing would be."""
    ids: dict[tuple[int, int], int] = {}
    x0 = -(cols - 1) * block / 2.0
    y0 = -(rows - 1) * block / 2.0
    for r in range(rows):
        for c in range(cols):
            if skip is not None and (r, c) == skip:
                continue
            edge = r in (0, rows - 1) or c in (0, cols - 1)
            ids[(r, c)] = net.add_node(x0 + c * block, y0 + r * block,
                                       "tee" if edge else "cross")
    return ids


def _build_grid(net: RoadNetwork, spec: dict) -> None:
    rows, cols = max(2, int(spec["rows"])), max(2, int(spec["cols"]))
    block = float(spec["block"])
    ids = _grid_nodes(net, rows, cols, block)
    for (r, c), node in ids.items():
        for dr, dc in ((0, 1), (1, 0)):
            other = ids.get((r + dr, c + dc))
            if other is not None:
                net.add_straight(node, other)


def _build_roundabout(net: RoadNetwork, spec: dict) -> None:
    radius = max(float(spec["radius"]), net.half_width * 1.6)
    arms = max(3, int(spec["arms"]))
    arm = float(spec["arm"])
    ring: list[int] = []
    for k in range(arms):
        a = 2.0 * math.pi * k / arms
        ring.append(net.add_node(radius * math.cos(a), radius * math.sin(a),
                                 "roundabout", pad=net.half_width))
    for k in range(arms):
        a0 = 2.0 * math.pi * k / arms
        net.add_arc(ring[k], ring[(k + 1) % arms], (0.0, 0.0), radius,
                    a0, 2.0 * math.pi / arms)
        # The approach arm runs radially outward from its ring node.
        a = a0
        end = net.add_node((radius + arm) * math.cos(a), (radius + arm) * math.sin(a), "end")
        net.add_straight(ring[k], end)
    net.islands.append((0.0, 0.0, max(radius - net.half_width, 0.5)))


def _build_loop(net: RoadNetwork, spec: dict) -> None:
    """A stadium circuit: two straights joined by two semicircles. The one
    layout with no junctions at all — useful as a "can it drive at all"
    baseline before an intersection task."""
    r = max(float(spec["radius"]), net.half_width * 2.0)   # half-width of the oval
    half = max(float(spec["straight"]), 1.0) / 2.0         # half-length of each straight
    n_rt_bot = net.add_node(r, -half, "end")
    n_rt_top = net.add_node(r, half, "end")
    n_lf_top = net.add_node(-r, half, "end")
    n_lf_bot = net.add_node(-r, -half, "end")
    net.add_straight(n_rt_bot, n_rt_top)          # right straight, north-bound
    net.add_straight(n_lf_top, n_lf_bot)          # left straight, south-bound
    # Semicircles centred on the ends of the straights, so each arc's own
    # endpoints land exactly on the nodes it connects — if they do not, the
    # graph distances silently stop matching the geometry.
    net.add_arc(n_rt_top, n_lf_top, (0.0, half), r, 0.0, math.pi)
    net.add_arc(n_lf_bot, n_rt_bot, (0.0, -half), r, math.pi, math.pi)


def _build_town(net: RoadNetwork, spec: dict) -> None:
    """A grid with its central junction replaced by a roundabout — the mixed
    layout, so one trained policy has to handle plain crossings AND a
    roundabout AND the tee junctions round the edge of the grid."""
    rows, cols = max(3, int(spec["rows"])), max(3, int(spec["cols"]))
    block = float(spec["block"])
    radius = max(float(spec["radius"]), net.half_width * 1.6)
    # The ring has to fit inside the block it replaces, or the connecting stubs
    # would run backwards through their own junction.
    radius = max(min(radius, block / 2.0 - net.half_width), net.half_width * 1.2)

    mid = (rows // 2, cols // 2)
    ids = _grid_nodes(net, rows, cols, block, skip=mid)
    cx = -(cols - 1) * block / 2.0 + mid[1] * block
    cy = -(rows - 1) * block / 2.0 + mid[0] * block

    for (r, c), node in ids.items():
        for dr, dc in ((0, 1), (1, 0)):
            other = ids.get((r + dr, c + dc))
            if other is not None:
                net.add_straight(node, other)

    # Ring nodes at E/N/W/S so each one lines up with the grid street it
    # replaces — that alignment is what lets the connecting stubs be straight.
    ring = []
    for k, ang in enumerate((0.0, math.pi / 2, math.pi, 3 * math.pi / 2)):
        ring.append(net.add_node(cx + radius * math.cos(ang), cy + radius * math.sin(ang),
                                 "roundabout", pad=net.half_width))
    for k, ang in enumerate((0.0, math.pi / 2, math.pi, 3 * math.pi / 2)):
        net.add_arc(ring[k], ring[(k + 1) % 4], (cx, cy), radius, ang, math.pi / 2)

    neighbours = {0: (mid[0], mid[1] + 1), 1: (mid[0] + 1, mid[1]),
                  2: (mid[0], mid[1] - 1), 3: (mid[0] - 1, mid[1])}
    for k, cell in neighbours.items():
        target = ids.get(cell)
        if target is not None:
            net.add_straight(ring[k], target)
    net.islands.append((cx, cy, max(radius - net.half_width, 0.5)))


# ── Scenario builders ────────────────────────────────────────────────────
#
# The eight collision-prone environments. Three things every one of them
# guarantees, because they are what makes a layout a usable RL scenario
# rather than decoration:
#
#   1. drivable vs non-drivable is unambiguous — the road corridors and
#      junction pads are drivable, the verge and any roundabout island are
#      not, and `road_margin` answers with a signed distance either way;
#   2. at least one declared SOURCE pose, in the correct lane, facing the
#      direction the scenario expects the vehicle to set off in;
#   3. at least one declared DESTINATION point, reachable from every source.
#
# They also deliberately mix carriageway widths: a three-lane arterial
# crossing two-lane streets, one-lane parking bays off a two-lane aisle. The
# widths are per piece (see `_Piece`), so "which road am I on, and where is
# its edge" has a different answer on either side of a junction — which is the
# situation a lane-boundary cost function has to be tested against.

def _minor(spec: dict, net: RoadNetwork) -> int:
    """The side-street lane count: whatever the spec says, but never wider
    than the main carriageway (a 'minor' street wider than the arterial it
    joins would invert every junction's pad)."""
    return max(1, min(int(spec.get("minor", 2)), net.lanes))


def _source_at(net: RoadNetwork, node_xy, toward_xy, lanes: int,
               inset: float = 6.0) -> None:
    """Declare a source a few metres inside the road at `node_xy`, ON THE
    CENTRELINE, facing `toward_xy`.

    Centreline and not lane centre on purpose: every caller (the env's
    `_sample_episode`, the tab's RL preview) already shifts the spawn into the
    right-hand lane of the direction the route runs, and a source stored
    pre-shifted would be shifted twice and start the episode in the oncoming
    lane. The heading is stored because a scenario cares which way the vehicle
    sets off even when a caller recomputes the lane offset.

    The inset exists because a source sitting exactly on an end node puts the
    vehicle's nose past the end of the road: it starts the episode already
    off-road, and the first thing the off-road cost records is an event the
    policy could not have avoided."""
    dx = float(toward_xy[0]) - float(node_xy[0])
    dy = float(toward_xy[1]) - float(node_xy[1])
    d = math.hypot(dx, dy)
    if d < _EPS:
        return
    tx, ty = dx / d, dy / d
    net.add_source(float(node_xy[0]) + tx * inset,
                   float(node_xy[1]) + ty * inset,
                   math.degrees(math.atan2(-tx, ty)))


def _goal_at(net: RoadNetwork, node_xy, toward_xy, inset: float = 6.0) -> None:
    """A destination pulled the same distance back inside the road, so the
    goal disc sits on tarmac rather than half on the verge."""
    dx = float(toward_xy[0]) - float(node_xy[0])
    dy = float(toward_xy[1]) - float(node_xy[1])
    d = math.hypot(dx, dy)
    if d < _EPS:
        net.add_goal(node_xy[0], node_xy[1])
        return
    net.add_goal(float(node_xy[0]) + dx / d * inset,
                 float(node_xy[1]) + dy / d * inset)


def _chain(net: RoadNetwork, nodes: list[int], lanes: int) -> None:
    """Connect an ordered run of nodes with straights — a street that has been
    split at every junction along it."""
    for a, b in zip(nodes, nodes[1:]):
        net.add_straight(a, b, lanes)


def _build_parking_lot(net: RoadNetwork, spec: dict) -> None:
    """Env 1 — parking lot. A perimeter loop with parking aisles across it and
    one-lane bays off the aisles, entered from a three-lane access road.

    The curriculum's baseline: everything happens at manoeuvring speed, the
    conflict is geometric rather than fast, and the source and destination are
    both bays, so the episode is the canonical 'leave one bay, reach another'
    task. The one-lane bays are the narrowest pieces in any layout, which is
    what makes this the first place a width-aware off-road test is exercised.
    """
    aisles = max(1, int(spec["aisles"]))
    bays = max(2, int(spec["bays"]))
    block = float(spec["block"])
    bay = float(spec["bay"])
    minor = _minor(spec, net)          # perimeter + aisles
    main = net.lanes                   # access road

    hx, hy = block, block * 0.8
    bay_x = [(-hx + 2 * hx * (i + 1) / (aisles + 1)) for i in range(aisles)]
    entry_x = 0.0 if aisles % 2 == 0 else bay_x[0] / 2.0

    # Bottom and top perimeter edges, split at every aisle and at the entry.
    def edge(y: float, xs: list[float], kinds: list[str]) -> list[int]:
        out = [net.add_node(-hx, y, "tee")]
        for x, kind in zip(xs, kinds):
            out.append(net.add_node(x, y, kind))
        out.append(net.add_node(hx, y, "tee"))
        return out

    bottom_xs = sorted(bay_x + [entry_x])
    bottom_kinds = ["cross" if abs(x - entry_x) < 1e-6 else "tee" for x in bottom_xs]
    bot = edge(-hy, bottom_xs, bottom_kinds)
    top = edge(hy, bay_x, ["tee"] * aisles)
    _chain(net, bot, minor)
    _chain(net, top, minor)
    net.add_straight(bot[0], top[0], minor)          # left edge
    net.add_straight(bot[-1], top[-1], minor)        # right edge

    # Access road, in from outside the lot.
    gate = net.add_node(entry_x, -hy - 55.0, "end")
    net.add_straight(gate, bot[bottom_xs.index(entry_x) + 1], main)
    _source_at(net, (entry_x, -hy - 55.0), (entry_x, -hy), main)

    # Aisles, split at every bay pair, with a one-lane stub each side.
    bay_ends: list[tuple[float, float, float, float]] = []
    for k, x in enumerate(bay_x):
        lower = bot[bottom_xs.index(x) + 1]
        upper = top[k + 1]
        run = [lower]
        for i in range(bays):
            y = -hy + 2 * hy * (i + 1) / (bays + 1)
            run.append(net.add_node(x, y, "cross"))
        run.append(upper)
        _chain(net, run, minor)
        for node in run[1:-1]:
            nx, ny = net.nodes[node]
            for side in (-1.0, 1.0):
                tip = net.add_node(nx + side * bay, ny, "end")
                net.add_straight(node, tip, 1)
                bay_ends.append((nx + side * bay, ny, nx, ny))

    # Every bay is both a place to start and a place to be sent. Sources are
    # nose-out (the vehicle has already reversed out of its space and is
    # pointing down the aisle); goals sit just inside the bay mouth.
    for bx, by, ax, ay in bay_ends:
        _source_at(net, (bx, by), (ax, ay), 1, inset=bay * 0.4)
        _goal_at(net, (bx, by), (ax, ay), inset=bay * 0.35)


def _build_intersection_x(net: RoadNetwork, spec: dict) -> None:
    """Env 2 — unsignalised crossings. A three-lane arterial crossed by two
    two-lane streets, with a connector between them closing a block.

    Two full four-way crossings on the same arterial, so a policy cannot learn
    a single junction's timing by heart, plus the connector's tees, which give
    three different right-of-way geometries in one layout."""
    arm = float(spec["arm"])
    block = float(spec["block"])
    minor = _minor(spec, net)
    main = net.lanes
    span = arm * 0.75
    conn_x = block * 0.85

    south = net.add_node(0.0, -arm, "end")
    north = net.add_node(0.0, arm, "end")
    xs_lo = net.add_node(0.0, -block / 2.0, "cross")
    xs_hi = net.add_node(0.0, block / 2.0, "cross")
    _chain(net, [south, xs_lo, xs_hi, north], main)

    streets = []
    for node, y in ((xs_lo, -block / 2.0), (xs_hi, block / 2.0)):
        west = net.add_node(-span, y, "end")
        east = net.add_node(span, y, "end")
        conn = net.add_node(conn_x, y, "cross")
        _chain(net, [west, node, conn, east], minor)
        streets.append((west, east, conn, y))
    net.add_straight(streets[0][2], streets[1][2], minor)

    _source_at(net, (0.0, -arm), (0.0, -block / 2.0), main)
    _source_at(net, (-span, streets[0][3]), (0.0, streets[0][3]), minor)
    _goal_at(net, (0.0, arm), (0.0, block / 2.0))
    _goal_at(net, (span, streets[1][3]), (conn_x, streets[1][3]))


def _build_merge_ramp(net: RoadNetwork, spec: dict) -> None:
    """Env 3 — on-ramp merge. A three-lane carriageway with a two-lane ramp
    joining it through an S-bend, and a two-lane cross street downstream.

    The scenario with a deadline: the ramp ends, so 'wait for a better gap' is
    not available indefinitely, which is exactly the time-against-safety
    trade-off a learned multiplier is supposed to arbitrate."""
    arm = float(spec["arm"])
    lead = max(float(spec["ramp"]), 20.0)
    minor = _minor(spec, net)
    main = net.lanes

    rise = 70.0                 # lateral offset of the ramp from the main road
    r = rise / 2.0              # two equal quarter-arcs make the S-bend
    merge_x = -arm * 0.2
    cross_x = arm * 0.35

    west = net.add_node(-arm, 0.0, "end")
    merge = net.add_node(merge_x, 0.0, "tee")
    cross = net.add_node(cross_x, 0.0, "cross")
    east = net.add_node(arm, 0.0, "end")
    _chain(net, [west, merge, cross, east], main)

    south = net.add_node(cross_x, -rise, "end")
    north = net.add_node(cross_x, rise, "end")
    _chain(net, [south, cross, north], minor)

    # Ramp: a straight lead-in, then two quarter-circles that arrive at the
    # merge node already parallel to the carriageway. Both arcs are written
    # from their own centres so their endpoints land exactly on their nodes —
    # `finalize` checks this, and a ramp that misses by a metre would make
    # every route through it shorter than the road actually is.
    bend_x = merge_x - 2.0 * r
    start = net.add_node(bend_x - lead, -rise, "end")
    bend = net.add_node(bend_x, -rise, "tee")
    mid = net.add_node(bend_x + r, -r, "tee")
    net.add_straight(start, bend, minor)
    net.add_arc(bend, mid, (bend_x, -r), r, -math.pi / 2.0, math.pi / 2.0, minor)
    net.add_arc(mid, merge, (bend_x + 2.0 * r, -r), r, math.pi, -math.pi / 2.0, minor)

    _source_at(net, (bend_x - lead, -rise), (bend_x, -rise), minor)
    _source_at(net, (-arm, 0.0), (merge_x, 0.0), main)
    _goal_at(net, (arm, 0.0), (cross_x, 0.0))
    _goal_at(net, (cross_x, rise), (cross_x, 0.0))


def _build_roundabout_yield(net: RoadNetwork, spec: dict) -> None:
    """Env 4 — roundabout with a yield on entry. A two-lane ring, approach
    arms that alternate three and two lanes, and an outer distributor road
    linking the approaches.

    Curvature makes the lane-boundary cost continuously non-zero here rather
    than event-like, and the alternating arm widths mean the entry geometry is
    different at every approach."""
    radius = max(float(spec["radius"]), net.max_half_width * 1.6)
    arms = max(3, int(spec["arms"]))
    arm = float(spec["arm"])
    minor = _minor(spec, net)
    main = net.lanes

    ring = []
    for k in range(arms):
        a = 2.0 * math.pi * k / arms
        ring.append(net.add_node(radius * math.cos(a), radius * math.sin(a),
                                 "roundabout", pad=minor * net.lane_width / 2.0))
    for k in range(arms):
        net.add_arc(ring[k], ring[(k + 1) % arms], (0.0, 0.0), radius,
                    2.0 * math.pi * k / arms, 2.0 * math.pi / arms, minor)

    hub_r = radius + arm
    hubs, tips = [], []
    for k in range(arms):
        a = 2.0 * math.pi * k / arms
        lanes = main if k % 2 == 0 else minor
        hub = net.add_node(hub_r * math.cos(a), hub_r * math.sin(a), "cross")
        net.add_straight(ring[k], hub, lanes)
        tip = net.add_node((hub_r + 30.0) * math.cos(a),
                           (hub_r + 30.0) * math.sin(a), "end")
        net.add_straight(hub, tip, lanes)
        hubs.append(hub)
        tips.append((tip, lanes, a))

    # The distributor: a polygon through the hubs, so an approach can also be
    # reached without passing through the roundabout at all.
    for k in range(arms):
        net.add_straight(hubs[k], hubs[(k + 1) % arms], minor)

    for k, (tip, lanes, a) in enumerate(tips):
        tip_xy = net.nodes[tip]
        hub_xy = net.nodes[hubs[k]]
        if k % 2 == 0:
            _source_at(net, tip_xy, hub_xy, lanes)
        else:
            _goal_at(net, tip_xy, hub_xy)
    net.islands.append((0.0, 0.0, max(radius - minor * net.lane_width / 2.0, 0.5)))


def _build_manhattan(net: RoadNetwork, spec: dict) -> None:
    """Env 5 — city grid. A rows x cols lattice whose middle row and middle
    column are three-lane avenues and whose remaining streets are two-lane,
    with short stubs at the corners to start from and aim at.

    Nine crossings on one map: the long-horizon, route-choice member of the
    set, and the one where congestion emerges rather than being placed."""
    rows, cols = max(2, int(spec["rows"])), max(2, int(spec["cols"]))
    block = float(spec["block"])
    minor = _minor(spec, net)
    main = net.lanes
    mid_r, mid_c = rows // 2, cols // 2

    ids = _grid_nodes(net, rows, cols, block)
    for (r, c), node in ids.items():
        right = ids.get((r, c + 1))
        if right is not None:                      # an east-west street
            net.add_straight(node, right, main if r == mid_r else minor)
        up = ids.get((r + 1, c))
        if up is not None:                         # a north-south street
            net.add_straight(node, up, main if c == mid_c else minor)

    # Corner stubs: a grid has no dead ends of its own, and a source sitting
    # on a crossing would start every episode inside a junction.
    x0 = -(cols - 1) * block / 2.0
    y0 = -(rows - 1) * block / 2.0
    stub = block * 0.35
    corners = (((0, 0), (-stub, 0.0)), ((0, cols - 1), (stub, 0.0)),
               ((rows - 1, 0), (-stub, 0.0)), ((rows - 1, cols - 1), (stub, 0.0)))
    for (r, c), (dx, dy) in corners:
        node = ids[(r, c)]
        cx, cy = x0 + c * block, y0 + r * block
        tip = net.add_node(cx + dx, cy + dy, "end")
        lanes = main if r == mid_r else minor
        net.add_straight(node, tip, lanes)
        if r == 0:
            _source_at(net, (cx + dx, cy + dy), (cx, cy), lanes)
        else:
            _goal_at(net, (cx + dx, cy + dy), (cx, cy))


def _build_left_turn(net: RoadNetwork, spec: dict) -> None:
    """Env 6 — unprotected turn across oncoming traffic. A three-lane arterial
    with two two-lane side roads hanging off it, joined into a block so the
    turn can be approached from either end.

    The highest-severity conflict class in real crash statistics, and the one
    where the cost of waiting and the cost of going are both real."""
    arm = float(spec["arm"])
    block = float(spec["block"])
    minor = _minor(spec, net)
    main = net.lanes

    west = net.add_node(-arm, 0.0, "end")
    t_a = net.add_node(0.0, 0.0, "tee")
    t_b = net.add_node(block, 0.0, "cross")
    east = net.add_node(arm, 0.0, "end")
    _chain(net, [west, t_a, t_b, east], main)

    # The block south of the arterial: down from the first tee, east, then
    # back up to the second — a vehicle can reach the same destination either
    # by turning across the oncoming stream or by going the long way round,
    # which is what makes the turn a decision rather than a requirement.
    a_s = net.add_node(0.0, -block, "tee")
    b_s = net.add_node(block, -block, "cross")
    net.add_straight(t_a, a_s, minor)
    net.add_straight(a_s, b_s, minor)
    net.add_straight(b_s, t_b, minor)

    north = net.add_node(block, block * 0.8, "end")
    net.add_straight(t_b, north, minor)
    tail = net.add_node(block, -block - 35.0, "end")
    net.add_straight(b_s, tail, minor)

    _source_at(net, (-arm, 0.0), (0.0, 0.0), main)
    _source_at(net, (arm, 0.0), (block, 0.0), main)
    _goal_at(net, (block, block * 0.8), (block, 0.0))
    _goal_at(net, (block, -block - 35.0), (block, -block))


def _build_lane_closure(net: RoadNetwork, spec: dict) -> None:
    """Env 7 — lane closure. A three-lane road pinched to a single lane over a
    stretch in the middle, with a two-lane detour round it and a cross street
    beyond.

    The sharpest test of the constrained formulation: the narrow section is
    genuinely narrower than the road either side of it, so holding the lane
    the vehicle arrived in and reaching the destination are incompatible. A
    fixed penalty weight has to be either too small to matter or large enough
    to stall the vehicle at the taper; a learned multiplier has a third
    option."""
    arm = float(spec["arm"])
    closure = max(float(spec["closure"]), 10.0)
    minor = _minor(spec, net)
    main = net.lanes
    half = closure / 2.0
    detour_y = 45.0

    west = net.add_node(-arm, 0.0, "end")
    taper_w = net.add_node(-half, 0.0, "cross")
    taper_e = net.add_node(half, 0.0, "cross")
    cross = net.add_node(arm * 0.55, 0.0, "cross")
    east = net.add_node(arm, 0.0, "end")
    net.add_straight(west, taper_w, main)
    net.add_straight(taper_w, taper_e, 1)          # the closure itself
    net.add_straight(taper_e, cross, main)
    net.add_straight(cross, east, main)

    d_w = net.add_node(-half, detour_y, "tee")
    d_e = net.add_node(half, detour_y, "tee")
    net.add_straight(taper_w, d_w, minor)
    net.add_straight(d_w, d_e, minor)
    net.add_straight(d_e, taper_e, minor)

    south = net.add_node(arm * 0.55, -detour_y - 25.0, "end")
    north = net.add_node(arm * 0.55, detour_y + 25.0, "end")
    _chain(net, [south, cross, north], minor)

    _source_at(net, (-arm, 0.0), (-half, 0.0), main)
    _goal_at(net, (arm, 0.0), (arm * 0.55, 0.0))
    _goal_at(net, (arm * 0.55, detour_y + 25.0), (arm * 0.55, 0.0))


def _build_weave_roundabout(net: RoadNetwork, spec: dict) -> None:
    """Env 8 — multi-lane roundabout with weaving. A three-lane ring, five
    approaches alternating three and two lanes, and a concentric outer ring
    joining the approaches partway out.

    The composite, and the one held out of training in the curriculum: a
    vehicle crossing between the two rings has to change lanes inside a bend
    while yielding, which is both conflicts in the set at once."""
    radius = max(float(spec["radius"]), net.max_half_width * 1.6)
    arms = max(4, int(spec["arms"]))
    arm = float(spec["arm"])
    minor = _minor(spec, net)
    main = net.lanes
    weave_r = radius + arm * 0.55

    ring, weave = [], []
    for k in range(arms):
        a = 2.0 * math.pi * k / arms
        ring.append(net.add_node(radius * math.cos(a), radius * math.sin(a),
                                 "roundabout", pad=main * net.lane_width / 2.0))
        weave.append(net.add_node(weave_r * math.cos(a), weave_r * math.sin(a),
                                  "cross"))
    sweep = 2.0 * math.pi / arms
    for k in range(arms):
        a = 2.0 * math.pi * k / arms
        net.add_arc(ring[k], ring[(k + 1) % arms], (0.0, 0.0), radius, a, sweep, main)
        # The outer ring is concentric, so its arcs land exactly on the weave
        # nodes without any trimming.
        net.add_arc(weave[k], weave[(k + 1) % arms], (0.0, 0.0), weave_r, a, sweep,
                    minor)
        lanes = main if k % 2 == 0 else minor
        net.add_straight(ring[k], weave[k], lanes)
        tip = net.add_node((radius + arm) * math.cos(a),
                           (radius + arm) * math.sin(a), "end")
        net.add_straight(weave[k], tip, lanes)
        if k % 2 == 0:
            _source_at(net, net.nodes[tip], net.nodes[weave[k]], lanes)
        else:
            _goal_at(net, net.nodes[tip], net.nodes[weave[k]])
    net.islands.append((0.0, 0.0, max(radius - main * net.lane_width / 2.0, 0.5)))


_BUILDERS = {
    "cross": _build_cross,
    "tee": _build_tee,
    "grid": _build_grid,
    "roundabout": _build_roundabout,
    "loop": _build_loop,
    "town": _build_town,
    "parking_lot": _build_parking_lot,
    "intersection_x": _build_intersection_x,
    "merge_ramp": _build_merge_ramp,
    "roundabout_yield": _build_roundabout_yield,
    "manhattan": _build_manhattan,
    "left_turn": _build_left_turn,
    "lane_closure": _build_lane_closure,
    "weave_roundabout": _build_weave_roundabout,
}
