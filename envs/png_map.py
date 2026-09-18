"""PNG maps — paint an environment, or export one you built.

Both directions, over one palette:

    save(world, "town.png")     World  -> classified PNG
    load("town.png")            PNG    -> World

The PNG is a **classified top-down map**, not a picture: every pixel is one
of a handful of exact palette colours, and the class is what it means. Paint
one in any image editor with the palette below and the loader turns it into a
routable road network with buildings, trees, and episode endpoints.

    colour      hex        class
    ----------  ---------  ------------------------------------------
    grass       #6f9642    nothing (background — anything unmatched)
    asphalt     #2a2b30    drivable road
    building    #8d8579    building footprint
    tree        #2f7a2a    tree
    source      #2b7fd9    an episode start point
    goal        #c43b3b    an episode destination

## How painted asphalt becomes a graph

The reward design needs distance ALONG the roads and a lane offset, and
neither is answerable from a drivable mask — so the loader recovers the
centreline graph rather than keeping the pixels:

1. **Skeletonise** the asphalt mask down to a one-pixel-wide medial axis.
2. **Classify** each skeleton pixel by its neighbour count: 1 is a dead end,
   3 or more is a junction, 2 is interior.
3. **Trace** each run of interior pixels between two of those, and simplify
   it (Ramer-Douglas-Peucker) into a short chain of straight pieces.
4. **Measure** the width from the distance transform at the skeleton — the
   value there IS the half-width in pixels — and round it to a whole number
   of lanes, so a painted stroke that is two lanes wide loads as two lanes.

Curves come back as chains of short straights rather than as arcs. That is a
deliberate loss: fitting arcs to a skeleton is fiddly and every query in
`RoadNetwork` is piecewise, so a 2 m chord on a bend costs nothing and
removes a whole class of fitting bug.

## What a round trip does and does not preserve

Preserved: layout, widths to the nearest lane, buildings (position, footprint
and yaw), trees, sources and goals. Not preserved: arcs (they come back as
chains), building and tree HEIGHTS (not representable in a top-down map — the
loader re-rolls them), and sub-pixel geometry. `save` then `load` gives an
equivalent world to drive in, not a byte-identical one.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

from envs import road_network, scenery
from envs.world import World

# -- palette --------------------------------------------------------------
GRASS = (0x6f, 0x96, 0x42)
ASPHALT = (0x2a, 0x2b, 0x30)
BUILDING = (0x8d, 0x85, 0x79)
TREE = (0x2f, 0x7a, 0x2a)
SOURCE = (0x2b, 0x7f, 0xd9)
GOAL = (0xc4, 0x3b, 0x3b)

PALETTE = {"grass": GRASS, "asphalt": ASPHALT, "building": BUILDING,
           "tree": TREE, "source": SOURCE, "goal": GOAL}

DEFAULT_PX_PER_M = 2.0
RDP_TOLERANCE_M = 1.2     # how far a simplified chain may stray from the skeleton
MIN_PIECE_M = 3.0         # shorter runs than this are junction noise, not roads
MARKER_RADIUS_M = 2.0     # how big a source/goal dot is drawn by `save`


# ── PNG out ──────────────────────────────────────────────────────────────

def save(world: World, path: str, px_per_m: float = DEFAULT_PX_PER_M) -> str:
    """Write `world` as a classified PNG. Round-trips through `load`."""
    x0, y0, x1, y1 = world.bounds()
    w = int(math.ceil((x1 - x0) * px_per_m)) + 2
    h = int(math.ceil((y1 - y0) * px_per_m)) + 2
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = GRASS

    def to_px(x, y):
        # Image rows run downward, world y runs upward.
        return ((np.asarray(x) - x0) * px_per_m + 1.0,
                (y1 - np.asarray(y)) * px_per_m + 1.0)

    yy, xx = np.mgrid[0:h, 0:w]

    def stamp_disc(cx, cy, r, colour):
        px, py = to_px(cx, cy)
        img[(xx - px) ** 2 + (yy - py) ** 2 <= (r * px_per_m) ** 2] = colour

    def stamp_rect(cx, cy, hl, hw, yaw, colour):
        px, py = to_px(cx, cy)
        dx, dy = xx - px, yy - py
        # Image y is flipped, so the rotation is too.
        cos, sin = math.cos(-yaw), math.sin(-yaw)
        lx = cos * dx - sin * (-dy)
        ly = sin * dx + cos * (-dy)
        img[(np.abs(lx) <= hl * px_per_m) & (np.abs(ly) <= hw * px_per_m)] = colour

    for piece in world.net.pieces:
        poly = piece.polyline(0.0, piece.length)
        for (ax, ay), (bx, by) in zip(poly, poly[1:]):
            mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
            seg = math.hypot(bx - ax, by - ay)
            if seg < 1e-6:
                continue
            stamp_rect(mx, my, seg / 2.0 + 0.5 / px_per_m, piece.half_width,
                       math.atan2(by - ay, bx - ax), ASPHALT)
    for (nx, ny), pad in zip(world.net.nodes, world.net.node_pads):
        if pad > 0.0:
            stamp_disc(nx, ny, pad, ASPHALT)
    for cx, cy, r in world.net.islands:
        stamp_disc(cx, cy, r, GRASS)

    for cx, cy, hl, hw, yaw, _h in world.scenery.boxes:
        stamp_rect(float(cx), float(cy), float(hl), float(hw), float(yaw), BUILDING)
    for cx, cy, r, _h in world.scenery.trees:
        stamp_disc(float(cx), float(cy), float(r), TREE)

    for sx, sy, _h in world.net.sources:
        stamp_disc(sx, sy, MARKER_RADIUS_M, SOURCE)
    for gx, gy in world.net.goals:
        stamp_disc(gx, gy, MARKER_RADIUS_M, GOAL)

    Image.fromarray(img).save(path)
    return path


# ── PNG in ───────────────────────────────────────────────────────────────

def _masks(path: str) -> dict:
    """Nearest-palette-colour classification of every pixel.

    Nearest rather than exact: a PNG that has been through a lossy editor, a
    resize or a colour profile arrives a few units off, and an exact match
    would silently classify the whole map as grass.
    """
    img = np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)
    names = list(PALETTE)
    colours = np.array([PALETTE[n] for n in names], dtype=np.int16)
    d = ((img[:, :, None, :] - colours[None, None, :, :]) ** 2).sum(axis=3)
    nearest = d.argmin(axis=2)
    return {name: nearest == i for i, name in enumerate(names)}


def _rdp(points: np.ndarray, tol: float) -> np.ndarray:
    """Ramer-Douglas-Peucker, iteratively (a painted street can be thousands
    of skeleton pixels long and recursion would blow the stack)."""
    if len(points) < 3:
        return points
    keep = np.zeros(len(points), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = points[i], points[j]
        ab = b - a
        n = math.hypot(*ab)
        seg = points[i + 1:j] - a
        if n < 1e-9:
            dist = np.hypot(seg[:, 0], seg[:, 1])
        else:
            dist = np.abs(seg[:, 0] * ab[1] - seg[:, 1] * ab[0]) / n
        k = int(dist.argmax())
        if dist[k] > tol:
            keep[i + 1 + k] = True
            stack.append((i, i + 1 + k))
            stack.append((i + 1 + k, j))
    return points[keep]


def _trace_skeleton(skel: np.ndarray):
    """Runs of skeleton pixels between endpoints/junctions.

    Yields `(row, col)` arrays, each one path. Junction pixels appear at both
    ends of every path that meets them, which is what lets the caller weld
    the paths back into a connected graph.
    """
    neigh = np.zeros(skel.shape, dtype=np.int16)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            neigh += np.roll(np.roll(skel, dr, axis=0), dc, axis=1).astype(np.int16)
    neigh[~skel] = 0
    nodes = skel & ((neigh == 1) | (neigh >= 3))

    def neighbours(r, c):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < skel.shape[0] and 0 <= cc < skel.shape[1] and skel[rr, cc]:
                    yield rr, cc

    # `used` marks INTERIOR pixels already consumed by a path. Junction and
    # endpoint pixels are deliberately never marked, because every path that
    # meets one has to be able to start or stop there — that shared pixel is
    # what welds the paths into a connected graph.
    used = np.zeros(skel.shape, dtype=bool)

    for r, c in zip(*np.nonzero(nodes)):
        for start in neighbours(r, c):
            if used[start] or nodes[start] and start < (r, c):
                continue
            path = [(r, c), start]
            prev, cur = (r, c), start
            while not nodes[cur]:
                used[cur] = True
                nxt = [p for p in neighbours(*cur) if p != prev and not used[p]]
                if not nxt:
                    break
                prev, cur = cur, nxt[0]
                path.append(cur)
            if len(path) > 1:
                yield np.array(path, dtype=float)

    # A closed loop — a ring road with no junction anywhere on it — contains
    # no endpoint and no junction, so nothing above ever reaches it. Whatever
    # interior pixels are still unused after the walk above are exactly those
    # rings, and closing each one back onto its own first pixel is correct
    # here in a way it would not be for a half-traced street.
    for r, c in zip(*np.nonzero(skel & ~nodes & ~used)):
        if used[r, c]:
            continue
        path = [(r, c)]
        used[r, c] = True
        cur, prev = (r, c), None
        while True:
            nxt = [p for p in neighbours(*cur) if p != prev and not used[p]]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            used[cur] = True
            path.append(cur)
        if len(path) > 8:
            path.append(path[0])
            yield np.array(path, dtype=float)


def _components(mask: np.ndarray, min_px: int = 6):
    """Connected components of a boolean mask, as label images."""
    import cv2

    n, labels = cv2.connectedComponents(mask.astype(np.uint8))
    for i in range(1, n):
        where = labels == i
        if where.sum() >= min_px:
            yield where


def load(path: str, px_per_m: float = DEFAULT_PX_PER_M,
         lane_width: float = road_network.DEFAULT_LANE_WIDTH,
         rng: np.random.Generator | None = None) -> World:
    """Read a classified PNG back into a drivable World."""
    import cv2
    from scipy import ndimage
    from skimage.morphology import skeletonize

    rng = rng or np.random.default_rng(0)
    masks = _masks(path)
    road = masks["asphalt"]
    if not road.any():
        raise ValueError(f"{path}: no asphalt-coloured pixels — nothing drivable")

    h, w = road.shape

    def to_world(rc):
        """(row, col) pixels -> (x, y) metres, with the origin centred so a
        loaded map sits around (0, 0) like a built one."""
        rc = np.asarray(rc, dtype=float).reshape(-1, 2)
        x = (rc[:, 1] - w / 2.0) / px_per_m
        y = (h / 2.0 - rc[:, 0]) / px_per_m
        return np.stack([x, y], axis=1)

    # -- roads: skeleton -> chains of straights -------------------------
    skel = skeletonize(road)
    halfwidth_px = ndimage.distance_transform_edt(road)

    net = road_network.RoadNetwork(
        half_width=lane_width, lanes=road_network.DEFAULT_LANES,
        lane_width=lane_width, spec={"kind": "png", "source": path})

    node_ids: dict[tuple[int, int], int] = {}

    def node_at(xy) -> int:
        key = (round(float(xy[0]), 2), round(float(xy[1]), 2))
        if key not in node_ids:
            node_ids[key] = net.add_node(key[0], key[1], kind="tee")
        return node_ids[key]

    for path_px in _trace_skeleton(skel):
        pts = to_world(path_px)
        simple = _rdp(pts, RDP_TOLERANCE_M)
        if len(simple) < 2:
            continue
        rows = path_px[:, 0].astype(int)
        cols = path_px[:, 1].astype(int)
        hw_m = float(np.median(halfwidth_px[rows, cols])) / px_per_m
        lanes = max(1, int(round(2.0 * hw_m / lane_width)))

        # Spur pruning. Skeletonising a blunt road end or a junction pad
        # leaves a little star of branches shorter than the road is wide;
        # they are an artefact of the medial axis, not geometry anyone
        # painted, and left in they give the projection something to snap to
        # inside the carriageway.
        total = float(np.hypot(*(simple[-1] - simple[0])))
        if len(simple) == 2 and total < max(MIN_PIECE_M, 2.0 * hw_m):
            continue

        ids = [node_at(p) for p in simple]
        for a, b in zip(ids, ids[1:]):
            if a != b:
                net.add_straight(a, b, lanes)

    if not net.pieces:
        raise ValueError(f"{path}: asphalt found but no road longer than "
                         f"{MIN_PIECE_M} m — is px_per_m right?")

    for where in _components(masks["source"]):
        rc = np.argwhere(where).mean(axis=0)
        x, y = to_world(rc)[0]
        # A painted dot says WHERE, not which way; the heading is taken from
        # the road it sits on, which is the only direction it could mean.
        net.add_source(float(x), float(y), _heading_deg_from_net(net, x, y))
    for where in _components(masks["goal"]):
        rc = np.argwhere(where).mean(axis=0)
        x, y = to_world(rc)[0]
        net.add_goal(float(x), float(y))

    net.finalize()

    # -- scenery: connected components -> footprints ---------------------
    boxes = []
    for where in _components(masks["building"], min_px=12):
        (px, py), (pw, ph), ang = cv2.minAreaRect(
            np.argwhere(where)[:, ::-1].astype(np.int32))
        centre = to_world((py, px))[0]
        hl, hw = pw / 2.0 / px_per_m, ph / 2.0 / px_per_m
        # cv2 angles are clockwise in image space; world y is flipped.
        boxes.append((centre[0], centre[1], hl, hw, math.radians(-ang),
                      float(rng.uniform(*scenery.BUILDING_HEIGHT))))

    trees = []
    for where in _components(masks["tree"], min_px=4):
        rc = np.argwhere(where).mean(axis=0)
        centre = to_world(rc)[0]
        r = math.sqrt(where.sum() / math.pi) / px_per_m
        trees.append((centre[0], centre[1], r,
                      float(rng.uniform(*scenery.TREE_HEIGHT))))

    objects = scenery.Scenery(np.array(boxes or np.zeros((0, 6))),
                              np.array(trees or np.zeros((0, 4))))
    return World(net, objects, dict(net.spec))


def _heading_deg_from_net(net, x: float, y: float) -> float:
    """The H angle of the road under (x, y). Falls back to +X when the graph
    is still empty, which only happens for a source painted off the road."""
    if not net.pieces:
        return -90.0
    rad = net.heading_at(float(x), float(y))
    return math.degrees(rad) - 90.0


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    if args and args[0].endswith(".png"):
        world = load(args[0])
        print(f"loaded {args[0]}: {world}")
    else:
        kind = args[0] if args else "intersection_x"
        out = save(World.build(kind), f"{kind}_map.png")
        print(f"wrote {out}")
