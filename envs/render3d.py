"""Panda3D renderer — the 3D view, and the PNG frames that come out of it.

Same `World` the env drives in, extruded: the road ribbons and junction pads
are laid flat on the ground, each scenery box becomes a building of its own
height, each scenery circle a tree. Nothing here invents geometry the
sensors cannot see, and nothing the sensors see is missing here.

Three camera modes, and the third is the point of the file:

    "orbit"  the whole network from above and behind — the audit view
    "close"  a low oblique on one point — for detail a whole-map view
             cannot show. A 5.5 m bridge deck on a 450 m layout is two
             pixels from the orbit camera; it is the scale, not the render
    "chase"  over one vehicle's shoulder — the rollout video
    "ego"    from the driver's eye — THE PIXEL OBSERVATION HOOK

`frame(..., camera="ego", size=(64, 64))` returns exactly the array
`Sedan`'s observation space already reserves under `rgb_camera`, so wiring
pixels into the policy later is a matter of calling this from the env's
`_observations` and paying the render cost, not of reworking anything.

Run offscreen by default, so it works over ssh and in CI:

    python -m envs.render3d manhattan          # writes manhattan_3d.png
    python -m envs.render3d manhattan --window # opens a window instead

WHAT THIS COSTS, because it decides whether you can afford it in the loop:
one offscreen frame is a GPU round-trip of a few milliseconds. At 20-30
agents that is slower than the entire rest of the step put together, which
is why the env's default observation is analytic lidar and this stays a
viewer until an experiment actually calls for pixels.
"""

from __future__ import annotations

import math

import numpy as np
from panda3d.core import (
    AmbientLight, DirectionalLight, Geom, GeomNode, GeomTriangles,
    GeomVertexData, GeomVertexFormat, GeomVertexWriter, NodePath,
    PerspectiveLens, Point3, Vec3, Vec4, loadPrcFileData,
)

GROUND = (0.55, 0.60, 0.42, 1.0)
ASPHALT = (0.20, 0.205, 0.22, 1.0)
MARKING = (0.90, 0.90, 0.84, 1.0)
ISLAND = (0.30, 0.46, 0.28, 1.0)
BUILDING = (0.62, 0.58, 0.52, 1.0)
TRUNK = (0.33, 0.25, 0.18, 1.0)
CANOPY = (0.22, 0.45, 0.20, 1.0)
VEHICLE = (0.85, 0.51, 0.17, 1.0)
BLOCKAGE = (0.55, 0.13, 0.13, 1.0)      # a stalled vehicle in the lane
STRUCTURE = (0.58, 0.57, 0.55, 1.0)     # bridge abutment / ramp embankment
EGO = (0.17, 0.50, 0.85, 1.0)

# Draw heights. The order is load-bearing rather than cosmetic: pads sit
# above markings so lane lines stop at a junction instead of being painted
# through it, and islands above pads because a ring node's pad overlaps the
# middle of a roundabout.
ROAD_Z, MARK_Z, PAD_Z, ISLAND_Z = 0.04, 0.06, 0.08, 0.10
EYE_HEIGHT = 1.25                 # driver's eye above the road, metres

_EDGE_INSET = 0.28
_EDGE_WIDTH = 0.14
_CENTRE_WIDTH = 0.16
_DASH_ON, _DASH_OFF = 3.0, 4.5


# ── geometry helpers ─────────────────────────────────────────────────────

class _Mesh:
    """Accumulates coloured triangles into one Geom. Flat surfaces get a
    constant +Z normal; anything with walls passes its own."""

    def __init__(self, name: str, color):
        self._name = name
        self._color = color
        self._vdata = GeomVertexData(name, GeomVertexFormat.getV3n3c4(),
                                     Geom.UHStatic)
        self._vw = GeomVertexWriter(self._vdata, "vertex")
        self._nw = GeomVertexWriter(self._vdata, "normal")
        self._cw = GeomVertexWriter(self._vdata, "color")
        self._tris = GeomTriangles(Geom.UHStatic)
        self._n = 0

    def vertex(self, x, y, z, normal=(0.0, 0.0, 1.0)) -> int:
        self._vw.addData3(x, y, z)
        self._nw.addData3(*normal)
        self._cw.addData4(*self._color)
        self._n += 1
        return self._n - 1

    def quad(self, p0, p1, p2, p3, z=0.0, normal=(0.0, 0.0, 1.0)) -> None:
        """Four 2-D corners at height `z`, ANTICLOCKWISE seen from above.

        Winding is not a detail here: Panda3D back-face culls, so a clockwise
        upward-facing quad is simply not drawn, and a road that silently does
        not render looks exactly like a road that was never built.
        """
        idx = [self.vertex(p[0], p[1], z, normal) for p in (p0, p1, p2, p3)]
        self._tris.addVertices(idx[0], idx[1], idx[2])
        self._tris.addVertices(idx[0], idx[2], idx[3])

    def quad3(self, p0, p1, p2, p3, normal) -> None:
        """Four 3-D corners — a wall."""
        idx = [self.vertex(*p, normal=normal) for p in (p0, p1, p2, p3)]
        self._tris.addVertices(idx[0], idx[1], idx[2])
        self._tris.addVertices(idx[0], idx[2], idx[3])

    def disc(self, cx, cy, r, z, segs: int = 28) -> None:
        centre = self.vertex(cx, cy, z)
        prev = None
        for i in range(segs + 1):
            a = 2.0 * math.pi * i / segs
            v = self.vertex(cx + r * math.cos(a), cy + r * math.sin(a), z)
            if prev is not None:
                self._tris.addVertices(centre, prev, v)
            prev = v

    def prism(self, cx, cy, r, z0, z1, segs: int, yaw: float = 0.0) -> None:
        """A capped extruded regular polygon — `segs=4` is a box (rotated to
        `yaw`), `segs=12` is close enough to a cylinder at any distance a
        camera will see a tree from. One routine instead of two because the
        only thing that differs is the side count."""
        pts = [(cx + r * math.cos(yaw + 2 * math.pi * i / segs),
                cy + r * math.sin(yaw + 2 * math.pi * i / segs))
               for i in range(segs)]
        top = [self.vertex(x, y, z1) for x, y in pts]
        for i in range(1, segs - 1):
            self._tris.addVertices(top[0], top[i], top[i + 1])
        for i in range(segs):
            a, b = pts[i], pts[(i + 1) % segs]
            nx, ny = b[1] - a[1], a[0] - b[0]
            n = math.hypot(nx, ny) or 1.0
            self.quad3((a[0], a[1], z0), (b[0], b[1], z0),
                       (b[0], b[1], z1), (a[0], a[1], z1),
                       normal=(nx / n, ny / n, 0.0))

    def box(self, cx, cy, hl, hw, yaw, z0, z1) -> None:
        """A rectangular block — the one shape `prism` cannot do, since its
        sides are two different lengths."""
        cos, sin = math.cos(yaw), math.sin(yaw)
        # Anticlockwise, like `prism`: it makes the roof face up and puts the
        # walls' right-hand normals on the outside, both from one ordering.
        corners = [(hl, hw), (-hl, hw), (-hl, -hw), (hl, -hw)]
        pts = [(cx + cos * x - sin * y, cy + sin * x + cos * y)
               for x, y in corners]
        self.quad(*pts, z=z1)
        for i in range(4):
            a, b = pts[i], pts[(i + 1) % 4]
            nx, ny = b[1] - a[1], a[0] - b[0]
            n = math.hypot(nx, ny) or 1.0
            self.quad3((a[0], a[1], z0), (b[0], b[1], z0),
                       (b[0], b[1], z1), (a[0], a[1], z1),
                       normal=(nx / n, ny / n, 0.0))

    def node(self) -> NodePath | None:
        if self._n == 0:
            return None
        geom = Geom(self._vdata)
        geom.addPrimitive(self._tris)
        gnode = GeomNode(self._name)
        gnode.addGeom(geom)
        return NodePath(gnode)


def _offset(poly: np.ndarray, d: float) -> np.ndarray:
    """A polyline shifted `d` metres to the right of travel."""
    poly = np.asarray(poly, dtype=float)
    if len(poly) < 2:
        return poly
    seg = np.gradient(poly, axis=0)
    n = np.hypot(seg[:, 0], seg[:, 1])
    n[n < 1e-9] = 1.0
    return np.stack([poly[:, 0] + seg[:, 1] / n * d,
                     poly[:, 1] - seg[:, 0] / n * d], axis=1)


def _ribbon(mesh: _Mesh, poly, half_width: float, z: float, zs=None) -> None:
    """A strip along a centreline. `zs` gives a per-vertex height, for a
    ramp or a deck; without it the whole strip sits flat at `z`."""
    left = _offset(poly, -half_width)
    right = _offset(poly, half_width)
    for i in range(len(poly) - 1):
        if zs is None:
            mesh.quad(right[i], right[i + 1], left[i + 1], left[i], z=z)
        else:
            mesh.quad3((right[i][0], right[i][1], z + zs[i]),
                       (right[i + 1][0], right[i + 1][1], z + zs[i + 1]),
                       (left[i + 1][0], left[i + 1][1], z + zs[i + 1]),
                       (left[i][0], left[i][1], z + zs[i]),
                       normal=(0.0, 0.0, 1.0))


# ── scene ────────────────────────────────────────────────────────────────

def build_scene(world, root: NodePath | None = None) -> NodePath:
    """One NodePath holding the whole static world. Render-only: it carries
    no collision geometry, because collisions are decided analytically in
    `World` against the same numbers this is built from."""
    root = root or NodePath("world")

    x0, y0, x1, y1 = world.bounds()
    ground = _Mesh("ground", GROUND)
    pad = 80.0
    ground.quad((x0 - pad, y0 - pad), (x1 + pad, y0 - pad),
                (x1 + pad, y1 + pad), (x0 - pad, y1 + pad), z=0.0)

    surface = _Mesh("road", ASPHALT)
    paint = _Mesh("markings", MARKING)
    pads = _Mesh("pads", ASPHALT)
    islands = _Mesh("islands", ISLAND)
    for i, piece in enumerate(world.net.pieces):
        poly = piece.polyline(0.0, piece.length)
        # Height per vertex, so a ramp climbs smoothly instead of stepping.
        heights = None
        if piece.z0 != 0.0 or piece.z1 != 0.0:
            span = max(len(poly) - 1, 1)
            heights = [piece.z0 + (piece.z1 - piece.z0) * k / span
                       for k in range(len(poly))]
        _ribbon(surface, poly, piece.half_width, ROAD_Z, heights)
        inset = max(piece.half_width - _EDGE_INSET, 0.05)
        for side in (inset, -inset):
            _ribbon(paint, _offset(poly, side), _EDGE_WIDTH / 2.0, MARK_Z)
        # One dashed line per lane divider, from the lane graph — so what is
        # painted is what the lane-keeping cost is measured against.
        for offset in world.lanes.dividers(i):
            s = _DASH_OFF / 2.0
            while s < piece.length:
                end = min(s + _DASH_ON, piece.length)
                if end - s > 0.4:
                    _ribbon(paint, _offset(piece.polyline(s, end), -offset),
                            _CENTRE_WIDTH / 2.0, MARK_Z)
                s = end + _DASH_OFF
    for (nx, ny), r in zip(world.net.nodes, world.net.node_pads):
        if r > 0.0:
            pads.disc(nx, ny, r, PAD_Z)
    for cx, cy, r in world.net.islands:
        islands.disc(cx, cy, r, ISLAND_Z, segs=48)

    # Stalled vehicles, drawn at vehicle height so they read as an
    # obstruction in the lane rather than as a very small building.
    blocked = _Mesh("blockages", BLOCKAGE)
    for cx, cy, hl, hw, yaw in world.blockages:
        blocked.box(float(cx), float(cy), float(hl), float(hw), float(yaw),
                    0.2, 1.45)

    # Sides for anything elevated: an embankment under a ramp, an abutment
    # under a deck. Without them a raised road is a ribbon floating in mid
    # air with no visible support, which reads as a rendering fault rather
    # than as a bridge.
    structure = _Mesh("structure", STRUCTURE)
    for piece in world.net.pieces:
        if piece.z0 == 0.0 and piece.z1 == 0.0:
            continue
        poly = piece.polyline(0.0, piece.length)
        span = max(len(poly) - 1, 1)
        heights = [piece.z0 + (piece.z1 - piece.z0) * k / span
                   for k in range(len(poly))]
        for sign in (1.0, -1.0):
            edge = _offset(poly, sign * piece.half_width)
            for i in range(len(poly) - 1):
                if max(heights[i], heights[i + 1]) < 0.15:
                    continue
                structure.quad3(
                    (edge[i][0], edge[i][1], 0.0),
                    (edge[i + 1][0], edge[i + 1][1], 0.0),
                    (edge[i + 1][0], edge[i + 1][1], heights[i + 1] + ROAD_Z),
                    (edge[i][0], edge[i][1], heights[i] + ROAD_Z),
                    normal=(sign, 0.0, 0.0))

    buildings = _Mesh("buildings", BUILDING)
    for cx, cy, hl, hw, yaw, h in world.scenery.boxes:
        buildings.box(float(cx), float(cy), float(hl), float(hw), float(yaw),
                      0.0, float(h))

    trunks = _Mesh("trunks", TRUNK)
    canopies = _Mesh("canopies", CANOPY)
    for cx, cy, r, h in world.scenery.trees:
        cx, cy, r, h = float(cx), float(cy), float(r), float(h)
        trunks.prism(cx, cy, r * 0.22, 0.0, h * 0.45, segs=8)
        canopies.prism(cx, cy, r, h * 0.40, h, segs=12)

    for mesh, offset in ((ground, 0), (surface, 2), (pads, 4), (paint, 3),
                         (islands, 5), (buildings, 0), (trunks, 0),
                         (canopies, 0), (blocked, 0), (structure, 0)):
        node = mesh.node()
        if node is not None:
            # A few centimetres of lift is below the depth buffer's
            # resolution from a few hundred metres back, which is the normal
            # way to look at a whole network; the bias is resolution-free and
            # fixes the speckling the lift alone leaves at that distance.
            node.setDepthOffset(offset)
            node.reparentTo(root)
    return root


# Body height per fleet type, for drawing only — the sensors are top-down
# and never ask. A bus drawn at a car's height reads as a very long car.
VEHICLE_HEIGHT = {"motorcycle": 1.30, "tuktuk": 1.70, "sedan": 1.45,
                  "truck": 3.00, "bus": 3.20}


def vehicle_nodes(rects, ego: int = 0, height: float = 1.45,
                  types=None) -> NodePath:
    """Every vehicle as a block, ego in a different colour. Rebuilt per frame
    — a few dozen boxes is cheaper to regenerate than to keep in sync."""
    root = NodePath("vehicles")
    others = _Mesh("traffic", VEHICLE)
    mine = _Mesh("ego", EGO)
    for k, (cx, cy, hl, hw, yaw) in enumerate(np.asarray(rects).reshape(-1, 5)):
        mesh = mine if k == ego else others
        top = height if types is None else VEHICLE_HEIGHT.get(types[k], height)
        mesh.box(float(cx), float(cy), float(hl), float(hw), float(yaw),
                 0.2, top)
    for mesh in (others, mine):
        node = mesh.node()
        if node is not None:
            node.reparentTo(root)
    return root


class Renderer3D:
    """A Panda3D window (or offscreen buffer) showing one World.

    One instance owns one Panda3D `ShowBase`, and Panda3D allows exactly one
    per process — so hold on to the renderer rather than constructing one per
    frame.
    """

    def __init__(self, world, size=(960, 720), window: bool = False):
        from direct.showbase.ShowBase import ShowBase

        if not window:
            loadPrcFileData("", "window-type offscreen")
        loadPrcFileData("", f"win-size {size[0]} {size[1]}")
        loadPrcFileData("", "audio-library-name null")

        self.base = ShowBase()
        self.base.setBackgroundColor(0.53, 0.68, 0.84, 1.0)
        self.world = world
        self.scene = build_scene(world)
        self.scene.reparentTo(self.base.render)
        self._vehicles: NodePath | None = None

        sun = DirectionalLight("sun")
        sun.setColor(Vec4(0.9, 0.88, 0.82, 1.0))
        sun_np = self.base.render.attachNewNode(sun)
        sun_np.setHpr(35.0, -55.0, 0.0)
        ambient = AmbientLight("ambient")
        ambient.setColor(Vec4(0.45, 0.47, 0.52, 1.0))
        self.base.render.setLight(sun_np)
        self.base.render.setLight(self.base.render.attachNewNode(ambient))

        self.base.disableMouse()

    # -- cameras ----------------------------------------------------------

    def _place_camera(self, mode: str, rects, ego: int,
                      focus=None, distance: float = 70.0) -> None:
        cam = self.base.camera
        if mode == "close":
            # Low and near, so height reads. `focus` defaults to the centre
            # of the road network — which for a grade separation is exactly
            # the crossing.
            if focus is None:
                x0, y0, x1, y1 = self.world.net.bounds()
                focus = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
            fx, fy = float(focus[0]), float(focus[1])
            cam.setPos(fx - distance * 0.75, fy - distance * 0.75,
                       distance * 0.38)
            cam.lookAt(Point3(fx, fy, 2.0))
            return
        if mode == "orbit" or rects is None or not len(rects):
            # Frame the ROADS, not the ground pad. World bounds include the
            # scenery and the margin around it, which on a 400 m layout put
            # the camera far enough back that a 5.5 m bridge deck was a
            # single pixel.
            x0, y0, x1, y1 = self.world.net.bounds()
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            span = max(x1 - x0, y1 - y0) * 1.15
            # A lower angle than a plan view: elevation is invisible from
            # directly overhead, and these layouts have a bridge in them.
            cam.setPos(cx, cy - span * 0.80, span * 0.42)
            cam.lookAt(Point3(cx, cy, 0.0))
            return

        vx, vy, hl, _hw, yaw = np.asarray(rects).reshape(-1, 5)[ego]
        cos, sin = math.cos(yaw), math.sin(yaw)
        if mode == "chase":
            back, up = 14.0, 6.0
            cam.setPos(vx - cos * back, vy - sin * back, up)
            cam.lookAt(Point3(vx + cos * 8.0, vy + sin * 8.0, 1.0))
        elif mode == "ego":
            # At the driver's eye, just behind the nose, looking down the
            # vehicle's own axis. Panda's H is +90 degrees off the maths one.
            nose = hl * 0.4
            cam.setPos(vx + cos * nose, vy + sin * nose, EYE_HEIGHT)
            cam.setHpr(math.degrees(yaw) - 90.0, 0.0, 0.0)
        else:
            raise ValueError(f"unknown camera mode {mode!r}")

    # -- frames -----------------------------------------------------------

    def frame(self, rects=None, ego: int = 0, camera: str = "orbit",
              fov: float = 70.0, path: str | None = None, types=None,
              focus=None, distance: float = 70.0):
        """Render one frame. Returns an (H, W, 3) uint8 array, or writes a
        PNG and returns the path when `path` is given."""
        if self._vehicles is not None:
            self._vehicles.removeNode()
            self._vehicles = None
        if rects is not None and len(rects):
            self._vehicles = vehicle_nodes(rects, ego=ego, types=types)
            self._vehicles.reparentTo(self.base.render)

        lens = PerspectiveLens()
        lens.setFov(fov)
        lens.setNearFar(0.3, 3000.0)
        self.base.cam.node().setLens(lens)
        self._place_camera(camera, rects, ego, focus, distance)

        self.base.graphicsEngine.renderFrame()
        self.base.graphicsEngine.renderFrame()   # one to build, one to settle

        if path:
            self.base.win.saveScreenshot(path)
            return path

        tex = self.base.win.getScreenshot()
        raw = tex.getRamImageAs("RGB")
        arr = np.frombuffer(bytes(raw), dtype=np.uint8)
        arr = arr.reshape(tex.getYSize(), tex.getXSize(), 3)
        return np.flipud(arr).copy()             # Panda's origin is bottom-left

    def close(self) -> None:
        self.base.destroy()


if __name__ == "__main__":
    import sys

    from envs.world import World

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    kind = args[0] if args else "manhattan"
    windowed = "--window" in sys.argv

    renderer = Renderer3D(World.build(kind), window=windowed)
    out = renderer.frame(path=f"{kind}_3d.png")
    print(f"wrote {out}")
    if windowed:
        renderer.base.run()
