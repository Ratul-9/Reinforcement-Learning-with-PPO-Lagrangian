"""Analytic sensors — lidar and radar computed against world geometry.

No renderer is involved. The world is already a small set of convex
footprints (rotated rectangles for buildings and vehicles, circles for
trees), so a lidar sweep is a closed-form ray/shape intersection rather than
a depth buffer read back off a GPU. At 20-30 agents x 20 Hz that difference
is the difference between a training run that fits on a laptop and one that
does not, and the answers are exact instead of being quantised to a
framebuffer.

The geometry here is the SAME geometry the renderer draws (see
`envs/scenery.py`), so a pixel observation added later — an offscreen camera
over the same scene — will agree with what the lidar reported, not describe a
second, parallel world.

Everything is vectorised over (rays x objects). A ray that hits nothing
returns `max_range`, which is what the observation space is clipped to.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-9


def ray_dirs(heading: float, n_rays: int, fov: float) -> np.ndarray:
    """(n_rays, 2) unit directions, fanned symmetrically about `heading`.

    A full 360 sweep (`fov = 2*pi`) drops the duplicate ray at the far end so
    the first and last beams are not the same beam.
    """
    full = fov >= 2.0 * np.pi - 1e-6
    angles = heading + np.linspace(-fov / 2.0, fov / 2.0, n_rays,
                                   endpoint=not full)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)


def _hit_circles(origin: np.ndarray, dirs: np.ndarray,
                 circles: np.ndarray, max_range: float) -> np.ndarray:
    """Nearest positive t along each ray against (M, 3) cx, cy, r."""
    if len(circles) == 0:
        return np.full(len(dirs), max_range)
    # f = origin - centre, per circle: (M, 2)
    f = origin[None, :] - circles[:, :2]
    # Broadcast rather than `dirs @ f.T`. macOS's Accelerate BLAS leaves
    # spurious divide-by-zero / overflow flags set on small matmuls, which
    # surface as a page of RuntimeWarnings per reset (or a FloatingPointError
    # under `np.seterr(all="raise")`) on finite inputs. At (rays x objects x 2)
    # this is small enough that the explicit product costs nothing.
    b = (dirs[:, None, :] * f[None, :, :]).sum(axis=2)   # (R, M)
    c = (f * f).sum(axis=1) - circles[:, 2] ** 2     # (M,)
    disc = b * b - c[None, :]
    ok = disc > 0.0
    root = np.sqrt(np.where(ok, disc, 0.0))
    # Near root first; if the ray starts inside the circle it is negative, so
    # fall back to the far root before giving up on this circle.
    t = -b - root
    t = np.where(t > _EPS, t, -b + root)
    t = np.where(ok & (t > _EPS), t, np.inf)
    return np.minimum(t.min(axis=1), max_range)


def _hit_boxes(origin: np.ndarray, dirs: np.ndarray,
               boxes: np.ndarray, max_range: float) -> np.ndarray:
    """Nearest positive t against (N, 5) cx, cy, half_l, half_w, yaw.

    Slab test in each box's own frame: rotate the ray into the box, clip it
    against the two axis-aligned slabs, and the interval that survives is the
    crossing.
    """
    if len(boxes) == 0:
        return np.full(len(dirs), max_range)
    cx, cy, hl, hw, yaw = boxes.T
    cos, sin = np.cos(yaw), np.sin(yaw)

    # Ray origin in each box's local frame: (N, 2)
    dx, dy = origin[0] - cx, origin[1] - cy
    ox = cos * dx + sin * dy
    oy = -sin * dx + cos * dy

    # Ray direction in each box's local frame: (R, N)
    dxr, dyr = dirs[:, 0:1], dirs[:, 1:2]
    lx = cos[None, :] * dxr + sin[None, :] * dyr
    ly = -sin[None, :] * dxr + cos[None, :] * dyr

    half = np.stack([hl, hw])                        # (2, N)
    o = np.stack([ox, oy])                           # (2, N)
    d = np.stack([lx, ly])                           # (2, R, N)

    # Parallel rays (|d| ~ 0) either miss the slab entirely or are inside it
    # for all t; the sign trick below keeps them from producing a false hit.
    safe = np.where(np.abs(d) < _EPS, _EPS, d)
    t1 = (-half[:, None, :] - o[:, None, :]) / safe
    t2 = (half[:, None, :] - o[:, None, :]) / safe
    lo = np.minimum(t1, t2).max(axis=0)              # (R, N)
    hi = np.maximum(t1, t2).min(axis=0)
    outside = (np.abs(d) < _EPS) & (np.abs(o[:, None, :]) > half[:, None, :])
    hit = (hi >= np.maximum(lo, 0.0)) & ~outside.any(axis=0)

    t = np.where(lo > _EPS, lo, hi)                  # inside the box -> exit
    t = np.where(hit & (t > _EPS), t, np.inf)
    return np.minimum(t.min(axis=1), max_range)


def lidar(origin, heading: float, boxes: np.ndarray, circles: np.ndarray,
          n_rays: int = 120, fov: float = 2.0 * np.pi,
          max_range: float = 100.0) -> np.ndarray:
    """Ranges (n_rays,) from `origin` facing `heading`.

    `boxes` is (N, 5) — cx, cy, half_l, half_w, yaw — and holds buildings AND
    other vehicles; `circles` is (M, 3) — cx, cy, r — and holds trees. Callers
    concatenate their own moving obstacles into `boxes` rather than this
    taking a list of worlds, so there is one array layout in play everywhere.
    """
    origin = np.asarray(origin, dtype=float)
    dirs = ray_dirs(heading, n_rays, fov)
    d = np.minimum(_hit_boxes(origin, dirs, np.asarray(boxes, dtype=float).reshape(-1, 5), max_range),
                   _hit_circles(origin, dirs, np.asarray(circles, dtype=float).reshape(-1, 3), max_range))
    return d.astype(np.float32)


def radar(origin, heading: float, velocity, others: np.ndarray,
          max_objects: int = 5, max_range: float = 100.0) -> np.ndarray:
    """The `max_objects` nearest moving objects, in the ego vehicle's frame.

    `others` is (K, 4) — x, y, vx, vy in WORLD coordinates. Returns
    (max_objects, 4) rows of [forward, left, relative forward speed,
    relative left speed], zero-padded when fewer than `max_objects` are in
    range. Ego-frame because a policy steering the car cares where a thing is
    relative to its own nose, and world coordinates would make it learn the
    rotation itself.
    """
    out = np.zeros((max_objects, 4), dtype=np.float32)
    others = np.asarray(others, dtype=float).reshape(-1, 4)
    if len(others) == 0:
        return out

    origin = np.asarray(origin, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    rel = others[:, :2] - origin[None, :]
    dist = np.hypot(rel[:, 0], rel[:, 1])
    keep = np.argsort(dist)[:max_objects]
    keep = keep[dist[keep] <= max_range]
    if len(keep) == 0:
        return out

    cos, sin = np.cos(-heading), np.sin(-heading)
    rel = rel[keep]
    dv = others[keep, 2:] - velocity[None, :]
    out[:len(keep), 0] = cos * rel[:, 0] - sin * rel[:, 1]
    out[:len(keep), 1] = sin * rel[:, 0] + cos * rel[:, 1]
    out[:len(keep), 2] = cos * dv[:, 0] - sin * dv[:, 1]
    out[:len(keep), 3] = sin * dv[:, 0] + cos * dv[:, 1]
    return out
