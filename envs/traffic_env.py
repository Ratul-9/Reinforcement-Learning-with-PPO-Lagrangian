"""TrafficEnv — N vehicles driving a road network to their own destinations.

The training target. Every agent is an independent vehicle with its own
spawn, destination and route, all of them in the same world at the same time,
so conflicts arise from the geometry rather than from a script.

## Why the step signature is a batch, not one agent

`paper-idea.md`'s working assumption is a shared policy with the agents
acting as parallel rollout sources (agents-as-VecEnv) and one averaged PPO
update. So `step` takes a list of actions and returns lists — that IS the
vectorised interface, with the agents' shared world as the only coupling.
An agent that finishes or crashes is respawned in place rather than removed,
which keeps the arrays rectangular and keeps traffic density constant; its
`terminated` flag for that step tells the learner where the episode boundary
was.

## Reward and cost are separate, on purpose

This is the constrained-MDP split, and it is the reason the env exists rather
than a reward-weighted scalar:

    reward   goal reached, progress along the route toward it. Terminal and
             unarguable things only.
    cost     collision, off-road, lane departure, low time-to-collision,
             jerk. Returned as `info["cost"]`, a dict of named channels, each
             a per-step non-negative number.

Nothing in here multiplies a cost by a weight. The Lagrange multipliers do
that, outside, and the env deliberately provides no place to hand-tune one.

## What is deliberately not modelled yet

Traffic lights, right-of-way rules and pedestrians. The scenario set is
collision-prone by geometry (unsignalised crossings, yields, merges,
weaves), which is enough to make the constraints bite; signals add a discrete
state the policy has to observe and would change the observation space.
"""

from __future__ import annotations

import math

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from envs import sensors
from envs.world import World, wrap_pi
from vehicle import Sedan, Gear

# -- episode shape --------------------------------------------------------
MAX_EPISODE_SECONDS = 90.0
GOAL_RADIUS = 6.0            # metres along the route
MIN_ROUTE = 40.0             # a destination nearer than this is not a journey
OFFROAD_GRACE = 1.5          # seconds off the road before the episode ends

# -- observation shape ----------------------------------------------------
LIDAR_RAYS = 120
LIDAR_RANGE = 100.0
RADAR_OBJECTS = 5
# Two route waypoints: the near one says which way this lane runs, the far
# one is the steering target that reaches round the next corner.
LOOKAHEAD_NEAR = 8.0
LOOKAHEAD_FAR = 18.0

# -- cost thresholds (what counts as a violation, NOT what it is worth) ---
TTC_THRESHOLD = 2.0          # seconds; below this the TTC cost is charged
JERK_THRESHOLD = 5.0         # m/s^3
SPEED_LIMIT = 13.9           # m/s (50 km/h)


class TrafficEnv(gym.Env):
    """Multi-agent driving on one scenario.

    >>> env = TrafficEnv("roundabout_yield", n_agents=8, seed=0)
    >>> obs, info = env.reset()
    >>> acts = [env.action_space.sample() for _ in range(env.n_agents)]
    >>> obs, rew, term, trunc, info = env.step(acts)
    """

    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(self, scenario="intersection_x", n_agents: int = 20,
                 dt: float = 0.1, seed: int | None = None,
                 scenery_density: float = 1.0, vehicle_cls=Sedan,
                 max_seconds: float = MAX_EPISODE_SECONDS,
                 world: World | None = None):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.dt = float(dt)
        self.n_agents = int(n_agents)
        self.max_steps = int(max_seconds / self.dt)
        self.vehicle_cls = vehicle_cls

        # `world` wins over `scenario` when given, which is how a map loaded
        # from a PNG (`envs.png_map.load`) is trained on: the env does not
        # care where a World came from.
        self.world = world or World.build(
            scenario, rng=np.random.default_rng(seed),
            scenery_density=scenery_density)

        self.vehicles = [vehicle_cls(dt=self.dt) for _ in range(self.n_agents)]
        v = self.vehicles[0]
        # Body footprint: axles plus an overhang each end, track plus mirrors.
        self.half_length = (v.lf + v.lr) / 2.0 + 0.6
        self.half_width = v.track_width / 2.0 + 0.2

        self.action_space = v.action_space
        self.observation_space = spaces.Dict({
            # [vx, vy, yaw_rate, steering_angle, gear,
            #  road_margin, lateral_offset, heading_error]
            "state": spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32),
            # [route_dist, near wp forward, near wp left,
            #  far wp forward, far wp left, bend ahead]
            "navigation": spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
            "lidar": spaces.Box(0.0, LIDAR_RANGE, shape=(LIDAR_RAYS,), dtype=np.float32),
            "radar": spaces.Box(-np.inf, np.inf, shape=(RADAR_OBJECTS, 4), dtype=np.float32),
        })

        self._goals = np.zeros((self.n_agents, 2))
        self._route0 = np.ones(self.n_agents)
        self._route_prev = np.ones(self.n_agents)
        self._offroad_for = np.zeros(self.n_agents)
        self._accel_prev = np.zeros(self.n_agents)
        self._steps = 0

    # -- reset ------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._steps = 0
        self._offroad_for[:] = 0.0
        self._accel_prev[:] = 0.0
        # Park every vehicle far outside the world first. `_respawn` rejects
        # spawns that overlap another vehicle, and vehicles still sitting at
        # their constructor default would all be stacked on the origin and
        # veto every candidate near it.
        far = max(abs(c) for c in self.world.bounds()) + 500.0
        for k, veh in enumerate(self.vehicles):
            veh.x, veh.y, veh.heading = far + 10.0 * k, far, 0.0
        for i in range(self.n_agents):
            self._respawn(i)
        return self._observations(), {"cost": self._zero_costs()}

    def _respawn(self, i: int) -> None:
        """Put agent `i` at a fresh start with a fresh destination.

        Retried against the vehicles already placed: two agents spawned into
        the same parking bay would register a collision on step zero, and a
        policy cannot be charged for a cost it was handed. The first few
        attempts use the scenario's declared sources; once those are taken
        the rest of the traffic is placed anywhere on the network, because a
        layout declaring two approaches still has to hold twenty vehicles.
        """
        veh = self.vehicles[i]
        others = np.delete(self._rects(), i, axis=0)
        for attempt in range(24):
            x, y, heading = self.world.sample_start(self.rng, declared=attempt < 6)
            rect = (x, y, self.half_length, self.half_width, heading)
            if not self.world.rect_hits_rects(rect, others).any():
                break
        gx, gy, route = self.world.sample_goal(self.rng, (x, y), MIN_ROUTE)
        x, y, heading = self._snap_to_lane(x, y, (gx, gy), heading)

        veh.reset(spawn_point=(x, y), destination=(gx, gy))
        veh.heading = heading
        veh.gear = Gear.DRIVE
        self._goals[i] = (gx, gy)
        self._route0[i] = max(route, 1.0)
        self._route_prev[i] = route
        self._offroad_for[i] = 0.0
        self._accel_prev[i] = 0.0

    def _snap_to_lane(self, x: float, y: float, goal, fallback: float):
        """Move a spawn into the right-hand lane of the direction its route
        actually runs, and face it that way.

        A layout's declared sources are stored ON THE CENTRELINE, because
        which lane is the right-hand one depends on which way the vehicle is
        about to go, and that is not knowable until a destination has been
        drawn. Skipping this leaves every scenario spawn straddling the
        centre line, on the wrong side half the time, and charges the lane
        cost from step zero for a position the policy did not choose.
        """
        i, s, _lateral, _d = self.world.net.project(x, y)
        piece = self.world.net.pieces[i]
        cx, cy = piece.point(s)
        (ahead,), _ = self.world.route_probe((cx, cy), goal, (6.0,))
        dx, dy = ahead[0] - cx, ahead[1] - cy
        if math.hypot(dx, dy) < 1e-6:
            return x, y, fallback
        travel = math.atan2(dy, dx)
        if piece.lanes < 2:
            return cx, cy, travel
        # Right of travel is the heading rotated -90 degrees, half this
        # piece's width out — a one-lane bay has no right-hand lane to sit in.
        off = piece.half_width / 2.0
        return (cx + math.sin(travel) * off,
                cy - math.cos(travel) * off,
                travel)

    # -- geometry helpers -------------------------------------------------

    def _rects(self) -> np.ndarray:
        """(n_agents, 5) footprints of every vehicle, this instant."""
        return np.array([(v.x, v.y, self.half_length, self.half_width, v.heading)
                         for v in self.vehicles], dtype=float)

    def _moving(self) -> np.ndarray:
        """(n_agents, 4) world x, y, vx, vy — what radar reads."""
        out = np.zeros((self.n_agents, 4))
        for i, v in enumerate(self.vehicles):
            cos, sin = math.cos(v.heading), math.sin(v.heading)
            out[i] = (v.x, v.y,
                      v.vx * cos - v.vy * sin,
                      v.vx * sin + v.vy * cos)
        return out

    # -- step -------------------------------------------------------------

    def step(self, actions):
        """One control step for every agent. `actions` is a sequence of
        `n_agents` action dicts; everything returned is a list of the same
        length, plus `info["cost"]` — a dict of (n_agents,) cost arrays."""
        if len(actions) != self.n_agents:
            raise ValueError(f"expected {self.n_agents} actions, got {len(actions)}")

        speed_before = np.array([v.vx for v in self.vehicles])
        for veh, action in zip(self.vehicles, actions):
            veh.step(action)
        self._steps += 1

        rects = self._rects()
        moving = self._moving()
        costs = self._zero_costs()
        rewards = np.zeros(self.n_agents, dtype=np.float32)
        terminated = np.zeros(self.n_agents, dtype=bool)
        truncated = np.zeros(self.n_agents, dtype=bool)
        events = []

        for i, veh in enumerate(self.vehicles):
            margin, lateral, tangent = self.world.road_state(veh.x, veh.y)
            route = self.world.net.route_distance((veh.x, veh.y), tuple(self._goals[i]))

            # -- reward: progress along the route, and arriving ------------
            progress = self._route_prev[i] - route
            rewards[i] = progress / self._route0[i]
            self._route_prev[i] = route
            event = ""
            if route <= GOAL_RADIUS:
                rewards[i] += 1.0
                terminated[i] = True
                event = "goal"

            # -- costs: every guidance term, unweighted --------------------
            others = np.delete(rects, i, axis=0)
            hit_vehicle = bool(self.world.rect_hits_rects(rects[i], others).any())
            hit_static = self.world.hits_scenery(rects[i])
            if hit_vehicle or hit_static:
                costs["collision"][i] = 1.0
                terminated[i] = True
                event = "collision_vehicle" if hit_vehicle else "collision_static"

            if margin < 0.0:
                costs["offroad"][i] = 1.0
                self._offroad_for[i] += self.dt
                if self._offroad_for[i] > OFFROAD_GRACE:
                    terminated[i] = True
                    event = event or "offroad"
            else:
                self._offroad_for[i] = 0.0

            # Lane departure: the vehicle is on the road but not in a lane —
            # measured as being on the wrong side of the centreline for the
            # direction it is travelling, which is what makes a head-on
            # conflict a rule violation and not just bad luck.
            piece = self.world.net.pieces[self.world.net.project(veh.x, veh.y)[0]]
            forward = math.cos(wrap_pi(veh.heading - tangent)) >= 0.0
            if piece.lanes >= 2 and margin >= 0.0:
                wrong_side = (lateral > 0.0) if forward else (lateral < 0.0)
                costs["lane"][i] = float(wrong_side)

            ttc = self._time_to_collision(i, moving)
            if ttc < TTC_THRESHOLD:
                costs["ttc"][i] = 1.0 - ttc / TTC_THRESHOLD

            accel = (veh.vx - speed_before[i]) / self.dt
            jerk = abs(accel - self._accel_prev[i]) / self.dt
            self._accel_prev[i] = accel
            costs["jerk"][i] = max(0.0, jerk - JERK_THRESHOLD) / JERK_THRESHOLD

            speed = math.hypot(veh.vx, veh.vy)
            costs["speeding"][i] = max(0.0, speed - SPEED_LIMIT) / SPEED_LIMIT

            events.append(event)

        if self._steps >= self.max_steps:
            truncated[:] = True

        obs = self._observations()
        info = {"cost": costs, "events": events,
                "route": self._route_prev.copy()}

        # Respawn whatever finished, so density stays constant. Done AFTER
        # the observation is taken: the learner's last observation of an
        # episode must be the state the terminal flag refers to.
        for i in range(self.n_agents):
            if terminated[i] or truncated[i]:
                self._respawn(i)
        if truncated.any():
            self._steps = 0

        return obs, list(rewards), list(terminated), list(truncated), info

    def _zero_costs(self) -> dict:
        return {k: np.zeros(self.n_agents, dtype=np.float32)
                for k in ("collision", "offroad", "lane", "ttc", "jerk", "speeding")}

    def _time_to_collision(self, i: int, moving: np.ndarray) -> float:
        """Seconds to the nearest constant-velocity closing conflict.

        A straight-line extrapolation of both vehicles, treating each as a
        disc of the body's own circumscribed radius. It is not a prediction —
        neither vehicle will actually hold its velocity — but it is the
        standard surrogate-safety measure, and as a COST it only has to be
        monotone in danger, not accurate.
        """
        r = math.hypot(self.half_length, self.half_width) * 2.0
        rel_p = np.delete(moving[:, :2] - moving[i, :2], i, axis=0)
        rel_v = np.delete(moving[:, 2:] - moving[i, 2:], i, axis=0)
        if len(rel_p) == 0:
            return math.inf

        a = (rel_v * rel_v).sum(axis=1)
        b = 2.0 * (rel_p * rel_v).sum(axis=1)
        c = (rel_p * rel_p).sum(axis=1) - r * r
        disc = b * b - 4.0 * a * c
        ok = (a > 1e-6) & (disc > 0.0)
        if not ok.any():
            return math.inf
        t = (-b[ok] - np.sqrt(disc[ok])) / (2.0 * a[ok])
        t = t[t > 0.0]
        return float(t.min()) if len(t) else math.inf

    # -- observation ------------------------------------------------------

    def _observations(self) -> list:
        rects = self._rects()
        moving = self._moving()
        static_boxes = self.world.static_boxes
        out = []
        for i, veh in enumerate(self.vehicles):
            margin, lateral, tangent = self.world.road_state(veh.x, veh.y)
            goal = tuple(self._goals[i])
            (near, far), route = self.world.route_probe(
                (veh.x, veh.y), goal, (LOOKAHEAD_NEAR, LOOKAHEAD_FAR))

            # Other vehicles are obstacles to the lidar exactly as buildings
            # are; the beam does not know the difference and neither should
            # the observation.
            boxes = np.vstack([static_boxes, np.delete(rects, i, axis=0)])
            ranges = sensors.lidar((veh.x, veh.y), veh.heading, boxes,
                                   self.world.static_circles,
                                   n_rays=LIDAR_RAYS, max_range=LIDAR_RANGE)

            nf, nl = self._to_ego(veh, near)
            ff, fl = self._to_ego(veh, far)
            bend = math.atan2(far[1] - near[1], far[0] - near[0])

            out.append({
                "state": np.array([
                    veh.vx, veh.vy, veh.yaw_rate, veh.steering_angle,
                    float(veh.gear.value), margin, lateral,
                    wrap_pi(veh.heading - tangent),
                ], dtype=np.float32),
                "navigation": np.array([
                    route, nf, nl, ff, fl, wrap_pi(bend - veh.heading),
                ], dtype=np.float32),
                "lidar": ranges,
                "radar": sensors.radar((veh.x, veh.y), veh.heading,
                                       moving[i, 2:], np.delete(moving, i, axis=0),
                                       max_objects=RADAR_OBJECTS,
                                       max_range=LIDAR_RANGE),
            })
        return out

    @staticmethod
    def _to_ego(veh, point) -> tuple[float, float]:
        """A world point as (forward, left) in the vehicle's own frame."""
        dx = point[0] - veh.x
        dy = point[1] - veh.y
        cos, sin = math.cos(-veh.heading), math.sin(-veh.heading)
        return cos * dx - sin * dy, sin * dx + cos * dy

    # -- rendering --------------------------------------------------------

    def render(self, mode: str = "rgb_array", path: str | None = None):
        """Top-down PNG of the world and every vehicle in it. The 3D view
        lives in `envs/render3d.py` and takes the same World."""
        from envs import render_mpl
        return render_mpl.draw(self.world, vehicles=self._rects(),
                               goals=self._goals, path=path)
