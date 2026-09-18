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

# -- arrivals -------------------------------------------------------------
# Vehicles do not all appear at once. Spawning the full fleet at reset makes
# every conflict in the run a consequence of one simultaneous placement:
# the same cars meet at the same junctions at the same times, and a policy
# can learn that schedule instead of learning to drive. Staggering arrivals
# is what makes a conflict a property of the traffic rather than of the
# reset.
#
# `initial_active` of the fleet is on the road at reset; the rest arrive
# over the next `arrival_spread` seconds. After an episode ends the vehicle
# waits a further Exponential(`respawn_delay`) before returning, which is
# what keeps arrivals from re-synchronising into a convoy over a long run.
ARRIVAL_SPREAD = 25.0
RESPAWN_DELAY = 4.0
INITIAL_ACTIVE = 0.5

# -- observation shape ----------------------------------------------------
LIDAR_RAYS = 120
LIDAR_RANGE = 100.0
RADAR_OBJECTS = 5
# Two route waypoints: the near one says which way this lane runs, the far
# one is the steering target that reaches round the next corner.
LOOKAHEAD_NEAR = 8.0
LOOKAHEAD_FAR = 18.0

# The constrained-MDP cost channels, in one place: the env emits exactly
# these, one Lagrange multiplier and one budget `d_i` belongs to each, and a
# learner can enumerate them without hard-coding the names a second time.
#
# EVERY CHANNEL IS BOUNDED TO [0, 1] PER STEP. That is not tidiness, it is
# what makes a budget writable. An episode's cost J_c is the sum over its
# steps, so a bounded per-step cost gives J_c a unit anyone can read:
#
#   collision   events per episode        "<= 0.01 collisions"
#   offroad     step-equivalents off it   "<= 5 steps, i.e. 0.5 s"
#   wrong_way   step-equivalents          "<= 20 steps in the wrong lane"
#   lane_keep   step-equivalents at full  "<= 30 steps of maximum drift"
#   ttc         step-equivalents at zero  "<= 10 steps of imminent conflict"
#   jerk        step-equivalents at full  "<= 15 steps of maximum harshness"
#   speeding    step-equivalents at 2x    "<= 5 steps at double the limit"
#
# Unbounded channels break that. Before this was enforced, `jerk` reached
# 21.3 in a single step and totalled 2160 over a window where the whole
# reward totalled 4.4 — so lambda_jerk would have had to converge near 1e-3
# while lambda_collision sat near 1, and with six simultaneous constraints
# that spread is how plain dual ascent oscillates into a degenerate policy.
# The budget is also the number the paper claims is interpretable; a budget
# in units of "unclamped ramped jerk" is not.
COST_CHANNELS = ("collision", "offroad", "wrong_way", "lane_keep",
                 "ttc", "jerk", "speeding")

# -- cost thresholds (what counts as a violation, NOT what it is worth) ---
TTC_THRESHOLD = 2.0          # seconds; below this the TTC cost is charged
SPEED_LIMIT = 13.9           # m/s (50 km/h)

# Comfort. Jerk is measured from a SMOOTHED acceleration, not from the raw
# step-to-step difference, and the smoothing is the whole point: at a 0.1 s
# control period, differencing raw acceleration turns the controller's own
# quantisation into jerk. A 5 m/s^3 threshold on that trips whenever
# acceleration moves 0.5 m/s^2 in one step, which every 10 Hz policy does
# constantly — it was measuring the control rate, not the ride.
#
# The filter time constant is the shortest change a passenger actually feels
# as a jolt rather than as steady acceleration; the threshold is then a real
# comfort figure rather than a number chosen to make the cost quiet.
JERK_TAU = 0.3               # s, acceleration low-pass time constant
JERK_THRESHOLD = 2.5         # m/s^3 of FILTERED jerk


def batch_obs(observations) -> dict:
    """A list of per-agent observation dicts as one dict of stacked arrays.

    `{"lidar": (n_agents, 120), ...}` rather than `n_agents` dicts each
    holding four small arrays. Two places want this and for different
    reasons: a policy wants a batch to forward in one call, and
    `envs/vec.py` has to put observations through a pipe, where 4 arrays
    pickle far faster than 80.
    """
    if not observations:
        return {}
    return {key: np.stack([o[key] for o in observations])
            for key in observations[0]}


def _ramp(value: float, threshold: float) -> float:
    """A violation's severity as a number in [0, 1]: zero at the threshold,
    one at twice it, flat after that.

    Ramped rather than flat because a flat charge is as bad at 31 km/h as at
    90, which leaves a policy that has already overshot no reason to come
    back down. Clamped rather than open-ended for the reason above the
    channel list.
    """
    if threshold <= 0.0:
        return float(value > 0.0)
    return float(min(max(value - threshold, 0.0) / threshold, 1.0))


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
                 world: World | None = None,
                 arrival_spread: float = ARRIVAL_SPREAD,
                 respawn_delay: float = RESPAWN_DELAY,
                 initial_active: float = INITIAL_ACTIVE):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.dt = float(dt)
        self.n_agents = int(n_agents)
        self.max_steps = int(max_seconds / self.dt)
        self.vehicle_cls = vehicle_cls
        self.arrival_spread = float(arrival_spread)
        self.respawn_delay = float(respawn_delay)
        self.initial_active = float(initial_active)

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
            # [vx, vy, yaw_rate, steering_angle, gear, road_margin,
            #  heading_error, lane_offset, lane_index_norm, wrong_way]
            #
            # `lane_offset` is measured from the centre of the vehicle's OWN
            # lane, not from the road's centreline: on a three-lane arterial
            # the centreline offset of a correctly-driven vehicle is several
            # metres and carries no information about whether it is driving
            # well. Absolute x/y are deliberately absent — a policy given its
            # world coordinates memorises the map.
            "state": spaces.Box(-np.inf, np.inf, shape=(10,), dtype=np.float32),
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
        self._accel_filt = np.zeros(self.n_agents)
        # Per-agent, not global. With staggered arrivals every vehicle is at
        # a different point of its own episode, so one shared step counter
        # would time all of them out together — re-synchronising exactly what
        # the staggering is for.
        self._age = np.zeros(self.n_agents, dtype=int)
        self._active = np.zeros(self.n_agents, dtype=bool)
        self._wait = np.zeros(self.n_agents)     # seconds until arrival
        self._far = 0.0                          # where parked vehicles go

    # -- reset ------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._offroad_for[:] = 0.0
        self._accel_filt[:] = 0.0
        self._age[:] = 0
        self._active[:] = False

        # Park every vehicle far outside the world. This does double duty:
        # `_respawn` rejects spawns that overlap another vehicle, and
        # vehicles still at their constructor default would all be stacked on
        # the origin and veto every candidate near it — AND it is where a
        # vehicle waiting to arrive sits, so every geometry query (collision,
        # lidar, radar) keeps working on the full rectangular array without
        # any of them needing to know about the active mask.
        self._far = max(abs(c) for c in self.world.bounds()) + 500.0
        for k in range(self.n_agents):
            self._park(k)

        # Who is on the road at t=0, and when the rest turn up.
        n_now = int(round(self.n_agents * np.clip(self.initial_active, 0.0, 1.0)))
        order = self.rng.permutation(self.n_agents)
        self._wait[:] = self.rng.uniform(0.0, self.arrival_spread, self.n_agents)
        self._wait[order[:n_now]] = 0.0
        for i in order[:n_now]:
            self._arrive(int(i))

        return self._observations(), {"cost": self._zero_costs(),
                                      "active": self._active.copy()}

    def _park(self, i: int) -> None:
        """Take agent `i` off the road until its next arrival."""
        veh = self.vehicles[i]
        veh.x, veh.y, veh.heading = self._far + 10.0 * i, self._far, 0.0
        veh.vx = veh.vy = veh.yaw_rate = 0.0
        self._active[i] = False

    def _arrive(self, i: int) -> bool:
        """Try to put a waiting agent on the road. False when every candidate
        start was blocked, in which case it waits one more step and retries —
        a vehicle cannot be forced into a space another vehicle is in."""
        if self._respawn(i):
            self._active[i] = True
            self._age[i] = 0
            return True
        self._park(i)
        return False

    def _respawn(self, i: int) -> bool:
        """Put agent `i` at a fresh start with a fresh destination.

        Returns False when every attempt was blocked by another vehicle. The
        caller leaves it parked and tries again next step rather than forcing
        it in: with staggered arrivals there is always a later moment, and a
        vehicle materialising inside another one is a collision cost the
        policy was handed rather than earned.

        The first few attempts use the scenario's declared sources; once
        those are taken the rest of the traffic is placed anywhere on the
        network, because a layout declaring two approaches still has to hold
        twenty vehicles.

        The destination is drawn and the pose snapped to its lane INSIDE the
        retry loop, not after it. Which lane a spawn belongs in depends on
        which way its route runs, so the snap moves the vehicle — and a
        clearance test run before that move tests a position the vehicle
        does not end up in. With every spawn now landing exactly on a lane
        centre, that stale test let almost every vehicle spawn on top of
        another one.
        """
        veh = self.vehicles[i]
        others = np.delete(self._rects(), i, axis=0)
        placed = False
        for attempt in range(24):
            x, y, heading = self.world.sample_start(self.rng, declared=attempt < 6)
            gx, gy, route = self.world.sample_goal(self.rng, (x, y), MIN_ROUTE)
            x, y, heading = self._snap_to_lane(x, y, (gx, gy), heading)
            rect = (x, y, self.half_length, self.half_width, heading)
            clear = not self.world.rect_hits_rects(rect, others).any()
            # Re-ask the lane graph rather than trusting the snap. Near a
            # junction — a roundabout entry especially — the piece whose edge
            # is nearest after the move is not always the piece the snap
            # measured against, so a pose that was placed correctly on the
            # approach can read as wrong-way on the ring.
            if clear and not self.world.locate_lane(x, y, heading).wrong_way:
                placed = True
                break
        if not placed:
            return False

        veh.reset(spawn_point=(x, y), destination=(gx, gy))
        veh.heading = heading
        veh.gear = Gear.DRIVE
        self._goals[i] = (gx, gy)
        self._route0[i] = max(route, 1.0)
        self._route_prev[i] = route
        self._offroad_for[i] = 0.0
        self._accel_filt[i] = 0.0
        return True

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
        tx, ty = piece.tangent(s)
        forward = (math.cos(travel) * tx + math.sin(travel) * ty) >= 0.0
        lane = self.world.lanes.lane_for_travel(i, forward)
        # Lane offsets are LEFT-positive relative to the piece's tangent, so
        # step left of the tangent — not of the travel direction, which is
        # the opposite way round on a lane running against the piece.
        return (cx - ty * lane.offset, cy + tx * lane.offset, travel)

    # -- geometry helpers -------------------------------------------------

    def _rects(self) -> np.ndarray:
        """(n_agents, 5) footprints of every vehicle, this instant."""
        return np.array([(v.x, v.y, self.half_length, self.half_width, v.heading)
                         for v in self.vehicles], dtype=float)

    def vehicle_rects(self) -> np.ndarray:
        """Footprints of the vehicles actually ON the road — what a renderer
        wants. `_rects` keeps a row per agent, including the ones parked far
        outside the world, because the geometry queries index by agent."""
        return self._rects()[self._active]

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
        length, plus `info["cost"]` — a dict of (n_agents,) cost arrays — and
        `info["active"]`, a boolean mask.

        The arrays stay rectangular whether or not a vehicle is on the road,
        because a ragged interface would have to be un-ragged again by every
        caller. A waiting vehicle gets a zeroed observation, zero reward, no
        costs and no terminal flag; `info["active"]` says which rows those
        are, and the learner drops them from its batch. Its action is
        ignored, so a policy that keeps producing one costs nothing.
        """
        if len(actions) != self.n_agents:
            raise ValueError(f"expected {self.n_agents} actions, got {len(actions)}")

        speed_before = np.array([v.vx for v in self.vehicles])
        for veh, action, live in zip(self.vehicles, actions, self._active):
            if live:
                veh.step(action)
        self._age[self._active] += 1

        rects = self._rects()
        moving = self._moving()
        costs = self._zero_costs()
        rewards = np.zeros(self.n_agents, dtype=np.float32)
        terminated = np.zeros(self.n_agents, dtype=bool)
        truncated = np.zeros(self.n_agents, dtype=bool)
        events = []

        for i, veh in enumerate(self.vehicles):
            if not self._active[i]:
                events.append("")
                continue
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

            # Two separate lane rules, because they are two different
            # mistakes and a Lagrange multiplier per cost channel can only
            # price them separately if they arrive separately.
            #
            #   wrong_way   in a lane that runs the other way. A rule
            #               violation, and the thing that turns a near-miss
            #               into a head-on.
            #   lane_keep   drifting off the centre of whatever lane you ARE
            #               in. Ramped by how far, and only charged past a
            #               dead band — a vehicle tracking its lane to within
            #               a few tens of centimetres is driving, not
            #               violating, and charging it there would make the
            #               constraint bind on noise.
            fix = self.world.locate_lane(veh.x, veh.y, veh.heading)
            if margin >= 0.0:
                costs["wrong_way"][i] = float(fix.wrong_way)
                # Saturating at one slack-width past the dead band is not a
                # loss: by then the vehicle is most of a lane out, and
                # whatever it does next is already being charged as
                # wrong_way or offroad.
                slack = max(fix.lane.width / 2.0 - self.half_width, 0.25)
                costs["lane_keep"][i] = _ramp(abs(fix.offset), slack)

            ttc = self._time_to_collision(i, moving)
            if ttc < TTC_THRESHOLD:
                costs["ttc"][i] = 1.0 - ttc / TTC_THRESHOLD

            accel = (veh.vx - speed_before[i]) / self.dt
            alpha = self.dt / (JERK_TAU + self.dt)
            filtered = self._accel_filt[i] + alpha * (accel - self._accel_filt[i])
            jerk = abs(filtered - self._accel_filt[i]) / self.dt
            self._accel_filt[i] = filtered
            costs["jerk"][i] = _ramp(jerk, JERK_THRESHOLD)

            speed = math.hypot(veh.vx, veh.vy)
            costs["speeding"][i] = _ramp(speed, SPEED_LIMIT)

            events.append(event)

        truncated = self._active & (self._age >= self.max_steps)

        obs = self._observations()
        info = {"cost": costs, "events": events, "active": self._active.copy(),
                "route": self._route_prev.copy()}

        # Retire whatever finished. Done AFTER the observation is taken: the
        # learner's last observation of an episode must be the state the
        # terminal flag refers to.
        for i in range(self.n_agents):
            if terminated[i] or truncated[i]:
                self._park(i)
                self._wait[i] = self.rng.exponential(self.respawn_delay)

        # Then bring in whoever is due. An Exponential wait re-rolled on every
        # retirement is what stops arrivals settling into a convoy: a fixed
        # delay would have the whole fleet keep whatever spacing the first
        # round of collisions happened to give it.
        for i in np.flatnonzero(~self._active):
            self._wait[i] -= self.dt
            if self._wait[i] <= 0.0:
                self._arrive(int(i))

        return obs, list(rewards), list(terminated), list(truncated), info

    def _zero_costs(self) -> dict:
        return {k: np.zeros(self.n_agents, dtype=np.float32)
                for k in COST_CHANNELS}

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
            if not self._active[i]:
                # A zeroed observation, not a computed one. A waiting vehicle
                # is parked far outside the world, so its lidar would be 120
                # rays of max_range and its route a straight line across
                # empty space — the expensive way to compute nothing.
                out.append(self._blank_observation())
                continue
            margin, lateral, tangent = self.world.road_state(veh.x, veh.y)
            goal = tuple(self._goals[i])
            # Lane-aware, not centreline: steering straight at a centreline
            # waypoint is steering into the oncoming lane on a two-way road.
            (near, far), route = self.world.lane_waypoints(
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

            fix = self.world.locate_lane(veh.x, veh.y, veh.heading)
            n_lanes = len(self.world.lanes.lanes(fix.piece))
            lane_norm = fix.lane.index / max(n_lanes - 1, 1)

            out.append({
                "state": np.array([
                    veh.vx, veh.vy, veh.yaw_rate, veh.steering_angle,
                    float(veh.gear.value), margin,
                    wrap_pi(veh.heading - tangent),
                    fix.offset, lane_norm, float(fix.wrong_way),
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
    def _blank_observation() -> dict:
        """The observation of a vehicle that is not on the road. Zeros, and
        the lidar at full range — the shape the space promises, carrying no
        claim about a world this vehicle is not in."""
        return {"state": np.zeros(10, dtype=np.float32),
                "navigation": np.zeros(6, dtype=np.float32),
                "lidar": np.full(LIDAR_RAYS, LIDAR_RANGE, dtype=np.float32),
                "radar": np.zeros((RADAR_OBJECTS, 4), dtype=np.float32)}

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
        return render_mpl.draw(self.world, vehicles=self.vehicle_rects(),
                               goals=self._goals[self._active], path=path)
