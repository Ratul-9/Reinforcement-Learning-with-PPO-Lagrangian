"""Self-check for the env stack. Run it after touching anything in envs/.

    conda run -n py310 python test_envs.py

Asserts the things that are expensive to notice later: that every scenario
builds and steps, that each cost channel can actually fire (a cost that never
fires is a constraint the Lagrange multiplier will drive to zero and a result
that quietly means nothing), that the lidar agrees with the geometry it is
cast against, and that the two heading conventions round-trip.
"""

import math

import numpy as np

from envs import road_network, sensors
from envs.world import World, h_to_rad, rad_to_h
from envs.traffic_env import TrafficEnv, COST_CHANNELS
from vehicle import Gear


def bare(kind="cross", seed=0):
    """A world with no bays and no scenery — a plain road to stage a single
    cost on. Bays are one lane wide and therefore bidirectional, so a vehicle
    that spawns in one can never be wrong-way, and a test that wants to
    provoke that cost has to start on a real two-way road."""
    return World.build(kind, rng=np.random.default_rng(seed),
                       scenery_density=0.0, bays=False)


ALL_ON = dict(initial_active=1.0, arrival_spread=0.0, respawn_delay=0.0,
              vehicle_types="sedan")
"""Every vehicle on the road at reset and back instantly after it finishes.

Staggered arrivals and a mixed fleet are the defaults and are what a
training run wants, but a test that provokes one specific cost on one
specific vehicle cannot also be waiting to find out whether that vehicle
exists yet, nor discovering that this time it drew a motorcycle whose
footprint, braking and top speed are all different."""


def drive(throttle=0.6, steering=0.0, brake=0.0, gear=3):
    return {"steering": np.array([steering], dtype=np.float32),
            "throttle": np.array([throttle], dtype=np.float32),
            "brake": np.array([brake], dtype=np.float32),
            "gear": gear}


def test_headings_round_trip():
    for deg in (-180.0, -90.0, 0.0, 45.0, 179.0):
        assert abs(rad_to_h(h_to_rad(deg)) - deg) < 1e-9, deg
    # The convention itself: H = -90 is travel along +X.
    assert abs(h_to_rad(-90.0)) < 1e-9


def test_every_scenario_builds():
    for kind in road_network.KINDS:
        world = World.build(kind, rng=np.random.default_rng(0))
        assert len(world.net.pieces) > 0, kind
        x0, y0, x1, y1 = world.bounds()
        assert x1 > x0 and y1 > y0, kind
    for kind in road_network.SCENARIO_KINDS:
        net = World.build(kind).net
        assert net.sources and net.goals, f"{kind} declares no source/goal"


def test_scenery_is_off_the_road():
    world = World.build("manhattan", rng=np.random.default_rng(3))
    assert len(world.scenery.boxes) > 0 and len(world.scenery.trees) > 0
    for cx, cy, *_ in world.scenery.boxes:
        assert not world.net.is_on_road(float(cx), float(cy)), "building on the road"
    for cx, cy, *_ in world.scenery.trees:
        assert not world.net.is_on_road(float(cx), float(cy)), "tree on the road"


def test_lidar_matches_geometry():
    # One box 20 m dead ahead, one tree 10 m to the left, nothing else.
    boxes = np.array([[20.0, 0.0, 2.0, 2.0, 0.0]])
    circles = np.array([[0.0, 10.0, 1.0]])
    r = sensors.lidar((0.0, 0.0), 0.0, boxes, circles, n_rays=4, fov=2 * math.pi,
                      max_range=100.0)
    # A 4-ray full sweep runs -180, -90, 0, +90 relative to the heading.
    assert abs(r[0] - 100.0) < 1e-3, r          # behind: nothing
    assert abs(r[1] - 100.0) < 1e-3, r          # right: nothing
    assert abs(r[2] - 18.0) < 1e-3, r           # ahead: box face at 20-2
    assert abs(r[3] - 9.0) < 1e-3, r            # left: tree surface at 10-1


def test_overlap_tests():
    world = bare()
    a = (0.0, 0.0, 2.5, 1.0, 0.0)
    # Overlapping, touching-but-rotated, and clear.
    near = np.array([[3.0, 0.0, 2.5, 1.0, 0.0],
                     [0.0, 0.0, 2.5, 1.0, math.pi / 2],
                     [20.0, 0.0, 2.5, 1.0, 0.0]])
    hits = world.rect_hits_rects(a, near)
    assert list(hits) == [True, True, False], hits
    circles = np.array([[2.0, 0.0, 1.0], [30.0, 0.0, 1.0]])
    assert list(world.rect_hits_circles(a, circles)) == [True, False]


def test_offroad_cost_fires():
    """Drive straight off the side of the road and the off-road cost must
    charge, then the episode must end once the grace window is used up."""
    env = TrafficEnv("cross", n_agents=1, seed=0, world=bare(), **ALL_ON)
    env.reset()
    veh = env.vehicles[0]
    # Point it across the carriageway rather than along it.
    _, _, tangent = env.world.road_state(veh.x, veh.y)
    veh.heading = tangent + math.pi / 2
    charged = ended = False
    for _ in range(60):
        _, _, term, _, info = env.step([drive(throttle=1.0)])
        charged |= info["cost"]["offroad"][0] > 0.0
        if term[0]:
            ended = True
            break
    assert charged, "off-road cost never fired"
    assert ended, "off-road episode never terminated"


def test_collision_cost_fires():
    """Two vehicles placed nose to nose must register a collision."""
    env = TrafficEnv("cross", n_agents=2, seed=0, world=bare(), **ALL_ON)
    env.reset()
    a, b = env.vehicles
    b.x, b.y, b.heading = a.x + 1.0, a.y, a.heading
    _, _, term, _, info = env.step([drive(throttle=0.0), drive(throttle=0.0)])
    assert info["cost"]["collision"].sum() == 2.0, info["cost"]["collision"]
    assert all(term)


def test_progress_reward_is_signed_by_direction():
    """Driving along the route must pay; driving away from it must not."""
    env = TrafficEnv("cross", n_agents=1, seed=2, world=bare(seed=2), **ALL_ON)
    env.reset()
    veh = env.vehicles[0]
    goal = tuple(env._goals[0])
    (waypoint,), _ = env.world.route_probe((veh.x, veh.y), goal, (10.0,))
    veh.heading = math.atan2(waypoint[1] - veh.y, waypoint[0] - veh.x)
    veh.gear = Gear.DRIVE
    toward = sum(env.step([drive(throttle=1.0)])[1][0] for _ in range(20))

    env.reset()
    veh = env.vehicles[0]
    goal = tuple(env._goals[0])
    (waypoint,), _ = env.world.route_probe((veh.x, veh.y), goal, (10.0,))
    veh.heading = math.atan2(waypoint[1] - veh.y, waypoint[0] - veh.x) + math.pi
    veh.gear = Gear.DRIVE
    away = sum(env.step([drive(throttle=1.0)])[1][0] for _ in range(20))

    assert toward > 0.0, f"driving toward the goal paid {toward}"
    assert toward > away, f"toward={toward} away={away}"


def test_every_scenario_steps_with_traffic():
    for kind in road_network.SCENARIO_KINDS:
        env = TrafficEnv(kind, n_agents=12, seed=1)
        obs, info = env.reset()
        assert info["cost"]["collision"].sum() == 0.0, f"{kind} spawns in collision"
        assert len(obs) == 12
        for _ in range(10):
            obs, rew, term, trunc, info = env.step(
                [env.action_space.sample() for _ in range(12)])
        for key, value in info["cost"].items():
            assert value.shape == (12,), (kind, key)
            assert np.all(value >= 0.0), f"{kind}: {key} went negative"
        assert np.all(np.isfinite(obs[0]["lidar"]))


def test_lane_graph_layout():
    """Lane centres, directions and dividers must follow from the lane count
    alone — that is what lets a PNG-loaded network have lanes without the
    loader knowing lanes exist."""
    from envs.lanes import lanes_of

    class FakePiece:
        def __init__(self, lanes, half_width):
            self.lanes, self.half_width = lanes, half_width

    one = lanes_of(FakePiece(1, 1.75))
    assert len(one) == 1 and one[0].offset == 0.0
    assert one[0].direction == 0, "a single-lane bay has no wrong side"

    two = lanes_of(FakePiece(2, 3.5))
    assert [ln.offset for ln in two] == [1.75, -1.75]
    assert [ln.direction for ln in two] == [-1, 1], "right-hand traffic"

    three = lanes_of(FakePiece(3, 5.25))
    assert [ln.direction for ln in three] == [-1, 0, 1], "odd lane is shared"
    assert abs(three[1].offset) < 1e-9

    world = World.build("intersection_x", rng=np.random.default_rng(0))
    for i, piece in enumerate(world.net.pieces):
        assert len(world.lanes.dividers(i)) == piece.lanes - 1


def test_spawns_are_lane_centred_and_right_way():
    """Every spawn must land on a lane centre, facing the way that lane runs,
    and clear of the vehicles already placed. All three at once: the lane
    snap MOVES the vehicle, so a clearance test taken before it is testing a
    pose the vehicle never occupies."""
    from envs.traffic_env import TrafficEnv as Env

    for kind in road_network.SCENARIO_KINDS:
        env = Env(kind, n_agents=20, seed=0, **ALL_ON)
        _, info = env.reset()
        assert info["cost"]["collision"].sum() == 0.0, f"{kind} spawns in collision"
        assert info["active"].any(), f"{kind} put nobody on the road"
        for veh, live in zip(env.vehicles, info["active"]):
            if not live:
                continue
            fix = env.world.locate_lane(veh.x, veh.y, veh.heading)
            assert not fix.wrong_way, f"{kind} spawns against the traffic"
            assert abs(fix.offset) < 0.5, f"{kind} spawns off its lane centre"


def test_wrong_way_cost_fires():
    """Turn a vehicle round on a two-way road and the wrong-way cost must
    charge while the lane-keeping cost does not — they are separate
    mistakes and each gets its own multiplier."""
    from envs.traffic_env import TrafficEnv as Env

    env = Env("cross", n_agents=1, seed=0, world=bare(), **ALL_ON)
    env.reset()
    veh = env.vehicles[0]
    veh.heading += math.pi                       # same lane, facing back
    _, _, _, _, info = env.step([drive(throttle=0.0)])
    assert info["cost"]["wrong_way"][0] == 1.0, "wrong-way cost never fired"
    assert info["cost"]["lane_keep"][0] == 0.0, "still centred in its lane"


def test_lane_keep_cost_fires():
    """Slide a vehicle off its lane centre, staying on the road, and the
    lane-keeping cost must charge and grow with the error."""
    from envs.traffic_env import TrafficEnv as Env

    env = Env("cross", n_agents=1, seed=0, world=bare(), **ALL_ON)
    env.reset()
    veh = env.vehicles[0]
    fix = env.world.locate_lane(veh.x, veh.y, veh.heading)
    piece = env.world.net.pieces[fix.piece]
    tx, ty = piece.tangent(fix.s)
    x0, y0 = veh.x, veh.y
    # Drift toward the road's centreline, not away from it: the spawn lane
    # may be either side, and drifting outward would leave the carriageway
    # and charge the off-road cost instead of the one under test.
    inward = -1.0 if fix.lane.offset > 0 else 1.0

    charged = []
    for shift in (0.0, 1.1, 1.6):
        shift *= inward
        veh.x, veh.y = x0 - ty * shift, y0 + tx * shift
        _, _, _, _, info = env.step([drive(throttle=0.0)])
        assert info["cost"]["offroad"][0] == 0.0, "drifted clean off the road"
        charged.append(float(info["cost"]["lane_keep"][0]))
    assert charged[0] == 0.0, "a centred vehicle was charged"
    assert charged[2] > charged[1] > 0.0, f"cost did not ramp: {charged}"


def test_speeding_cost_fires():
    """Over the limit must charge, and charge more the further over."""
    from envs.traffic_env import SPEED_LIMIT

    env = TrafficEnv("cross", n_agents=1, seed=0, world=bare(), **ALL_ON)
    env.reset()
    veh = env.vehicles[0]
    charged = []
    for speed in (SPEED_LIMIT * 0.9, SPEED_LIMIT * 1.2, SPEED_LIMIT * 1.5):
        veh.vx = speed
        _, _, _, _, info = env.step([drive(throttle=0.0)])
        charged.append(float(info["cost"]["speeding"][0]))
    assert charged[0] == 0.0, "charged under the limit"
    assert charged[2] > charged[1] > 0.0, f"cost did not ramp: {charged}"


def test_ttc_cost_fires():
    """Closing head-on with a stationary vehicle must charge the TTC cost,
    and charge more the closer the conflict is."""
    env = TrafficEnv("cross", n_agents=2, seed=0, world=bare(), **ALL_ON)
    env.reset()
    a, b = env.vehicles
    charged = []
    for gap in (60.0, 25.0, 14.0):
        a.vx = 10.0
        b.x = a.x + math.cos(a.heading) * gap
        b.y = a.y + math.sin(a.heading) * gap
        b.heading, b.vx = a.heading, 0.0
        _, _, _, _, info = env.step([drive(throttle=0.0), drive(throttle=0.0)])
        charged.append(float(info["cost"]["ttc"][0]))
    assert charged[0] == 0.0, "charged for a conflict six seconds away"
    assert charged[2] > charged[1] > 0.0, f"cost did not ramp: {charged}"


def test_jerk_cost_fires():
    """A hard throttle-to-brake reversal must charge the comfort cost."""
    env = TrafficEnv("cross", n_agents=1, seed=0, world=bare(), **ALL_ON)
    env.reset()
    env.vehicles[0].vx = 8.0
    charged = 0.0
    for action in (drive(throttle=1.0), drive(throttle=0.0, brake=1.0),
                   drive(throttle=1.0)):
        _, _, _, _, info = env.step([action])
        charged = max(charged, float(info["cost"]["jerk"][0]))
    assert charged > 0.0, "jerk cost never fired on a throttle-brake reversal"


def test_arrivals_are_staggered():
    """Vehicles must trickle in rather than all appearing at reset, and a
    waiting vehicle must be inert: no reward, no cost, no terminal flag, and
    invisible to everyone else's sensors."""
    env = TrafficEnv("manhattan", n_agents=20, seed=0,
                     initial_active=0.5, arrival_spread=20.0)
    obs, info = env.reset()
    at_reset = int(info["active"].sum())
    assert 8 <= at_reset <= 12, f"{at_reset} active at reset, expected about half"

    seen = {at_reset}
    for _ in range(300):
        obs, rew, term, trunc, info = env.step(
            [env.action_space.sample() for _ in range(20)])
        live = info["active"]
        seen.add(int(live.sum()))
        idle = ~live
        assert not np.any(np.array(rew)[idle]), "a parked vehicle earned reward"
        assert not np.any(np.array(term)[idle]), "a parked vehicle terminated"
        for channel in info["cost"].values():
            assert not np.any(channel[idle]), "a parked vehicle was charged"
        for i in np.flatnonzero(idle):
            assert obs[i]["state"].sum() == 0.0, "a parked vehicle observed itself"

    assert max(seen) > at_reset, "nobody ever arrived after reset"
    # Parked vehicles must not be visible to the ones that are driving: they
    # are held far outside the world, and a lidar that could see them would
    # be reporting a wall where there is none.
    driving = np.flatnonzero(info["active"])
    assert obs[driving[0]]["lidar"].max() <= 100.0


def test_all_on_mode_disables_staggering():
    """The knob the tests and any fixed-density experiment rely on."""
    env = TrafficEnv("cross", n_agents=6, seed=0, world=bare(), **ALL_ON)
    _, info = env.reset()
    assert info["active"].all(), "initial_active=1.0 left someone waiting"


def test_route_waypoints_are_in_lane():
    """A route waypoint must sit in the lane the route runs in, not on the
    centreline between two of them.

    A vehicle steering at a centreline waypoint drives down the middle of
    the carriageway, which on a two-way road is the oncoming lane. That is
    not cosmetic: it made `wrong_way` fire on a third of all steps by
    construction, before the policy had done anything wrong, and a
    constraint violated that often at initialisation is what drives plain
    dual ascent into oscillation.
    """
    world = bare("manhattan")
    rng = np.random.default_rng(0)
    x, y, _h = world.sample_start(rng, declared=False)
    gx, gy, _route = world.sample_goal(rng, (x, y), 80.0)

    lookaheads = (8.0, 18.0, 30.0)
    centre, _ = world.route_probe((x, y), (gx, gy), lookaheads)
    inlane, _ = world.lane_waypoints((x, y), (gx, gy), lookaheads)

    moved = 0
    for (cx, cy), (lx, ly) in zip(centre, inlane):
        i, s, _lat, _d = world.net.project(cx, cy)
        if len(world.lanes.lanes(i)) < 2:
            continue                      # a one-lane bay has no lane to pick
        assert math.hypot(lx - cx, ly - cy) > 0.5, "waypoint stayed on the centreline"
        assert world.net.is_on_road(lx, ly), "waypoint was shifted off the road"
        moved += 1
    assert moved, "no multi-lane waypoint on this route to check"


def test_costs_are_bounded():
    """Every channel must stay in [0, 1] per step, under any behaviour.

    This is what makes a budget writable: with a bounded per-step cost, the
    episode sum J_c has a unit a human can read ("<= 5 steps off the road"),
    and no single channel's multiplier has to live orders of magnitude away
    from the others. An unbounded channel silently reintroduces exactly the
    arbitrary weighting the method exists to remove.
    """
    env = TrafficEnv("manhattan", n_agents=12, seed=3, **ALL_ON)
    obs, _ = env.reset()
    peak = {k: 0.0 for k in COST_CHANNELS}
    for _ in range(200):
        obs, _r, _t, _tr, info = env.step(
            [env.action_space.sample() for _ in range(12)])
        for k in COST_CHANNELS:
            channel = info["cost"][k]
            assert np.all(channel >= 0.0), f"{k} went negative"
            assert np.all(channel <= 1.0), f"{k} exceeded 1.0: {channel.max()}"
            peak[k] = max(peak[k], float(channel.max()))
    # No assertion that a particular channel fires here — this test is about
    # the BOUND. Random actions mostly leave vehicles parked, and once the
    # integrator was sub-stepped they stopped generating jerk at all, which
    # was the point: that jerk had been integration noise. Each channel has
    # its own firing test.
    assert any(v > 0.0 for v in peak.values()), "no cost fired at all"


def test_jerk_ignores_control_rate():
    """Jerk must be measured from a SMOOTHED acceleration.

    Held full throttle is one continuous acceleration and must not be
    charged, however fast the controller is ticking. Differencing raw
    step-to-step acceleration charged it anyway — it was measuring the 10 Hz
    control period rather than the ride.
    """
    env = TrafficEnv("cross", n_agents=1, seed=0, world=bare(), **ALL_ON)
    env.reset()
    steady = 0.0
    for _ in range(30):
        _, _, _, _, info = env.step([drive(throttle=0.35)])
        steady = max(steady, float(info["cost"]["jerk"][0]))
    assert steady < 0.5, f"steady acceleration charged as jerk: {steady}"


def test_vec_env_runs_mixed_scenarios():
    """Workers must run different scenarios at once and hand back batched
    observations — that is what lets a shared policy see a roundabout and a
    merge in the same update instead of overfitting one layout per run."""
    from envs.vec import VecTrafficEnv, performance_cores

    assert performance_cores() >= 1

    with VecTrafficEnv(["manhattan", "merge_ramp"], n_agents=6, seed=0) as vec:
        obs = vec.reset()
        assert len(obs) == 2
        assert obs[0]["lidar"].shape == (6, 120), obs[0]["lidar"].shape
        assert obs[0]["state"].shape == (6, 10)

        actions = [[vec.action_space.sample() for _ in range(6)]
                   for _ in range(2)]
        obs, rew, term, trunc, info = vec.step(actions)
        assert len(rew) == 2 and len(rew[0]) == 6
        assert set(info[0]["cost"]) == set(COST_CHANNELS)
        assert info[0]["active"].shape == (6,)


def test_batch_obs_round_trips():
    """`batch_obs` must stack a single env's observations into the same
    shape the vectorised wrapper sends, so both paths feed a policy the same
    thing."""
    from envs.traffic_env import batch_obs

    env = TrafficEnv("cross", n_agents=4, seed=0, world=bare(), **ALL_ON)
    obs, _ = env.reset()
    batched = batch_obs(obs)
    assert batched["lidar"].shape == (4, 120)
    assert batched["radar"].shape == (4, 5, 4)
    assert np.array_equal(batched["state"][2], obs[2]["state"])


def test_ttc_reports_imminent_conflicts():
    """The surrogate must report the dangerous cases, not just the tidy ones.

    A stationary vehicle three metres ahead used to come back as `inf`: with
    the pair already inside the conflict disc the quadratic's near root goes
    negative, and filtering to positive roots threw away exactly the
    geometry that matters. A separating pair must still be `inf` rather than
    the time it would have met travelling the other way.
    """
    env = TrafficEnv("cross", n_agents=2, seed=0, world=bare(), **ALL_ON)
    env.reset()
    a, b = env.vehicles

    def ttc(gap, lead_speed):
        a.vx = 10.0
        b.heading, b.vx = a.heading, lead_speed
        b.x = a.x + math.cos(a.heading) * gap
        b.y = a.y + math.sin(a.heading) * gap
        return env._time_to_collision(0, env._moving())

    assert ttc(3.0, 0.0) == 0.0, "missed a stationary vehicle three metres ahead"
    assert ttc(30.0, 15.0) == math.inf, "charged for a vehicle pulling away"
    assert ttc(30.0, 0.0) == math.inf or 2.0 < ttc(30.0, 0.0) < 3.0
    near, far = ttc(20.0, 5.0), ttc(40.0, 5.0)
    assert far > near > 0.0, f"not monotone in distance: {near}, {far}"


def test_metrics_recorder():
    """The recorder must close one episode per terminal flag and compare
    realised cost against the budget the env reported."""
    from envs.metrics import EpisodeRecorder

    # The scripted driver, not random actions: a random gear is DRIVE one
    # time in four, so a random fleet mostly sits still and no episode ever
    # ends inside a short test.
    from rollout import pure_pursuit

    env = TrafficEnv("manhattan", n_agents=12, seed=1)
    obs, _ = env.reset()
    rec = EpisodeRecorder(dt=env.dt)
    for _ in range(400):
        obs, rew, term, trunc, info = env.step([pure_pursuit(o) for o in obs])
        rec.update(info, term, trunc, rewards=rew)

    assert rec.episodes, "no episodes closed in 40 seconds of traffic"
    summary = rec.summary()
    rates = sum(summary[f"rate_{name}"] for name in
                ("goal", "collision_vehicle", "collision_static",
                 "offroad", "timeout"))
    assert abs(rates - 1.0) < 1e-9, f"outcomes do not partition: {rates}"

    rows = rec.constraints()
    assert rows, "no constraint rows"
    for row in rows:
        assert row["channel"] in COST_CHANNELS
        assert row["cost"] >= 0.0
        assert row["budget"] > 0.0, "budget did not reach the recorder"
    assert "episodes" in rec.report()


def test_layout_varies_per_reset():
    """Successive resets must give geometrically different maps, and a
    policy-visible difference rather than just different scenery.

    A fixed layout is a map a policy can memorise: where the junctions are,
    which way the roads run. Re-rolling the continuous dimensions is what
    leaves driving as the only transferable thing to learn.
    """
    env = TrafficEnv("manhattan", n_agents=8, seed=0,
                     layout_jitter=0.15, blockages=4)
    extents, blocked = set(), set()
    for _ in range(4):
        _, info = env.reset()
        extents.add(round(env.world.net.extent(), 2))
        blocked.add(tuple(np.round(env.world.blockages[:, 0], 2)))
        assert info["cost"]["collision"].sum() == 0.0, "spawned into a blockage"
        assert len(env.world.blockages) == 4
    assert len(extents) == 4, f"layout repeated: {extents}"
    assert len(blocked) == 4, "blockages landed in the same places"


def test_blockages_are_on_the_road_and_seen():
    """A stalled vehicle must sit ON the carriageway — that is what makes it
    different from scenery — and must be visible to the lidar, because the
    route will happily steer straight through it."""
    world = World.build("manhattan", rng=np.random.default_rng(2),
                        scenery_density=0.0, blockages=6)
    assert len(world.blockages) == 6
    for cx, cy, *_ in world.blockages:
        assert world.net.is_on_road(float(cx), float(cy)), "blockage off the road"

    # Cast from a few metres back along the same piece; something must be
    # closer than open ground.
    cx, cy, _hl, _hw, yaw = world.blockages[0]
    origin = (float(cx) - math.cos(yaw) * 20.0, float(cy) - math.sin(yaw) * 20.0)
    ranges = sensors.lidar(origin, float(yaw), world.static_boxes,
                           world.static_circles, n_rays=72, max_range=100.0)
    assert ranges.min() < 25.0, "lidar cannot see a stalled vehicle 20 m ahead"


def test_dynamics_match_theory():
    """The vehicle model must agree with theory where theory is unambiguous.

    Two regimes, and a correct model has to hit both:

    * **Low speed** — the vehicle turns on geometry. Steady-state yaw rate is
      the bicycle value `vx / L * tan(delta)`, and body slip at the CG is
      `atan(lr * tan(delta) / L)` (non-zero, and not a defect: the CG is not
      on the rear axle).
    * **High speed** — the tyres run out. Yaw rate is capped by grip at
      `mu * g / vx`, so a vehicle *must* understeer rather than follow the
      kinematic value, which at 20 m/s and a third of lock would demand 2.7 g.

    This caught a real bug. Explicit Euler on the stiff lateral dynamics
    needs `dt * (Cf + Cr) / (m * vx) < 2` — vx > 6.7 m/s at a 0.1 s step —
    and the friction clip bounded the resulting instability into a limit
    cycle instead of a blow-up, so it read as plausible noise. At 3 m/s the
    tuktuk oscillated between -0.19 and +1.16 rad/s for a constant steering
    input. Sub-stepping the integration fixed it; this test is what stops it
    coming back.
    """
    from vehicle import Vehicle, Gear
    from fleet import FLEET

    def hold(spec, vx, steer_cmd, steps=80):
        veh = Vehicle(spec=spec, dt=0.1)
        veh.vx, veh.vy, veh.yaw_rate, veh.heading = vx, 0.0, 0.0, 0.0
        veh.gear = Gear.DRIVE
        trace = []
        for _ in range(steps):
            veh.step({"steering": np.array([steer_cmd], np.float32),
                      "throttle": np.array([0.0], np.float32),
                      "brake": np.array([0.0], np.float32), "gear": 3})
            veh.vx = vx                      # hold speed, isolate lateral
            trace.append(veh.yaw_rate)
        return veh, np.array(trace)

    for name in ("sedan", "bus", "tuktuk", "truck", "motorcycle"):
        spec = FLEET[name]
        for vx in (1.5, 3.0, 5.0):
            veh, trace = hold(spec, vx, 0.3)
            delta = veh.steering_angle

            # Settled, not ringing: the last quarter must be flat.
            spread = float(np.ptp(trace[-20:]))
            assert spread < 0.02, \
                f"{name} at {vx} m/s oscillates: spread {spread:.3f} rad/s"

            r_kin = vx / spec.wheelbase * math.tan(delta)
            assert abs(veh.yaw_rate - r_kin) < 0.12 * abs(r_kin) + 0.02, \
                f"{name} at {vx} m/s: yaw {veh.yaw_rate:.3f} vs kinematic {r_kin:.3f}"

            slip_expected = math.atan(spec.lr * math.tan(delta) / spec.wheelbase)
            slip = math.atan2(veh.vy, vx)
            assert abs(slip - slip_expected) < math.radians(4.0), \
                f"{name} at {vx} m/s: body slip {math.degrees(slip):.1f} deg " \
                f"vs expected {math.degrees(slip_expected):.1f}"

    # High speed: grip-limited, and the limit is the right number.
    veh, _ = hold(FLEET["sedan"], 20.0, 0.3)
    lateral_g = 20.0 * abs(veh.yaw_rate)
    assert lateral_g <= FLEET["sedan"].effective_mu * 9.81 * 1.1, \
        f"cornering at {lateral_g:.1f} m/s^2 exceeds available grip"
    assert lateral_g > 0.6 * FLEET["sedan"].effective_mu * 9.81, \
        f"only {lateral_g:.1f} m/s^2 of a possible " \
        f"{FLEET['sedan'].effective_mu * 9.81:.1f} — not using the tyres"


def test_rear_axle_carries_its_own_load():
    """Each tyre is limited by the load on ITS axle.

    Fz_r was computed and then never used: the rear tyres were clipped at the
    front axle's limit, which on a front-heavy car is ~33% more rear grip
    than physics allows, and quietly biases every vehicle toward understeer.
    """
    from vehicle import Vehicle
    from fleet import SEDAN

    veh = Vehicle(spec=SEDAN)
    front_load = veh.mass * veh.gravity * veh.lr / veh.wheelbase
    rear_load = veh.mass * veh.gravity * veh.lf / veh.wheelbase
    assert abs(front_load + rear_load - veh.mass * veh.gravity) < 1.0
    assert front_load != rear_load, "a symmetric CG cannot detect this bug"

    import inspect
    source = inspect.getsource(Vehicle._advance)
    assert "Fz_rear_wheel" in source and "max_rear" in source, \
        "rear tyres are not clipped against the rear axle load"


def test_progress_cannot_exceed_distance_travelled():
    """The progress reward must never pay more than the vehicle moved.

    `route_distance` re-solved the shortest path from the vehicle's
    projection every step, and that answer jumps when the projection
    switches piece — at a junction, a bay mouth, anywhere two pieces
    overlap. Measured on manhattan with 16 agents: 257 steps where the route
    distance moved further than the vehicle did, largest +255 m. On a 150 m
    route that single step pays 1.7, where arriving pays 1.0. A policy finds
    that long before a human notices it.

    Fixed twice over, because the two halves fail differently: the route is
    now solved once per episode and walked (which also stops the observation
    and the goal test jumping, which a reward clamp cannot), and the reward
    is clamped to distance driven (which catches a vehicle re-attaching to
    its route after being shoved off it).
    """
    from rollout import pure_pursuit

    env = TrafficEnv("manhattan", n_agents=16, seed=1)
    obs, info = env.reset()
    fresh = np.ones(env.n_agents, dtype=bool)
    prev = np.array([(v.x, v.y) for v in env.vehicles])

    worst = 0.0
    for _ in range(300):
        obs, reward, term, trunc, info = env.step([pure_pursuit(o) for o in obs])
        now = np.array([(v.x, v.y) for v in env.vehicles])
        for i in range(env.n_agents):
            # Terminal steps are excluded because the env respawns inside
            # step(), so _route0 already belongs to the next episode.
            if not info["active"][i] or fresh[i] or term[i] or trunc[i]:
                continue
            moved = math.hypot(*(now[i] - prev[i]))
            gained = abs(float(reward[i]) * env._route0[i])
            worst = max(worst, gained - moved)
        fresh = np.array(term) | np.array(trunc) | ~np.array(info["active"])
        prev = now

    assert worst < 0.5, f"unearned progress of {worst:.2f} m in one step"


def test_route_tracker_is_monotone():
    """Walking the route must not jump backwards when the path passes near
    itself — a loop, a block circled twice, a roundabout."""
    from envs.route import RouteTracker

    world = bare("loop")
    net = world.net
    rng = np.random.default_rng(0)
    x, y, _h = world.sample_start(rng, declared=False)
    gx, gy, _ = world.sample_goal(rng, (x, y), 60.0)
    tracker = RouteTracker(net, (x, y), (gx, gy))

    # Walk the polyline itself: s must increase monotonically the whole way.
    last_s = -1.0
    for point in tracker.poly:
        tracker.advance(float(point[0]), float(point[1]))
        assert tracker.s >= last_s - 1e-6, \
            f"route position went backwards: {last_s:.2f} -> {tracker.s:.2f}"
        last_s = tracker.s
    assert tracker.remaining() < 1.0, "walking the whole route did not finish it"


def test_sensor_noise_does_not_reach_the_costs():
    """Noise corrupts what the agent PERCEIVES, never what it is judged on.

    If a cost were measured through a noisy sensor, the budget would stop
    meaning what it says — `<= 5 steps off the road` would become `<= 5
    steps the localiser THOUGHT were off the road`, and the constraint would
    be measuring the sensor.

    Two envs, same seed, same actions, different noise: every cost,
    termination and reward must match exactly.
    """
    actions = None
    runs = {}
    for noise in (0.0, 1.0):
        env = TrafficEnv("manhattan", n_agents=8, seed=4, sensor_noise=noise)
        obs, _ = env.reset()
        if actions is None:
            rng = np.random.default_rng(0)
            actions = [[{"steering": np.array([rng.uniform(-1, 1)], np.float32),
                         "throttle": np.array([rng.uniform(0, 1)], np.float32),
                         "brake": np.array([rng.uniform(0, 1)], np.float32),
                         "gear": 3} for _ in range(8)] for _ in range(60)]
        costs, rewards = [], []
        for step_actions in actions:
            _obs, reward, _t, _tr, info = env.step(step_actions)
            costs.append({k: v.copy() for k, v in info["cost"].items()})
            rewards.append(list(reward))
        runs[noise] = (costs, rewards)

    clean, noisy = runs[0.0], runs[1.0]
    for step, (a, b) in enumerate(zip(clean[0], noisy[0])):
        for channel in COST_CHANNELS:
            assert np.array_equal(a[channel], b[channel]), \
                f"sensor noise changed the {channel} cost at step {step}"
    assert clean[1] == noisy[1], "sensor noise changed the reward"


def test_sensor_noise_actually_perturbs():
    """...and it must actually be doing something when switched on."""
    env = TrafficEnv("manhattan", n_agents=8, seed=4, sensor_noise=1.0)
    obs, _ = env.reset()
    obs, _r, _t, _tr, info = env.step(
        [env.action_space.sample() for _ in range(8)])
    live = int(np.flatnonzero(info["active"])[0])

    # The vehicle and budget blocks are known exactly and must be untouched.
    from envs.budgets import as_vector
    assert np.allclose(obs[live]["budget"],
                       as_vector(info["vehicle_type"][live]))

    quiet = TrafficEnv("manhattan", n_agents=8, seed=4, sensor_noise=0.0)
    clean, _ = quiet.reset()
    clean, *_ = quiet.step([quiet.action_space.sample() for _ in range(8)])
    assert not np.array_equal(obs[live]["lidar"], clean[live]["lidar"]), \
        "sensor_noise=1.0 produced an identical lidar scan"


def test_actuators_lag_and_rate_limit():
    """Steering must not teleport, and heavy vehicles must brake slowly.

    A policy trained against instantaneous actuators learns to depend on a
    response no real vehicle has; the symptom on hardware is high-frequency
    steering chatter. Steering carries both a lag and a rate limit because
    they bound different things — the lag is how fast assistance builds, the
    rate is the ceiling on turning the wheel at all.
    """
    from vehicle import Vehicle, Gear
    from fleet import FLEET

    spec = FLEET["sedan"]
    veh = Vehicle(spec=spec, dt=0.1)
    veh.gear = Gear.DRIVE
    veh.vx = 10.0
    full = drive(throttle=0.0, steering=1.0)

    veh.step(full)
    first = abs(veh.steering_angle)
    assert first < veh.max_steer_angle, "steering reached full lock in one step"
    assert first <= veh.max_steer_rate * veh.dt + 1e-9, \
        f"steering moved {math.degrees(first):.1f} deg in one step, " \
        f"limit is {math.degrees(veh.max_steer_rate * veh.dt):.1f}"

    for _ in range(40):
        veh.step(full)
    assert abs(veh.steering_angle) > 0.95 * veh.max_steer_angle, \
        "steering never reached the commanded angle"

    # Air brakes: the bus must need materially more road than the car.
    def stopping_distance(name):
        v = Vehicle(spec=FLEET[name], dt=0.1)
        v.gear, v.vx, v.x = Gear.DRIVE, 20.0, 0.0
        for _ in range(200):
            v.step(drive(throttle=0.0, brake=1.0))
            if v.vx <= 0.01:
                break
        return v.x

    car, bus = stopping_distance("sedan"), stopping_distance("bus")
    assert bus > car * 1.4, \
        f"bus stops in {bus:.1f} m against the car's {car:.1f} — air brake " \
        f"lag and mass are not showing up"

    # And the ablation switch has to actually disable it.
    instant = Vehicle(spec=spec, dt=0.1, actuator_lag=False)
    instant.gear = Gear.DRIVE
    instant.step(full)
    assert abs(instant.steering_angle - instant.max_steer_angle) < 1e-9


def test_contract_is_stable():
    """Pin the interface Handoff.md documents.

    The trainer is written by someone else against that document. A shape
    or a key that changes without this failing is a silent break in another
    person's code, discovered as a confusing loss curve rather than as an
    error.
    """
    env = TrafficEnv("manhattan", n_agents=6, seed=0)

    obs_shapes = {"state": (10,), "navigation": (6,), "lidar": (120,),
                  "radar": (5, 4), "vehicle": (8,), "budget": (7,)}
    info_keys = {"cost", "active", "events", "budget", "ttc", "route",
                 "vehicle_type", "scenario"}

    obs, info = env.reset()
    # reset and step must agree: a caller that has to branch on which one
    # produced an info gets it wrong exactly once, at the episode boundary.
    assert set(info) == info_keys, sorted(info)

    obs, reward, terminated, truncated, info = env.step(
        [env.action_space.sample() for _ in range(6)])

    assert set(info) == info_keys, sorted(info)
    assert set(obs[0]) == set(obs_shapes), sorted(obs[0])
    for key, shape in obs_shapes.items():
        assert obs[0][key].shape == shape, (key, obs[0][key].shape)
        assert obs[0][key].dtype == np.float32, (key, obs[0][key].dtype)

    assert len(reward) == len(terminated) == len(truncated) == 6
    assert set(info["cost"]) == set(COST_CHANNELS)
    for channel in COST_CHANNELS:
        assert info["cost"][channel].shape == (6,)
    assert info["active"].dtype == bool and info["active"].shape == (6,)
    assert info["budget"].shape == (6, len(COST_CHANNELS))
    assert info["ttc"].shape == (6,)
    assert len(info["events"]) == 6 and len(info["vehicle_type"]) == 6
    assert isinstance(info["scenario"], str)

    # The budget in the observation must be the same vector as the one in
    # info, or a policy conditioned on one and a dual update using the other
    # are enforcing different constraints.
    live = int(np.flatnonzero(info["active"])[0])
    assert np.allclose(obs[live]["budget"], info["budget"][live])

    # And it must match the budget table for that agent's own vehicle type.
    from envs.budgets import as_vector
    assert np.allclose(info["budget"][live],
                       as_vector(info["vehicle_type"][live]))


def test_png_round_trip():
    """A world saved as a classified PNG must load back as an equivalent
    world — same layout, same buildings, same endpoints — and must be
    drivable. Lengths are compared loosely: the skeleton wanders a little
    inside a wide carriageway and arcs come back as chains of chords."""
    import tempfile, os

    from envs import png_map
    from envs.traffic_env import TrafficEnv as Env

    world = World.build("intersection_x", rng=np.random.default_rng(0))
    with tempfile.TemporaryDirectory() as tmp:
        path = png_map.save(world, os.path.join(tmp, "map.png"))
        loaded = png_map.load(path)

    assert len(loaded.scenery.boxes) == len(world.scenery.boxes)
    assert len(loaded.net.sources) == len(world.net.sources)
    assert len(loaded.net.goals) == len(world.net.goals)

    # The assertion that matters is not how long the roads are, it is that
    # every endpoint came back STANDING ON ONE. Parking bays lose some of
    # their length through a round trip — a short wide stub is largely
    # absorbed into its parent road's medial axis — so the length tolerance
    # is loose on purpose, while "can a vehicle actually start here" is not.
    for x, y, _h in loaded.net.sources:
        assert loaded.net.is_on_road(x, y), "source loaded off the road"
    for x, y in loaded.net.goals:
        assert loaded.net.is_on_road(x, y), "goal loaded off the road"
    ratio = loaded.net.total_length / world.net.total_length
    assert 0.7 < ratio < 1.2, f"road length changed by {ratio:.2f}x"

    env = Env("intersection_x", n_agents=4, seed=0, world=loaded, **ALL_ON)
    env.reset()
    for _ in range(10):
        env.step([env.action_space.sample() for _ in range(4)])


def test_renderers_produce_frames():
    """Both renderers must return a real image of the same world. The 3D one
    is skipped when no GL context can be created (a headless CI box without
    a GPU), because a missing renderer is not a broken env."""
    from envs import render_mpl

    world = World.build("cross", rng=np.random.default_rng(0))
    frame = render_mpl.draw(world)
    assert frame.ndim == 3 and frame.shape[2] == 3 and frame.dtype == np.uint8
    assert frame.std() > 1.0, "top-down frame is a flat colour"

    try:
        from envs.render3d import Renderer3D
        renderer = Renderer3D(world, size=(160, 120))
    except Exception as exc:                     # no GL context available
        print(f"    (3D renderer skipped: {exc})")
        return
    frame = renderer.frame(camera="orbit")
    assert frame.shape == (120, 160, 3), frame.shape
    assert frame.std() > 1.0, "3D frame is a flat colour"
    renderer.close()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in tests:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(tests)} checks passed")
