"""Drive a scenario with a hand-written controller and report what happened.

    conda run -n py310 python rollout.py                       # all scenarios
    conda run -n py310 python rollout.py manhattan             # one
    conda run -n py310 python rollout.py manhattan --frames 6  # plus PNGs

This is not a baseline for the paper and it is not a policy. It is the
sanity check that has to pass before any training run is worth starting:

    can anything at all reach a goal in these environments?

If a pure-pursuit controller that simply steers at the route waypoint and
holds a sensible speed cannot complete a single episode, the problem is the
environment — a route that does not exist, a goal radius that cannot be hit,
a spawn that starts off-road — and no amount of PPO will find that out for
you. It reports per-scenario goal rate, collision rate and the mean of every
cost channel, so a channel that never fires anywhere is visible immediately.

The controller deliberately has no collision avoidance. Its collisions are
information: they say the traffic density is producing conflicts, which is
the whole point of the scenario set.
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from envs import road_network
from envs.metrics import EpisodeRecorder
from envs.traffic_env import TrafficEnv, COST_CHANNELS

TARGET_SPEED = 7.0        # m/s — brisk for a town, slow enough to take a bend
STEER_GAIN = 1.4          # bearing (rad) to steering command
SLOW_FOR_BEND = 0.55      # target speed multiplier, scaled by how hard it bends


def pure_pursuit(obs) -> dict:
    """Steer at the far route waypoint, hold a speed, slow for bends.

    The waypoint is already in the vehicle's own frame — `navigation` carries
    (forward, left) for a near and a far probe — so the bearing to it is one
    atan2 and no transform.
    """
    _route, _nf, _nl, far_fwd, far_left, bend = obs["navigation"]
    bearing = math.atan2(float(far_left), max(float(far_fwd), 0.1))
    steering = float(np.clip(STEER_GAIN * bearing, -1.0, 1.0))

    # Ease off for a corner, using the bend the route makes ahead rather than
    # the yaw rate: by the time a vehicle is yawing it is already in the bend.
    ease = 1.0 - SLOW_FOR_BEND * min(abs(float(bend)) / (math.pi / 2), 1.0)
    target = TARGET_SPEED * max(ease, 0.25)
    # Never ask for more than this vehicle has. The last element of the
    # `vehicle` block is its governed top speed, scaled by 50 m/s — a tuktuk
    # tops out at 15 m/s, and a controller demanding 7 m/s uphill of that is
    # holding full throttle forever and reporting it as a tracking error.
    if "vehicle" in obs:
        target = min(target, float(obs["vehicle"][7]) * 50.0 * 0.85)

    speed = float(obs["state"][0])
    throttle = float(np.clip((target - speed) * 0.5, 0.0, 1.0))
    brake = float(np.clip((speed - target) * 0.4, 0.0, 1.0))
    return {"steering": np.array([steering], dtype=np.float32),
            "throttle": np.array([throttle], dtype=np.float32),
            "brake": np.array([brake], dtype=np.float32),
            "gear": 3}


def run(scenario: str, n_agents: int = 20, steps: int = 900, seed: int = 0,
        frames: int = 0, recorder: EpisodeRecorder | None = None) -> dict:
    """One rollout. Returns the summary row printed by `main`."""
    env = TrafficEnv(scenario, n_agents=n_agents, seed=seed)
    obs, _ = env.reset()

    renderer = None
    if frames:
        from envs.render3d import Renderer3D
        renderer = Renderer3D(env.world)

    totals = {k: 0.0 for k in COST_CHANNELS}
    goals = collisions = episodes = 0
    # Costs are averaged over ACTIVE agent-steps, not over the rectangular
    # array: a vehicle waiting to arrive contributes a zero to every channel
    # and would silently deflate every mean in the table.
    live_steps = 0
    every = max(steps // frames, 1) if frames else 0

    for step in range(steps):
        obs, reward, terminated, truncated, info = env.step(
            [pure_pursuit(o) for o in obs])
        if recorder is not None:
            recorder.update(info, terminated, truncated, rewards=reward)
        live_steps += int(info["active"].sum())
        for k in COST_CHANNELS:
            totals[k] += float(info["cost"][k].sum())
        for event in info["events"]:
            if event == "goal":
                goals += 1
            elif event.startswith("collision"):
                collisions += 1
            if event:
                episodes += 1
        if renderer and step % every == 0:
            renderer.frame(env.vehicle_rects(), camera="chase",
                           path=f"rollout_{scenario}_{step:04d}.png")

    if renderer:
        renderer.close()
    live_steps = max(live_steps, 1)
    return {"scenario": scenario, "episodes": episodes, "goals": goals,
            "collisions": collisions, "active": live_steps / steps,
            **{k: v / live_steps for k, v in totals.items()}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scenario", nargs="?", help="one scenario, or all if omitted")
    ap.add_argument("--agents", type=int, default=20)
    ap.add_argument("--steps", type=int, default=900, help="control steps (0.1 s each)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frames", type=int, default=0,
                    help="write this many chase-camera PNGs")
    ap.add_argument("--metrics", metavar="PATH",
                    help="write the per-episode CSV and the constraint "
                         "table here (PATH.csv and PATH.json)")
    args = ap.parse_args()

    kinds = [args.scenario] if args.scenario else list(road_network.SCENARIO_KINDS)
    recorder = EpisodeRecorder() if args.metrics else None
    head = f"{'scenario':18s} {'episodes':>8s} {'goals':>6s} {'crashes':>8s} " \
           f"{'active':>7s}  " + "  ".join(f"{k:>9s}" for k in COST_CHANNELS)
    print(head)
    print("-" * len(head))
    for kind in kinds:
        row = run(kind, args.agents, args.steps, args.seed, args.frames,
                  recorder=recorder)
        print(f"{row['scenario']:18s} {row['episodes']:8d} {row['goals']:6d} "
              f"{row['collisions']:8d} {row['active']:7.1f}  " +
              "  ".join(f"{row[k]:9.4f}" for k in COST_CHANNELS))
    print("\ncost columns are per ACTIVE agent-step; `active` is the mean "
          "number of vehicles on the road.\ngoals+crashes < episodes means the "
          "rest timed out.")
    if recorder is not None:
        print()
        print(recorder.report())
        print()
        print("wrote", recorder.write_csv(args.metrics + ".csv"))
        print("wrote", recorder.write_json(args.metrics + ".json"))


if __name__ == "__main__":
    main()
