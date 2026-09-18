"""Render a set of presentation figures into `figures/`.

    conda run -n py310 python figures.py

Writes, for every scenario: a top-down plan, a 3D aerial with traffic on it,
and a driver's-eye view from one of the vehicles. Plus the combined
`scenarios.png` contact sheet.

Each scenario renders in its own subprocess. That is not caution, it is a
hard constraint: Panda3D allows exactly one `ShowBase` per process, so eight
3D scenes in one process is eight crashes.

Traffic is settled first — the scripted driver from `rollout.py` runs for a
few seconds of simulated time before anything is captured — because at reset
every vehicle is parked in a bay, and a picture of a town where nothing has
moved yet does not show that it is a traffic simulator.
"""

from __future__ import annotations

import os
import subprocess
import sys

OUT = "figures"
SETTLE_STEPS = 120        # 12 s of simulated time before the shutter
AGENTS = 24


def render_one(scenario: str) -> None:
    """Runs inside a fresh subprocess — one ShowBase, one scenario."""
    from envs.traffic_env import TrafficEnv
    from envs.render3d import Renderer3D
    from envs import render_mpl
    from rollout import pure_pursuit

    env = TrafficEnv(scenario, n_agents=AGENTS, seed=7)
    obs, _ = env.reset()
    for _ in range(SETTLE_STEPS):
        obs, *_ = env.step([pure_pursuit(o) for o in obs])

    rects, goals = env._rects(), env._goals
    render_mpl.draw(env.world, vehicles=rects, goals=goals,
                    path=f"{OUT}/{scenario}_plan.png")

    renderer = Renderer3D(env.world, size=(1600, 1000))
    renderer.frame(rects, camera="orbit", path=f"{OUT}/{scenario}_aerial.png")
    # Frame the ego on whichever vehicle is actually moving, so the driver's
    # view is of a road being driven rather than of the inside of a bay.
    ego = max(range(len(env.vehicles)), key=lambda i: abs(env.vehicles[i].vx))
    renderer.frame(rects, ego=ego, camera="chase", path=f"{OUT}/{scenario}_chase.png")
    renderer.frame(rects, ego=ego, camera="ego", path=f"{OUT}/{scenario}_driver.png")
    renderer.close()


def main() -> None:
    from envs import road_network, render_mpl

    os.makedirs(OUT, exist_ok=True)
    render_mpl.contact_sheet(f"{OUT}/all_scenarios.png")
    print(f"wrote {OUT}/all_scenarios.png")

    for scenario in road_network.SCENARIO_KINDS:
        subprocess.run([sys.executable, __file__, scenario], check=True,
                       stdout=subprocess.DEVNULL)
        print(f"wrote {OUT}/{scenario}_"
              "{plan,aerial,chase,driver}.png")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        render_one(sys.argv[1])
    else:
        main()
