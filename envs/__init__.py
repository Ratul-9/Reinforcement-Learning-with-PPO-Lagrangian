"""Environments for constrained multi-agent driving RL.

    road_network.py   graph of centrelines + routing. Pure geometry, no deps
                      beyond numpy. Ported unchanged from LANCER-3D.
    scenery.py        buildings and trees, placed procedurally off the graph
    world.py          network + scenery + overlap tests + heading conversion
    sensors.py        analytic lidar / radar against that geometry
    traffic_env.py    N vehicles, one world, reward and costs kept separate
    render_mpl.py     top-down PNG (audit sheet, rollout frames)
    render3d.py       Panda3D 3D view and offscreen PNG frames

`vehicle.Sedan` supplies the physics; nothing in here duplicates it.
"""

from envs.world import World
from envs.traffic_env import TrafficEnv

__all__ = ["World", "TrafficEnv"]
