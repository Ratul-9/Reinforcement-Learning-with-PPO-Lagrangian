# Environment Module Documentation (`envs/`)

## Overview

`envs/` is the world the driving policy is trained in: eight collision-prone
road scenarios, populated with buildings and trees, holding 20–30 vehicles
that each drive from their own source to their own destination.

It is built for **PPO-Lagrangian**, and that shows in exactly one place — the
env returns a reward and a *separate dict of named costs*, and never combines
them. Nothing in `envs/` multiplies a cost by a weight, because the whole
point of the method is that the optimiser derives those weights from
constraint budgets instead of a developer picking them.

```
envs/
  road_network.py   graph of centrelines, routing, 8 scenario builders
  scenery.py        buildings and trees, placed procedurally off the graph
  world.py          network + scenery + collision tests + heading conversion
  sensors.py        analytic lidar / radar against that geometry
  traffic_env.py    N vehicles, one world, reward and costs kept apart
  render_mpl.py     top-down PNG — audit sheet, rollout frames
  render3d.py       Panda3D 3D view, offscreen PNG, driver's-eye camera
  png_map.py        PNG map <-> World, both directions
```

`vehicle.Sedan` supplies the physics. Nothing in `envs/` duplicates it.

---

## Quick start

```python
from envs import TrafficEnv

env = TrafficEnv("roundabout_yield", n_agents=20, seed=0)
obs, info = env.reset()

obs, reward, terminated, truncated, info = env.step(
    [env.action_space.sample() for _ in range(env.n_agents)])

info["cost"]["collision"]      # (n_agents,) — one entry per cost channel
```

Everything is a list of length `n_agents`: this IS the vectorised interface,
with the shared world as the only coupling between agents.

```bash
python test_envs.py                        # 11 checks, run after any change
python -m envs.render_mpl                  # scenarios.png — all eight, top down
python -m envs.render3d manhattan          # manhattan_3d.png — 3D, offscreen
python -m envs.png_map intersection_x      # intersection_x_map.png — a paintable map
```

---

## The eight scenarios

Built parametrically by `road_network.build(kind)`, in curriculum order.
Each declares its own source and destination points and mixes carriageway
widths across its pieces.

| kind | what it is |
|---|---|
| `parking_lot` | perimeter loop, aisles, one-lane bays. Manoeuvring speed; "leave one bay, reach another" |
| `intersection_x` | unsignalised crossings of an arterial and side streets |
| `merge_ramp` | on-ramp merging into a carriageway |
| `roundabout_yield` | four-arm roundabout with a non-drivable island |
| `manhattan` | city grid |
| `left_turn` | unprotected left turn across oncoming traffic |
| `lane_closure` | a road closed over a stretch, with a detour |
| `weave_roundabout` | five-arm roundabout with an outer weave ring |

Six generic layouts (`cross`, `tee`, `grid`, `roundabout`, `loop`, `town`)
also exist for "can it drive at all" testing. `road_network.KINDS` lists all
fourteen; `road_network.SCENARIO_KINDS` just the eight.

Geometry parameters are ordinary spec fields, so a sweep widens a
carriageway or adds an arm without touching a builder:

```python
World.build({"kind": "roundabout_yield", "radius": 28.0, "arms": 5, "lanes": 3})
```

---

## Reward and cost

**Reward** — terminal and unarguable only:

| term | value |
|---|---|
| progress | route distance closed this step, as a fraction of the episode's starting route |
| goal reached | `+1.0`, episode ends |

**Cost** — `info["cost"]`, one `(n_agents,)` array per channel, all
non-negative, all unweighted:

| channel | fires when |
|---|---|
| `collision` | footprint overlaps another vehicle, a building or a tree. Episode ends |
| `offroad` | outside the drivable surface. Episode ends after 1.5 s |
| `lane` | on the wrong side of the centreline for the direction of travel |
| `ttc` | constant-velocity time-to-collision under 2 s, ramped |
| `jerk` | jerk over 5 m/s³, ramped |
| `speeding` | over 50 km/h, ramped |

Every one of these is exercised by `test_envs.py` — a cost channel that can
never fire is a constraint whose multiplier goes to zero and a result that
quietly means nothing.

The thresholds above say **what counts as a violation**. That is a modelling
decision worth owning. What a violation is *worth* is not set here.

---

## Observation

```python
{
  "state":      (8,)   vx, vy, yaw_rate, steering, gear,
                       road_margin, lateral_offset, heading_error
  "navigation": (6,)   route_distance, near waypoint (fwd, left),
                       far waypoint (fwd, left), bend ahead
  "lidar":      (120,) ranges, 360° sweep, 100 m
  "radar":      (5, 4) nearest 5 movers: fwd, left, rel fwd speed, rel left speed
}
```

Deliberately **no absolute x/y**. A policy given its world coordinates
memorises the map; one given a route waypoint and a lane offset learns to
drive. Waypoints and offsets are in the vehicle's own frame, so the policy is
not also learning a rotation.

Route distance is measured **along the roads**, never as the crow flies. With
a straight-line distance the progress term pays for cutting the corner of an
intersection or crossing the middle of a roundabout — exactly what the
off-road cost is there to stop — and the two terms spend the whole run
fighting each other.

---

## Sensors are analytic, not rendered

The lidar is a closed-form ray/shape intersection against the same rotated
rectangles (buildings, vehicles) and circles (trees) the renderers draw — not
a depth buffer read back off a GPU. At 20–30 agents × 20 Hz that is the
difference between a run that fits on a laptop and one that does not, and
the answers are exact rather than quantised to a framebuffer.

Measured: **~1000–1500 agent-steps/s** on one CPU core, all eight scenarios.

Because both renderers and the sensors read the *same* arrays, a pixel
observation added later cannot describe a different world from the one the
lidar reported.

---

## 3D rendering

`render3d.Renderer3D` extrudes the same `World`: road ribbons and junction
pads flat on the ground, scenery boxes as buildings of their own height,
scenery circles as trees. Three cameras:

| mode | what it is for |
|---|---|
| `orbit` | the whole network from above and behind — the audit view |
| `chase` | over one vehicle's shoulder — the rollout video |
| `ego` | the driver's eye — **the pixel-observation hook** |

```python
renderer = Renderer3D(world)                    # offscreen by default
renderer.frame(env._rects(), camera="chase", path="frame.png")
pixels = renderer.frame(env._rects(), camera="ego")   # (H, W, 3) uint8
```

`Sedan`'s observation space already reserves a `rgb_camera` slot, so feeding
`camera="ego"` into it later is a matter of calling this from
`TrafficEnv._observations` and paying the render cost — one offscreen frame
is a GPU round-trip of a few milliseconds, which at 20–30 agents is slower
than the entire rest of the step. That is why the default observation is
analytic lidar and this stays a viewer until an experiment calls for pixels.

Panda3D allows exactly one `ShowBase` per process — hold on to the renderer
rather than constructing one per frame.

---

## PNG maps, both directions

```python
from envs import png_map

png_map.save(world, "town.png")      # World -> classified PNG
world = png_map.load("town.png")     # PNG   -> World
TrafficEnv("png", n_agents=20, world=world)
```

The PNG is a **classified map**, not a picture: every pixel is one of six
exact palette colours and the class is what it means.

| colour | hex | class |
|---|---|---|
| grass | `#6f9642` | background (anything unmatched) |
| asphalt | `#2a2b30` | drivable road |
| building | `#8d8579` | building footprint |
| tree | `#2f7a2a` | tree |
| source | `#2b7fd9` | an episode start point |
| goal | `#c43b3b` | an episode destination |

Paint one in any image editor and `load` recovers a routable graph: the
asphalt mask is skeletonised to a medial axis, skeleton pixels are classed by
neighbour count into dead ends, junctions and interior, each run between
those is simplified (Ramer–Douglas–Peucker) into a chain of straights, and
the width is read off the distance transform and rounded to whole lanes.

**A round trip is equivalent, not identical.** Preserved: layout, widths to
the nearest lane, buildings (position, footprint, yaw), trees, sources,
goals. Not preserved: arcs (they return as chains of chords, which every
`RoadNetwork` query handles natively), building and tree *heights* (a
top-down map cannot carry them — the loader re-rolls them), and sub-pixel
geometry. Measured road length after a round trip lands within about 6% of
the original.

---

## Scenery

Not authored per scenario. The road graph already says where the road is
*not*, so candidates are drawn on a jittered grid and kept by distance from
the nearest road edge:

| distance from road edge | what goes there |
|---|---|
| < 2.5 m | nothing (kerb) |
| 2.5–7 m | trees |
| 12–55 m | buildings |
| > 55 m | nothing (open country) |

Closing the building band at the top is what makes the result read as a town
rather than a field of boxes, and keeps the object count proportional to the
network instead of to the square of the map margin. Buildings are yawed to
the nearest centreline so they face their street. Roundabout islands are
excluded explicitly — they are off-road by the margin test and would
otherwise be the most attractive plot on the map.

`scenery_density=0.0` gives a bare network, which is what the collision and
off-road tests use.

---

## Coordinate and heading conventions

`road_network.py` is a verbatim port from the earlier Panda3D project and
speaks that engine's **H angle** — degrees, zero along +Y. `vehicle.Sedan`
speaks the ordinary maths convention — **radians, zero along +X**. Every
crossing goes through `world.h_to_rad` / `world.rad_to_h` and nowhere else.

The port is byte-identical to `LANCER3D-dev/sim/world/road_network.py` on
purpose: it is pure numpy geometry with no Panda3D or Qt import, it is the
most load-bearing file here, and keeping it diffable against its origin is
worth more than tidying its heading convention.

---

## Known gaps

- **No traffic lights, right-of-way rules, or pedestrians.** The scenarios
  are collision-prone by geometry, which is enough for the constraints to
  bite; signals would add a discrete state to the observation space.
- **One vehicle type.** `TrafficEnv(vehicle_cls=...)` takes any class with
  `Sedan`'s interface, but only `Sedan` exists so far.
- **Arcs are lost through a PNG round trip** (see above).
- **No validation against real trajectory data.** `paper-idea.md` flags this
  as a Phase 5 deliverable, not an afterthought.
