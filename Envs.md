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
  lanes.py          discrete lanes derived from each piece's lane count
  bays.py           parking bays attached to any network; bay-to-bay task
  scenery.py        buildings and trees, placed procedurally off the graph
  world.py          network + scenery + collision tests + heading conversion
  sensors.py        analytic lidar / radar against that geometry
  traffic_env.py    N vehicles, one world, reward and costs kept apart
  metrics.py        per-episode statistics: the results table
  render_mpl.py     top-down PNG — audit sheet, rollout frames
  render3d.py       Panda3D 3D view, offscreen PNG, driver's-eye camera
  png_map.py        PNG map <-> World, both directions
```

```
fleet.py            physical parameters for the five vehicle types
vehicle.py          the 4-wheel dynamic model, driven by a VehicleSpec
envs/budgets.py     per-type cost budgets — the d_i in the Lagrangian
```

`vehicle.Vehicle` supplies the physics; nothing in `envs/` duplicates it.

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

### Arrivals are staggered

Vehicles do not all appear at once. Half the fleet is on the road at reset
and the rest trickle in over the next 25 s; a vehicle that crashes or
arrives waits an Exponential(4 s) before returning. Spawning everything at
reset makes every conflict a consequence of one simultaneous placement —
the same cars meet at the same junctions at the same times, and a policy can
learn that schedule instead of learning to drive.

`info["active"]` is the boolean mask of who is actually on the road. A
waiting vehicle keeps its row — the arrays stay rectangular, because a
ragged interface just has to be un-ragged again by every caller — but gets a
zeroed observation, zero reward, no costs and no terminal flag, and is
invisible to every other vehicle's sensors. **The learner drops those rows
from its batch.** Its action is ignored, so a policy that keeps emitting one
costs nothing.

Each agent also has its own episode clock, so a timeout retires one vehicle
rather than resetting the fleet together.

`n_agents` is therefore an upper bound on density, not the density. With the
scripted driver's crash rate the mean on-road count sits around 11-16 out of
20; a policy that crashes less will sit nearer the cap. Set `n_agents`
higher than the density you want, or pass `initial_active=1.0,
arrival_spread=0.0, respawn_delay=0.0` for the old fixed-density behaviour.

```bash
python test_envs.py                        # 29 checks, run after any change
python fleet.py                            # the vehicle table
python -m envs.budgets                     # the budget table
python -m envs.vec                         # parallel throughput benchmark
python rollout.py                          # scripted driver on all 8 scenarios
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
| `parking_lot` | perimeter loop, aisles, one-lane bays. Manoeuvring speed |
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

### The task is bay to bay, everywhere

Every scenario gets parking bays attached (`envs/bays.py`) and they become
its sources and goals, so the episode is always "leave a parking space,
drive the scenario, park in another one". Bays inherit the positions of
whatever endpoints the scenario declared, so the route still enters the
merge from the ramp and still crosses the unsignalised junction — it just
now starts and ends stationary, which is the part a road-to-road task never
exercises. Another ~18 bays are scattered for the rest of the traffic.

`World.build(..., bays=False)` turns this off; `parking_lot` already is bays
and is left alone.

### Lanes

`envs/lanes.py` derives lane centrelines from each piece's lane count rather
than storing them, so a network from any source — a builder, a PNG — gets
lanes without knowing lanes exist. Right-hand traffic: a lane on the right
half runs along the piece's tangent, one on the left runs against it. One
lane means a bidirectional bay or aisle. An odd count puts a **shared
turning lane** on the centreline, direction 0, which is never a wrong-way
violation.

Both renderers paint one dashed divider per lane boundary, so what is drawn
is what `lane_keep` is measured against.

Geometry parameters are ordinary spec fields, so a sweep widens a
carriageway or adds an arm without touching a builder:

```python
World.build({"kind": "roundabout_yield", "radius": 28.0, "arms": 5, "lanes": 3})
```

---

## The fleet

Five vehicle types, each earning its slot by changing the *conflict* rather
than the paint. `python fleet.py` prints the table.

| | L × W (m) | mass | turn radius | top speed |
|---|---|---|---|---|
| motorcycle | 2.00 × 0.75 | 180 kg | 2.3 m | 119 km/h |
| tuktuk (auto rickshaw) | 2.63 × 1.30 | 450 kg | 2.0 m | 54 km/h |
| sedan | 4.50 × 1.80 | 1500 kg | 4.0 m | 180 km/h |
| truck | 7.50 × 2.40 | 5800 kg | 8.6 m | 101 km/h |
| bus | 12.20 × 2.55 | 12000 kg | 8.6 m | 79 km/h |

One set of equations; what differs is the `VehicleSpec` handed in. The
default mix is 40% sedan, 25% motorcycle, 20% tuktuk, 10% truck, 5% bus,
realised as **counts** rather than sampled — at 20 agents an independently
sampled 5% bus share gives a run with no bus about a third of the time.

```python
TrafficEnv("manhattan", n_agents=20)                      # the default mix
TrafficEnv("manhattan", vehicle_types="sedan")            # one type
TrafficEnv("manhattan", vehicle_types={"bus": 1, "sedan": 3})
```

### The policy is told its physics, not its name

The observation carries a `vehicle` block of eight scaled physical
parameters — length, width, mass, wheelbase, max steer, accel, brake, top
speed — rather than a one-hot type ID. One shared policy then spans the
fleet and could in principle drive a type it never saw; a one-hot cannot,
and grows every time a type is added.

### Budgets come from the vehicle

`envs/budgets.py` holds `d_i` per type. This is `paper-idea.md`'s
personality-as-budget idea, except grounded in physics rather than an
invented trait vector — every difference is one you can argue about:

- a **bus** gets a tight jerk budget (standing passengers) and a loose
  `lane_keep` budget (twelve metres cannot hold a lane through a turn)
- a **truck** gets a tight speeding budget (stopping distance)
- a **motorcycle** and **tuktuk** get tight TTC budgets (no crumple zone)

The observation carries the agent's own budget vector, so one network can
drive a bus cautiously and a rickshaw loosely instead of averaging them
into a vehicle that is neither.

**These differences are not decorative.** Cost per active agent-step,
manhattan, 24 agents, scripted driver:

| type | lane_keep | ttc | jerk | offroad |
|---|---|---|---|---|
| motorcycle | 0.018 | 0.042 | 0.144 | 0.094 |
| tuktuk | 0.066 | 0.051 | 0.179 | 0.155 |
| sedan | 0.117 | 0.069 | 0.157 | 0.081 |
| truck | 0.283 | 0.094 | 0.276 | 0.056 |
| bus | 0.332 | 0.106 | 0.179 | 0.030 |

`lane_keep` rises monotonically with size, unprompted — big vehicles
physically cannot hold a lane centre. That is the evidence for the loose bus
budget, rather than an assertion about it.

### What the vehicle model does not do

**Roll-over.** A bus tips at about 7.4 m/s² lateral, below where its tyres
would slide, so a real one rolls rather than slides. Modelling that needs
suspension and load transfer, and charging it needs an eighth cost channel —
at six simultaneous constraints, where dual ascent already oscillates.
Instead tippy vehicles carry reduced grip, so they lose traction roughly
where they would have tipped, and the failure reads as understeer.

**Leaning.** A motorcycle here is a very narrow four-wheeler. Right
footprint, mass and acceleration — which is what decides whether it fits a
gap — and the wrong reason for staying upright.

**Articulated vehicles.** A bendy bus or semi-trailer needs a hitch angle
and a second body: different code, not different numbers. Left out.

---

## Reward and cost

**Reward** — terminal and unarguable only:

| term | value |
|---|---|
| progress | route distance closed this step, as a fraction of the episode's starting route |
| goal reached | `+1.0`, episode ends |

**Cost** — `info["cost"]`, one `(n_agents,)` array per channel, all
non-negative, all unweighted, **all bounded to [0, 1] per step**:

| channel | fires when | a budget reads as |
|---|---|---|
| `collision` | footprint overlaps another vehicle, a building or a tree. Episode ends | collisions per episode |
| `offroad` | outside the drivable surface. Episode ends after 1.5 s | steps off the road |
| `wrong_way` | in a lane that runs the other way | steps in the wrong lane |
| `lane_keep` | off its own lane's centre, ramped past a dead band | steps at full drift |
| `ttc` | constant-velocity time-to-collision under 2 s, ramped | steps of imminent conflict |
| `jerk` | filtered jerk over 2.5 m/s³, ramped | steps at full harshness |
| `speeding` | over 50 km/h, ramped | steps at double the limit |

### Why the [0, 1] bound matters

An episode's cost `J_c` is the sum over its steps, so a bounded per-step
cost gives `J_c` a unit anyone can read — which is the whole interpretable-
budget claim. Unbounded channels destroy it. Before the bound was enforced,
`jerk` hit 21.3 in one step and totalled 2160 over a window where the entire
reward totalled 4.4: `λ_jerk` would have had to converge near 1e-3 while
`λ_collision` sat near 1, and at six simultaneous constraints that spread is
how plain dual ascent oscillates into a do-nothing policy.

Measured per active agent-step with the scripted driver, the channels now
span 0.005 (`collision`) to 0.54 (`lane_keep`) — and that remaining spread
is real rather than an artefact: collisions are rare events, lane drift is
continuous.

### Jerk is measured from smoothed acceleration

Differencing raw step-to-step acceleration measures the control rate, not
the ride: at a 0.1 s control period a 5 m/s³ threshold trips whenever
acceleration moves 0.5 m/s² in one step, which every 10 Hz policy does
constantly. Acceleration is low-passed (τ = 0.3 s) before differencing, and
the threshold is then a real comfort figure. Held throttle is one continuous
acceleration and is not charged; a throttle-to-brake reversal is.

All seven are exercised by `test_envs.py`, one test each, asserting both
that the channel fires and that the ramped ones ramp, plus one test holding
every channel inside [0, 1] under random actions. A cost channel that
can never fire is a constraint whose multiplier goes to zero and a result
that quietly means nothing.

The thresholds above say **what counts as a violation**. That is a modelling
decision worth owning. What a violation is *worth* is not set here.

---

## Observation

```python
{
  "state":      (10,)  vx, vy, yaw_rate, steering, gear, road_margin,
                       heading_error, lane_offset, lane_index, wrong_way
  "navigation": (6,)   route_distance, near waypoint (fwd, left),
                       far waypoint (fwd, left), bend ahead
  "lidar":      (120,) ranges, 360° sweep, 100 m
  "radar":      (5, 4) nearest 5 movers: fwd, left, rel fwd speed, rel left speed
  "vehicle":    (8,)   own length, width, mass, wheelbase, steer, accel,
                       brake, top speed — scaled, not a one-hot type
  "budget":     (7,)   own cost budgets, in COST_CHANNELS order
}
```

`lane_offset` is measured from the centre of the vehicle's OWN lane, not
from the road's centreline: on a three-lane arterial a correctly-driven
vehicle is several metres off the centreline, and that number says nothing
about whether it is driving well.

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

Measured: **~2300–2500 agent-steps/s** on one CPU core, all eight scenarios.
Two exact optimisations got it there from ~1150, neither changing an answer:

- **Broad-phase road projection.** `_project_full` was the hottest call in
  the run — twenty agents projecting six times each per step, against every
  piece of the network. A per-piece bounding box gives a lower bound on the
  distance, hence an upper bound on the margin; visiting pieces in
  descending order of that bound and stopping once it falls below the best
  margin measured is exact and evaluates a handful instead of all sixty.
  Verified against brute force on 3000 random points: 0 mismatches.
- **Range-pruned lidar.** An object whose centre is beyond `max_range` plus
  its own circumradius cannot be hit, so it never enters the O(rays ×
  objects) raycast.

Because both renderers and the sensors read the *same* arrays, a pixel
observation added later cannot describe a different world from the one the
lidar reported.

---

## Running many environments at once

`envs/vec.py` runs one `TrafficEnv` per worker process, optionally each on a
different scenario — which is what a curriculum wants, since a shared policy
then sees a roundabout, a merge and a parking lot in the same update.

```python
from envs.vec import VecTrafficEnv, performance_cores

with VecTrafficEnv(["manhattan", "merge_ramp", "roundabout_yield"],
                   n_agents=20) as vec:
    obs = vec.reset()                    # per env, BATCHED: (n_agents, 120) lidar
    obs, rew, term, trunc, info = vec.step(actions)
```

A worker owns a whole env rather than a slice of one: agents inside an env
collide with each other and appear in each other's lidar, so splitting there
would mean shipping every pose to every worker each step.

**Use one worker per *performance* core.** `os.cpu_count()` counts efficiency
cores, which are several times slower at this workload, and the parent waits
for every worker each step — so the slowest one sets the pace.
`performance_cores()` reads the real number.

Measured on an Apple M4 (4 performance + 6 efficiency), 20 agents each:

| workers | wall | per-worker busy | agent-steps/s |
|---|---|---|---|
| 1 | 0.95 s | 0.93 s | 3161 |
| 4 | 2.35 s | 1.96 s | **5114** |
| 8 | 5.12 s | 3.46 s | 4683 |

What grows is the per-worker *compute*, not the plumbing: identical work
takes 2.1× longer at four workers and 3.7× at eight, because past the
fourth it lands on an efficiency core. The pipe itself sustains ~4600
round-trips/s — a ceiling of ~90k agent-steps/s at 20 agents, two orders of
magnitude clear of anything measured. **So the plateau is this laptop, not
the design**; coordination efficiency is 0.84, and a box with 16 real
performance cores should scale close to linearly.

---

## Layout variation

A fixed map is a map a policy can memorise — where its junctions are, which
way its roads run. Two knobs re-roll it, both applied **at `reset` and
nowhere else**: moving geometry under vehicles that are driving on it would
teleport them off the road, so a trainer that never calls `reset` never
varies.

```python
TrafficEnv("manhattan", layout_jitter=0.15, blockages=5)
```

**`layout_jitter`** randomises the layout's *continuous* dimensions by
±15% — arm lengths, block sizes, radii, ramp and closure lengths. Counts
(`arms`, `rows`, `cols`, `lanes`) are deliberately excluded: those change a
layout's topology, and the builders make assumptions about it that a random
integer would quietly break. Stretching an arm cannot.

**`blockages`** parks stalled vehicles in live lanes. These are the
interesting ones, because **the route does not know about them** — routing
is over centrelines — so the agent is steered straight into an obstruction
and has to leave its lane to get past. That puts `lane_keep` and `wrong_way`
in genuine tension with the progress reward, which is a conflict a fixed
layout never produces. Only multi-lane pieces are used, and never within 12 m
of a bay, so there is always a way round and nobody spawns nose-first into a
lorry.

A rebuild costs ~40 ms against a ~7.5 s episode, so the variation is 0.5%
overhead. Measured over three episodes on manhattan with the scripted
driver:

| | goal | collision | offroad |
|---|---|---|---|
| fixed layout | 16.4% | 45.4% | 38.2% |
| varied + 5 blockages | 20.8% | 49.5% | 29.7% |

---

## Episode metrics

`envs/metrics.py` turns per-step costs into the statistics the results
section is made of. An episode is a per-agent span between terminal flags —
with staggered arrivals every agent is at a different point of its own
journey, so anything aggregating by step averages a bus three seconds in
with a motorcycle ninety seconds in.

```python
from envs.metrics import EpisodeRecorder

recorder = EpisodeRecorder(dt=env.dt)
for ...:
    obs, rew, term, trunc, info = env.step(actions)
    recorder.update(info, term, trunc, rewards=rew)
print(recorder.report())
recorder.write_csv("run.csv")       # one row per episode
recorder.write_json("run.json")     # summary + constraint table
```

```bash
python rollout.py manhattan --metrics run   # writes run.csv and run.json
```

It reports goal / collision / off-road / timeout rates (which partition —
timeout is neither success nor failure), time-to-goal over the episodes that
*reached* the goal (averaging it over failures lets a policy improve the
number by crashing sooner), the minimum-TTC distribution, and **realised
cost against budget per channel per vehicle type**.

That last table is the headline. The claim under test is that the
constraint is actually met — `J_c` inside `d_i` at convergence — and a run
whose collision rate looks good while its collision cost sits at three times
its budget has not shown what the paper says it shows. Budgets are read from
`info["budget"]`, so a per-type budget table is compared per type without
the recorder knowing one exists.

λ trajectories are deliberately absent: they belong to the optimiser, and a
recorder that invented a place for them would be guessing at the update
schedule. Pass them to `report(extra=...)`.

Two caveats on reading the output from the *scripted* driver: it violates
most budgets, because the budgets are calibrated for a trained policy and
that driver has no collision avoidance; and its min-TTC median is 0.00 s
because 39% of its episodes end in a collision, where TTC is zero by
definition. Filter on `outcome` in the CSV for a near-miss distribution that
excludes actual crashes.

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
| endpoint | `#9b59b6` | both — a parking bay, usable either way |

Marker pixels count as drivable: a source is a point a vehicle stands on, so
painting one is a dot *on* the road, not a hole in it.

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
geometry, and **parking bays lose most of their stub length** — a short wide
protrusion is largely absorbed into its parent road's medial axis, so the
loader rebuilds the stub from the surviving marker dot instead. Measured road
length after a round trip lands within 6% on a bay-free network and 10-25%
on one with bays; every source and goal comes back standing on drivable
surface, which is the property that matters and the one the test asserts.

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

`road_network.py` began as a byte-identical port of
`LANCER3D-dev/sim/world/road_network.py` — pure numpy geometry, no Panda3D
or Qt import. It has since diverged in exactly one place: `_project_full`
gained a bounding-box broad phase, and `finalize` builds the index for it.
That was worth breaking the clean diff for, since it was the hottest call in
a training run; nothing else in the file has been touched, and the heading
convention is deliberately left as the original's.

---

## Route waypoints are lane-aware

`RoadNetwork.route_probe` answers with points on the road's **centreline**,
because a centreline graph is all it has. `World.lane_waypoints` shifts each
one into the lane the route actually runs in, and that is what the
observation carries.

This was a deliberate change, not a tidy-up. Steering at a centreline
waypoint means driving down the middle of the carriageway, which on a
two-way road is the oncoming lane — so `wrong_way` fired on a third of all
steps *by construction*, before any policy had done anything wrong. A
constraint violated that often at initialisation drives its multiplier hard
and is precisely the dual-ascent collapse the method has to design around.

Direction of travel at each waypoint comes from the route itself (the step
from the previous waypoint), not from the vehicle's heading, so the shift is
still right on the far side of a junction the vehicle has not reached.

Measured with the scripted driver, per active agent-step. (These figures
predate the mixed fleet and the TTC fix below, and were taken on a
sedan-only fleet; the *direction* of every column is what they are cited
for, not the absolute values.)

| | centreline | in-lane |
|---|---|---|
| goals (all 8 scenarios) | 101 | **157** |
| crashes | 781 | **574** |
| `wrong_way` | 0.18–0.39 | **0.13–0.30** |
| `lane_keep` | 0.29–0.54 | **0.21–0.38** |
| `collision` | 0.005–0.015 | **0.003–0.012** |
| `offroad` | 0.04–0.11 | 0.07–0.09 |

`offroad` went slightly **up**, and that is the honest trade: a vehicle
correctly in its lane sits nearer the kerb than one straddling the middle.
`lane_keep` still binds where it should — cornering, overtaking, avoiding a
conflict — so it keeps its dial.

---

## Known gaps

- **No traffic lights, right-of-way rules, or pedestrians.** The scenarios
  are collision-prone by geometry, which is enough for the constraints to
  bite; signals would add a discrete state to the observation space.
- **No pedestrians or other moving non-vehicles.** Dropped from scope.
- **No SB3 or PettingZoo adapter.** `VecTrafficEnv` is the vectorised
  interface, but it is this project's shape, not either library's.
- **No roll-over, leaning or articulated vehicles** (see The fleet, above).
- **Arcs are lost through a PNG round trip** (see above).
- **No validation against real trajectory data.** `paper-idea.md` flags this
  as a Phase 5 deliverable, not an afterthought.
