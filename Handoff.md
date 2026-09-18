# Handoff — the environment/learner contract

This is the interface the training code is written against. It is pinned by
`test_contract_is_stable` in `test_envs.py`, so changing anything here
breaks a test loudly rather than silently changing what a trainer receives.

Read [`Envs.md`](Envs.md) for what the environment *is*. This document is
only the boundary.

---

## The call

```python
from envs import TrafficEnv

env = TrafficEnv(
    scenario="manhattan",     # or any of road_network.SCENARIO_KINDS
    n_agents=20,              # upper bound on density, not the density
    seed=0,
    layout_jitter=0.15,       # re-roll geometry each reset
    blockages=5,              # stalled vehicles in live lanes
)

obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(actions)
```

`actions` is a list of `n_agents` dicts. Everything returned is a list or
array of length `n_agents`, in agent order, every step. **The arrays are
always rectangular** — an agent waiting to arrive still has a row.

---

## Action, per agent

| key | type | range |
|---|---|---|
| `steering` | `float32 (1,)` | `[-1, 1]`, scaled to that vehicle's max steer |
| `throttle` | `float32 (1,)` | `[0, 1]` |
| `brake` | `float32 (1,)` | `[0, 1]` |
| `gear` | `int` | `0` park, `1` reverse, `2` neutral, `3` drive |

`env.action_space` is the matching `gymnasium.spaces.Dict`.

---

## Observation, per agent

| key | shape | contents |
|---|---|---|
| `state` | `(10,)` | vx, vy, yaw rate, steering angle, gear, road margin, heading error, lane offset, lane index, wrong-way flag |
| `navigation` | `(6,)` | route distance, near waypoint (fwd, left), far waypoint (fwd, left), bend ahead |
| `lidar` | `(120,)` | 360° sweep, metres, clipped to 100 |
| `radar` | `(5, 4)` | nearest 5 movers: fwd, left, rel fwd speed, rel left speed |
| `vehicle` | `(8,)` | own length, width, mass, wheelbase, max steer, accel, brake, top speed — scaled |
| `budget` | `(7,)` | own cost budgets `d_i`, in `COST_CHANNELS` order |

All `float32`. Everything positional is in the **vehicle's own frame**;
there are no absolute world coordinates, deliberately.

`envs.traffic_env.batch_obs(obs)` stacks the list into
`{key: (n_agents, ...)}` — the shape a policy forwards, and what
`VecTrafficEnv` sends over its pipe.

---

## Return values

| name | type | notes |
|---|---|---|
| `reward` | list of `float32` | progress along the route, `+1.0` on arrival |
| `terminated` | list of `bool` | goal, collision, or off-road past the grace window |
| `truncated` | list of `bool` | that agent's own 90 s clock expired |

Episode clocks are **per agent**, so `truncated` is not all-or-nothing.

### `info`

Identical keys from `reset` and `step`.

| key | type | notes |
|---|---|---|
| `cost` | dict of 7 × `(n_agents,) float32` | each in `[0, 1]` per step |
| `active` | `(n_agents,) bool` | **the mask — see below** |
| `events` | list of `n_agents` str | `""`, `goal`, `collision_vehicle`, `collision_static`, `offroad` |
| `budget` | `(n_agents, 7) float32` | `d_i` per agent, `COST_CHANNELS` order |
| `ttc` | `(n_agents,) float64` | raw seconds, `inf` when no conflict |
| `route` | `(n_agents,) float64` | remaining route distance, metres |
| `vehicle_type` | list of `n_agents` str | `sedan`, `tuktuk`, `motorcycle`, `truck`, `bus` |
| `scenario` | str | layout name |

`envs.traffic_env.COST_CHANNELS` defines the channel order. **Use it**
rather than hard-coding names — a learner that zips its multipliers against
a differently-ordered budget optimises the wrong constraint and nothing
will say so.

---

## The two things a trainer must get right

### 1. Drop inactive rows

Arrivals are staggered, so not every agent is on the road. A waiting agent
has a zeroed observation, zero reward, no costs and no terminal flag, and
its action is ignored.

```python
mask = info["active"]
# keep only mask==True rows in the rollout buffer
```

Including them trains the policy on transitions that never happened.

### 2. Costs are per step; budgets are per episode

`info["cost"][channel][i]` is this step's cost. `d_i` in
`info["budget"][i]` is a budget on the **episode sum**:

```python
J_c[i] += info["cost"][channel][i]          # while the episode runs
# on terminated[i] or truncated[i]:
violation = J_c[i] - budget[i, channel]     # feeds the dual update
J_c[i] = 0.0
```

Budgets differ per vehicle type, so the dual variables are either per type
or conditioned on the budget the policy already observes. Averaging a bus's
loose lane budget with a sedan's tight one gives a constraint neither vehicle
has.

---

## Running many environments

```python
from envs.vec import VecTrafficEnv, performance_cores

with VecTrafficEnv(["manhattan", "merge_ramp", "roundabout_yield"],
                   n_agents=20) as vec:
    obs = vec.reset()                     # list per env, BATCHED per env
    obs, rew, term, trunc, info = vec.step(actions)   # actions: list per env
```

Observations are already batched per env — `obs[e]["lidar"]` is
`(n_agents, 120)`. Results are lists indexed by worker, not flattened, so
per-scenario metrics stay attributable.

Use **one worker per performance core** (`performance_cores()`), not per
logical CPU. The parent waits for every worker each step, so the slowest
sets the pace.

---

## Metrics

```python
from envs.metrics import EpisodeRecorder

recorder = EpisodeRecorder(dt=env.dt)
recorder.update(info, terminated, truncated, rewards=reward)   # every step
print(recorder.report())
recorder.write_csv("run.csv"); recorder.write_json("run.json")
```

Produces the goal/collision/off-road/timeout rates, time-to-goal, minimum-TTC
distribution, and realised cost against budget per channel per vehicle type.

λ trajectories are **not** recorded here — they belong to the optimiser.
Pass them via `report(extra=...)` or log them alongside.

---

## What the environment does not decide

- **Shared policy vs one network per vehicle.** `step` takes a list of
  actions and does not ask who produced them. Either architecture works
  unchanged.
- **The numeric budgets.** `envs/budgets.py` ships defaults; they are a
  starting point, and sweeping them is an experiment, not a tuning chore.
- **Dual ascent vs PID-Lagrangian.** Entirely outside.

---

## Known sharp edges

- **Layout variation only happens at `reset`.** A trainer that never resets
  trains on one map forever. Geometry cannot move under driving vehicles.
- **`n_agents` is an upper bound on density.** With arrivals staggered and
  crashes retiring vehicles, the mean on-road count is lower — set
  `n_agents` above the density you want.
- **Determinism is per seed.** Same seed, same trajectories; verified. But
  `reset()` without a seed advances the RNG, so an evaluation loop that
  wants repeatability must pass one.
- **No validation against real trajectory data.** Nobody owns this yet, and
  `paper-idea.md` flags it as the gap most likely to sink a submission.
