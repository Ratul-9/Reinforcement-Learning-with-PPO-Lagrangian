# Project context

Read this first in a new session. It is orientation and *rationale* — the
decisions a fresh reader would otherwise re-litigate. The reference material
lives elsewhere and is not duplicated here:

| document | what it covers |
|---|---|
| [`Envs.md`](Envs.md) | what the environment **is** — scenarios, fleet, costs, sensors, rendering, PNG maps |
| [`Handoff.md`](Handoff.md) | the env/learner **contract**. Read before touching the interface |
| [`Vehicle.md`](Vehicle.md) | the original single-vehicle physics doc (predates the fleet) |
| `../LANCER3D-dev/agent-files/paper-idea.md` | the **research framing**: the claim, the baselines, the venues |

---

## What this repo is

The environments for a constrained multi-agent driving RL study. The method
is **PPO-Lagrangian**: guidance rewards (lane keeping, smoothness, headway)
are reformulated as *constraints with learned Lagrange multipliers*, so the
arbitrary penalty weight is replaced by an interpretable safety budget.

The one-line thesis, and the reason the code is shaped the way it is:

> The env measures cost and **never weighs it**. Nothing in `envs/`
> multiplies a cost by a weight, because the optimiser is supposed to derive
> that weight from a budget. A `reward_weights` dict appearing anywhere in
> this repo would defeat the entire point of the study.

**Scope split:** this repo is the environments (Shinjini). The trainer is
someone else's (Ratul), written against `Handoff.md`.

## Working rules

- **Always `conda run -n py310 python ...`** — never base/system Python.
- **No Claude attribution** in commits or PRs. No `Co-Authored-By`, no
  generated-with footer. Four commits were rewritten once to strip them.
- Run `conda run -n py310 python test_envs.py` after any change — 37 checks,
  ~90 s. It is the contract and the regression net.
- `figures/` is gitignored; regenerate with `python figures.py`.

## Lineage

This repo is a deliberate **simplification** of `../LANCER3D-dev` (24k lines:
PyQt6 GUI, Panda3D Bullet physics, a studio road editor, a single-agent env).
All of that was dropped. What came across:

- `envs/road_network.py` — ported from `LANCER3D-dev/sim/world/road_network.py`.
  Pure numpy geometry, the eight scenario builders, routing. It began
  byte-identical and has diverged in **exactly one place**: `_project_full`
  gained a bounding-box broad phase and `finalize` builds its index. Keep it
  otherwise diffable against the original.
- `vehicle_data.xlsx` in that repo supplied the fleet's physical parameters.

---

## Design decisions worth not re-litigating

Each of these cost real time to arrive at. The *why* matters more than the
what.

**Reward carries only goal and progress; everything else is a named cost
channel.** Seven channels, all bounded to `[0, 1]` per step. The bound is
what makes a budget writable: an episode's cost is a sum, so `offroad: 5`
reads as "five steps off the road". Before the bound, `jerk` hit 21.3 in one
step and totalled 2160 over a window where the whole reward totalled 4.4 —
`λ_jerk` would have had to converge near 1e-3 while `λ_collision` sat near 1,
which is how dual ascent oscillates into a do-nothing policy.

**Sensors are analytic, not rendered.** Closed-form ray/shape intersection
against the same geometry the renderers draw. At 20–30 agents a GPU
round-trip per agent per step would cost more than the entire rest of the
step. Rendering exists as a *viewer*, with a driver's-eye camera reserved as
the pixel-observation hook if an experiment ever calls for one.

**Route waypoints are shifted into their lane.** `route_probe` answers with
centreline points because a centreline graph is all it has. Steering at one
means driving down the middle of the carriageway — the oncoming lane half
the time — so `wrong_way` fired on a third of all steps *by construction*.
A constraint violated that often at initialisation is the collapse the
method has to design around.

**Vehicle type sets the cost budgets.** This is `paper-idea.md`'s
personality-as-budget idea, grounded in physics rather than an invented
trait vector. The heterogeneity is real and measured, not asserted:
`lane_keep` per step runs motorcycle 0.018 → tuktuk 0.066 → sedan 0.117 →
truck 0.283 → bus 0.332, unprompted, because big vehicles physically cannot
hold a lane centre.

**The policy is told its physics, not its name** — eight scaled parameters,
not a one-hot. One shared policy then spans the fleet and could drive a type
it never saw; a one-hot cannot, and grows with every type added.

**Arrivals are staggered and layouts vary per reset.** Both exist so the
policy cannot memorise a schedule or a map. Layout variation happens at
`reset` **only** — moving geometry under driving vehicles would teleport
them off the road.

**Actuators lag and sensors are noisy, both on by default.** A policy
trained against instant actuators and exact sensors learns to depend on
both. Noise touches the observation only — costs and metrics are ground
truth, because a constraint measured through a noisy sensor measures the
sensor.

**Things deliberately NOT modelled**, so nobody rebuilds them by accident:
pedestrians and moving non-vehicles (dropped from scope by the user),
traffic lights and right-of-way (the scenarios are collision-prone by
geometry instead), roll-over (tippy vehicles carry reduced grip as a
stand-in; a real model needs an eighth cost channel, and six constraints is
already where dual ascent oscillates), leaning, and articulated vehicles.

---

## Bugs that were found the hard way

Worth knowing, because each was invisible until something specific was built.

- **Spawn clearance was tested before the lane snap moved the vehicle.**
  Harmless while spawns were spread across a half-carriageway; once every
  spawn landed exactly on a lane centre it put 18 of 20 vehicles on top of
  each other.
- **TTC reported `inf` for a stationary vehicle three metres ahead.** Inside
  the conflict disc the quadratic's near root is negative, and filtering to
  positive roots discarded exactly the most dangerous geometry. The channel
  was blind to the conflicts it exists to catch.
- **Jerk was measuring the control rate, not the ride.** Differencing raw
  step-to-step acceleration at 10 Hz trips a 5 m/s³ threshold constantly.
  Now low-passed (τ = 0.3 s) before differencing.
- **PNG round trip silently lost every parking bay.** A 5.5 m stub on a 10 m
  road is absorbed into the road's own medial axis, so skeletonising leaves
  no branch. Bays are rebuilt from their marker dot instead.
- **Panda3D back-face culling made every road invisible.** Winding order is
  load-bearing: a clockwise upward-facing quad is simply not drawn.
- **Explicit Euler was unstable below 6.7 m/s** — and the friction clip hid
  it, turning divergence into a limit cycle that read as noise. The tuktuk
  oscillated ±1.16 rad/s for a constant steering input. Physics is now
  sub-stepped ten times per control step. ~70% of the measured `jerk` cost
  had been integration noise.
- **The progress reward was farmable.** Re-solving the route every step gave
  jumps of +255 m, worth more than arriving. The route is now solved once
  per episode and walked, and reward is clamped to distance driven.
- **The rear tyres used the front axle's load** — `Fz_r` computed, never
  used, ~33% free rear grip.

---

## Running things

```bash
conda run -n py310 python test_envs.py       # 37 checks — run after any change
conda run -n py310 python rollout.py         # scripted driver, all 8 scenarios
conda run -n py310 python figures.py         # presentation renders -> figures/
conda run -n py310 python fleet.py           # the vehicle table
conda run -n py310 python -m envs.budgets    # the budget table
conda run -n py310 python -m envs.vec        # parallel throughput benchmark
```

`rollout.py` is the sanity check that must pass before any training run is
worth starting: *can anything at all reach a goal here?* It has no collision
avoidance, so its crashes are information, not a failure.

## Performance, and this laptop

~2000 agent-steps/s on one core. `envs/vec.py` runs one env per worker
process; use `performance_cores()`, which reads macOS performance cores,
Linux physical cores and cgroup quotas — `os.cpu_count()` is wrong in both
directions and the parent waits for every worker each step.

This machine is an M4 MacBook Air (4 performance + 6 efficiency cores). The
parallel plateau at ~5000 agent-steps/s is the laptop, not the design:
coordination efficiency measured 0.84, and per-worker *compute* slows 2.1×
at four workers and 3.7× at eight. Measured throughput also swings ~3× with
whatever else is open — a video call cut it to 1500.

For real training: a Linux box, many physical cores, and
`requirements-train.txt` (numpy, scipy, gymnasium, torch). The training path
imports **no** Panda3D, matplotlib, OpenCV or PIL, so a headless container
needs no OpenGL and no system packages. The bottleneck is CPU, not GPU — the
policy is a small MLP, so size the machine by core count.

---

## State

Env side is complete and pushed. Remaining:

1. **The trainer** — not this repo's scope. `Handoff.md` is the contract.
2. **Validation against real trajectory data** — *unowned*. `paper-idea.md`
   names it as the gap most likely to sink a submission. Raise it rather
   than letting it surface at review.
3. Optional, not blocking: a budget-sweep harness (claim 4 in
   `paper-idea.md` needs one) and an IDM/MOBIL rule-based baseline (a
   required baseline, and env-side rather than learner-side).
