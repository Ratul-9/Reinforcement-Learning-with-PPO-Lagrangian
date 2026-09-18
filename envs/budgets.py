"""Per-type constraint budgets — the `d_i` in the Lagrangian.

    maximize E[sum r]   subject to   E[sum c_i] <= d_i

This module holds the `d_i`, one set per vehicle type. It is the single
place in the repo where a human number about *how much* a violation matters
is allowed to live, and it is deliberately separate from the env: the env
measures cost, and never weighs it.

## Why the budgets differ by vehicle

`paper-idea.md` proposes that per-agent personality should set the budgets
rather than modulate reward weights. Vehicle type is a better carrier for
that than an invented trait vector, because every difference is one someone
can argue about from physics:

    bus         tight jerk        standing passengers fall over
                loose lane_keep   twelve metres cannot stay in one lane
                                  through a turn; charging it for that is
                                  charging it for being a bus
    truck       tight speeding    stopping distance is the whole problem
                tight offroad     it does not recover from a soft verge
    motorcycle  tight ttc         no crumple zone, and it is the road user
                loose lane_keep   most likely to be killed by a mistake
                                  that a car would survive
    tuktuk      loose wrong_way   it genuinely does, everywhere it exists
                tight ttc         same vulnerability argument as the bike
    sedan       the baseline every other row is stated relative to

## The units

Costs are bounded to [0, 1] per step (see `traffic_env.COST_CHANNELS`), so
an episode's cost is a sum in "step-equivalents" and a budget reads
directly: `offroad: 5` is five steps — half a second — off the road per
episode. `collision` is the exception and is in events, because a collision
ends the episode and so can only happen once.

## These are defaults, not findings

Every number below is a starting point a human chose, and the paper's claim
is precisely that this is a *better place* to put a human number than a
penalty weight — not that the number has vanished. Sweeping them is an
experiment (`paper-idea.md`, claims 3 and 4), not a tuning chore.
"""

from __future__ import annotations

from envs.traffic_env import COST_CHANNELS

# episode length is 900 steps (90 s at 10 Hz), so a budget of 45 is 5% of
# an episode spent in violation.
DEFAULT = {
    "collision": 0.02,     # events per episode
    "offroad": 5.0,        # steps outside the drivable surface
    "wrong_way": 10.0,     # steps in a lane running the other way
    "lane_keep": 40.0,     # steps at full drift off the lane centre
    "ttc": 20.0,           # steps with a conflict under two seconds away
    "jerk": 30.0,          # steps at full harshness
    "speeding": 10.0,      # steps at double the limit
}

BY_TYPE = {
    "sedan": DEFAULT,
    "tuktuk": dict(DEFAULT, wrong_way=40.0, lane_keep=120.0, ttc=10.0,
                   jerk=40.0, speeding=20.0, offroad=10.0),
    "motorcycle": dict(DEFAULT, collision=0.01, wrong_way=40.0,
                       lane_keep=120.0, ttc=8.0, jerk=40.0, speeding=20.0,
                       offroad=10.0),
    "truck": dict(DEFAULT, collision=0.01, offroad=3.0, wrong_way=5.0,
                  lane_keep=60.0, ttc=25.0, jerk=15.0, speeding=3.0),
    "bus": dict(DEFAULT, collision=0.01, offroad=2.0, wrong_way=5.0,
                lane_keep=100.0, ttc=25.0, jerk=8.0, speeding=3.0),
}


def for_type(type_name: str) -> dict:
    """The budget vector for one vehicle type, as a plain dict."""
    return dict(BY_TYPE.get(type_name, DEFAULT))


def as_vector(type_name: str) -> list:
    """The same thing ordered by `COST_CHANNELS`, which is the order the
    observation and any multiplier vector use. One ordering, defined once —
    a learner that zips its lambdas against a differently-ordered budget
    silently optimises the wrong constraint."""
    budget = for_type(type_name)
    return [float(budget[channel]) for channel in COST_CHANNELS]


def describe() -> str:
    """The table, generated from the data so it cannot go stale."""
    head = f"{'type':12s}" + "".join(f"{c:>11s}" for c in COST_CHANNELS)
    rows = [head, "-" * len(head)]
    for name in BY_TYPE:
        budget = for_type(name)
        rows.append(f"{name:12s}" +
                    "".join(f"{budget[c]:11.2f}" for c in COST_CHANNELS))
    return "\n".join(rows)


if __name__ == "__main__":
    print(describe())
