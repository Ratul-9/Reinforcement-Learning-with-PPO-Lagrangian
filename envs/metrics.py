"""Episode metrics — the numbers the results section is made of.

`TrafficEnv` reports per-step costs and nothing else. That is correct for an
env, and useless for a paper: `paper-idea.md` asks for collision rate,
goal-completion rate, time-to-goal, a near-miss / minimum-TTC distribution,
constraint-violation rate and sample efficiency, and every one of those is a
statistic over *episodes*.

    recorder = EpisodeRecorder()
    for ... :
        obs, rew, term, trunc, info = env.step(actions)
        recorder.update(info, term, trunc, rewards=rew)
    print(recorder.report())
    recorder.write_csv("run.csv")

## Why it tracks agents rather than steps

With staggered arrivals every agent is at a different point of its own
episode, and the env respawns one the moment it finishes. So an "episode" is
a per-agent span between terminal flags, not a slice of wall-clock, and
anything that aggregates by step silently averages a bus three seconds into
its journey with a motorcycle ninety seconds into its own.

## Constraint satisfaction is the headline

The claim under test is that the constraint is actually *met* — that `J_c`
at convergence sits inside `d_i`. `report()` puts the realised per-episode
cost next to the budget for every channel, because a run whose collision
rate looks good while its collision cost sits at three times its budget has
not demonstrated the thing the paper says it demonstrates.

Cost budgets are read from `info["budget"]`, so a per-vehicle-type budget
table is compared per vehicle type without this module knowing one exists.

## Deliberately not here

Lambda trajectories. They belong to the optimiser, not the environment, and
a recorder that invented a place to put them would be guessing at the
learner's update schedule. Pass them to `report(extra=...)` or log them
alongside.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict

import numpy as np

from envs.traffic_env import COST_CHANNELS

# Outcomes an episode can have. "timeout" is not a failure and not a
# success — a vehicle that was still driving sensibly when the clock ran out
# is a different thing from one that crashed, and folding them together is
# how a goal rate quietly becomes uninterpretable.
OUTCOMES = ("goal", "collision_vehicle", "collision_static", "offroad",
            "timeout")


class EpisodeRecorder:
    """Accumulates per-agent episodes from what `TrafficEnv.step` returns."""

    def __init__(self, cost_channels=COST_CHANNELS, dt: float = 0.1):
        self.channels = tuple(cost_channels)
        self.dt = float(dt)
        self.episodes: list[dict] = []
        self._open: dict[int, dict] = {}

    # -- collection -------------------------------------------------------

    def update(self, info, terminated, truncated, rewards=None) -> None:
        """One env step. Call it every step, including ones where nothing
        finished — that is where the cost is accumulated."""
        active = np.asarray(info["active"])
        terminated = np.asarray(terminated)
        truncated = np.asarray(truncated)
        costs = info["cost"]
        types = info.get("vehicle_type")
        budgets = info.get("budget")
        ttc = np.asarray(info.get("ttc", np.full(len(active), np.inf)))
        scenario = info.get("scenario", "?")

        for i in np.flatnonzero(active):
            ep = self._open.get(i)
            if ep is None:
                ep = self._open[i] = {
                    "scenario": scenario,
                    "vehicle_type": types[i] if types else "?",
                    "steps": 0, "reward": 0.0, "min_ttc": math.inf,
                    **{f"cost_{c}": 0.0 for c in self.channels},
                }
                if budgets is not None:
                    ep.update({f"budget_{c}": float(budgets[i][k])
                               for k, c in enumerate(self.channels)})
            ep["steps"] += 1
            if rewards is not None:
                ep["reward"] += float(rewards[i])
            for c in self.channels:
                ep[f"cost_{c}"] += float(costs[c][i])
            if np.isfinite(ttc[i]):
                ep["min_ttc"] = min(ep["min_ttc"], float(ttc[i]))

        events = info.get("events") or [""] * len(active)
        for i in np.flatnonzero(terminated | truncated):
            ep = self._open.pop(int(i), None)
            if ep is None:
                continue
            # The event names the cause; a truncation that names no event is
            # the clock running out, which is its own outcome.
            ep["outcome"] = events[i] or "timeout"
            ep["seconds"] = ep["steps"] * self.dt
            self.episodes.append(ep)

    # -- reporting --------------------------------------------------------

    def summary(self) -> dict:
        """Headline numbers over every completed episode."""
        if not self.episodes:
            return {"episodes": 0}

        out = {"episodes": len(self.episodes)}
        outcomes = [e["outcome"] for e in self.episodes]
        for name in OUTCOMES:
            out[f"rate_{name}"] = outcomes.count(name) / len(outcomes)
        out["rate_collision"] = (out["rate_collision_vehicle"]
                                 + out["rate_collision_static"])

        # Time-to-goal over the episodes that REACHED the goal. Averaging it
        # over failures too would let a policy improve the number by
        # crashing sooner.
        reached = [e["seconds"] for e in self.episodes if e["outcome"] == "goal"]
        out["time_to_goal_mean"] = float(np.mean(reached)) if reached else math.nan
        out["reward_mean"] = float(np.mean([e["reward"] for e in self.episodes]))

        near = [e["min_ttc"] for e in self.episodes if math.isfinite(e["min_ttc"])]
        if near:
            out["min_ttc_p05"] = float(np.percentile(near, 5))
            out["min_ttc_median"] = float(np.median(near))
        return out

    def constraints(self) -> list[dict]:
        """Realised per-episode cost against the budget, per channel.

        Grouped by vehicle type, because the budgets are per type: pooling a
        bus's loose lane budget with a sedan's tight one and comparing the
        pool to either is meaningless.
        """
        rows = []
        by_type = defaultdict(list)
        for ep in self.episodes:
            by_type[ep["vehicle_type"]].append(ep)

        for vehicle_type, episodes in sorted(by_type.items()):
            for channel in self.channels:
                realised = np.mean([e[f"cost_{channel}"] for e in episodes])
                budget_key = f"budget_{channel}"
                budget = (float(np.mean([e[budget_key] for e in episodes]))
                          if budget_key in episodes[0] else math.nan)
                violated = np.mean([e[f"cost_{channel}"] > e[budget_key]
                                    for e in episodes]) \
                    if budget_key in episodes[0] else math.nan
                rows.append({"vehicle_type": vehicle_type, "channel": channel,
                             "episodes": len(episodes), "cost": float(realised),
                             "budget": budget, "violation_rate": float(violated),
                             "satisfied": bool(realised <= budget)
                             if math.isfinite(budget) else None})
        return rows

    def report(self, extra: dict | None = None) -> str:
        """The whole thing as text, for a log or a terminal."""
        summary = self.summary()
        if not summary["episodes"]:
            return "no completed episodes"

        lines = [f"episodes {summary['episodes']}   "
                 f"goal {summary['rate_goal']:.1%}   "
                 f"collision {summary['rate_collision']:.1%}   "
                 f"offroad {summary['rate_offroad']:.1%}   "
                 f"timeout {summary['rate_timeout']:.1%}"]
        if not math.isnan(summary["time_to_goal_mean"]):
            lines.append(f"time to goal {summary['time_to_goal_mean']:.1f}s   "
                         f"mean reward {summary['reward_mean']:.3f}")
        if "min_ttc_median" in summary:
            lines.append(f"min TTC  median {summary['min_ttc_median']:.2f}s   "
                         f"5th pct {summary['min_ttc_p05']:.2f}s")

        lines.append("")
        head = (f"{'type':11s}{'channel':11s}{'cost':>9s}{'budget':>9s}"
                f"{'viol':>7s}  ok")
        lines.append(head)
        lines.append("-" * len(head))
        for row in self.constraints():
            mark = "" if row["satisfied"] is None else ("ok" if row["satisfied"] else "OVER")
            lines.append(f"{row['vehicle_type']:11s}{row['channel']:11s}"
                         f"{row['cost']:9.3f}{row['budget']:9.3f}"
                         f"{row['violation_rate']:7.1%}  {mark}")
        if extra:
            lines.append("")
            lines.extend(f"{k}: {v}" for k, v in extra.items())
        return "\n".join(lines)

    # -- artifacts --------------------------------------------------------

    def write_csv(self, path: str) -> str:
        """One row per episode — the raw material for any plot later."""
        if not self.episodes:
            raise ValueError("no completed episodes to write")
        fields = list(self.episodes[0])
        for ep in self.episodes:
            fields.extend(k for k in ep if k not in fields)
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.episodes)
        return path

    def write_json(self, path: str, extra: dict | None = None) -> str:
        """Summary and constraint table, for a run manifest."""
        payload = {"summary": self.summary(), "constraints": self.constraints()}
        if extra:
            payload["extra"] = extra
        with open(path, "w") as handle:
            json.dump(payload, handle, indent=2, default=str)
        return path
