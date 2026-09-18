"""VecTrafficEnv — several TrafficEnvs in worker processes.

    conda run -n py310 python -m envs.vec          # benchmark

One `TrafficEnv` is about 2400 agent-steps/s on one core, and the work is
numpy under the GIL rather than anything a thread pool could overlap, so the
only way past that number is more processes. Each worker owns one whole
environment — its own scenario, its own world, its own fleet — and the
parent batches actions to them and collects results.

## Why a worker holds a whole env rather than a slice of one

The agents inside a `TrafficEnv` share a world: they collide with each
other, they appear in each other's lidar. Splitting one env's agents across
processes would mean shipping every vehicle's pose to every worker on every
step, which is the entire coupling and most of the state. Splitting at the
env boundary has no shared state at all — the only thing crossing a pipe is
one env's observations.

## Mixing scenarios

Pass a list of scenario names and the workers run different ones
simultaneously. That is what a curriculum wants: a shared policy sees a
roundabout, a merge and a parking lot in the same batch, instead of
overfitting one layout for a whole run before moving on.

The per-env results are returned as lists indexed by worker, NOT concatenated
into one flat batch. Which agent belongs to which world matters to anything
that reports per-scenario metrics, and flattening throws that away — a
learner that wants a flat batch can concatenate in one line.

## How many workers

**One per PERFORMANCE core, not per logical CPU.** `os.cpu_count()` counts
efficiency cores too, and on a heterogeneous CPU those are several times
slower at this workload — use `performance_cores()` below.

Measured on an Apple M4 (4 performance + 6 efficiency cores), 20 agents,
identical work per worker:

    workers   wall    per-worker busy   parallel efficiency   agent-steps/s
      1       0.95s        0.93s              0.98                3161
      4       2.35s        1.96s              0.84                5114
      8       5.12s        3.46s              0.67                4683

Note what grows: it is not the pipe, it is the per-worker COMPUTE. Each
worker does the same amount of numpy and takes 2.1x longer at four workers
and 3.7x at eight, because past the fourth it is running on an efficiency
core. The pipe itself tops out around 4600 round-trips/s, which at 20 agents
is a ceiling of ~90k agent-steps/s — two orders of magnitude clear of
anything measured here, so plumbing is not the constraint and making it
faster would not help.

The implication for a real training box: this plateau is a property of this
laptop, not of the design. Parallel efficiency at the coordination layer is
0.84, so a machine with 16 genuine performance cores should scale close to
linearly.
"""

from __future__ import annotations

import multiprocessing as mp

import numpy as np

_CLOSE = "close"
_RESET = "reset"
_STEP = "step"


def performance_cores() -> int:
    """How many workers to run: performance cores, not logical CPUs.

    `os.cpu_count()` counts efficiency cores, and on a heterogeneous CPU
    (every Apple Silicon machine, and increasingly Intel too) a worker
    landing on one runs several times slower — so oversubscribing does not
    just fail to help, it drags the whole batch down to the slowest worker
    because the parent waits for all of them each step.
    """
    import os
    import subprocess

    try:
        out = subprocess.run(["sysctl", "-n", "hw.perflevel0.logicalcpu"],
                             capture_output=True, text=True, timeout=2)
        n = int(out.stdout.strip())
        if n > 0:
            return n
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return max(1, (os.cpu_count() or 2))


def _worker(conn, scenario: str, seed: int, kwargs: dict) -> None:
    """One environment, driven over a pipe until told to close."""
    from envs.traffic_env import TrafficEnv, batch_obs

    env = TrafficEnv(scenario, seed=seed, **kwargs)
    try:
        while True:
            cmd, payload = conn.recv()
            if cmd == _STEP:
                obs, rew, term, trunc, info = env.step(payload)
                # Batched before it crosses the pipe. A list of 20 dicts is
                # 80 small numpy arrays to pickle per env per step, and that
                # — not the physics — is what caps the process speedup.
                out = (batch_obs(obs), np.asarray(rew, dtype=np.float32),
                       np.asarray(term), np.asarray(trunc), info)
            elif cmd == _RESET:
                obs, info = env.reset(seed=payload)
                out = (batch_obs(obs), info)
            elif cmd == _CLOSE:
                conn.send(("ok", None))
                return
            else:
                raise ValueError(f"unknown command {cmd!r}")
            conn.send(("ok", out))
    except Exception as exc:                      # surface it in the parent
        # The traceback is formatted HERE: an exception object can carry
        # references that do not survive a pipe, and a worker dying with a
        # bare "connection closed" in the parent is the worst possible
        # version of this failure to debug.
        import traceback
        conn.send(("error", traceback.format_exc()))
    finally:
        conn.close()


class VecTrafficEnv:
    """`n_envs` TrafficEnvs, one per process.

    >>> vec = VecTrafficEnv(["manhattan", "merge_ramp"], n_agents=20)
    >>> obs = vec.reset()                      # list, one entry per env
    >>> acts = [[vec.action_space.sample() for _ in range(vec.n_agents)]
    ...         for _ in range(vec.n_envs)]
    >>> obs, rew, term, trunc, info = vec.step(acts)
    >>> vec.close()
    """

    def __init__(self, scenarios, n_agents: int = 20, seed: int = 0,
                 start_method: str | None = None, **env_kwargs):
        if isinstance(scenarios, str):
            scenarios = [scenarios]
        self.scenarios = list(scenarios)
        self.n_envs = len(self.scenarios)
        self.n_agents = int(n_agents)
        self._closed = False

        # "spawn" rather than "fork". macOS defaults to spawn already, and a
        # forked child that inherits a half-initialised numpy or an open GL
        # context from the renderer deadlocks in ways that look random.
        ctx = mp.get_context(start_method or "spawn")
        kwargs = dict(env_kwargs, n_agents=self.n_agents)

        self._conns = []
        self._procs = []
        for i, scenario in enumerate(self.scenarios):
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=_worker,
                               args=(child, scenario, seed + i, kwargs),
                               daemon=True)
            proc.start()
            child.close()
            self._conns.append(parent)
            self._procs.append(proc)

        # Built in the parent rather than fetched from a worker: the action
        # space is a property of the vehicle, not of the world, so asking a
        # worker would be a round trip for something already known here.
        from vehicle import Sedan
        self.action_space = Sedan().action_space

    # -- plumbing ---------------------------------------------------------

    def _send(self, cmd: str, payloads) -> list:
        for conn, payload in zip(self._conns, payloads):
            conn.send((cmd, payload))
        out = []
        for conn in self._conns:
            status, value = conn.recv()
            if status == "error":
                self.close()
                raise RuntimeError(f"worker failed:\n{value}")
            out.append(value)
        return out

    def reset(self, seed: int | None = None):
        """Returns a list of `n_envs` batched observation dicts."""
        seeds = [None if seed is None else seed + i for i in range(self.n_envs)]
        results = self._send(_RESET, seeds)
        self._infos = [info for _obs, info in results]
        return [obs for obs, _info in results]

    def step(self, actions):
        """`actions` is a list of `n_envs` action lists.

        Returns `(obs, rewards, terminated, truncated, infos)`, each a list
        of `n_envs` entries. Unlike a single `TrafficEnv`, the observations
        are BATCHED — `obs[e]["lidar"]` is `(n_agents, 120)` rather than a
        list of per-agent dicts. That is both what a policy wants to forward
        and what makes the pipe affordable; use
        `envs.traffic_env.batch_obs` to get the same shape from one env.
        """
        if len(actions) != self.n_envs:
            raise ValueError(f"expected {self.n_envs} action lists, "
                             f"got {len(actions)}")
        results = self._send(_STEP, actions)
        return tuple(list(column) for column in zip(*results))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for conn in self._conns:
            try:
                conn.send((_CLOSE, None))
            except (BrokenPipeError, OSError):
                pass
        for proc in self._procs:
            proc.join(timeout=5.0)
            if proc.is_alive():
                proc.terminate()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def benchmark(n_envs: int = 4, n_agents: int = 20, steps: int = 150) -> None:
    """One env on this core versus `n_envs` in processes, same total work."""
    import time

    from envs import road_network
    from envs.traffic_env import TrafficEnv
    from rollout import pure_pursuit

    scenarios = [road_network.SCENARIO_KINDS[i % len(road_network.SCENARIO_KINDS)]
                 for i in range(n_envs)]

    env = TrafficEnv(scenarios[0], n_agents=n_agents, seed=0)
    obs, _ = env.reset()
    start = time.time()
    for _ in range(steps):
        obs, *_ = env.step([pure_pursuit(o) for o in obs])
    single = steps * n_agents / (time.time() - start)
    print(f"1 env  in-process : {single:8.0f} agent-steps/s")

    with VecTrafficEnv(scenarios, n_agents=n_agents, seed=0) as vec:
        batch = vec.reset()
        start = time.time()
        for _ in range(steps):
            actions = [[pure_pursuit({k: v[i] for k, v in per_env.items()})
                        for i in range(n_agents)] for per_env in batch]
            batch, *_ = vec.step(actions)
        many = steps * n_agents * n_envs / (time.time() - start)
    print(f"{n_envs} envs in processes: {many:8.0f} agent-steps/s "
          f"({many / single:.1f}x)")


if __name__ == "__main__":
    import sys

    benchmark(n_envs=int(sys.argv[1]) if len(sys.argv) > 1
              else performance_cores())
