"""PettingZoo adapter tests.

The adapter is a debugging and API-compliance surface, not the training path.
Its value depends entirely on it showing the *same* world the training path
sees, so the parity test below is the one that matters -- an adapter that
quietly diverges is worse than no adapter, because every hand-inspection made
through it would be misleading.
"""

from __future__ import annotations

import numpy as np
import torch
from pettingzoo.test import parallel_api_test

from .core import ACTION_DIM, FLAT_DIM, BatchedSwarmEnv, EnvConfig
from .swarm_env import SwarmRelayEnv

FAST = {"no_buildings": True, "stage_weights": (1.0, 0.0, 0.0, 0.0)}  # stage 1: 150 steps


def test_pettingzoo_parallel_api_compliance():
    env = SwarmRelayEnv(num_drones=3, **FAST)
    parallel_api_test(env, num_cycles=200)


def test_adapter_reproduces_the_core_step_for_step():
    """The whole point of keeping it: what you inspect here is what trains."""
    n, seed = 5, 17
    adapter = SwarmRelayEnv(num_drones=n, seed=seed, **FAST)
    core = BatchedSwarmEnv(
        EnvConfig(
            num_envs=1,
            num_drones=n,
            seed=seed,
            auto_reset=False,
            compile_occlusion=False,
            **FAST,
        )
    )
    obs_a, _ = adapter.reset()
    obs_c = core.reset()
    for i, agent in enumerate(adapter.possible_agents):
        assert np.allclose(obs_a[agent], obs_c["flat"][0, i].numpy(), atol=1e-5)

    rng = np.random.default_rng(0)
    for _ in range(40):
        raw = rng.uniform(-1.0, 1.0, size=(n, ACTION_DIM)).astype(np.float32)
        obs_a, rew_a, term_a, trunc_a, _ = adapter.step(
            {a: raw[i] for i, a in enumerate(adapter.possible_agents)}
        )
        obs_c, rew_c, term_c, trunc_c, _ = core.step(torch.from_numpy(raw).unsqueeze(0))

        for i, agent in enumerate(adapter.possible_agents):
            assert np.allclose(obs_a[agent], obs_c["flat"][0, i].numpy(), atol=1e-5)
            assert abs(rew_a[agent] - float(rew_c[0, i])) < 1e-5
            assert term_a[agent] == bool(term_c[0])
            assert trunc_a[agent] == bool(trunc_c[0])


def test_episode_ends_and_can_be_restarted():
    env = SwarmRelayEnv(num_drones=3, **FAST)
    env.reset()
    zero = {a: np.zeros(ACTION_DIM, dtype=np.float32) for a in env.possible_agents}
    for step in range(1, 151):
        _, _, term, trunc, _ = env.step(zero)
        assert not any(term.values()), "hovering must never terminate"
        if step < 150:
            assert not any(trunc.values())
    assert env.agents == [], "PettingZoo requires agents to empty at episode end"
    env.reset()
    assert env.agents == env.possible_agents


def test_mission_status_is_reported_but_never_terminal():
    """Mission failure is a per-step condition. Terminating on it teaches the
    policy never to acquire (docs/DECISIONS.md)."""
    env = SwarmRelayEnv(num_drones=3, **FAST)
    env.reset()
    zero = {a: np.zeros(ACTION_DIM, dtype=np.float32) for a in env.possible_agents}
    for _ in range(30):
        _, _, term, _, infos = env.step(zero)
        assert not any(term.values())
        for info in infos.values():
            assert set(info) >= {"mission_capable", "e2e_capacity_mbps", "hop_count"}


def test_observation_matches_the_declared_space():
    env = SwarmRelayEnv(num_drones=5, **FAST)
    obs, _ = env.reset()
    for agent, value in obs.items():
        assert value.shape == (FLAT_DIM,)
        assert value.dtype == np.float32
        assert env.observation_space(agent).contains(value)
