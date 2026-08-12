"""PettingZoo ParallelEnv adapter over the batched core.

SCOPE: this is the *adapter*, not the training path. It exists for PettingZoo
API-compliance tests and single-env visual debugging only.

Training runs against `core.BatchedSwarmEnv` through a custom skrl multi-agent
wrapper (`src/training/skrl_wrapper.py`). Reason, verified against the installed
stack: skrl's own `PettingZooWrapper` round-trips every action and observation
through NumPy on each step (`untensorize_space` / `tensorize_space`) and exposes
`num_envs == 1`, which contradicts the project's stay-in-VRAM rule and caps
throughput at single-env Python speed. See AGENTS.md and docs/DECISIONS.md.

Because this is explicitly off the hot path, it is the one place in `src/env/`
allowed to call `.cpu()` / `.numpy()`: PettingZoo's API is defined in terms of
NumPy, and refusing to convert would mean not implementing it.

The core is constructed with `auto_reset=False`: PettingZoo ends the episode and
waits for an explicit `reset()`, whereas the training path restarts in place.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import torch
from gymnasium import spaces
from pettingzoo import ParallelEnv

from .core import ACTION_DIM, FLAT_DIM, BatchedSwarmEnv, EnvConfig


class SwarmRelayEnv(ParallelEnv):
    """One environment, agent-keyed dicts, NumPy in and out."""

    metadata: ClassVar[dict] = {"name": "swarm_relay_v0", "render_modes": []}

    def __init__(self, num_drones: int = 5, device: str = "cpu", seed: int = 0, **cfg_kw: Any):
        cfg_kw.setdefault("compile_occlusion", False)  # single env: warmup dominates
        self.core = BatchedSwarmEnv(
            EnvConfig(
                num_envs=1,
                num_drones=num_drones,
                device=device,
                seed=seed,
                auto_reset=False,
                **cfg_kw,
            )
        )
        self.possible_agents = [f"tactical_node_{i}" for i in range(num_drones)]
        self.agents: list[str] = []
        self.render_mode = None

        self.observation_spaces = {
            a: spaces.Box(-np.inf, np.inf, shape=(FLAT_DIM,), dtype=np.float32)
            for a in self.possible_agents
        }
        # Motion only. Transmit power is fixed at 30 dBm -- adaptive Ptx was
        # tested under three separate justifications and came out null each
        # time, see docs/NEGATIVE_RESULTS.md before adding a 4th dimension back.
        self.action_spaces = {
            a: spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,), dtype=np.float32)
            for a in self.possible_agents
        }

    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    # ------------------------------------------------------------------ #

    def _split(self, flat: torch.Tensor) -> dict[str, np.ndarray]:
        arr = flat[0].detach().cpu().numpy().astype(np.float32)
        return {a: arr[i] for i, a in enumerate(self.possible_agents)}

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        self.agents = self.possible_agents[:]
        obs = self.core.reset(seed=seed)
        return self._split(obs["flat"]), {a: {} for a in self.agents}

    def step(self, actions: dict[str, np.ndarray]):
        if not self.agents:
            return {}, {}, {}, {}, {}

        act = torch.zeros(1, len(self.possible_agents), ACTION_DIM, device=self.core.device)
        for i, a in enumerate(self.possible_agents):
            if a in actions:
                act[0, i] = torch.as_tensor(
                    np.asarray(actions[a], dtype=np.float32), device=self.core.device
                )

        obs, rew, terminated, truncated, extras = self.core.step(act)
        term, trunc = bool(terminated[0]), bool(truncated[0])
        rewards = {a: float(rew[0, i]) for i, a in enumerate(self.agents)}

        # Mission status is per-step, never terminal -- it belongs in `infos`.
        info = {
            "mission_capable": bool(extras["mission_capable"][0]),
            "e2e_capacity_mbps": float(extras["e2e_capacity_mbps"][0]),
            "hop_count": int(extras["hop_count"][0]),
            "chain_occluded": bool(extras["chain_occluded"][0]),
        }
        infos = {a: dict(info) for a in self.agents}
        terminations = {a: term for a in self.agents}
        truncations = {a: trunc for a in self.agents}
        observations = self._split(obs["flat"])

        if term or trunc:
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def render(self) -> None:
        raise NotImplementedError("Use scripts/view_episode.py for visualization")

    def close(self) -> None:
        pass
