"""skrl multi-agent wrapper over the batched env core.

This is the training path. It exists because skrl's own `PettingZooWrapper`
round-trips every action and observation through NumPy on each step
(`untensorize_space` / `tensorize_space`, verified in the installed 2.1.0) and
exposes `num_envs == 1` -- which contradicts the project's stay-in-VRAM rule and
caps throughput at single-env Python speed. See docs/DECISIONS.md.

The wrapper is deliberately thin: reshape tensors that are already on the right
device into the agent-keyed dicts skrl expects. Nothing is copied to the host.

    core                          skrl
    obs["flat"]  (B, N, 108)  ->  {agent_i: (B, 108)}
    obs["state"] (B, S)       ->  {agent_i: (B, S)}   shared, for the critic
    reward       (B, N)       ->  {agent_i: (B, 1)}
    terminated   (B,)         ->  {agent_i: (B, 1)}

Two skrl defaults must be overridden or results are silently wrong; see
`MAPPO_OVERRIDES` below.
"""

from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np
import torch
from skrl.envs.wrappers.torch import MultiAgentEnvWrapper

from ..env.core import ACTION_DIM, FLAT_DIM, GAMMA, BatchedSwarmEnv

# --------------------------------------------------------------------------- #
# skrl defaults that are wrong for this project
# --------------------------------------------------------------------------- #
#
# `time_limit_bootstrap=False` is skrl's default and it is the dangerous one.
# Episodes here truncate at a fixed horizon that is NOT part of the task
# (docs/ENVIRONMENT.md deliberately omits a time feature on exactly that
# reasoning, following Pardo et al. 2018). Left at the default, every truncation
# is treated as a genuine terminal state, the critic learns that the world ends
# at 600 steps, and with gamma ~0.999 that bias is large and completely silent.
#
# `discount_factor=0.99` is likewise skrl's default and AGENTS.md rules it out:
# 0.99 is blind to the hard end of the episode, which is precisely where the
# 3-hop escalation lives. It is taken from `core.GAMMA` rather than written here,
# because the env's PBRS shaping uses the same constant and the invariance proof
# requires the two to be identical -- see the comment on `core.GAMMA`.
#
# `gae_lambda` is left alone: skrl already defaults it to 0.95.
#
# TODO(Block G): pair this discount with skrl's `value_preprocessor`
# (`RunningStandardScaler`). Returns are of order 300 at gamma=0.997 and the
# critic has to fit that scale. Not set here because the preprocessor needs the
# state width and belongs with the training config rather than the env seam.
MAPPO_OVERRIDES: dict[str, Any] = {
    "time_limit_bootstrap": True,
    "discount_factor": GAMMA,
}

# skrl 2.1.0 cannot construct MAPPO with its own default config. `MAPPO_CFG`
# declares four `*_kwargs` fields defaulting to `{}`, and `Config.expand()`
# rejects any dict whose keys are a strict subset of `possible_agents` -- which
# an empty dict always is. Supplying them pre-expanded per agent is the fix.
_PER_AGENT_KWARGS = (
    "learning_rate_scheduler_kwargs",
    "observation_preprocessor_kwargs",
    "state_preprocessor_kwargs",
    "value_preprocessor_kwargs",
)


def mappo_cfg(possible_agents: list[str], **overrides: Any):
    """Build a `MAPPO_CFG` that is both constructible and correct for this task.

    Applies `MAPPO_OVERRIDES` and works around the empty-dict expansion bug
    above. Always build the config through here rather than instantiating
    `MAPPO_CFG` directly -- the two defaults it overrides fail silently, not
    loudly.
    """
    from skrl.multi_agents.torch.mappo import MAPPO_CFG

    kwargs: dict[str, Any] = dict(MAPPO_OVERRIDES)
    kwargs.update(overrides)
    for name in _PER_AGENT_KWARGS:
        kwargs.setdefault(name, {uid: {} for uid in possible_agents})
    return MAPPO_CFG(**kwargs)


class SwarmMultiAgentWrapper(MultiAgentEnvWrapper):
    """Agent-keyed view of `BatchedSwarmEnv`, all tensors, all on device."""

    def __init__(self, env: BatchedSwarmEnv):
        super().__init__(env)
        self.core = env
        n = env.cfg.num_drones
        self._agents = [f"tactical_node_{i}" for i in range(n)]
        self._state: torch.Tensor | None = None

        obs_space = gymnasium.spaces.Box(-np.inf, np.inf, shape=(FLAT_DIM,), dtype=np.float32)
        act_space = gymnasium.spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,), dtype=np.float32)
        state_dim = env.cfg.state_dim  # derived in core, pinned by a test there
        st_space = gymnasium.spaces.Box(-np.inf, np.inf, shape=(state_dim,), dtype=np.float32)

        self._observation_spaces = dict.fromkeys(self._agents, obs_space)
        self._action_spaces = dict.fromkeys(self._agents, act_space)
        self._state_spaces = dict.fromkeys(self._agents, st_space)

    # --- identity -------------------------------------------------------- #

    @property
    def num_envs(self) -> int:
        return self.core.cfg.num_envs

    @property
    def num_agents(self) -> int:
        return len(self._agents)

    @property
    def max_num_agents(self) -> int:
        return len(self._agents)

    @property
    def agents(self) -> list[str]:
        return self._agents

    @property
    def possible_agents(self) -> list[str]:
        return self._agents

    @property
    def observation_spaces(self) -> dict[str, gymnasium.Space]:
        return self._observation_spaces

    @property
    def action_spaces(self) -> dict[str, gymnasium.Space]:
        return self._action_spaces

    @property
    def state_spaces(self) -> dict[str, gymnasium.Space]:
        return self._state_spaces

    # --- interaction ------------------------------------------------------ #

    def _split(self, flat: torch.Tensor) -> dict[str, torch.Tensor]:
        return {uid: flat[:, i] for i, uid in enumerate(self._agents)}

    def reset(self) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        obs = self.core.reset()
        self._state = obs["state"]
        return self._split(obs["flat"]), {uid: {} for uid in self._agents}

    def step(self, actions: dict[str, torch.Tensor]):
        act = torch.stack([actions[uid] for uid in self._agents], dim=1)
        obs, rew, terminated, truncated, extras = self.core.step(act)
        self._state = obs["state"]

        term = terminated.view(-1, 1)
        trunc = truncated.view(-1, 1)
        infos = dict.fromkeys(self._agents, extras)
        return (
            self._split(obs["flat"]),
            {uid: rew[:, i : i + 1] for i, uid in enumerate(self._agents)},
            dict.fromkeys(self._agents, term),
            dict.fromkeys(self._agents, trunc),
            infos,
        )

    def state(self) -> dict[str, torch.Tensor | None]:
        """Global state for the centralized critic.

        Shared across agents on purpose: MODELS.md requires the critic to be
        identical across all three architecture conditions, so that RQ2 isolates
        the actor rather than confounding actor and critic.
        """
        return dict.fromkeys(self._agents, self._state)

    def render(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Use scripts/view_episode.py for visualization")

    def close(self) -> None:
        pass
