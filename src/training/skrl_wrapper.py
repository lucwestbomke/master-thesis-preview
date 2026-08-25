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
from skrl.resources.preprocessors.torch import RunningStandardScaler
from torch import Tensor

from ..env.core import ACTION_DIM, EGO_DIM, FLAT_DIM, GAMMA, BatchedSwarmEnv, unpack_flat

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
# `value_preprocessor=RunningStandardScaler` is the third, and it fails quietly
# rather than loudly too. Returns here are of order 300 (gamma=0.997 over a
# 600-step episode whose dominant term is a per-step indicator), and an
# unnormalised critic target of that scale makes the value loss dwarf the policy
# loss at a shared learning rate. The scaler standardises the target and inverts
# on the way back out, so nothing downstream sees the change. `size=1` because
# it normalises the value *output*, not the state -- the critic's global state is
# already unit-scaled by `core._critic_state`, which is why no
# `state_preprocessor` is set alongside it.
#
# `gae_lambda` is left alone: skrl already defaults it to 0.95.
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


class Float32RunningStandardScaler(RunningStandardScaler):
    """skrl's value scaler with float32 moments, for Apple silicon only.

    The stock scaler keeps `running_mean`, `running_variance` and
    `current_count` in **float64**, which MPS cannot allocate at all -- so on a
    local dev box the correct learner config simply refuses to build. The
    arithmetic is unchanged: `_compute` already casts the buffers to float32
    before using them, so only the accumulators differ.

    Used **only** where float64 is unavailable, and `mappo_cfg` selects it by
    device rather than by preference. float64 accumulators are the right choice
    on CUDA: the parallel-variance update adds `delta * count / total`, and over
    a 10 M-step run `total` reaches 1e8, where a float32 increment can vanish
    into the mean it is updating. Local MPS runs are pilots and toy configs
    (AGENTS.md), so the looser accumulator is acceptable there and nowhere else.
    """

    def __init__(self, size, *, epsilon: float = 1e-8, clip_threshold: float = 5.0, device=None):
        super().__init__(size, epsilon=epsilon, clip_threshold=clip_threshold, device="cpu")
        for name in ("running_mean", "running_variance", "current_count"):
            self.register_buffer(name, getattr(self, name).to(torch.float32))
        self.device = torch.device(device) if device is not None else self.device
        self.to(self.device)


def _require_device(device: torch.device | str | None) -> None:
    """The value scaler is a module with buffers, so it lands on a device.

    Left to skrl it resolves `None` to the *global default* device -- which on a
    GPU box is `cuda` even when the env and the models are on CPU, and the
    mismatch only surfaces at the first inverse transform, several hundred lines
    from the cause. Found the first time the suite ran on a CUDA machine
    (2026-08-24): `test_mappo_runs_against_the_batched_core` builds a CPU env and
    the scaler silently landed on the GPU.
    """
    if device is None:
        raise ValueError(
            "pass device= to mappo_cfg/ppo_cfg when scale_values is on: the value "
            "preprocessor holds buffers, and skrl resolves None to the global "
            "default device, which is cuda on a GPU box even for a CPU env"
        )


def value_scaler_for(device: torch.device | str | None) -> type[RunningStandardScaler]:
    return (
        Float32RunningStandardScaler
        if torch.device(device or "cpu").type == "mps"
        else RunningStandardScaler
    )


def mappo_cfg(
    possible_agents: list[str],
    device: torch.device | str | None = None,
    scale_values: bool = True,
    **overrides: Any,
):
    """Build a `MAPPO_CFG` that is both constructible and correct for this task.

    Applies `MAPPO_OVERRIDES` and works around the empty-dict expansion bug
    above. Always build the config through here rather than instantiating
    `MAPPO_CFG` directly -- the three defaults it overrides fail silently, not
    loudly.

    `scale_values=False` exists only so the ablation "is the value scaler doing
    anything?" can be run; reported runs leave it on.
    """
    from skrl.multi_agents.torch.mappo import MAPPO_CFG

    kwargs: dict[str, Any] = dict(MAPPO_OVERRIDES)
    if scale_values:
        _require_device(device)
        kwargs["value_preprocessor"] = value_scaler_for(device)
        kwargs["value_preprocessor_kwargs"] = {
            uid: {"size": 1, "device": device} for uid in possible_agents
        }
    kwargs.update(overrides)
    for name in _PER_AGENT_KWARGS:
        kwargs.setdefault(name, {uid: {} for uid in possible_agents})
    return MAPPO_CFG(**kwargs)


def ppo_cfg(device: torch.device | str | None = None, scale_values: bool = True, **overrides: Any):
    """`PPO_CFG` with the same three non-default settings `mappo_cfg` applies.

    Used for the **recurrent** path. skrl 2.1.0 ships `PPO_RNN` but no recurrent
    MAPPO -- `skrl/multi_agents/` contains no RNN handling at all -- and since
    `SharedPolicyWrapper` already presents the swarm as ONE parameter-shared
    agent, skrl's MAPPO with a single agent id *is* PPO with a centralized
    state-based critic. The two paths run the same algorithm; only the class
    that carries the RNN state differs. `PPO_CFG` is a flat dataclass, so the
    empty-dict expansion bug `mappo_cfg` works around does not arise here.

    ⚠️ Say this accurately in the write-up: the algorithm is MAPPO (decentralized
    actors, centralized critic on the global state, parameters shared across
    homogeneous agents). `PPO_RNN` is the *implementation vehicle*, not a
    different method.
    """
    from skrl.agents.torch.ppo import PPO_CFG

    kwargs: dict[str, Any] = dict(MAPPO_OVERRIDES)
    if scale_values:
        _require_device(device)
        kwargs["value_preprocessor"] = value_scaler_for(device)
        kwargs["value_preprocessor_kwargs"] = {"size": 1, "device": device}
    kwargs.update(overrides)
    return PPO_CFG(**kwargs)


class SwarmMultiAgentWrapper(MultiAgentEnvWrapper):
    """Agent-keyed view of `BatchedSwarmEnv`, all tensors, all on device."""

    def __init__(self, env: BatchedSwarmEnv):
        super().__init__(env)
        self.core = env
        n = env.cfg.num_drones
        self._agents = [f"tactical_node_{i}" for i in range(n)]
        self._state: torch.Tensor | None = None
        self._final_flat: torch.Tensor | None = None
        self._final_state: torch.Tensor | None = None

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
        self._state, self._flat = obs["state"], obs["flat"]
        self._final_flat = extras["final_observation"]
        self._final_state = extras.get("final_state")

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

    # --- the pre-reset pair, which is what the learner must bootstrap from --- #

    def final_observations(self) -> dict[str, torch.Tensor]:
        """`next_observations` for `record_transition` -- NOT what `step` returns.

        With `auto_reset=True` the env's returned observation is already the
        *next episode's* opening wherever an episode ended, because the policy
        needs something to act on. skrl's truncation bootstrap adds
        `gamma * V(next_observations, next_states)` to the reward, so handing it
        the returned tensors values a state from an unrelated episode -- silently,
        and at gamma = 0.997 on returns of order 300. That is precisely the bias
        `time_limit_bootstrap=True` is set to remove, so getting this wrong
        cancels the fix while leaving the flag looking correct.

        On rows where no episode ended these are element-wise equal to what
        `step` returned (the env's second physics pass re-evaluates an unchanged
        state), so a training loop can pass them unconditionally.
        """
        if self._final_flat is None:
            raise RuntimeError("call step() before final_observations()")
        return self._split(self._final_flat)

    def final_states(self) -> dict[str, torch.Tensor]:
        """The critic's half of the same pair. See `final_observations`."""
        if self._final_state is None:
            raise RuntimeError(
                "the env is not emitting `final_state`. Construct it with "
                "EnvConfig(training_extras=True): without it the truncation bootstrap "
                "takes the value of a fresh episode's opening state and "
                "time_limit_bootstrap=True silently does the wrong thing."
            )
        return self._split_state(self._final_state)

    def _split_state(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
        return dict.fromkeys(self._agents, state)

    def render(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Use scripts/view_episode.py for visualization")

    def close(self) -> None:
        pass


#: The one agent a parameter-shared swarm presents to skrl. See below.
SWARM_UID = "swarm"


class SharedPolicyWrapper(MultiAgentEnvWrapper):
    """The swarm as ONE homogeneous skrl agent over `num_envs * N` rows.

    This is the wrapper training uses. `SwarmMultiAgentWrapper` above keys the
    same tensors per drone, which is the natural reading of the API and is what
    the observation-contract smoke tests exercise -- but it cannot be what a
    reported run uses, for two reasons that are both structural rather than
    performance-related:

    1. **RQ2's zero-shot transfer needs one policy, not `N` of them.** The
       matrix trains at `N = 5` and evaluates at `N in {3, 5, 8}`
       (`docs/DECISIONS.md`, settled twice). Five per-drone policies cannot be
       evaluated at eight drones at all -- there is no policy for drones 6, 7
       and 8 -- so the transfer columns RQ2 exists to measure would not exist.
    2. **Homogeneity is a claim this project makes.** `docs/REWARD.md`: the
       reward "must not depend on agent index, or homogeneity breaks and the
       'roles emerge rather than being assigned' claim collapses". A distinct
       network per drone breaks it from the other side -- roles would be
       assigned by which network you are, not discovered.

    So the actor's parameters are shared across drones, which is also what MAPPO
    does as published (Yu et al., 2022). The remaining question is only *how* to
    express that through skrl, and there the obvious route is a trap: handing
    the same `Model` object to skrl under five agent ids builds **five Adam
    optimizers over the same parameters** and runs five sequential PPO updates
    per rollout, so four of them compute their ratios against log-probabilities
    collected under a policy that has already moved. Silent, and it degrades
    exactly the trust-region property PPO is chosen for.

    Collapsing the drones into the batch dimension instead gives one optimizer,
    one update, and correct ratios -- and it is not a weaker form of MAPPO but
    the standard one: decentralized agent-local actors, one centralized critic
    on the shared global state (repeated per drone here), parameters shared
    across homogeneous agents. CTDE is unaffected; the actor still reads only
    `observations` and never `states`.

        core                              skrl
        obs["flat"]  (B, N, 108)  ->  {swarm: (B*N, 108)}
        obs["state"] (B, S)       ->  {swarm: (B*N, S)}   repeated per drone
        reward       (B, N)       ->  {swarm: (B*N, 1)}
        terminated   (B,)         ->  {swarm: (B*N, 1)}   repeated per drone
    """

    def __init__(self, env: BatchedSwarmEnv, agent_specific_state: bool = False):
        super().__init__(env)
        self.core = env
        self._agents = [SWARM_UID]
        self._state: torch.Tensor | None = None
        self._flat: torch.Tensor | None = None
        self._final_flat: torch.Tensor | None = None
        self._final_state: torch.Tensor | None = None
        self.agent_specific_state = agent_specific_state

        obs_space = gymnasium.spaces.Box(-np.inf, np.inf, shape=(FLAT_DIM,), dtype=np.float32)
        act_space = gymnasium.spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,), dtype=np.float32)
        width = env.cfg.state_dim + (EGO_DIM if agent_specific_state else 0)
        st_space = gymnasium.spaces.Box(-np.inf, np.inf, shape=(width,), dtype=np.float32)
        self._observation_spaces = {SWARM_UID: obs_space}
        self._action_spaces = {SWARM_UID: act_space}
        self._state_spaces = {SWARM_UID: st_space}

    # --- identity -------------------------------------------------------- #

    @property
    def num_envs(self) -> int:
        """Rows skrl sees: one per drone per environment."""
        return self.core.cfg.num_envs * self.core.cfg.num_drones

    @property
    def num_agents(self) -> int:
        return 1

    @property
    def max_num_agents(self) -> int:
        return 1

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

    # --- reshaping ------------------------------------------------------- #

    def _rows(self, per_drone: torch.Tensor) -> dict[str, torch.Tensor]:
        """`(B, N, D)` or `(B, N)` -> `{swarm: (B*N, D)}`."""
        flat = per_drone.reshape(self.num_envs, -1)
        return {SWARM_UID: flat}

    def _repeat(self, per_env: torch.Tensor) -> dict[str, torch.Tensor]:
        """`(B, D)` or `(B,)` -> `{swarm: (B*N, D)}`, repeated per drone."""
        n = self.core.cfg.num_drones
        x = per_env if per_env.dim() > 1 else per_env.unsqueeze(-1)
        return {SWARM_UID: x.repeat_interleave(n, dim=0)}

    def _critic_rows(self, global_state: torch.Tensor, flat: torch.Tensor) -> dict[str, Tensor]:
        """The critic's input rows: global state, optionally + the drone's own ego.

        ## Why the option exists, and it is a measured deficit rather than a knob

        Without it the critic receives **one global state, repeat_interleaved
        across the N drones**, so `V(s)` is bit-identical for every drone of an
        environment -- not approximately, exactly. The reward is team-dominated,
        so `A_i = r_i + gamma*V(s') - V(s)` is then nearly identical too.
        Measured by `scripts/probe_credit.py`: **0.015-0.06 %** of advantage
        variance distinguishes one drone from another, and `max |V_i - V_j|` is
        `0.000e+00`, trained or not.

        Every drone's policy gradient is therefore
        `grad log pi(a_i | o_i) * A` with the same `A` -- each is told *the team
        did well*, never *your action was the good one*. Role differentiation
        cannot be learned from a signal that is constant across the agents it
        would differentiate, which is the Block G diagnosis in one sentence.

        Concatenating the drone's **own ego block** makes `V_i` a function of
        drone `i`, which is the term the probe measures as exactly zero. This is
        Yu et al. (2022)'s *agent-specific global state* -- their single largest
        MAPPO recommendation -- and it is the configuration this project has not
        been running.

        ⛔ It does **not** break "never give the actor an agent index"
        (`AGENTS.md`). That rule is about the **actor**: this is the critic, it is
        training-only and discarded at evaluation, and the appended features are
        the drone's own *state* -- position, velocity, battery, clearances,
        `sees_hvt`, `on_path` -- never its identity. The construction stays
        permutation-equivariant, so roles must still emerge.

        ⚠️ The ego block is taken through `core.unpack_flat`, the sanctioned
        inverse of the env's own packing, so the critic sees exactly the block the
        actor does rather than a second hand-rolled slice of it.
        """
        rows = self._repeat(global_state)[SWARM_UID]
        if not self.agent_specific_state:
            return {SWARM_UID: rows}
        ego = unpack_flat(flat)["ego"].reshape(self.num_envs, EGO_DIM)
        return {SWARM_UID: torch.cat([rows, ego], dim=-1)}

    # --- interaction ------------------------------------------------------ #

    def reset(self) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        obs = self.core.reset()
        self._state, self._flat = obs["state"], obs["flat"]
        return self._rows(obs["flat"]), {SWARM_UID: {}}

    def step(self, actions: dict[str, torch.Tensor]):
        b, n = self.core.cfg.num_envs, self.core.cfg.num_drones
        act = actions[SWARM_UID].view(b, n, ACTION_DIM)
        obs, rew, terminated, truncated, extras = self.core.step(act)
        self._state, self._flat = obs["state"], obs["flat"]
        self._final_flat = extras["final_observation"]
        self._final_state = extras.get("final_state")
        return (
            self._rows(obs["flat"]),
            self._rows(rew),
            self._repeat(terminated),
            self._repeat(truncated),
            {SWARM_UID: extras},
        )

    def state(self) -> dict[str, torch.Tensor | None]:
        return self._critic_rows(self._state, self._flat)

    def final_observations(self) -> dict[str, torch.Tensor]:
        """The pre-reset observation -- see `SwarmMultiAgentWrapper` for why."""
        if self._final_flat is None:
            raise RuntimeError("call step() before final_observations()")
        return self._rows(self._final_flat)

    def final_states(self) -> dict[str, torch.Tensor]:
        if self._final_state is None:
            raise RuntimeError(
                "the env is not emitting `final_state`. Construct it with "
                "EnvConfig(training_extras=True): without it the truncation bootstrap "
                "takes the value of a fresh episode's opening state and "
                "time_limit_bootstrap=True silently does the wrong thing."
            )
        # ⚠️ Paired with the PRE-reset observation, not the post-reset one: the
        # bootstrap values the state the episode actually ended in, and the ego
        # half has to come from the same instant as the global half.
        return self._critic_rows(self._final_state, self._final_flat)

    def render(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Use scripts/view_episode.py for visualization")

    def close(self) -> None:
        pass
