"""D2.5 smoke test: does skrl's MAPPO accept this observation contract?

NOT a training test. No curriculum, no learning claim, nothing reported. It
exists to answer two questions while the observation contract is still free to
change, because discovering either in Block G would mean rewriting a contract
that E and F have already been built against:

1. Does skrl's rollout storage accept the 108-dim max-N packing?
2. Does the wrapper keep `terminated` and `truncated` distinct, and is
   truncation bootstrapped rather than treated as termination?

(2) is the one that bites: `time_limit_bootstrap` defaults to False in skrl
2.1.0, and with no time feature in the observation that default makes the critic
believe the world ends at 600 steps.
"""

from __future__ import annotations

import torch
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.multi_agents.torch.mappo import MAPPO
from torch import nn

from ..env.core import ACTION_DIM, FLAT_DIM, GAMMA, BatchedSwarmEnv, EnvConfig
from .skrl_wrapper import MAPPO_OVERRIDES, SwarmMultiAgentWrapper, mappo_cfg

FAST = {"use_occlusion": False, "compile_occlusion": False, "stage_weights": (1.0, 0.0, 0.0, 0.0)}


class Policy(GaussianMixin, Model):
    """Actor: agent-local only. Reads `observations`, never `states`."""

    def __init__(self, obs_space, act_space, device):
        Model.__init__(self, observation_space=obs_space, action_space=act_space, device=device)
        GaussianMixin.__init__(self, clip_actions=True)
        self.net = nn.Sequential(nn.Linear(FLAT_DIM, 32), nn.Tanh(), nn.Linear(32, ACTION_DIM))
        self.log_std = nn.Parameter(torch.zeros(ACTION_DIM))

    def compute(self, inputs, role=""):
        # skrl 2.1: compute returns (value, outputs) and log_std travels in outputs
        return torch.tanh(self.net(inputs["observations"])), {"log_std": self.log_std}


class Value(DeterministicMixin, Model):
    """Centralized critic: reads the shared `states`, discarded at evaluation."""

    def __init__(self, state_space, act_space, device):
        Model.__init__(self, observation_space=state_space, action_space=act_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=False)
        self.net = nn.Sequential(nn.Linear(state_space.shape[0], 32), nn.Tanh(), nn.Linear(32, 1))

    def compute(self, inputs, role=""):
        return self.net(inputs["states"]), {}


def build(num_envs=4, num_drones=3, rollouts=8):
    core = BatchedSwarmEnv(EnvConfig(num_envs=num_envs, num_drones=num_drones, seed=0, **FAST))
    env = SwarmMultiAgentWrapper(core)
    dev = env.device
    models, memories = {}, {}
    for uid in env.agents:
        models[uid] = {
            "policy": Policy(env.observation_spaces[uid], env.action_spaces[uid], dev),
            "value": Value(env.state_spaces[uid], env.action_spaces[uid], dev),
        }
        memories[uid] = RandomMemory(memory_size=rollouts, num_envs=env.num_envs, device=dev)

    cfg = mappo_cfg(env.possible_agents, rollouts=rollouts, learning_epochs=1, mini_batches=1)
    agent = MAPPO(
        possible_agents=env.possible_agents,
        models=models,
        memories=memories,
        cfg=cfg,
        observation_spaces=env.observation_spaces,
        state_spaces=env.state_spaces,
        action_spaces=env.action_spaces,
        device=dev,
    )
    return env, agent


def test_wrapper_shapes_match_what_skrl_expects():
    env = SwarmMultiAgentWrapper(BatchedSwarmEnv(EnvConfig(num_envs=4, num_drones=3, **FAST)))
    obs, _ = env.reset()
    assert set(obs) == set(env.agents)
    for uid in env.agents:
        assert obs[uid].shape == (4, FLAT_DIM)

    actions = {uid: torch.zeros(4, ACTION_DIM, device=env.device) for uid in env.agents}
    nobs, rew, term, trunc, _ = env.step(actions)
    for uid in env.agents:
        assert nobs[uid].shape == (4, FLAT_DIM)
        assert rew[uid].shape == (4, 1)
        assert term[uid].shape == (4, 1) and term[uid].dtype == torch.bool
        assert trunc[uid].shape == (4, 1) and trunc[uid].dtype == torch.bool
    assert env.state()[env.agents[0]].shape == (4, env.state_spaces[env.agents[0]].shape[0])


def test_truncation_is_distinct_from_termination_and_bootstrapped():
    """The failure this guards against is silent: with `time_limit_bootstrap`
    left at skrl's default, a truncated episode is treated as terminal and the
    critic learns a horizon the task does not have."""
    assert MAPPO_OVERRIDES["time_limit_bootstrap"] is True
    assert MAPPO_OVERRIDES["discount_factor"] > 0.99, "AGENTS.md rules out 0.99"

    env, agent = build()
    cfg = agent.cfg
    for uid in env.possible_agents:
        assert cfg.time_limit_bootstrap[uid] is True
        assert cfg.discount_factor[uid] > 0.99

    # and the two flags really do differ at an episode boundary
    env.reset()
    zero = {uid: torch.zeros(4, ACTION_DIM, device=env.device) for uid in env.agents}
    saw_truncation = False
    for _ in range(150):
        _, _, term, trunc, _ = env.step(zero)
        assert not term[env.agents[0]].any(), "hovering must never terminate"
        saw_truncation |= bool(trunc[env.agents[0]].any())
    assert saw_truncation, "a 150-step stage must truncate within 150 steps"


def test_mappo_runs_against_the_batched_core():
    """The seam itself: rollout storage accepts the 108-dim packing, MAPPO
    completes an update, and parameters actually move."""
    rollouts = 8
    env, agent = build(rollouts=rollouts)
    agent.init()
    agent.enable_training_mode(True)  # skrl agents start in eval mode

    before = {
        uid: [p.detach().clone() for p in agent.policies[uid].parameters()]
        for uid in env.possible_agents
    }

    obs, _ = env.reset()
    states = env.state()
    total = rollouts * 2
    for timestep in range(total):
        agent.pre_interaction(timestep=timestep, timesteps=total)
        with torch.no_grad():
            actions = agent.act(obs, states, timestep=timestep, timesteps=total)[0]
        next_obs, rewards, terminated, truncated, infos = env.step(actions)
        next_states = env.state()
        agent.record_transition(
            observations=obs,
            states=states,
            actions=actions,
            rewards=rewards,
            next_observations=next_obs,
            next_states=next_states,
            terminated=terminated,
            truncated=truncated,
            infos=infos,
            timestep=timestep,
            timesteps=total,
        )
        agent.post_interaction(timestep=timestep, timesteps=total)
        obs, states = next_obs, next_states

    moved = any(
        not torch.equal(p0, p1)
        for uid in env.possible_agents
        for p0, p1 in zip(before[uid], agent.policies[uid].parameters(), strict=True)
    )
    assert moved, "MAPPO ran but no policy parameter changed -- the update did not happen"


def test_learner_and_env_discount_the_same_way():
    """PBRS invariance holds only if the shaping gamma equals the agent's gamma.

    `reward.shaping` adds `gamma*Phi(s') - Phi(s)`. That telescopes to a
    policy-independent constant -- and so provably cannot move the optimum -- only
    when the two gammas match. If the env shapes at 0.997 while MAPPO discounts
    at skrl's default 0.99, the shaping stops being potential-based and silently
    becomes a bias on exactly the term chosen for being unbiased.
    """
    env, agent = build(num_envs=2, num_drones=2)
    assert env.core.cfg.gamma == GAMMA
    # post-expansion, which is the form MAPPO actually discounts with
    for uid in env.possible_agents:
        assert agent.cfg.discount_factor[uid] == env.core.cfg.gamma
