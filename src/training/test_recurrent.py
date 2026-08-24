"""The recurrent path's one non-negotiable invariant.

`docs/BLOCK_G.md` records that the GRU actor trains at `--seq-len 1` and
collapses at `--seq-len 16`, while the model itself reproduces step-mode
collection to 1.19e-7 standalone. That localises the defect to skrl's
sequence-replay path and nothing narrower.

This file holds the invariant that has to be true for PPO to mean anything at
all with a recurrent policy:

    at the FIRST learning epoch, before any parameter has moved, the recomputed
    log-probability of the stored action must equal the stored one.

If it does not, every importance ratio is wrong from the first gradient step --
the same class of silent failure as `clip_actions`, which cost this project a
day. `test_actor.py` pins the feedforward version of the same idea.
"""

from __future__ import annotations

import torch

from .skrl_wrapper import SWARM_UID
from .train import TrainConfig, build

TOY = {
    "num_envs": 16,
    "num_drones": 3,
    "device": "cpu",
    "no_buildings": True,
    "stage_weights": (1.0, 0.0, 0.0, 0.0),
    "recurrent": True,
}


def collect(cfg: TrainConfig):
    """Fill the rollout buffer exactly once, the way `train()` does."""
    env, agent, _ = build(cfg)
    agent.init()
    agent.enable_training_mode(True)
    obs, _ = env.reset()
    states = env.state()
    for t in range(cfg.rollouts):
        with torch.no_grad():
            actions = agent.act(
                obs[SWARM_UID], states[SWARM_UID], timestep=t, timesteps=cfg.rollouts
            )[0]
        next_obs, rewards, terminated, truncated, infos = env.step({SWARM_UID: actions})
        agent.record_transition(
            observations=obs[SWARM_UID],
            states=states[SWARM_UID],
            actions=actions,
            rewards=rewards[SWARM_UID],
            next_observations=env.final_observations()[SWARM_UID],
            next_states=env.final_states()[SWARM_UID],
            terminated=terminated[SWARM_UID],
            truncated=truncated[SWARM_UID],
            infos=infos,
            timestep=t,
            timesteps=cfg.rollouts,
        )
        obs, states = next_obs, env.state()
    return env, agent


def epoch_zero_delta(cfg: TrainConfig) -> torch.Tensor:
    """`|log pi_new(a) - log pi_stored(a)|` per row, replayed as `update` does."""
    _env, agent = collect(cfg)
    memory, length = agent.memory, agent._rnn_sequence_length
    names = ["observations", "actions", "terminated", "truncated", "log_prob"]
    obs, act, term, trunc, stored = memory.sample_all(
        names=names, mini_batches=1, sequence_length=length
    )[0]
    rnn = memory.sample_all(names=agent._rnn_tensors_names, mini_batches=1, sequence_length=length)[
        0
    ]

    agent.policy.enable_training_mode(True)
    with torch.no_grad():
        _, outputs = agent.policy.act(
            {
                "observations": obs,
                "taken_actions": act,
                "rnn": [s.transpose(0, 1) for s in rnn],
                "terminated": term,
                "truncated": trunc,
            },
            role="policy",
        )
    return (outputs["log_prob"] - stored).abs()


def test_sequence_length_one_replays_exactly():
    """The control: no sequence replay at all, so this must be exact."""
    delta = epoch_zero_delta(TrainConfig(env_steps=1, rollouts=8, sequence_length=1, **TOY))
    assert float(delta.max()) < 1e-5, f"max |delta log p| = {float(delta.max()):.3e}"


def test_sequence_replay_reproduces_collection():
    """What must hold before a recurrent run can be believed."""
    delta = epoch_zero_delta(TrainConfig(env_steps=1, rollouts=8, sequence_length=4, **TOY))
    assert float(delta.max()) < 1e-5, f"max |delta log p| = {float(delta.max()):.3e}"


def test_the_stored_hidden_state_is_the_one_the_action_was_taken_from():
    """The root cause, pinned directly rather than through its symptom.

    skrl 2.1.0 binds `_rnn_initial_states` and `_rnn_final_states` to the same
    dict, so after the first step `act()` overwrites the "initial" state with the
    post-step one before `record_transition` can store it. Measured:
    `stored[t] == h_in[t+1]` for every t >= 1. `PPO_RNN_Aligned` restores the
    invariant; this asserts it holds rather than trusting the subclass.
    """
    cfg = TrainConfig(env_steps=1, rollouts=4, sequence_length=1, **TOY)
    env, agent, _ = build(cfg)
    agent.init()
    agent.enable_training_mode(True)
    obs, _ = env.reset()
    states = env.state()

    for t in range(cfg.rollouts):
        used = agent._rnn_initial_states["policy"][0].clone()
        with torch.no_grad():
            actions = agent.act(
                obs[SWARM_UID], states[SWARM_UID], timestep=t, timesteps=cfg.rollouts
            )[0]
        next_obs, rewards, terminated, truncated, infos = env.step({SWARM_UID: actions})
        agent.record_transition(
            observations=obs[SWARM_UID],
            states=states[SWARM_UID],
            actions=actions,
            rewards=rewards[SWARM_UID],
            next_observations=env.final_observations()[SWARM_UID],
            next_states=env.final_states()[SWARM_UID],
            terminated=terminated[SWARM_UID],
            truncated=truncated[SWARM_UID],
            infos=infos,
            timestep=t,
            timesteps=cfg.rollouts,
        )
        stored = agent.memory.get_tensor_by_name("rnn_policy_0")[t].transpose(0, 1)
        assert torch.equal(stored, used), (
            f"step {t}: the memory holds a hidden state the action was not taken from"
        )
        obs, states = next_obs, env.state()
