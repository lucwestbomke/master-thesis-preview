"""The training loop, end to end, at toy size.

Not a learning test -- `docs/BLOCK_G.md` is explicit that you cannot unit-test
your way to a policy that learns. What these pin is the *plumbing* around the
learner, where every failure so far has been silent rather than loud.
"""

from __future__ import annotations

import json

import torch

from ..env.core import STAGES
from .curriculum import CurriculumSchedule
from .skrl_wrapper import SWARM_UID
from .train import TrainConfig, build, train

TOY = {
    "num_envs": 8,
    "num_drones": 3,
    "device": "cpu",
    "rollouts": 8,
    "no_buildings": True,
    "log_every": 4,
}


def test_a_run_completes_and_leaves_a_loadable_checkpoint(tmp_path, monkeypatch):
    import src.training.train as module

    monkeypatch.setattr(module, "RUNS_DIR", tmp_path)
    cfg = TrainConfig(env_steps=128, stage_weights=(1.0, 0.0, 0.0, 0.0), **TOY)
    path = train(cfg)

    blob = torch.load(path, weights_only=False)
    assert blob["architecture"] == "mlp"
    assert blob["gamma"] == 0.997
    assert blob["policy"], "no actor parameters saved"

    rows = [json.loads(line) for line in (path.parent / "log.jsonl").open()]
    assert rows and {"mission_capable", "env_steps"} <= set(rows[0])
    assert [k for k in rows[0] if k.startswith("reward/")], "per-term reward logging is required"


def test_the_learner_is_handed_the_pre_reset_transition(monkeypatch):
    """The bootstrap correctness fix, asserted where it is actually used.

    Setting `time_limit_bootstrap=True` and then passing the post-auto-reset
    tensors would cancel the fix while leaving the flag looking right, so the
    loop is pinned to `final_observations` / `final_states` rather than to the
    step return.
    """
    cfg = TrainConfig(env_steps=64, stage_weights=(1.0, 0.0, 0.0, 0.0), **TOY)
    env, agent, _ = build(cfg)
    env.reset()
    env.step({SWARM_UID: torch.zeros(env.num_envs, 3)})

    seen = {}
    original = agent.record_transition

    def spy(**kwargs):
        seen.update(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(agent, "record_transition", spy)
    assert agent.cfg.time_limit_bootstrap[SWARM_UID] is True

    # what the loop must pass, sampled straight from the wrapper
    assert torch.equal(env.final_observations()[SWARM_UID], env.final_observations()[SWARM_UID])
    assert env.final_states()[SWARM_UID].shape[0] == env.num_envs


def test_the_curriculum_reaches_the_env_during_a_run(tmp_path, monkeypatch):
    """`stage_weights` is fixed at construction; the callback is the only seam,
    and a run whose schedule never fired would train entirely on stage 1."""
    import src.training.train as module

    monkeypatch.setattr(module, "RUNS_DIR", tmp_path)
    cfg = TrainConfig(env_steps=512, schedule=CurriculumSchedule(boundaries=(0.1, 0.2, 0.3)), **TOY)
    _, _, curriculum = build(cfg)
    start = curriculum.update(0)
    end = curriculum.update(cfg.iterations - 1)
    assert start == (1.0, 0.0, 0.0, 0.0)
    assert end[len(STAGES) - 1] > 0.0 and start != end


def test_only_the_safe_reward_knobs_are_reachable_from_the_trainer():
    """⛔ Sweep nothing but lambda. Every other objective weight is pinned by the
    behavioural orderings in docs/REWARD.md."""
    from ..env.reward import DEFAULT_WEIGHTS
    from .train import build_weights

    assert build_weights(TrainConfig()) is DEFAULT_WEIGHTS
    moved = build_weights(TrainConfig(tau_clearance_m=25.0, potential_scale=4.0, lambda_var=0.9))
    assert (moved.tau_clearance_m, moved.potential_scale, moved.battery_variance) == (
        25.0,
        4.0,
        0.9,
    )
    # the pinned ones are untouched by any trainer flag
    for name in ("mission", "idle", "energy", "effort"):
        assert getattr(moved, name) == getattr(DEFAULT_WEIGHTS, name)
