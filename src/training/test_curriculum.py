"""The schedule must not be able to differ between fidelity conditions.

This is the training-level sibling of Block F's env-level guarantee. `Block F`
asserts the env's *draws* (`jammer_on`, `speed_scale`, `episode_len`,
`route_id`) are bit-identical across all five rungs at a fixed seed; these
assert the *schedule* that reweights those draws is too. Together they are what
stop RQ1's independent variable leaking into how much of the hard stage each
condition ever sees.
"""

from __future__ import annotations

import pytest
import torch

from ..env.core import LADDER, STAGES, BatchedSwarmEnv, EnvConfig
from .curriculum import CurriculumCallback, CurriculumSchedule

FAST = {"no_buildings": True, "compile_occlusion": False}


def test_weights_are_a_distribution_that_ends_on_the_design_stage():
    s = CurriculumSchedule()
    for progress in (0.0, 0.1, 0.2, 0.5, 0.9, 1.0):
        w = s.weights(progress)
        assert len(w) == len(STAGES)
        assert all(x >= 0.0 for x in w)
        assert sum(w) == pytest.approx(1.0)
    assert s.weights(0.0) == (1.0, 0.0, 0.0, 0.0)
    assert s.weights(1.0)[-1] == pytest.approx(1.0 - s.mix)


def test_earlier_stages_are_mixed_in_rather_than_switched_away_from():
    """Hard-switching makes the policy forget the opening it still has to
    execute every single episode (docs/ENVIRONMENT.md)."""
    s = CurriculumSchedule()
    late = s.weights(1.0)
    assert sum(late[:-1]) == pytest.approx(s.mix)
    assert all(x > 0.0 for x in late[:-1]), "every earlier stage must stay alive"


def test_the_schedule_is_monotone_in_difficulty():
    s = CurriculumSchedule()
    focus = [s.focus(i / 100) for i in range(101)]
    assert focus == sorted(focus)
    assert focus[0] == 0 and focus[-1] == len(STAGES) - 1


@pytest.mark.parametrize("fidelity", sorted(LADDER))
def test_the_schedule_is_identical_at_every_fidelity_rung(fidelity):
    """The one that protects RQ1.

    Not merely that the numbers agree: `weights()` is a pure function of
    progress and is never handed the env, the rung or any measure of how
    training is going, so there is no channel through which they *could*
    disagree. This pins that.
    """
    env = BatchedSwarmEnv(EnvConfig(num_envs=2, num_drones=3, seed=0, fidelity=fidelity, **FAST))
    cb = CurriculumCallback(env, total_timesteps=1000)
    emitted = [cb.update(t) for t in range(0, 1000, 7)]

    reference = CurriculumSchedule()
    assert emitted == [reference.weights(t / 1000) for t in range(0, 1000, 7)]


def test_the_callback_actually_moves_the_env_and_only_when_it_changes():
    env = BatchedSwarmEnv(EnvConfig(num_envs=8, num_drones=3, seed=0, **FAST))
    cb = CurriculumCallback(env, total_timesteps=100)

    cb.update(0)
    first = env.stage_cdf.clone()
    cb.update(1)
    assert torch.equal(env.stage_cdf, first), "no boundary crossed: no write"

    cb.update(99)
    assert not torch.equal(env.stage_cdf, first)
    # and what it wrote is the schedule's own answer, not an approximation of it
    want = torch.tensor(CurriculumSchedule().weights(0.99), device=env.device)
    assert torch.allclose(env.stage_cdf, (want / want.sum()).cumsum(0))


def test_a_schedule_that_does_not_match_the_stage_count_is_refused():
    with pytest.raises(ValueError):
        CurriculumSchedule(boundaries=(0.5,))
    with pytest.raises(ValueError):
        CurriculumSchedule(boundaries=(0.6, 0.3, 0.9))
    with pytest.raises(ValueError):
        CurriculumSchedule(mix=1.0)
