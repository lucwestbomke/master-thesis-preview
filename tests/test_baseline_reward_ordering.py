"""The reward's policy ordering, for policies that actually fly. Block E.

`docs/REWARD.md` rests on a ranking -- heuristic > formation > chase > lazy --
and `src/env/test_reward.py` asserts it against **hand-written `Snapshot`
stubs**. That is the right scope for a unit test of `reward.py`, but it means
the ordering the whole reward design depends on had never been checked against
behaviour the environment actually produces. Stubs cannot tell you that the
reward ranks *reachable* states correctly.

This closes that loop: real policies, real physics, real snapshots. It is a
cross-module test, so it lives here rather than beside `reward.py`
(`AGENTS.md`: "tests/ is cross-module only -- unit tests are CO-LOCATED").

Deliberately short episodes and few environments. The gaps being asserted are
tens of percentage points wide, not marginal, so this buys its runtime.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines import B0Policy, rollout
from src.env.core import BatchedSwarmEnv, EnvConfig

B, N, STEPS = 8, 5, 220


def env_for(seed: int) -> BatchedSwarmEnv:
    return BatchedSwarmEnv(
        EnvConfig(
            num_envs=B,
            num_drones=N,
            seed=seed,
            auto_reset=False,
            compile_occlusion=False,
            stage_weights=(0.0, 0.0, 0.0, 1.0),
        )
    )


def score(policy_name: str, seed: int = 21):
    env = env_for(seed)
    if policy_name == "lazy":
        # Never launches. The optimum the idle penalty exists to destroy, and
        # the one that fixed-length episodes do NOT kill on their own.
        return rollout(env, lambda _o: torch.zeros(B, N, 3), STEPS)
    if policy_name == "random":
        gen = torch.Generator().manual_seed(seed)
        return rollout(env, lambda _o: torch.empty(B, N, 3).uniform_(-1, 1, generator=gen), STEPS)
    pol = B0Policy(B, N, variant=policy_name, device=env.device)
    return rollout(env, lambda o: pol.act(o["flat"]), STEPS, on_reset=pol.reset)


def test_reward_ranks_real_policies_the_way_reward_md_claims():
    """B0 > geodesic-ish > random > lazy, in the reward's own units.

    If this ordering ever breaks, the reward is wrong -- and it is found in
    seconds rather than after a three-hour training run.
    """
    b0 = score("b0")
    geo = score("geodesic")
    rnd = score("random")
    lazy = score("lazy")

    r = {
        k: float(v.episode_return.median())
        for k, v in (("b0", b0), ("geodesic", geo), ("random", rnd), ("lazy", lazy))
    }
    assert r["b0"] > r["random"] > r["lazy"], r
    assert r["geodesic"] > r["random"], r
    # Both designed rungs must beat the untuned floor by a wide margin, or the
    # reward is not discriminating between competence and noise.
    assert r["b0"] - r["random"] > 50.0, r


def test_loitering_accrues_unbounded_negative_reward():
    """REWARD.md's `w_idle`: never acquiring must be strictly worse than trying.

    The stub version of this asserts it on synthetic snapshots; here the drones
    genuinely sit on the MCV for the whole episode.
    """
    lazy = score("lazy")
    assert float(lazy.episode_return.median()) < 0.0
    assert float(lazy.mission_capable.mean()) == 0.0


def test_the_headline_metric_and_the_reward_agree_on_the_ordering():
    """Mission-capable fraction IS the dominant reward term by construction
    (REWARD.md), so the two must not disagree about which policy is better.
    A divergence would mean the policy optimises something other than the
    number that gets reported."""
    for a, b in (("b0", "random"), ("geodesic", "random"), ("random", "lazy")):
        ma, mb = score(a), score(b)
        by_reward = float(ma.episode_return.median()) > float(mb.episode_return.median())
        by_metric = float(ma.mission_capable.mean()) >= float(mb.mission_capable.mean())
        assert by_reward and by_metric, (a, b)
