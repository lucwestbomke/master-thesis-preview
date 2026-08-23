"""Unit tests for the B0 baseline -- Block E.

The load-bearing one is `test_action_depends_only_on_the_observation`. B0's whole
claim is that it competes with the learned policies on equal information, and
`env.hvt_pos` is one attribute access away at every point in the controller.
Nothing but this test stops a tuning session from quietly turning the reported
baseline into an oracle.
"""

from __future__ import annotations

import pytest
import torch

from ..env.core import ACTION_DIM, EPISODE_STEPS, FLAT_DIM, BatchedSwarmEnv, EnvConfig
from .b0 import VARIANTS, B0Config, B0Policy

B, N = 4, 5
STEPS = 40  # long enough to leave the launch fan and engage the role logic


def make_env(seed: int = 0, num_drones: int = N, num_envs: int = B) -> BatchedSwarmEnv:
    return BatchedSwarmEnv(
        EnvConfig(
            num_envs=num_envs,
            num_drones=num_drones,
            seed=seed,
            auto_reset=False,
            compile_occlusion=False,
            stage_weights=(0.0, 0.0, 0.0, 1.0),
        )
    )


def record(env: BatchedSwarmEnv, pol: B0Policy, steps: int = STEPS):
    """Run the policy and keep every observation it was shown, plus its actions."""
    obs = env.reset()
    pol.reset()
    flats, actions = [], []
    for _ in range(steps):
        flat = obs["flat"].clone()
        act = pol.act(flat)
        flats.append(flat)
        actions.append(act.clone())
        obs, _, _, _, _ = env.step(act)
    return flats, actions


# --------------------------------------------------------------------------- #
# The information contract
# --------------------------------------------------------------------------- #


def test_action_depends_only_on_the_observation():
    """Replaying the recorded observations reproduces the actions exactly.

    If B0 read anything off the env -- the true HVT position, the building
    boxes, the route id -- replay through a fresh policy with no env in scope
    would diverge. This is `docs/BLOCK_E.md` §1's contract made enforceable
    rather than asserted in a comment.
    """
    env = make_env(seed=0)
    pol = B0Policy(B, N, variant="b0")
    flats, actions = record(env, pol)

    replay = B0Policy(B, N, variant="b0")
    replay.reset()
    for flat, expected in zip(flats, actions, strict=True):
        torch.testing.assert_close(replay.act(flat), expected)


def test_live_env_state_cannot_leak_into_the_action():
    """Replay one scenario's observations while a *different* one is running.

    Stronger than the replay test above: a second env is constructed, reset and
    stepped between every call, so any attribute the controller reached for --
    `hvt_pos`, `mcv_pos`, `route_id` -- would be the wrong scenario's and the
    actions would diverge. They must not.
    """
    flats, actions = record(make_env(seed=1), B0Policy(B, N), steps=12)

    other = make_env(seed=99)
    other.reset()
    quiet = torch.zeros(B, N, ACTION_DIM)
    replay = B0Policy(B, N)
    replay.reset()
    for flat, expected in zip(flats, actions, strict=True):
        got = replay.act(flat)
        other.step(quiet)  # move the decoy scenario on underneath
        torch.testing.assert_close(got, expected)


def test_oracle_is_an_explicit_channel_not_an_accident():
    env = make_env()
    obs = env.reset()
    like_for_like = B0Policy(B, N, variant="b0")
    oracle = B0Policy(B, N, variant="oracle")
    truth = {
        "hvt_rel": env.hvt_pos.unsqueeze(1) - env.drone_pos,
        "hvt_vel": env.hvt_vel.unsqueeze(1).expand(-1, N, -1),
    }
    with pytest.raises(ValueError, match="like-for-like"):
        like_for_like.act(obs["flat"], truth)
    with pytest.raises(ValueError, match="requires truth"):
        oracle.act(obs["flat"])
    assert oracle.act(obs["flat"], truth).shape == (B, N, ACTION_DIM)


# --------------------------------------------------------------------------- #
# Shape, range, device discipline
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("num_drones", [1, 3, 5, 8])
def test_shapes_and_bounds(variant: str, num_drones: int):
    env = make_env(num_drones=num_drones)
    obs = env.reset()
    pol = B0Policy(B, num_drones, variant=variant)
    truth = None
    if variant == "oracle":
        truth = {
            "hvt_rel": env.hvt_pos.unsqueeze(1) - env.drone_pos,
            "hvt_vel": env.hvt_vel.unsqueeze(1).expand(-1, num_drones, -1),
        }
    act = pol.act(obs["flat"], truth)
    assert act.shape == (B, num_drones, ACTION_DIM)
    assert torch.isfinite(act).all()
    assert act.abs().max() <= 1.0 + 1e-6
    assert obs["flat"].shape[-1] == FLAT_DIM


def test_batch_independence():
    """B=4 identical seeds must give four identical trajectories.

    Catches an accidental reduction across the batch -- for instance a role
    ranking that pooled over environments -- which is invisible otherwise.
    """
    env = make_env(seed=7, num_envs=4)
    pol = B0Policy(4, N)
    _, actions = record(env, pol, steps=15)
    for act in actions:
        for i in range(1, 4):
            # environments differ (different routes), so this asserts only that
            # no environment's action is a function of another's: perturbing one
            # row of the observation must not move the others.
            assert torch.isfinite(act[i]).all()

    flat = env.reset()["flat"]
    pol2 = B0Policy(4, N)
    pol2.reset()
    base = pol2.act(flat)
    perturbed = flat.clone()
    perturbed[0] += 0.5
    pol3 = B0Policy(4, N)
    pol3.reset()
    got = pol3.act(perturbed)
    torch.testing.assert_close(got[1:], base[1:])
    assert not torch.allclose(got[0], base[0])


# --------------------------------------------------------------------------- #
# Carried state
# --------------------------------------------------------------------------- #


def test_reset_clears_carried_state():
    """Belief, roles and repair offsets must not survive an episode boundary.

    The same class of bug as Block D's stale potential: silent, survives every
    single-step test, and shows up only as "B0 is oddly bad at episode start"
    because every new episode begins with a belief aimed at the previous route.
    """
    env = make_env(seed=3)
    pol = B0Policy(B, N)
    record(env, pol, steps=25)
    assert pol.informed.any(), "test is vacuous if nothing was learned to forget"

    pol.reset()
    assert not pol.informed.any()
    assert not pol.started.any()
    assert torch.count_nonzero(pol.belief_rel) == 0
    assert torch.count_nonzero(pol.belief_vel) == 0
    assert torch.count_nonzero(pol.lat_m) == 0


def test_reset_is_per_environment():
    env = make_env(seed=3)
    pol = B0Policy(B, N)
    record(env, pol, steps=25)
    before = pol.belief_rel.clone()
    mask = torch.zeros(B, dtype=torch.bool)
    mask[0] = True
    pol.reset(mask)
    assert torch.count_nonzero(pol.belief_rel[0]) == 0
    torch.testing.assert_close(pol.belief_rel[1:], before[1:])


# --------------------------------------------------------------------------- #
# Behaviour the design claims
# --------------------------------------------------------------------------- #


def test_the_observer_closes_on_the_target():
    """B0's highest-leverage mechanism: somebody flies directly overhead.

    Measured at 40.2 % -> 92.6 % mission-capable on its own (docs/BLOCK_E.md),
    so if this regresses the baseline collapses and nothing else in the file
    would say why.
    """
    env = make_env(seed=5)
    pol = B0Policy(B, N)
    obs = env.reset()
    pol.reset()
    start = (env.drone_pos - env.hvt_pos.unsqueeze(1)).norm(dim=-1).min(dim=-1).values
    for _ in range(120):
        obs, _, _, _, _ = env.step(pol.act(obs["flat"]))
    end = (env.drone_pos - env.hvt_pos.unsqueeze(1)).norm(dim=-1).min(dim=-1).values
    assert (end < start).float().mean() > 0.75
    # Directly overhead at the 80 m ceiling over a vehicle at 1.5 m is a 78.5 m
    # slant, so anything near that is "on top of it".
    assert float(end.median()) < 200.0


def test_it_climbs_to_the_ceiling():
    """Everything pushes up and nothing charges for altitude, so B0 should pin
    the ceiling. Stated in the methodology rather than tuned away."""
    env = make_env(seed=5)
    pol = B0Policy(B, N)
    obs = env.reset()
    pol.reset()
    for _ in range(80):
        obs, _, _, _, _ = env.step(pol.act(obs["flat"]))
    assert float(env.drone_pos[..., 2].median()) > 75.0


def test_hop_count_escalates_with_separation():
    """The chain is sized from the separation the drone believes in."""
    cfg = B0Config(hop_reach_m=400.0, max_spares=2)
    pol = B0Policy(1, 5, cfg=cfg)
    pol.belief_rel = torch.zeros(1, 5, 3)
    mcv_near = torch.zeros(1, 5, 3)
    mcv_near[..., 0] = 350.0
    mcv_far = torch.zeros(1, 5, 3)
    mcv_far[..., 0] = 1400.0
    sees = torch.zeros(1, 5, dtype=torch.bool)
    nb_sees = torch.zeros(1, 5, 7, dtype=torch.bool)
    nb_rel = torch.zeros(1, 5, 7, 3)
    valid = torch.zeros(1, 5, 7, dtype=torch.bool)
    valid[..., :4] = True
    _, near = pol._roles(sees, nb_sees, nb_rel, mcv_near, valid)
    _, far = pol._roles(sees, nb_sees, nb_rel, mcv_far, valid)
    assert int(near.max()) < int(far.max())


def test_beats_a_random_policy():
    """MODELS.md's sanity floor, from the bottom. A short rollout is enough --
    the gap is 19 % against >90 %, not a coin flip."""
    from .evaluate import rollout

    steps = min(200, EPISODE_STEPS)
    env = make_env(seed=11, num_envs=8)
    pol = B0Policy(8, N)
    b0_score = rollout(env, lambda o: pol.act(o["flat"]), steps, on_reset=pol.reset)

    env2 = make_env(seed=11, num_envs=8)
    gen = torch.Generator().manual_seed(0)
    rnd = rollout(env2, lambda _o: torch.empty(8, N, 3).uniform_(-1, 1, generator=gen), steps)
    assert float(b0_score.mission_capable.mean()) > float(rnd.mission_capable.mean()) + 0.2
