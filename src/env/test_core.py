"""Batched env core tests.

Two classes of test, and the second is the one that matters:

- *wiring*: does the env call Block A the way PHYSICS.md and REWARD.md say?
- *semantics*: episode structure, auto-reset, determinism, batch independence,
  and actor locality. These pin properties whose violation is silent -- an
  accidental reduction across the batch dimension, a potential carried across an
  episode boundary, or global state leaking into the actor, none of which show up
  as a crash and all of which invalidate results.
"""

from __future__ import annotations

import pytest
import torch

from . import channel, core
from .core import (
    ALT_MAX_M,
    ALT_MIN_M,
    BANDWIDTH_HZ,
    BOX_HALF_M,
    EGO_DIM,
    FLAT_DIM,
    N_MAX,
    NEIGHBOUR_DIM,
    STAGES,
    BatchedSwarmEnv,
    EnvConfig,
)
from .reward import CAPACITY_THRESHOLD_MBPS, potential

STAGE4 = (0.0, 0.0, 0.0, 1.0)


def make(num_envs=8, num_drones=5, seed=0, **kw) -> BatchedSwarmEnv:
    kw.setdefault("stage_weights", STAGE4)
    kw.setdefault("compile_occlusion", False)  # tests should not pay compile warmup
    env = BatchedSwarmEnv(EnvConfig(num_envs=num_envs, num_drones=num_drones, seed=seed, **kw))
    env.reset()
    return env


def zeros_like_actions(env: BatchedSwarmEnv) -> torch.Tensor:
    return torch.zeros(env.cfg.num_envs, env.cfg.num_drones, 3, device=env.device)


# --------------------------------------------------------------------------- #
# Physics wiring
# --------------------------------------------------------------------------- #


def test_hvt_follows_the_baked_route_exactly():
    """No graph search in the hot loop -- the route is an index, and it must be
    the *same* index the artefact holds."""
    env = make(num_envs=4)
    a = zeros_like_actions(env)
    for step in range(1, 25):
        env.step(a)
        expect = env.route_xy[env.route_id, step]
        assert torch.allclose(env.hvt_pos[:, :2], expect, atol=1e-4)


def test_stationary_stage_freezes_the_hvt():
    """Stage 1's speed_scale=0 must actually stop the target, since decoupling
    'learn to relay' from 'learn to chase' is the whole point of that rung."""
    env = make(num_envs=4, stage_weights=(1.0, 0.0, 0.0, 0.0))
    start = env.hvt_pos.clone()
    for _ in range(30):
        env.step(zeros_like_actions(env))
    assert torch.allclose(env.hvt_pos, start, atol=1e-5)


def test_hovering_burns_about_seven_percent_of_the_pack():
    """PHYSICS.md predicts ~7 % over a 240 s episode, and RQ3's reframing from
    energy rotation to geometric handoff depends on batteries not binding."""
    env = make(num_envs=4, no_buildings=True)
    env.battery = torch.ones_like(env.battery)
    env.battery_scale = torch.ones_like(env.battery_scale)
    # One step short of truncation: crossing it re-randomises the charge, and
    # measuring after that reads the fresh battery rather than the used one.
    for _ in range(core.EPISODE_STEPS - 1):
        env.step(zeros_like_actions(env))
    used = 1.0 - env.battery
    assert torch.allclose(used, used[:1, :1].expand_as(used), atol=1e-5), "hovering is uniform"
    assert 0.03 < used.mean().item() < 0.25, used.mean().item()


def test_link_class_follows_the_index_rule():
    """Drone-drone is A2A; anything touching the MCV is A2G. Getting this wrong
    applies a ground street-canyon model to nodes above rooftop."""
    env = make(num_drones=4)
    n = env.cfg.num_drones
    assert env.is_a2a[:n, :n].all()
    assert not env.is_a2a[n, :].any()
    assert not env.is_a2a[:, n].any()


def test_scheduled_mac_matches_a_one_hot_tx_mask():
    """The shortcut in `_capacity`: with only one transmitter live per slot,
    intra-swarm interference is identically zero and SINR reduces to S/(J+N0).

    This asserts the shortcut equals `channel.sinr_db` driven with a one-hot
    mask, which is the form PHYSICS.md actually specifies.
    """
    env = make(num_envs=4, num_drones=4)
    r = env.cfg.n_radio
    pos_k = torch.cat([env.drone_pos, env.mcv_pos.unsqueeze(1), env.hvt_pos.unsqueeze(1)], dim=1)
    _true, clearance = env._clearance(pos_k)
    got, jam_mw = env._capacity(pos_k, clearance)

    radio = pos_k[:, :r]
    d3d = channel.pairwise_distance_m(radio)
    occluded = clearance[:, :r, :r] < 0.0
    z = radio[..., 2]
    h_uav = torch.maximum(z.unsqueeze(-1), z.unsqueeze(-2))
    pathloss = torch.where(
        env.is_a2a,
        channel.pathloss_a2a_db(d3d, occluded),
        channel.pathloss_a2g_umi_av_db(d3d, h_uav, ~occluded),
    )
    prx = channel.received_power_dbm(env.ptx, pathloss)
    jam_dbm = channel.mw_to_dbm(jam_mw)

    for i in range(r):
        mask = torch.zeros(env.cfg.num_envs, r, dtype=torch.bool, device=env.device)
        mask[:, i] = True
        sinr = channel.sinr_db(prx, jam_dbm, env.n0_dbm, mask)
        expect = channel.capacity_mbps(sinr, BANDWIDTH_HZ)[:, i, :] * env.no_self[i]
        assert torch.allclose(got[:, i, :], expect, atol=1e-3), f"transmitter {i}"


def test_positions_stay_in_the_box_and_the_altitude_band():
    """The band is a model-validity envelope: below 40 m the A2G model is out of
    spec and occlusion's endpoint convention starts letting drones see through
    their own building (docs/BLOCK_D.md)."""
    env = make(num_envs=8, no_buildings=True)
    torch.manual_seed(0)
    for _ in range(100):
        env.step(torch.empty_like(zeros_like_actions(env)).uniform_(-1.0, 1.0))
        assert env.drone_pos[..., :2].abs().max() <= BOX_HALF_M + 1e-4
        assert env.drone_pos[..., 2].min() >= ALT_MIN_M - 1e-4
        assert env.drone_pos[..., 2].max() <= ALT_MAX_M + 1e-4


def test_hitting_a_limit_zeroes_that_velocity_component():
    """Otherwise the drone presses into the wall and the energy term charges for
    motion that never happened."""
    env = make(num_envs=2)
    a = zeros_like_actions(env)
    a[..., 2] = 1.0  # climb hard into the ceiling
    for _ in range(40):
        env.step(a)
    assert torch.allclose(env.drone_pos[..., 2], torch.full_like(env.drone_pos[..., 2], ALT_MAX_M))
    assert torch.allclose(env.drone_vel[..., 2], torch.zeros_like(env.drone_vel[..., 2]))


# --------------------------------------------------------------------------- #
# Episode semantics
# --------------------------------------------------------------------------- #


def test_episode_truncates_at_the_stage_length():
    for idx, stage in enumerate(STAGES):
        weights = [0.0] * len(STAGES)
        weights[idx] = 1.0
        env = make(num_envs=4, stage_weights=tuple(weights), no_buildings=True)
        for step in range(1, stage.episode_steps + 1):
            _, _, term, trunc, _ = env.step(zeros_like_actions(env))
            assert not term.any(), "hovering must not terminate"
            expect = step == stage.episode_steps
            assert bool(trunc.all()) == expect, f"stage {idx + 1} step {step}"


def test_mission_failure_never_terminates():
    """Terminating on it teaches the policy to never acquire, and kills a random
    initial policy before it reaches the tracking phase (docs/DECISIONS.md)."""
    env = make(num_envs=8)
    for _ in range(60):
        _, _, term, _, extras = env.step(zeros_like_actions(env))
        assert not term.any()
    # parked on the MCV the whole time: the mission is failing every step
    assert not extras["mission_capable"].any()


def test_battery_death_terminates():
    """Physical, and unhackable -- hovering at the MCV burns power too."""
    env = make(num_envs=4)
    env.battery = torch.full_like(env.battery, 1e-6)
    _, _, term, trunc, _ = env.step(zeros_like_actions(env))
    assert term.all()
    assert not trunc.any(), "termination and truncation must be exclusive"


def test_terminal_flag_reaches_the_shaping_term(monkeypatch):
    """`Phi(terminal) = 0` is required for PBRS invariance; if the flag never
    arrives, `gamma^T Phi(s_T)` survives the telescoping as a policy-dependent
    bias (docs/REWARD.md)."""
    seen: dict[str, torch.Tensor] = {}
    real = core.reward

    def spy(snap, next_snap, w=None, gamma=0.999, next_is_terminal=None, craft=None):
        seen["flag"] = next_is_terminal
        return real(snap, next_snap, w, gamma, next_is_terminal, craft)

    monkeypatch.setattr(core, "reward", spy)
    env = make(num_envs=4)
    env.battery = torch.full_like(env.battery, 1e-6)
    _, _, term, _, _ = env.step(zeros_like_actions(env))
    assert seen["flag"] is not None
    assert torch.equal(seen["flag"], term)


def test_auto_reset_restarts_the_episode_cleanly():
    env = make(num_envs=4, stage_weights=(1.0, 0.0, 0.0, 0.0), no_buildings=True)
    for _ in range(STAGES[0].episode_steps):
        env.step(zeros_like_actions(env))
    assert torch.all(env.t == 0)
    assert torch.all(env.steps_since_link == 0)
    assert torch.allclose(env.drone_vel, torch.zeros_like(env.drone_vel))
    assert torch.allclose(env.drone_pos[..., 2], torch.full_like(env.drone_pos[..., 2], ALT_MIN_M))
    # drones are back on the MCV ring
    radius = (env.drone_pos[..., :2] - env.mcv_pos[:, None, :2]).norm(dim=-1)
    assert torch.allclose(radius, torch.full_like(radius, core.SPAWN_RING_M), atol=1e-3)


def test_potential_is_recomputed_after_a_reset_not_carried_over():
    """The silent one. `reward.shaping` reads Phi(s) from the stored snapshot; if
    that snapshot survives an episode boundary it injects a large spurious
    shaping term on the first step of every new episode, and it attacks exactly
    the invariance PBRS was chosen to guarantee."""
    env = make(num_envs=4, stage_weights=(1.0, 0.0, 0.0, 0.0), no_buildings=True)
    for _ in range(STAGES[0].episode_steps):
        env.step(zeros_like_actions(env))

    # The stored snapshot must describe the FRESH state, not the finished one.
    fresh_dist = (env.drone_pos - env.hvt_pos.unsqueeze(1)).norm(dim=-1).min(dim=-1).values
    assert torch.allclose(env.snap.nearest_dist_m, fresh_dist, atol=1e-3)
    assert torch.allclose(env.snap.battery, env.battery)
    assert torch.allclose(env.snap.speed_ms, torch.zeros_like(env.snap.speed_ms))

    # The invariant proper: the cached snapshot equals a fresh evaluation of the
    # current state, so the Phi(s) the next step subtracts is the right one.
    again, _ = env._evaluate()
    assert torch.allclose(env.snap.e2e_capacity_mbps, again.e2e_capacity_mbps, atol=1e-5)
    assert torch.allclose(env.snap.best_clearance_m, again.best_clearance_m, atol=1e-3)
    assert torch.equal(env.snap.observed, again.observed)
    assert torch.allclose(
        potential(env.snap, env.weights), potential(again, env.weights), atol=1e-5
    )


def test_same_seed_gives_identical_trajectories():
    a = make(num_envs=8, seed=7)
    b = make(num_envs=8, seed=7)
    torch.manual_seed(0)
    acts = [torch.empty(8, 5, 3).uniform_(-1, 1) for _ in range(30)]
    for act in acts:
        oa, ra, ta, ua, _ = a.step(act)
        ob, rb, tb, ub, _ = b.step(act)
        assert torch.equal(ra, rb)
        assert torch.equal(oa["flat"], ob["flat"])
        assert torch.equal(ta, tb) and torch.equal(ua, ub)


def test_environments_do_not_leak_into_each_other():
    """A reduction that accidentally spans the batch dimension is invisible in
    every other test: shapes stay right and values stay finite."""
    env = make(num_envs=4, seed=3)
    # force all four environments into environment 0's state
    for attr in ("drone_pos", "drone_vel", "battery", "mcv_pos", "hvt_pos", "hvt_vel", "cue"):
        t = getattr(env, attr)
        setattr(env, attr, t[:1].expand_as(t).clone())
    for attr in ("route_id", "t", "steps_since_link", "episode_len", "speed_scale", "jammer_on"):
        t = getattr(env, attr)
        setattr(env, attr, t[:1].expand_as(t).clone())
    env.snap, _ = env._evaluate()

    torch.manual_seed(1)
    act = torch.empty(1, 5, 3).uniform_(-1, 1).expand(4, 5, 3).contiguous()
    for _ in range(20):
        obs, rew, _, _, extras = env.step(act)
        for name, t in (("reward", rew), ("flat", obs["flat"]), ("state", obs["state"])):
            assert torch.allclose(t, t[:1].expand_as(t), atol=1e-5), name
        assert torch.allclose(
            extras["e2e_capacity_mbps"],
            extras["e2e_capacity_mbps"][:1].expand_as(extras["e2e_capacity_mbps"]),
        )


def test_batched_result_matches_a_single_environment():
    """Complements the leak test: pins the absolute values, not just agreement
    among copies."""
    big = make(num_envs=4, seed=11)
    small = make(num_envs=1, seed=11)
    for attr in ("drone_pos", "drone_vel", "battery", "mcv_pos", "hvt_pos", "hvt_vel", "cue"):
        t = getattr(big, attr)
        setattr(big, attr, t[:1].expand_as(t).clone())
        setattr(small, attr, t[:1].clone())
    for attr in ("route_id", "t", "steps_since_link", "episode_len", "speed_scale", "jammer_on"):
        t = getattr(big, attr)
        setattr(big, attr, t[:1].expand_as(t).clone())
        setattr(small, attr, t[:1].clone())
    big.snap, _ = big._evaluate()
    small.snap, _ = small._evaluate()

    torch.manual_seed(2)
    act = torch.empty(1, 5, 3).uniform_(-1, 1)
    for _ in range(15):
        ob, rb, *_ = big.step(act.expand(4, 5, 3).contiguous())
        os_, rs, *_ = small.step(act)
        assert torch.allclose(rb[:1], rs, atol=1e-5)
        assert torch.allclose(ob["flat"][:1], os_["flat"], atol=1e-5)


# --------------------------------------------------------------------------- #
# Observations
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n", [3, 5, 8])
def test_observation_shapes_hold_across_swarm_sizes(n):
    """RQ2 evaluates zero-shot at N in {3,5,8}; `flat` must not change width."""
    env = make(num_envs=4, num_drones=n)
    obs, *_ = env.step(zeros_like_actions(env))
    assert obs["ego"].shape == (4, n, EGO_DIM)
    assert obs["neighbour"].shape == (4, n, n - 1, NEIGHBOUR_DIM)
    assert obs["edge"].shape == (4, n, n - 1, 2)
    assert obs["flat"].shape == (4, n, FLAT_DIM)


def test_flat_packing_marks_padded_neighbours_invalid():
    env = make(num_envs=2, num_drones=5)
    obs, *_ = env.step(zeros_like_actions(env))
    valid = obs["flat"][..., -(N_MAX - 1) :]
    assert torch.equal(valid[..., :4], torch.ones_like(valid[..., :4]))
    assert torch.equal(valid[..., 4:], torch.zeros_like(valid[..., 4:]))
    # the padded slots themselves must be zero, not stale memory
    nb_block = obs["flat"][..., EGO_DIM : EGO_DIM + (N_MAX - 1) * NEIGHBOUR_DIM]
    nb = nb_block.unflatten(-1, (N_MAX - 1, NEIGHBOUR_DIM))
    assert torch.equal(nb[..., 4:, :], torch.zeros_like(nb[..., 4:, :]))


def test_flat_packing_contains_the_structured_blocks():
    env = make(num_envs=2, num_drones=5)
    obs, *_ = env.step(zeros_like_actions(env))
    flat, ego, nb, edge = obs["flat"], obs["ego"], obs["neighbour"], obs["edge"]
    k = N_MAX - 1
    assert torch.equal(flat[..., :EGO_DIM], ego)
    off = EGO_DIM
    assert torch.equal(flat[..., off : off + 4 * NEIGHBOUR_DIM], nb.flatten(2))
    off += k * NEIGHBOUR_DIM
    assert torch.equal(flat[..., off : off + 4 * 2], edge.flatten(2))


def test_actor_sees_no_absolute_position():
    """CTDE depends on it, and DECISIONS.md's MCV-quadrant argument depends on
    it: nothing in the ego vector may reveal *where on the map* the drone is.

    Occlusion is switched off so the map itself is the only remaining source of
    absolute reference; the whole scenario is then translated and the actor's
    observation must not move.
    """
    env = make(num_envs=4, no_buildings=True)
    obs_a, *_ = env.step(zeros_like_actions(env))
    before = {k: v.clone() for k, v in obs_a.items()}

    shift = torch.tensor([137.0, -91.0, 0.0], device=env.device)
    env.drone_pos = env.drone_pos + shift
    env.mcv_pos = env.mcv_pos + shift
    env.hvt_pos = env.hvt_pos + shift
    env.cue = env.cue + shift
    _, aux = env._evaluate()
    after = env._observe(aux)

    assert torch.allclose(before["ego"], after["ego"], atol=1e-4)
    assert torch.allclose(before["neighbour"], after["neighbour"], atol=1e-4)


def test_cue_is_observable_and_never_refreshed():
    """It occupies its own 3 dims and persists all episode -- it decays in range,
    not in direction, which is why carrying it is safe (docs/BLOCK_D.md)."""
    env = make(num_envs=4)
    cue0 = env.cue.clone()
    for _ in range(60):
        obs, *_ = env.step(zeros_like_actions(env))
    assert torch.equal(env.cue, cue0), "the cue must never be refreshed"
    expect = (env.cue.unsqueeze(1) - env.drone_pos) / core.POS_SCALE_M
    assert torch.allclose(obs["ego"][..., 4:7], expect, atol=1e-5)


def test_observations_are_finite_and_roughly_unit_scale():
    """Unbounded inputs are a silent training failure; occlusion returns 1e4 for
    'nothing in the way', which must be clamped before it reaches a network."""
    env = make(num_envs=8)
    torch.manual_seed(4)
    for _ in range(40):
        obs, *_ = env.step(torch.empty_like(zeros_like_actions(env)).uniform_(-1, 1))
        for name, t in obs.items():
            assert torch.isfinite(t).all(), name
            assert t.abs().max() <= 10.0, (name, t.abs().max().item())


def test_extras_report_the_headline_metric_and_its_attribution():
    env = make(num_envs=8)
    _, _, _, _, extras = env.step(zeros_like_actions(env))
    for key in ("mission_capable", "e2e_capacity_mbps", "hop_count", "chain_occluded", "sees_any"):
        assert key in extras, key
    assert extras["mission_capable"].dtype == torch.bool
    # mission-capable implies both an observation and a live link, by definition
    capable = extras["mission_capable"]
    assert torch.all(~capable | extras["sees_any"])
    assert torch.all(~capable | (extras["e2e_capacity_mbps"] >= CAPACITY_THRESHOLD_MBPS))


def test_state_dim_matches_the_critic_state_it_describes():
    """`EnvConfig.state_dim` is what the skrl wrapper declares to MAPPO. If it
    drifts from `_critic_state`, the mismatch surfaces as a shape error deep in
    the learner rather than here."""
    for n in (3, 5, 8):
        env = make(num_envs=2, num_drones=n, no_buildings=True)
        obs, *_ = env.step(zeros_like_actions(env))
        assert obs["state"].shape[-1] == env.cfg.state_dim, n


# --------------------------------------------------------------------------- #
# The flat packing, and its inverse
# --------------------------------------------------------------------------- #


def test_unpack_flat_inverts_pack():
    """`unpack_flat` must recover exactly what `_pack` put in.

    Everything downstream of the observation contract unpacks `flat`: the B0
    baseline and all three of Block G's architectures (`docs/MODELS.md`). A
    second, hand-rolled unpacking is how the max-N padding stops being identical
    across rungs, so the inverse lives next to the packing and is pinned here.
    """
    env = BatchedSwarmEnv(EnvConfig(num_envs=3, num_drones=5, seed=0, compile_occlusion=False))
    obs = env.reset()
    got = core.unpack_flat(obs["flat"])

    n_real = env.cfg.num_drones - 1
    torch.testing.assert_close(got["ego"], obs["ego"])
    torch.testing.assert_close(got["neighbour"][:, :, :n_real], obs["neighbour"])
    torch.testing.assert_close(got["edge"][:, :, :n_real], obs["edge"])
    # Padding is zero and flagged invalid, so an off-N model cannot read noise.
    assert torch.count_nonzero(got["neighbour"][:, :, n_real:]) == 0
    assert torch.count_nonzero(got["edge"][:, :, n_real:]) == 0
    torch.testing.assert_close(
        got["valid"][:, :, :n_real], torch.ones_like(got["valid"][:, :, :n_real])
    )
    assert torch.count_nonzero(got["valid"][:, :, n_real:]) == 0


def test_neighbour_index_table_matches_the_packing_order():
    """Slot k of drone i holds drone k if k < i else k+1.

    A policy reconstructs this from N alone -- it is part of the contract, not
    env state -- so if the env ever reorders neighbours this must fail loudly.
    """
    n = 5
    table = core.neighbour_index_table(n)
    assert table.shape == (n, n - 1)
    for i in range(n):
        expected = [k for k in range(n) if k != i]
        assert table[i].tolist() == expected
    torch.testing.assert_close(
        table,
        BatchedSwarmEnv(
            EnvConfig(num_envs=1, num_drones=n, seed=0, compile_occlusion=False)
        ).nb_idx,
    )


# --------------------------------------------------------------------------- #
# The Block G seams: curriculum reweighting and the training-only extras
# --------------------------------------------------------------------------- #


def test_training_extras_are_off_by_default():
    """They widen the `extras` contract, which `test_golden.py` pins. Off by
    default is what keeps the frozen trace valid without a re-capture."""
    env = make(num_envs=2, no_buildings=True)
    assert env.cfg.training_extras is False
    _, _, _, _, extras = env.step(zeros_like_actions(env))
    assert not [k for k in extras if k == "final_state" or k.startswith("reward/")]


def test_final_state_is_the_pre_reset_state_not_the_fresh_episode():
    """The one that matters for correctness.

    skrl bootstraps a truncation as `gamma * V(next_observations, next_states)`.
    With auto_reset the tensors `step()` returns are a FRESH episode's opening,
    so bootstrapping off them values an unrelated state -- silently, and at
    gamma = 0.997 on returns of order 300. `final_state` is what the learner
    must be handed instead, and this asserts the two really do differ exactly
    where it matters and agree everywhere else.
    """
    stage1 = (1.0, 0.0, 0.0, 0.0)
    env = make(num_envs=4, stage_weights=stage1, no_buildings=True, training_extras=True)
    steps = STAGES[0].episode_steps
    for step in range(1, steps + 1):
        obs, _, term, trunc, extras = env.step(zeros_like_actions(env))
        assert "final_state" in extras
        assert extras["final_state"].shape == obs["state"].shape
        done = term | trunc
        if not done.any():
            # No reset happened: the second physics pass re-evaluates an
            # unchanged state, so the two must agree element for element.
            assert torch.equal(extras["final_state"], obs["state"]), f"step {step}"
        else:
            assert done.all() and step == steps
            assert not torch.equal(extras["final_state"], obs["state"])
            # and `final_observation` is the same story for the actor's view
            assert not torch.equal(extras["final_observation"], obs["flat"])


def test_reward_terms_sum_to_the_reward_they_decompose():
    """Instrumentation that disagreed with the objective would be worse than
    none: it would attribute a flat return curve to the wrong term."""
    env = make(num_envs=4, no_buildings=True, training_extras=True)
    for _ in range(5):
        actions = torch.empty(env.cfg.num_envs, env.cfg.num_drones, 3, device=env.device).uniform_(
            -1, 1
        )
        _, rew, _, _, extras = env.step(actions)
        terms = [v for k, v in extras.items() if k.startswith("reward/")]
        # 7 since Block G added the per-drone `relay` term. The count is asserted
        # so a term cannot be added to the reward and forgotten in the
        # decomposition -- which is how a flat return curve gets attributed to
        # the wrong place. Update it deliberately when the reward gains a term.
        assert len(terms) == 7
        assert torch.allclose(torch.stack(terms).sum(0), rew, atol=1e-5)


def test_set_stage_weights_moves_which_stage_fresh_episodes_draw():
    """The curriculum's only seam into the env. It must reach *new* episodes and
    nothing else -- the physics, the route bank and the reward are untouched."""
    env = make(num_envs=64, stage_weights=(1.0, 0.0, 0.0, 0.0), no_buildings=True)
    assert float(env.episode_len.max()) == float(STAGES[0].episode_steps)

    env.set_stage_weights((0.0, 0.0, 0.0, 1.0))
    for _ in range(STAGES[0].episode_steps):
        env.step(zeros_like_actions(env))
    assert float(env.episode_len.min()) == float(STAGES[3].episode_steps)

    with pytest.raises(ValueError):
        env.set_stage_weights((1.0, 0.0))
    with pytest.raises(ValueError):
        env.set_stage_weights((0.0, 0.0, 0.0, 0.0))
