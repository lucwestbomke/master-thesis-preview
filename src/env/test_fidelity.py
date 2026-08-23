"""The fidelity ladder F0-F4 -- RQ1's independent variable.

Block F is small in code and large in consequence: three of its decisions, taken
carelessly, destroy RQ1's attribution **silently**. The runs complete, the
numbers look plausible, and the primary result means something other than what it
says. So most of this file asserts properties whose violation would not crash
anything:

- the **sensor** runs on true geometry at every rung, so the F0->F1 gap is the
  cost of ignoring buildings *in the radio* rather than the cost of deleting the
  city (decision 1);
- the **diagnostics** run on true geometry at every rung, so `chain_occluded`
  does not read 0.0 % under F0 by construction (decision 2);
- **F3's jammer** is a construction-time flag, not the curriculum's `jammer_on`,
  so RQ1's jammer rung is not confounded with the curriculum ramp (decision 4);
- the rungs are **one composed enum**, so a half-specified condition that is not
  on the ladder cannot be built at all (decision 5);
- the **reward** is untouched by fidelity: only the physics feeding it changes.

`test_golden.py` carries the fifth and heaviest property -- that `F4` reproduces
the environment Blocks D and E measured, exactly.
"""

from __future__ import annotations

import dataclasses
import functools

import pytest
import torch

from . import channel
from .core import (
    BANDWIDTH_HZ,
    F0_CAPACITY_MBPS,
    F0_RADIUS_M,
    LADDER,
    BatchedSwarmEnv,
    EnvConfig,
)
from .reward import DEFAULT_WEIGHTS

RUNGS = ("F0", "F1", "F2", "F3", "F4")
STAGE4 = (0.0, 0.0, 0.0, 1.0)


def make(fidelity="F4", num_envs=4, num_drones=5, seed=0, **kw) -> BatchedSwarmEnv:
    kw.setdefault("stage_weights", STAGE4)
    kw.setdefault("compile_occlusion", False)
    env = BatchedSwarmEnv(
        EnvConfig(num_envs=num_envs, num_drones=num_drones, seed=seed, fidelity=fidelity, **kw)
    )
    env.reset()
    return env


def fly(env: BatchedSwarmEnv, steps: int, seed: int = 7) -> None:
    """Push the swarm somewhere interesting with a seeded action sequence.

    Identical across rungs because the actions do not read the observation --
    which is what lets the comparisons below hold geometry fixed and vary only
    the channel.
    """
    gen = torch.Generator(device=env.device).manual_seed(seed)
    b, n = env.cfg.num_envs, env.cfg.num_drones
    for _ in range(steps):
        env.step(torch.rand(b, n, 3, generator=gen, device=env.device) * 2.0 - 1.0)


@functools.cache
def evaluate_at(fidelity: str, steps: int = 40, **kw):
    """`(env, snapshot, aux)` after an identical rollout at the given rung.

    Memoised: occlusion is ~99 % of a step and this file asks for the same five
    rollouts a dozen times over. Callers only read the result.
    """
    env = make(fidelity=fidelity, **kw)
    fly(env, steps)
    snap, aux = env._evaluate()
    return env, snap, aux


# --------------------------------------------------------------------------- #
# Decision 5 -- one composed enum, not five independent flags
# --------------------------------------------------------------------------- #


def test_the_ladder_matches_the_table_in_thesis_plan():
    """Cumulative, one effect per rung. If this table drifts, RQ1's attribution
    stops meaning what THESIS_PLAN §2 says it means."""
    assert tuple(LADDER) == RUNGS
    assert [LADDER[r].channel_occlusion for r in RUNGS] == [False, True, True, True, True]
    assert [LADDER[r].binary_capacity for r in RUNGS] == [True, True, False, False, False]
    assert [LADDER[r].channel_jammer for r in RUNGS] == [False, False, False, True, True]
    assert [LADDER[r].reuse_limit for r in RUNGS] == [1, 1, 1, 1, 3]


def test_the_flags_cannot_be_set_independently():
    """`channel_occlusion=False, channel_jammer=True` is not on the ladder, and
    nothing else would stop it running or stop its number reaching a table."""
    for flag in ("channel_occlusion", "binary_capacity", "channel_jammer", "reuse_limit"):
        with pytest.raises(TypeError):
            EnvConfig(num_envs=1, **{flag: True})
    fields = {f.name for f in dataclasses.fields(EnvConfig)}
    assert not fields & {"channel_occlusion", "binary_capacity", "channel_jammer", "reuse_limit"}


def test_an_unknown_rung_is_refused_at_construction():
    with pytest.raises(ValueError, match="fidelity must be one of"):
        EnvConfig(num_envs=1, fidelity="F5")
    with pytest.raises(ValueError, match="fidelity must be one of"):
        EnvConfig(num_envs=1, fidelity="F0-nogeo")  # a real rung, but not this seam


def test_the_duplexing_override_is_confined_to_the_full_model():
    """PHYSICS.md wants the result under more than one duplexing assumption, and
    that is a robustness check on F4. At F0-F3 the rung pins the divisor, so an
    override there would silently produce an off-ladder condition."""
    assert EnvConfig(num_envs=1, fidelity="F4", duplexing_override=1).reuse_limit == 1
    assert EnvConfig(num_envs=1, fidelity="F4", duplexing_override=5).reuse_limit == 5
    for rung in ("F0", "F1", "F2", "F3"):
        with pytest.raises(ValueError, match="only valid at fidelity='F4'"):
            EnvConfig(num_envs=1, fidelity=rung, duplexing_override=3)
    with pytest.raises(ValueError, match="must be >= 1"):
        EnvConfig(num_envs=1, fidelity="F4", duplexing_override=0)


def test_no_buildings_is_orthogonal_to_the_ladder():
    """It removes buildings from the WORLD, which is a different claim from any
    rung. `F0-nogeo` is that combination, named and reported as such -- never
    folded into F0, where it would confound the primary result."""
    assert "no_buildings" in {f.name for f in dataclasses.fields(EnvConfig)}
    for rung in RUNGS:
        assert EnvConfig(num_envs=1, fidelity=rung).no_buildings is False
    nogeo = EnvConfig(num_envs=1, fidelity="F0", no_buildings=True)
    assert nogeo.fidelity == "F0" and nogeo.no_buildings


# --------------------------------------------------------------------------- #
# Decision 3 -- C_max is the modulation ceiling, not a new free parameter
# --------------------------------------------------------------------------- #


def test_c_max_is_the_channel_modules_own_ceiling():
    """Traceable to `channel.capacity_mbps` rather than asserted: it is where a
    link lands when nothing degrades it, which is the honest reading of "a
    connected link runs at full rate"."""
    saturated = channel.capacity_mbps(torch.tensor([200.0]), BANDWIDTH_HZ)
    assert torch.allclose(saturated, torch.tensor([F0_CAPACITY_MBPS]))
    assert F0_CAPACITY_MBPS == pytest.approx(74.0)


@pytest.mark.parametrize("rung", ("F0", "F1"))
def test_binary_rungs_deliver_c_max_or_nothing(rung):
    _env, _snap, aux = evaluate_at(rung)
    cap = aux["capacity_mbps"]
    assert torch.isin(cap, torch.tensor([0.0, F0_CAPACITY_MBPS])).all()
    assert (cap > 0).any(), "a binary rung with no usable link at all pins nothing"


@pytest.mark.parametrize("rung", ("F2", "F3", "F4"))
def test_continuous_rungs_are_not_binary(rung):
    _env, _snap, aux = evaluate_at(rung)
    cap = aux["capacity_mbps"]
    interior = (cap > 1e-6) & (cap < F0_CAPACITY_MBPS - 1e-6)
    assert interior.any(), "a continuous rung should produce partially-degraded links"


def test_f0_ignores_buildings_and_f1_does_not():
    """The one difference between the two binary rungs, isolated."""
    _e0, _s0, aux0 = evaluate_at("F0")
    _e1, _s1, aux1 = evaluate_at("F1")
    blocked = aux0["true_clearance"][:, :6, :6] < 0.0
    assert blocked.any(), "the rollout must actually meet some buildings"
    # F0 carries traffic over blocked pairs; F1 refuses every one of them.
    assert (aux0["capacity_mbps"] * blocked).sum() > 0.0
    assert (aux1["capacity_mbps"] * blocked).sum() == 0.0


# --------------------------------------------------------------------------- #
# Decision 1 -- fidelity gates the CHANNEL, never the sensor
# --------------------------------------------------------------------------- #


def test_the_true_geometry_is_identical_across_rungs():
    ref = evaluate_at("F4")[2]["true_clearance"]
    for rung in RUNGS:
        got = evaluate_at(rung)[2]["true_clearance"]
        assert torch.equal(got, ref), f"{rung} sees different geometry from F4"


def test_the_sensor_is_identical_across_rungs():
    """RQ1 asks which effects a *channel model* must include, and a camera is
    not part of a channel model. Letting the sensor vary would also let the
    larger, uninteresting effect swamp the smaller, interesting one -- Block E
    measured observation as ~93 % solved by geometry while the chain binds."""
    _env, ref_snap, ref_aux = evaluate_at("F4")
    for rung in RUNGS:
        _env, snap, aux = evaluate_at(rung)
        assert torch.equal(aux["sees_hvt"], ref_aux["sees_hvt"]), f"{rung}: sensor moved"
        assert torch.equal(snap.observed, ref_snap.observed), f"{rung}: `observed` moved"
        assert torch.equal(snap.best_clearance_m, ref_snap.best_clearance_m), rung
        assert torch.equal(snap.nearest_dist_m, ref_snap.nearest_dist_m), rung


def test_the_sensor_still_works_when_the_channel_ignores_buildings():
    """The failure this guards is the one the old `use_occlusion` seam would
    have produced: under it, "F0" meant a city with no buildings and every drone
    saw the HVT anywhere within 830 m."""
    _env, _snap, aux = evaluate_at("F0")
    blocked_sightline = aux["true_clearance"][:, :5, -1] < 0.0
    assert blocked_sightline.any(), "the rollout must contain a blocked sightline"
    assert not (aux["sees_hvt"] & blocked_sightline).any(), (
        "a drone saw the HVT through a building at F0 -- the sensor is being gated"
    )


# --------------------------------------------------------------------------- #
# Decision 2 -- the diagnostics always use TRUE occlusion
# --------------------------------------------------------------------------- #


def test_chain_occluded_is_not_zero_under_f0():
    """The headline failure-attribution metric, in the one condition it exists
    to expose. Computed from the fidelity-gated clearance it would read 0.0 % by
    construction: the F0 policy routes straight through towers and the metric
    would report that it never happens.

    ⚠️ 0.0 % here is a bug, not a finding.
    """
    _env, _snap, aux = evaluate_at("F0", steps=60)
    assert aux["chain_occluded"].any(), (
        "chain_occluded is zero at F0. Either the diagnostic is reading the gated "
        "clearance (decision 2 violated) or the rollout never routed through a "
        "building -- check the second before believing the first."
    )


def test_chain_occluded_is_a_true_geometry_reading_at_every_rung():
    """Cross the chain the router actually chose against the real buildings, by
    hand, and check the env agrees -- at every rung, including the ones where
    the channel pretends the buildings are not there."""
    for rung in RUNGS:
        _env, _snap, aux = evaluate_at(rung, steps=60)
        r = 6  # n_radio for num_drones=5
        expect = (aux["on_edge"] & (aux["true_clearance"][:, :r, :r] < 0.0)).any(dim=-1).any(dim=-1)
        assert torch.equal(aux["chain_occluded"], expect), rung


def test_f1_never_routes_through_a_building():
    """Not a tautology about the metric -- a statement about the physics. F1 is
    the only rung where occlusion is a hard veto rather than a graded penalty,
    so a chosen chain cannot contain a blocked link. That the diagnostic reports
    0 here and non-zero at F0 is the difference the rung exists to create."""
    _env, _snap, aux = evaluate_at("F1", steps=60)
    assert not aux["chain_occluded"].any()


# --------------------------------------------------------------------------- #
# Decision 4 -- F3's jammer is NOT the curriculum's jammer
# --------------------------------------------------------------------------- #


def test_the_curriculum_ramp_is_identical_across_rungs():
    """`jammer_on` is drawn per episode from the stage table, and
    docs/ENVIRONMENT.md requires that ramp to run identically in every fidelity
    condition with the rung deciding whether it *does* anything. Driving F3 from
    it would confound RQ1's jammer rung with the curriculum, unrecoverably.

    Drawn over a mixed stage weighting so the jammer axis actually varies.
    """
    mixed = (0.25, 0.25, 0.25, 0.25)
    draws = {}
    for rung in RUNGS:
        env = make(fidelity=rung, num_envs=64, stage_weights=mixed, seed=11)
        rows = []
        for _ in range(8):
            env._sample_episode(torch.ones(64, dtype=torch.bool, device=env.device))
            rows.append(
                torch.stack(
                    [env.jammer_on, env.speed_scale, env.episode_len, env.route_id.float()]
                ).clone()
            )
        draws[rung] = torch.stack(rows)
    assert draws["F4"][:, 0].min() == 0.0 and draws["F4"][:, 0].max() == 1.0, (
        "the mixed weighting must actually exercise both jammer states"
    )
    for rung in RUNGS:
        assert torch.equal(draws[rung], draws["F4"]), f"{rung}: the curriculum ramp moved"


def test_the_jammer_reaches_the_denominator_only_at_f3_and_above():
    """The rung decides whether the ramp does anything. With `jammer_on = 1`
    throughout (stage 4), jam power must be identically zero below F3 and
    strictly positive at or above it."""
    for rung in RUNGS:
        env, _snap, aux = evaluate_at(rung)
        assert (env.jammer_on == 1.0).all(), "stage 4 must have the curriculum jammer on"
        if rung in ("F0", "F1", "F2"):
            assert (aux["jam_mw"] == 0.0).all(), f"{rung} put the jammer in the channel"
        else:
            assert (aux["jam_mw"] > 0.0).all(), f"{rung} left the jammer out of the channel"


def test_the_jammer_only_costs_capacity_where_the_channel_is_continuous():
    """F2 -> F3 is the rung that adds the threat, and it must be visible in the
    capacity matrix. F1 -> a hypothetical binary-with-jammer is not on the
    ladder precisely because a binary link cannot express partial degradation --
    which is why F2 has to come before F3."""
    _e2, _s2, aux2 = evaluate_at("F2")
    _e3, _s3, aux3 = evaluate_at("F3")
    assert (aux3["capacity_mbps"] <= aux2["capacity_mbps"] + 1e-4).all()
    assert (aux3["capacity_mbps"] < aux2["capacity_mbps"] - 1e-4).any()


# --------------------------------------------------------------------------- #
# The reward is untouched by fidelity
# --------------------------------------------------------------------------- #


def test_the_reward_weights_are_untouched_by_fidelity():
    for rung in RUNGS:
        assert make(fidelity=rung).weights == DEFAULT_WEIGHTS


def test_only_the_channel_term_of_the_snapshot_moves_across_rungs():
    """`REWARD.md`: only the physics feeding the reward changes. Every non-channel
    input to `reward()` must be bit-identical across rungs for a fixed state, so
    the F0-F4 comparison cannot be contaminated by a differently-shaped
    objective. `e2e_capacity_mbps` is the one field allowed to move -- it is the
    channel."""
    _env, ref, _aux = evaluate_at("F4")
    for rung in RUNGS:
        _env, snap, _aux = evaluate_at(rung)
        for field in (
            "observed",
            "nearest_dist_m",
            "best_clearance_m",
            "battery",
            "speed_ms",
            "accel_ms2",
        ):
            assert torch.equal(getattr(snap, field), getattr(ref, field)), f"{rung}/{field}"
    # ...and the channel really does move. Asserted on the capacity matrix
    # rather than on `e2e`: with the swarm still near the MCV a single strong
    # chain saturates under every rung, so `e2e` can coincide while the physics
    # underneath is completely different.
    caps = {r: evaluate_at(r)[2]["capacity_mbps"] for r in RUNGS}
    assert not torch.equal(caps["F0"], caps["F4"]), "the rungs must differ somewhere"


# --------------------------------------------------------------------------- #
# Monotonicity of permissiveness
# --------------------------------------------------------------------------- #


def test_permissiveness_orderings_that_hold():
    """Three of them, end to end.

    `F0 >= F1` holds because F1's capacity is F0's times an unoccluded mask,
    elementwise, and `best_relay_path` is monotone in the capacity matrix at a
    fixed `reuse_limit`. (docs/BLOCK_F.md folded this in with `F1 >= F2` and
    called the pair not guaranteed; the first half is in fact guaranteed and is
    worth asserting.)

    `F2 >= F3` is the jammer entering the denominator; `F3 >= F4` is the divisor
    `min(n, 3) >= 1` at an unchanged capacity matrix.
    """
    caps, e2e = {}, {}
    for rung in RUNGS:
        _env, snap, aux = evaluate_at(rung, steps=60)
        caps[rung], e2e[rung] = aux["capacity_mbps"], snap.e2e_capacity_mbps
    for hi, lo in (("F0", "F1"), ("F2", "F3"), ("F3", "F4")):
        assert (caps[hi] >= caps[lo] - 1e-4).all(), f"per-link {hi} >= {lo} violated"
        assert (e2e[hi] >= e2e[lo] - 1e-4).all(), f"end-to-end {hi} >= {lo} violated"


def test_f2_can_exceed_f1_because_f2_has_no_radius_cutoff():
    """The non-monotone direction that is guaranteed by construction, and the
    reason `F1 >= F2` is not an ordering the ladder may assume.

    F2 is not a restriction of F1. It drops the radius cutoff, so a link beyond
    `R` reads 0 under F1 and positive under F2; and it turns occlusion from a
    hard veto into a blockage penalty, so a blocked-but-usable link reads 0
    under F1 and positive under F2.
    """
    _e1, _s1, aux1 = evaluate_at("F1", steps=60)
    _e2, _s2, aux2 = evaluate_at("F2", steps=60)
    assert (aux2["capacity_mbps"] > aux1["capacity_mbps"] + 1e-4).any()


def test_f1_can_exceed_f2_only_where_a_connected_link_is_below_the_cap():
    """The other direction, and it depends on `R` -- which is why the ordering
    is stated as absent rather than reversed.

    At the calibrated `R` a link just inside the radius is typically still
    saturated at `C_max` (a clear 500 m link runs ~30 dB above what 7.4 b/s/Hz
    needs), so F1 = F2 = 74 Mbps there and the F1 > F2 direction may not occur
    at all in a given rollout. Widen `R` until connected links are genuinely
    long and it appears immediately: F1 pays a connected link the ceiling while
    F2 pays it Shannon.

    Stated this way the test measures the mechanism instead of the sample, and
    it does not silently start failing when `calibrate_r.py` moves `R`.
    """
    wide = 5_000.0  # every pair in the box is "connected"
    _e1, _s1, aux1 = evaluate_at("F1", steps=60, radius_m=wide)
    _e2, _s2, aux2 = evaluate_at("F2", steps=60, radius_m=wide)
    assert (aux1["capacity_mbps"] > aux2["capacity_mbps"] + 1e-4).any()


# --------------------------------------------------------------------------- #
# Every rung runs
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("rung", RUNGS)
def test_every_rung_steps_cleanly(rung):
    env = make(fidelity=rung, num_envs=8)
    fly(env, 20)
    _snap, aux = env._evaluate()
    obs = env._observe(aux)
    for name, tensor in list(aux.items()) + list(obs.items()):
        if tensor.is_floating_point():
            assert torch.isfinite(tensor).all(), f"{rung}: {name} is not finite"


@pytest.mark.parametrize("rung", RUNGS)
def test_every_rung_packs_the_same_observation_shape(rung):
    """The observation contract is identical across rungs -- MODELS.md needs one
    architecture to consume all five. Only the *values* in the channel-derived
    dims move."""
    env = make(fidelity=rung)
    obs = env._observe(env._evaluate()[1])
    assert obs["flat"].shape == (env.cfg.num_envs, env.cfg.num_drones, 108)
    assert obs["state"].shape == (env.cfg.num_envs, env.cfg.state_dim)


# --------------------------------------------------------------------------- #
# `R` -- the fairness requirement on RQ1
# --------------------------------------------------------------------------- #

#: What `scripts/calibrate_r.py` reported, 8 seeds x 64 eval episodes under B0.
#: Pinned here so a change to `F0_RADIUS_M` has to come with a re-measurement.
CALIBRATED_R_M = 524.0
DEGREE_MATCHED_R_M = 418.0


def test_r_is_the_calibrated_value_and_not_a_guess():
    """THESIS_PLAN §2 makes `R` the fairness requirement on RQ1 and warns it is
    "the first thing an examiner will probe". An arbitrary `R` makes the whole
    F0 arm meaningless, so the constant is pinned to the measurement."""
    assert F0_RADIUS_M == CALIBRATED_R_M


def test_the_two_calibration_methods_agree_within_the_reported_sensitivity():
    """The substantive property, not a restatement of the constant.

    The pre-registered method (median link *range*) and the degree-matching
    cross-check are independent -- one is a quantile of the usability curve, the
    other solves for equal mean node degree. docs/BLOCK_F.md's position is that
    nothing in RQ1 turns on the choice between them **because** they agree to
    within the +-25 % sensitivity that is swept and reported anyway. If that
    stops being true, the disagreement becomes the finding and the deviation
    has to be argued with fresh evidence attached.
    """
    ratio = DEGREE_MATCHED_R_M / CALIBRATED_R_M
    assert 0.75 <= ratio <= 1.25, f"the two methods now disagree by {ratio:.2f}x"
