"""
Reward validation. Scores hand-written policies and asserts their ranking.

This is the cheapest bug-catcher in the project: if a competent policy does not
outscore a lazy one, the reward is wrong, and it is found in milliseconds instead
of after a three-hour training run. The reward is a specification artefact like
the channel model, so it gets the same treatment.

The policies are trajectories of `Snapshot`s, so none of this needs an
environment -- which also means these tests survive the batched env replacing the
PettingZoo stub.
"""

from itertools import pairwise

import pytest
import torch

from .reward import (
    CAPACITY_THRESHOLD_MBPS,
    RewardWeights,
    Snapshot,
    episode_return,
    hover_reference_power_w,
    individual_reward,
    mission_capable,
    potential,
    shaping,
    team_reward,
    weight_constraints_satisfied,
)

N = 5
W = RewardWeights()


# Capacity levels, expressed RELATIVE to the requirement rather than as magic
# numbers. When CAPACITY_THRESHOLD_MBPS moved 5 -> 15 (docs/BLOCK_E.md), the
# hardcoded 9.0 and 12.0 in these stubs silently crossed from "comfortably
# delivering" to "failing" and three tests inverted. Naming the intent means the
# stubs track the constant instead of drifting behind it.
GOOD = 2.4 * CAPACITY_THRESHOLD_MBPS  # a healthy chain
OK = 1.8 * CAPACITY_THRESHOLD_MBPS  # delivering, with margin
POOR = 0.4 * CAPACITY_THRESHOLD_MBPS  # chain up, below the bar
BAD = 0.1 * CAPACITY_THRESHOLD_MBPS  # barely a link at all


def _snap(observed, cap, dist, clear, speed, accel=1.0, battery=0.8):
    """One timestep for a single environment with N drones."""
    return Snapshot(
        observed=torch.tensor([bool(observed)]),
        e2e_capacity_mbps=torch.tensor([float(cap)]),
        nearest_dist_m=torch.tensor([float(dist)]),
        best_clearance_m=torch.tensor([float(clear)]),
        battery=torch.full((1, N), float(battery)),
        speed_ms=torch.full((1, N), float(speed)),
        accel_ms2=torch.full((1, N), float(accel)),
    )


# --------------------------------------------------------------------------- #
# Four scripted policies
# --------------------------------------------------------------------------- #


def lazy(steps=100):
    """Never leaves the MCV. The optimum the idle penalty exists to destroy."""
    return [_snap(False, 0.0, 1400, -60, 0.0, 0.0) for _ in range(steps)]


def all_chase(steps=100):
    """All five drones pile onto the HVT. Sees it perfectly, relays nothing."""
    return [_snap(True, 0.0, 20, 30, 18.0, 3.0) for _ in range(steps)]


def fixed_formation(steps=100):
    """Flies out and holds a rigid line. Works until the target turns."""
    out = []
    for t in range(steps):
        ok = (t % 10) < 5  # link holds half the time
        seen = (t % 10) < 6  # sight held a little more often
        out.append(
            _snap(seen, OK if ok else POOR, 60 if seen else 200, 20 if seen else -20, 8.0, 1.0)
        )
    return out


def heuristic(steps=100):
    """The B0 scripted geometric baseline: relays on the MCV->HVT geodesic."""
    out = []
    for t in range(steps):
        ok = (t % 20) < 17
        seen = (t % 10) < 9
        out.append(
            _snap(seen, GOOD if ok else POOR, 30 if seen else 90, 25 if seen else -5, 14.0, 2.0)
        )
    return out


# --------------------------------------------------------------------------- #
# The ordering -- the whole point of this file
# --------------------------------------------------------------------------- #


def test_policies_rank_in_the_expected_order():
    r_heur = episode_return(heuristic()).item()
    r_form = episode_return(fixed_formation()).item()
    r_chase = episode_return(all_chase()).item()
    r_lazy = episode_return(lazy()).item()

    assert r_heur > r_form > r_chase > r_lazy, (
        f"heuristic={r_heur:.1f} formation={r_form:.1f} chase={r_chase:.1f} lazy={r_lazy:.1f}"
    )


def test_lazy_is_strictly_punished():
    """Loitering must accrue unbounded negative reward, not merely zero.

    Fixed-length episodes do not by themselves kill the lazy optimum: never
    acquiring also means never flying out, and at 25 m/s dash a chasing drone
    briefly costs *more* than hovering. The idle penalty is what breaks the tie.
    """
    assert episode_return(lazy()).item() < 0.0
    assert episode_return(lazy(200)).item() < 2.0 * episode_return(lazy(100)).item() + 1e-6


def test_seeing_without_relaying_beats_seeing_nothing():
    """Acquisition is real progress even when the chain has not formed."""
    assert episode_return(all_chase()).item() > episode_return(lazy()).item()


def test_but_seeing_without_relaying_never_beats_the_mission():
    """Guards the clustering optimum: all five drones observing and nobody
    relaying must not outscore an actual working chain."""
    assert episode_return(heuristic()).item() > episode_return(all_chase()).item()


# --------------------------------------------------------------------------- #
# Potential-based shaping: the invariance property
# --------------------------------------------------------------------------- #


def test_shaping_depends_only_on_endpoints():
    """The property that makes PBRS safe. Two trajectories, same start and end,
    wildly different middles -- identical total shaping."""
    start = _snap(False, 0.0, 1400, -60, 0.0)
    end = _snap(True, 4.0 * CAPACITY_THRESHOLD_MBPS, 30, 30, 12.0)
    direct = [start, _snap(True, OK, 300, 5, 15.0), end]
    wandering = [
        start,
        _snap(False, 0.0, 200, -40, 20.0),
        _snap(True, GOOD, 40, 25, 10.0),
        _snap(False, 0.0, 900, -50, 22.0),  # throws it all away
        _snap(True, POOR, 120, 10, 18.0),
        end,
    ]

    def total_shaping(traj):
        return sum(shaping(a, b, W, gamma=1.0).item() for a, b in pairwise(traj))

    assert total_shaping(direct) == pytest.approx(total_shaping(wandering), abs=1e-5)
    # and equals Phi(end) - Phi(start) exactly
    assert total_shaping(direct) == pytest.approx(
        (potential(end, W) - potential(start, W)).item(), abs=1e-5
    )


def test_a_round_trip_earns_nothing():
    """Nothing to farm: returning to where you started returns the potential."""
    a = _snap(False, 0.0, 800, -30, 10.0)
    b = _snap(True, OK, 40, 20, 10.0)
    out = shaping(a, b, W, gamma=1.0).item()
    back = shaping(b, a, W, gamma=1.0).item()
    assert out + back == pytest.approx(0.0, abs=1e-6)
    assert out > 0.0  # progress genuinely pays on the way there


def test_terminal_potential_is_zeroed():
    """Required by the invariance proof -- otherwise gamma^T*Phi(s_T) survives."""
    a = _snap(True, OK, 40, 20, 10.0)
    b = _snap(True, GOOD, 35, 22, 10.0)
    normal = shaping(a, b, W, 0.999).item()
    terminal = shaping(a, b, W, 0.999, next_is_terminal=torch.tensor([True])).item()
    assert terminal == pytest.approx(-potential(a, W).item(), abs=1e-5)
    assert terminal < normal


# --------------------------------------------------------------------------- #
# Potential structure
# --------------------------------------------------------------------------- #


def test_potential_has_gradient_at_episode_start():
    """The failure a product form would cause: drones parked on the MCV with
    nothing observed and no chain, and the potential flat in every direction."""
    parked = _snap(False, 0.0, 1400, -60, 0.0)
    stepped_out = _snap(False, 0.0, 1200, -60, 0.0)
    assert potential(stepped_out, W).item() > potential(parked, W).item()


def test_each_potential_component_moves_independently():
    """Sum, not product: improving one axis pays even when the others are zero."""
    base = _snap(False, 0.0, 1400, -60, 0.0)
    closer = _snap(False, 0.0, 700, -60, 0.0)
    clearer = _snap(False, 0.0, 1400, 20, 0.0)
    linked = _snap(False, GOOD, 1400, -60, 0.0)
    p0 = potential(base, W).item()
    assert potential(closer, W).item() > p0
    assert potential(clearer, W).item() > p0
    assert potential(linked, W).item() > p0


def test_potential_is_bounded_by_its_scale():
    best = _snap(True, 5.0 * CAPACITY_THRESHOLD_MBPS, 0, 200, 0.0)
    worst = _snap(False, 0.0, 5000, -300, 0.0)
    assert potential(worst, W).item() >= 0.0
    assert potential(best, W).item() <= W.potential_scale + 1e-6


def test_clearance_matters_more_than_distance_for_observation():
    """The wedge, not the disc. Close but blocked must score below far but
    clear on the observation component."""
    close_blocked = _snap(False, 0.0, 20, -40, 0.0)
    far_clear = _snap(False, 0.0, 300, 40, 0.0)
    obs_close = torch.sigmoid(torch.tensor(-40.0 / W.tau_clearance_m)).item()
    obs_far = torch.sigmoid(torch.tensor(40.0 / W.tau_clearance_m)).item()
    assert obs_far > obs_close
    assert potential(far_clear, W).item() > potential(close_blocked, W).item()


def test_capacity_potential_gives_gradient_below_threshold():
    """Where the binary indicator has none."""
    dead = _snap(False, BAD * 0.3, 500, 0, 10.0)
    improving = _snap(False, POOR, 500, 0, 10.0)
    assert potential(improving, W).item() > potential(dead, W).item() + 0.05


# --------------------------------------------------------------------------- #
# Objective terms
# --------------------------------------------------------------------------- #


def test_mission_capable_needs_both_conditions():
    assert mission_capable(_snap(True, OK, 30, 20, 10.0)).item()
    assert not mission_capable(_snap(True, POOR, 30, 20, 10.0)).item()
    assert not mission_capable(_snap(False, OK, 30, 20, 10.0)).item()


def test_mission_threshold_is_inclusive():
    assert mission_capable(_snap(True, CAPACITY_THRESHOLD_MBPS, 30, 20, 10.0)).item()


def test_energy_term_reads_one_at_hover():
    hovering = _snap(False, 0.0, 500, 0, 0.0, accel=0.0)
    expected = -W.energy * 1.0
    assert individual_reward(hovering, W)[0, 0].item() == pytest.approx(expected, rel=1e-4)


def test_cruising_costs_less_than_hovering():
    """The U-shape reaching the reward. If this ever flips, energy would push
    the swarm to hover rather than fly, inverting the observer's incentive."""
    hovering = _snap(False, 0.0, 500, 0, 0.0, accel=0.0)
    cruising = _snap(False, 0.0, 500, 0, 13.3, accel=0.0)
    assert individual_reward(cruising, W)[0, 0].item() > individual_reward(hovering, W)[0, 0].item()


def test_battery_variance_penalises_spread():
    even = Snapshot(
        observed=torch.tensor([True]),
        e2e_capacity_mbps=torch.tensor([OK]),
        nearest_dist_m=torch.tensor([50.0]),
        best_clearance_m=torch.tensor([20.0]),
        battery=torch.full((1, N), 0.6),
        speed_ms=torch.zeros(1, N),
        accel_ms2=torch.zeros(1, N),
    )
    spread = Snapshot(
        observed=torch.tensor([True]),
        e2e_capacity_mbps=torch.tensor([OK]),
        nearest_dist_m=torch.tensor([50.0]),
        best_clearance_m=torch.tensor([20.0]),
        battery=torch.tensor([[0.1, 0.35, 0.6, 0.85, 1.0]]),
        speed_ms=torch.zeros(1, N),
        accel_ms2=torch.zeros(1, N),
    )
    assert team_reward(even, W).item() > team_reward(spread, W).item()


def test_all_hover_does_not_beat_the_mission():
    """The variance term's own degenerate optimum: everyone hovers, batteries
    stay identical, Var(B)=0 and that term scores perfectly. Mission reward
    must dominate it."""
    hovering_balanced = _snap(False, 0.0, 1400, -60, 0.0, accel=0.0, battery=0.7)
    working = _snap(True, GOOD, 30, 25, 14.0, accel=2.0, battery=0.7)
    assert team_reward(working, W).item() > team_reward(hovering_balanced, W).item()


# --------------------------------------------------------------------------- #
# The weight-setting method
# --------------------------------------------------------------------------- #


def test_all_weight_constraints_hold_for_the_defaults():
    failed = [k for k, ok in weight_constraints_satisfied(W).items() if not ok]
    assert not failed, f"violated behavioural orderings: {failed}"


def test_constraints_actually_bite():
    """A guard against vacuous constraints -- each must reject a bad setting."""
    assert not weight_constraints_satisfied(RewardWeights(idle=2.0))["success_beats_partial"]
    assert not weight_constraints_satisfied(RewardWeights(battery_variance=10.0))[
        "mission_beats_balance"
    ]
    assert not weight_constraints_satisfied(RewardWeights(energy=5.0))["energy_cannot_veto_mission"]
    assert not weight_constraints_satisfied(RewardWeights(idle=0.0))["trying_beats_loitering"]


def test_lambda_is_the_only_weight_expected_to_move():
    """Sweeping lambda must not break the other orderings."""
    for lam in (0.0, 0.25, 0.5, 1.0, 2.0):
        checks = weight_constraints_satisfied(RewardWeights(battery_variance=lam))
        assert all(checks.values()), f"lambda={lam} broke {checks}"


# --------------------------------------------------------------------------- #
# Shapes / hygiene
# --------------------------------------------------------------------------- #


def test_reward_is_per_agent_and_batched():
    from .reward import reward

    b = 8
    snap = Snapshot(
        observed=torch.rand(b) > 0.5,
        e2e_capacity_mbps=torch.rand(b) * 20,
        nearest_dist_m=torch.rand(b) * 1500,
        best_clearance_m=(torch.rand(b) - 0.5) * 100,
        battery=torch.rand(b, N),
        speed_ms=torch.rand(b, N) * 25,
        accel_ms2=torch.rand(b, N) * 5,
    )
    r = reward(snap, snap)
    assert r.shape == (b, N)
    assert torch.isfinite(r).all()


def test_team_terms_are_shared_and_individual_terms_are_not():
    from .reward import reward

    snap = Snapshot(
        observed=torch.tensor([True]),
        e2e_capacity_mbps=torch.tensor([OK]),
        nearest_dist_m=torch.tensor([50.0]),
        best_clearance_m=torch.tensor([20.0]),
        battery=torch.full((1, N), 0.6),
        speed_ms=torch.tensor([[0.0, 5.0, 13.3, 20.0, 25.0]]),
        accel_ms2=torch.zeros(1, N),
    )
    r = reward(snap, snap)[0]
    assert r.std().item() > 0.0, "energy is individual, so agents must differ"
    # The spread comes only from the individual terms.
    ind = individual_reward(snap, W)[0]
    assert torch.allclose(r - ind, torch.full((N,), (r - ind)[0].item()), atol=1e-5)


def test_hover_reference_matches_the_energy_module():
    from .energy import DEFAULT_AIRFRAME, total_power_w

    # float32 tensor path vs float64 python path -- agreement to fp32 precision.
    direct = total_power_w(torch.tensor(0.0), torch.tensor(0.0), DEFAULT_AIRFRAME).item()
    assert hover_reference_power_w(DEFAULT_AIRFRAME) == pytest.approx(direct, rel=1e-6)
