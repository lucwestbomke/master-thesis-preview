"""
Reward function, as a pure function of a state summary.

Pure on purpose: a "policy" in the tests is then just a hand-written list of
snapshots, so the reward can be validated with no environment, no simulator and
no training run. It also means those tests survive the batched env replacing the
PettingZoo stub, because they never depended on either.

Design rationale, weight-setting method and the known degenerate optima are in
AGENTS.md -> Reward. The short version:

- The dominant term IS the headline metric (fraction of steps mission-capable),
  so the policy optimises exactly the number that gets reported.
- All *guidance* is potential-based (Ng, Harada & Russell 1999), which provably
  cannot move the optimum. A plain proximity bonus is a salary that grows with
  time loitering; PBRS pays once for real progress and round trips cancel.
- Weights are pinned by behavioural orderings, not swept. Only lambda is swept.

Everything is batched, pure torch, device-agnostic, free of .item()/.cpu().
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import torch

from .energy import DEFAULT_AIRFRAME, Rotorcraft, hover_power_w, radio_dc_power_w, total_power_w

CAPACITY_THRESHOLD_MBPS = 5.0


def hover_reference_power_w(craft: Rotorcraft) -> float:
    """Total *electrical* draw while hovering -- the normaliser for the energy
    term, so it reads 1.0 at hover and ~0.65 at the minimum-power airspeed.

    Must match what `total_power_w` returns (electrical, radio included);
    normalising electrical draw by *shaft* hover power would silently shift the
    whole energy term by the drivetrain efficiency.
    """
    return hover_power_w(craft) / craft.drivetrain_efficiency + radio_dc_power_w()


@dataclass(frozen=True)
class RewardWeights:
    """Objective weights, then the potential.

    Objective weights change what is optimal, so they are pinned by the
    orderings in `weight_constraints_satisfied`. Potential weights cannot move
    the optimum, so they are free to tune for learning speed.
    """

    # --- objective: these define the mission ---
    mission: float = 1.0  # the unit. One step of full mission capability.
    idle: float = 0.3  # per step with no observation at all
    energy: float = 0.15  # relative to hover power
    battery_variance: float = 0.5  # lambda -- the ONLY swept weight
    effort: float = 0.01  # control-effort heuristic, deliberately tiny

    # --- potential: guidance only, provably optimum-preserving ---
    potential_scale: float = 10.0  # k: full swing worth ~10 steps of mission
    w_approach: float = 0.25
    w_observe: float = 0.35
    w_link: float = 0.40
    tau_clearance_m: float = 15.0  # ~building height, ~2 steps of travel
    tau_capacity_mbps: float = 2.0  # 40% of threshold
    d_ref_m: float = 1500.0  # map scale

    # --- physical references for normalisation ---
    max_accel_ms2: float = 10.0


DEFAULT_WEIGHTS = RewardWeights()


@dataclass(frozen=True)
class Snapshot:
    """Everything the reward needs. Team quantities are (B,), per-drone (B, N)."""

    observed: torch.Tensor  # (B,) bool -- does ANY drone hold the HVT
    e2e_capacity_mbps: torch.Tensor  # (B,)
    nearest_dist_m: torch.Tensor  # (B,) closest drone to the HVT
    best_clearance_m: torch.Tensor  # (B,) best ray clearance, signed metres
    battery: torch.Tensor  # (B, N) in [0, 1]
    speed_ms: torch.Tensor  # (B, N)
    accel_ms2: torch.Tensor  # (B, N)

    @property
    def n_agents(self) -> int:
        return self.battery.shape[-1]


# --------------------------------------------------------------------------- #
# Potential
# --------------------------------------------------------------------------- #


def potential(snap: Snapshot, w: RewardWeights) -> torch.Tensor:
    """(B,) team potential. Three components with a deliberate handover.

    All three are TEAM quantities. Per-drone potentials would pull every drone
    onto the HVT and leave nobody relaying.

    Summed, not multiplied: a product is flat at episode start, when the drones
    are parked on the MCV and both observation and link are ~0, so neither can
    improve without the other.
    """
    # Coarse: non-zero anywhere on the map, so the agent is never blind.
    approach = 1.0 - (snap.nearest_dist_m / w.d_ref_m).clamp(0.0, 1.0)

    # Fine: rewards correct GEOMETRY, not mere proximity. The observation
    # envelope is a wedge down the street plus an overhead cone, so a drone
    # 20 m away across the street sees nothing while one 300 m down it sees
    # fine -- clearance captures that where distance cannot.
    observe = torch.sigmoid(snap.best_clearance_m / w.tau_clearance_m)

    # Gradient below threshold, where the binary link indicator has none.
    link = torch.sigmoid((snap.e2e_capacity_mbps - CAPACITY_THRESHOLD_MBPS) / w.tau_capacity_mbps)

    return w.potential_scale * (w.w_approach * approach + w.w_observe * observe + w.w_link * link)


def shaping(
    snap: Snapshot,
    next_snap: Snapshot,
    w: RewardWeights,
    gamma: float,
    next_is_terminal: torch.Tensor | None = None,
) -> torch.Tensor:
    """(B,) gamma*Phi(s') - Phi(s).

    `next_is_terminal` must flag genuine terminal states (battery death). The
    invariance proof requires Phi(terminal) = 0; otherwise gamma^T * Phi(s_T)
    survives the telescoping and reintroduces a policy-dependent bias.
    Truncation is not terminal -- bootstrap the value there instead.
    """
    phi_next = potential(next_snap, w)
    if next_is_terminal is not None:
        phi_next = torch.where(next_is_terminal, torch.zeros_like(phi_next), phi_next)
    return gamma * phi_next - potential(snap, w)


# --------------------------------------------------------------------------- #
# Objective
# --------------------------------------------------------------------------- #


def mission_capable(snap: Snapshot) -> torch.Tensor:
    """(B,) bool. The headline metric, and the dominant reward term."""
    return snap.observed & (snap.e2e_capacity_mbps >= CAPACITY_THRESHOLD_MBPS)


def team_reward(snap: Snapshot, w: RewardWeights) -> torch.Tensor:
    """(B,) terms that are properties of the swarm, not of any one drone."""
    capable = mission_capable(snap).to(snap.e2e_capacity_mbps.dtype)
    idle = (~snap.observed).to(capable.dtype)
    # Population variance: the actual spread, not a sample estimate.
    var_b = snap.battery.var(dim=-1, unbiased=False)
    return w.mission * capable - w.idle * idle - w.battery_variance * var_b


def individual_reward(
    snap: Snapshot, w: RewardWeights, craft: Rotorcraft = DEFAULT_AIRFRAME
) -> torch.Tensor:
    """(B, N) costs each drone pays for itself.

    Energy is individual while the mission reward is shared, which creates a
    free-rider incentive -- let the others work and hover cheaply. `lambda *
    Var(B)` is the counter-mechanism, not merely a rotation device.
    """
    power = total_power_w(snap.speed_ms, snap.accel_ms2, craft)
    energy = power / hover_reference_power_w(craft)  # 1.0 at hover, ~0.65 cruising
    effort = (snap.accel_ms2 / w.max_accel_ms2) ** 2
    return -w.energy * energy - w.effort * effort


def reward(
    snap: Snapshot,
    next_snap: Snapshot,
    w: RewardWeights | None = None,
    gamma: float = 0.999,
    next_is_terminal: torch.Tensor | None = None,
    craft: Rotorcraft = DEFAULT_AIRFRAME,
) -> torch.Tensor:
    """(B, N) per-agent reward: shared team terms plus individual costs."""
    w = w or DEFAULT_WEIGHTS
    team = team_reward(snap, w) + shaping(snap, next_snap, w, gamma, next_is_terminal)
    return team.unsqueeze(-1) + individual_reward(snap, w, craft)


def episode_return(
    trajectory: list[Snapshot],
    w: RewardWeights | None = None,
    gamma: float = 0.999,
    craft: Rotorcraft = DEFAULT_AIRFRAME,
) -> torch.Tensor:
    """(B,) undiscounted mean-over-agents return. Diagnostic / test use."""
    w = w or DEFAULT_WEIGHTS
    total = torch.zeros_like(trajectory[0].e2e_capacity_mbps)
    for s, s_next in pairwise(trajectory):
        total = total + reward(s, s_next, w, gamma, craft=craft).mean(dim=-1)
    return total


# --------------------------------------------------------------------------- #
# The weight-setting method, expressed as checkable predicates
# --------------------------------------------------------------------------- #


def weight_constraints_satisfied(
    w: RewardWeights, craft: Rotorcraft = DEFAULT_AIRFRAME
) -> dict[str, bool]:
    """Each entry is a behavioural ordering the reward must reproduce.

    This is how the objective weights are *set* -- write down pairs of policies
    whose ranking you already know, require the reward to rank them correctly,
    and solve the resulting inequalities. Sweeping six weights is forbidden by
    the plan; this is what replaces it.
    """
    p_ref = hover_reference_power_w(craft)
    # Energy cost, in units of hover draw, of the cheapest and most expensive
    # flight a chasing drone plausibly sustains.
    e_loiter = total_power_w(torch.tensor(13.3), torch.tensor(0.0), craft).item() / p_ref
    e_dash = total_power_w(torch.tensor(25.0), torch.tensor(0.0), craft).item() / p_ref

    max_variance = 0.25  # battery in [0,1]; worst case is half at each extreme

    return {
        # Chasing must beat loitering even at the most expensive airspeed,
        # otherwise "never acquire" is cheaper than trying and failing.
        "trying_beats_loitering": w.idle > w.energy * (e_dash - e_loiter),
        # Full mission success must beat the safe partial success of observing
        # forever without ever closing the link.
        "success_beats_partial": w.mission > w.idle,
        # All drones hovering gives Var(B)=0 and so scores perfectly on the
        # variance term. Mission reward must dominate that.
        "mission_beats_balance": w.mission > w.battery_variance * max_variance,
        # Energy must not be able to veto flying at all.
        "energy_cannot_veto_mission": w.mission > w.energy * e_dash,
        # Control effort is a heuristic, not an objective.
        "effort_stays_negligible": w.effort < 0.1 * w.energy,
        # Potential guidance must be able to dominate when mission reward is
        # zero (all of early training) yet stay negligible over a full episode.
        "potential_guides_without_dominating": 3.0 < w.potential_scale < 50.0,
    }
