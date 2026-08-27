"""
Reward function, as a pure function of a state summary.

Pure on purpose: a "policy" in the tests is then just a hand-written list of
snapshots, so the reward can be validated with no environment, no simulator and
no training run. It also means those tests survive the batched env replacing the
PettingZoo stub, because they never depended on either.

Design rationale, weight-setting method and the known degenerate optima are in
docs/REWARD.md. The short version:

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

# The end-to-end rate the mission requires, defined ONCE and read by the env,
# the metrics, the baselines and the renderer.
#
# **15, not 5.** Both are defensible for the stated payload -- 5 Mbps is a
# single compressed HD stream, 15 is a dual EO/IR feed at low latency, which is
# what a tracking ISR sortie actually carries -- and Block E measured what the
# choice does to the task. At 5 Mbps the radio link NEVER binds: the chain's
# bottleneck carries a median 37.6 Mbps, 8x the bar, so `mission_capable`
# reduces to `observed` and the whole mission collapses to "put one drone over
# the car". A scripted baseline reaches 93.2 % that way and the metric
# saturates. Measured for B0 at N=5 on the eval split:
#
#     requirement    5      10     15     20     30     40   Mbps
#     capable      93.3%  69.6%  54.7%  44.3%  19.4%  11.2%
#     sensor-only ceiling 93.4 %
#
# At 15 the binding constraint moves from the SENSOR to the RELAY CHAIN -- the
# drone can see the car and still cannot get the video home -- which is the
# swarm problem this project exists to study, and it is invisible at 5. It also
# revives F4's rate-division rung, inert at 5 because 37.6/3 still cleared it.
# See docs/BLOCK_E.md.
CAPACITY_THRESHOLD_MBPS = 15.0


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
    tau_capacity_mbps: float = 6.0  # 40% of threshold; tracks CAPACITY_THRESHOLD_MBPS
    d_ref_m: float = 1500.0  # map scale
    # The "hold" factor on Phi_observe. `w_hold = 0.0` reproduces the shipped
    # potential BITWISE, so this ships off. See `potential()` for the argument.
    # Sane range is [0, 0.6]: at 1.0 a distant-but-clear sightline is worth
    # nothing, which would discourage acquiring at all.
    w_hold: float = 0.0
    d_hold_m: float = 400.0
    #: The **per-drone** relay potential. 0.0 = off, and the reward is then
    #: byte-identical to the shipped one. See `relay_shaping()` -- this is the
    #: only per-drone term in the whole reward, and it exists because the
    #: measured deficit is that the relay role has no learning signal at all.
    w_relay: float = 0.0

    # --- Phi v2: two components added 2026-08-27, both OFF by default -------
    #
    # 📏 The measurement that motivates them, `scripts/measure_potential.py` on
    # the eval split at stage 4 under F4:
    #
    #   * along the closing axis the whole swarm turns on -- observer 250 -> 60 m
    #     with the ray clear and a chain at 25 Mbps -- the shipped `Phi` moves a
    #     total of **0.320**, i.e. **0.0133 per 8 m step** against the **0.0544**
    #     per step the objective's energy term can pay for cruising instead;
    #   * every shipped component reduces the swarm with a hard `min` / `max` /
    #     routing choice, so a drone that is not currently the nearest, the
    #     clearest or on the path contributes **exactly zero** to `Phi`. Measured
    #     consequence: learned policies spend **15-23 %** of steps pressed
    #     against the map boundary, where B0 spends 0.9 %.
    #
    # `w_standoff = w_cover = 0.0` reproduces the shipped potential BITWISE, so
    # this ships off and the golden trace is untouched.

    #: Grades the sightline by the range of the drone HOLDING it, as an additive
    #: component with its own budget. ⚠️ Not `w_hold`, which multiplied the same
    #: idea INTO `Phi_observe` and so could never be worth more than
    #: `w_observe * w_hold` -- measured at 0.74 over the whole closing band, and
    #: a null at 5 seeds. See `potential()`.
    w_standoff: float = 0.0
    #: Where `Phi_standoff` is steepest. 📏 Block B measured the along-street
    #: sightline median at **127 m**: B0's observer stands at 88.8 m, INSIDE it,
    #: and the learned observer at 184 m, outside it. The gradient belongs at the
    #: threshold, not spread evenly over the map.
    d_standoff_m: float = 127.0
    #: Width of that logistic. Max slope is `k*w_standoff / (4*tau)` per metre.
    tau_standoff_m: float = 40.0

    #: Coverage of the MCV -> HVT axis. The one component that is smooth in
    #: EVERY drone's position -- see `axis_coverage()`.
    w_cover: float = 0.0
    #: Coverage radius. 📏 **Chosen by sweep, not derived** -- and the derivation
    #: that looked obvious is the one the sweep rejected. `R`/2 = 262 m ("two
    #: drones whose discs touch are one hop apart") saturates the term at the
    #: behaviour it is supposed to reward: B0 scores p1 2.446 of a 3.0 maximum,
    #: and the gap between B0's states and a learned policy's shrinks to 0.872.
    #: Swept on real state banks (`scripts/measure_potential.py`), B0 against the
    #: learned policy, at `w_cover = 0.30`:
    #:
    #:     r_cover      120     180     262     400   m
    #:     separation  1.247   1.114   0.872   0.575
    #:     B0 p1       1.428   2.000   2.446   2.750   (of 3.0)
    #:     trip home   0.335   0.286   0.242   0.184
    #:
    #: 120 m wins on all three. It is corroborated -- not derived -- by Block B's
    #: **127 m** along-street sightline median: a drone within one sightline of
    #: the axis is where the relay chain wants it.
    r_cover_m: float = 120.0
    #: Sample points along the axis. 16 puts a sample every ~75 m on a 1.2 km
    #: axis, well inside `r_cover_m`, so the term is smooth in the drones.
    n_cover_samples: int = 16

    # --- physical references for normalisation ---
    max_accel_ms2: float = 10.0


DEFAULT_WEIGHTS = RewardWeights()

#: `Phi` v2 -- the whole rebuild as ONE flag, so the experiment has one variable.
#:
#: Two structural changes and a redistribution, all inside the potential and so
#: all optimum-preserving (Ng, Harada & Russell 1999; Devlin & Kudenko 2011 for
#: the multi-agent case). The rationale is in `potential()` and
#: `axis_coverage()`; the sizing rule is the only thing that belongs here:
#:
#: 🔒 **The component weights sum to 1.0 and `potential_scale` stays 10.**
#: `Phi` is redistributed, never inflated. 📏 The reason is measured: PBRS pays
#: `gamma*Phi(s') - Phi(s)`, so a policy HOLDING a state pays `(gamma-1)*Phi`
#: every step -- at `gamma = 0.997` and `Phi ~ 9` that is **-0.027/step**, and
#: `scripts/measure_potential.py` measures B0's mean shaping at **-0.018/step**
#: against the learned policy's -0.016. The drag is proportional to `Phi` and is
#: therefore *largest for the best policy*. Tripling `k` triples it -- which is
#: the most likely reason `potential_scale = 30` measured as a null rather than
#: as an improvement in the 81-run sweep.
PHI_V2 = RewardWeights(
    # `Phi_cover`'s muster half does `Phi_approach`'s job better -- for every
    # drone rather than for whichever one is momentarily nearest, and toward the
    # relay corridor rather than toward the HVT, which is where four of the five
    # drones actually need to be. Approach keeps a trace so the term is visibly
    # retained rather than silently deleted.
    w_approach=0.05,
    # Acquisition. Still a near-step function -- `occlusion` returns 1e4 for a
    # clear ray, so the sigmoid reads exactly 1.0 the moment any ray is clear --
    # but it is the gate everything else sits behind, so it keeps its budget.
    w_observe=0.20,
    # The closing decision. 📏 `k*w = 2.0` gives **0.077 per 8 m step** across
    # 250 -> 60 m, **1.41x** the objective's 0.0544, against the shipped
    # potential's 0.0133 (0.25x).
    w_standoff=0.20,
    # ⬇️ from 0.40, the largest weight in the shipped potential. It was spent on
    # the relay half of the mission, and 📏 the corrected diagnosis is that there
    # is no separate relay failure: `hop | observed` measures geometry (random
    # 1.83, every learned policy 1.86-1.93, B0 2.26) and chain length follows
    # from where the observer stands. `docs/DECISIONS.md`.
    w_link=0.15,
    # The largest single weight, because it is the only component that is not
    # blind to four drones out of five.
    w_cover=0.40,
)


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
    #: (B,) range of the drone that actually HOLDS that ray -- the argmax of
    #: clearance, not the argmin of distance. A drone 50 m away on the wrong
    #: side of a building is near but blind, and `nearest_dist_m` cannot tell
    #: the difference. Only read when `w_hold > 0`; optional so the reward's
    #: own tests can build a Snapshot without it.
    observer_dist_m: torch.Tensor | None = None
    #: (B, N) bool -- is drone `i` carrying the delivery path this step? The one
    #: per-drone quantity the reward uses, and only when `w_relay > 0`.
    on_path: torch.Tensor | None = None
    #: Raw geometry, `(B, N, 3)` and `(B, 3)`. Read only when `w_cover > 0`.
    #: `Phi_cover` needs where every drone is, not a reduction over them -- that
    #: is the whole point of it -- so it cannot be expressed as one of the scalar
    #: summaries above. The env has these already; passing them costs nothing.
    drone_pos: torch.Tensor | None = None
    mcv_pos: torch.Tensor | None = None
    hvt_pos: torch.Tensor | None = None

    @property
    def n_agents(self) -> int:
        return self.battery.shape[-1]


# --------------------------------------------------------------------------- #
# Potential
# --------------------------------------------------------------------------- #


def axis_coverage(
    drone_xy: torch.Tensor,
    mcv_xy: torch.Tensor,
    hvt_xy: torch.Tensor,
    *,
    r_cover_m: float,
    n_samples: int,
) -> torch.Tensor:
    """(B,) how well the swarm covers the MCV -> HVT axis, in [0, 1].

    ## Why this component exists

    📏 Every other component of `Phi` reduces the swarm with a hard `min`
    (`nearest_dist_m`), a hard `max` (`best_clearance_m`) or the router's chosen
    path (`e2e_capacity_mbps`). So at any instant **`Phi` is a function of one
    drone and is exactly constant in the other four.** A drone that is not
    currently the nearest, the clearest or on the path can fly anywhere at all
    without moving the potential by a single bit -- and those are precisely the
    drones that have to pre-position for the relay chain and the handoff.

    The measured consequence, `scripts/measure_potential.py` on the eval split at
    stage 4: learned policies sit against the map boundary on **15-23 %** of
    steps against B0's **0.9 %**, and `off_axis_m` is **252 m** against B0's
    **105 m**. Out there the shipped potential is not weak, it is *absent*.

    ## The construction

    Sample `n_samples` points evenly along the segment, and ask how covered each
    one is:

        f_im    = 1 / (1 + (d_im / r_cover)^2)      drone i's cover of sample m
        covered = mean_m [ 1 - prod_i (1 - f_im) ]  soft OR over the swarm
        mustered= mean_i [ max_m f_im ]             each drone's own proximity
        Phi     = (covered + mustered) / 2

    Three properties, each of them the reason for a choice above:

    1. **Every drone always has a gradient.** The kernel is heavy-tailed
       (Cauchy, not Gaussian) *on purpose*: at 800 m and `r_cover = 262 m` it
       still reads 0.10 and still has slope. An exponential kernel -- or the hard
       `min` the shipped terms use -- is numerically zero out there, which is the
       failure this component exists to fix. A drone stranded at the map edge
       must be told to come back, and it is the far field that has to tell it.

    2. **It cannot be satisfied by clustering, and needs no agent index to do
       it.** `d cover_m / d f_jm = prod_{i != j} (1 - f_im)`: a drone's marginal
       value at a sample point is exactly *how uncovered that point is by
       everybody else*. Two drones in the same place are worth barely more than
       one; spreading along the axis is worth much more. That is a differentiated
       role pressure out of a **team** quantity with no `i` in it, which is what
       `docs/REWARD.md`'s homogeneity constraint requires and what
       `docs/BLOCK_G.md`'s per-drone `w_relay` could not produce.

    3. **Huddling at either end scores badly.** Samples run the whole segment, so
       five drones parked on the MCV cover the near end and nothing else. That is
       the failure mode a plain "distance to the MCV-HVT line" term would have --
       sitting *on* the MCV is distance zero.

    ⚠️ **Horizontal distance only.** The drones fly at 40-80 m and both endpoints
    are at ground level, so a 3D distance would add a near-constant offset and
    would quietly reward descending. Altitude is governed by the band, not here.

    Args:
        drone_xy: `(B, N, 2)`.
        mcv_xy, hvt_xy: `(B, 2)`.
    """
    b = drone_xy.shape[0]
    # Midpoints of `n_samples` equal cells, so no sample sits exactly on an
    # endpoint: a sample AT the MCV would be permanently covered by nothing the
    # swarm does and would only dilute the mean.
    frac = (torch.arange(n_samples, device=drone_xy.device, dtype=drone_xy.dtype) + 0.5) / n_samples
    samples = mcv_xy[:, None, :] + frac[None, :, None] * (hvt_xy - mcv_xy)[:, None, :]  # (B, M, 2)
    d = torch.cdist(samples, drone_xy)  # (B, M, N)
    f = 1.0 / (1.0 + (d / r_cover_m) ** 2)

    # Half 1 -- COVERAGE. How much of the axis the swarm collectively holds.
    # Rewards spreading and refuses to be satisfied by huddling at either end.
    covered = (1.0 - (1.0 - f).prod(dim=-1)).mean(dim=-1)  # (B,)

    # Half 2 -- MUSTER. Each drone's own proximity to the axis, averaged over the
    # swarm rather than reduced with a `max`.
    #
    # ⚠️ Both halves are needed and neither is redundant, because each one is the
    # other's degenerate case:
    #
    #   * 📏 Coverage alone gives a REDUNDANT drone no gradient, and that is
    #     correct arithmetic rather than a bug -- `d covered / d f_j` is
    #     `prod_{i != j} (1 - f_i)`, which is ~0 once the axis is held. Measured:
    #     with four drones on the axis the fifth's recall gradient is
    #     **+0.0003/step at 200 m off-axis**, i.e. still nothing. But the drones
    #     that end up at the map boundary are exactly the redundant ones.
    #   * Muster alone is maximised by every drone sitting ON the MCV, which is
    #     distance zero from the segment and covers none of it.
    #
    # Fixed 50/50 rather than a knob: they are two halves of one statement --
    # *be on the axis, and be spread along it* -- and a weight between them would
    # be a knob with no measurement behind it.
    mustered = f.max(dim=1).values.mean(dim=-1)  # (B,), nearest sample per drone
    out = 0.5 * (covered + mustered)
    assert out.shape == (b,)
    return out


def potential(snap: Snapshot, w: RewardWeights) -> torch.Tensor:
    """(B,) team potential. Five components with a deliberate handover; two of
    them ship off, and with `w_standoff = w_cover = 0` this is bitwise the
    potential every number before 2026-08-27 was measured under.

    Every component is a TEAM quantity. Per-drone potentials would pull every
    drone onto the HVT and leave nobody relaying -- which `Phi_cover` respects
    without going blind to the individual, because its marginal value to a drone
    is *how uncovered a point is by everybody else*.

    Summed, not multiplied: a product is flat at episode start, when the drones
    are parked on the MCV and both observation and link are ~0, so neither can
    improve without the other.

    📏 What the three shipped components are actually worth, measured on real
    states by `scripts/measure_potential.py` -- and this is the audit the v2
    components exist because of:

        closing the observer 250 -> 60 m, ray clear, chain at 25 Mbps
            shipped     +0.320 total,  0.0133 per 8 m step   (0.25x the 0.0544
                                                              the energy term
                                                              pays for cruising)
            v2          +1.717 total,  0.0774 per 8 m step   (1.42x)

        moving a drone that holds no role 8 m back toward the MCV-HVT axis
            shipped      0.0000 at 200 m off-axis, and at 500 m, and at 800 m
            v2           0.0078 / 0.0010 / 0.0003, and +0.34..+0.47 for the trip
    """
    # Coarse: non-zero anywhere on the map, so the agent is never blind.
    approach = 1.0 - (snap.nearest_dist_m / w.d_ref_m).clamp(0.0, 1.0)

    # Fine: rewards correct GEOMETRY, not mere proximity. The observation
    # envelope is a wedge down the street plus an overhead cone, so a drone
    # 20 m away across the street sees nothing while one 300 m down it sees
    # fine -- clearance captures that where distance cannot.
    observe = torch.sigmoid(snap.best_clearance_m / w.tau_clearance_m)

    # ⚠️ The reason this factor exists, and it is not "more pull toward the
    # target" -- that hypothesis was tested and died.
    #
    # `occlusion` returns 1e4 for "nothing in the way", so `best_clearance_m` is
    # 1e4 the moment ANY ray is clear and the sigmoid above reads exactly 1.0.
    # Meanwhile `mission` is 1.0, `idle` is 0.0, and `Phi_link` is 0.999 because
    # a formed chain carries ~4x the 15 Mbps bar. **While the swarm is
    # succeeding, every term in the reward is flat**, so nothing distinguishes an
    # action that will hold the sightline from one that will drift out of it. The
    # policy only hears about the drift ~30 steps later, through a GAE window
    # whose effective horizon at lambda = 0.95 is ~20 steps.
    #
    # That is a ZERO gradient, not a weak one, which is why scaling the existing
    # terms could not fix it: the 81-run sweep moved `d_ref_m` 1500 -> 400
    # (3.8x the closing gradient) and `potential_scale` 10 -> 30, and both were
    # nulls. You cannot fix a zero by multiplying it.
    #
    # `hold` grades the sightline by the range of the drone holding it. At a
    # 40-80 m ceiling that is a cheap monotone stand-in for elevation angle:
    # B0 parks its observer at 79 m (~37 deg, a short near-vertical ray that
    # survives the HVT moving down a street) where the learned policies loiter at
    # 291 m (~12 deg, a long canyon ray one building corner kills).
    #
    # It is a TEAM quantity -- one observer, the best one -- so once somebody is
    # parked the pull stops for everyone else, exactly as `d_min` and
    # `clearance_best` already do. Per-drone would cluster the swarm on the HVT
    # and leave nobody relaying.
    #
    # ✅ It lives in the potential, so PBRS proves it cannot move the optimum
    # (Ng, Harada & Russell 1999). The worst it can do is slow learning down.
    #
    # ⛔ **Superseded by `w_standoff` below, and a null at 5 seeds.** Kept because
    # `DECISIONS.md` records the null and the flag is how it gets re-run -- but
    # the diagnosis above is right about the deficit and wrong about the fix: a
    # FACTOR on `observe` is capped at `w_observe * w_hold` however it is tuned,
    # which is 0.03/step where the objective pays 0.054. Do not reach for this
    # one; reach for `w_standoff`, which is the same variable with its own budget.
    if w.w_hold > 0.0:
        if snap.observer_dist_m is None:
            raise ValueError(
                "w_hold > 0 needs Snapshot.observer_dist_m; the env supplies it, "
                "a hand-built Snapshot must too"
            )
        hold = 1.0 - (snap.observer_dist_m / w.d_hold_m).clamp(0.0, 1.0)
        observe = observe * (1.0 - w.w_hold + w.w_hold * hold)

    # Gradient below threshold, where the binary link indicator has none.
    link = torch.sigmoid((snap.e2e_capacity_mbps - CAPACITY_THRESHOLD_MBPS) / w.tau_capacity_mbps)

    total = w.w_approach * approach + w.w_observe * observe + w.w_link * link

    # --- Phi v2 ------------------------------------------------------------
    #
    # Both off by default, and the branch is what keeps the shipped potential
    # bitwise identical rather than merely equal-to-float-error.

    # `Phi_standoff` -- the closing decision, graded where the decision is made.
    #
    # ⚠️ Read the difference from `w_hold` before treating this as a retune of
    # it. `w_hold` MULTIPLIED this idea into `observe`, so (a) its whole budget
    # was `w_observe * w_hold` -- 0.74 across the entire 291 -> 79 m band, or
    # 0.03 per 8 m step against the objective's 0.054 -- and (b) it worked by
    # *removing* potential from a distant sightline rather than adding it to a
    # close one, which is why `w_hold = 1.0` discourages acquiring at all. Here
    # it is an ADDITIVE component with its own weight, so its per-step gradient
    # is set directly, and `Phi_observe` still pays in full for acquiring at any
    # range.
    #
    # The shape is a logistic centred on `d_standoff_m`, not a ramp to a far
    # reference, because 📏 the deficit is a THRESHOLD: Block B measured the
    # along-street sightline median at 127 m, B0's observer stands at 88.8 m
    # inside it and the learned observer at 184 m outside it. A ramp spends its
    # budget uniformly over hundreds of metres; a logistic spends it at the
    # threshold, where the true value function has its step.
    #
    # Gated on `observed` so it cannot pay for closing while BLIND -- which is
    # `docs/REWARD.md`'s first trap, a drone 20 m away on the wrong side of a
    # building seeing nothing while one 300 m down the street sees fine.
    #
    # ⚠️ The gate is the BOOLEAN, not the graded `observe` factor, and a test
    # forced the distinction: `occlusion` returns a signed margin, so a ray
    # blocked by 20 m of building still reads `sigmoid(-20/15) = 0.21` and would
    # leak a fifth of the closing pull to a blind drone. The component grades the
    # sightline the swarm is ACTUALLY holding, so it is off when there is none.
    if w.w_standoff > 0.0:
        if snap.observer_dist_m is None:
            raise ValueError(
                "w_standoff > 0 needs Snapshot.observer_dist_m; the env supplies it, "
                "a hand-built Snapshot must too"
            )
        standoff = torch.sigmoid((w.d_standoff_m - snap.observer_dist_m) / w.tau_standoff_m)
        total = total + w.w_standoff * snap.observed.to(standoff.dtype) * standoff

    # `Phi_cover` -- the only component that is not blind to four drones out of
    # five. See `axis_coverage()`.
    if w.w_cover > 0.0:
        missing = [
            name for name in ("drone_pos", "mcv_pos", "hvt_pos") if getattr(snap, name) is None
        ]
        if missing:
            raise ValueError(
                f"w_cover > 0 needs Snapshot.{'/'.join(missing)}; the env supplies "
                "them, a hand-built Snapshot must too"
            )
        total = total + w.w_cover * axis_coverage(
            snap.drone_pos[..., :2],
            snap.mcv_pos[..., :2],
            snap.hvt_pos[..., :2],
            r_cover_m=w.r_cover_m,
            n_samples=w.n_cover_samples,
        )

    return w.potential_scale * total


def relay_shaping(
    snap: Snapshot,
    next_snap: Snapshot,
    w: RewardWeights,
    gamma: float,
    next_is_terminal: torch.Tensor | None = None,
) -> torch.Tensor:
    """(B, N) per-drone shaping: `gamma*Phi_i(s') - Phi_i(s)`, `Phi_i = k*w_relay*on_path_i`.

    ## The measured deficit this exists for

    `scripts/probe_credit.py`: the critic is handed one global state repeated per
    drone, so `max |V_i - V_j| = 0.000e+00`, and the reward is team-dominated --
    which leaves **0.015-0.06 %** of advantage variance distinguishing one drone
    from another. Every drone's policy gradient is
    `grad log pi(a_i | o_i) * A` with the *same* `A`: each is told *the team did
    well*, never *your action was the good one*. Role differentiation cannot be
    learned from a signal that is constant across the agents it would
    differentiate.

    ## ⚠️ Why ONLY the relay term goes per-drone

    `docs/REWARD.md` warns that per-drone potentials cluster the swarm: pay every
    drone for its own proximity or its own sightline and all `N` fly at the HVT
    and nobody relays. **That warning is correct and this respects it** --
    `Phi_approach` and `Phi_observe` stay TEAM quantities, so once one drone has
    the target the pull stops for everyone else, exactly as before.

    `on_path` is the opposite kind of quantity. A drone can only be on the
    delivery path by sitting *between* the source and the MCV, so paying for it
    rewards **spreading out**, not clustering. And it is the role with no signal
    at all today: measured on the eval split, conditioned on observing, every
    learned policy's chain is indistinguishable from a random policy's
    (1.86-1.91 hops against random's 1.83, B0's 2.26).

    ## Sizing, which is NOT free even though the optimum is

    PBRS makes any `Phi` optimum-preserving -- in the multi-agent case too
    (Devlin & Kudenko, 2011), so this cannot move the equilibrium. **But
    optimum-preserving is not signal-preserving.** `probe_credit.py` measured a
    condition where 50 % of *reward* variance was per-drone and the advantage
    still washed it out to 0.06 %: GAE accumulates the team component coherently
    over ~19 effective steps (`lambda = 0.95`) while per-drone terms largely
    cancel. A per-drone term has to clear that filter before it reaches the
    gradient.

    The payment is concentrated at transitions, which is the intent: joining the
    path pays `~k*w_relay` once, holding it pays only the `(gamma - 1)*Phi` decay,
    and joining-then-leaving nets zero. So the incentive is *take a relay slot and
    keep it* -- precisely the commitment the swarm does not currently make.
    At `k = 10`, `w_relay = 0.2` makes joining the chain worth two steps of full
    mission capability. **0.2-0.5 is the range to test; 0.0 ships.**

    ⚠️ The acceptance test is `probe_credit.py`'s **advantage** column, not its
    value column. If the between-drone share does not rise, the term is not
    reaching the gradient and its weight is too small -- the one honest reason to
    raise it.
    """
    if w.w_relay <= 0.0:
        return torch.zeros_like(snap.battery)
    if snap.on_path is None or next_snap.on_path is None:
        raise ValueError(
            "w_relay > 0 needs Snapshot.on_path; the env supplies it, "
            "a hand-built Snapshot must too"
        )
    scale = w.potential_scale * w.w_relay
    phi = scale * snap.on_path.to(snap.battery.dtype)
    phi_next = scale * next_snap.on_path.to(snap.battery.dtype)
    if next_is_terminal is not None:
        # Same rule as the team potential: Phi(terminal) = 0, or `gamma^T Phi`
        # survives the telescoping and reintroduces a policy-dependent bias.
        phi_next = torch.where(next_is_terminal.unsqueeze(-1), torch.zeros_like(phi_next), phi_next)
    return gamma * phi_next - phi


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
    return (
        team.unsqueeze(-1)
        + individual_reward(snap, w, craft)
        + relay_shaping(snap, next_snap, w, gamma, next_is_terminal)
    )


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


def reward_terms(
    snap: Snapshot,
    next_snap: Snapshot,
    w: RewardWeights | None = None,
    gamma: float = 0.999,
    next_is_terminal: torch.Tensor | None = None,
    craft: Rotorcraft = DEFAULT_AIRFRAME,
) -> dict[str, torch.Tensor]:
    """The same reward, decomposed. `(B, N)` per term; they sum to `reward(...)`.

    Diagnostic only -- nothing consumes this to compute anything. It exists
    because `docs/REWARD.md` requires every term logged separately: the total is
    nearly useless for diagnosis, and one term carrying 95 % of the magnitude is
    the signature of a scaling error that is invisible in the aggregate. Block G
    is where a flat return curve has to be attributed to a term, so this is the
    instrumentation that has to exist before any tuning starts.

    Broadcast to `(B, N)` rather than left as `(B,)` for the team terms, so a
    caller can stack them and check the sum against `reward` element-wise --
    which `test_reward.py` does, and which is what stops this drifting away from
    the function it claims to decompose.
    """
    w = w or DEFAULT_WEIGHTS
    n = snap.n_agents
    capable = mission_capable(snap).to(snap.e2e_capacity_mbps.dtype)
    idle = (~snap.observed).to(capable.dtype)
    var_b = snap.battery.var(dim=-1, unbiased=False)
    power = total_power_w(snap.speed_ms, snap.accel_ms2, craft)

    def team(x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1).expand(-1, n)

    return {
        "mission": team(w.mission * capable),
        "idle": team(-w.idle * idle),
        "battery_variance": team(-w.battery_variance * var_b),
        "shaping": team(shaping(snap, next_snap, w, gamma, next_is_terminal)),
        "energy": -w.energy * power / hover_reference_power_w(craft),
        "effort": -w.effort * (snap.accel_ms2 / w.max_accel_ms2) ** 2,
        "relay": relay_shaping(snap, next_snap, w, gamma, next_is_terminal),
    }
