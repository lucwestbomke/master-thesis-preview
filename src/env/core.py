"""Batched env core -- Block D.

The training path. Carries a leading `num_envs` dimension and never leaves the
GPU: no `.item()`, no `.cpu()`, no Python loop over environments. `swarm_env.py`
is the thin PettingZoo adapter over this, for API-compliance tests and visual
debugging only (`AGENTS.md`, `docs/DECISIONS.md`).

Node layout -- two sets, deliberately different sizes. Confusing them feeds the
HVT into the routing DP as a relay:

    geometric  K = N + 2     0..N-1 drones,  N = MCV,  N+1 = HVT
    radio      R = N + 1     0..N-1 drones,  N = MCV

Occlusion runs over all K, so the drone-HVT rays serve the sensor *and* the
jammer path from one call. The HVT is never a radio node: it is the target and
the emitter, not a relay.

Two conventions that are easy to get backwards and are pinned by tests:

**Reward timing.** `reward.episode_return` pairs consecutive snapshots as
`reward(s_t, s_t+1)`, so the objective terms are evaluated at the state the
transition *starts* from and only the shaping looks forward. `step()` therefore
keeps the pre-transition snapshot in `self.snap` and calls `reward(self.snap,
new_snap)`. Energy is charged on the same one-step lag, which is why the battery
drain and the reward's energy term are consistent in aggregate over an episode
rather than step by step.

**The snapshot after an auto-reset.** `gamma*Phi(s') - Phi(s)` needs `Phi` of the
*fresh* state on the first step of every new episode. Zeroing it instead would
leave `gamma*Phi(s_1)` in the telescoped return, which is policy-dependent and
so breaks exactly the invariance PBRS is chosen for. `step()` therefore
evaluates the physics a second time after resetting. That second pass also
produces the observation that auto-reset must return, so it is the price of the
API rather than of the reward -- and at ~3170x margin on the throughput gate
(`docs/BLOCK_C.md`) the 2x is affordable.

Design decisions and the measurements behind them: `docs/BLOCK_D.md`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from . import channel, occlusion, routing
from .energy import DEFAULT_AIRFRAME, Rotorcraft, climb_power_w, total_power_w
from .reward import (
    CAPACITY_THRESHOLD_MBPS,
    DEFAULT_WEIGHTS,
    RewardWeights,
    Snapshot,
    mission_capable,
    reward,
    reward_terms,
)

ARTEFACT = Path(__file__).resolve().parents[2] / "data" / "frankfurt_box.npz"

# --- scenario, from AGENTS.md "Settled parameters". Do not re-derive here. ---
BOX_HALF_M = 750.0
ALT_MIN_M = 40.0  # model-validity floor, not a flight rule -- see BLOCK_D.md
# The ceiling is DERIVED, not chosen: it is the altitude above which a single
# best-placed drone can do the mission on its own, which would dissolve the
# scenario. 3.3 % solo success at 80 m vs 57.4 % at 120 m. See docs/BLOCK_D.md.
ALT_MAX_M = 80.0
HVT_Z_M = 1.5
MCV_Z_M = 2.0
DT_S = 0.4
EPISODE_STEPS = 600
DRONE_DASH_MS = 25.0
# Cruise is a scenario parameter (AGENTS.md), not an env constraint: `step()`
# enforces only the dash ceiling. It lives here so policies and the sizing
# scripts read one number rather than three copies of it.
DRONE_CRUISE_MS = 20.0
MAX_ACCEL_MS2 = 10.0
SENSOR_RANGE_M = 830.0  # non-binding ceiling; 99.8 % of sightlines are shorter
SPAWN_RING_M = 5.0
BATTERY_WH = 548.0

# The discount, defined ONCE. It is not just a learner hyperparameter here: the
# reward's PBRS term is `gamma*Phi(s') - Phi(s)`, and the invariance proof holds
# only if that gamma equals the one the agent actually discounts with. Env and
# learner reading different values silently turns provably-neutral shaping into a
# bias. `src/training/skrl_wrapper.py` imports this, and a test asserts they agree.
#
# 0.997 rather than 0.999: horizon 1/(1-g) = 333 steps covers 55 % of a 600-step
# episode, which reaches the escalation onset, while more than halving the value
# scale the critic has to fit. AGENTS.md pins the band 0.997-0.999, so this is a
# choice inside the range, not a change to it. 0.99 (skrl's default, horizon 100
# steps) is blind to the hard end of the episode -- see docs/BLOCK_D.md.
GAMMA = 0.997

PTX_DBM = 30.0
JAMMER_DBM = 30.0
BANDWIDTH_HZ = 10e6
NOISE_FIGURE_DB = 7.0

# Route bank split. Held out so a generalisation check exists later at no cost;
# the bank was sampled i.i.d., so a contiguous split is valid.
N_EVAL_ROUTES = 256

# --- observation shapes. Fixed here because E, F and G all build against them.
EGO_DIM = 24  # 21 from ENVIRONMENT.md + 3 for the persistent cue vector
NEIGHBOUR_DIM = 9
EDGE_DIM = 2
ACTION_DIM = 3  # dv_x, dv_y, dv_z. Ptx is NOT an action -- NEGATIVE_RESULTS.md
N_MAX = 8  # max-N padding, so the MLP rung can be evaluated off-N at all
FLAT_DIM = EGO_DIM + (N_MAX - 1) * (NEIGHBOUR_DIM + EDGE_DIM + 1)  # 24 + 77 = 108

# Feature scaling. Networks see roughly unit-scale inputs; these are the divisors
# and they are part of the observation contract, not free knobs.
POS_SCALE_M = BOX_HALF_M
VEL_SCALE_MS = DRONE_DASH_MS
CLEARANCE_CLAMP_M = 150.0  # occlusion returns 1e4 for "nothing in the way"
# Capacity features are stored in THRESHOLD units and clamped here, so this sets
# how much of the physical range the network can actually see. 5.0 x 15 Mbps =
# 75 Mbps, just above `capacity_mbps`'s own ceiling of 7.4 b/s/Hz x 10 MHz =
# 74 Mbps -- so the observation now saturates only where the PHYSICS saturates,
# never before it.
#
# It used to be 4.0, which at the old 5 Mbps threshold clamped at 20 Mbps while
# real drone-drone links run to 74. Measured under B0: **57.5 % of link-capacity
# values sat pinned at the clamp**, against 6.3 % genuinely at zero -- so barely
# a third of the feature carried any information. That matters more than it
# sounds: `edge` is the ONLY input the GNN rung has and the DeepSets rung does
# not (docs/MODELS.md), so a pinned capacity feature quietly handicaps the one
# comparison RQ2 exists to make. At 5.0 the informative share is 93.7 %.
CAPACITY_CLAMP = 5.0
NOISE_REF_DBM = -97.0  # thermal floor at 10 MHz / 7 dB NF
NOISE_SCALE_DB = 30.0
LINK_TIMEOUT_SCALE = 50.0
SOFT_SEE_TAU_M = 15.0  # matches RewardWeights.tau_clearance_m


def unpack_flat(flat: Tensor) -> dict[str, Tensor]:
    """Inverse of `BatchedSwarmEnv._pack`. `(B, N, 108)` -> the structured views.

    Lives here, next to the packing it inverts, because everything that consumes
    `flat` has to agree on the layout: the B0 baseline (`src/baselines/b0.py`)
    and all three of Block G's architectures, which `docs/MODELS.md` requires to
    consume `flat` and unpack it so the max-N padding is identical across rungs
    *by construction* rather than by discipline. A second, hand-rolled unpacking
    somewhere else is how that guarantee is lost.

    Returns `ego (B,N,24)`, `neighbour (B,N,7,9)`, `edge (B,N,7,2)` and
    `valid (B,N,7)` -- the padding mask, 1.0 for real neighbours. The structured
    keys the env returns alongside `flat` are the *unpadded* `(B,N,N-1,·)`
    versions; these are always at `N_MAX - 1 = 7`.
    """
    k = N_MAX - 1
    a = EGO_DIM
    b = a + k * NEIGHBOUR_DIM
    c = b + k * EDGE_DIM
    return {
        "ego": flat[..., :a],
        "neighbour": flat[..., a:b].unflatten(-1, (k, NEIGHBOUR_DIM)),
        "edge": flat[..., b:c].unflatten(-1, (k, EDGE_DIM)),
        "valid": flat[..., c:],
    }


def neighbour_index_table(num_drones: int, device: torch.device | str = "cpu") -> Tensor:
    """`(N, N-1)` global index of each neighbour slot, matching `_pack` order.

    Slot `k` of drone `i` holds drone `k` if `k < i`, else `k + 1`. Part of the
    observation contract (`docs/BLOCK_D.md`: "neighbour ordering is a
    precomputed index table, fixed for the run"), so a policy may reconstruct it
    from `N` alone -- it is not env state.
    """
    eye = torch.eye(num_drones, dtype=torch.bool, device=device)
    return (
        torch.arange(num_drones, device=device)
        .repeat(num_drones, 1)[~eye]
        .view(num_drones, max(num_drones - 1, 0))
    )


@dataclass(frozen=True)
class CurriculumStage:
    """One rung of `docs/ENVIRONMENT.md`'s curriculum.

    `speed_scale` multiplies the index into the pre-baked route, so it is a
    scale on the *realised* speed of the bank (median 5.8 m/s after
    CONGESTION_FACTOR), not an OSM free-flow class speed.
    """

    episode_steps: int
    speed_scale: float
    jammer: float
    battery_scale: float
    cue_sigma_m: float
    charge_min: float  # initial charge drawn from [charge_min, 1]


STAGES: tuple[CurriculumStage, ...] = (
    CurriculumStage(150, 0.00, 0.0, 3.0, 0.0, 1.0),
    CurriculumStage(300, 0.50, 0.0, 2.0, 0.0, 1.0),
    CurriculumStage(450, 0.75, 1.0, 1.5, 150.0, 1.0),
    CurriculumStage(600, 1.00, 1.0, 1.0, 150.0, 0.3),
)


# --------------------------------------------------------------------------- #
# The fidelity ladder -- RQ1's independent variable. docs/BLOCK_F.md.
# --------------------------------------------------------------------------- #

Fidelity = Literal["F0", "F1", "F2", "F3", "F4"]


@dataclass(frozen=True)
class FidelityRung:
    """What one rung of THESIS_PLAN §2's ladder changes about the CHANNEL.

    Nothing here touches the sensor, the diagnostics or the curriculum. That is
    the whole content of `docs/BLOCK_F.md` decisions 1, 2 and 4, and it is what
    keeps RQ1's attribution interpretable: the F0->F1 gap is the cost of
    ignoring buildings *in the radio*, not the cost of deleting the city.
    """

    channel_occlusion: bool  # does a blocked ray cost the LINK anything?
    binary_capacity: bool  # C_max-or-nothing, vs path loss -> SINR -> Shannon
    channel_jammer: bool  # is the emitter in the SINR denominator?
    reuse_limit: int  # 1 = no rate division; 3 = min(n, 3), the full model


LADDER: dict[str, FidelityRung] = {
    # rung  occlusion  binary  jammer  reuse
    "F0": FidelityRung(False, True, False, 1),  # the connectivity-radius abstraction
    "F1": FidelityRung(True, True, False, 1),  # + buildings
    "F2": FidelityRung(True, False, False, 1),  # + continuous capacity
    "F3": FidelityRung(True, False, True, 1),  # + the threat
    "F4": FidelityRung(True, False, True, 3),  # + relay cost == today's env
}

# `C_max` for the binary rungs (F0, F1). NOT a new free parameter: it is
# `channel.capacity_mbps`'s own ceiling, which is where a "connected" link lands
# when nothing degrades it. docs/BLOCK_F.md decision 3.
#
# The consequence is the point of the abstraction under test: 74 Mbps against a
# 15 Mbps requirement means a chain that exists geometrically always delivers,
# so under F0 `mission_capable` reduces to "someone sees the HVT and a chain of
# at most `max_hops` links exists". F0 is meant to be permissive.
F0_CAPACITY_MBPS = channel.DEFAULT_SE_CAP_BPS_HZ * BANDWIDTH_HZ / 1e6

# The connectivity radius `R`. **Measured, not chosen** -- THESIS_PLAN §2 makes
# it the fairness requirement on RQ1 ("an arbitrary `R` makes the comparison
# meaningless, and it is the first thing an examiner will probe"), and the
# pre-registered method is the median range of a link usable under F4 in the
# same city.
#
# **Measured: 524 m** [IQR 22] -- `scripts/calibrate_r.py`, 8 seeds x 64 eval
# episodes under B0. It is the distance at which a link's probability of
# carrying the 15 Mbps requirement under F4 crosses 0.5, which is the "median
# link RANGE" reading of the pre-registration: a range is a reach, and half of
# link geometries reach further than this.
#
# Cross-checked by degree matching (choose R so F0 and F4 have the same mean
# usable links per node): **418 m**, which is 0.80x and falls inside the +-25 %
# sensitivity band the same script sweeps. Replicated on a second device and
# sample size at 536 m [39]. The competing "median realised link LENGTH" reading
# gives 266 m and is rejected in docs/BLOCK_F.md -- it measures B0's spacing
# rather than the channel's reach, and it makes F0 *stricter* than F4, inverting
# the abstraction under test.
#
# Nothing in RQ1 turns on the exact value: B0's mission success under F0 is flat
# at 93.4 % from 0.75x to 1.5x of it. Full tables in docs/BLOCK_F.md.
F0_RADIUS_M = 524.0


@dataclass(frozen=True)
class EnvConfig:
    num_envs: int
    num_drones: int = 5
    device: str = "cpu"
    seed: int = 0
    dt_s: float = DT_S
    occlusion_chunk: int = 512
    gamma: float = GAMMA
    eval_routes: bool = False
    # Training wants auto-reset; the PettingZoo adapter must NOT have it, because
    # that API ends the episode and waits for an explicit reset(). Turning it off
    # also skips the second physics pass, so a manual-reset loop costs half.
    auto_reset: bool = True
    # Compile the occlusion kernel, which is ~99.7 % of the step's eager cost.
    # The whole `step()` is deliberately NOT compiled -- see `_clearance`.
    compile_occlusion: bool = True
    # Which curriculum stages episodes are drawn from. Default is the design
    # condition only; Block G's callback reweights this during training, using
    # a schedule that must be IDENTICAL across fidelity levels or RQ1 is
    # confounded (docs/ENVIRONMENT.md).
    stage_weights: tuple[float, ...] = (0.0, 0.0, 0.0, 1.0)

    # --- RQ1's independent variable. One composed enum, never loose flags. --- #
    #
    # `F4` is the default because `F4` IS the environment Blocks D and E
    # measured; `src/env/test_golden.py` proves it against a frozen trace.
    #
    # The individual flags are DERIVED (below) and deliberately not settable.
    # `channel_occlusion=False, channel_jammer=True` is not a rung on the
    # ladder, nothing would stop it running, and the number it produced would go
    # into a table -- docs/BLOCK_F.md decision 5.
    fidelity: Fidelity = "F4"
    # `R` for the binary rungs. Measured, not chosen: scripts/calibrate_r.py,
    # and `test_fidelity.py` pins this default against what that script reports.
    #
    # Settable because the sensitivity sweep docs/BLOCK_F.md requires (+-25 %,
    # +-50 %) has to vary it. INERT at F2-F4, which model capacity continuously
    # and have no radius at all -- a sweep over `radius_m` at those rungs
    # produces identical numbers, which is the correct behaviour rather than a
    # silent one.
    radius_m: float = F0_RADIUS_M

    # ⚠️ NOT a fidelity flag, and not on the ladder. This removes buildings from
    # the WORLD -- sensor, jammer line of sight and diagnostics included -- which
    # is a *city with no buildings*, not a channel abstraction. Two legitimate
    # uses and no others:
    #
    #   1. tests that are not about geometry (occlusion is ~37x the rest of the
    #      step on CPU, so the suite would take minutes without it);
    #   2. the `F0-nogeo` sensitivity rung docs/BLOCK_F.md decision 1 records as
    #      the defensible alternative reading -- a reviewer may argue that papers
    #      using a radius channel model no buildings at all. Constructed as
    #      `fidelity="F0", no_buildings=True`, reported under that name, and
    #      NEVER folded into F0: it confounds the primary result.
    no_buildings: bool = False

    # PHYSICS.md asks for the main result under more than one duplexing
    # assumption; `routing.py` exposes `reuse_limit` for exactly that. It is a
    # robustness check on the FULL model, so it is refused anywhere else -- at
    # F0-F3 the rung already pins the divisor and an override would silently
    # produce an off-ladder condition.
    duplexing_override: int | None = None

    # ⚠️ Widens the `extras` contract, so it is OFF by default and the default
    # build is byte-identical to the one `test_golden.py` pins. Block G turns it
    # on for training runs; nothing else should. Two additions, both diagnostic:
    #
    #   `final_state`  -- the critic's global state BEFORE auto-reset. The
    #       companion to `final_observation`, and it is not optional for
    #       correctness: skrl bootstraps truncation as `gamma * V(next_obs,
    #       next_state)`, and with auto_reset the post-step return is a FRESH
    #       episode's opening. Without this key the learner would bootstrap the
    #       value of an unrelated state at every truncation -- silently, and at
    #       gamma = 0.997 on returns of order 300.
    #
    #   `reward/<term>` -- the six reward terms, separately. docs/REWARD.md:
    #       "Log every term separately. The total is nearly useless for
    #       diagnosis; one term contributing 95 % of the magnitude is the
    #       signature of a scaling error and is invisible in the aggregate."
    #
    # Neither feeds anything the env computes; both cost one extra pass over
    # already-computed tensors, which is noise against occlusion.
    training_extras: bool = False

    def __post_init__(self) -> None:
        if self.fidelity not in LADDER:
            raise ValueError(f"fidelity must be one of {sorted(LADDER)}, got {self.fidelity!r}")
        if self.duplexing_override is not None:
            if self.fidelity != "F4":
                raise ValueError(
                    "duplexing_override is a robustness check on the full model and is "
                    f"only valid at fidelity='F4', not {self.fidelity!r}. At F0-F3 the "
                    "rung pins the divisor; overriding it produces a condition that is "
                    "not on the ladder (docs/BLOCK_F.md decision 5)."
                )
            if self.duplexing_override < 1:
                raise ValueError(f"duplexing_override must be >= 1, got {self.duplexing_override}")
        if self.radius_m <= 0.0:
            raise ValueError(f"radius_m must be positive, got {self.radius_m}")

    @property
    def rung(self) -> FidelityRung:
        return LADDER[self.fidelity]

    @property
    def channel_occlusion(self) -> bool:
        """Does a blocked ray cost the LINK anything? The sensor and every
        diagnostic use true geometry regardless -- decisions 1 and 2."""
        return self.rung.channel_occlusion

    @property
    def binary_capacity(self) -> bool:
        return self.rung.binary_capacity

    @property
    def channel_jammer(self) -> bool:
        """F3's switch. Multiplied in ALONGSIDE the curriculum's `jammer_on`,
        never instead of it: the ramp must run identically in every fidelity
        condition or RQ1's jammer rung is confounded with the curriculum and the
        two cannot be separated afterwards (decision 4)."""
        return self.rung.channel_jammer

    @property
    def reuse_limit(self) -> int:
        return self.rung.reuse_limit if self.duplexing_override is None else self.duplexing_override

    @property
    def n_geometric(self) -> int:
        return self.num_drones + 2

    @property
    def n_radio(self) -> int:
        return self.num_drones + 1

    @property
    def state_dim(self) -> int:
        """Width of the critic's global state; see `_critic_state`.

        Derived here rather than duplicated in the skrl wrapper, and pinned by a
        test against the real tensor so the two cannot drift apart.
        """
        n = self.num_drones
        return 9 * n + 9


class BatchedSwarmEnv:
    """`num_envs` copies of the relay-tracking mission, stepped in lockstep."""

    def __init__(
        self,
        cfg: EnvConfig,
        artefact: Path = ARTEFACT,
        weights: RewardWeights = DEFAULT_WEIGHTS,
    ):
        if not 1 <= cfg.num_drones <= N_MAX:
            raise ValueError(f"num_drones must be in 1..{N_MAX}, got {cfg.num_drones}")
        self.cfg = cfg
        self.weights = weights
        dev = torch.device(cfg.device)
        self.device = dev
        self.craft: Rotorcraft = DEFAULT_AIRFRAME
        self.gen = torch.Generator(device=dev).manual_seed(cfg.seed)

        art = np.load(artefact)
        self.boxes = torch.from_numpy(art["building_boxes"]).float().to(dev)
        self.heights = torch.from_numpy(art["building_heights"]).float().to(dev)
        self.route_xy = torch.from_numpy(art["route_xy"]).float().to(dev)
        self.route_mcv = torch.from_numpy(art["route_mcv"]).float().to(dev)

        b, n, r = cfg.num_envs, cfg.num_drones, cfg.n_radio
        self.mcv_idx, self.hvt_idx = n, n + 1

        # Link class is pure index arithmetic: drone-drone is A2A, anything
        # touching the MCV is A2G. Static, so build it once.
        drone = torch.arange(r, device=dev) < n
        self.is_a2a = drone.unsqueeze(0) & drone.unsqueeze(1)  # (R, R)
        self.no_self = 1.0 - torch.eye(r, device=dev)

        # nb_idx[i] lists the other drones, in fixed order. The whole neighbour
        # block is one gather against it.
        eye = torch.eye(n, dtype=torch.bool, device=dev)
        self.nb_idx = torch.arange(n, device=dev).repeat(n, 1)[~eye].view(n, max(n - 1, 0))
        self.self_idx = torch.arange(n, device=dev).unsqueeze(1)

        # Compiling occlusion alone captures essentially the whole speedup
        # (99.7 % of eager cost, docs/BLOCK_D.md) and is the form already proven
        # to fuse on CPU, MPS and CUDA by scripts/bench_occlusion.py.
        self._pairwise = (
            torch.compile(occlusion.pairwise_clearance, dynamic=False)
            if cfg.compile_occlusion
            else occlusion.pairwise_clearance
        )

        self.n0_dbm = channel.noise_floor_dbm(BANDWIDTH_HZ, NOISE_FIGURE_DB)
        self.noise_mw = channel.dbm_to_mw(torch.tensor(self.n0_dbm, device=dev))
        self.ptx = torch.full((b, r), PTX_DBM, device=dev)

        stage_rows = [
            [s.episode_steps, s.speed_scale, s.jammer, s.battery_scale, s.cue_sigma_m, s.charge_min]
            for s in STAGES
        ]
        self.stage_table = torch.tensor(stage_rows, device=dev, dtype=torch.float32)
        w = torch.tensor(cfg.stage_weights, device=dev, dtype=torch.float32)
        self.stage_cdf = (w / w.sum()).cumsum(0)

        # Position limits, as device tensors built once. See `_advance_drones`.
        self.pos_lo = torch.tensor([-BOX_HALF_M, -BOX_HALF_M, ALT_MIN_M], device=dev)
        self.pos_hi = torch.tensor([BOX_HALF_M, BOX_HALF_M, ALT_MAX_M], device=dev)

        self.drone_pos = torch.zeros(b, n, 3, device=dev)
        self.drone_vel = torch.zeros(b, n, 3, device=dev)
        self.last_accel = torch.zeros(b, n, device=dev)
        self.battery = torch.ones(b, n, device=dev)
        self.mcv_pos = torch.zeros(b, 3, device=dev)
        self.hvt_pos = torch.zeros(b, 3, device=dev)
        self.hvt_vel = torch.zeros(b, 3, device=dev)
        self.cue = torch.zeros(b, 3, device=dev)
        self.route_id = torch.zeros(b, dtype=torch.long, device=dev)
        self.t = torch.zeros(b, dtype=torch.long, device=dev)
        self.steps_since_link = torch.zeros(b, device=dev)
        self.episode_len = torch.full((b,), float(EPISODE_STEPS), device=dev)
        self.speed_scale = torch.ones(b, device=dev)
        self.jammer_on = torch.ones(b, device=dev)
        self.battery_scale = torch.ones(b, device=dev)
        self.snap: Snapshot | None = None

    # ------------------------------------------------------------------ #
    # Episode setup
    # ------------------------------------------------------------------ #

    def set_stage_weights(self, weights: Sequence[float]) -> None:
        """Reweight which curriculum stages fresh episodes are drawn from.

        The curriculum is the one thing that legitimately varies *during* a run
        (docs/ENVIRONMENT.md: "Curriculum varies within one training run.
        Fidelity varies between runs"), and `EnvConfig` is frozen, so the
        callback needs a seam. This is it, and it is the whole seam: it touches
        the sampling distribution over `STAGES` and nothing else.

        Takes effect at the next auto-reset, per environment -- episodes already
        in flight keep the stage they were drawn under, which is what makes the
        transition smooth rather than a discontinuity mid-episode.

        ⛔ Never reachable from `fidelity`. The schedule that drives this must be
        a pure function of the training step and identical in every fidelity
        condition, or easier rungs reach the final stage sooner and RQ1 is
        confounded past repair -- `src/training/curriculum.py` and its test.
        """
        if len(weights) != len(STAGES):
            raise ValueError(f"expected {len(STAGES)} stage weights, got {len(weights)}")
        w = torch.tensor(tuple(weights), device=self.device, dtype=torch.float32)
        if (w < 0).any() or float(w.sum()) <= 0.0:
            raise ValueError(f"stage weights must be non-negative with a positive sum, got {w}")
        self.stage_cdf = (w / w.sum()).cumsum(0)

    def _sample_episode(self, mask: Tensor) -> None:
        """Draw fresh episodes where `mask`, in place.

        Fresh values are generated for the whole batch and written with
        `torch.where`. Boolean indexing would give data-dependent shapes, which
        breaks `fullgraph=True` compilation and can trigger a recompilation loop.
        """
        cfg = self.cfg
        b, n, dev = cfg.num_envs, cfg.num_drones, self.device
        m1 = mask.unsqueeze(-1)
        m2 = mask.unsqueeze(-1).unsqueeze(-1)

        stage = torch.searchsorted(
            self.stage_cdf, torch.rand(b, device=dev, generator=self.gen)
        ).clamp_max(len(STAGES) - 1)
        row = self.stage_table[stage]  # (B, 6)
        self.episode_len = torch.where(mask, row[:, 0], self.episode_len)
        self.speed_scale = torch.where(mask, row[:, 1], self.speed_scale)
        self.jammer_on = torch.where(mask, row[:, 2], self.jammer_on)
        self.battery_scale = torch.where(mask, row[:, 3], self.battery_scale)

        n_routes = self.route_xy.shape[0]
        lo, hi = (
            (n_routes - N_EVAL_ROUTES, n_routes)
            if cfg.eval_routes
            else (0, n_routes - N_EVAL_ROUTES)
        )
        route = torch.randint(lo, hi, (b,), device=dev, generator=self.gen)
        self.route_id = torch.where(mask, route, self.route_id)

        mcv = torch.cat(
            [self.route_mcv[self.route_id], torch.full((b, 1), MCV_Z_M, device=dev)], dim=-1
        )
        hvt = torch.cat(
            [self.route_xy[self.route_id, 0], torch.full((b, 1), HVT_Z_M, device=dev)], dim=-1
        )
        self.mcv_pos = torch.where(m1, mcv, self.mcv_pos)
        self.hvt_pos = torch.where(m1, hvt, self.hvt_pos)
        self.hvt_vel = torch.where(m1, torch.zeros_like(hvt), self.hvt_vel)

        # Drones launch from the MCV, on a ring so they neither share a position
        # (zero-length links) nor all sit in the same building box.
        #
        # They start at ALT_MIN rather than on the ground: the A2G model is not
        # valid below 40 m (TR 36.777 stops at 22.5 m) and 37 % of ground
        # positions sit inside a building box, where occlusion's endpoint
        # convention would let a drone see through its own building. What the
        # launch phase is *for* -- the chain forming during transit and the
        # energy cost of flying out -- is horizontal and survives intact.
        phase = torch.rand(b, 1, device=dev, generator=self.gen) * 2 * torch.pi
        ring = phase + torch.arange(n, device=dev).unsqueeze(0) * (2 * torch.pi / n)
        spawn = torch.stack(
            [
                self.mcv_pos[:, None, 0] + SPAWN_RING_M * torch.cos(ring),
                self.mcv_pos[:, None, 1] + SPAWN_RING_M * torch.sin(ring),
                torch.full((b, n), ALT_MIN_M, device=dev),
            ],
            dim=-1,
        )
        self.drone_pos = torch.where(m2, spawn, self.drone_pos)
        self.drone_vel = torch.where(m2, torch.zeros_like(spawn), self.drone_vel)
        self.last_accel = torch.where(m1, torch.zeros_like(self.last_accel), self.last_accel)

        # Initial charge is randomised in [charge_min, 1] at the design stage --
        # a swarm mid-sortie has heterogeneous charge, which is what gives
        # Var(B) something to act on from step 1 (docs/ENVIRONMENT.md).
        floor = row[:, 5].unsqueeze(-1)
        charge = floor + (1.0 - floor) * torch.rand(b, n, device=dev, generator=self.gen)
        self.battery = torch.where(m1, charge, self.battery)

        self.t = torch.where(mask, torch.zeros_like(self.t), self.t)
        self.steps_since_link = torch.where(
            mask, torch.zeros_like(self.steps_since_link), self.steps_since_link
        )

        # One-shot cue, never refreshed. It decays in range rather than in
        # direction, which is why it stays observable all episode (BLOCK_D.md).
        sigma = row[:, 4].unsqueeze(-1)
        noise = torch.randn(b, 2, device=dev, generator=self.gen) * sigma
        cue = torch.cat(
            [self.hvt_pos[:, :2] + noise, torch.full((b, 1), HVT_Z_M, device=dev)], dim=-1
        )
        self.cue = torch.where(m1, cue, self.cue)

    def reset(self, seed: int | None = None) -> dict[str, Tensor]:
        """Start fresh episodes everywhere and return the first observation."""
        if seed is not None:
            self.gen.manual_seed(seed)
        all_envs = torch.ones(self.cfg.num_envs, dtype=torch.bool, device=self.device)
        self._sample_episode(all_envs)
        self.snap, aux = self._evaluate()
        return self._observe(aux)

    # ------------------------------------------------------------------ #
    # Physics
    # ------------------------------------------------------------------ #

    def _advance_drones(self, actions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Point-mass kinematics with box and altitude limits."""
        accel = actions.clamp(-1.0, 1.0) * MAX_ACCEL_MS2
        vel = self.drone_vel + accel * self.cfg.dt_s

        speed = vel.norm(dim=-1, keepdim=True)
        vel = vel * (DRONE_DASH_MS / speed.clamp_min(1e-6)).clamp(max=1.0)

        want = self.drone_pos + vel * self.cfg.dt_s
        # ⚠️ Built once in `__init__`, NOT here. `torch.tensor([...])` from a
        # Python list copies host memory to the device and forces a
        # synchronisation -- a full pipeline stall on every step, invisible in a
        # diff. Caught by `test_step_never_syncs_to_the_host`, which is CUDA-only
        # and so had never executed until the first GPU session (2026-08-24).
        pos = torch.maximum(torch.minimum(want, self.pos_hi), self.pos_lo)

        # Zero the component that hit a limit. Without this the drone presses
        # into the wall and the energy term charges for motion that never
        # happened.
        vel = torch.where(pos != want, torch.zeros_like(vel), vel)
        return pos, vel, accel

    def _advance_hvt(self, t: Tensor) -> tuple[Tensor, Tensor]:
        """Index the pre-baked route. No graph search in the hot loop."""
        last = self.route_xy.shape[1] - 1
        idx = (t.to(torch.float32) * self.speed_scale).long().clamp(0, last)
        prev = ((t - 1).to(torch.float32) * self.speed_scale).long().clamp(0, last)
        xy = self.route_xy[self.route_id, idx]
        vel_xy = (xy - self.route_xy[self.route_id, prev]) / self.cfg.dt_s
        z = torch.full((xy.shape[0], 1), HVT_Z_M, device=self.device)
        return torch.cat([xy, z], dim=-1), torch.cat([vel_xy, torch.zeros_like(z)], dim=-1)

    def _free_clearance(self, pos_k: Tensor) -> Tensor:
        k = pos_k.shape[1]
        return torch.full((pos_k.shape[0], k, k), occlusion.FREE_CLEARANCE_M, device=pos_k.device)

    def _clearance(self, pos_k: Tensor) -> tuple[Tensor, Tensor]:
        """Line of sight for every node pair, twice: `(true, channel)`.

        **`true` is the world and `channel` is the model of it.** The sensor,
        the reward's observation potential and every diagnostic read `true` at
        every rung; only `_capacity` and the channel-derived observation
        features read `channel`. That split is docs/BLOCK_F.md decisions 1 and 2
        and it is what keeps RQ1 interpretable:

        - if the sensor were gated, F0 would be *a city with no buildings* and
          the F0->F1 gap would conflate sensor occlusion with link occlusion --
          and Block E measured those at wildly different sizes (observation is
          ~93 % solved by geometry while the chain binds), so the larger,
          uninteresting effect would swamp the smaller, interesting one;
        - if `chain_occluded` were gated it would read 0.0 % under F0 *by
          construction*, destroying the headline failure-attribution metric in
          the one condition it exists to expose.

        The two are the same tensor whenever the rung models occlusion, so the
        split costs an allocation at F0 and nothing anywhere else. Occlusion
        itself runs at every rung, which is why **no rung is cheaper to
        simulate** -- good for the budget and good for comparability, since a
        faster rung would quietly get more samples per GPU-hour.

        Only this kernel is compiled, not `step()`. Two independent reasons,
        both measured at D1 and both to be re-checked on CUDA:

        - `fullgraph=True` over `step()` fails with "failed to convert
          args/kwargs to proxy" -- dynamo cannot proxy the `torch.Generator`
          that `_sample_episode` uses. Dropping the generator would fix it but
          would cost per-seed reproducibility, which is worth more than fusing
          stages that are 0.3 % of the cost.
        - Inductor's Metal backend then fails to codegen the whole step at all
          (`float64 cast requested`; MPS has no fp64). CUDA/Triton is a
          different backend and is expected to be fine, but that is untested.

        Since occlusion is 99.7 % of the eager step, compiling it alone captures
        essentially all the available speedup anyway.
        """
        true = (
            self._free_clearance(pos_k)
            if self.cfg.no_buildings
            else self._pairwise(pos_k, self.boxes, self.heights, chunk=self.cfg.occlusion_chunk)
        )
        channel_clr = true if self.cfg.channel_occlusion else self._free_clearance(pos_k)
        return true, channel_clr

    def _jammer_mw(self, radio: Tensor, channel_clr: Tensor) -> Tensor:
        """Barrage emitter riding the HVT, reaching every radio node.

        `channel_jammer` is F3's switch and is multiplied in ALONGSIDE the
        curriculum's `jammer_on`, never instead of it -- docs/BLOCK_F.md
        decision 4. `jammer_on` is drawn per episode from the stage table and
        must ramp identically in every fidelity condition, with the rung
        deciding whether it *does* anything; driving F3 from it would confound
        RQ1's jammer rung with the curriculum unrecoverably.

        Reads the channel's clearance, not the true one, so the emitter's line
        of sight is modelled at the same fidelity as the links it degrades.
        Under F3 and F4 the two are identical; under F0-F2 the result is zeroed
        anyway.
        """
        r = self.cfg.n_radio
        d = (radio - self.hvt_pos.unsqueeze(1)).norm(dim=-1)
        los = channel_clr[:, :r, self.hvt_idx] >= 0.0
        pathloss = channel.pathloss_a2g_umi_av_db(d, radio[..., 2], los)
        return (
            channel.dbm_to_mw(JAMMER_DBM - pathloss)
            * self.jammer_on.unsqueeze(-1)
            * float(self.cfg.channel_jammer)
        )

    def _capacity(self, pos_k: Tensor, channel_clr: Tensor) -> tuple[Tensor, Tensor]:
        """Per-link capacity **under the rung's channel model**.

        F0/F1 -- binary. `C_max` if the pair is within `R`, and (F1 only) if the
        ray is unoccluded. This is the connectivity-radius abstraction RQ1 is
        about, and the two rungs share one branch because at F0 `channel_clr` is
        free everywhere, so the occlusion term is vacuously true.

        F2/F3/F4 -- continuous. Path loss -> SINR -> Shannon with the modulation
        cap, which is the model Block A built and PHYSICS.md documents.

        `docs/PHYSICS.md` requires `tx_mask` to hold only the transmitters active
        in the evaluated slot, which for a <=3-hop reuse-3 chain is one node. No
        single mask can be "only i" for every link at once, and the capacity
        matrix has to exist before routing picks a path -- so every candidate
        link is evaluated as if alone in its slot. Intra-swarm interference is
        then identically zero and SINR reduces to S / (J + N0). Pinned against
        `channel.sinr_db` with a one-hot mask in the tests.
        """
        cfg = self.cfg
        r = cfg.n_radio
        radio = pos_k[:, :r]
        d3d = channel.pairwise_distance_m(radio)
        occluded = channel_clr[:, :r, :r] < 0.0

        jam_mw = self._jammer_mw(radio, channel_clr)

        if cfg.binary_capacity:
            # No path loss, no SINR: "connected" is a predicate on geometry, and
            # a connected link runs at the modulation ceiling. The jammer cannot
            # enter here even in principle, which is why F2 -- and not F1 -- is
            # the rung that has to come before F3.
            usable = (d3d < cfg.radius_m) & ~occluded
            return usable.to(d3d.dtype) * F0_CAPACITY_MBPS * self.no_self, jam_mw

        z = radio[..., 2]
        h_uav = torch.maximum(z.unsqueeze(-1), z.unsqueeze(-2))
        pathloss = torch.where(
            self.is_a2a,
            channel.pathloss_a2a_db(d3d, occluded),
            channel.pathloss_a2g_umi_av_db(d3d, h_uav, ~occluded),
        )
        prx_dbm = channel.received_power_dbm(self.ptx, pathloss)

        denom_mw = jam_mw.unsqueeze(1) + self.noise_mw  # (B, 1, R): landing on rx j
        sinr_db = prx_dbm - channel.mw_to_dbm(denom_mw)
        return channel.capacity_mbps(sinr_db, BANDWIDTH_HZ) * self.no_self, jam_mw

    def _evaluate(self) -> tuple[Snapshot, dict[str, Tensor]]:
        """Physics of the *current* state. Pure: advances nothing."""
        cfg = self.cfg
        n = cfg.num_drones

        pos_k = torch.cat(
            [self.drone_pos, self.mcv_pos.unsqueeze(1), self.hvt_pos.unsqueeze(1)], dim=1
        )
        true_clr, channel_clr = self._clearance(pos_k)

        # The SENSOR runs on true geometry at every rung. RQ1 asks which effects
        # a *channel model* must include, and a camera is not part of a channel
        # model -- docs/BLOCK_F.md decision 1.
        clr_hvt = true_clr[:, :n, self.hvt_idx]
        dist_hvt = (self.drone_pos - self.hvt_pos.unsqueeze(1)).norm(dim=-1)
        sees = (clr_hvt >= 0.0) & (dist_hvt <= SENSOR_RANGE_M)

        capacity, jam_mw = self._capacity(pos_k, channel_clr)
        source = torch.cat([sees, torch.zeros_like(sees[:, :1])], dim=1)
        e2e, on_path, on_edge, hops = routing.best_relay_path(
            capacity,
            source,
            dst_index=self.mcv_idx,
            max_hops=cfg.n_radio - 1,
            reuse_limit=cfg.reuse_limit,
        )

        # The drone holding the best ray, and its range. Two extra kernels, no
        # host sync -- and it is NOT `dist_hvt.min()`: a drone can be nearest and
        # blind (wrong side of a building), which is exactly the case
        # `nearest_dist_m` cannot express. Only `reward.potential` reads it, and
        # only when `w_hold > 0`.
        best_clr, observer = clr_hvt.max(dim=-1)

        snap = Snapshot(
            observed=sees.any(dim=-1),
            e2e_capacity_mbps=e2e,
            nearest_dist_m=dist_hvt.min(dim=-1).values,
            best_clearance_m=best_clr,
            observer_dist_m=dist_hvt.gather(1, observer.unsqueeze(1)).squeeze(1),
            # The one per-drone quantity the reward can use. Only read when
            # `w_relay > 0`; free here, since routing already produced it.
            on_path=on_path[:, :n],
            # Raw geometry for `Phi_cover`, which is a function of where EVERY
            # drone is rather than of a reduction over them -- read only when
            # `w_cover > 0`. Views, not copies: no work when the term is off.
            drone_pos=self.drone_pos,
            mcv_pos=self.mcv_pos,
            hvt_pos=self.hvt_pos,
            battery=self.battery,
            speed_ms=self.drone_vel.norm(dim=-1),
            accel_ms2=self.last_accel,
        )
        aux = {
            # What the OBSERVATION's channel features report -- the model in
            # force, not the world. See `_observe`.
            "clearance": channel_clr,
            # What the SENSOR and every DIAGNOSTIC report. Never gated.
            "true_clearance": true_clr,
            "capacity_mbps": capacity,
            "e2e_capacity_mbps": e2e,
            "sees_hvt": sees,
            "on_path": on_path,
            "on_edge": on_edge,
            "hop_count": hops,
            "jam_mw": jam_mw,
            # RQ1's headline diagnostic: does the chain the router actually chose
            # run through a building? A radius-trained policy's signature.
            #
            # ⚠️ TRUE clearance, always. Computed from the fidelity-gated
            # clearance it would read 0.0 % under F0 *by construction* -- the F0
            # policy routes straight through towers and the metric would report
            # that it never happens, destroying the failure-attribution number
            # in the one condition it exists to expose. Decision 2, and the
            # reason decision 1 makes it free: the real clearance is computed at
            # every rung anyway.
            "chain_occluded": (on_edge & (true_clr[:, : cfg.n_radio, : cfg.n_radio] < 0.0))
            .any(dim=-1)
            .any(dim=-1),
        }
        return snap, aux

    # ------------------------------------------------------------------ #
    # Observations
    # ------------------------------------------------------------------ #

    def _observe(self, aux: dict[str, Tensor]) -> dict[str, Tensor]:
        """Actor-local views plus the critic's global state.

        The actor may only see what a real drone could sense or receive; global
        state belongs to the critic. Violating that turns decentralized execution
        into centralized execution and invalidates CTDE (docs/ENVIRONMENT.md).

        Three of the 108 dims are **channel state** rather than sensing, and they
        report the channel model in force -- the measured noise floor, the
        clearance margin on the link to the MCV, and the per-edge clearance
        margin. The other three channel-derived features (`on_path`, e2e
        capacity, per-edge capacity) follow the rung for free, since they are
        computed from its capacity matrix.

        The rule, in one line: **sensor features report the sensor, channel
        features report the channel model, diagnostics report the truth.**

        Why gate them at all -- docs/BLOCK_F.md decisions 1 and 2 settle the
        sensor and the diagnostics but not this, and it is their third sibling.
        A radius simulator has no building data to put in an observation, so
        reporting true clearance under F0 would be reporting a quantity that
        model does not possess. Worse, it would leave F0's observation
        *internally contradictory* -- an edge reporting 74 Mbps beside a
        clearance feature reading -150 m -- and that contradiction is learnable
        in exactly the direction that would understate the F0->F1 gap RQ1 exists
        to measure. It is not hypothetical: B0's link repair hill-climbs on
        these two features, and ungated it would keep repairing against
        buildings the channel never charges it for.
        """
        cfg = self.cfg
        b, n = cfg.num_envs, cfg.num_drones
        pos, vel = self.drone_pos, self.drone_vel
        clearance, capacity = aux["clearance"], aux["capacity_mbps"]
        sees = aux["sees_hvt"]

        def clr(x: Tensor) -> Tensor:
            return x.clamp(-CLEARANCE_CLAMP_M, CLEARANCE_CLAMP_M) / CLEARANCE_CLAMP_M

        def cap(x: Tensor) -> Tensor:
            return (x / CAPACITY_THRESHOLD_MBPS).clamp(0.0, CAPACITY_CLAMP)

        # Sensor quantity -> true geometry. It is the same ray `sees_hvt` is
        # computed from, so gating it would put the soft flag and the hard gate
        # into disagreement under F0.
        clr_hvt = aux["true_clearance"][:, :n, self.hvt_idx]
        # Radio quantity: the drone's link to the MCV -> the channel's geometry.
        clr_mcv = clearance[:, :n, self.mcv_idx]
        rel_hvt = (self.hvt_pos.unsqueeze(1) - pos) / POS_SCALE_M
        noise_dbm = channel.mw_to_dbm(aux["jam_mw"][:, :n] + self.noise_mw)

        ego = torch.cat(
            [
                vel / VEL_SCALE_MS,  # 3  own velocity, INS
                ((pos[..., 2] - ALT_MIN_M) / (ALT_MAX_M - ALT_MIN_M)).unsqueeze(-1),  # 1
                (self.cue.unsqueeze(1) - pos) / POS_SCALE_M,  # 3  briefed cue
                self.battery.unsqueeze(-1),  # 1
                torch.sigmoid(clr_hvt / SOFT_SEE_TAU_M).unsqueeze(-1),  # 1  soft sees
                rel_hvt * sees.unsqueeze(-1),  # 3  zeroed when unseen
                ((self.hvt_vel.unsqueeze(1) - vel) / VEL_SCALE_MS) * sees.unsqueeze(-1),  # 3
                (self.mcv_pos.unsqueeze(1) - pos) / POS_SCALE_M,  # 3
                ((noise_dbm - NOISE_REF_DBM) / NOISE_SCALE_DB).unsqueeze(-1),  # 1
                clr(clr_hvt).unsqueeze(-1),  # 1
                clr(clr_mcv).unsqueeze(-1),  # 1
                aux["on_path"][:, :n].float().unsqueeze(-1),  # 1
                cap(aux["e2e_capacity_mbps"]).unsqueeze(-1).expand(b, n).unsqueeze(-1),  # 1
                (self.steps_since_link / LINK_TIMEOUT_SCALE)
                .unsqueeze(-1)
                .expand(b, n)
                .unsqueeze(-1),  # 1
            ],
            dim=-1,
        )

        # --- neighbours: standard MANET position reporting, nothing global ---
        nb = self.nb_idx
        neighbour = torch.cat(
            [
                (pos[:, nb] - pos.unsqueeze(2)) / POS_SCALE_M,
                (vel[:, nb] - vel.unsqueeze(2)) / VEL_SCALE_MS,
                self.battery[:, nb].unsqueeze(-1),
                sees[:, nb].float().unsqueeze(-1),
                aux["on_path"][:, :n][:, nb].float().unsqueeze(-1),
            ],
            dim=-1,
        )
        edge = torch.stack(
            [
                cap(capacity[:, self.self_idx, nb]),
                clr(clearance[:, self.self_idx, nb]),
            ],
            dim=-1,
        )

        return {
            "ego": ego,
            "neighbour": neighbour,
            "edge": edge,
            "flat": self._pack(ego, neighbour, edge),
            "state": self._critic_state(aux),
        }

    def _pack(self, ego: Tensor, neighbour: Tensor, edge: Tensor) -> Tensor:
        """Max-N padded flat vector, `(B, N, 108)`.

        skrl's rollout storage wants one fixed-shape tensor per agent, and
        `docs/MODELS.md` needs max-N padding so the MLP rung can be evaluated at
        N in {3, 8} at all. Every architecture consumes this and unpacks it, so
        the padding is identical across rungs by construction.

        `unpack_flat` (module level) is the inverse and is what they unpack with.
        """
        b, n, k = ego.shape[0], self.cfg.num_drones, N_MAX - 1
        real = self.cfg.num_drones - 1
        nb = torch.zeros(b, n, k, NEIGHBOUR_DIM, device=ego.device)
        eg = torch.zeros(b, n, k, EDGE_DIM, device=ego.device)
        valid = torch.zeros(b, n, k, device=ego.device)
        if real > 0:
            nb[:, :, :real] = neighbour
            eg[:, :, :real] = edge
            valid[:, :, :real] = 1.0
        return torch.cat([ego, nb.flatten(2), eg.flatten(2), valid], dim=-1)

    def _critic_state(self, aux: dict[str, Tensor]) -> Tensor:
        """Global state, training only, discarded at evaluation.

        Need not be size-agnostic: zero-shot transfer to N in {3, 8} runs the
        actor alone (docs/MODELS.md). Positions are MCV-relative so the critic
        is translation-invariant across map locations.
        """
        n = self.cfg.num_drones
        rel_pos = (self.drone_pos - self.mcv_pos.unsqueeze(1)) / POS_SCALE_M
        return torch.cat(
            [
                rel_pos.flatten(1),
                (self.drone_vel / VEL_SCALE_MS).flatten(1),
                self.battery,
                (self.hvt_pos - self.mcv_pos) / POS_SCALE_M,
                self.hvt_vel / VEL_SCALE_MS,
                (aux["e2e_capacity_mbps"] / CAPACITY_THRESHOLD_MBPS)
                .clamp(0.0, CAPACITY_CLAMP)
                .unsqueeze(-1),
                (aux["hop_count"].float() / 3.0).unsqueeze(-1),
                aux["sees_hvt"].float(),
                aux["on_path"][:, :n].float(),
                (self.steps_since_link / LINK_TIMEOUT_SCALE).unsqueeze(-1),
            ],
            dim=-1,
        )

    # ------------------------------------------------------------------ #
    # Step
    # ------------------------------------------------------------------ #

    def step(
        self, actions: Tensor
    ) -> tuple[dict[str, Tensor], Tensor, Tensor, Tensor, dict[str, Tensor]]:
        """Advance every environment one tick.

        Returns `(obs, reward, terminated, truncated, extras)`. `obs` is of the
        state *after* auto-reset; the pre-reset observation is in
        `extras["final_observation"]`, which the learner needs to bootstrap
        correctly at truncation.
        """
        cfg = self.cfg
        pos, vel, accel = self._advance_drones(actions)
        self.drone_pos, self.drone_vel = pos, vel
        self.last_accel = accel.norm(dim=-1)

        self.t = self.t + 1
        self.hvt_pos, self.hvt_vel = self._advance_hvt(self.t)

        # Battery drains on the post-transition speed (explicit Euler). The
        # reward's energy term charges the pre-transition speed, per
        # `reward(s, s_next)`; the two agree in aggregate over an episode.
        power = total_power_w(vel.norm(dim=-1), self.last_accel, self.craft) + climb_power_w(
            vel[..., 2], self.craft
        )
        drain = power * cfg.dt_s / (BATTERY_WH * 3600.0 * self.battery_scale.unsqueeze(-1))
        self.battery = (self.battery - drain).clamp_min(0.0)

        new_snap, aux = self._evaluate()

        # Battery death is physical and unhackable -- hovering at the MCV burns
        # power too. Mission failure is NEVER terminal: terminating on it teaches
        # the policy to never acquire, and kills a random initial policy before
        # it ever reaches the tracking phase (docs/DECISIONS.md).
        terminated = (self.battery <= 0.0).any(dim=-1)
        truncated = (self.t.to(torch.float32) >= self.episode_len) & ~terminated

        rew = reward(
            self.snap,
            new_snap,
            self.weights,
            cfg.gamma,
            next_is_terminal=terminated,
            craft=self.craft,
        )

        alive = new_snap.e2e_capacity_mbps >= CAPACITY_THRESHOLD_MBPS
        self.steps_since_link = torch.where(
            alive, torch.zeros_like(self.steps_since_link), self.steps_since_link + 1.0
        )

        capable = mission_capable(new_snap)
        final_obs = self._observe(aux)
        extras = {
            "final_observation": final_obs["flat"],
            "mission_capable": capable,
            "e2e_capacity_mbps": new_snap.e2e_capacity_mbps,
            "hop_count": aux["hop_count"],
            "chain_occluded": aux["chain_occluded"],
            "on_path": aux["on_path"],
            "on_edge": aux["on_edge"],
            "sees_any": new_snap.observed,
            # Per-drone sighting and the raw link matrix. Block E needs both:
            # `sees_hvt` is the observer identity RQ3's handoff metrics are built
            # from, and `capacity_mbps` lets an evaluator re-run the routing DP
            # under a different `reuse_limit` without re-running the physics --
            # which is how the rate-division counterfactual in docs/BLOCK_E.md
            # is measured. Both are already computed; this only stops them being
            # thrown away.
            "sees_hvt": aux["sees_hvt"],
            "capacity_mbps": aux["capacity_mbps"],
            "altitude_m": self.drone_pos[..., 2],
            "battery": self.battery,
        }

        # Off by default: these widen the output contract, which `test_golden.py`
        # pins deliberately. See `EnvConfig.training_extras`.
        if cfg.training_extras:
            extras["final_state"] = final_obs["state"]
            for name, value in reward_terms(
                self.snap,
                new_snap,
                self.weights,
                cfg.gamma,
                next_is_terminal=terminated,
                craft=self.craft,
            ).items():
                extras[f"reward/{name}"] = value

        if not cfg.auto_reset:
            self.snap = new_snap
            return final_obs, rew, terminated, truncated, extras

        # Auto-reset, then re-evaluate. The second pass produces both the
        # observation auto-reset must return AND Phi of the fresh state, which
        # the next step's shaping needs -- see the module docstring.
        self._sample_episode(terminated | truncated)
        self.snap, aux_new = self._evaluate()
        return self._observe(aux_new), rew, terminated, truncated, extras
