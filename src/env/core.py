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

from dataclasses import dataclass
from pathlib import Path

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
)

ARTEFACT = Path(__file__).resolve().parents[2] / "data" / "frankfurt_box.npz"

# --- scenario, from AGENTS.md "Settled parameters". Do not re-derive here. ---
BOX_HALF_M = 750.0
ALT_MIN_M = 40.0  # model-validity floor, not a flight rule -- see BLOCK_D.md
ALT_MAX_M = 120.0  # load-bearing for RQ1: above ~180 m A2A occlusion vanishes
HVT_Z_M = 1.5
MCV_Z_M = 2.0
DT_S = 0.4
EPISODE_STEPS = 600
DRONE_DASH_MS = 25.0
MAX_ACCEL_MS2 = 10.0
SENSOR_RANGE_M = 830.0  # non-binding ceiling; 99.8 % of sightlines are shorter
SPAWN_RING_M = 5.0
BATTERY_WH = 548.0

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
CAPACITY_CLAMP = 4.0  # in threshold units, so 4 == 20 Mbps
NOISE_REF_DBM = -97.0  # thermal floor at 10 MHz / 7 dB NF
NOISE_SCALE_DB = 30.0
LINK_TIMEOUT_SCALE = 50.0
SOFT_SEE_TAU_M = 15.0  # matches RewardWeights.tau_clearance_m


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


@dataclass(frozen=True)
class EnvConfig:
    num_envs: int
    num_drones: int = 5
    device: str = "cpu"
    seed: int = 0
    dt_s: float = DT_S
    occlusion_chunk: int = 512
    reuse_limit: int = 3
    gamma: float = 0.999
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
    # Fidelity seams for Block F. Only the full model (=F4) is wired in D.
    use_occlusion: bool = True

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
        lo = torch.tensor([-BOX_HALF_M, -BOX_HALF_M, ALT_MIN_M], device=self.device)
        hi = torch.tensor([BOX_HALF_M, BOX_HALF_M, ALT_MAX_M], device=self.device)
        pos = torch.maximum(torch.minimum(want, hi), lo)

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

    def _clearance(self, pos_k: Tensor) -> Tensor:
        """Line of sight for every node pair.

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
        if not self.cfg.use_occlusion:  # F0 seam -- Block F turns this off
            k = pos_k.shape[1]
            return torch.full(
                (pos_k.shape[0], k, k), occlusion.FREE_CLEARANCE_M, device=pos_k.device
            )
        return self._pairwise(pos_k, self.boxes, self.heights, chunk=self.cfg.occlusion_chunk)

    def _jammer_mw(self, radio: Tensor, clearance: Tensor) -> Tensor:
        """Barrage emitter riding the HVT, reaching every radio node."""
        r = self.cfg.n_radio
        d = (radio - self.hvt_pos.unsqueeze(1)).norm(dim=-1)
        los = clearance[:, :r, self.hvt_idx] >= 0.0
        pathloss = channel.pathloss_a2g_umi_av_db(d, radio[..., 2], los)
        return channel.dbm_to_mw(JAMMER_DBM - pathloss) * self.jammer_on.unsqueeze(-1)

    def _capacity(self, pos_k: Tensor, clearance: Tensor) -> tuple[Tensor, Tensor]:
        """Per-link capacity under the scheduled-MAC assumption.

        `docs/PHYSICS.md` requires `tx_mask` to hold only the transmitters active
        in the evaluated slot, which for a <=3-hop reuse-3 chain is one node. No
        single mask can be "only i" for every link at once, and the capacity
        matrix has to exist before routing picks a path -- so every candidate
        link is evaluated as if alone in its slot. Intra-swarm interference is
        then identically zero and SINR reduces to S / (J + N0). Pinned against
        `channel.sinr_db` with a one-hot mask in the tests.
        """
        r = self.cfg.n_radio
        radio = pos_k[:, :r]
        d3d = channel.pairwise_distance_m(radio)
        occluded = clearance[:, :r, :r] < 0.0

        z = radio[..., 2]
        h_uav = torch.maximum(z.unsqueeze(-1), z.unsqueeze(-2))
        pathloss = torch.where(
            self.is_a2a,
            channel.pathloss_a2a_db(d3d, occluded),
            channel.pathloss_a2g_umi_av_db(d3d, h_uav, ~occluded),
        )
        prx_dbm = channel.received_power_dbm(self.ptx, pathloss)

        jam_mw = self._jammer_mw(radio, clearance)
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
        clearance = self._clearance(pos_k)

        clr_hvt = clearance[:, :n, self.hvt_idx]
        dist_hvt = (self.drone_pos - self.hvt_pos.unsqueeze(1)).norm(dim=-1)
        sees = (clr_hvt >= 0.0) & (dist_hvt <= SENSOR_RANGE_M)

        capacity, jam_mw = self._capacity(pos_k, clearance)
        source = torch.cat([sees, torch.zeros_like(sees[:, :1])], dim=1)
        e2e, on_path, on_edge, hops = routing.best_relay_path(
            capacity,
            source,
            dst_index=self.mcv_idx,
            max_hops=cfg.n_radio - 1,
            reuse_limit=cfg.reuse_limit,
        )

        snap = Snapshot(
            observed=sees.any(dim=-1),
            e2e_capacity_mbps=e2e,
            nearest_dist_m=dist_hvt.min(dim=-1).values,
            best_clearance_m=clr_hvt.max(dim=-1).values,
            battery=self.battery,
            speed_ms=self.drone_vel.norm(dim=-1),
            accel_ms2=self.last_accel,
        )
        aux = {
            "clearance": clearance,
            "capacity_mbps": capacity,
            "e2e_capacity_mbps": e2e,
            "sees_hvt": sees,
            "on_path": on_path,
            "on_edge": on_edge,
            "hop_count": hops,
            "jam_mw": jam_mw,
            # RQ1's headline diagnostic: does the chain the router actually chose
            # run through a building? A radius-trained policy's signature.
            "chain_occluded": (on_edge & (clearance[:, : cfg.n_radio, : cfg.n_radio] < 0.0))
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

        clr_hvt = clearance[:, :n, self.hvt_idx]
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
            "altitude_m": self.drone_pos[..., 2],
            "battery": self.battery,
        }

        if not cfg.auto_reset:
            self.snap = new_snap
            return final_obs, rew, terminated, truncated, extras

        # Auto-reset, then re-evaluate. The second pass produces both the
        # observation auto-reset must return AND Phi of the fresh state, which
        # the next step's shaping needs -- see the module docstring.
        self._sample_episode(terminated | truncated)
        self.snap, aux_new = self._evaluate()
        return self._observe(aux_new), rew, terminated, truncated, extras
