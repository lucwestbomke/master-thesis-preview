"""Batched env core -- Block D.

The training path. Carries a leading `num_envs` dimension and never leaves the
GPU: no `.item()`, no `.cpu()`, no Python loop over environments. `swarm_env.py`
is the thin PettingZoo adapter over this, for API-compliance tests and visual
debugging only (`AGENTS.md`, `docs/DECISIONS.md`).

**Milestone D0: physics only.** Kinematics, occlusion, channel, routing, energy.
Observations, reward, termination and auto-reset arrive in D1 -- deliberately,
because `AGENTS.md` requires the throughput profile to be measured before
anything is built on top of it. `scripts/bench_env.py` is what measures it.

Node layout -- two sets, deliberately different sizes. Confusing them feeds the
HVT into the routing DP as a relay:

    geometric  K = N + 2     0..N-1 drones,  N = MCV,  N+1 = HVT
    radio      R = N + 1     0..N-1 drones,  N = MCV

Occlusion runs over all K, so the drone-HVT rays serve the sensor *and* the
jammer path from one call. The HVT is never a radio node: it is the target and
the emitter, not a relay.

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


@dataclass(frozen=True)
class EnvConfig:
    num_envs: int
    num_drones: int = 5
    device: str = "cpu"
    seed: int = 0
    dt_s: float = DT_S
    episode_steps: int = EPISODE_STEPS
    occlusion_chunk: int = 512
    reuse_limit: int = 3
    eval_routes: bool = False
    # Fidelity seams for Block F. Only the full model (=F4) is wired in D.
    use_occlusion: bool = True
    use_jammer: bool = True

    @property
    def n_geometric(self) -> int:
        return self.num_drones + 2

    @property
    def n_radio(self) -> int:
        return self.num_drones + 1


class BatchedSwarmEnv:
    """`num_envs` copies of the relay-tracking mission, stepped in lockstep."""

    def __init__(self, cfg: EnvConfig, artefact: Path = ARTEFACT):
        self.cfg = cfg
        dev = torch.device(cfg.device)
        self.device = dev
        self.craft: Rotorcraft = DEFAULT_AIRFRAME
        self.gen = torch.Generator(device="cpu").manual_seed(cfg.seed)

        art = np.load(artefact)
        self.boxes = torch.from_numpy(art["building_boxes"]).float().to(dev)
        self.heights = torch.from_numpy(art["building_heights"]).float().to(dev)
        self.route_xy = torch.from_numpy(art["route_xy"]).float().to(dev)
        self.route_mcv = torch.from_numpy(art["route_mcv"]).float().to(dev)

        n, r = cfg.num_drones, cfg.n_radio
        self.mcv_idx, self.hvt_idx = n, n + 1

        # Link class is pure index arithmetic: drone-drone is A2A, anything
        # touching the MCV is A2G. Static, so build it once.
        drone = torch.arange(r, device=dev) < n
        self.is_a2a = drone.unsqueeze(0) & drone.unsqueeze(1)  # (R, R)
        self.no_self = 1.0 - torch.eye(r, device=dev)

        self.n0_dbm = channel.noise_floor_dbm(BANDWIDTH_HZ, NOISE_FIGURE_DB)
        self.noise_mw = channel.dbm_to_mw(torch.tensor(self.n0_dbm, device=dev))
        self.ptx = torch.full((cfg.num_envs, r), PTX_DBM, device=dev)

        b = cfg.num_envs
        self.drone_pos = torch.zeros(b, n, 3, device=dev)
        self.drone_vel = torch.zeros(b, n, 3, device=dev)
        self.battery = torch.ones(b, n, device=dev)
        self.mcv_pos = torch.zeros(b, 3, device=dev)
        self.hvt_pos = torch.zeros(b, 3, device=dev)
        self.cue = torch.zeros(b, 3, device=dev)
        self.route_id = torch.zeros(b, dtype=torch.long, device=dev)
        self.t = torch.zeros(b, dtype=torch.long, device=dev)
        self.speed_scale = torch.ones(b, device=dev)
        self.jammer_on = torch.ones(b, device=dev)
        self.capacity_scale = torch.ones(b, device=dev)

    # ------------------------------------------------------------------ #
    # Episode setup
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Fresh episodes in every environment.

        D0 resets all of them; D1 adds selective auto-reset of finished ones.
        """
        cfg = self.cfg
        b, n = cfg.num_envs, cfg.num_drones
        dev = self.device

        n_routes = self.route_xy.shape[0]
        lo, hi = (
            (n_routes - N_EVAL_ROUTES, n_routes)
            if cfg.eval_routes
            else (0, n_routes - N_EVAL_ROUTES)
        )
        self.route_id = torch.randint(lo, hi, (b,), generator=self.gen).to(dev)

        self.mcv_pos = torch.cat(
            [self.route_mcv[self.route_id], torch.full((b, 1), MCV_Z_M, device=dev)], dim=-1
        )
        self.hvt_pos = torch.cat(
            [self.route_xy[self.route_id, 0], torch.full((b, 1), HVT_Z_M, device=dev)], dim=-1
        )

        # Drones launch from the MCV, on a ring so they do not share a position
        # (zero-length links, and every one of them inside the same box).
        #
        # They start at ALT_MIN rather than on the ground: the model is not valid
        # below 40 m (TR 36.777 stops at 22.5 m, and 37 % of ground positions sit
        # inside a building box), and simulating a take-off through a band where
        # the physics is wrong buys nothing. What the launch phase is *for* --
        # the chain forming during transit, and the energy cost of flying out --
        # is horizontal and survives intact. See docs/BLOCK_D.md.
        phase = torch.rand(b, 1, generator=self.gen).to(dev) * 2 * torch.pi
        k = torch.arange(n, device=dev).unsqueeze(0) * (2 * torch.pi / n)
        ring = phase + k
        self.drone_pos = torch.stack(
            [
                self.mcv_pos[:, None, 0] + SPAWN_RING_M * torch.cos(ring),
                self.mcv_pos[:, None, 1] + SPAWN_RING_M * torch.sin(ring),
                torch.full((b, n), ALT_MIN_M, device=dev),
            ],
            dim=-1,
        )
        self.drone_vel = torch.zeros(b, n, 3, device=dev)
        self.battery = torch.ones(b, n, device=dev)
        self.t = torch.zeros(b, dtype=torch.long, device=dev)

        # One-shot cue: never refreshed, and it decays in range rather than in
        # direction, which is why it can be observed all episode (BLOCK_D.md).
        noise = torch.randn(b, 2, generator=self.gen).to(dev) * 150.0
        self.cue = torch.cat(
            [self.hvt_pos[:, :2] + noise, torch.full((b, 1), HVT_Z_M, device=dev)], dim=-1
        )

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
        if not self.cfg.use_occlusion:  # F0 seam -- Block F turns this off
            return torch.full(
                pos_k.shape[:1] + (pos_k.shape[1],) * 2,
                occlusion.FREE_CLEARANCE_M,
                device=pos_k.device,
            )
        return occlusion.pairwise_clearance(
            pos_k, self.boxes, self.heights, chunk=self.cfg.occlusion_chunk
        )

    def _capacity(self, pos_k: Tensor, clearance: Tensor) -> Tensor:
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
        prx_dbm = channel.received_power_dbm(self.ptx[: radio.shape[0]], pathloss)

        jam_mw = self._jammer_mw(radio, clearance)
        denom_mw = jam_mw.unsqueeze(1) + self.noise_mw  # (B, 1, R): landing on rx j
        sinr_db = prx_dbm - channel.mw_to_dbm(denom_mw)
        return channel.capacity_mbps(sinr_db, BANDWIDTH_HZ) * self.no_self

    def _jammer_mw(self, radio: Tensor, clearance: Tensor) -> Tensor:
        """Barrage emitter riding the HVT, reaching every radio node."""
        r = self.cfg.n_radio
        d = (radio - self.hvt_pos.unsqueeze(1)).norm(dim=-1)
        los = clearance[:, :r, self.hvt_idx] >= 0.0
        pathloss = channel.pathloss_a2g_umi_av_db(d, radio[..., 2], los)
        return channel.dbm_to_mw(JAMMER_DBM - pathloss) * self.jammer_on.unsqueeze(-1)

    def physics_step(self, actions: Tensor) -> dict[str, Tensor]:
        """One tick of the world. D0: no reward, no observations, no reset.

        Args:
            actions: `(B, N, 3)` normalised acceleration in [-1, 1].
        """
        cfg = self.cfg
        n = cfg.num_drones

        pos, vel, accel = self._advance_drones(actions)
        t = self.t + 1
        hvt_pos, hvt_vel = self._advance_hvt(t)

        pos_k = torch.cat([pos, self.mcv_pos.unsqueeze(1), hvt_pos.unsqueeze(1)], dim=1)
        clearance = self._clearance(pos_k)

        # Sensor: an unoccluded ray within range. The range test essentially
        # never binds and is kept so BLOCK_B's non-binding claim stays literal.
        clr_hvt = clearance[:, :n, self.hvt_idx]
        dist_hvt = (pos - hvt_pos.unsqueeze(1)).norm(dim=-1)
        sees = (clr_hvt >= 0.0) & (dist_hvt <= SENSOR_RANGE_M)

        capacity = self._capacity(pos_k, clearance)
        source = torch.cat([sees, torch.zeros_like(sees[:, :1])], dim=1)
        e2e, on_path, hops = routing.best_relay_path(
            capacity,
            source,
            dst_index=self.mcv_idx,
            max_hops=cfg.n_radio - 1,
            reuse_limit=cfg.reuse_limit,
        )

        speed = vel.norm(dim=-1)
        power = total_power_w(speed, accel.norm(dim=-1), self.craft) + climb_power_w(
            vel[..., 2], self.craft
        )
        drain = power * cfg.dt_s / (BATTERY_WH * 3600.0 * self.capacity_scale.unsqueeze(-1))
        battery = (self.battery - drain).clamp_min(0.0)

        self.drone_pos, self.drone_vel, self.battery = pos, vel, battery
        self.hvt_pos, self.t = hvt_pos, t

        return {
            "clearance": clearance,
            "capacity_mbps": capacity,
            "e2e_capacity_mbps": e2e,
            "on_path": on_path,
            "hop_count": hops,
            "sees_hvt": sees,
            "power_w": power,
            "battery": battery,
            "hvt_vel": hvt_vel,
            "speed_ms": speed,
        }
