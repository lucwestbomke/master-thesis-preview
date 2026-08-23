"""B0 -- the scripted geometric baseline. Block E.

THESIS_PLAN §3 calls B0 "the non-learned control: relays placed on the MCV->HVT
geodesic, one observer". `docs/MODELS.md` turns that into a requirement -- every
architecture must at least match it -- so B0 is not a formality. **A weak B0
flatters every result in Chapter 6 and is the first thing an examiner attacks.**

The information contract, which is the decision that sets what the headline
comparison means (`docs/BLOCK_E.md` §1):

    B0 is a pure function of `obs["flat"]` and its own carried state.

It never reads `env.hvt_pos`, `env.boxes` or any other env attribute -- it sees
exactly the `(B, N, 108)` tensor Block G's actors see, so "MARL earns its keep"
becomes a statement about *control* with information held fixed. The `oracle`
variant is the deliberate exception and takes ground truth through an explicit
argument to `act()`, so it cannot be acquired by accident. `test_b0.py` pins the
contract by feeding identical observations from two differently-seeded envs.

Two advantages B0 is granted on purpose, both stated in the write-up rather than
hidden: it carries **memory** (Block G's actors are feed-forward), and it takes
**roles from the agent index** (Block G's actors are homogeneous so that roles
must emerge). A scripted controller is programmed per airframe; both are what a
real one would do, and both make B0 harder to beat.

What it may *not* do: gossip the target's coordinates. The neighbour channel
carries a `sees_hvt` **bit**, not the observer's fix, because sharing the fix
would dissolve RQ3's coordination problem. B0 inherits that restriction, or it
would have a communication channel the learned policies do not.

Why that costs less than it looks: **the relays do not need the target, they need
the observer**, whose relative position is in the observation exactly, every
step. Only the observer and the spares need a target estimate -- and the
observer, by definition of its role, is looking at it.

The variant ladder (`VARIANTS`), which is how the design effort is made visible
rather than asserted:

    geodesic  THESIS_PLAN's literal sketch, built properly: velocity servo,
              80 m, adaptive hop count, roles fixed by index
    b0        + ranked roles from the sees bits, target-belief filter with
              dead reckoning, spare observation posts, local link repair
    oracle    + ground-truth target state. A stated upper bound, not the
              headline; the b0->oracle gap measures what target information is
              worth in this task

Design, measurements and the pre-registered decision rules: `docs/BLOCK_E.md`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
from torch import Tensor

from ..env.core import (
    ALT_MAX_M,
    ALT_MIN_M,
    CLEARANCE_CLAMP_M,
    DRONE_CRUISE_MS,
    DRONE_DASH_MS,
    DT_S,
    MAX_ACCEL_MS2,
    POS_SCALE_M,
    VEL_SCALE_MS,
    neighbour_index_table,
    unpack_flat,
)
from ..env.reward import CAPACITY_THRESHOLD_MBPS

# Any seeing drone outranks any non-seeing one. Larger than the map diagonal, so
# it dominates every real distance without needing a branch.
_SEE_BONUS_M = 1.0e5
_BIG_M = 1.0e6


@dataclass(frozen=True)
class B0Config:
    """B0's tunable constants -- the whole of its tuning budget.

    Swept on the TRAINING route split (ids 0..1791) and reported on the held-out
    evaluation split; `docs/BLOCK_E.md` §3. A baseline tuned on its own test set
    is not a baseline. Defaults here are the pre-sweep starting values, reasoned
    from the measurements named beside each one.
    """

    # Hop reach: separation, in metres, that one hop is expected to cover. Sets
    # how many drones go on the chain, K = ceil(sep/L) - 1. Starting value from
    # the escalation table (ENVIRONMENT.md): 2 hops by ~1000 m, 3 by ~1400 m.
    hop_reach_m: float = 520.0
    # Observer leads the target by this many seconds of its believed velocity.
    lead_s: float = 2.0
    # Spare observation posts: range from the believed target, and the bearings
    # they fan to either side of its direction of travel.
    spare_radius_m: float = 120.0
    spare_bearings_deg: tuple[float, ...] = (0.0, 55.0, -55.0, 110.0, -110.0)
    # How many drones may be held OFF the chain as spare observers. Default 0:
    # measured at N=5, the routing DP uses up to four hops, so every drone the
    # chain can have is worth more than a redundant observation post. The sweep
    # revisits it -- this is a measured default, not a assumed one.
    max_spares: int = 0
    # Local link repair: a relay slides perpendicular to the chain, hill-climbing
    # on the clearance it can observe. Bounded, or it stops being a relay.
    # 200 m, from the sweep: 53.7 % / 58.7 % / 60.7 % at 0 / 60 / 200 m on the
    # training split, 5 seeds. The 0-vs-200 gap (+7.0 pp) is the largest effect
    # any B0 constant has. Consequence worth stating: with a 200 m lateral
    # search B0 no longer places relays ON the MCV->HVT geodesic, it places them
    # NEAR it and then hunts for a clear line -- which is exactly the difference
    # between the `b0` rung and the `geodesic` rung.
    repair_amplitude_m: float = 200.0
    repair_step_m: float = 8.0
    repair_settle_m: float = 60.0  # only climb once roughly on station
    # What the hill climb maximises. "clearance" is signed metres of roofline
    # margin; "capacity" is the link rate itself, which is what actually decides
    # the mission -- a long clear hop can carry less than a short blocked one, so
    # clearance is only a proxy. Both are in the observation, so neither costs
    # information. Swept, not assumed.
    repair_score: str = "capacity"
    # Relays also slide ALONG the chain, not only across it. Equal geometric
    # spacing is not equal-capacity spacing once buildings are in the way, and
    # the chain's rate is set by its worst hop -- so the right station is the one
    # that balances the two adjacent links, which is what this searches for.
    repair_along: float = 0.0  # fraction of the belief->MCV span, 0 disables
    # Acquisition fan half-angle, about the MCV->cue bearing. Block D measured
    # that spreading, not knowing the direction, is what makes search fast.
    fan_half_angle_deg: float = 50.0
    # Velocity servo.
    gain_per_s: float = 0.5
    dash_range_m: float = 150.0  # sprint when this far from station


VARIANTS: tuple[str, ...] = ("geodesic", "b0", "oracle")


class B0Policy:
    """Batched scripted controller. `(B, N, 108)` observation -> `(B, N, 3)`.

    Runs on device inside the same loop the env does: no `.item()`, no `.cpu()`,
    no Python loop over environments, no data-dependent shapes.
    """

    def __init__(
        self,
        num_envs: int,
        num_drones: int,
        variant: str = "b0",
        device: torch.device | str = "cpu",
        cfg: B0Config | None = None,
        dt_s: float = DT_S,
        **overrides: float,
    ):
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        self.variant = variant
        self.cfg = replace(cfg or B0Config(), **overrides) if overrides else (cfg or B0Config())
        self.num_envs, self.num_drones = num_envs, num_drones
        self.dt = dt_s
        dev = torch.device(device)
        self.device = dev

        b, n, k = num_envs, num_drones, 7  # N_MAX - 1
        self.own_idx = torch.arange(n, device=dev).view(1, n)

        # Global index of each neighbour slot. Padded slots get an index above
        # every real one so they can never win an index tie-break.
        nb_idx = torch.full((n, k), float(n + 100), device=dev)
        if n > 1:
            nb_idx[:, : n - 1] = neighbour_index_table(n, dev).float()
        self.nb_idx = nb_idx.unsqueeze(0)  # (1, N, 7)

        ang = torch.tensor(
            [math.radians(a) for a in self.cfg.spare_bearings_deg], device=dev
        )  # (S,)
        self.spare_cos, self.spare_sin = ang.cos(), ang.sin()

        # Carried state. Cleared per environment by `reset`.
        self.belief_rel = torch.zeros(b, n, 3, device=dev)
        self.belief_vel = torch.zeros(b, n, 3, device=dev)
        self.informed = torch.zeros(b, n, dtype=torch.bool, device=dev)
        self.started = torch.zeros(b, n, dtype=torch.bool, device=dev)
        self.lat_m = torch.zeros(b, n, device=dev)
        self.lat_dir = torch.ones(b, n, device=dev)
        self.along = torch.zeros(b, n, device=dev)
        self.along_dir = torch.ones(b, n, device=dev)
        self.prev_score = torch.full((b, n), -_BIG_M, device=dev)

    # ------------------------------------------------------------------ #

    def reset(self, mask: Tensor | None = None) -> None:
        """Clear carried state where `mask` (default: everywhere).

        **Must be called on every auto-reset.** Belief, roles and repair offsets
        otherwise survive an episode boundary, and every new episode then starts
        with a belief pointing at the previous route's target -- silent, and the
        same class of bug as Block D's stale potential.
        """
        if mask is None:
            mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        m1 = mask.view(-1, 1)
        m2 = m1.unsqueeze(-1)
        z = torch.zeros_like(self.belief_rel)
        self.belief_rel = torch.where(m2, z, self.belief_rel)
        self.belief_vel = torch.where(m2, z, self.belief_vel)
        self.informed = torch.where(m1, torch.zeros_like(self.informed), self.informed)
        self.started = torch.where(m1, torch.zeros_like(self.started), self.started)
        self.lat_m = torch.where(m1, torch.zeros_like(self.lat_m), self.lat_m)
        self.lat_dir = torch.where(m1, torch.ones_like(self.lat_dir), self.lat_dir)
        self.along = torch.where(m1, torch.zeros_like(self.along), self.along)
        self.along_dir = torch.where(m1, torch.ones_like(self.along_dir), self.along_dir)
        self.prev_score = torch.where(
            m1, torch.full_like(self.prev_score, -_BIG_M), self.prev_score
        )

    # ------------------------------------------------------------------ #

    def act(self, flat: Tensor, truth: dict[str, Tensor] | None = None) -> Tensor:
        """One control step. `flat` is `(B, N, 108)`; returns `(B, N, 3)`.

        `truth` is accepted **only** by the `oracle` variant and must carry
        `hvt_rel` and `hvt_vel`, both `(B, N, 3)` in metres and m/s. Passing it
        to any other variant raises, so the like-for-like B0 cannot quietly
        acquire ground truth during a tuning session.
        """
        if truth is not None and self.variant != "oracle":
            raise ValueError(
                f"variant {self.variant!r} is the like-for-like B0 and must not be given "
                "ground truth; use variant='oracle' if that is what you meant"
            )
        if self.variant == "oracle" and truth is None:
            raise ValueError("variant='oracle' requires truth={'hvt_rel':..., 'hvt_vel':...}")

        o = unpack_flat(flat)
        ego, nb, edge, valid = o["ego"], o["neighbour"], o["edge"], o["valid"] > 0.5

        own_vel = ego[..., 0:3] * VEL_SCALE_MS
        own_alt = ALT_MIN_M + ego[..., 3] * (ALT_MAX_M - ALT_MIN_M)
        cue_rel = ego[..., 4:7] * POS_SCALE_M
        hvt_rel = ego[..., 9:12] * POS_SCALE_M
        hvt_relvel = ego[..., 12:15] * VEL_SCALE_MS
        mcv_rel = ego[..., 15:18] * POS_SCALE_M
        clr_mcv = ego[..., 20] * CLEARANCE_CLAMP_M
        # The env zeroes the HVT block when the drone does not see it, and the
        # z component is never zero when it does (78.5 m of altitude separation),
        # so this recovers the hard `sees` gate exactly.
        sees = hvt_rel.abs().amax(-1) > 0.0

        nb_rel = nb[..., 0:3] * POS_SCALE_M
        nb_relvel = nb[..., 3:6] * VEL_SCALE_MS
        nb_sees = (nb[..., 7] > 0.5) & valid
        nb_onpath = (nb[..., 8] > 0.5) & valid
        edge_clr = edge[..., 1] * CLEARANCE_CLAMP_M
        edge_cap = edge[..., 0] * CAPACITY_THRESHOLD_MBPS  # obs stores it in threshold units

        self._update_belief(cue_rel, own_vel, hvt_rel, hvt_relvel, sees, nb_rel, nb_relvel, nb_sees)
        if self.variant == "oracle":
            self.belief_rel = truth["hvt_rel"].to(self.belief_rel.dtype)
            self.belief_vel = truth["hvt_vel"].to(self.belief_vel.dtype)
            self.informed = torch.ones_like(self.informed)

        rank, n_relay = self._roles(sees, nb_sees, nb_rel, mcv_rel, valid)
        station = self._station(rank, n_relay, mcv_rel, cue_rel)

        # Local link repair, for next step: hill-climb the lateral offset on the
        # clearance this drone can actually observe. Only once on station, or it
        # climbs while still in transit and the signal is meaningless.
        self._update_repair(station, clr_mcv, edge_clr, edge_cap, nb_onpath, rank, n_relay)

        delta = torch.cat([station, (ALT_MAX_M - own_alt).unsqueeze(-1)], dim=-1)
        return self._servo(delta, own_vel)

    # ------------------------------------------------------------------ #
    # Target belief
    # ------------------------------------------------------------------ #

    def _update_belief(
        self,
        cue_rel: Tensor,
        own_vel: Tensor,
        hvt_rel: Tensor,
        hvt_relvel: Tensor,
        sees: Tensor,
        nb_rel: Tensor,
        nb_relvel: Tensor,
        nb_sees: Tensor,
    ) -> None:
        """Per-drone target estimate, from the observation and memory only.

        own sighting  -> exact fix
        neighbour bit -> that neighbour's position. Loose (the observation
                         envelope is ~43 m across-street but hundreds along it,
                         `docs/BLOCK_B.md`) and loose in the direction that does
                         not matter, since a relay needs the *bearing* to the
                         observer, not the target.
        neither       -> dead reckon: advance by the believed target velocity and
                         subtract this drone's own displacement.
        """
        # First step of an episode: the cue is the only thing there is, and it is
        # persistent in the observation, so this is a genuine sensor input.
        start = ~self.started
        self.belief_rel = torch.where(start.unsqueeze(-1), cue_rel, self.belief_rel)
        self.belief_vel = torch.where(
            start.unsqueeze(-1), torch.zeros_like(self.belief_vel), self.belief_vel
        )
        self.started = torch.ones_like(self.started)

        pred = self.belief_rel + (self.belief_vel - own_vel) * self.dt

        # Pick the observing neighbour most consistent with the current belief.
        # Non-observers are pushed out of the argmin rather than indexed out, so
        # the shape stays static.
        dist = (nb_rel - pred.unsqueeze(2)).norm(dim=-1)
        dist = torch.where(nb_sees, dist, torch.full_like(dist, _BIG_M))
        pick = dist.argmin(dim=-1, keepdim=True)  # (B, N, 1)
        any_nb = nb_sees.any(dim=-1)
        nb_fix = nb_rel.gather(2, pick.unsqueeze(-1).expand(-1, -1, -1, 3)).squeeze(2)
        nb_vel = own_vel + nb_relvel.gather(2, pick.unsqueeze(-1).expand(-1, -1, -1, 3)).squeeze(2)

        rel = torch.where(any_nb.unsqueeze(-1), nb_fix, pred)
        vel = torch.where(any_nb.unsqueeze(-1), nb_vel, self.belief_vel)
        self.belief_rel = torch.where(sees.unsqueeze(-1), hvt_rel, rel)
        self.belief_vel = torch.where(sees.unsqueeze(-1), own_vel + hvt_relvel, vel)
        self.informed = self.informed | sees | any_nb

    # ------------------------------------------------------------------ #
    # Roles
    # ------------------------------------------------------------------ #

    def _roles(
        self, sees: Tensor, nb_sees: Tensor, nb_rel: Tensor, mcv_rel: Tensor, valid: Tensor
    ) -> tuple[Tensor, Tensor]:
        """`(rank, n_relay)`. Rank 0 observes; 1..K relay; the rest are spares.

        **Only the observer role is dynamic.** It goes to the drone nearest the
        believed target, with any seeing drone outranking any non-seeing one --
        which is what supplies hysteresis, since the incumbent is the one that
        can see and the role therefore follows sight rather than chattering when
        two drones swap places by a metre. Every other drone takes a station in
        index order, skipping whichever index is observing.

        > ⚠️ Ranking *everyone* by distance to the target was tried first and is
        > wrong, in a way that only a rollout shows: the station layout is
        > observer (0 m), spares (~120 m), relays (~430 m, ~870 m), so a
        > distance ordering hands relay duty to whichever drones happen to be
        > near the target. They fly out, stop seeing, and their rank flips back.
        > The roles chatter, no station is ever held, and the chain never forms
        > -- 55 % mission-capable against 94 % for the same controller with
        > fixed assignments. Keeping the assignment stable and letting only the
        > observer move is both more realistic and much stronger.

        Consistency needs no extra communication: the sees bits are shared
        exactly, the observer's own fix anchors everyone's belief, and each drone
        can locate every other from the neighbour block.
        """
        n = self.num_drones
        if self.variant == "geodesic":
            # THESIS_PLAN's sketch: roles fixed by index, every non-observer on
            # the chain, so there are no spare observation posts to allocate.
            rank = self.own_idx.expand(self.num_envs, n)
            return rank, torch.full_like(rank, max(n - 1, 0))

        d_self = self.belief_rel.norm(dim=-1) - _SEE_BONUS_M * sees.float()
        d_nb = (nb_rel - self.belief_rel.unsqueeze(2)).norm(dim=-1)
        d_nb = torch.where(
            valid, d_nb - _SEE_BONUS_M * nb_sees.float(), torch.full_like(d_nb, _BIG_M)
        )

        best = d_nb.argmin(dim=-1, keepdim=True)
        d_best = d_nb.gather(-1, best).squeeze(-1)
        idx_best = self.nb_idx.expand_as(d_nb).gather(-1, best).squeeze(-1)

        own = self.own_idx.expand_as(d_self).float()
        i_observe = (d_self < d_best - 1e-4) | (
            ((d_self - d_best).abs() <= 1e-4) & (own < idx_best)
        )
        obs_idx = torch.where(i_observe, own, idx_best)
        # Non-observers take stations 1..N-1 in index order, closing the gap the
        # observer leaves. Stable under a handoff: each drone shifts by at most
        # one station rather than the whole assignment re-sorting.
        rank = torch.where(i_observe, torch.zeros_like(own), own + 1.0 - (own > obs_idx).float())

        # How many drones the chain needs, from the separation this drone
        # believes in, so the escalation is automatic. Capped at N-1 -- every
        # non-observer -- not N-2: the routing DP picks the best sub-chain from
        # whatever geometry exists, so an extra drone on the line is never worse
        # than one off it, and at N=5 the chain uses all four (docs/BLOCK_E.md).
        sep = (mcv_rel[..., :2] - self.belief_rel[..., :2]).norm(dim=-1)
        max_relay = max(n - 1 - self.cfg.max_spares, 0)
        n_relay = ((sep / self.cfg.hop_reach_m).ceil().long() - 1).clamp(0, max_relay)
        if self.cfg.max_spares == 0:
            n_relay = torch.full_like(n_relay, max_relay)
        return rank.long(), n_relay

    # ------------------------------------------------------------------ #
    # Stations
    # ------------------------------------------------------------------ #

    def _station(self, rank: Tensor, n_relay: Tensor, mcv_rel: Tensor, cue_rel: Tensor) -> Tensor:
        """Horizontal vector from each drone to where it should be, `(B, N, 2)`."""
        cfg = self.cfg
        belief = self.belief_rel[..., :2]
        mcv = mcv_rel[..., :2]
        to_mcv = mcv - belief
        sep = to_mcv.norm(dim=-1, keepdim=True).clamp_min(1.0)
        u = to_mcv / sep
        perp = torch.stack([-u[..., 1], u[..., 0]], dim=-1)

        # --- observer: directly over the believed target, led by its velocity.
        # Measured to be worth 40.2 % -> 92.6 % mission-capable on its own
        # (docs/BLOCK_E.md), which makes it the highest-leverage line in B0.
        lead = 0.0 if self.variant == "geodesic" else cfg.lead_s
        observer = belief + self.belief_vel[..., :2] * lead

        # --- relays: equally spaced along the geodesic, plus the repair offset.
        frac = (rank.float() / (n_relay.float() + 1.0)).unsqueeze(-1)
        frac = (frac + self.along.unsqueeze(-1)).clamp(0.05, 0.95)
        relay = belief + to_mcv * frac + perp * self.lat_m.unsqueeze(-1)

        # --- spares: alternative observation posts, fanned about the direction
        # the target is travelling. Observation binds and the link does not, so
        # a spare is worth more watching where the target is going than adding
        # redundancy to a chain that already closes.
        vel_xy = self.belief_vel[..., :2]
        moving = vel_xy.norm(dim=-1, keepdim=True) > 1.0
        head = torch.where(moving, vel_xy / vel_xy.norm(dim=-1, keepdim=True).clamp_min(1e-6), -u)
        s = (rank - n_relay - 1).clamp(0, len(cfg.spare_bearings_deg) - 1)
        cos_s, sin_s = self.spare_cos[s].unsqueeze(-1), self.spare_sin[s].unsqueeze(-1)
        rot = torch.cat(
            [
                head[..., :1] * cos_s - head[..., 1:] * sin_s,
                head[..., :1] * sin_s + head[..., 1:] * cos_s,
            ],
            dim=-1,
        )
        spare = belief + rot * cfg.spare_radius_m

        is_obs = (rank == 0).unsqueeze(-1)
        is_relay = ((rank > 0) & (rank <= n_relay)).unsqueeze(-1)
        station = torch.where(is_obs, observer, torch.where(is_relay, relay, spare))

        # --- acquisition: before anyone has reported a sighting, fly a radial
        # fan about the MCV->cue bearing. Block D measured 100 % acquisition at
        # t50 = 8 s for a fan against 59 % for a single shared bearing -- what
        # matters is spreading out, not knowing the direction.
        if self.variant != "geodesic":
            cue = cue_rel[..., :2]
            ray = cue - mcv
            half = math.radians(cfg.fan_half_angle_deg)
            n = max(self.num_drones - 1, 1)
            offs = (self.own_idx.float() - (self.num_drones - 1) / 2.0) * (2.0 * half / n)
            c, s_ = offs.cos().unsqueeze(-1), offs.sin().unsqueeze(-1)
            fan = mcv + torch.cat(
                [ray[..., :1] * c - ray[..., 1:] * s_, ray[..., :1] * s_ + ray[..., 1:] * c],
                dim=-1,
            )
            station = torch.where(self.informed.unsqueeze(-1), station, fan)
        return station

    # ------------------------------------------------------------------ #
    # Local link repair
    # ------------------------------------------------------------------ #

    def _update_repair(
        self,
        station: Tensor,
        clr_mcv: Tensor,
        edge_clr: Tensor,
        edge_cap: Tensor,
        nb_onpath: Tensor,
        rank: Tensor,
        n_relay: Tensor,
    ) -> None:
        """One step of a 1-D hill climb on observable clearance.

        A relay sees the signed clearance of its own links in metres -- to the
        MCV in the ego block, to each neighbour in the edge block. It slides
        perpendicular to the chain and keeps going while the worst of those
        improves, reversing when it does not. Gradient-free, batched, and the
        only part of B0 aimed at `chain_occluded` rather than at sightlines.

        Clearance is clamped at +-150 m by the observation contract, so a fully
        clear chain saturates and the climb stalls -- which is correct: there is
        nothing to repair.
        """
        if self.variant == "geodesic":
            return
        cfg = self.cfg
        if cfg.repair_score == "capacity":
            # The bottleneck of the links this drone carries. Capacity saturates
            # far above the bar, so the climb naturally stops pushing once the
            # hop is comfortable and spends its effort on the marginal ones.
            chain = torch.where(nb_onpath, edge_cap, torch.full_like(edge_cap, _BIG_M))
            score = chain.amin(dim=-1).clamp(max=_BIG_M)
        else:
            chain_clr = torch.where(nb_onpath, edge_clr, torch.full_like(edge_clr, _BIG_M))
            score = torch.minimum(clr_mcv, chain_clr.amin(dim=-1))

        is_relay = (rank > 0) & (rank <= n_relay)
        settled = station.norm(dim=-1) < cfg.repair_settle_m
        active = is_relay & settled

        improved = score > self.prev_score + 1e-3
        direction = torch.where(improved, self.lat_dir, -self.lat_dir)
        moved = (self.lat_m + direction * cfg.repair_step_m).clamp(
            -cfg.repair_amplitude_m, cfg.repair_amplitude_m
        )
        self.lat_dir = torch.where(active, direction, self.lat_dir)
        if cfg.repair_along > 0.0:
            a_dir = torch.where(improved, self.along_dir, -self.along_dir)
            a_new = (self.along + a_dir * cfg.repair_along * 0.25).clamp(
                -cfg.repair_along, cfg.repair_along
            )
            self.along_dir = torch.where(active, a_dir, self.along_dir)
            self.along = torch.where(
                active, a_new, torch.where(is_relay, self.along, torch.zeros_like(self.along))
            )
        self.lat_m = torch.where(
            active, moved, torch.where(is_relay, self.lat_m, torch.zeros_like(self.lat_m))
        )
        self.prev_score = torch.where(active, score, self.prev_score)

    # ------------------------------------------------------------------ #

    def _servo(self, delta: Tensor, own_vel: Tensor) -> Tensor:
        """Proportional velocity servo -> normalised acceleration command.

        The Block D waypoint harness used a position-P with velocity damping and
        never approached the speed envelope; commanding a *velocity* and letting
        the accel limit do the rest is what uses the 20 m/s cruise and 25 m/s
        dash the airframe actually has.
        """
        cfg = self.cfg
        want = delta * cfg.gain_per_s
        far = delta.norm(dim=-1, keepdim=True) > cfg.dash_range_m
        v_max = torch.where(
            far,
            torch.full_like(far, DRONE_DASH_MS, dtype=delta.dtype),
            torch.full_like(far, DRONE_CRUISE_MS, dtype=delta.dtype),
        )
        speed = want.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        want = want * (v_max / speed).clamp(max=1.0)
        return ((want - own_vel) / (MAX_ACCEL_MS2 * self.dt)).clamp(-1.0, 1.0)
