"""Rollout harness and the pre-registered metrics -- Block E.

Every policy goes through this one code path: `random`, the Block D waypoint
harness, the three B0 rungs, and (in Block G) a trained checkpoint. That is the
point -- numbers produced by two different loops are not comparable, and the
comparison is the entire deliverable.

Metrics are `docs/THESIS_PLAN.md` §4, pre-registered before any results were
seen. Three groups:

  mission       mission-capable fraction (the headline, and the dominant reward
                term by construction), observed, link-alive, capacity quantiles
  attribution   chain-occluded, the full hop histogram, and the rate-division
                counterfactual that answers Block D's open question
  behavioural   observer identity over time -> handoff rate, coverage gap,
                anticipation lead time (RQ3)

Everything accumulates **on device**; only the final reduction crosses to the
host. `docs/AGENTS.md` forbids `.item()` in the hot loop and this is one.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch import Tensor

from ..env import routing
from ..env.core import ALT_MAX_M, N_MAX, BatchedSwarmEnv
from ..env.reward import CAPACITY_THRESHOLD_MBPS

Policy = Callable[[dict[str, Tensor]], Tensor]

# Chain lengths are 0..N+1; the divisor saturates at `reuse_limit`, so 4- and
# 5-hop chains are charged exactly like 3-hop ones. Block D counted only the
# exactly-3 cell and the worry that produced was an artefact of that.
MAX_HOPS_TRACKED = 8


@dataclass
class RolloutMetrics:
    """Per-episode metrics, `(n_episodes,)` on the host. Reduce with `summary`."""

    mission_capable: Tensor
    observed: Tensor
    link_alive: Tensor
    chain_occluded: Tensor
    capacity_mean: Tensor
    capacity_p5: Tensor
    altitude_mean: Tensor
    battery_end: Tensor
    battery_var_end: Tensor
    episode_return: Tensor
    # attribution
    hop_hist: Tensor  # (n_episodes, MAX_HOPS_TRACKED) share of steps
    hop_hist_last_third: Tensor
    capable_no_division: Tensor  # mission-capable with reuse_limit = 1
    capable_strict_tdma: Tensor  # mission-capable with reuse_limit = max_hops
    bottleneck_mbps: Tensor  # median capacity of the chain's worst link
    bottleneck_marginal: Tensor  # share of chain-steps the divisor actually flips
    # behavioural, RQ3 -- observer handoff (the rare half)
    handoffs: Tensor  # count per episode
    handoff_gap_steps: Tensor  # mean steps uncovered across a handoff
    anticipation_steps: Tensor  # mean lead of the successor over the incumbent
    # behavioural, RQ3 -- chain re-rooting (the abundant half). Measured in
    # Block E at ~51 per episode against ~1 observer handoff, which is why the
    # RQ is pointed here: this is the role dynamic the environment actually
    # forces, and it is forced by occlusion changing link quality -- the effect
    # the thesis is about.
    reroots: Tensor  # steps on which relay membership changed
    chain_churn: Tensor  # drone enter/leave events
    chain_compositions: Tensor  # distinct relay sets visited
    reroot_lead: Tensor  # steps a drone was link-viable BEFORE it was recruited
    # failure attribution: of the steps that were not mission-capable, why
    fail_no_observation: Tensor
    fail_link: Tensor
    # --- role emergence -----------------------------------------------------
    #
    # Added 2026-08-25. Nothing in this harness could distinguish "one drone
    # committed to observing and four relayed behind it" from "five drones did
    # the same mediocre thing at the same radius" -- and that is precisely the
    # difference the Block G diagnosis turns on. Measured: given a sightline the
    # GNN converts it as well as B0 does (0.620 vs 0.617), but conditioned on
    # observing its chain is indistinguishable from a RANDOM policy's (1.91 hops
    # vs 1.83, B0 2.26). The swarm learned to fly at the target and nothing about
    # relaying. These four say whether that is a role failure.
    #
    # ⛔ These are DIAGNOSTICS, computed by the evaluator from `env.drone_pos`.
    # That is allowed here and forbidden in `b0.py` -- B0 is a policy and must
    # see only `obs["flat"]`; the evaluator is a measuring instrument.
    observer_share_max: Tensor  # share of observing steps held by the top observer
    role_entropy: Tensor  # normalised entropy of observer identity, 0 = one drone owns it
    relay_entropy: Tensor  # same for chain membership
    standoff_gap_m: Tensor  # median drone-HVT range minus the closest drone's
    #: (n_episodes,) mean range of the drone actually HOLDING the sightline,
    #: over observing steps. ⚠️ **This is the number the Block G diagnosis rests
    #: on** -- B0 parks its observer at ~79 m and every learned policy loiters at
    #: ~291 m -- and until now it existed only as a one-off measurement rather
    #: than a reported metric. `nan` on an episode that never observed.
    observer_range_m: Tensor
    #: The last third of the episode, where `MODELS.md` says the difficulty is
    #: ("capable decays 84 % -> 35 % as the HVT drives out ... report the second
    #: half separately") and which nothing in Block G had ever reported. The
    #: coordination-trap hypothesis predicts the swarm is fine early and fails
    #: late, because the chain only has to extend once the HVT is far out.
    capable_last_third: Tensor
    observed_last_third: Tensor
    observer_range_last_third: Tensor
    #: (n_episodes,) mean distance from a drone to the MCV--HVT segment, over
    #: drones and steps. **The swarm's useful work happens near that line** --
    #: an observer at one end, relays strung along it -- so this is "is the
    #: swarm where the mission is". Rendering route 12 showed B0's tracks
    #: confined to the HVT corridor while the learned policy's sprawled across
    #: the whole map, including the half the HVT never enters; no aggregate in
    #: this block could see that.
    off_axis_m: Tensor
    #: Share of episodes scoring above 80 % / below 20 % mission-capable. ⚠️ The
    #: mean hides shape: the same policy that averages 40.7 % scored **80.8 %**
    #: on route 12, near B0's 97.2 %. These say whether it is uniformly mediocre
    #: or solves some routes outright and fails others.
    capable_share_high: Tensor
    capable_share_low: Tensor

    # ⚠️ `standoff_gap_m` MUST be read together with `role_entropy`, never alone.
    # A large gap means either "one drone went in and the rest held back" or
    # "the drones are scattered at random". Measured at stage 1, which separates
    # the three cases cleanly:
    #
    #     policy        role_entropy   standoff_gap_m   reading
    #     B0                0.0            88.6         one observer, rest held back
    #     MAPPO MLP         0.2            34.1         an observer emerges, nobody
    #                                                   drops back to relay
    #     random            0.5           163.7         no structure -- pure scatter
    #
    # So the learned policy is much closer to B0 than to random on WHO observes,
    # and closer to neither on WHERE the others go: it bunches. That is the
    # Block G diagnosis in two numbers -- **the observer role partly emerges and
    # the relay role does not** -- and it matches the hop evidence exactly
    # (1.9 hops conditioned on observing, against random's 1.83 and B0's 2.26).
    meta: dict = field(default_factory=dict)

    def summary(self) -> dict[str, float]:
        """Across-episode **means** -- this seed's point estimate for each metric.

        Deliberately not medians. AGENTS.md's "median + IQR, never mean +- std"
        governs how *seeds* are aggregated, and the caller does that. Taking a
        median here instead would silently report **0.0** for every rare-event
        metric -- `fail_link` is zero in most episodes, so its median is zero
        even when the mean is a real 1.3 % -- which is exactly the number that
        says whether the relay premise ever binds.
        """
        out: dict[str, float] = {}
        for name in (
            "mission_capable",
            "observed",
            "link_alive",
            "chain_occluded",
            "capacity_mean",
            "capacity_p5",
            "altitude_mean",
            "battery_end",
            "battery_var_end",
            "episode_return",
            "capable_no_division",
            "capable_strict_tdma",
            "bottleneck_mbps",
            "bottleneck_marginal",
            "handoffs",
            "handoff_gap_steps",
            "anticipation_steps",
            "reroots",
            "chain_churn",
            "chain_compositions",
            "reroot_lead",
            "fail_no_observation",
            "fail_link",
            "observer_share_max",
            "observer_range_m",
            "capable_last_third",
            "observed_last_third",
            "observer_range_last_third",
            "off_axis_m",
            "capable_share_high",
            "capable_share_low",
            "battery_end",
            "role_entropy",
            "relay_entropy",
            "standoff_gap_m",
        ):
            v = getattr(self, name)
            ok = torch.isfinite(v)
            out[name] = float(v[ok].mean()) if ok.any() else 0.0
        return out

    def hop_distribution(self, last_third: bool = False) -> Tensor:
        """Pooled hop histogram, `(MAX_HOPS_TRACKED,)`, summing to 1.

        Pooled across episodes rather than reduced element-wise: a per-bin
        median across seeds does **not** sum to one (medians of parts are not
        the median of the whole), which is how a "chain exists on 100.9 % of
        steps" line gets printed. Scalar summaries of the distribution --
        multi-hop share, saturated-divisor share -- are the things to take a
        median + IQR over.
        """
        h = self.hop_hist_last_third if last_third else self.hop_hist
        return h.mean(0)


@torch.no_grad()
def rollout(
    env: BatchedSwarmEnv,
    policy: Policy,
    steps: int,
    on_reset: Callable[[Tensor], None] | None = None,
    rate_division_counterfactual: bool = True,
) -> RolloutMetrics:
    """Run `steps` ticks of `env` under `policy`, one episode per environment.

    `env.cfg.auto_reset` must be off: an episode boundary mid-rollout would mix
    two routes into one row of metrics, and the RQ3 handoff series would record
    a spurious handoff at the seam. `steps` should be the episode length.

    `on_reset` is called with the done mask so a stateful policy can clear its
    carried state; B0 needs it, and forgetting it is the failure mode BLOCK_E
    §14 warns about.
    """
    if env.cfg.auto_reset:
        raise ValueError(
            "rollout needs auto_reset=False: an episode boundary mid-rollout mixes two "
            "routes into one metrics row and fakes a handoff at the seam"
        )
    b, n = env.cfg.num_envs, env.cfg.num_drones
    dev = env.device

    obs = env.reset()
    if on_reset is not None:
        on_reset(torch.ones(b, dtype=torch.bool, device=dev))

    acc = {
        k: torch.zeros(b, device=dev)
        for k in (
            "capable",
            "observed",
            "alive",
            "occluded",
            "cap_sum",
            "alt_sum",
            "ret",
            "cap_nodiv",
            "cap_tdma",
            "marginal",
            "chain_steps",
            "reroot",
            "churn",
            "join_lead",
            "joins",
            "fail_obs",
            "fail_link",
        )
    }
    hop_hist = torch.zeros(b, MAX_HOPS_TRACKED, device=dev)
    hop_hist_late = torch.zeros(b, MAX_HOPS_TRACKED, device=dev)
    cap_series = torch.zeros(b, steps, device=dev)
    bottleneck_series = torch.full((b, steps), float("nan"), device=dev)

    # RQ3 bookkeeping: who is observing, and since when.
    prev_obs_idx = torch.full((b,), -1, dtype=torch.long, device=dev)
    handoffs = torch.zeros(b, device=dev)
    gap_total = torch.zeros(b, device=dev)
    gap_run = torch.zeros(b, device=dev)
    lead_total = torch.zeros(b, device=dev)
    # How long each drone has been seeing the target. The successor's run length
    # at the moment of handoff IS the anticipation lead: it says the successor
    # had already acquired before the incumbent lost the target.
    see_run = torch.zeros(b, n, device=dev)
    # Relay-side bookkeeping. `viable` = not currently carrying the chain, but
    # holding a usable link to something that is -- i.e. standing by. A drone
    # recruited after a long viable run pre-positioned; one recruited the instant
    # it became viable reacted.
    prev_path = torch.zeros(b, n, dtype=torch.bool, device=dev)
    viable_run = torch.zeros(b, n, device=dev)
    # Role emergence. `obs_count[e, i]` = steps drone i was THE observer of
    # episode e; `path_count[e, i]` = steps it carried the chain.
    obs_count = torch.zeros(b, n, device=dev)
    path_count = torch.zeros(b, n, device=dev)
    standoff_sum = torch.zeros(b, device=dev)
    range_sum = torch.zeros(b, device=dev)
    off_axis_sum = torch.zeros(b, device=dev)
    range_late = torch.zeros(b, device=dev)
    covered_late = torch.zeros(b, device=dev)
    capable_late = torch.zeros(b, device=dev)
    observed_late = torch.zeros(b, device=dev)
    seen_comp = torch.zeros(b, 1 << N_MAX, device=dev)
    pow2 = (2 ** torch.arange(n, device=dev)).float()

    late_from = steps - steps // 3
    for t in range(steps):
        action = policy(obs)
        obs, rew, terminated, truncated, ex = env.step(action)

        capable = ex["mission_capable"].float()
        seen = ex["sees_any"].float()
        cap_mbps = ex["e2e_capacity_mbps"]
        alive = (cap_mbps >= CAPACITY_THRESHOLD_MBPS).float()

        acc["capable"] += capable
        acc["observed"] += seen
        acc["alive"] += alive
        acc["occluded"] += ex["chain_occluded"].float()
        acc["cap_sum"] += cap_mbps
        acc["alt_sum"] += ex["altitude_m"].mean(dim=-1)
        acc["ret"] += rew.mean(dim=-1)
        cap_series[:, t] = cap_mbps
        # Of the steps that failed, separate the two causes. They are not
        # symmetric: no observation means there is no feed at all, whereas a
        # link failure means the feed exists and cannot be delivered.
        acc["fail_obs"] += 1.0 - seen
        acc["fail_link"] += seen * (1.0 - alive)

        hops = ex["hop_count"].clamp(max=MAX_HOPS_TRACKED - 1)
        one_hot = torch.zeros(b, MAX_HOPS_TRACKED, device=dev)
        one_hot.scatter_(1, hops.unsqueeze(-1), 1.0)
        hop_hist += one_hot
        if t >= late_from:
            hop_hist_late += one_hot

        if rate_division_counterfactual:
            acc["cap_nodiv"] += _capable_at_reuse(env, ex, 1)
            acc["cap_tdma"] += _capable_at_reuse(env, ex, env.cfg.n_radio - 1)

        # The chain's worst link -- what the divisor actually divides. This is
        # what explains a null on the counterfactual above: the divisor can only
        # change the outcome for a chain whose bottleneck would clear the bar
        # undivided and fails once divided. That is the *exact* flip condition,
        # not the "bottleneck is in the 5-15 Mbps window" proxy it replaced --
        # the proxy counts 1-hop chains, whose divisor is 1, and so reports a
        # window that is populated while the flip rate is genuinely zero.
        on_edge = ex["on_edge"]
        link = torch.where(on_edge, ex["capacity_mbps"], torch.full_like(ex["capacity_mbps"], 1e9))
        worst = link.amin(dim=(-1, -2))
        has_chain = ex["hop_count"] > 0
        bottleneck_series[:, t] = torch.where(has_chain, worst, torch.nan)
        divisor = ex["hop_count"].clamp(min=1, max=env.cfg.reuse_limit).float()
        flips = (
            has_chain
            & (worst >= CAPACITY_THRESHOLD_MBPS)
            & (worst / divisor < CAPACITY_THRESHOLD_MBPS)
        )
        acc["marginal"] += flips.float()
        acc["chain_steps"] += has_chain.float()

        # --- RQ3: chain re-rooting, the abundant role dynamic --------------
        path = ex["on_path"][:, :n]
        joined = path & ~prev_path
        acc["reroot"] += (path != prev_path).any(dim=-1).float()
        acc["churn"] += (path != prev_path).float().sum(dim=-1)
        acc["join_lead"] += (joined.float() * viable_run).sum(dim=-1)
        acc["joins"] += joined.float().sum(dim=-1)
        seen_comp.scatter_(1, (path.float() * pow2).sum(-1, keepdim=True).long(), 1.0)
        # Link-viable: off the chain, but able to reach something on it.
        cap_to_path = torch.where(
            path.unsqueeze(1),
            ex["capacity_mbps"][:, :n, :n],
            torch.zeros_like(ex["capacity_mbps"][:, :n, :n]),
        ).amax(dim=-1)
        viable = ~path & (cap_to_path >= CAPACITY_THRESHOLD_MBPS)
        viable_run = torch.where(viable, viable_run + 1.0, torch.zeros_like(viable_run))
        prev_path = path

        # --- RQ3: observer identity, handoffs, anticipation ---------------
        sees = ex["sees_hvt"]
        see_run = torch.where(sees, see_run + 1.0, torch.zeros_like(see_run))
        # The observer is the longest-standing seer; ties go to the lower index,
        # which is what `argmax` on a descending-priority key gives.
        key = see_run + sees.float() * 1e6
        cur = torch.where(sees.any(dim=-1), key.argmax(dim=-1), torch.full_like(prev_obs_idx, -1))

        covered = sees.any(dim=-1)
        # The coverage gap belongs to the handoff that ENDS it, so read it
        # before this step's coverage resets the run.
        gap_before = gap_run
        gap_run = torch.where(covered, torch.zeros_like(gap_run), gap_run + 1.0)

        changed = (cur != prev_obs_idx) & (cur >= 0) & (prev_obs_idx >= 0)
        handoffs += changed.float()
        gap_total += changed.float() * gap_before
        # Lead time: how many steps the successor had ALREADY been observing when
        # it took over. The incumbent holds the role until it loses sight (it has
        # the longest run), so at the moment of handoff the successor's run is
        # exactly its head start. 0 = reactive, > 0 = anticipatory.
        succ_run = see_run.gather(1, cur.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        lead_total += changed.float() * (succ_run - 1.0).clamp_min(0.0)
        prev_obs_idx = torch.where(cur >= 0, cur, prev_obs_idx)

        # --- role emergence -------------------------------------------------
        # Reuses `cur` -- the same observer identity `handoffs` and the tenure
        # figure are built from -- so all of them agree by construction rather
        # than by coincidence. `covered` gates it: a step where nobody sees has
        # no observer to attribute.
        obs_count.scatter_add_(1, cur.clamp_min(0).unsqueeze(-1), covered.float().unsqueeze(-1))
        path_count += path.float()
        # How much further back the rest of the swarm sits than its closest
        # member. Differentiation shows here as a LARGE gap -- one drone in at
        # ~79 m and the others held back to relay -- while a clustered or
        # uniformly standing-off swarm shows a small one.
        d_hvt = (env.drone_pos - env.hvt_pos.unsqueeze(1)).norm(dim=-1)
        standoff_sum += d_hvt.median(dim=-1).values - d_hvt.min(dim=-1).values
        # The range of the drone that actually holds the ray -- B0 79 m, learned
        # ~291 m. Accumulated only over steps where somebody sees, so it is the
        # observer's stand-off and not an average over blind steps.
        # Distance from each drone to the MCV--HVT segment, clamped to the
        # segment rather than the infinite line so a drone beyond either end is
        # measured from that end.
        axis = env.hvt_pos - env.mcv_pos  # (B, 3)
        rel = env.drone_pos - env.mcv_pos.unsqueeze(1)  # (B, N, 3)
        t_par = (
            (rel * axis.unsqueeze(1)).sum(-1) / axis.pow(2).sum(-1, keepdim=True).clamp_min(1.0)
        ).clamp(0.0, 1.0)
        off_axis_sum += (rel - t_par.unsqueeze(-1) * axis.unsqueeze(1)).norm(dim=-1).mean(dim=-1)

        obs_range = d_hvt.gather(1, cur.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        range_sum += obs_range * covered.float()
        if t >= late_from:
            capable_late += capable
            observed_late += seen
            range_late += obs_range * covered.float()
            covered_late += covered.float()

        if on_reset is not None:
            # Unconditional: a masked reset is a no-op, and `if done.any()`
            # would be a host sync in disguise.
            on_reset(terminated | truncated)

    t_f = float(steps)
    late_f = float(steps - late_from)

    def _norm_entropy(counts: Tensor) -> Tensor:
        """Entropy of a per-drone share, divided by `log(n)` so it lands in [0, 1].

        **0 = one drone owns the role. 1 = every drone holds it equally**, which
        is the signature of no role at all. Normalising by `log(n)` is what makes
        it comparable across `N in {3, 5, 8}` -- without it the RQ2 transfer
        columns would be reporting swarm size rather than behaviour.
        """
        if n < 2:
            return torch.zeros_like(counts[:, 0])
        share = counts / counts.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        entropy = -(share * share.clamp_min(1e-12).log()).sum(dim=-1)
        return entropy / math.log(n)

    obs_total = obs_count.sum(dim=-1)

    def _mean_where(total: Tensor, count: Tensor) -> Tensor:
        """`nan` rather than 0 where nothing was observed -- `summary()` drops
        non-finite entries, so an episode that never saw the target contributes
        nothing instead of dragging the mean toward zero."""
        return torch.where(
            count > 0, total / count.clamp_min(1.0), torch.full_like(total, float("nan"))
        )

    return RolloutMetrics(
        mission_capable=(acc["capable"] / t_f).cpu(),
        observed=(acc["observed"] / t_f).cpu(),
        link_alive=(acc["alive"] / t_f).cpu(),
        chain_occluded=(acc["occluded"] / t_f).cpu(),
        capacity_mean=(acc["cap_sum"] / t_f).cpu(),
        capacity_p5=cap_series.quantile(0.05, dim=1).cpu(),
        altitude_mean=(acc["alt_sum"] / t_f).cpu(),
        battery_end=env.battery.mean(dim=-1).cpu(),
        battery_var_end=env.battery.var(dim=-1, unbiased=False).cpu(),
        episode_return=acc["ret"].cpu(),
        hop_hist=(hop_hist / t_f).cpu(),
        hop_hist_last_third=(hop_hist_late / late_f).cpu(),
        capable_no_division=(acc["cap_nodiv"] / t_f).cpu(),
        capable_strict_tdma=(acc["cap_tdma"] / t_f).cpu(),
        bottleneck_mbps=bottleneck_series.nanmedian(dim=1).values.cpu(),
        bottleneck_marginal=(acc["marginal"] / acc["chain_steps"].clamp_min(1)).cpu(),
        handoffs=handoffs.cpu(),
        handoff_gap_steps=(gap_total / handoffs.clamp_min(1)).cpu(),
        anticipation_steps=(lead_total / handoffs.clamp_min(1)).cpu(),
        reroots=acc["reroot"].cpu(),
        chain_churn=acc["churn"].cpu(),
        chain_compositions=seen_comp.sum(dim=-1).cpu(),
        reroot_lead=(acc["join_lead"] / acc["joins"].clamp_min(1)).cpu(),
        fail_no_observation=(acc["fail_obs"] / t_f).cpu(),
        fail_link=(acc["fail_link"] / t_f).cpu(),
        observer_share_max=torch.where(
            obs_total > 0,
            obs_count.max(dim=-1).values / obs_total.clamp_min(1e-9),
            torch.zeros_like(obs_total),
        ).cpu(),
        role_entropy=_norm_entropy(obs_count).cpu(),
        relay_entropy=_norm_entropy(path_count).cpu(),
        standoff_gap_m=(standoff_sum / t_f).cpu(),
        observer_range_m=_mean_where(range_sum, obs_total).cpu(),
        capable_last_third=(capable_late / late_f).cpu(),
        observed_last_third=(observed_late / late_f).cpu(),
        observer_range_last_third=_mean_where(range_late, covered_late).cpu(),
        off_axis_m=(off_axis_sum / t_f).cpu(),
        capable_share_high=(acc["capable"] / t_f > 0.8).float().cpu(),
        capable_share_low=(acc["capable"] / t_f < 0.2).float().cpu(),
        meta={"steps": steps, "num_envs": b, "num_drones": n, "alt_ceiling_m": ALT_MAX_M},
    )


def _capable_at_reuse(env: BatchedSwarmEnv, ex: dict[str, Tensor], reuse_limit: int) -> Tensor:
    """Mission-capable under a different half-duplex schedule.

    The direct measure of what F4's rate-division rung is worth, in the units the
    thesis reports -- see `docs/BLOCK_E.md` §6. Hop counts are a proxy for this;
    this is the thing itself. It is a **fixed-geometry** counterfactual: a
    scripted policy does not adapt to the divisor, so it says whether the rung
    has anything to act on, not how a policy would respond to it.

    `reuse_limit = 1` removes the penalty entirely and `= max_hops` recovers
    strict TDMA (`/n`); `routing.py` exposes both for exactly this. Reporting all
    three is also the duplexing robustness check `PHYSICS.md` asks for -- and the
    strict-TDMA arm is the sanity check on the other one: if *neither* direction
    moves the number, the counterfactual is broken rather than the rung being
    unimportant.
    """
    source = torch.cat([ex["sees_hvt"], torch.zeros_like(ex["sees_hvt"][:, :1])], dim=1)
    e2e = routing.best_relay_capacity(
        ex["capacity_mbps"],
        source,
        dst_index=env.mcv_idx,
        max_hops=env.cfg.n_radio - 1,
        reuse_limit=reuse_limit,
    )
    return (ex["sees_any"] & (e2e >= CAPACITY_THRESHOLD_MBPS)).float()
