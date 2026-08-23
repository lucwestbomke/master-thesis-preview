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

        if on_reset is not None:
            # Unconditional: a masked reset is a no-op, and `if done.any()`
            # would be a host sync in disguise.
            on_reset(terminated | truncated)

    t_f = float(steps)
    late_f = float(steps - late_from)
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
