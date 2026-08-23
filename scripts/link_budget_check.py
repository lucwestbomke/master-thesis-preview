"""
Scenario sizing tool: is the mission trivially easy, properly contested, or
impossible, for a given (map size, Ptx ceiling, bandwidth, rate target)?

Motivation
----------
The relay chain must be *geometrically necessary*. If one drone can observe the
HVT and still close a link to the MCV across the whole map, there is no
multi-hop problem, no role differentiation, and nothing for a MARL policy to
learn that a single-agent policy could not.

That is a link-budget question with a numeric answer, so it should be computed
rather than argued. Run this before committing to an operating area, and again
whenever the Ptx ceiling, bandwidth, carrier or rate target changes.

    uv run python scripts/link_budget_check.py

The trade-off table it produces is pinned by tests/test_scenario_sizing.py so a
later parameter change cannot silently make the mission trivial again.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from env.channel import (
    capacity_mbps,
    dbm_to_mw,
    fspl_db,
    noise_floor_dbm,
    pathloss_a2g_umi_av_db,
)
from env.reward import CAPACITY_THRESHOLD_MBPS

FC_GHZ = 3.5
BANDWIDTH_HZ = 10e6
NOISE_FIGURE_DB = 7.0
DRONE_ALT_M = 80.0
BLOCKAGE_DB = 20.0


def _t(x: float) -> torch.Tensor:
    return torch.tensor([float(x)])


def path_loss(distance_m: float, kind: str, alt_m: float = DRONE_ALT_M) -> torch.Tensor:
    """`kind` in {a2g_los, a2g_nlos, a2a_los, a2a_blocked}."""
    d, h = _t(distance_m), _t(alt_m)
    if kind == "a2g_los":
        return pathloss_a2g_umi_av_db(d, h, torch.tensor([True]), FC_GHZ)
    if kind == "a2g_nlos":
        return pathloss_a2g_umi_av_db(d, h, torch.tensor([False]), FC_GHZ)
    if kind == "a2a_los":
        return fspl_db(d, FC_GHZ)
    if kind == "a2a_blocked":
        return fspl_db(d, FC_GHZ) + BLOCKAGE_DB
    raise ValueError(kind)


def link_capacity_mbps(
    ptx_dbm: float,
    distance_m: float,
    kind: str,
    jammer_dbm: float | None = None,
    jammer_distance_m: float | None = None,
    alt_m: float = DRONE_ALT_M,
) -> float:
    """Single-link rate, noise-limited unless a jammer is supplied."""
    n0 = noise_floor_dbm(BANDWIDTH_HZ, NOISE_FIGURE_DB)
    prx = ptx_dbm - path_loss(distance_m, kind, alt_m)

    denom_mw = dbm_to_mw(torch.tensor(n0))
    if jammer_dbm is not None:
        jam_pl = path_loss(jammer_distance_m, "a2g_los", alt_m)
        denom_mw = denom_mw + dbm_to_mw(jammer_dbm - jam_pl)

    sinr = 10.0 * torch.log10(dbm_to_mw(prx) / denom_mw)
    return capacity_mbps(sinr, BANDWIDTH_HZ).item()


def max_range_m(ptx_dbm: float, target_mbps: float, kind: str) -> float:
    """Largest distance still delivering `target_mbps`. Bisection, noise-limited."""
    lo, hi = 1.0, 100_000.0
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        if link_capacity_mbps(ptx_dbm, mid, kind) >= target_mbps:
            lo = mid
        else:
            hi = mid
    return lo


def classify(
    map_size_m: float,
    ptx_dbm: float,
    threshold_mbps: float = CAPACITY_THRESHOLD_MBPS,
    hops: int = 3,
    reuse_limit: int = 3,
) -> tuple[str, float, float]:
    """Verdict for one (map, Ptx) design point.

    A chain of `hops` hops pays a `min(hops, reuse_limit)` divisor, so each hop
    must individually carry `threshold * divisor`.
    """
    per_hop_needed = threshold_mbps * min(hops, reuse_limit)
    single = link_capacity_mbps(ptx_dbm, map_size_m, "a2g_nlos")
    relayed = link_capacity_mbps(ptx_dbm, map_size_m / hops, "a2a_blocked")

    if single >= threshold_mbps:
        verdict = "TRIVIAL - one drone spans the map, no relay needed"
    elif relayed < per_hop_needed:
        verdict = "INFEASIBLE - even the relay chain cannot close"
    else:
        verdict = "CONTESTED - relay chain required and achievable"
    return verdict, single, relayed


def main() -> None:
    n0 = noise_floor_dbm(BANDWIDTH_HZ, NOISE_FIGURE_DB)
    print(
        f"fc={FC_GHZ} GHz  B={BANDWIDTH_HZ / 1e6:.0f} MHz  "
        f"N0={n0:.1f} dBm  drone alt={DRONE_ALT_M:.0f} m\n"
    )

    print("=" * 78)
    print("MAX LINK RANGE (m) -- noise-limited, no jammer")
    print("=" * 78)
    print(
        f"{'Ptx dBm':>8} {'A2G NLoS 5M':>13} {'A2G NLoS 15M':>13} "
        f"{'A2A LoS 5M':>12} {'A2A blk 15M':>13}"
    )
    for ptx in (0, 10, 20, 30, 40):
        print(
            f"{ptx:>8} {max_range_m(ptx, 5, 'a2g_nlos'):>13.0f} "
            f"{max_range_m(ptx, 15, 'a2g_nlos'):>13.0f} "
            f"{max_range_m(ptx, 5, 'a2a_los'):>12.0f} "
            f"{max_range_m(ptx, 15, 'a2a_blocked'):>13.0f}"
        )

    print()
    print("=" * 78)
    print(f"SCENARIO VERDICT -- 3-hop chain, {CAPACITY_THRESHOLD_MBPS:.0f} Mbps end-to-end target")
    print("=" * 78)
    print(f"{'map m':>7} {'Ptx':>5} {'1-hop':>8} {'per-hop':>9}  verdict")
    for map_size in (300, 600, 1200, 2000):
        for ptx in (10, 20, 30, 40):
            verdict, single, relayed = classify(map_size, ptx)
            print(f"{map_size:>7} {ptx:>5} {single:>8.1f} {relayed:>9.1f}  {verdict}")
        print()

    print("=" * 78)
    print("JAMMER BITE -- observer 150 m from the HVT, Ptx = 20 dBm")
    print("=" * 78)
    for jam in (20, 30, 40):
        for d in (100, 200, 400):
            clean = link_capacity_mbps(20, d, "a2a_los")
            jammed = link_capacity_mbps(20, d, "a2a_los", jammer_dbm=jam, jammer_distance_m=150)
            print(
                f"  jammer {jam:>2} dBm, observer->relay {d:>3} m: "
                f"{clean:>6.1f} -> {jammed:>6.1f} Mbps"
            )


if __name__ == "__main__":
    main()
