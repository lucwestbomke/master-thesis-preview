"""
Derives the operating area from physical parameters, instead of picking it.

The map size is not a free choice. Once the radio, the threat and the sensor are
fixed from defensible sources, the operating area is whatever makes the problem
*well-posed*, which means three conditions must hold simultaneously:

  W1  A single drone cannot do the mission.       (else there is no swarm problem)
  W2  The swarm can do the mission.               (else nothing is learnable)
  W3  Non-uniform transmit power beats the best
      uniform transmit power by a measurable
      margin.                                     (else RQ1 has nothing to measure)

W3 is the one that decides whether the thesis has a subject. Note that under
*uniform* power in the interference-limited regime, SIR is independent of the
power level -- raising everyone's power raises nobody's SINR. So a motion-only
policy can manage interference only geometrically. W3 measures what that costs.

Run:  uv run python scripts/scenario_design.py
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from env.channel import (
    capacity_mbps,
    fspl_db,
    noise_floor_dbm,
    pairwise_distance_m,
    pathloss_a2g_umi_av_db,
    received_power_dbm,
    sinr_db,
)
from env.reward import CAPACITY_THRESHOLD_MBPS
from env.routing import best_relay_capacity

# --------------------------------------------------------------------------- #
# Parameters, each traceable to something outside this file
# --------------------------------------------------------------------------- #

FC_GHZ = 3.5  # S/C-band tactical MANET allocation
BANDWIDTH_HZ = 10e6  # single 10 MHz channel
NOISE_FIGURE_DB = 7.0  # typical COTS receiver
# The mission rate requirement, imported rather than restated. Three copies of
# this number existed before Block E raised it 5 -> 15 Mbps, and a stale copy
# here would silently re-derive the altitude ceiling against the old bar.
THRESHOLD_MBPS = CAPACITY_THRESHOLD_MBPS  # dual EO/IR feed, low latency

# Ptx ceiling: UAV-mounted tactical MANET radios (Silvus SC4200, Doodle Labs
# Helix, TrellisWare TW-950 class) transmit 0.5-2 W. 1 W = 30 dBm.
PTX_MAX_DBM = 30.0
PTX_LEVELS = [-300.0, 0.0, 10.0, 20.0, 30.0]  # -300 == transmitter off

# Jammer: vehicle-mounted C-UAS emitter, in-band power in our 10 MHz slice.
# A barrage jammer spreading tens of watts over several hundred MHz lands near
# 1 W in-band; a spot jammer would be higher. Swept below.
JAMMER_DBM = 30.0

FLIGHT_ALT_M = 80.0
MCV_ALT_M = 2.0

# Street-canyon geometry -> ground-link LoS rule. A drone at altitude h sees a
# ground point only within a horizontal radius of (W/2)*h/H_b, otherwise the
# building line blocks it. H_b=20 m, W=25 m is a mixed European city core.
BUILDING_H_M = 20.0
STREET_W_M = 25.0
LOS_RATIO = (STREET_W_M / 2.0) / BUILDING_H_M

# Sensor: the observer must hold the HVT in an unoccluded ray, so it is pinned
# to the same canyon radius. The EO range limit (~500 m slant) never binds.
OBSERVER_RADIUS_M = LOS_RATIO * FLIGHT_ALT_M

N0_DBM = noise_floor_dbm(BANDWIDTH_HZ, NOISE_FIGURE_DB)


def ground_los(horizontal_m: torch.Tensor, alt_m: torch.Tensor | float) -> torch.Tensor:
    """Does a node at `alt_m` have an unoccluded ray to a ground node?

    Altitude must be per-node: the MCV itself sits at 2 m, so its own LoS radius
    to another ground point is metres, i.e. always blocked in a built-up area.
    """
    return horizontal_m <= LOS_RATIO * alt_m


# --------------------------------------------------------------------------- #
# Scenario construction
# --------------------------------------------------------------------------- #


def build_positions(map_size_m: float, n_bridges: int, n_spare: int) -> torch.Tensor:
    """Node layout. Order: observer, bridges..., gateway, spares..., MCV.

    The observer is pinned overhead the HVT (sensor constraint) and the gateway
    overhead the MCV (only place with ground LoS to it). Bridges are spaced
    evenly along the straight line between them; spares are parked off-axis.
    """
    L = map_size_m
    nodes = [[L, 0.0, FLIGHT_ALT_M]]  # observer, over HVT
    for k in range(1, n_bridges + 1):
        frac = k / (n_bridges + 1)
        nodes.append([L * (1.0 - frac), 0.0, FLIGHT_ALT_M])  # bridges
    nodes.append([0.0, 0.0, FLIGHT_ALT_M])  # gateway, over MCV
    for s in range(n_spare):
        nodes.append([L * 0.5, L * 0.15 * (1 + s), FLIGHT_ALT_M])  # spares
    nodes.append([0.0, 0.0, MCV_ALT_M])  # MCV
    return torch.tensor(nodes).unsqueeze(0)


def link_capacities(
    pos: torch.Tensor, ptx_dbm: torch.Tensor, jammer_dbm: float, hvt_xy: torch.Tensor
) -> torch.Tensor:
    """(1, M, M) per-link capacity for a given power vector."""
    m = pos.shape[1]
    mcv_i = m - 1
    d3d = pairwise_distance_m(pos)

    node_alt = pos[0, :, 2]

    # Drone<->drone above rooftop: free space. Anything touching the MCV is
    # air-to-ground and obeys the canyon LoS rule. The LoS flag for a link
    # belongs to the *aerial* endpoint, so it must be indexed by whichever end
    # is not the MCV -- indexing it by receiver silently grants every uplink a
    # clear ray, because the MCV is at zero distance from itself.
    horiz_to_mcv = torch.linalg.norm(pos[0, :, :2] - pos[0, mcv_i, :2], dim=-1)
    los_to_mcv = ground_los(horiz_to_mcv, node_alt)

    los_mat = los_to_mcv.view(1, m, 1).expand(1, m, m).clone()  # [i, j] -> LoS of i
    los_mat[0, mcv_i, :] = los_to_mcv  # [mcv, j] -> LoS of j

    alt = node_alt.view(1, m, 1).expand(1, m, m).clamp_min(22.5)
    alt = torch.where(
        torch.eye(m, dtype=torch.bool).unsqueeze(0) | (torch.arange(m) == mcv_i).view(1, m, 1),
        node_alt.view(1, 1, m).expand(1, m, m).clamp_min(22.5),
        alt,
    )
    pl_a2a = fspl_db(d3d, FC_GHZ)
    pl_a2g = pathloss_a2g_umi_av_db(d3d, alt, los_mat, FC_GHZ)

    touches_mcv = torch.zeros(1, m, m, dtype=torch.bool)
    touches_mcv[0, mcv_i, :] = True
    touches_mcv[0, :, mcv_i] = True
    pl = torch.where(touches_mcv, pl_a2g, pl_a2a)

    prx = received_power_dbm(ptx_dbm, pl)

    # Jammer rides the HVT. Same canyon rule -> only nodes nearly overhead the
    # HVT take it in LoS; everyone else is shadowed by the building line. The
    # MCV, being a ground node itself, is always shadowed.
    horiz_to_hvt = torch.linalg.norm(pos[0, :, :2] - hvt_xy, dim=-1)
    d_jam = torch.sqrt(horiz_to_hvt**2 + node_alt**2).unsqueeze(0)
    jam_los = ground_los(horiz_to_hvt, node_alt).unsqueeze(0)
    jam = jammer_dbm - pathloss_a2g_umi_av_db(d_jam, pos[..., 2].clamp_min(22.5), jam_los, FC_GHZ)

    tx = ptx_dbm > -200.0
    return capacity_mbps(sinr_db(prx, jam, N0_DBM, tx), BANDWIDTH_HZ)


def e2e_capacity(
    pos: torch.Tensor, ptx_vec: list[float], jammer_dbm: float, hvt_xy: torch.Tensor
) -> float:
    m = pos.shape[1]
    ptx = torch.tensor([[*ptx_vec, -300.0]])  # MCV is receive-only
    cap = link_capacities(pos, ptx, jammer_dbm, hvt_xy)
    src = torch.zeros(1, m, dtype=torch.bool)
    src[0, 0] = True  # only the observer holds a valid HVT observation
    return best_relay_capacity(cap, src, m - 1, max_hops=m - 1).item()


def best_uniform(pos, n_drones, jammer_dbm, hvt_xy) -> tuple[float, float]:
    best = (0.0, 0.0)
    for p in [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]:
        c = e2e_capacity(pos, [p] * n_drones, jammer_dbm, hvt_xy)
        if c > best[0]:
            best = (c, p)
    return best


def best_oracle(pos, n_drones, jammer_dbm, hvt_xy) -> tuple[float, tuple]:
    """Exhaustive per-drone power allocation with full state knowledge.

    Upper bound on what any decentralized policy could achieve in the power
    domain at this geometry -- so the learned policy can be reported as a
    fraction of it, not just as a relative gain over the baseline.
    """
    best = (0.0, ())
    for combo in itertools.product(PTX_LEVELS, repeat=n_drones):
        if combo[0] <= -200.0:
            continue  # the observer must transmit; it is the data source
        c = e2e_capacity(pos, list(combo), jammer_dbm, hvt_xy)
        if c > best[0]:
            best = (c, combo)
    return best


def single_drone_capacity(map_size_m: float, jammer_dbm: float) -> float:
    """One drone overhead the HVT, relaying straight to the MCV.

    It is maximally jammed (overhead the HVT means LoS to the jammer) and its
    link to the MCV is NLoS at full map range. This is condition W1.
    """
    pos = torch.tensor([[[map_size_m, 0.0, FLIGHT_ALT_M], [0.0, 0.0, MCV_ALT_M]]])
    return e2e_capacity(pos, [PTX_MAX_DBM], jammer_dbm, torch.tensor([map_size_m, 0.0]))


# --------------------------------------------------------------------------- #


def main() -> None:
    print(
        f"fc={FC_GHZ} GHz  B={BANDWIDTH_HZ / 1e6:.0f} MHz  N0={N0_DBM:.1f} dBm  "
        f"Ptx<={PTX_MAX_DBM:.0f} dBm  alt={FLIGHT_ALT_M:.0f} m"
    )
    print(
        f"canyon: buildings {BUILDING_H_M:.0f} m, street {STREET_W_M:.0f} m "
        f"-> ground LoS within {LOS_RATIO * FLIGHT_ALT_M:.0f} m horizontally"
    )
    print(f"observer must stay within {OBSERVER_RADIUS_M:.0f} m of the HVT\n")

    n_drones = 5
    for jam in (20.0, 30.0, 40.0):
        print("=" * 92)
        print(f"JAMMER {jam:.0f} dBm in-band")
        print("=" * 92)
        print(
            f"{'map':>6} {'1 drone':>9} | {'hops':>5} {'uniform':>9} {'@dBm':>5} "
            f"{'oracle':>9} {'gain':>7} | verdict"
        )
        for L in (400, 600, 800, 1200, 1600, 2400):
            hvt = torch.tensor([float(L), 0.0])
            solo = single_drone_capacity(float(L), jam)
            rows = []
            for n_bridges in (0, 1, 2):
                n_spare = n_drones - 2 - n_bridges
                pos = build_positions(float(L), n_bridges, n_spare)
                uni, uni_p = best_uniform(pos, n_drones, jam, hvt)
                orc, _ = best_oracle(pos, n_drones, jam, hvt)
                rows.append((n_bridges + 2, uni, uni_p, orc))
            best_row = max(rows, key=lambda r: r[3])
            hops, uni, uni_p, orc = best_row
            gain = (orc / uni - 1.0) * 100 if uni > 1e-6 else float("inf")

            w1 = solo < THRESHOLD_MBPS
            w2 = orc >= THRESHOLD_MBPS
            w3 = uni > 1e-6 and gain >= 20.0
            verdict = (
                "WELL-POSED"
                if (w1 and w2 and w3)
                else "trivial"
                if not w1
                else "infeasible"
                if not w2
                else "no RQ1 headroom"
            )
            gs = f"{gain:>6.0f}%" if gain != float("inf") else "   inf"
            print(
                f"{L:>6} {solo:>9.2f} | {hops:>5} {uni:>9.2f} {uni_p:>5.0f} "
                f"{orc:>9.2f} {gs:>7} | {verdict}"
            )
        print()


if __name__ == "__main__":
    main()
