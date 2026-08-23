"""Block D design measurements: altitude, sensor envelope, cue, link budget.

Four of Block D's design decisions rest on numbers that did not exist anywhere
before this script. It regenerates all of them from the shipped artefact and the
production kernels, so `docs/BLOCK_D.md` cites measurements rather than
assertions -- AGENTS.md forbids the alternative.

What each section decides:

  a2a       does the tower cluster still block air-to-air at the chosen ceiling?
            -> the altitude band. Above ~200 m the answer is no and RQ1's F1
               rung loses its A2A component entirely.
  a2g       can a drone at altitude h see a vehicle on the road h_off away?
            -> whether climbing is monotonically good (it is), and how tightly
               occlusion pins the observer to the target.
  inside    how often is a drone at altitude h standing inside a building box?
            -> the altitude FLOOR. occlusion.py ignores boxes containing an
               endpoint; below rooftop that convention stops being a corner case
               and starts letting drones see through their own building.
  cue       how stale is the one-shot cue, in range and in bearing?
            -> whether a persistent cue field is safe (it is: it decays in
               range, not direction).
  search    can five drones find the HVT with no cue at all?
            -> whether the cue hides difficulty (it does not) and what it is
               actually buying.
  budget    solo drone vs relay chain to a 1400 m HVT.
            -> whether the relay chain is necessary, and for which reason.
  solo      can a BEST-PLACED single drone do the mission on the real map?
            -> re-validates W1 ("one drone cannot do this"), which until now
               rested on scenario_design.py's analytic canyon rule -- measured
               to be more conservative than the real geometry.
  route     does the pre-baked route bank ever stall?
            -> the open question in DECISIONS.md about grow_outward lingering.
  policy    random and waypoint policies through the built env.
            -> the floor and ceiling Block D's numbers are quoted against.
               NOT B0: that is a designed baseline and belongs to Block E.

Usage:
    uv run python scripts/measure_envelope.py
    uv run python scripts/measure_envelope.py --only a2a inside
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.env.channel import (
    capacity_mbps,
    dbm_to_mw,
    noise_floor_dbm,
    pathloss_a2a_db,
    pathloss_a2g_umi_av_db,
)
from src.env.core import ALT_MAX_M, BatchedSwarmEnv, EnvConfig
from src.env.occlusion import segment_clearance
from src.env.reward import CAPACITY_THRESHOLD_MBPS

ARTEFACT = Path(__file__).resolve().parent.parent / "data" / "frankfurt_box.npz"

HALF_M = 740.0  # sample inside the box edge, not on it
HVT_Z = 1.5  # a vehicle
MCV_Z = 2.0
DT_S = 0.4
EPISODE_STEPS = 600
SENSOR_RANGE_M = 830.0  # non-binding ceiling -- see docs/BLOCK_B.md
DRONE_CRUISE_MS = 20.0

# Radio, from docs/PHYSICS.md. Fixed, not swept.
PTX_DBM = 30.0
BANDWIDTH_HZ = 10e6
JAMMER_DBM = 30.0
# The mission rate requirement, imported rather than restated. Three copies of
# this number existed before Block E raised it 5 -> 15 Mbps, and a stale copy
# here would silently re-derive the altitude ceiling against the old bar.
THRESHOLD_MBPS = CAPACITY_THRESHOLD_MBPS


def load() -> tuple[torch.Tensor, torch.Tensor, dict]:
    art = np.load(ARTEFACT)
    boxes = torch.from_numpy(art["building_boxes"]).float()
    heights = torch.from_numpy(art["building_heights"]).float()
    return boxes, heights, art


# --------------------------------------------------------------------------- #
# Altitude: what the ceiling and the floor are actually buying
# --------------------------------------------------------------------------- #


def sec_a2a(boxes: torch.Tensor, heights: torch.Tensor, n: int = 20000) -> None:
    """A2A blockage vs common altitude. Decides the CEILING."""
    print("\n== A2A: two drones at a common altitude, links 200-900 m ==")
    print("The ceiling must sit where the tower cluster still blocks something.")
    print(f"{'altitude':>10}{'blocked':>10}{'p50 clearance':>16}")
    g = torch.Generator().manual_seed(0)
    for h in (40, 60, 80, 100, 120, 150, 180, 200, 230):
        p0 = torch.empty(n, 3)
        p0[:, :2].uniform_(-HALF_M, HALF_M, generator=g)
        p0[:, 2] = h
        ang = torch.rand(n, generator=g) * 2 * math.pi
        length = 200.0 + torch.rand(n, generator=g) * 700.0
        p1 = p0.clone()
        p1[:, 0] = p0[:, 0] + length * torch.cos(ang)
        p1[:, 1] = p0[:, 1] + length * torch.sin(ang)
        keep = (p1[:, :2].abs() <= HALF_M).all(-1)
        c = segment_clearance(p0[keep], p1[keep], boxes, heights)
        print(f"{h:>8} m{(c < 0).float().mean() * 100:>9.1f}%{c.median():>14.1f} m")


def sec_a2g(boxes: torch.Tensor, heights: torch.Tensor, art, n: int = 30000) -> None:
    """Sensor envelope vs altitude and horizontal offset. Decides that up is always better."""
    print("\n== A2G: can a drone at altitude h see a vehicle on the road? ==")
    print("Monotone in altitude at every offset -> there is no low-flight tactic.")
    route = torch.from_numpy(art["route_xy"]).float().reshape(-1, 2)
    g = torch.Generator().manual_seed(1)
    bands = ((0, 50), (50, 100), (100, 200), (200, 400), (400, 800))
    print(f"{'altitude':>10}" + "".join(f"{f'{lo}-{hi} m':>11}" for lo, hi in bands))
    idx = torch.randint(0, route.shape[0], (n,), generator=g)
    hvt = torch.cat([route[idx], torch.full((n, 1), HVT_Z)], dim=1)
    for h in (40, 60, 80, 100, 120, 160, 200):
        row = f"{h:>8} m"
        for lo, hi in bands:
            ang = torch.rand(n, generator=g) * 2 * math.pi
            length = lo + torch.rand(n, generator=g) * (hi - lo)
            p = torch.stack(
                [
                    hvt[:, 0] + length * torch.cos(ang),
                    hvt[:, 1] + length * torch.sin(ang),
                    torch.full((n,), float(h)),
                ],
                dim=1,
            )
            keep = (p[:, :2].abs() <= HALF_M).all(-1)
            c = segment_clearance(p[keep], hvt[keep], boxes, heights)
            row += f"{(c >= 0).float().mean() * 100:>10.1f}%"
        print(row)


def sec_inside(boxes: torch.Tensor, heights: torch.Tensor, n: int = 200000) -> None:
    """Containment vs altitude. Decides the FLOOR.

    `occlusion.segment_clearance(ignore_endpoint_boxes=True)` drops any box
    containing an endpoint, so a contained drone sees through its own building.
    That is tolerable at 1 %; it is a falsification at 37 %.
    """
    print("\n== Inside a building: how often is a drone at altitude h contained? ==")
    print("occlusion.py ignores boxes containing an endpoint -> contained = sees through.")
    g = torch.Generator().manual_seed(3)
    p = torch.empty(n, 2)
    p.uniform_(-HALF_M, HALF_M, generator=g)
    dx = p[:, None, 0] - boxes[:, 0]
    dy = p[:, None, 1] - boxes[:, 1]
    lx = dx * boxes[:, 4] + dy * boxes[:, 5]
    ly = -dx * boxes[:, 5] + dy * boxes[:, 4]
    inside2d = (lx.abs() <= boxes[:, 2]) & (ly.abs() <= boxes[:, 3])
    roof = torch.where(inside2d, heights.expand_as(inside2d), torch.zeros_like(lx)).max(1).values
    print(f"{'altitude':>10}{'contained':>12}   note")
    for h in (5, 10, 20, 25, 30, 40, 50, 80, 120):
        note = "TR 36.777 A2G invalid below 22.5 m" if h < 22.5 else ""
        print(f"{h:>8} m{(roof >= h).float().mean() * 100:>11.2f}%   {note}")


# --------------------------------------------------------------------------- #
# The cue
# --------------------------------------------------------------------------- #


def sec_cue(art) -> None:
    """Cue staleness in range and in bearing. Decides that a persistent cue is safe."""
    print("\n== Cue staleness: does the t=0 cue ever point the wrong way? ==")
    print("grow_outward builds near-radial routes, so it decays in RANGE, not direction.")
    route = art["route_xy"]
    mcv = art["route_mcv"]
    v0 = route[:, 0, :] - mcv
    b0 = np.arctan2(v0[:, 1], v0[:, 0])
    print(
        f"{'t':>6}{'bearing drift p50':>19}{'p90':>8}"
        f"{'>90 deg':>10}{'cue->HVT p50':>15}{'cue beats MCV':>15}"
    )
    for t in (0, 50, 150, 300, 450, EPISODE_STEPS - 1):
        v = route[:, t, :] - mcv
        b = np.arctan2(v[:, 1], v[:, 0])
        drift = np.abs(np.degrees((b - b0 + np.pi) % (2 * np.pi) - np.pi))
        from_cue = np.linalg.norm(route[:, t, :] - route[:, 0, :], axis=-1)
        from_mcv = np.linalg.norm(route[:, t, :] - mcv, axis=-1)
        print(
            f"{t:>6}{np.percentile(drift, 50):>17.1f} d{np.percentile(drift, 90):>7.1f}"
            f"{(drift > 90).mean() * 100:>9.1f}%{np.percentile(from_cue, 50):>13.0f} m"
            f"{(from_cue < from_mcv).mean() * 100:>14.0f}%"
        )


def sec_search(
    boxes: torch.Tensor, heights: torch.Tensor, art, routes: int = 512, stride: int = 4
) -> None:
    """Uncued fan search vs cued. Decides that the cue does not hide difficulty.

    The drones fly a fixed radial fan at cruise speed -- a competent scripted
    search, so this is a LOWER bound on difficulty, which is the right bound for
    the question "could a good policy manage unaided?".
    """
    print("\n== Acquisition: can five drones find the HVT with no cue? ==")
    n_drones = 5
    route = torch.from_numpy(art["route_xy"]).float()
    mcv = torch.from_numpy(art["route_mcv"]).float()
    g = torch.Generator().manual_seed(7)
    sel = torch.randperm(route.shape[0], generator=g)[:routes]
    route, mcv = route[sel], mcv[sel]

    def run(bearings: torch.Tensor) -> torch.Tensor:
        first = torch.full((routes,), -1)
        for t in range(0, EPISODE_STEPS, stride):
            radius = min(DRONE_CRUISE_MS * t * DT_S, 700.0)
            dx = mcv[:, None, 0] + radius * torch.cos(bearings)
            dy = mcv[:, None, 1] + radius * torch.sin(bearings)
            z = torch.full_like(dx, min(80.0, 8.0 + DRONE_CRUISE_MS * t * DT_S))
            drones = torch.stack([dx.clamp(-HALF_M, HALF_M), dy.clamp(-HALF_M, HALF_M), z], -1)
            hvt = torch.cat([route[:, t, :], torch.full((routes, 1), HVT_Z)], 1)
            hvt = hvt[:, None, :].expand(-1, n_drones, -1)
            clear = segment_clearance(drones, hvt, boxes, heights) >= 0
            in_range = (drones - hvt).norm(dim=-1) <= SENSOR_RANGE_M
            seen = (clear & in_range).any(-1)
            first = torch.where((first < 0) & seen, torch.full_like(first, t), first)
        return first

    def report(name: str, first: torch.Tensor) -> None:
        found = first >= 0
        secs = first[found].float().numpy() * DT_S
        q = np.percentile(secs, [50, 75, 90]) if found.any() else [float("nan")] * 3
        print(
            f"{name:<36}{found.float().mean() * 100:>7.1f}%{q[0]:>9.0f} s{q[1]:>8.0f} s{q[2]:>8.0f} s"
        )

    phase = torch.rand(routes, 1, generator=g) * 2 * math.pi
    cue_xy = route[:, 0, :] + torch.randn(routes, 2, generator=g) * 150.0
    cue_b = torch.atan2(cue_xy[:, 1] - mcv[:, 1], cue_xy[:, 0] - mcv[:, 0])[:, None]

    print(f"{routes} routes, {n_drones} drones @ 80 m, {DRONE_CRUISE_MS:.0f} m/s radial")
    print(f"{'strategy':<36}{'found':>7}{'t50':>11}{'t75':>10}{'t90':>10}")
    report("no cue - 5-way fan", run(phase + torch.arange(n_drones).float() * (2 * math.pi / 5)))
    report("cue sigma=150 m - narrow fan", run(cue_b + torch.linspace(-0.35, 0.35, n_drones)))
    report("no cue - all five on one bearing", run(phase.expand(-1, n_drones)))


# --------------------------------------------------------------------------- #
# Link budget: is the relay chain necessary, and for which reason?
# --------------------------------------------------------------------------- #


def sec_budget(alt_m: float = 120.0) -> None:
    print("\n== Link budget: is the relay chain necessary? ==")
    n0 = noise_floor_dbm(BANDWIDTH_HZ, 7.0)
    noise_mw = dbm_to_mw(torch.tensor(n0))

    def cap(pathloss_db: float, jam_dbm: float | None = None) -> float:
        sig = dbm_to_mw(torch.tensor(PTX_DBM - pathloss_db))
        denom = noise_mw + (dbm_to_mw(torch.tensor(jam_dbm)) if jam_dbm is not None else 0.0)
        return capacity_mbps(10.0 * torch.log10(sig / denom), BANDWIDTH_HZ).item()

    def a2g(d: float, h: float, los: bool) -> float:
        return pathloss_a2g_umi_av_db(
            torch.tensor(float(d)), torch.tensor(float(h)), torch.tensor(los)
        ).item()

    def a2a(d: float, occluded: bool) -> float:
        return pathloss_a2a_db(torch.tensor(float(d)), torch.tensor(occluded)).item()

    print(f"Ptx {PTX_DBM:.0f} dBm, B 10 MHz, N0 {n0:.1f} dBm, drone at {alt_m:.0f} m")
    print("\nSolo drone -> MCV (A2G). Ground LoS radius is (W/2)*h/H_b, ~75 m at 120 m.")
    print(f"{'range':>9}{'graze':>9}{'LoS':>11}{'blocked':>11}{'blocked+jam':>14}")
    dz = alt_m - MCV_Z
    for d in (400, 700, 1000, 1400):
        d3d = math.hypot(d, dz)
        graze = math.degrees(math.atan2(dz, d))
        jam = JAMMER_DBM - a2g(math.hypot(d, MCV_Z), MCV_Z, False)
        print(
            f"{d:>7} m{graze:>8.1f} d{cap(a2g(d3d, alt_m, True)):>10.1f}"
            f"{cap(a2g(d3d, alt_m, False)):>11.1f}{cap(a2g(d3d, alt_m, False), jam):>13.1f}"
        )
    print(f"\n{'A2A hop':>9}{'clear':>11}{'blocked':>11}   (both drones at altitude)")
    for d in (200, 400, 700, 1000, 1400):
        print(f"{d:>7} m{cap(a2a(d, False)):>10.1f}{cap(a2a(d, True)):>11.1f}")

    last = cap(a2g(math.hypot(200.0, dz), alt_m, True))
    two = min(cap(a2a(1200, False)), last) / 2.0
    three = min(cap(a2a(600, False)), cap(a2a(600, False)), last) / 3.0
    print("\nChains to a 1400 m HVT, C_e2e = min(C_i)/min(n,3):")
    print(f"  2 hop  A2A 1200 m -> A2G 200 m            {two:>6.1f} Mbps")
    print(f"  3 hop  A2A 600 m x2 -> A2G 200 m          {three:>6.1f} Mbps")
    print(f"  threshold                                 {THRESHOLD_MBPS:>6.1f} Mbps")
    print("\nThe chain is required by BLOCKAGE, not by range: with a clear ray one")
    print("drone closes 1400 m with margin. That is what makes RQ1 a real question --")
    print("a radius model would capture a range requirement perfectly.")


# --------------------------------------------------------------------------- #
# Is the scenario premise still true on the real map?
# --------------------------------------------------------------------------- #


def sec_solo(boxes: torch.Tensor, heights: torch.Tensor, art, routes: int = 512) -> None:
    """Best-case solo drone. Decides whether W1 survives real geometry.

    The drone is placed in the most favourable position available to it --
    hovering directly over the HVT at the ceiling, where it sees the target
    ~97 % of the time and has the shortest possible slant to the MCV. If even
    that fails, "one drone cannot do the mission" holds for *any* policy, which
    is a far stronger statement than a scripted run could make.

    scenario_design.py answers the same question with an analytic canyon rule
    (ground LoS within 0.625*altitude) that the A2G measurement above shows is
    more conservative than the real map, so the premise needs checking here.
    """
    print("\n== Solo drone, best case: hovering over the HVT at the ceiling ==")
    n0 = noise_floor_dbm(BANDWIDTH_HZ, 7.0)
    noise_mw = dbm_to_mw(torch.tensor(n0))
    route = torch.from_numpy(art["route_xy"]).float()
    mcv2 = torch.from_numpy(art["route_mcv"]).float()
    g = torch.Generator().manual_seed(11)
    sel = torch.randperm(route.shape[0], generator=g)[:routes]
    route, mcv2 = route[sel], mcv2[sel]
    mcv = torch.cat([mcv2, torch.full((routes, 1), MCV_Z)], 1)

    print(
        f"{'alt':>6}{'t':>6}{'sep':>9}{'sees HVT':>11}{'link ok':>10}{'MISSION OK':>13}{'p50 Mbps':>11}"
    )
    for alt in (ALT_MAX_M, 100.0, 120.0):
        for t in (0, 300, EPISODE_STEPS - 1):
            hvt = torch.cat([route[:, t, :], torch.full((routes, 1), HVT_Z)], 1)
            drone = torch.cat([route[:, t, :], torch.full((routes, 1), alt)], 1)

            sees = segment_clearance(drone, hvt, boxes, heights) >= 0
            clr_mcv = segment_clearance(drone, mcv, boxes, heights)
            d3d = (drone - mcv).norm(dim=-1)
            pl = pathloss_a2g_umi_av_db(d3d, drone[:, 2], clr_mcv >= 0)

            # Jammer rides the HVT and lands on the MCV, which is what receives.
            d_jam = (hvt - mcv).norm(dim=-1)
            jam_los = segment_clearance(hvt, mcv, boxes, heights) >= 0
            jam_mw = dbm_to_mw(JAMMER_DBM - pathloss_a2g_umi_av_db(d_jam, mcv[:, 2], jam_los))

            sinr = 10.0 * torch.log10(dbm_to_mw(PTX_DBM - pl) / (jam_mw + noise_mw))
            cap = capacity_mbps(sinr, BANDWIDTH_HZ)
            ok = cap >= THRESHOLD_MBPS
            sep = (hvt[:, :2] - mcv[:, :2]).norm(dim=-1)
            print(
                f"{alt:>5.0f}m{t:>6}{sep.median():>8.0f}m{sees.float().mean() * 100:>10.1f}%"
                f"{ok.float().mean() * 100:>9.1f}%{(sees & ok).float().mean() * 100:>12.1f}%"
                f"{cap.median():>11.1f}"
            )
    print(
        "W1 holds only if MISSION OK stays low once the HVT is far out.\n"
        "NOTE: at the 15 Mbps requirement W1 holds at EVERY altitude in and above the\n"
        "band (0.4-0.8 % at 1336 m), so it no longer discriminates between ceilings and\n"
        "no longer pins the 80 m ceiling -- see docs/DECISIONS.md."
    )


def sec_route(art) -> None:
    """Does grow_outward stall? The open question in DECISIONS.md.

    One route once spent 333 steps (133 s) on a ~240 m bridge -- about 5x too
    long at the capped speed. The bridge decks are gone, but the *timing* was
    never re-checked, and the escalation profile is calibrated on medians so a
    few stalled routes would not show up in it.
    """
    print("\n== Route bank: does the outward walk ever stall? ==")
    r = art["route_xy"]
    step = np.linalg.norm(np.diff(r, axis=1), axis=-1)  # (R, T-1) metres per step
    # A "stall" is a run of steps covering less ground than a slow walk.
    stalled = step < 0.4  # < 1 m/s at dt = 0.4 s
    runs = np.zeros(len(r), dtype=int)
    for i in range(len(r)):
        best = cur = 0
        for v in stalled[i]:
            cur = cur + 1 if v else 0
            best = max(best, cur)
        runs[i] = best
    net = np.linalg.norm(r[:, -1, :] - r[:, 0, :], axis=-1)
    path = step.sum(axis=1)
    straightness = net / np.maximum(path, 1e-6)
    print(f"  routes                       {len(r)}")
    print(
        f"  longest stalled run   p50/p90/max  {np.percentile(runs, 50):.0f} /"
        f" {np.percentile(runs, 90):.0f} / {runs.max():.0f} steps"
    )
    print(f"  routes stalled > 50 steps    {(runs > 50).sum()}  ({(runs > 50).mean() * 100:.1f} %)")
    print(
        f"  straightness (net/path) p10/p50  {np.percentile(straightness, 10):.2f} /"
        f" {np.percentile(straightness, 50):.2f}"
    )
    print(f"  slowest route mean speed     {(path / (EPISODE_STEPS * DT_S)).min():.2f} m/s")


# --------------------------------------------------------------------------- #
# Policy floor and ceiling
# --------------------------------------------------------------------------- #


def waypoint_policy(env: BatchedSwarmEnv, altitude_m: float = 100.0) -> torch.Tensor:
    """Crude relay heuristic: string the drones along the MCV->HVT line.

    **This is not B0.** B0 is a designed geometric baseline that gets reported
    (THESIS_PLAN §3) and belongs to Block E. This exists only so the numbers
    Block D quotes have a regenerable ceiling.
    """
    n = env.cfg.num_drones
    frac = torch.linspace(1.0, 0.15, n, device=env.device).view(1, n, 1)
    target = env.hvt_pos.unsqueeze(1) * (1.0 - frac) + env.mcv_pos.unsqueeze(1) * frac
    target = torch.cat([target[..., :2], torch.full_like(target[..., :1], altitude_m)], dim=-1)
    drive = (target - env.drone_pos) * 0.25 - env.drone_vel * 1.2
    return drive.clamp(-10.0, 10.0) / 10.0


def sec_policy(num_envs: int = 64, num_drones: int = 5) -> None:
    """The floor and ceiling every Block D claim is quoted against."""
    print("\n== Policies through the built env (floor and ceiling) ==")
    print(f"{'policy':<12}{'mission-capable':>17}{'observed':>11}{'chain occl':>12}{'3-hop':>8}")
    for name in ("random", "waypoint"):
        env = BatchedSwarmEnv(
            EnvConfig(
                num_envs=num_envs,
                num_drones=num_drones,
                seed=1,
                stage_weights=(0.0, 0.0, 0.0, 1.0),
                compile_occlusion=False,
            )
        )
        env.reset()
        torch.manual_seed(0)
        cap = seen = occl = 0.0
        hops = torch.zeros(num_drones + 2)
        for _ in range(EPISODE_STEPS):
            if name == "random":
                act = torch.empty(num_envs, num_drones, 3, device=env.device).uniform_(-1, 1)
            else:
                act = waypoint_policy(env)
            _, _, _, _, ex = env.step(act)
            cap += ex["mission_capable"].float().mean().item()
            seen += ex["sees_any"].float().mean().item()
            occl += ex["chain_occluded"].float().mean().item()
            hops += torch.bincount(ex["hop_count"], minlength=num_drones + 2).float().cpu()
        t = EPISODE_STEPS
        print(
            f"{name:<12}{cap / t * 100:>16.1f}%{seen / t * 100:>10.1f}%"
            f"{occl / t * 100:>11.1f}%{hops[3] / hops.sum() * 100:>7.1f}%"
        )
    print(
        "The gap between observed and mission-capable is the LINK binding. At the old\n"
        "5 Mbps bar the two columns were identical for every policy -- the chain always\n"
        "delivered once anything was seen, so the relay premise was never exercised.\n"
        f"At {THRESHOLD_MBPS:.0f} Mbps they separate, which is why the bar was raised "
        "(docs/BLOCK_E.md)."
    )


SECTIONS = {
    "a2a": lambda b, h, a: sec_a2a(b, h),
    "a2g": lambda b, h, a: sec_a2g(b, h, a),
    "inside": lambda b, h, a: sec_inside(b, h),
    "cue": lambda b, h, a: sec_cue(a),
    "search": lambda b, h, a: sec_search(b, h, a),
    "budget": lambda b, h, a: sec_budget(),
    "solo": lambda b, h, a: sec_solo(b, h, a),
    "route": lambda b, h, a: sec_route(a),
    "policy": lambda b, h, a: sec_policy(),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", choices=sorted(SECTIONS), default=None)
    args = ap.parse_args()

    boxes, heights, art = load()
    print(f"artefact: {ARTEFACT.name}  M={boxes.shape[0]} boxes  {art['route_xy'].shape[0]} routes")
    for name in args.only or SECTIONS:
        SECTIONS[name](boxes, heights, art)


if __name__ == "__main__":
    main()
