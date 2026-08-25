"""Is `channel.pathloss_a2a_db`'s 20 dB blockage penalty defensible?

    uv run python scripts/verify_blockage.py
    uv run python scripts/verify_blockage.py --only depth      # skip the sweep

`blockage_db = 20.0` is an **assumed** constant -- not measured, not cited. RQ1's
F1 rung rests on occlusion mattering, so an unciteable number in the occlusion
path is a real exposure in the methodology chapter.

A parameter you cannot cite can still be defended, in one of two ways: derive it
from physics, or show the result does not depend on it. This does both, and they
disagree in an interesting way.

## 1. What the real geometry says the loss should be

3GPP has **no** air-to-air model -- TR 36.777 is air-to-ground only -- so the
reference physics is single knife-edge diffraction (ITU-R P.526), with the
Fresnel-Kirchhoff parameter

    v = h * sqrt( 2*(d1 + d2) / (lambda * d1 * d2) )

and the standard approximation
`L(v) = 6.9 + 20*log10( sqrt((v-0.1)^2 + 1) + v - 0.1 )` for `v > -0.78`.

`h` is taken from the env's own signed clearance, which is exactly the depth the
ray passes below the obstruction -- the quantity a binary flag throws away.

## 2. Whether it matters

A sensitivity sweep of the constant through B0 at stage 4 / F4, and a count of
which link class the occluded chain edges actually belong to.
"""

from __future__ import annotations

import argparse
import functools
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.baselines.b0 import B0Policy
from src.baselines.evaluate import rollout
from src.env import channel
from src.env.core import STAGES, BatchedSwarmEnv, EnvConfig

FC_GHZ = 3.5
LAMBDA_M = 0.299792458 / FC_GHZ


def knife_edge_db(h_m: np.ndarray, d_m: np.ndarray) -> np.ndarray:
    """ITU-R P.526 single knife-edge loss, obstruction taken at mid-path.

    Mid-path is the *optimistic* placement: it maximises `d1*d2` and so minimises
    `v` for a given depth. Any other position along the link gives a larger loss,
    so the figures below are a lower bound on the diffraction penalty.
    """
    v = h_m * np.sqrt(8.0 / (LAMBDA_M * np.maximum(d_m, 1.0)))
    loss = 6.9 + 20.0 * np.log10(np.sqrt((v - 0.1) ** 2 + 1.0) + v - 0.1)
    return np.maximum(np.where(v <= -0.78, 0.0, loss), 0.0)


def measure_depths(envs: int, drones: int, steps: int, seed: int):
    """How deep do occluded A2A rays actually pass through Frankfurt geometry?"""
    env = BatchedSwarmEnv(
        EnvConfig(
            num_envs=envs,
            num_drones=drones,
            device="cpu",
            seed=seed,
            stage_weights=(0.0, 0.0, 0.0, 1.0),
        )
    )
    env.reset()
    n = env.cfg.num_drones
    depths, dists = [], []
    eye = torch.eye(n, dtype=torch.bool).unsqueeze(0)
    for _ in range(steps):
        env.step(torch.empty(envs, n, 3).uniform_(-1, 1))
        pos = env.drone_pos
        true_clr, _ = env._clearance(
            torch.cat([pos, env.mcv_pos.unsqueeze(1), env.hvt_pos.unsqueeze(1)], 1)
        )
        clr = true_clr[:, :n, :n]
        d = (pos.unsqueeze(2) - pos.unsqueeze(1)).norm(dim=-1)
        blocked = (clr < 0.0) & ~eye
        depths.append((-clr[blocked]).flatten())
        dists.append(d[blocked].flatten())
    return torch.cat(depths).numpy(), torch.cat(dists).numpy()


def edge_class_split(envs: int, drones: int, steps: int, seed: int):
    """Of the occluded edges on B0's CHOSEN chain, how many are A2A vs A2G?

    The number that decides how much `blockage_db` can possibly matter: it
    governs A2A only, and A2G runs on the (verified) TR 36.777 NLoS branch.
    """
    env = BatchedSwarmEnv(
        EnvConfig(
            num_envs=envs,
            num_drones=drones,
            device="cpu",
            seed=seed,
            auto_reset=True,
            stage_weights=(0.0, 0.0, 0.0, 1.0),
        )
    )
    r = env.cfg.n_radio
    obs = env.reset()
    pol = B0Policy(envs, drones, variant="b0", device=env.device)
    a2a = a2g = 0
    for _ in range(steps):
        obs, _rew, _term, _trunc, ex = env.step(pol.act(obs["flat"]))
        true_clr, _ = env._clearance(
            torch.cat([env.drone_pos, env.mcv_pos.unsqueeze(1), env.hvt_pos.unsqueeze(1)], 1)
        )
        occ = ex["on_edge"] & (true_clr[:, :r, :r] < 0.0)
        a2a += int((occ & env.is_a2a).sum())
        a2g += int((occ & ~env.is_a2a).sum())
    return a2a, a2g


def sweep(values: list[float], envs: int, drones: int, seed: int):
    """B0's headline metric as the assumed constant is swept."""
    from src.env import core

    original = channel.pathloss_a2a_db
    out = []
    try:
        for blk in values:
            patched = functools.partial(original, blockage_db=blk)
            channel.pathloss_a2a_db = patched
            core.channel.pathloss_a2a_db = patched
            env = BatchedSwarmEnv(
                EnvConfig(
                    num_envs=envs,
                    num_drones=drones,
                    device="cpu",
                    seed=seed,
                    auto_reset=False,
                    stage_weights=(0.0, 0.0, 0.0, 1.0),
                )
            )
            pol = B0Policy(envs, drones, variant="b0", device=env.device)
            m = rollout(
                env,
                lambda o, _p=pol: _p.act(o["flat"]),
                steps=STAGES[3].episode_steps,
                on_reset=pol.reset,
                rate_division_counterfactual=False,
            )
            out.append((blk, m.summary()))
    finally:
        channel.pathloss_a2a_db = original
        core.channel.pathloss_a2a_db = original
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--drones", type=int, default=8)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--only", nargs="*", default=["depth", "split", "sweep"])
    a = ap.parse_args()

    if "depth" in a.only:
        depth, dist = measure_depths(a.envs, a.drones, a.steps, a.seed)
        loss = knife_edge_db(depth, dist)
        print(f"\n=== 1. Occluded A2A rays in real Frankfurt geometry ({len(depth):,} links) ===\n")
        print("obstruction DEPTH below the ray -- how far inside the building it passes")
        for q in (10, 25, 50, 75, 90):
            print(f"    p{q:<3} {np.percentile(depth, q):7.1f} m")
        print(f"    link distance: median {np.median(dist):.0f} m")
        print(
            f"    first Fresnel radius at that distance: {math.sqrt(LAMBDA_M * np.median(dist) / 4):.1f} m"
        )
        print("\nimplied single knife-edge diffraction loss at 3.5 GHz")
        for q in (10, 25, 50, 75, 90):
            print(f"    p{q:<3} {np.percentile(loss, q):7.1f} dB")
        print(f"\n    share exceeding the modelled 20 dB: {100 * (loss > 20).mean():.1f} %")

    if "split" in a.only:
        a2a, a2g = edge_class_split(a.envs, 5, 300, a.seed + 2)
        tot = a2a + a2g
        print("\n=== 2. Which link class carries the occluded chain edges? ===\n")
        print(f"    A2A (uses blockage_db)     {a2a:7,}  {100 * a2a / tot:5.1f} %")
        print(f"    A2G (uses TR 36.777 NLoS)  {a2g:7,}  {100 * a2g / tot:5.1f} %")

    if "sweep" in a.only:
        print("\n=== 3. Does the headline metric depend on it? (B0, stage 4, F4) ===\n")
        print(
            f"    {'blockage':>9}{'capable':>10}{'observed':>10}{'capacity':>10}{'chain occl':>12}"
        )
        for blk, s in sweep([20.0, 30.0, 40.0], 48, 5, a.seed + 6):
            print(
                f"    {blk:9.0f}{s['mission_capable'] * 100:9.1f}%{s['observed'] * 100:9.1f}%"
                f"{s['capacity_mean']:10.1f}{s['chain_occluded'] * 100:11.1f}%"
            )


if __name__ == "__main__":
    main()
