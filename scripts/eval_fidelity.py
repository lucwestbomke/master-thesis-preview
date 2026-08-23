"""Block F sanity: does each rung of the ladder run, and does it run sanely?

Regenerates the ladder tables in `docs/BLOCK_F.md`. `calibrate_r.py` is the
other half and carries `R`; this file carries everything that is a property of
the rungs themselves.

⚠️ **None of this is an RQ1 result.** RQ1 trains one policy per rung and
evaluates all of them under F4; B0 is scripted and fixed, so it cannot be
"trained under F0". What B0 measures here is the **environment** -- whether each
rung is constructible, finite, and shaped the way the ladder says it should be --
without waiting for a learner that does not exist yet. Read the rows as a
description of five worlds, not as five policies being compared.

Sections
--------
  ladder      B0 under each rung: mission success, the sensor, the RQ1
              diagnostic, the chain bottleneck. Plus `F0-nogeo`, the
              building-free variant docs/BLOCK_F.md decision 1 records as the
              defensible alternative reading of F0 -- reported under its own
              name, never folded into F0.
  hops        The hop distribution per rung. What the rungs do to chain
              topology, which is what F4's divisor acts on.
  throughput  Steps per second per rung. Decision 1's stated consequence is
              that occlusion runs at every rung, so no rung is cheaper -- and a
              rung that ran faster would quietly get more samples per GPU-hour.

Statistics: >=5 seeds, means across episodes *within* a seed, median + IQR
*across* seeds.

Usage:
    uv run python scripts/eval_fidelity.py
    uv run python scripts/eval_fidelity.py --only ladder --seeds 3 --num-envs 32
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines import B0Config, B0Policy, rollout
from src.env.core import EPISODE_STEPS, BatchedSwarmEnv, EnvConfig

DEFAULT_SEEDS = 5
DEFAULT_ENVS = 64
SEED0 = 400

#: The five rungs, plus the explicitly-named building-free variant. `F0-nogeo`
#: is NOT on the ladder: it removes buildings from the world rather than from
#: the channel, so its gap against F0 is the sensor's share of the effect.
CONDITIONS = ("F0", "F0-nogeo", "F1", "F2", "F3", "F4")


def condition_cfg(name: str) -> dict:
    return {"fidelity": "F0", "no_buildings": True} if name == "F0-nogeo" else {"fidelity": name}


def make_env(
    name: str,
    num_envs: int,
    seed: int,
    compile_: bool,
    device: str = "cpu",
    auto_reset: bool = False,
    **kw,
) -> BatchedSwarmEnv:
    return BatchedSwarmEnv(
        EnvConfig(
            num_envs=num_envs,
            num_drones=5,
            seed=seed,
            device=device,
            auto_reset=auto_reset,
            eval_routes=True,
            compile_occlusion=compile_,
            stage_weights=(0.0, 0.0, 0.0, 1.0),
            **condition_cfg(name),
            **kw,
        )
    )


def run_b0(env: BatchedSwarmEnv, steps: int):
    pol = B0Policy(
        env.cfg.num_envs, env.cfg.num_drones, variant="b0", device=env.device, cfg=B0Config()
    )
    return rollout(env, lambda obs: pol.act(obs["flat"]), steps, on_reset=pol.reset)


def med_iqr(values: list[float]) -> tuple[float, float]:
    t = torch.tensor(values, dtype=torch.float64)
    return float(t.median()), float(t.quantile(0.75) - t.quantile(0.25))


def cell(values: list[float], scale: float = 100.0, unit: str = "%", width: int = 15) -> str:
    m, i = med_iqr(values)
    return f"{m * scale:>{width - 8}.1f}{unit} [{i * scale:4.1f}]"


_CACHE: dict = {}


def summaries(name: str, a) -> list[dict]:
    """One `summary()` per seed. Cached across sections."""
    if name not in _CACHE:
        out = []
        for s in range(a.seeds):
            env = make_env(name, a.num_envs, SEED0 + s, a.compile, a.device)
            out.append(run_b0(env, a.steps))
        _CACHE[name] = out
    return _CACHE[name]


# --------------------------------------------------------------------------- #


def sec_ladder(a) -> None:
    print("\n== B0 under each rung ==   eval split, full-difficulty stage")
    print(f"   {a.seeds} seeds x {a.num_envs} episodes; median [IQR] across seeds")
    print("   NOT an RQ1 result -- B0 cannot be trained under F0. This describes the")
    print("   five environments, not five policies.\n")
    hdr = (
        f"   {'rung':<10}{'capable':>15}{'observed':>15}{'chain occl':>15}"
        f"{'bottleneck':>15}{'return':>14}"
    )
    print(hdr)
    print("   " + "-" * (len(hdr) - 3))
    for name in CONDITIONS:
        rows = [m.summary() for m in summaries(name, a)]
        line = f"   {name:<10}"
        for key in ("mission_capable", "observed", "chain_occluded"):
            line += cell([r[key] for r in rows])
        m, i = med_iqr([r["bottleneck_mbps"] for r in rows])
        line += f"{m:>10.1f}    [{i:4.1f}]"
        m, i = med_iqr([r["episode_return"] for r in rows])
        line += f"{m:>9.1f} [{i:4.1f}]"
        print(line)

    print("\n   What to check, in order of how quietly it would break RQ1:")
    print("   1. `observed` is IDENTICAL down the column except at F0-nogeo. The")
    print("      sensor runs on true geometry at every rung; only the world-level")
    print("      variant may move it. A rung that moves `observed` is gating the")
    print("      sensor (decision 1) and the primary result is uninterpretable.")
    print("   2. `chain occl` at F0 is LARGE, not zero. Zero means the diagnostic")
    print("      is reading the gated clearance (decision 2) -- it is a bug, not a")
    print("      finding. F1 reads zero legitimately: occlusion is a hard veto")
    print("      there, so a chosen chain cannot contain a blocked link.")
    print("   3. F0 is permissive: `capable` close to `observed`. That is the")
    print("      abstraction under test, not a defect.")


def sec_hops(a) -> None:
    print("\n== Chain topology per rung ==   pooled hop distribution, all steps")
    print("   F4's divisor acts on hop count, so this is what the rungs hand it.\n")
    hdr = f"   {'rung':<10}" + "".join(f"{h:>8}" for h in ("0 hop", "1", "2", "3", "4", "5+"))
    hdr += f"{'multi-hop':>13}{'divisor sat.':>15}"
    print(hdr)
    print("   " + "-" * (len(hdr) - 3))
    for name in CONDITIONS:
        metrics = summaries(name, a)
        pooled = torch.stack([m.hop_distribution() for m in metrics]).mean(0)
        line = f"   {name:<10}" + "".join(f"{float(v) * 100:>7.1f}%" for v in pooled[:5])
        line += f"{float(pooled[5:].sum()) * 100:>7.1f}%"
        # Conditioned on a chain existing -- the denominator Block E fixed.
        exists = 1.0 - pooled[0]
        multi = float(pooled[2:].sum() / exists) if exists > 0 else float("nan")
        sat = float(pooled[3:].sum() / exists) if exists > 0 else float("nan")
        line += f"{multi * 100:>12.1f}%{sat * 100:>14.1f}%"
        print(line)
    print("\n   'multi-hop' and 'divisor sat.' are conditioned on a chain existing:")
    print("   steps where nobody observes have no chain and would otherwise dilute")
    print("   every share (docs/BLOCK_E.md §6).")


def sec_throughput(a) -> None:
    print("\n== Throughput per rung ==   env-steps/s, one environment one tick")
    print("   Decision 1's consequence: occlusion runs at EVERY rung, so no rung is")
    print("   cheaper to simulate. A faster rung would quietly get more samples per")
    print("   GPU-hour and the comparison would stop being like-for-like.")
    print(f"   Batch {a.throughput_envs} (NOT --num-envs): below ~128 the step is")
    print("   dominated by per-call overhead rather than by occlusion, and the")
    print("   comparison inverts -- at 64 envs `F0-nogeo`, which genuinely skips")
    print("   the geometry, measures SLOWER than F4. A benchmark whose answer")
    print("   depends on an unrelated flag is not a benchmark.\n")
    steps, warmup = 40, 10
    b = a.throughput_envs
    print(f"   {'rung':<12}{'env-steps/s':>16}{'vs F4':>10}")
    print("   " + "-" * 36)
    rates = {}
    for name in CONDITIONS:
        env = make_env(name, b, SEED0, a.compile, a.device, auto_reset=True)
        env.reset()
        act = torch.zeros(b, env.cfg.num_drones, 3, device=env.device)
        for _ in range(warmup):
            env.step(act)
        if env.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(steps):
            env.step(act)
        if env.device.type == "cuda":
            torch.cuda.synchronize()
        rates[name] = steps * b / (time.perf_counter() - t0)
    for name in CONDITIONS:
        print(f"   {name:<12}{rates[name]:>16,.0f}{rates[name] / rates['F4']:>9.2f}x")
    spread = max(rates[c] for c in CONDITIONS if c != "F0-nogeo") / min(
        rates[c] for c in CONDITIONS if c != "F0-nogeo"
    )
    print(f"\n   Spread across the five LADDER rungs: {spread:.2f}x.")
    print("   F0-nogeo is excluded -- it genuinely skips occlusion, which is the")
    print("   point of it being a separate condition rather than the default F0.")


def sec_scale(a) -> None:
    print(f"\n== Every rung at num_envs = {a.scale_envs} ==   finiteness, not speed")
    print("   docs/BLOCK_F.md's definition of done. A rung that NaNs only at batch")
    print("   scale would pass every unit test in the repo and then poison a run.\n")
    print(f"   {'rung':<12}{'non-finite':>14}{'capacity max':>16}{'e2e max':>12}{'capable':>10}")
    print("   " + "-" * 61)
    for name in CONDITIONS:
        env = make_env(name, a.scale_envs, SEED0, a.compile, a.device, auto_reset=True)
        obs = env.reset()
        gen = torch.Generator(device=env.device).manual_seed(1)
        bad = 0
        for _ in range(a.scale_steps):
            act = (
                torch.rand(a.scale_envs, env.cfg.num_drones, 3, generator=gen, device=env.device)
                * 2.0
                - 1.0
            )
            obs, rew, _term, _trunc, ex = env.step(act)
            for t in (obs["flat"], obs["state"], rew, *ex.values()):
                if t.is_floating_point():
                    bad += int((~torch.isfinite(t)).sum())
        flag = "ok" if bad == 0 else f"{bad} !!"
        print(
            f"   {name:<12}{flag:>14}{float(ex['capacity_mbps'].max()):>15.1f}"
            f"{float(ex['e2e_capacity_mbps'].max()):>12.1f}"
            f"{float(ex['mission_capable'].float().mean()) * 100:>9.1f}%"
        )
        del env, obs, ex


SECTIONS = {
    "ladder": sec_ladder,
    "hops": sec_hops,
    "throughput": sec_throughput,
    "scale": sec_scale,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=sorted(SECTIONS), default=None)
    ap.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--num-envs", type=int, default=DEFAULT_ENVS)
    ap.add_argument("--steps", type=int, default=EPISODE_STEPS)
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.add_argument("--device", default="cpu", help="cpu | mps | cuda")
    ap.add_argument("--scale-envs", type=int, default=1024, help="batch for the scale check")
    ap.add_argument(
        "--throughput-envs",
        type=int,
        default=256,
        help="batch for the throughput comparison; must be large enough that "
        "occlusion dominates the step, or the comparison measures overhead",
    )
    ap.add_argument("--scale-steps", type=int, default=5)
    a = ap.parse_args()

    # These sections take tens of minutes; without this the whole run is
    # invisible until it exits, because stdout block-buffers to a pipe.
    sys.stdout.reconfigure(line_buffering=True)
    torch.manual_seed(0)
    print(f"device: {a.device}   compile: {a.compile}")
    from src.env.core import F0_RADIUS_M

    print(f"R = {F0_RADIUS_M:.0f} m  (scripts/calibrate_r.py)")
    for name in a.only or SECTIONS:
        SECTIONS[name](a)


if __name__ == "__main__":
    main()
