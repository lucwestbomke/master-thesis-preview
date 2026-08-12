"""Block D throughput: how fast does the whole env step, and where does it go?

`AGENTS.md` requires this measured before anything is built on top of the env.
The gate itself was already cleared by occlusion alone on CUDA
(`docs/BLOCK_C.md`), so the question this script answers is no longer "does it
pass" but **"has the profile inverted?"** -- occlusion is 0.32 ms on a 5090, so
scaffolding that costs a few milliseconds would make it irrelevant and move the
bottleneck somewhere nobody is looking.

Reports throughput in BOTH units, because the repo previously stated its gate in
two that differ by 1000x (`docs/BLOCK_D.md`, decision 1):

    env-steps/s     one environment advancing one tick, summed over the batch.
                    THIS is the gate's unit, and what the 120 GPU-hour budget in
                    THESIS_PLAN §3 is written in.
    calls/s         one batched step() over all num_envs.

Usage:
    uv run python scripts/bench_env.py
    uv run python scripts/bench_env.py --envs 1024 --drones 5 8 --no-compile
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.env import channel, routing
from src.env.core import BANDWIDTH_HZ, BatchedSwarmEnv, EnvConfig

GATE_ENV_STEPS_PER_S = 1000.0
RUN_STEPS = 10_000_000  # one reported training run, THESIS_PLAN §3


def pick_device(name: str | None) -> str:
    if name:
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(dev: str) -> None:
    if dev == "cuda":
        torch.cuda.synchronize()
    elif dev == "mps":
        torch.mps.synchronize()


def peak_mem_gb(dev: str) -> float:
    if dev == "cuda":
        return torch.cuda.max_memory_allocated() / 1e9
    if dev == "mps":
        return torch.mps.current_allocated_memory() / 1e9
    return float("nan")


def bench(fn, dev: str, n: int = 10, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    sync(dev)
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    sync(dev)
    return (time.perf_counter() - t0) / n


def stage_breakdown(
    env: BatchedSwarmEnv, actions: torch.Tensor, dev: str
) -> list[tuple[str, float]]:
    """Cost of each stage in isolation, eager.

    Only meaningful eager: `torch.compile` fuses across stage boundaries, so a
    compiled breakdown would not sum to the compiled total. Read this to find
    *which* stage grew, then trust the end-to-end number for the total.
    """
    cfg = env.cfg
    pos, _vel, _accel = env._advance_drones(actions)
    hvt_pos, _ = env._advance_hvt(env.t + 1)
    pos_k = torch.cat([pos, env.mcv_pos.unsqueeze(1), hvt_pos.unsqueeze(1)], dim=1)
    clearance = env._clearance(pos_k)
    capacity = env._capacity(pos_k, clearance)
    sees = torch.zeros(cfg.num_envs, cfg.num_drones, dtype=torch.bool, device=env.device)
    source = torch.cat([sees, torch.zeros_like(sees[:, :1])], dim=1)

    return [
        ("kinematics", bench(lambda: env._advance_drones(actions), dev)),
        ("hvt route", bench(lambda: env._advance_hvt(env.t + 1), dev)),
        ("occlusion", bench(lambda: env._clearance(pos_k), dev)),
        ("channel", bench(lambda: env._capacity(pos_k, clearance), dev)),
        (
            "routing DP",
            bench(
                lambda: routing.best_relay_path(
                    capacity, source, cfg.n_radio - 1, cfg.n_radio - 1, cfg.reuse_limit
                ),
                dev,
            ),
        ),
        (
            "capacity fn",
            bench(lambda: channel.capacity_mbps(torch.zeros_like(capacity), BANDWIDTH_HZ), dev),
        ),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--envs", type=int, nargs="*", default=[256, 1024, 4096])
    ap.add_argument("--drones", type=int, nargs="*", default=[5])
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--breakdown", action="store_true", help="per-stage costs at the first config")
    args = ap.parse_args()

    dev = pick_device(args.device)
    print(f"device={dev}  torch={torch.__version__}  chunk={args.chunk}")
    print("gate: >=1000 ENV-STEPS/s (transitions, not batched calls) -- docs/BLOCK_D.md\n")

    header = (
        f"{'drones':>7}{'num_envs':>10}{'ms/call':>10}{'calls/s':>10}"
        f"{'env-steps/s':>14}{'x gate':>9}{'10M run':>11}{'peak GB':>9}"
    )
    print(header)
    print("-" * len(header))

    worst = float("inf")
    for n_drones in args.drones:
        for b in args.envs:
            cfg = EnvConfig(num_envs=b, num_drones=n_drones, device=dev, occlusion_chunk=args.chunk)
            env = BatchedSwarmEnv(cfg)
            env.reset()
            actions = torch.zeros(b, n_drones, 3, device=dev)

            step = env.physics_step
            if not args.no_compile:
                try:
                    step = torch.compile(env.physics_step, dynamic=False)
                except Exception as exc:  # noqa: BLE001
                    print(f"[warn] torch.compile unavailable: {exc}")

            if dev == "cuda":
                torch.cuda.reset_peak_memory_stats()
            dt = bench(lambda _s=step, _a=actions: _s(_a), dev)

            eps = b / dt
            worst = min(worst, eps)
            run_h = RUN_STEPS / eps / 3600.0
            run = f"{run_h * 60:.1f} min" if run_h < 1 else f"{run_h:.1f} h"
            print(
                f"{n_drones:>7}{b:>10}{dt * 1e3:>10.2f}{1 / dt:>10.1f}{eps:>14,.0f}"
                f"{eps / GATE_ENV_STEPS_PER_S:>9,.0f}{run:>11}{peak_mem_gb(dev):>9.2f}"
            )

            if args.breakdown:
                stages = stage_breakdown(env, actions, dev)
                eager_total = sum(s for _, s in stages)
                print(f"\n  per-stage at num_envs={b} N={n_drones}, all EAGER.")
                print("  Shares are of the eager total, not of the compiled step above --")
                print("  fusion crosses stage boundaries, so the two are not comparable.")
                for name, secs in stages:
                    print(f"    {name:<14}{secs * 1e3:>9.3f} ms{secs / eager_total * 100:>8.1f} %")
                print(
                    f"    {'-- eager sum':<14}{eager_total * 1e3:>9.3f} ms"
                    f"   vs compiled step {dt * 1e3:.3f} ms  ({eager_total / dt:.0f}x)"
                )
                print()

    print()
    if dev != "cuda":
        print(f"PROVISIONAL -- not CUDA. Local {dev} is a lower bound, never a verdict.")
        print("Re-run on the rented GPU before any claim about the gate.")
    elif worst >= GATE_ENV_STEPS_PER_S:
        print(f"GATE MET at every configuration ({worst:,.0f} env-steps/s worst case).")
    else:
        print(f"*** GATE MISSED: worst case {worst:,.0f} env-steps/s < {GATE_ENV_STEPS_PER_S:.0f}.")


if __name__ == "__main__":
    main()
