"""Block C throughput: how fast is line-of-sight on the real map?

Block D's gate is >=1000 env-steps/s for the *whole* env, so occlusion must cost
a fraction of that budget. This script produces the number, on the real artefact
rather than a toy, and is the thing to re-run on the rented CUDA GPU -- the
answer on Apple MPS is a lower bound, not the verdict.

Usage:
    uv run python scripts/bench_occlusion.py
    uv run python scripts/bench_occlusion.py --device cpu --no-compile
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.env.occlusion import pairwise_clearance

ARTEFACT = Path(__file__).resolve().parent.parent / "data" / "frankfurt_box.npz"
K_NODES = 7  # 5 drones + MCV + HVT
GATE_STEPS_PER_S = 1000.0


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


def parity_check(compiled, boxes, heights, dev: str, chunk: int, tol: float = 1e-2) -> bool:
    """Do the compiled and CUDA paths still compute the right answer?

    A throughput number is worthless if the kernel it timed is wrong, and the two
    ways that happens here are invisible in a benchmark: `torch.compile` reorders
    the slab chain (the +/-inf sentinels are the fragile part), and no test in
    this repo has ever exercised CUDA -- they all run on the default device.
    Deviations should be fp32 rounding, ~1e-5 m; a mishandled sentinel shows up
    as ~1e4, so `tol` is deliberately loose and still catches it.
    """
    torch.manual_seed(0)
    pos = torch.empty(16, K_NODES, 3, device=dev)
    pos[..., 0].uniform_(-700, 700)
    pos[..., 1].uniform_(-700, 700)
    pos[..., 2].uniform_(1.5, 120.0)

    ref = pairwise_clearance(pos, boxes, heights, chunk=chunk)
    ok = True

    if dev != "cpu":
        on_cpu = pairwise_clearance(pos.cpu(), boxes.cpu(), heights.cpu(), chunk=chunk)
        delta = (ref.cpu() - on_cpu).abs().max().item()
        ok &= delta <= tol
        print(
            f"parity  {dev} vs cpu      max |delta| = {delta:.2e} m  {'ok' if delta <= tol else 'FAIL'}"
        )

    if compiled is not None:
        delta = (compiled(pos, boxes, heights, chunk=chunk) - ref).abs().max().item()
        ok &= delta <= tol
        print(
            f"parity  compiled vs eager max |delta| = {delta:.2e} m  {'ok' if delta <= tol else 'FAIL'}"
        )

    if not ok:
        print("\n*** PARITY FAILED -- the numbers below are meaningless. Do not report them.")
    return ok


def bench(fn, dev: str, n: int = 5) -> float:
    for _ in range(3):
        fn()
    sync(dev)
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    sync(dev)
    return (time.perf_counter() - t0) / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--envs", type=int, nargs="*", default=[64, 256, 1024, 2048])
    args = ap.parse_args()

    dev = pick_device(args.device)
    art = np.load(ARTEFACT)
    boxes = torch.from_numpy(art["building_boxes"]).float().to(dev)
    heights = torch.from_numpy(art["building_heights"]).float().to(dev)
    m = boxes.shape[0]
    links = K_NODES * (K_NODES - 1) // 2

    print(f"device={dev}  M={m} boxes  K={K_NODES} nodes -> {links} links/env  chunk={args.chunk}")
    print(f"gate: occlusion must cost well under 1/{GATE_STEPS_PER_S:.0f} s per step\n")

    compiled = None
    if not args.no_compile:
        try:
            compiled = torch.compile(pairwise_clearance, dynamic=False)
        except Exception as exc:  # noqa: BLE001 - compile is an optimisation
            print(f"[warn] torch.compile unavailable: {exc}")

    parity_check(compiled, boxes, heights, dev, args.chunk)
    print()

    header = f"{'num_envs':>9}{'tests/step':>13}{'eager st/s':>12}"
    if compiled is not None:
        header += f"{'compiled st/s':>15}{'speedup':>9}"
    print(header)

    for b in args.envs:
        pos = torch.empty(b, K_NODES, 3, device=dev)
        pos[..., 0].uniform_(-700, 700)
        pos[..., 1].uniform_(-700, 700)
        pos[..., 2].uniform_(1.5, 120.0)

        def run(fn, _pos=pos):
            return fn(_pos, boxes, heights, chunk=args.chunk)

        eager = bench(lambda: run(pairwise_clearance), dev)
        row = f"{b:>9}{b * links * m / 1e6:>11.1f}M{1 / eager:>12.1f}"
        if compiled is not None:
            c = bench(lambda: run(compiled), dev)
            row += f"{1 / c:>15.1f}{eager / c:>8.1f}x"
        print(row)

    print(
        "\nMemory traffic, not arithmetic, is the wall: the slab chain is elementwise,"
        "\nso unfused it writes ~20 intermediates of (links x M). Fusing them via"
        "\ntorch.compile keeps them in registers, which is where the speedup comes from."
        "\nIf more headroom is needed, the next lever is a spatial broad phase:"
        "\na segment's real candidate set is ~50-150 boxes, not all of M."
    )


if __name__ == "__main__":
    main()
