"""Freeze the pre-Block-F behavioural trace into `data/f4_golden.pt`.

⚠️ **Running this destroys evidence.** The artefact's whole value is that it was
recorded from the environment Blocks D and E measured, *before* the fidelity
ladder existed. `src/env/test_golden.py` then proves `fidelity="F4"` still
reproduces it. Re-capturing after a change to the env makes that test compare
the new code against itself, which proves nothing.

There are exactly two legitimate reasons to run this:

1. **The first capture**, taken at 5ce0a2f with Block F not yet started.
2. **A format change** -- a new scenario, a new recorded key, a different action
   rule. Bump `golden.GOLDEN_FORMAT`, and only ever alongside evidence that the
   trajectories themselves are untouched.

"The test fails" is not on that list. A failure means the environment changed,
and the question to answer is *which* Block D or Block E number moved.

Usage:
    uv run python scripts/capture_f4_golden.py            # refuses to overwrite
    uv run python scripts/capture_f4_golden.py --force    # deliberate re-capture
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.env.golden import GOLDEN_FORMAT, GOLDEN_PATH, GOLDEN_SCENARIOS, run_all, save_golden


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="overwrite an existing artefact")
    ap.add_argument("--out", type=Path, default=GOLDEN_PATH)
    a = ap.parse_args()

    if a.out.exists() and not a.force:
        raise SystemExit(
            f"{a.out} already exists.\n"
            "Overwriting it discards the pre-Block-F evidence that makes "
            "test_golden.py meaningful. Pass --force only if you have read the "
            "docstring and mean it."
        )

    print(f"capturing {len(GOLDEN_SCENARIOS)} scenarios (format {GOLDEN_FORMAT})")
    for s in GOLDEN_SCENARIOS:
        print(f"  {s.name:10s} {s.steps:4d} steps  {s.cfg}")

    torch.manual_seed(0)  # nothing should read the global RNG; belt and braces
    traces = run_all()

    for name, trace in traces.items():
        total = sum(t.numel() for t in trace.values())
        print(f"  {name:10s} {len(trace):3d} keys  {total:>9d} elements")

    save_golden(traces, a.out)
    print(f"wrote {a.out}  ({a.out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
