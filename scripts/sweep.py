"""The equal-budget sweep -- Block G, and the thing MODELS.md has always owed.

    uv run python scripts/sweep.py --device cuda                  # stage A
    uv run python scripts/sweep.py --device cuda --stage-b        # confirm winners
    uv run python scripts/sweep.py --device cuda --dry-run        # print the plan

`docs/MODELS.md` rule 2: *"Equal hyperparameter budget across all three, and say
so in the methodology. Tuning the GNN harder than the baselines is the single
most likely way this result gets dismissed."* BLOCK_G decision 3 turns that into
an operational definition -- **the same search space, the same number of trials,
the same selection rule, for each of the three architectures** -- which is what
this script executes and records.

⚠️ The search is run **per architecture**. Tuning on the GNN and applying the
winner to the others would be exactly the unequal budget the rule forbids, and it
is the tempting shortcut because the GNN is currently ahead.

## Why the axes are these

**Cadence.** The first CUDA session measured `ms/call` as *flat* from 256 to 4096
environments -- every stage but occlusion is kernel-launch bound, so a 16x larger
batch is free. But raising `num_envs` at fixed `rollouts` divides the number of
optimizer steps by the same factor, so batch size and update cadence cannot be
swept independently. Each preset below holds **gradient steps per env-step**
roughly constant and varies what the extra samples buy: a bigger batch (less
gradient noise) or a longer GAE horizon.

**Shaping.** The only reward knobs the design permits. `d_ref_m` and
`potential_scale` both live inside `Phi`, so by the PBRS invariance proof they
cannot move the optimum -- only learning speed. ⛔ Every other weight is pinned
by the behavioural orderings in `docs/REWARD.md`. `d_ref_m = 400` measured +3 pp
on MPS; `potential_scale` has never been moved.

**Not swept, deliberately:** the learning rate (fixed at 3e-4 so cadence is not
confounded with it -- if a winner sits at a boundary, sweep it separately and say
so), the curriculum schedule (measured on 3 seeds: the shipped one beats no
curriculum by +4.4 pp and cuts seed spread 4.5x), and fidelity (⛔ RQ1's
independent variable, never a tuning axis).

## Selection rule, declared before any result is seen

Median `mission_capable` across seeds, scored through `src/baselines/evaluate.py`
-- the harness B0's number came from -- at stage 4, F4, on the **train** route
split. Ties broken by smaller IQR. Stage B then re-runs each architecture's
winner at 5 seeds and reports on the **eval** split, which is touched exactly
once, at the end.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_policy import med_iqr, score

SWEEP_DIR = ROOT / "runs" / "sweep"
SUMMARY = SWEEP_DIR / "summary.jsonl"


@dataclass(frozen=True)
class Cadence:
    """A (batch, update) pairing that holds gradient steps per env-step fixed.

    `rows_per_update = rollouts * num_envs * N`; one update every
    `rollouts * num_envs` env-steps; `learning_epochs * mini_batches` gradient
    steps per update. Raising `num_envs` alone divides the optimizer steps by the
    same factor, which is why these move together.
    """

    name: str
    num_envs: int
    rollouts: int
    mini_batches: int

    def grad_steps_per_million(self, epochs: int = 4, num_drones: int = 5) -> float:
        updates = 1e6 / (self.rollouts * self.num_envs)
        return updates * epochs * self.mini_batches


CADENCES = (
    # today's configuration, and the control everything is measured against
    Cadence("base", num_envs=1024, rollouts=32, mini_batches=4),
    # 4x the batch for free (ms/call is flat to 4096); same horizon, same
    # gradient density, less gradient noise -- aimed straight at the seed spread
    Cadence("wide", num_envs=4096, rollouts=32, mini_batches=16),
    # 4x the batch spent on a longer GAE horizon instead. gamma = 0.997 has an
    # effective horizon of 333 steps and the rollout sees 32, so this is the
    # axis with a real reason behind it rather than a knob
    Cadence("deep", num_envs=4096, rollouts=64, mini_batches=32),
)

#: PBRS-safe reward shaping. Optimum-preserving by construction; learning speed
#: only. The names are what appears in the summary and in the write-up.
SHAPINGS = (
    ("shipped", {}),
    ("dref400", {"--d-ref": "400"}),
    ("dref400_k30", {"--d-ref": "400", "--potential-scale": "30"}),
)

ARCHITECTURES = ("mlp", "deepsets", "gnn")


def cells(architectures, cadences=None, shapings=None):
    """The grid. ⚠️ Filters exist for resuming a partial sweep and for smoke
    tests -- a *reported* equal-budget claim needs the full grid on every
    architecture, which is what `--arch` alone preserves."""
    for arch in architectures:
        for cadence in CADENCES:
            if cadences and cadence.name not in cadences:
                continue
            for shaping_name, shaping in SHAPINGS:
                if shapings and shaping_name not in shapings:
                    continue
                yield arch, cadence, shaping_name, shaping


def run_name(arch: str, cadence: str, shaping: str, seed: int) -> str:
    return f"sweep/{arch}-{cadence}-{shaping}-s{seed}"


def train_one(a, arch: str, cadence: Cadence, shaping_name: str, shaping: dict, seed: int) -> bool:
    """One training run. Returns True if it ran, False if it was already there."""
    name = run_name(arch, cadence.name, shaping_name, seed)
    if (ROOT / "runs" / name / "checkpoint.pt").exists():
        return False
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "src.training.train",
        "--fidelity",
        "F4",
        "--arch",
        arch,
        "--seed",
        str(seed),
        "--num-envs",
        str(cadence.num_envs),
        "--rollouts",
        str(cadence.rollouts),
        "--mini-batches",
        str(cadence.mini_batches),
        "--env-steps",
        str(a.env_steps),
        "--device",
        a.device,
        "--boundaries",
        "0.10",
        "0.20",
        "0.35",
        "--min-std",
        "0.2",
        "--log-every",
        "100000",  # the sweep reads checkpoints, not curves
        "--name",
        name,
    ]
    for flag, value in shaping.items():
        cmd += [flag, value]
    if a.dry_run:
        print("  " + " ".join(cmd))
        return True
    subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    return True


def evaluate_one(a, name: str, eval_split: bool) -> dict[str, float]:
    """Score a checkpoint through `evaluate.py`, the same path B0 went through."""
    args = SimpleNamespace(
        stage=4,
        fidelity="F4",
        num_envs=a.eval_envs,
        device=a.device,
        train_routes=not eval_split,
        seeds=1,
    )
    cols = score(args, name, ROOT / "runs" / name / "checkpoint.pt", 5)
    return {k: v[0] for k, v in cols.items()}


def append(row: dict) -> None:
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    with SUMMARY.open("a") as handle:
        handle.write(json.dumps(row) + "\n")


def load_summary() -> list[dict]:
    if not SUMMARY.exists():
        return []
    return [json.loads(line) for line in SUMMARY.open()]


def report(rows: list[dict], key: str = "mission_capable") -> list[tuple]:
    """Group by cell, aggregate across seeds, rank. The selection rule, applied."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["arch"], row["cadence"], row["shaping"], row["split"]), []).append(
            row
        )
    ranked = []
    for cell, members in groups.items():
        values = [m[key] for m in members]
        median, iqr = med_iqr(values)
        ranked.append((median, -iqr, cell, len(members), median, iqr, members))
    ranked.sort(reverse=True)
    return ranked


def print_table(ranked: list[tuple], title: str) -> None:
    print(f"\n{title}")
    header = (
        f"{'arch':<10}{'cadence':<8}{'shaping':<14}{'split':<7}{'n':>3}"
        f"{'capable':>16}{'observed':>12}{'tenure':>9}{'hops':>7}"
    )
    print(header)
    print("-" * len(header))
    for _, _, cell, n, median, iqr, members in ranked:
        arch, cadence, shaping, split = cell
        obs = med_iqr([m["observed"] for m in members])[0]
        ten = med_iqr([m["observer_tenure"] for m in members])[0]
        hop = med_iqr([m["hop_mean"] for m in members])[0]
        print(
            f"{arch:<10}{cadence:<8}{shaping:<14}{split:<7}{n:>3}"
            f"{median * 100:>11.1f} %[{iqr * 100:.1f}]{obs * 100:>11.1f} %"
            f"{ten:>9.1f}{hop:>7.2f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--env-steps", type=int, default=12_000_000)
    ap.add_argument("--seeds", type=int, default=3, help="stage A seeds per cell")
    ap.add_argument("--final-seeds", type=int, default=5, help="stage B seeds per winner")
    ap.add_argument("--eval-envs", type=int, default=256, help="episodes per scoring run")
    ap.add_argument("--arch", nargs="*", default=list(ARCHITECTURES))
    ap.add_argument("--cadence", nargs="*", default=None, help="filter; resuming and smoke tests")
    ap.add_argument("--shaping", nargs="*", default=None, help="filter; resuming and smoke tests")
    ap.add_argument("--stage-b", action="store_true", help="confirm the winners on the eval split")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()

    if a.report_only:
        rows = load_summary()
        print_table(report([r for r in rows if r["split"] == "train"]), "STAGE A (train split)")
        print_table(report([r for r in rows if r["split"] == "eval"]), "STAGE B (eval split)")
        return

    architectures = tuple(a.arch)
    plan = list(cells(architectures, a.cadence, a.shaping))
    if a.cadence or a.shaping:
        print("⚠️  FILTERED grid -- not an equal-budget result, only a partial run\n")
    total = len(plan) * a.seeds
    print(f"device={a.device}  {a.env_steps:,} env-steps per run")
    print(f"stage A: {len(plan)} cells x {a.seeds} seeds = {total} runs")
    print(
        f"  {len(architectures)} architectures x {len(CADENCES)} cadences x {len(SHAPINGS)} shapings"
    )
    for c in CADENCES:
        print(
            f"  cadence {c.name:<6} num_envs={c.num_envs:<5} rollouts={c.rollouts:<3} "
            f"mini_batches={c.mini_batches:<3} -> {c.grad_steps_per_million():.0f} grad steps / M env-steps"
        )
    print("⚠️  the SAME grid runs for every architecture -- that is the equal budget\n")

    if not a.stage_b:
        started = time.perf_counter()
        done = {(r["arch"], r["cadence"], r["shaping"], r["seed"]) for r in load_summary()}
        for arch, cadence, shaping_name, shaping in plan:
            for seed in range(a.seeds):
                key = (arch, cadence.name, shaping_name, seed)
                if key in done:
                    continue
                name = run_name(arch, cadence.name, shaping_name, seed)
                print(f"[{time.perf_counter() - started:7.0f}s] {name}", flush=True)
                train_one(a, arch, cadence, shaping_name, shaping, seed)
                if a.dry_run:
                    continue
                metrics = evaluate_one(a, name, eval_split=False)
                append(
                    {
                        "arch": arch,
                        "cadence": cadence.name,
                        "shaping": shaping_name,
                        "seed": seed,
                        "split": "train",
                        "env_steps": a.env_steps,
                        **asdict(cadence),
                        **metrics,
                    }
                )
        if not a.dry_run:
            print_table(
                report([r for r in load_summary() if r["split"] == "train"]),
                "STAGE A -- selection is on this table, train split",
            )
        return

    # --- stage B: each architecture's winner, more seeds, eval split ----------
    rows = [r for r in load_summary() if r["split"] == "train"]
    if not rows:
        raise SystemExit("no stage A results in the summary -- run without --stage-b first")
    for arch in architectures:
        ranked = report([r for r in rows if r["arch"] == arch])
        if not ranked:
            continue
        _, _, cell, _, median, iqr, _ = ranked[0]
        cadence = next(c for c in CADENCES if c.name == cell[1])
        shaping = dict(next(s for name, s in SHAPINGS if name == cell[2]))
        print(f"\n{arch}: winner {cell[1]}/{cell[2]} at {median * 100:.1f} % [{iqr * 100:.1f}]")
        for seed in range(a.final_seeds):
            name = run_name(arch, cadence.name, cell[2], seed)
            print(f"  {name}", flush=True)
            train_one(a, arch, cadence, cell[2], shaping, seed)
            if a.dry_run:
                continue
            metrics = evaluate_one(a, name, eval_split=True)
            append(
                {
                    "arch": arch,
                    "cadence": cadence.name,
                    "shaping": cell[2],
                    "seed": seed,
                    "split": "eval",
                    "env_steps": a.env_steps,
                    **asdict(cadence),
                    **metrics,
                }
            )
    if not a.dry_run:
        print_table(
            report([r for r in load_summary() if r["split"] == "eval"]),
            "STAGE B -- the reportable numbers, eval split",
        )


if __name__ == "__main__":
    main()
