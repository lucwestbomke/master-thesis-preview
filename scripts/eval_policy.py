"""Score a trained checkpoint through the harness B0 was scored with.

    uv run python scripts/eval_policy.py runs/F4-mlp-s0/checkpoint.pt --device mps
    uv run python scripts/eval_policy.py --policy random --stage 1 --device mps
    uv run python scripts/eval_policy.py runs/*/checkpoint.pt --n 3 5 8

⚠️ This deliberately adds no metrics of its own. `src/baselines/evaluate.py`
already produces every number the thesis reports -- mission-capable,
`chain_occluded`, the hop histogram, capacity quantiles, the rate-division
counterfactual, the RQ3 anticipation lead -- and B0's 57.2 % came out of it. A
learned policy scored by a *different* loop is not comparable to B0, and that
comparison is the entire reason B0 exists.

Two rules the defaults enforce:

* **Evaluation is on the held-out routes** (`eval_routes=True`), which is the
  only generalisation check left now that the second city is cut.
* **Every RQ1 number is measured under `--fidelity F4`**, whatever the policy
  was trained under. That is the whole design of the ladder: train on a rung,
  sit the same exam.

Aggregation follows Block E and F: **means across episodes within a seed**
(`RolloutMetrics.summary`), **median + IQR across seeds**. A median within a
seed reports 0.0 % for every rare-event metric.

⚠️ A device is part of a measurement's provenance. `torch.Generator` streams
differ per device, so the same seed draws *different episodes* on MPS than on
CPU. Never compare a number from one device with a number from another.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines.b0 import B0Policy
from src.baselines.evaluate import RolloutMetrics, rollout
from src.env.core import STAGES, BatchedSwarmEnv, EnvConfig
from src.models import SwarmActor

#: Metrics stored as a fraction in [0, 1] and read as a percentage. ⚠️ Keep this
#: in step with REPORT: `capable_last_third` shipped without it and printed as
#: "0.4", one decimal of a fraction -- a +-5 pp resolution on the column the
#: late-episode question is decided by.
_AS_PERCENT = frozenset(
    {
        "mission_capable",
        "observed",
        "link_alive",
        "chain_occluded",
        "capable_last_third",
        "observed_last_third",
        "capable_share_high",
        "capable_share_low",
        # ⚠️ Reads BACKWARDS from the obvious intuition, and that is the point. The
        # rotary-wing power curve is U-shaped: cruising at 13.3 m/s costs 0.638 of
        # hover. So a drone that wanders burns LESS battery than one holding station,
        # and `battery_end` HIGHER than B0's is evidence the swarm is cruising to
        # stay cheap rather than holding a useful position.
        "battery_end",
    }
)

REPORT = (
    "mission_capable",
    "observed",
    "link_alive",
    "chain_occluded",
    "capacity_mean",
    "capacity_p5",
    "episode_return",
    "hop_mean",
    # RQ3's own metrics, reported here because they turned out to be the
    # diagnosis rather than a side result: a swarm that cannot hold the observer
    # role hands it over constantly and leaves gaps where nobody is looking.
    "handoffs",
    # Mean steps one drone holds the observer role: observed steps / tenures.
    # The clearest single number for "does anyone commit?", and it is what
    # separates the learned policies from B0 by an order of magnitude.
    "observer_tenure",
    # Role emergence (Block G): normalised entropy of observer identity, 0 = one
    # drone owns the role and 1 = every drone holds it equally. `standoff_gap_m`
    # is how much further back the swarm sits than its closest member -- large
    # means one drone went in and the rest held back to relay.
    "role_entropy",
    "standoff_gap_m",
    # ⚠️ The number the Block G diagnosis rests on: B0 parks its observer at
    # ~79 m, every learned policy loiters at ~291 m, and everything else
    # (observed, tenure, hop count) is a view of that one gap.
    "observer_range_m",
    "capable_last_third",
    "observer_range_last_third",
    "off_axis_m",
    "capable_share_high",
    "capable_share_low",
    # ⚠️ Reads BACKWARDS from the obvious intuition, and that is the point. The
    # rotary-wing power curve is U-shaped: cruising at 13.3 m/s costs 0.638 of
    # hover. So a drone that wanders burns LESS battery than one holding station,
    # and `battery_end` HIGHER than B0's is evidence the swarm is cruising to
    # stay cheap rather than holding a useful position.
    "battery_end",
)


def med_iqr(values: list[float]) -> tuple[float, float]:
    """Median and inter-quartile range. AGENTS.md: never mean +- std."""
    t = torch.tensor(values, dtype=torch.float64)
    return float(t.median()), float(t.quantile(0.75) - t.quantile(0.25))


def make_env(a, seed: int, num_drones: int) -> BatchedSwarmEnv:
    weights = tuple(1.0 if i == a.stage - 1 else 0.0 for i in range(len(STAGES)))
    return BatchedSwarmEnv(
        EnvConfig(
            num_envs=a.num_envs,
            num_drones=num_drones,
            device=a.device,
            seed=seed,
            fidelity=a.fidelity,
            eval_routes=not a.train_routes,
            auto_reset=False,  # one episode per environment: clean metric rows
            stage_weights=weights,
            compile_occlusion=a.device != "cpu",
        )
    )


def load_actor(path: Path, env: BatchedSwarmEnv) -> tuple[SwarmActor, dict]:
    blob = torch.load(path, map_location=env.device, weights_only=False)
    import gymnasium
    import numpy as np

    from src.env.core import ACTION_DIM, FLAT_DIM

    obs_space = gymnasium.spaces.Box(-np.inf, np.inf, shape=(FLAT_DIM,), dtype=np.float32)
    act_space = gymnasium.spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,), dtype=np.float32)
    rows = env.cfg.num_envs * env.cfg.num_drones
    if blob.get("recurrent"):
        from src.models import SwarmActorRNN

        actor = SwarmActorRNN(
            obs_space,
            act_space,
            env.device,
            architecture=blob["architecture"],
            hidden=blob.get("hidden"),
            num_envs=rows,
            rnn_hidden=blob.get("rnn_hidden", 128),
            sequence_length=blob.get("sequence_length", 16),
        ).to(env.device)
    else:
        actor = SwarmActor(
            obs_space,
            act_space,
            env.device,
            architecture=blob["architecture"],
            hidden=blob.get("hidden"),
        ).to(env.device)
    actor.load_state_dict(blob["policy"])
    actor.eval()
    return actor, blob


def score(a, name: str, checkpoint: Path | None, num_drones: int) -> dict[str, list[float]]:
    cols: dict[str, list[float]] = {}
    steps = STAGES[a.stage - 1].episode_steps

    for seed in range(a.seeds):
        env = make_env(a, seed, num_drones)
        b, n = env.cfg.num_envs, env.cfg.num_drones
        on_reset = None

        if name == "random":
            gen = torch.Generator(device=env.device).manual_seed(seed)

            def policy(_obs, _b=b, _n=n, _gen=gen, _env=env):
                return torch.empty(_b, _n, 3, device=_env.device).uniform_(-1, 1, generator=_gen)

        elif name == "b0":
            pol = B0Policy(b, n, variant="b0", device=env.device)
            on_reset = pol.reset

            def policy(obs, _pol=pol):
                return _pol.act(obs["flat"])

        else:
            actor, blob = load_actor(checkpoint, env)
            # A recurrent actor carries hidden state across the episode, which is
            # the whole point of it -- so the evaluator has to carry it too. The
            # rollout is one episode per environment (auto_reset off), so a single
            # zeroed state at the start is the correct initialisation.
            state = (
                [torch.zeros(actor.rnn_layers, b * n, actor.rnn_hidden, device=env.device)]
                if blob.get("recurrent")
                else None
            )

            @torch.no_grad()
            def policy(obs, _actor=actor, _b=b, _n=n, _h=state):
                # The deterministic action: the Gaussian's mean, not a sample.
                # Evaluating a stochastic policy by sampling would report the
                # exploration noise as part of the result.
                flat = obs["flat"].reshape(_b * _n, -1)
                inputs = {"observations": flat}
                if _h is not None:
                    inputs["rnn"] = _h
                mean, extra = _actor.compute(inputs)
                if _h is not None:
                    _h[0] = extra["rnn"][0]
                return mean.view(_b, _n, 3)

        metrics: RolloutMetrics = rollout(env, policy, steps, on_reset=on_reset)
        summary = metrics.summary()
        summary["observer_tenure"] = summary["observed"] * steps / (summary["handoffs"] + 1.0)
        summary["hop_mean"] = float(
            (metrics.hop_hist * torch.arange(metrics.hop_hist.shape[-1])).sum(-1).mean()
        )
        for key in REPORT:
            cols.setdefault(key, []).append(summary[key])
    return cols


def _append(path: Path, row: dict) -> None:
    """One JSONL row, appended. Same contract as `sweep.py`'s summary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoints", nargs="*", type=Path)
    ap.add_argument("--policy", nargs="*", default=[], choices=["random", "b0"])
    ap.add_argument("--fidelity", default="F4", choices=["F0", "F1", "F2", "F3", "F4"])
    ap.add_argument("--stage", type=int, default=4, help="curriculum stage to evaluate at")
    ap.add_argument("--n", nargs="*", type=int, default=[5], help="swarm sizes (zero-shot)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--train-routes", action="store_true", help="tune on these, never report them")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="ALSO append one JSONL row per (policy, N) here, carrying the PER-SEED "
        "values and not just the median. Point it at results/ (tracked) rather than "
        "runs/ (gitignored): the summary is a result, the checkpoints are regenerable, "
        "so the machine that ran it can `git push` and the machine that analyses it "
        "can `git pull`. Per-seed matters -- BLOCK_G_PLAN's gate rules are declared "
        "on the WORST seed, and a median alone cannot be judged against them.",
    )
    ap.add_argument(
        "--group",
        default=None,
        help="treat the given checkpoints as SEEDS OF ONE CONDITION: score each on the "
        "same episodes and report median [IQR] across them. This is the aggregation "
        "AGENTS.md asks for -- means across episodes within a seed, median across seeds "
        "-- and it is the only one that reports training-seed variation. Without it, "
        "--seeds varies the EVALUATION episodes of a single policy, which says nothing "
        "about whether a second training run would have landed anywhere near the first.",
    )
    a = ap.parse_args()

    if a.seeds < 5:
        print(f"⚠️  {a.seeds} seeds. AGENTS.md requires >=5 for anything reported as a finding.\n")

    split = "train" if a.train_routes else "eval"
    print(
        f"device={a.device}  fidelity={a.fidelity}  stage={a.stage}  routes={split}  "
        f"{a.seeds} seeds x {a.num_envs} episodes; median [IQR] across seeds"
    )
    print("⚠️  device is part of provenance: never compare across devices.\n")

    header = f"{'policy':<24}{'N':>3}" + "".join(f"{k.replace('_', ' '):>18}" for k in REPORT)
    print(header)
    print("-" * len(header))

    jobs: list[tuple[str, Path | None]] = [(p, None) for p in a.policy]
    if a.group:
        jobs.append((a.group, None))  # handled below, from a.checkpoints
    else:
        jobs += [(cp.parent.name, cp) for cp in a.checkpoints]

    for name, checkpoint in jobs:
        for n in a.n:
            if a.group and name == a.group:
                # One row per training seed, all on the SAME episodes (paired), so
                # the spread reported is training-seed variation and nothing else.
                paired = argparse.Namespace(**{**vars(a), "seeds": 1})
                cols = {}
                for cp in a.checkpoints:
                    for key, values in score(paired, cp.parent.name, cp, n).items():
                        cols.setdefault(key, []).extend(values)
            else:
                cols = score(a, name, checkpoint, n)
            row = f"{name:<24}{n:>3}"
            for key in REPORT:
                m, i = med_iqr(cols[key])
                scale = 100.0 if key in _AS_PERCENT else 1.0
                unit = " %" if scale == 100.0 else ""
                row += f"{m * scale:>12.1f}{unit:<2}[{i * scale:.1f}]".rjust(18)
            print(row)

            if a.out:
                _append(
                    a.out,
                    {
                        "policy": name,
                        "n": n,
                        "split": split,
                        "stage": a.stage,
                        "fidelity": a.fidelity,
                        "device": a.device,
                        "num_envs": a.num_envs,
                        "grouped": bool(a.group and name == a.group),
                        "checkpoints": [str(cp) for cp in a.checkpoints]
                        if a.group and name == a.group
                        else ([str(checkpoint)] if checkpoint else []),
                        # ⚠️ Per-seed, not just the aggregate. `median [IQR]` cannot be
                        # judged against a rule declared on the worst seed, and every
                        # gate rule in BLOCK_G_PLAN.md is.
                        "seeds": {key: cols[key] for key in REPORT},
                        "median": {key: med_iqr(cols[key])[0] for key in REPORT},
                        "iqr": {key: med_iqr(cols[key])[1] for key in REPORT},
                    },
                )


if __name__ == "__main__":
    main()
