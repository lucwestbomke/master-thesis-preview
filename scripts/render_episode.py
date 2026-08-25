"""Presentation renderer: eval videos for wandb, vector figures for the thesis.

Not the inspection tool -- that is `view_episode.py`, which draws the map and
exists to find geometry bugs by looking. This one draws **what a policy did**:
the chain the router chose, the rate against its threshold, the hop escalation,
and which drone is carrying the sensor.

    uv run python scripts/render_episode.py --route 12 --policy b0
    uv run python scripts/render_episode.py --route 12 --policy b0 --video --zoom
    uv run python scripts/render_episode.py --compare --route 12
    uv run python scripts/render_episode.py --policy b0 --worst   # hardest route

Block G logs the video straight from the returned path:

    import wandb; wandb.log({"eval": wandb.Video(str(path))})

wandb is deliberately not imported here -- the renderer stays usable without it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.viz.episode import OUTDIR, animate, figure, fly
from src.viz.scene import inside_any_box, load_artefact

POLICIES = ("random", "waypoint", "b0-geodesic", "b0", "b0-oracle")


def compare_set(policy: str) -> tuple[str, ...]:
    """The policies `--compare` draws, given whatever `--policy` was asked for.

    ⚠️ `--compare` used to be `policies = POLICIES`, which **silently discarded
    `--policy`**. So

        render_episode.py --policy runs/<name>/checkpoint.pt --compare --route 12

    -- the exact command `BLOCK_G.md` recommends for turning an aggregate into a
    mechanism -- rendered the five scripted baselines and *not the checkpoint*,
    reporting success for all five. The learned policy is the one thing that
    command exists to look at.

    A checkpoint path is now appended to the baseline set rather than replacing
    or being replaced by it, because "compare" means *against* the baselines.
    """
    return POLICIES if policy in POLICIES else (*POLICIES, policy)


#: Block F rungs, plus the explicitly-named building-free variant. Same
#: policy, same route, five worlds -- the visual half of Block F.
FIDELITIES = ("F0", "F0-nogeo", "F1", "F2", "F3", "F4")


def worst_route(art: dict, stride: int = 8) -> int:
    """The route that spends the most time inside a building box.

    The same selector `view_episode.py --worst` uses. A renderer that only ever
    shows well-behaved routes is a renderer that never shows a problem.
    """
    frac = [
        inside_any_box(art["routes"][i], art["boxes"]).mean()
        for i in range(0, len(art["routes"]), stride)
    ]
    idx = int(np.argmax(frac) * stride)
    print(f"worst route is #{idx}: HVT inside a building for {max(frac):.1%} of the episode")
    return idx


def report(trace) -> None:
    hops = trace.hops
    chain = hops > 0
    obs = trace.observer
    handoffs = int(((obs[1:] != obs[:-1]) & (obs[1:] >= 0) & (obs[:-1] >= 0)).sum())
    print(
        f"  {trace.policy + '@' + trace.fidelity:<20} capable {trace.capable.mean() * 100:5.1f} %   "
        f"observed {trace.sees.any(-1).mean() * 100:5.1f} %   "
        f"chain occluded {trace.occluded.mean() * 100:5.1f} %"
    )
    print(
        f"  {'':<20} hops: median {np.median(hops[chain]) if chain.any() else 0:.0f}, "
        f"max {hops.max()}   e2e p5 {np.quantile(trace.capacity, 0.05):.1f} Mbps   "
        f"{handoffs} handoffs"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--route", type=int, default=0)
    ap.add_argument("--worst", action="store_true", help="pick the route most often indoors")
    ap.add_argument(
        "--policy",
        default="b0",
        help=f"one of {POLICIES}, or a path to a Block G checkpoint (runs/<name>/checkpoint.pt)",
    )
    ap.add_argument("--compare", action="store_true", help="one figure per policy, same route")
    ap.add_argument("--fidelity", choices=FIDELITIES, default="F4", help="Block F rung")
    ap.add_argument(
        "--compare-fidelity",
        action="store_true",
        help="one figure per Block F rung, same policy and route",
    )
    ap.add_argument("--drones", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video", action="store_true", help="also write an mp4")
    ap.add_argument("--zoom", action="store_true", help="video follows the HVT")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--stride", type=int, default=4, help="steps per rendered frame")
    ap.add_argument("--out", type=Path, default=OUTDIR)
    a = ap.parse_args()

    art = load_artefact()
    route = worst_route(art) if a.worst else min(a.route, len(art["routes"]) - 1)
    policies = compare_set(a.policy) if a.compare else (a.policy,)
    rungs = FIDELITIES if a.compare_fidelity else (a.fidelity,)

    print(f"route #{route}, N={a.drones}, seed {a.seed}")
    for rung in rungs:
        for name in policies:
            trace = fly(
                route,
                policy=name,
                num_drones=a.drones,
                seed=a.seed,
                fidelity="F0" if rung == "F0-nogeo" else rung,
                no_buildings=rung == "F0-nogeo",
            )
            report(trace)
            label = name if name in POLICIES else Path(name).parent.name
            stem = f"route{route}_{label}" + ("" if rung == "F4" else f"_{rung}")
            # Vector, because these go into the thesis. Raster only for the video.
            fig_path = a.out / f"{stem}.pdf"
            figure(trace, out=fig_path, art=art)
            print(f"    figure -> {fig_path}")
            if a.video:
                vid = animate(
                    trace,
                    a.out / f"{stem}.mp4",
                    fps=a.fps,
                    stride=a.stride,
                    zoom=a.zoom,
                    art=art,
                )
                print(f"    video  -> {vid}")


if __name__ == "__main__":
    main()
