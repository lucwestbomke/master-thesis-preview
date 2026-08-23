"""Watch an episode: buildings, roads, MCV, HVT, and live line-of-sight.

An inspection tool for `data/frankfurt_box.npz`, not the Block E presentation
renderer. It exists because every map problem so far (axis-aligned boxes filling
94 % of the city, boxes swallowing road network, LoD2 bridge decks duplicating
the road surface) was found by happening to compute the right statistic. Looking
at the thing is cheaper.

It draws exactly what the env will consume -- the oriented boxes, not the
original polygons -- so what you see is what occlusion tests against. The
MCV-to-HVT ray is coloured by the real `src/env/occlusion.py` clearance, so a
blocked ray on screen is a blocked link in training.

Usage:
    uv run python scripts/view_episode.py --route 0
    uv run python scripts/view_episode.py --route 1936 --save ep.mp4
    uv run python scripts/view_episode.py --worst          # most time indoors
    uv run python scripts/view_episode.py --route 3 --zoom
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling scripts
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # src package

from src.env.occlusion import segment_clearance

# Shared with scripts/render_episode.py. Extracted rather than duplicated: the
# reason this drawing code is trusted is that it draws the ORIENTED BOXES the
# env consumes, and two copies of that would drift apart (docs/BLOCK_E.md §8).
from src.viz.scene import (
    draw_static_scene,
    inside_any_box,
)

ARTEFACT = Path(__file__).resolve().parent.parent / "data" / "frankfurt_box.npz"
OUTDIR = Path(__file__).resolve().parent.parent / ".cache" / "view"

HALF = 750.0
DT_S = 0.4
HVT_Z = 1.5  # a vehicle
MCV_Z = 2.0


def _use_bundled_ffmpeg() -> bool:
    """Point matplotlib at imageio-ffmpeg's binary if there is no system one.

    `ffmpeg` is usually not on PATH, but `imageio-ffmpeg` is already a
    dependency and ships one, so mp4 works without asking anyone to install
    anything.
    """
    import shutil

    import matplotlib

    if shutil.which("ffmpeg"):
        return True
    try:
        import imageio_ffmpeg

        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        return True
    except Exception:  # noqa: BLE001 - fall back to GIF
        return False


def clearance_series(mcv: np.ndarray, traj: np.ndarray, boxes, heights) -> np.ndarray:
    """Signed MCV->HVT clearance at every step, from the production kernel."""
    n = len(traj)
    p0 = torch.tensor(np.c_[np.repeat(mcv[None], n, 0), np.full(n, MCV_Z)], dtype=torch.float64)
    p1 = torch.tensor(np.c_[traj, np.full(n, HVT_Z)], dtype=torch.float64)
    return segment_clearance(
        p0, p1, torch.tensor(boxes, dtype=torch.float64), torch.tensor(heights, dtype=torch.float64)
    ).numpy()


def fly_swarm(route_idx: int, num_drones: int = 5) -> dict:
    """Run one episode of the real env on this route and record what to draw.

    Uses the crude waypoint policy from `measure_envelope.py`, NOT B0 -- this is
    an inspection aid, and the presentation renderer plus the real scripted
    baseline are Block E. The point is that every geometry problem in this
    project so far was found by looking, and Block D added a lot of geometry:
    chain selection, sensor gating, the altitude band.
    """
    import torch
    from measure_envelope import waypoint_policy

    from src.env.core import EPISODE_STEPS, BatchedSwarmEnv, EnvConfig

    env = BatchedSwarmEnv(
        EnvConfig(
            num_envs=1,
            num_drones=num_drones,
            seed=0,
            auto_reset=False,
            compile_occlusion=False,
            stage_weights=(0.0, 0.0, 0.0, 1.0),
        )
    )
    env.reset()
    # Pin the episode to the route being viewed, so the picture matches the
    # trajectory drawn from the artefact rather than a random other one.
    env.route_id = torch.full_like(env.route_id, route_idx)
    env.mcv_pos[:, :2] = env.route_mcv[env.route_id]
    env.hvt_pos[:, :2] = env.route_xy[env.route_id, 0]
    env.drone_pos[:, :, 0] = env.mcv_pos[:, None, 0]
    env.drone_pos[:, :, 1] = env.mcv_pos[:, None, 1]
    env.snap, _ = env._evaluate()

    pos, chain, capable = [], [], []
    for _ in range(EPISODE_STEPS):
        _, _, _, _, ex = env.step(waypoint_policy(env))
        pos.append(env.drone_pos[0].clone())
        chain.append(ex["on_edge"][0].clone())
        capable.append(bool(ex["mission_capable"][0]))
    return {
        "pos": torch.stack(pos).numpy(),  # (T, N, 3)
        "chain": torch.stack(chain).numpy(),  # (T, R, R) bool, R = N+1, MCV last
        "capable": np.asarray(capable),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", type=int, default=0)
    ap.add_argument("--worst", action="store_true", help="pick the route most often indoors")
    ap.add_argument("--save", type=str, default=None, help="output .mp4 or .gif")
    ap.add_argument("--zoom", action="store_true", help="follow the HVT instead of the whole box")
    ap.add_argument("--stride", type=int, default=4, help="steps per rendered frame")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument(
        "--drones",
        action="store_true",
        help="fly the swarm through the env and overlay it plus the chosen relay chain",
    )
    ap.add_argument(
        "--polygons",
        action="store_true",
        help="outline the source LoD2 footprints over the boxes, to see the approximation",
    )
    args = ap.parse_args()

    art = np.load(ARTEFACT)
    boxes = art["building_boxes"].astype(np.float64)
    heights = art["building_heights"].astype(np.float64)
    nodes = art["road_nodes"].astype(np.float64)
    edges = art["road_edges"]
    mcvs = art["route_mcv"].astype(np.float64)
    routes = art["route_xy"].astype(np.float64)

    idx = args.route
    if args.worst:
        frac = [inside_any_box(routes[i], boxes).mean() for i in range(0, len(routes), 8)]
        idx = int(np.argmax(frac) * 8)
        print(f"worst route is #{idx}: inside a building for {max(frac):.1%} of the episode")
    idx = min(idx, len(routes) - 1)

    swarm = fly_swarm(idx) if args.drones else None

    mcv, traj = mcvs[idx], routes[idx]
    indoors = inside_any_box(traj, boxes)
    clear = clearance_series(mcv, traj, boxes, heights)
    dist = np.linalg.norm(traj - mcv, axis=1)

    print(f"route #{idx}")
    print(f"  MCV at ({mcv[0]:+.0f}, {mcv[1]:+.0f})")
    print(f"  separation {dist[0]:.0f} m -> {dist[-1]:.0f} m")
    print(f"  steps inside a building: {indoors.sum()} / {len(traj)} ({indoors.mean():.1%})")
    print(f"  MCV->HVT direct LoS clear on {100 * (clear >= 0).mean():.0f}% of steps")
    if swarm is not None:
        print(f"  swarm mission-capable on {100 * swarm['capable'].mean():.0f}% of steps")

    # ---- figure ----------------------------------------------------------
    fig, (ax, axc) = plt.subplots(2, 1, figsize=(10, 12), gridspec_kw={"height_ratios": [4, 1]})

    draw_static_scene(
        ax,
        {"boxes": boxes, "heights": heights, "nodes": nodes, "edges": edges},
        polygons=args.polygons,
    )

    ax.plot(traj[:, 0], traj[:, 1], color="#f39c12", lw=1.2, alpha=0.5, zorder=3)
    ax.plot(*mcv, "k*", ms=20, zorder=6, label="MCV")
    (hvt_dot,) = ax.plot([], [], "o", ms=11, color="#e74c3c", mec="k", zorder=7, label="HVT")
    (ray,) = ax.plot([], [], lw=2.0, zorder=6)
    (trail,) = ax.plot([], [], color="#e67e22", lw=2.5, zorder=5)
    drone_dots = None
    chain_lines: list = []
    if swarm is not None:
        (drone_dots,) = ax.plot(
            [], [], "^", ms=8, color="#2980b9", mec="k", zorder=7, label="drones"
        )
        # One line per possible hop; a chain is at most R-1 = num_drones long.
        chain_lines = [
            ax.plot([], [], color="#27ae60", lw=2.0, alpha=0.9, zorder=6)[0]
            for _ in range(swarm["chain"].shape[1])
        ]
        ax.plot([], [], color="#27ae60", lw=2.0, label="relay chain")
    title = ax.set_title("")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlabel("metres east of box centre")
    ax.set_ylabel("metres north of box centre")

    axc.plot(np.arange(len(clear)) * DT_S, np.clip(clear, -80, 120), color="#2c3e50", lw=1.0)
    axc.axhline(0, color="#c0392b", lw=1.0, ls="--")
    axc.fill_between(
        np.arange(len(clear)) * DT_S,
        -80,
        120,
        where=indoors,
        color="#c0392b",
        alpha=0.25,
        label="HVT inside a building box",
    )
    (cursor,) = axc.plot([], [], color="#e74c3c", lw=1.5)
    axc.set_xlim(0, len(clear) * DT_S)
    axc.set_ylim(-80, 120)
    axc.set_xlabel("time (s)")
    axc.set_ylabel("MCV→HVT clearance (m)")
    axc.legend(fontsize=8, loc="upper right")

    frames = range(0, len(traj), args.stride)

    def draw(i):
        p = traj[i]
        hvt_dot.set_data([p[0]], [p[1]])
        trail.set_data(traj[max(0, i - 100) : i + 1, 0], traj[max(0, i - 100) : i + 1, 1])
        blocked = clear[i] < 0
        ray.set_data([mcv[0], p[0]], [mcv[1], p[1]])

        if swarm is not None:
            dp = swarm["pos"][i]
            drone_dots.set_data(dp[:, 0], dp[:, 1])
            # node R-1 is the MCV; everything below it is a drone
            nodes_xy = np.vstack([dp[:, :2], mcv[None, :2]])
            hops = np.argwhere(swarm["chain"][i])
            for k, line in enumerate(chain_lines):
                if k < len(hops):
                    a, b = hops[k]
                    line.set_data(
                        [nodes_xy[a, 0], nodes_xy[b, 0]], [nodes_xy[a, 1], nodes_xy[b, 1]]
                    )
                else:
                    line.set_data([], [])
        ray.set_color("#c0392b" if blocked else "#27ae60")
        ray.set_alpha(0.8 if blocked else 0.9)
        cursor.set_data([i * DT_S, i * DT_S], [-80, 120])
        state = "BLOCKED" if blocked else "clear"
        extra = "  [HVT inside a building box]" if indoors[i] else ""
        title.set_text(
            f"route #{idx}   t = {i * DT_S:5.1f} s   separation {dist[i]:4.0f} m   "
            f"direct LoS {state} ({clear[i]:+.0f} m){extra}"
        )
        if args.zoom:
            ax.set_xlim(p[0] - 250, p[0] + 250)
            ax.set_ylim(p[1] - 250, p[1] + 250)
        else:
            ax.set_xlim(-HALF, HALF)
            ax.set_ylim(-HALF, HALF)
        return hvt_dot, ray, trail, cursor, title

    anim = FuncAnimation(fig, draw, frames=frames, interval=1000 / args.fps, blit=False)
    fig.tight_layout()

    if args.save:
        OUTDIR.mkdir(parents=True, exist_ok=True)
        out = OUTDIR / args.save
        if out.suffix != ".gif" and _use_bundled_ffmpeg():
            anim.save(out, writer=FFMpegWriter(fps=args.fps, bitrate=2400))
        else:
            # no ffmpeg anywhere: GIF always works, Pillow is already a dep
            out = out.with_suffix(".gif")
            anim.save(out, writer=PillowWriter(fps=args.fps))
        print(f"saved -> {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
