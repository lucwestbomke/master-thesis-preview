"""Presentation rendering: eval videos for wandb, vector figures for the thesis.

The counterpart to `scripts/view_episode.py`, which stays the inspection tool.
The difference that matters is what gets drawn: this module draws **what the
policy did and why the mission succeeded or failed** -- the chain the router
chose, the rate against its threshold, the hop escalation, and which drone is
carrying the sensor. `view_episode.py` draws the map.

The observer panel is the one that did not exist before Block E. RQ3 asks
whether the observer role hands off and whether the handoff is anticipatory;
that is a picture, and if the handoff logic misbehaves this is the only place it
is visible before the metric silently reports a clean zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..env.core import DT_S, EPISODE_STEPS, HVT_Z_M, MCV_Z_M, SPAWN_RING_M, BatchedSwarmEnv
from ..env.reward import CAPACITY_THRESHOLD_MBPS
from .scene import COLOURS, HALF_M, draw_static_scene, load_artefact

OUTDIR = Path(__file__).resolve().parents[2] / ".cache" / "render"


@dataclass
class EpisodeTrace:
    """Everything a figure or a video needs, as host arrays."""

    route_idx: int
    policy: str
    mcv: np.ndarray  # (2,)
    hvt: np.ndarray  # (T, 2)
    pos: np.ndarray  # (T, N, 3)
    chain: np.ndarray  # (T, R, R) bool, R = N+1, MCV last
    capable: np.ndarray  # (T,)
    capacity: np.ndarray  # (T,)
    hops: np.ndarray  # (T,)
    sees: np.ndarray  # (T, N) bool
    occluded: np.ndarray  # (T,)
    # Which rung of Block F's ladder this was flown under. Part of the trace
    # because a figure of an F0 episode that does not say so is a figure of a
    # chain running through a tower with no explanation attached.
    fidelity: str = "F4"

    @property
    def observer(self) -> np.ndarray:
        """(T,) index of the drone carrying the sensor, -1 when nobody is.

        The longest-standing seer, which is the same rule
        `src/baselines/evaluate.py` uses for the handoff metrics -- so the
        picture and the number can never disagree.
        """
        t, n = self.sees.shape
        run = np.zeros(n, dtype=int)
        out = np.full(t, -1)
        for i in range(t):
            run = np.where(self.sees[i], run + 1, 0)
            if self.sees[i].any():
                out[i] = int(np.argmax(run + self.sees[i] * 10**6))
        return out


def pin_route(env: BatchedSwarmEnv, route_idx: int, cue_sigma_m: float = 150.0) -> None:
    """Force every environment onto one route, so the picture matches the story.

    `reset()` draws a random route; a figure captioned "route 1936" has to be
    route 1936. Mirrors what `_sample_episode` does rather than reaching into
    it, because the private method redraws the route it is being pinned away
    from.
    """
    b, n, dev = env.cfg.num_envs, env.cfg.num_drones, env.device
    env.route_id = torch.full_like(env.route_id, route_idx)
    env.mcv_pos = torch.cat(
        [env.route_mcv[env.route_id], torch.full((b, 1), MCV_Z_M, device=dev)], dim=-1
    )
    env.hvt_pos = torch.cat(
        [env.route_xy[env.route_id, 0], torch.full((b, 1), HVT_Z_M, device=dev)], dim=-1
    )
    env.hvt_vel = torch.zeros_like(env.hvt_pos)
    noise = torch.randn(b, 2, device=dev, generator=env.gen) * cue_sigma_m
    env.cue = torch.cat(
        [env.hvt_pos[:, :2] + noise, torch.full((b, 1), HVT_Z_M, device=dev)], dim=-1
    )
    phase = torch.rand(b, 1, device=dev, generator=env.gen) * 2 * torch.pi
    ring = phase + torch.arange(n, device=dev).unsqueeze(0) * (2 * torch.pi / n)
    from ..env.core import ALT_MIN_M

    env.drone_pos = torch.stack(
        [
            env.mcv_pos[:, None, 0] + SPAWN_RING_M * torch.cos(ring),
            env.mcv_pos[:, None, 1] + SPAWN_RING_M * torch.sin(ring),
            torch.full((b, n), ALT_MIN_M, device=dev),
        ],
        dim=-1,
    )
    env.drone_vel = torch.zeros_like(env.drone_pos)
    env.t = torch.zeros_like(env.t)
    env.steps_since_link = torch.zeros_like(env.steps_since_link)
    env.snap, _ = env._evaluate()


#: Scripted policy names. Anything else `_make_policy` sees is a checkpoint path.
POLICY_NAMES = ("random", "waypoint", "b0-geodesic", "b0", "b0-oracle")


def fly(
    route_idx: int,
    policy: str = "b0",
    num_drones: int = 5,
    seed: int = 0,
    steps: int = EPISODE_STEPS,
    fidelity: str = "F4",
    no_buildings: bool = False,
) -> EpisodeTrace:
    """Run one episode of the real env on one route and record it.

    `fidelity` selects the Block F rung. Flying the same route and the same
    policy at F0 and at F4 and putting the two figures side by side is the
    cheapest check that the ladder does what it claims -- under F0 the chain
    visibly runs straight through the tower cluster, which is the whole content
    of the hypothesis RQ1 tests.
    """
    from ..env.core import EnvConfig

    env = BatchedSwarmEnv(
        EnvConfig(
            num_envs=1,
            num_drones=num_drones,
            seed=seed,
            auto_reset=False,
            compile_occlusion=False,
            stage_weights=(0.0, 0.0, 0.0, 1.0),
            fidelity=fidelity,
            no_buildings=no_buildings,
        )
    )
    obs = env.reset()
    pin_route(env, route_idx)
    obs = env._observe(env._evaluate()[1])

    act_fn = _make_policy(policy, env, num_drones)
    rec: dict[str, list] = {
        k: [] for k in ("pos", "chain", "capable", "capacity", "hops", "sees", "occluded")
    }
    hvt = []
    for _ in range(steps):
        obs, _, _, _, ex = env.step(act_fn(obs))
        hvt.append(env.hvt_pos[0, :2].clone())
        rec["pos"].append(env.drone_pos[0].clone())
        rec["chain"].append(ex["on_edge"][0].clone())
        rec["capable"].append(ex["mission_capable"][0].clone())
        rec["capacity"].append(ex["e2e_capacity_mbps"][0].clone())
        rec["hops"].append(ex["hop_count"][0].clone())
        rec["sees"].append(ex["sees_hvt"][0].clone())
        rec["occluded"].append(ex["chain_occluded"][0].clone())

    def stack(key: str) -> np.ndarray:
        return torch.stack(rec[key]).cpu().numpy()

    return EpisodeTrace(
        route_idx=route_idx,
        policy=policy,
        mcv=env.mcv_pos[0, :2].cpu().numpy(),
        hvt=torch.stack(hvt).cpu().numpy(),
        pos=stack("pos"),
        chain=stack("chain"),
        capable=stack("capable"),
        capacity=stack("capacity"),
        hops=stack("hops"),
        sees=stack("sees"),
        occluded=stack("occluded"),
        fidelity="F0-nogeo" if no_buildings else fidelity,
    )


def _make_policy(name: str, env: BatchedSwarmEnv, n: int):
    """Any policy, one signature.

    Block G's addition: a **path to a checkpoint** is a policy name. The actor
    is evaluated at its distribution's mean rather than sampled, so the figure
    shows what the policy *does* and not what its exploration noise did on one
    roll. Everything else in the renderer is unchanged, exactly as intended.
    """
    from ..baselines import B0Policy

    if name not in POLICY_NAMES and Path(name).exists():
        import gymnasium
        import numpy as np

        from ..env.core import ACTION_DIM, FLAT_DIM
        from ..models import SwarmActor

        blob = torch.load(name, map_location=env.device, weights_only=False)
        obs_space = gymnasium.spaces.Box(-np.inf, np.inf, shape=(FLAT_DIM,), dtype=np.float32)
        act_space = gymnasium.spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,), dtype=np.float32)
        kw = {"architecture": blob["architecture"], "hidden": blob.get("hidden")}
        if blob.get("recurrent"):
            from ..models import SwarmActorRNN

            actor = SwarmActorRNN(
                obs_space,
                act_space,
                env.device,
                num_envs=n,
                rnn_hidden=blob.get("rnn_hidden", 128),
                sequence_length=blob.get("sequence_length", 16),
                **kw,
            ).to(env.device)
            hidden = [torch.zeros(actor.rnn_layers, n, actor.rnn_hidden, device=env.device)]
        else:
            actor = SwarmActor(obs_space, act_space, env.device, **kw).to(env.device)
            hidden = None
        actor.load_state_dict(blob["policy"])
        actor.eval()

        @torch.no_grad()
        def act_checkpoint(obs):
            inputs = {"observations": obs["flat"].reshape(n, FLAT_DIM)}
            if hidden is not None:
                inputs["rnn"] = hidden
            mean, extra = actor.compute(inputs)
            if hidden is not None:
                hidden[0] = extra["rnn"][0]
            return mean.view(1, n, ACTION_DIM)

        return act_checkpoint

    if name == "random":
        gen = torch.Generator(device=env.device).manual_seed(0)
        return lambda _obs: torch.empty(1, n, 3, device=env.device).uniform_(-1, 1, generator=gen)
    if name == "waypoint":
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        from measure_envelope import waypoint_policy

        return lambda _obs: waypoint_policy(env)

    variant = {"b0": "b0", "b0-geodesic": "geodesic", "b0-oracle": "oracle"}[name]
    pol = B0Policy(1, n, variant=variant, device=env.device)
    pol.reset()

    def act(obs):
        truth = None
        if variant == "oracle":
            truth = {
                "hvt_rel": env.hvt_pos.unsqueeze(1) - env.drone_pos,
                "hvt_vel": env.hvt_vel.unsqueeze(1).expand(-1, n, -1),
            }
        return pol.act(obs["flat"], truth)

    return act


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def figure(trace: EpisodeTrace, out: Path | None = None, art: dict | None = None):
    """Four-panel thesis figure. Vector by default -- pass a `.pdf` path."""
    import matplotlib.pyplot as plt

    art = art or load_artefact()
    t = np.arange(len(trace.capable)) * DT_S
    fig = plt.figure(figsize=(11, 13))
    gs = fig.add_gridspec(4, 1, height_ratios=[5, 1.3, 1.0, 1.2], hspace=0.28)
    ax = fig.add_subplot(gs[0])

    draw_static_scene(ax, art, road_colour=COLOURS["road_muted"])
    ax.plot(
        trace.hvt[:, 0],
        trace.hvt[:, 1],
        color=COLOURS["route"],
        lw=1.4,
        alpha=0.65,
        zorder=3,
        label="HVT route",
    )
    for i in range(trace.pos.shape[1]):
        ax.plot(
            trace.pos[:, i, 0],
            trace.pos[:, i, 1],
            lw=1.0,
            alpha=0.6,
            color=COLOURS["track"],
            zorder=4,
        )
    ax.plot(*trace.mcv, "k*", ms=20, zorder=7, label="MCV")
    ax.plot(
        trace.hvt[-1, 0],
        trace.hvt[-1, 1],
        "o",
        ms=11,
        color=COLOURS["hvt"],
        mec="k",
        zorder=8,
        label="HVT (final)",
    )

    nodes = np.vstack([trace.pos[-1, :, :2], trace.mcv[None]])
    for a, b in np.argwhere(trace.chain[-1]):
        ax.plot(
            [nodes[a, 0], nodes[b, 0]],
            [nodes[a, 1], nodes[b, 1]],
            color=COLOURS["chain"],
            lw=2.4,
            zorder=6,
        )
    ax.plot([], [], color=COLOURS["chain"], lw=2.4, label="relay chain (final)")
    ax.plot([], [], color=COLOURS["track"], lw=1.0, label="drone tracks")
    ax.set_xlim(-HALF_M, HALF_M)
    ax.set_ylim(-HALF_M, HALF_M)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(
        f"{trace.policy} @ {trace.fidelity} — route #{trace.route_idx} — "
        f"mission-capable {trace.capable.mean() * 100:.0f} % of {len(t) * DT_S:.0f} s"
    )
    ax.set_xlabel("metres east of box centre")
    ax.set_ylabel("metres north of box centre")

    # --- rate against the threshold it has to clear ------------------------
    axc = fig.add_subplot(gs[1])
    axc.plot(t, np.clip(trace.capacity, 0, 60), color="#2c3e50", lw=1.0)
    axc.axhline(
        CAPACITY_THRESHOLD_MBPS,
        color=COLOURS["blocked"],
        ls="--",
        lw=1.0,
        label=f"{CAPACITY_THRESHOLD_MBPS:.0f} Mbps requirement",
    )
    axc.fill_between(
        t,
        0,
        60,
        where=trace.capable.astype(bool),
        color=COLOURS["clear"],
        alpha=0.12,
        label="mission-capable",
    )
    axc.set_ylim(0, 60)
    axc.set_ylabel("e2e (Mbps)")
    axc.legend(fontsize=8, loc="upper right", ncol=2)

    # --- the escalation the episode is built around ------------------------
    axh = fig.add_subplot(gs[2])
    axh.step(t, trace.hops, where="post", color="#2c3e50", lw=1.0)
    axh.fill_between(
        t,
        0,
        trace.hops.max() + 1,
        where=trace.occluded.astype(bool),
        color=COLOURS["blocked"],
        alpha=0.15,
        label="chosen chain crosses a building",
    )
    axh.set_ylim(0, max(int(trace.hops.max()) + 1, 2))
    axh.set_ylabel("hops")
    axh.legend(fontsize=8, loc="upper left")

    # --- who is carrying the sensor: RQ3's picture -------------------------
    axo = fig.add_subplot(gs[3])
    obs_idx = trace.observer
    n = trace.sees.shape[1]
    for i in range(n):
        seen = np.where(trace.sees[:, i], i, np.nan)
        axo.plot(t, seen, lw=5, alpha=0.55, color="#b0b8c0", solid_capstyle="butt")
    axo.plot(t, np.where(obs_idx >= 0, obs_idx, np.nan), lw=2.0, color=COLOURS["observer"])
    handoffs = int(((obs_idx[1:] != obs_idx[:-1]) & (obs_idx[1:] >= 0) & (obs_idx[:-1] >= 0)).sum())
    axo.set_yticks(range(n))
    axo.set_ylim(-0.5, n - 0.5)
    axo.set_ylabel("drone")
    axo.set_xlabel("time (s)")
    axo.set_title(
        f"observer (purple) over every drone that can see (grey) — {handoffs} handoffs",
        fontsize=9,
    )
    for a in (axc, axh, axo):
        a.set_xlim(0, t[-1])

    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
    return fig


# --------------------------------------------------------------------------- #
# Video
# --------------------------------------------------------------------------- #


def _ffmpeg_available() -> bool:
    """imageio-ffmpeg is already a dependency and ships a binary, so mp4 works
    without asking anyone to install anything."""
    import shutil

    import matplotlib

    if shutil.which("ffmpeg"):
        return True
    try:
        import imageio_ffmpeg

        matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        return True
    except Exception:  # noqa: BLE001 - GIF always works
        return False


def animate(
    trace: EpisodeTrace,
    out: Path,
    fps: int = 25,
    stride: int = 4,
    zoom: bool = False,
    art: dict | None = None,
) -> Path:
    """Write an eval video. Returns the path, which is what a wandb call wants."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

    art = art or load_artefact()
    t_s = np.arange(len(trace.capable)) * DT_S
    fig, (ax, axc) = plt.subplots(2, 1, figsize=(10, 12), gridspec_kw={"height_ratios": [4, 1]})
    draw_static_scene(ax, art, road_colour=COLOURS["road_muted"])
    ax.plot(trace.hvt[:, 0], trace.hvt[:, 1], color=COLOURS["route"], lw=1.2, alpha=0.45, zorder=3)
    ax.plot(*trace.mcv, "k*", ms=20, zorder=7, label="MCV")
    (hvt_dot,) = ax.plot([], [], "o", ms=11, color=COLOURS["hvt"], mec="k", zorder=8, label="HVT")
    (drones,) = ax.plot(
        [], [], "^", ms=8, color=COLOURS["drone"], mec="k", zorder=7, label="drones"
    )
    (obs_dot,) = ax.plot(
        [], [], "^", ms=13, color=COLOURS["observer"], mec="k", zorder=9, label="observer"
    )
    chain_lines = [
        ax.plot([], [], color=COLOURS["chain"], lw=2.2, alpha=0.9, zorder=6)[0]
        for _ in range(trace.chain.shape[1])
    ]
    ax.plot([], [], color=COLOURS["chain"], lw=2.2, label="relay chain")
    title = ax.set_title("")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlabel("metres east of box centre")
    ax.set_ylabel("metres north of box centre")

    axc.plot(t_s, np.clip(trace.capacity, 0, 60), color="#2c3e50", lw=1.0)
    axc.axhline(CAPACITY_THRESHOLD_MBPS, color=COLOURS["blocked"], ls="--", lw=1.0)
    axc.fill_between(
        t_s,
        0,
        60,
        where=trace.capable.astype(bool),
        color=COLOURS["clear"],
        alpha=0.15,
        label="mission-capable",
    )
    (cursor,) = axc.plot([], [], color=COLOURS["hvt"], lw=1.5)
    axc.set_xlim(0, t_s[-1])
    axc.set_ylim(0, 60)
    axc.set_xlabel("time (s)")
    axc.set_ylabel("end-to-end (Mbps)")
    axc.legend(fontsize=8, loc="upper right")

    obs_idx = trace.observer

    def draw(i):
        p = trace.hvt[i]
        dp = trace.pos[i]
        hvt_dot.set_data([p[0]], [p[1]])
        drones.set_data(dp[:, 0], dp[:, 1])
        if obs_idx[i] >= 0:
            obs_dot.set_data([dp[obs_idx[i], 0]], [dp[obs_idx[i], 1]])
        else:
            obs_dot.set_data([], [])
        nodes = np.vstack([dp[:, :2], trace.mcv[None]])
        hops = np.argwhere(trace.chain[i])
        for k, line in enumerate(chain_lines):
            if k < len(hops):
                a, b = hops[k]
                line.set_data([nodes[a, 0], nodes[b, 0]], [nodes[a, 1], nodes[b, 1]])
            else:
                line.set_data([], [])
        cursor.set_data([t_s[i], t_s[i]], [0, 60])
        sep = float(np.linalg.norm(p - trace.mcv))
        state = "CAPABLE" if trace.capable[i] else "no feed"
        title.set_text(
            f"{trace.policy} @ {trace.fidelity}  route #{trace.route_idx}   t = {t_s[i]:5.1f} s   "
            f"separation {sep:4.0f} m   {trace.hops[i]} hops   "
            f"{trace.capacity[i]:5.1f} Mbps   {state}"
        )
        if zoom:
            ax.set_xlim(p[0] - 300, p[0] + 300)
            ax.set_ylim(p[1] - 300, p[1] + 300)
        else:
            ax.set_xlim(-HALF_M, HALF_M)
            ax.set_ylim(-HALF_M, HALF_M)
        return hvt_dot, drones, obs_dot, cursor, title

    anim = FuncAnimation(
        fig, draw, frames=range(0, len(t_s), stride), interval=1000 / fps, blit=False
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix != ".gif" and _ffmpeg_available():
        anim.save(out, writer=FFMpegWriter(fps=fps, bitrate=2400))
    else:
        out = out.with_suffix(".gif")
        anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out
