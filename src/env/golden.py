"""The pre-Block-F behavioural trace: what `F4` has to reproduce, exactly.

Block F makes the env constructible at five channel fidelities, and `F4` **is**
the environment Blocks D and E measured. `docs/BLOCK_F.md`:

> `F4` must reproduce today's environment exactly -- it *is* the current
> environment. [...] If that test does not pass, every Block D and Block E
> number is invalidated.

So the check cannot be "the code looks equivalent". This module runs a fixed set
of scenarios through the env with a fixed action sequence and records every
tensor the env produces; `scripts/capture_f4_golden.py` freezes that into
`data/f4_golden.pt` and `test_golden.py` asserts the current code still
reproduces it, element for element.

**The capture was taken at 5ce0a2f, before any Block F change**, which is the
only moment at which it can be taken -- afterwards there is nothing left to
compare against. Regenerating it is therefore a deliberate act that discards the
evidence, not a routine step: see `scripts/capture_f4_golden.py`.

Why the scenario definitions live here rather than in the script
---------------------------------------------------------------
The capture and the check must run *identical* code, or the test degrades into
comparing two slightly different rollouts. One definition, two consumers.

Why the driver is written here and not borrowed
-----------------------------------------------
A golden driven by `B0` would couple this artefact to that policy's tuning, and
a later B0 change would surface as an env regression. So the driver is frozen in
this file and depends on nothing outside it.

It is a *stringer*: it servos drone `i` toward fraction `i` of the way along the
MCV -> HVT line, with a seeded lateral jitter and an oscillating altitude. That
is deliberately crude and it is **not a baseline** -- it reads `env.hvt_pos`
straight off the env, which `AGENTS.md` forbids B0 to do. A capture harness is
allowed to; a reported policy is not.

The point of stringing the swarm out is coverage. A first attempt fanned the
drones onto fixed bearings, and the resulting trace never once produced a
multi-hop chain while the jammer was on -- so it would have pinned neither the
`F3` nor the `F4` path, which are the two rungs Block F adds. The stringer puts
relays between the observer and the MCV, so the chain lengthens as the HVT
drives away and the trace covers 0-, 1-, 2- and 3-hop routing, occluded chains,
both altitude clamps and a jammed SINR denominator.
"""

from __future__ import annotations

import gzip
import io
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor

from .core import (
    ALT_MAX_M,
    ALT_MIN_M,
    DRONE_CRUISE_MS,
    DT_S,
    MAX_ACCEL_MS2,
    BatchedSwarmEnv,
    EnvConfig,
)

# Gzipped: the trace is mostly max-N padding zeros and gates that are off, so it
# compresses ~3x (3.6 -> 1.2 MB). Worth it for a file that lives in git forever.
GOLDEN_PATH = Path(__file__).resolve().parents[2] / "data" / "f4_golden.pt.gz"

# Bumped only when the *format* changes (scenarios, recorded keys, action rule).
# A mismatch means the artefact and this file disagree about what is compared,
# which is a different and louder failure than a behavioural regression.
GOLDEN_FORMAT = 1


@dataclass(frozen=True)
class Scenario:
    """One rollout to record. `cfg` is passed straight to `EnvConfig`."""

    name: str
    why: str
    steps: int
    action_seed: int
    cfg: dict = field(default_factory=dict)


#: Three rollouts, chosen so that between them every branch of `step()` runs.
#: Small batches, long horizons: occlusion cost scales with `num_envs`, and what
#: a golden needs is code-path coverage rather than route diversity. `design` at
#: 2 envs x 300 steps covers 0- through 5-hop routing; at 4 envs x 150 steps it
#: reached only 2 hops and cost three times as much.
GOLDEN_SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="design",
        why=(
            "Stage 4, the design condition: full HVT speed, jammer on, randomised "
            "initial charge, training route split. 300 steps is 120 s, far enough "
            "into the escalation that the trace covers every hop count the router "
            "can return (0-5), with the chain occluded on ~56 % of steps and "
            "mission-capable well below observed -- i.e. the relay chain binding, "
            "which is the regime Block E's numbers live in."
        ),
        steps=300,
        action_seed=101,
        cfg={"num_envs": 2, "num_drones": 5, "seed": 5, "stage_weights": (0.0, 0.0, 0.0, 1.0)},
    ),
    Scenario(
        name="reset",
        why=(
            "Stage 1 truncates at 150 steps, so 180 crosses an auto-reset boundary "
            "and exercises the second physics pass, `final_observation`, and "
            "Phi(fresh state) -- the PBRS convention `core.py`'s module docstring "
            "warns is easy to get backwards. Stage 1 also pins the jammer-OFF "
            "branch and a frozen HVT."
        ),
        steps=180,
        action_seed=202,
        cfg={"num_envs": 2, "num_drones": 5, "seed": 1, "stage_weights": (1.0, 0.0, 0.0, 0.0)},
    ),
    Scenario(
        name="offn_eval",
        why=(
            "N=3 on the held-out route split. Pins the max-N padding at N != 5 "
            "(where `_pack` has to zero four neighbour slots) and the eval-route "
            "index range, neither of which the other two touch."
        ),
        steps=120,
        action_seed=303,
        cfg={
            "num_envs": 2,
            "num_drones": 3,
            "seed": 3,
            "eval_routes": True,
            "stage_weights": (0.0, 0.0, 0.0, 1.0),
        },
    ),
)

#: Config the capture forces regardless of what a scenario asks for. CPU because
#: the artefact must be comparable on any machine; uncompiled because compilation
#: is a performance choice that must not sit between the golden and the code.
FORCED_CFG = {"device": "cpu", "compile_occlusion": False, "auto_reset": True}


#: How far along the MCV -> HVT line each drone is commanded to sit. Drone 0
#: goes over the target (Block E: a drone parked overhead sees it ~always, and
#: one parked 15 % back sees it 40 % of the time); the last stops short of the
#: MCV so no link has zero length.
_STATION_NEAR, _STATION_FAR = 1.0, 0.2
_JITTER_M = 90.0  # lateral spread, large enough that chains meet real buildings
_ALT_OVERSHOOT_M = 15.0  # command past both altitude limits, so the clamp fires


def golden_actions(step: int, env: BatchedSwarmEnv, gen: torch.Generator) -> Tensor:
    """`(B, N, 3)` actions from the frozen stringer. See the module docstring.

    A proportional servo: command a station, ask for the velocity that closes
    the gap at cruise, and accelerate toward it. Altitude sweeps the whole
    40-80 m band on a per-drone phase, so both clamps are hit and the A2A
    geometry keeps changing.
    """
    dev = env.device
    b, n = env.cfg.num_envs, env.cfg.num_drones

    frac = (
        torch.linspace(_STATION_NEAR, _STATION_FAR, n, device=dev)
        if n > 1
        else torch.full((1,), _STATION_NEAR, device=dev)
    )
    line = env.hvt_pos[:, :2] - env.mcv_pos[:, :2]  # (B, 2)
    perp = torch.stack([-line[:, 1], line[:, 0]], dim=-1)
    perp = perp / perp.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    # Jitter is redrawn every step, so the formation never settles into a
    # perfectly collinear chain -- which would make every link identical and
    # leave the occluded-link branch of the router unexercised.
    jitter = (torch.rand(b, n, generator=gen, device=dev) * 2.0 - 1.0) * _JITTER_M
    station_xy = (
        env.mcv_pos[:, None, :2]
        + frac[None, :, None] * line[:, None, :]
        + jitter.unsqueeze(-1) * perp[:, None, :]
    )

    # Commanded deliberately OUTSIDE the 40-80 m band. A servo that merely aims
    # at the ceiling asymptotes to 79.99 m and never trips it, leaving
    # `_advance_drones`'s limit clamp -- which zeroes the velocity component that
    # hit the wall, so the energy term is not charged for motion that did not
    # happen -- unexercised by the golden.
    phase = torch.arange(n, device=dev, dtype=torch.float32) * (2.0 * math.pi / n)
    sweep = 0.5 * (1.0 + torch.sin(step * (2.0 * math.pi / 80.0) + phase))  # in [0, 1]
    station_z = (ALT_MIN_M - _ALT_OVERSHOOT_M) + (
        (ALT_MAX_M - ALT_MIN_M) + 2.0 * _ALT_OVERSHOOT_M
    ) * sweep
    station = torch.cat([station_xy, station_z.unsqueeze(0).expand(b, n).unsqueeze(-1)], dim=-1)

    delta = station - env.drone_pos
    want_vel = delta * (DRONE_CRUISE_MS / delta.norm(dim=-1, keepdim=True).clamp_min(1e-6))
    want_vel = torch.where(
        delta.norm(dim=-1, keepdim=True) < DRONE_CRUISE_MS * DT_S, delta, want_vel
    )
    return ((want_vel - env.drone_vel) / (MAX_ACCEL_MS2 * DT_S)).clamp(-1.0, 1.0)


#: Env attributes recorded after every step, alongside the `step()` return and
#: `extras`. These are the state the next step is computed from, so a divergence
#: here localises a regression to the transition rather than to the observation.
_STATE_KEYS = (
    "drone_pos",
    "drone_vel",
    "last_accel",
    "battery",
    "hvt_pos",
    "hvt_vel",
    "mcv_pos",
    "cue",
    "t",
    "steps_since_link",
    "episode_len",
    "speed_scale",
    "jammer_on",
    "battery_scale",
    "route_id",
)


def run_scenario(scenario: Scenario, **cfg_overrides) -> dict[str, Tensor]:
    """Roll `scenario` out and return every tensor it produced, stacked on time.

    `cfg_overrides` exists so the *check* can construct the same rollout through
    a new seam (`fidelity="F4"`) that did not exist when the artefact was
    captured. It must never be used to change the scenario.
    """
    cfg_kw = {**scenario.cfg, **FORCED_CFG, **cfg_overrides}
    env = BatchedSwarmEnv(EnvConfig(**cfg_kw))

    gen = torch.Generator(device=env.device).manual_seed(scenario.action_seed)
    obs = env.reset()

    rec: dict[str, list[Tensor]] = {}

    def put(key: str, value: Tensor) -> None:
        rec.setdefault(key, []).append(value.detach().clone())

    put("reset/flat", obs["flat"])
    put("reset/state", obs["state"])

    for step in range(scenario.steps):
        actions = golden_actions(step, env, gen)
        put("action", actions)
        obs, rew, terminated, truncated, extras = env.step(actions)
        put("reward", rew)
        put("terminated", terminated)
        put("truncated", truncated)
        put("obs/flat", obs["flat"])
        put("obs/state", obs["state"])
        for key, value in extras.items():
            put(f"extras/{key}", value)
        for key in _STATE_KEYS:
            put(f"env/{key}", getattr(env, key))

    return {key: torch.stack(values) for key, values in rec.items()}


def run_all(**cfg_overrides) -> dict[str, dict[str, Tensor]]:
    """Every scenario, keyed by name."""
    return {s.name: run_scenario(s, **cfg_overrides) for s in GOLDEN_SCENARIOS}


def save_golden(traces: dict[str, dict[str, Tensor]], path: Path = GOLDEN_PATH) -> None:
    buf = io.BytesIO()
    torch.save({"format": GOLDEN_FORMAT, "traces": traces}, buf)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(buf.getvalue(), 9))


def load_golden(path: Path = GOLDEN_PATH) -> dict[str, dict[str, Tensor]]:
    """The frozen trace. Raises if the artefact predates the current format."""
    blob = torch.load(io.BytesIO(gzip.decompress(path.read_bytes())), weights_only=True)
    if blob["format"] != GOLDEN_FORMAT:
        raise ValueError(
            f"{path} is format {blob['format']}, this code expects {GOLDEN_FORMAT}. "
            "The artefact and the comparison disagree about what is being compared; "
            "see scripts/capture_f4_golden.py before re-capturing."
        )
    return blob["traces"]
