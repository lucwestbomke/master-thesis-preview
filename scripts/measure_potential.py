"""Measure `Phi` over the states a policy actually visits -- the instrument the
`Phi` rebuild is judged on.

Every `Phi`-side intervention in Block G so far was sized by intuition and
returned a null, and `docs/REWARD.md` eventually explained why: measured across
the whole operating band the shipped potential moves a **total of 0.32**, while
the objective's energy term pays **0.054 per step** to cruise rather than hold
station. The shaping was not badly tuned, it was switched off.

That comparison is the only one that matters, and nothing in this repo could
compute it. This script does:

1. **Bank the states.** Roll a policy through the real env and keep exactly the
   fields `reward.potential` reads -- `nearest_dist_m`, `best_clearance_m`,
   `observer_dist_m`, `e2e_capacity_mbps` -- plus the diagnostics that say where
   in the band each step sat. Read straight off `env.snap`, so this measures the
   potential the env computes rather than a reimplementation of it.
2. **Score a candidate.** Evaluate any `RewardWeights` over the bank and report
   the two numbers a design target can be written in:
   * **swing** -- `p99 - p1` of `Phi` across visited states, and
   * **per-step gradient** -- the distribution of `|Phi(s_t+1) - Phi(s_t)|`,
     against `ENERGY_STEP_DIFFERENTIAL = 0.054`.

⚠️ A bank is device- and policy-specific. `torch.Generator` streams differ per
device, so a bank collected on MPS is not comparable with one collected on CUDA
-- the physics is identical, the episodes are not.

    uv run python scripts/measure_potential.py --policy b0 --device mps
    uv run python scripts/measure_potential.py --policy b0 random --candidates
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines.b0 import B0Policy
from src.env.core import STAGES, BatchedSwarmEnv, EnvConfig
from src.env.energy import DEFAULT_AIRFRAME, total_power_w
from src.env.reward import DEFAULT_WEIGHTS, PHI_V2, RewardWeights, Snapshot, potential

#: What the objective pays, per step, for cruising at the minimum-power airspeed
#: instead of holding station. 📏 `docs/REWARD.md`, measured 2026-08-26. This is
#: the bar `Phi`'s per-step gradient has to clear: a drone weighing "hold this
#: sightline" against "keep moving" sees this much certain reward for moving.
ENERGY_STEP_DIFFERENTIAL = 0.054

#: Fields of `Snapshot` that `potential()` reads. Banking exactly these keeps the
#: instrument honest -- anything a candidate `Phi` wants that is not here has to
#: be added to `Snapshot` and to the env, not smuggled into the script.
BANKED = ("nearest_dist_m", "best_clearance_m", "observer_dist_m", "e2e_capacity_mbps")
#: `Phi_cover` is a function of where every drone is, so the bank has to carry
#: the geometry too -- that is the whole point of the component.
BANKED_GEOMETRY = ("drone_pos", "mcv_pos", "hvt_pos")

#: The candidates this script knows how to score. `shipped` is the control.
CANDIDATES: dict[str, RewardWeights] = {"shipped": DEFAULT_WEIGHTS, "v2": PHI_V2}


def energy_differential(craft=DEFAULT_AIRFRAME) -> float:
    """Recompute `ENERGY_STEP_DIFFERENTIAL` from the airframe, so the bar cannot
    silently drift away from the power curve it is derived from."""
    from src.env.reward import hover_reference_power_w

    p_ref = hover_reference_power_w(craft)
    zero = torch.tensor(0.0)
    hover = total_power_w(zero, zero, craft).item() / p_ref
    cruise = total_power_w(torch.tensor(13.3), zero, craft).item() / p_ref
    return DEFAULT_WEIGHTS.energy * (hover - cruise)


@torch.no_grad()
def bank_states(
    policy_name: str,
    checkpoint: Path | None,
    *,
    device: str,
    num_envs: int,
    num_drones: int,
    seed: int,
    stage: int,
    fidelity: str,
    train_routes: bool,
) -> dict[str, Tensor]:
    """`(steps, B)` tensors of everything `Phi` reads, from a real rollout."""
    weights = tuple(1.0 if i == stage - 1 else 0.0 for i in range(len(STAGES)))
    env = BatchedSwarmEnv(
        EnvConfig(
            num_envs=num_envs,
            num_drones=num_drones,
            device=device,
            seed=seed,
            fidelity=fidelity,
            eval_routes=not train_routes,
            auto_reset=False,
            stage_weights=weights,
            compile_occlusion=device != "cpu",
        )
    )
    steps = STAGES[stage - 1].episode_steps
    b, n = env.cfg.num_envs, env.cfg.num_drones

    obs = env.reset()
    on_reset = None
    if policy_name == "random":
        gen = torch.Generator(device=env.device).manual_seed(seed)

        def act(_obs):
            return torch.empty(b, n, 3, device=env.device).uniform_(-1, 1, generator=gen)

    elif policy_name == "b0":
        pol = B0Policy(b, n, variant="b0", device=env.device)
        on_reset = pol.reset
        on_reset(torch.ones(b, dtype=torch.bool, device=env.device))

        def act(o):
            return pol.act(o["flat"])

    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from eval_policy import load_actor

        actor, blob = load_actor(checkpoint, env)
        if blob.get("recurrent"):
            raise SystemExit("recurrent checkpoints are not banked: state handling differs")

        def act(o):
            mean, _ = actor.compute({"observations": o["flat"].reshape(b * n, -1)})
            return mean.view(b, n, 3)

    cols: dict[str, list[Tensor]] = {
        k: [] for k in (*BANKED, *BANKED_GEOMETRY, "observed", "capable", "speed")
    }
    for _ in range(steps):
        obs, _rew, _term, _trunc, ex = env.step(act(obs))
        snap = env.snap
        for k in (*BANKED, *BANKED_GEOMETRY):
            cols[k].append(getattr(snap, k).clone())
        cols["observed"].append(snap.observed.clone())
        cols["capable"].append(ex["mission_capable"].clone())
        cols["speed"].append(snap.speed_ms.clone())  # (B, N) -- the energy term is per-drone
    return {k: torch.stack(v) for k, v in cols.items()}


def bank_snapshot(bank: dict[str, Tensor]) -> Snapshot:
    """A `(steps*B,)` Snapshot, so `potential()` scores the whole bank at once.

    `battery`, `speed_ms` and `accel_ms2` are dummies: `potential()` reads none
    of them, and giving them real values would invite a candidate `Phi` to read
    a field this instrument is not actually banking.
    """
    flat = {k: bank[k].reshape(-1) for k in BANKED}
    geom = {k: bank[k].flatten(0, 1) for k in BANKED_GEOMETRY}
    zeros = torch.zeros(flat["nearest_dist_m"].shape[0], 1, device=flat["nearest_dist_m"].device)
    return Snapshot(
        observed=bank["observed"].reshape(-1),
        battery=zeros,
        speed_ms=zeros,
        accel_ms2=zeros,
        **flat,
        **geom,
    )


def phi_of_bank(bank: dict[str, Tensor], w: RewardWeights) -> Tensor:
    """`(steps, B)` potential over the banked states."""
    shape = bank["nearest_dist_m"].shape
    return potential(bank_snapshot(bank), w).reshape(shape)


def q(t: Tensor, *qs: float) -> list[float]:
    finite = t[torch.isfinite(t)].float()
    return [float(finite.quantile(x)) for x in qs]


def report(name: str, bank: dict[str, Tensor], w: RewardWeights, gamma: float = 0.999) -> None:
    phi = phi_of_bank(bank, w)
    d_phi = phi[1:] - phi[:-1]
    # What the learner actually receives, decay included: PBRS pays
    # `gamma*Phi(s') - Phi(s)`, so a policy that holds a high-potential state
    # still pays `(gamma-1)*Phi` every step. At k=10 that is -0.01/step, and it
    # is the reason `Phi` cannot simply be scaled up without limit.
    shaping = gamma * phi[1:] - phi[:-1]
    p1, p50, p99 = q(phi, 0.01, 0.5, 0.99)
    bar = energy_differential()

    print(f"\n  {name}")
    print(f"    Phi         p1 {p1:7.3f}   p50 {p50:7.3f}   p99 {p99:7.3f}   swing {p99 - p1:7.3f}")
    g50, g90, g99 = q(d_phi.abs(), 0.5, 0.9, 0.99)
    print(f"    |dPhi|/step p50 {g50:7.4f}   p90 {g90:7.4f}   p99 {g99:7.4f}   bar {bar:6.4f}")
    print(
        f"                p90 / bar = {g90 / bar:5.2f}x      share of steps over bar: "
        f"{float((d_phi.abs() > bar).float().mean()) * 100:5.1f} %"
    )
    hold = float(shaping[d_phi.abs() < 1e-6].mean()) if (d_phi.abs() < 1e-6).any() else float("nan")
    print(
        f"    gamma decay while stationary: {hold:+.4f}/step "
        f"(vs {-bar:+.4f} for holding instead of cruising)"
    )

    for term, val in term_values(bank, w).items():
        lo, hi = q(val, 0.01, 0.99)
        print(f"      {term:<10} p1 {lo:7.3f}  p99 {hi:7.3f}  swing {hi - lo:7.3f}")


def term_values(bank: dict[str, Tensor], w: RewardWeights) -> dict[str, Tensor]:
    """Each weighted component of `Phi`, so a saturated term is visible."""
    terms = ("w_approach", "w_observe", "w_standoff", "w_link", "w_cover")
    out = {}
    for term in terms:
        if getattr(w, term) <= 0.0:
            continue
        only = replace(w, **{t: (getattr(w, t) if t == term else 0.0) for t in terms})
        out[term[2:]] = phi_of_bank(bank, only)
    return out


def closing_curve(w: RewardWeights, ranges: Tensor, *, capacity_mbps: float) -> Tensor:
    """`Phi` along the one axis the diagnosis turns on: the observer closing in.

    Holds everything else at the operating point -- a clear ray
    (`FREE_CLEARANCE_M`, which is what `occlusion` returns for "nothing in the
    way") and a chain already carrying `capacity_mbps` -- and sweeps the range.
    📏 This is the slice `docs/REWARD.md` reports the shipped potential moving a
    **total of 0.32** across, and it is the number a candidate has to beat: it is
    exactly the decision "close to B0's 89 m or stand off at 184 m", with every
    other quantity held where the policy already has it.
    """
    from src.env.occlusion import FREE_CLEARANCE_M

    n = ranges.shape[0]
    zeros = torch.zeros(n, 1)
    # A concrete swarm to go with the scalars, so `Phi_cover` is defined: MCV at
    # the origin, HVT 1000 m out (📏 the median HVT-MCV range in the last third
    # of a stage-4 episode), the observer closing along that axis at `ranges` and
    # the other four drones spread evenly behind it. Only the observer moves, so
    # the cover term is near-constant and the curve isolates the closing
    # decision -- which is exactly what it is for.
    mcv = torch.zeros(n, 3)
    hvt = torch.tensor([1000.0, 0.0, 0.0]).expand(n, 3)
    behind = torch.linspace(0.2, 0.8, 4)
    others = torch.stack(
        [
            torch.stack([1000.0 * f + 0 * ranges, torch.zeros(n), torch.full((n,), 80.0)], dim=-1)
            for f in behind
        ],
        dim=1,
    )
    observer = torch.stack([1000.0 - ranges, torch.zeros(n), torch.full((n,), 80.0)], dim=-1)
    drones = torch.cat([observer.unsqueeze(1), others], dim=1)
    snap = Snapshot(
        observed=torch.ones(n, dtype=torch.bool),
        e2e_capacity_mbps=torch.full((n,), capacity_mbps),
        nearest_dist_m=ranges,
        best_clearance_m=torch.full((n,), FREE_CLEARANCE_M),
        observer_dist_m=ranges,
        drone_pos=drones,
        mcv_pos=mcv,
        hvt_pos=hvt,
        battery=zeros,
        speed_ms=zeros,
        accel_ms2=zeros,
    )
    return potential(snap, w)


def _stranded_phi(w: RewardWeights, off: float) -> Tensor:
    """`Phi` with four drones working the axis and the fifth `off` m to the side."""
    from src.env.occlusion import FREE_CLEARANCE_M

    xy = [(200.0, 0.0), (450.0, 0.0), (700.0, 0.0), (940.0, 0.0), (600.0, off)]
    return potential(
        Snapshot(
            observed=torch.ones(1, dtype=torch.bool),
            e2e_capacity_mbps=torch.full((1,), 25.0),
            nearest_dist_m=torch.full((1,), 60.0),
            best_clearance_m=torch.full((1,), FREE_CLEARANCE_M),
            observer_dist_m=torch.full((1,), 60.0),
            drone_pos=torch.tensor([[[x, y, 80.0] for x, y in xy]]),
            mcv_pos=torch.zeros(1, 3),
            hvt_pos=torch.tensor([[1000.0, 0.0, 0.0]]),
            battery=torch.zeros(1, 1),
            speed_ms=torch.zeros(1, 1),
            accel_ms2=torch.zeros(1, 1),
        ),
        w,
    )


def report_recall(w: RewardWeights, *, offsets: tuple[float, ...] = (200.0, 500.0, 800.0)) -> None:
    """What `Phi` pays a drone that is NOT holding a role to come back.

    📏 The measured failure this answers: learned policies sit against the map
    boundary on 15-23 % of steps. Under the shipped potential every component is
    a `min` / `max` / routing reduction, so a drone out there moves `Phi` by
    **exactly 0.0** -- there is no gradient to come back along, at any distance.
    """

    print(
        "    recall gradient for a drone that holds no role (one 8 m step toward the MCV-HVT axis):"
    )
    for off in offsets:
        # Four drones doing the mission on the axis; the fifth stranded at
        # `off` metres perpendicular to it. Only the stranded drone moves.
        gain = float(_stranded_phi(w, off - 8.0) - _stranded_phi(w, off))
        home = float(_stranded_phi(w, 0.0) - _stranded_phi(w, off))
        bar = energy_differential()
        print(
            f"      {off:5.0f} m off-axis:  {gain:+.4f}/step   ({gain / bar:5.2f}x bar)"
            f"    whole trip home {home:+.3f}"
        )


def report_closing(w: RewardWeights, *, capacity_mbps: float = 25.0) -> None:
    """The design target, printed: swing and per-step gradient over the band."""
    bar = energy_differential()
    # 8 m is one step of travel at the 20 m/s cruise and the 0.4 s tick.
    ranges = torch.arange(60.0, 258.0, 8.0)
    phi = closing_curve(w, ranges, capacity_mbps=capacity_mbps)
    grad = (phi[:-1] - phi[1:]).abs()  # per 8 m step of closing
    print(
        f"    closing 250 -> 60 m (clear ray, chain at {capacity_mbps:.0f} Mbps): "
        f"swing {float(phi[0] - phi[-1]):+.3f}"
    )
    print(
        f"      per 8 m step  min {float(grad.min()):.4f}  median {float(grad.median()):.4f}  "
        f"max {float(grad.max()):.4f}   bar {bar:.4f}  "
        f"(median / bar = {float(grad.median()) / bar:.2f}x)"
    )
    marks = [
        (r, float(p))
        for r, p in zip(ranges.tolist(), phi.tolist(), strict=True)
        if int(r) in (60, 92, 132, 188, 252)
    ]
    print("      Phi at  " + "   ".join(f"{int(r)} m: {v:6.3f}" for r, v in marks))


def report_band(name: str, bank: dict[str, Tensor], w: RewardWeights) -> None:
    """`Phi` restricted to the states where the swarm is already SUCCEEDING.

    The whole-rollout swing is dominated by two step functions -- acquiring a
    sightline at all, and a chain existing at all -- and both are already solved
    by every learned policy most of the time. What decides the 16.1 pp gap is the
    gradient *inside* the succeeding set, which is where `docs/REWARD.md`'s
    flat-success problem lives.
    """
    from src.env.reward import CAPACITY_THRESHOLD_MBPS

    ok = bank["observed"] & (bank["e2e_capacity_mbps"] >= CAPACITY_THRESHOLD_MBPS)
    if not bool(ok.any()):
        print("    (no mission-capable steps in the bank)")
        return
    phi = phi_of_bank(bank, w)[ok]
    p1, p50, p99 = q(phi, 0.01, 0.5, 0.99)
    print(
        f"    Phi | capable   p1 {p1:7.3f}  p50 {p50:7.3f}  p99 {p99:7.3f}  "
        f"swing {p99 - p1:7.3f}   ({float(ok.float().mean()) * 100:.1f} % of steps)"
    )


def energy_term(bank: dict[str, Tensor], craft=DEFAULT_AIRFRAME) -> float:
    """The objective's mean energy payment per drone per step, as flown.

    ⚠️ Read against `energy_differential()`, not instead of it. That constant is
    the *maximum* the power curve can pay -- hover against the minimum-power
    airspeed -- and this is what the policy in the bank actually collected. The
    gap between them is how much of the opposing force is real for THIS policy.
    """
    from src.env.reward import hover_reference_power_w

    speed = bank["speed"]
    power = total_power_w(speed, torch.zeros_like(speed), craft)
    return float(-DEFAULT_WEIGHTS.energy * (power / hover_reference_power_w(craft)).mean())


def band(bank: dict[str, Tensor]) -> None:
    """Where in the operating band this policy actually sits."""
    obs = bank["observed"]
    r = bank["observer_dist_m"]
    print(
        "    observer range  p10/p50/p90  "
        + " / ".join(f"{x:6.1f}" for x in q(r[obs], 0.1, 0.5, 0.9))
    )
    print(
        "    nearest range   p10/p50/p90  "
        + " / ".join(f"{x:6.1f}" for x in q(bank["nearest_dist_m"], 0.1, 0.5, 0.9))
    )
    print(
        "    clearance(obs)  p10/p50/p90  "
        + " / ".join(f"{x:9.1f}" for x in q(bank["best_clearance_m"], 0.1, 0.5, 0.9))
    )
    cap = bank["e2e_capacity_mbps"]
    print(
        "    e2e capacity    p10/p50/p90  " + " / ".join(f"{x:6.1f}" for x in q(cap, 0.1, 0.5, 0.9))
    )
    print(
        f"    observed {float(obs.float().mean()) * 100:5.1f} %   "
        f"capable {float(bank['capable'].float().mean()) * 100:5.1f} %   "
        f"speed p50 {q(bank['speed'], 0.5)[0]:5.2f} m/s   "
        f"energy term {energy_term(bank):+.4f}/step/drone"
    )


def discounted_future_capable(capable: Tensor, gamma: float = 0.997) -> Tensor:
    """(steps, B) discounted future `mission_capable` from each step onward."""
    out = torch.zeros_like(capable, dtype=torch.float64)
    acc = torch.zeros(capable.shape[1], dtype=torch.float64)
    for t in range(capable.shape[0] - 1, -1, -1):
        acc = capable[t].to(torch.float64) + gamma * acc
        out[t] = acc
    return out


def report_value_correlation(banks: dict[str, dict[str, Tensor]]) -> None:
    """Does `Phi` predict what a policy is about to achieve?

    ⚠️ The sharpest offline test of a potential there is, and the reason it is
    worth stating: **the ideal `Phi` is `V*` itself** (Ng, Harada & Russell show
    the shaped problem's value function is `V - Phi`, so `Phi = V*` makes every
    advantage immediate). So a candidate can be scored against the discounted
    future `mission_capable` return of the states in the bank, with no training
    run at all -- which is the only offline check in this block that could have
    told the four failed interventions apart from a good one.

    Two readouts, and they answer different questions:

    * **correlation** -- does `Phi` rank *states* by what follows them;
    * **separation** -- does `Phi` rank B0's states above a learned policy's. A
      potential that cannot do this is guiding toward the wrong place.
    """
    if len(banks) < 2:
        return
    print("\n=== does Phi predict what happens next? ===")
    print("    (corr with discounted future mission-capable; the ideal Phi is V*)")
    for cname, cw in CANDIDATES.items():
        xs = torch.cat([phi_of_bank(b, cw).flatten().double() for b in banks.values()])
        ys = torch.cat([discounted_future_capable(b["capable"]).flatten() for b in banks.values()])
        corr = float(torch.corrcoef(torch.stack([xs, ys]))[0, 1])
        line = f"    {cname:<9} pooled corr {corr:+.3f}"
        if "b0" in banks:
            others = [k for k in banks if k != "b0"]
            gap = float(
                phi_of_bank(banks["b0"], cw).mean()
                - torch.stack([phi_of_bank(banks[k], cw).mean() for k in others]).mean()
            )
            line += f"    mean Phi(B0) - mean Phi(learned) = {gap:+.3f}"
        print(line)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", nargs="+", default=["b0"], help="b0 | random | a checkpoint path")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--num-drones", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stage", type=int, default=4)
    ap.add_argument("--fidelity", default="F4")
    ap.add_argument("--train-routes", action="store_true")
    ap.add_argument(
        "--bank-dir",
        default=None,
        help="cache banked rollouts here (a bank is ~1 min to collect and is reused "
        "by every candidate Phi, so caching is what makes iterating cheap)",
    )
    a = ap.parse_args()

    print(f"energy step differential (recomputed): {energy_differential():.4f}")
    for cname, cw in CANDIDATES.items():
        print(f"\n=== the closing decision, Phi `{cname}` ===")
        report_closing(cw)
        report_recall(cw)
    collected: dict[str, dict[str, Tensor]] = {}
    for name in a.policy:
        ckpt = None if name in ("b0", "random") else Path(name)
        cache = None
        if a.bank_dir:
            tag = name.replace("/", "_")
            split = "train" if a.train_routes else "eval"
            cache = Path(a.bank_dir) / (
                f"{tag}-{a.fidelity}-s{a.stage}-{split}-n{a.num_drones}"
                f"-b{a.num_envs}-seed{a.seed}-{a.device}.pt"
            )
        if cache is not None and cache.exists():
            bank = torch.load(cache, map_location="cpu", weights_only=True)
        else:
            bank = bank_states(
                name if ckpt is None else "checkpoint",
                ckpt,
                device=a.device,
                num_envs=a.num_envs,
                num_drones=a.num_drones,
                seed=a.seed,
                stage=a.stage,
                fidelity=a.fidelity,
                train_routes=a.train_routes,
            )
            bank = {k: v.cpu() for k, v in bank.items()}
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                torch.save(bank, cache)
        print(f"\n=== {name} ===")
        band(bank)
        collected[name] = bank
        for cname, cw in CANDIDATES.items():
            report(f"Phi `{cname}`", bank, cw)
            report_band(f"Phi `{cname}`", bank, cw)
    report_value_correlation({("b0" if k == "b0" else k): v for k, v in collected.items()})


if __name__ == "__main__":
    main()
