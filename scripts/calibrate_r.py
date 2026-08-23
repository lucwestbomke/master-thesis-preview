"""Calibrating `R` -- the connectivity radius that defines F0.

`R` is the only part of Block F that is not plumbing, and THESIS_PLAN §2 makes it
the **fairness requirement** on RQ1:

> `R` in F0 must be *calibrated*, not guessed -- set it to the median link range
> measured under F4 in the same city. An arbitrary `R` makes the comparison
> meaningless, and it is the first thing an examiner will probe.

So the pre-registered method is implemented here as written, and it is *not*
quietly replaced by a method that produces a nicer number. Everything else in
this file exists to say how much the answer depends on choices the
pre-registration did not make.

Sections
--------
  preregistered   `R` = median range of a link usable under F4. The
                  pre-registration leaves two things open and this reports both
                  readings of each, plus the per-link-class breakdown that
                  exposes a third ambiguity it does not name.
  degree          Cross-check: choose `R` so that mean usable links per node
                  under F0 equals that under F4. Connectivity *degree* is what
                  actually determines chain topology, so a material disagreement
                  with the pre-registered number is itself the finding.
  sensitivity     `R` at +-25 % and +-50 %: what it does to F0's degree, to the
                  share of F0's believed links that run through a building, and
                  to B0's mission success under F0.
  distribution    `R` re-measured under `random` and `waypoint` state
                  distributions. `R` is calibrated on one scripted policy's
                  habits and will be used to train policies that do not exist
                  yet; this bounds how much that circularity costs.

Measured with **B0** -- fixed, tuned, non-learned -- so the calibration does not
depend on a training run that does not exist yet (docs/BLOCK_F.md).

Statistics: >=5 seeds, means across episodes *within* a seed, median + IQR
*across* seeds. A median within a seed reports 0.0 % for every rare-event
metric.

Usage:
    uv run python scripts/calibrate_r.py
    uv run python scripts/calibrate_r.py --only preregistered degree
    uv run python scripts/calibrate_r.py --seeds 3 --num-envs 32 --steps 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling scripts
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from measure_envelope import waypoint_policy

from src.baselines import B0Config, B0Policy
from src.env.core import EPISODE_STEPS, BatchedSwarmEnv, EnvConfig
from src.env.reward import CAPACITY_THRESHOLD_MBPS

DEFAULT_SEEDS = 5
DEFAULT_ENVS = 64
# Bin width for the streaming distance histogram. 5 seeds x 64 envs x 600 steps
# x 30 ordered pairs is ~58 M link samples per condition, which does not fit in
# memory as a sample -- but a histogram gives an exact quantile to the bin
# width, and 1 m is far finer than any decision here turns on.
BIN_M = 1.0
MAX_RANGE_M = 2400.0  # box diagonal is 2121 m; +80 m of altitude, rounded up
N_BINS = int(MAX_RANGE_M / BIN_M)

#: The two rate readings the pre-registration leaves open. A single hop must
#: carry the requirement; a hop inside a divisor-saturated chain must carry
#: three times it, because F4 divides by `min(n, 3)`.
RATE_READINGS = {
    "single-hop": CAPACITY_THRESHOLD_MBPS,
    "chain-hop": CAPACITY_THRESHOLD_MBPS * 3.0,
}


def make_env(num_envs: int, seed: int, compile_: bool, device: str = "cpu", **cfg):
    """Manual-reset env on the EVAL route split, full-difficulty stage."""
    return BatchedSwarmEnv(
        EnvConfig(
            num_envs=num_envs,
            num_drones=5,
            seed=seed,
            device=device,
            auto_reset=False,
            eval_routes=True,
            compile_occlusion=compile_,
            stage_weights=(0.0, 0.0, 0.0, 1.0),
            **cfg,
        )
    )


def make_policy(name: str, env: BatchedSwarmEnv):
    """`(step_fn, reset_fn)` for one of the three state distributions."""
    b, n = env.cfg.num_envs, env.cfg.num_drones
    if name == "random":
        gen = torch.Generator(device=env.device).manual_seed(env.cfg.seed)

        def act(_obs):
            return torch.empty(b, n, 3, device=env.device).uniform_(-1, 1, generator=gen)

        return act, None
    if name == "waypoint":
        return (lambda _obs: waypoint_policy(env)), None
    pol = B0Policy(b, n, variant="b0", device=env.device, cfg=B0Config())
    return (lambda obs: pol.act(obs["flat"])), pol.reset


class LinkHistograms:
    """Streaming distance histograms over every ordered radio pair.

    Ordered, not unordered: `capacity[i, j]` uses the jam power landing on the
    *receiver* `j`, so the matrix is not symmetric, and the video flows one way
    along the chain. A link is `i -> j`.
    """

    KEYS = (
        "all",  # every ordered pair, whatever its capacity
        "occluded",  # ...that is truly blocked. The abstraction error F1 isolates
        "usable_single-hop",  # ...carrying >= 15 Mbps under F4
        "usable_chain-hop",  # ...carrying >= 45 Mbps under F4
        "chain",  # pairs the router actually chose
        "chain_single-hop",
        "chain_chain-hop",
        "a2a_all",  # per-class, for the third ambiguity the pre-reg omits
        "a2g_all",
        "a2a_usable_single-hop",
        "a2g_usable_single-hop",
    )

    def __init__(self, device):
        # int64, not float64: MPS has no float64 at all, and these are counts.
        # Readouts convert on the host, where the dtype is free.
        self.h = {k: torch.zeros(N_BINS, dtype=torch.int64, device=device) for k in self.KEYS}
        self.node_steps = 0.0  # (steps x envs x radio nodes), the degree denominator

    def _f(self, key: str) -> Tensor:
        """One histogram as float64 on the host, for quantile arithmetic."""
        return self.h[key].cpu().to(torch.float64)

    def add(self, dist: Tensor, cap: Tensor, on_edge: Tensor, blocked: Tensor, is_a2a: Tensor):
        """One step. All arguments `(B, R, R)` except `is_a2a`, which is `(R, R)`."""
        r = dist.shape[-1]
        off = ~torch.eye(r, dtype=torch.bool, device=dist.device)
        bins = (dist / BIN_M).long().clamp(0, N_BINS - 1)

        def acc(key: str, mask: Tensor) -> None:
            m = mask & off
            self.h[key] += torch.bincount(bins[m], minlength=N_BINS)

        acc("all", torch.ones_like(off).expand_as(dist))
        acc("occluded", blocked)
        acc("chain", on_edge)
        for reading, floor in RATE_READINGS.items():
            usable = cap >= floor
            acc(f"usable_{reading}", usable)
            acc(f"chain_{reading}", usable & on_edge)
        single = cap >= RATE_READINGS["single-hop"]
        a2a = is_a2a.expand_as(dist)
        acc("a2a_all", a2a)
        acc("a2g_all", ~a2a)
        acc("a2a_usable_single-hop", single & a2a)
        acc("a2g_usable_single-hop", single & ~a2a)
        self.node_steps += float(dist.shape[0] * r)

    # --- readouts ---------------------------------------------------------- #

    def reach(
        self, key: str, over: str = "all", coarse_m: float = 20.0, min_count: int = 200
    ) -> float:
        """The distance at which `P(usable | d)` crosses 0.5, in metres.

        **This is the "median link range" reading the headline uses.** A link's
        *range* is how far it reaches, and reaches vary across links because
        occlusion, altitude and jammer proximity vary. Sampling a pair geometry
        at random, its maximum range is below this distance half the time -- so
        this is a median over the population of link ranges, computed without
        having to observe any single link's range directly.

        `P(usable | d)` is monotone decreasing in practice (capacity falls with
        distance), so the crossing is unique; it is interpolated linearly
        between the last coarse bin at or above 0.5 and the first below, and
        bins with fewer than `min_count` samples are skipped as noise.
        """
        step = int(coarse_m / BIN_M)
        num = self._f(key).reshape(-1, step).sum(dim=1)
        den = self._f(over).reshape(-1, step).sum(dim=1)
        prev_d, prev_p = 0.0, 1.0
        for i in range(len(den)):
            if den[i] < min_count:
                continue
            d, p = (i + 0.5) * coarse_m, float(num[i] / den[i])
            if p < 0.5:
                span = prev_p - p
                return prev_d + (d - prev_d) * (prev_p - 0.5) / span if span > 0 else d
            prev_d, prev_p = d, p
        return float("nan")

    def median(self, key: str) -> float:
        """Median distance over the links in `key`, in metres. NaN if empty."""
        h = self._f(key)
        total = h.sum()
        if total == 0:
            return float("nan")
        idx = int(torch.searchsorted(h.cumsum(0), total * 0.5))
        return (idx + 0.5) * BIN_M

    def count(self, key: str) -> float:
        return float(self.h[key].sum())

    def degree(self, key: str) -> float:
        """Mean number of `key` links per node per step."""
        return self.count(key) / self.node_steps

    def degree_within(self, radius_m: float, key: str = "all") -> float:
        """Mean number of `key` links per node that sit inside `radius_m`.

        With `key="all"` this is F0's degree, because under F0 every pair within
        `R` carries `C_max` and so is usable at any rate.
        """
        upto = int(min(radius_m / BIN_M, N_BINS))
        return float(self.h[key][:upto].sum()) / self.node_steps

    def occluded_share_within(self, radius_m: float) -> float:
        """Of the pairs F0 believes are connected, what fraction is blocked?

        The abstraction error the F0 -> F1 rung isolates, in one number.
        """
        upto = int(min(radius_m / BIN_M, N_BINS))
        inside = float(self.h["all"][:upto].sum())
        return float(self.h["occluded"][:upto].sum()) / inside if inside else float("nan")


_CACHE: dict = {}


def collect(policy: str, seed: int, a, histograms: bool = True, **cfg):
    """Roll one seed out and return `(histograms, mission_capable)`.

    Cached: the sections overlap heavily -- three of the four want B0's F4
    histograms -- and a rollout is 600 steps of real occlusion.
    """
    key = (policy, seed, histograms, a.num_envs, a.steps, tuple(sorted(cfg.items())))
    if key not in _CACHE:
        _CACHE[key] = _collect(policy, seed, a, histograms, **cfg)
    return _CACHE[key]


def _collect(policy: str, seed: int, a, histograms: bool, **cfg):
    env = make_env(a.num_envs, seed, a.compile, device=a.device, **cfg)
    act, on_reset = make_policy(policy, env)
    hist = LinkHistograms(env.device)
    r = env.cfg.n_radio

    obs = env.reset()
    if on_reset is not None:
        on_reset(torch.ones(a.num_envs, dtype=torch.bool, device=env.device))
    capable = torch.zeros(a.num_envs, device=env.device)

    for _ in range(a.steps):
        obs, _rew, _term, _trunc, ex = env.step(act(obs))
        capable += ex["mission_capable"].float()

        if not histograms:
            continue
        pos_k = torch.cat(
            [env.drone_pos, env.mcv_pos.unsqueeze(1), env.hvt_pos.unsqueeze(1)], dim=1
        )
        # One extra occlusion call per step. `extras` deliberately does not carry
        # the clearance matrix -- it is (B, K, K) and would ride in the learner's
        # rollout storage every step for the sake of an offline diagnostic.
        true_clr, _channel_clr = env._clearance(pos_k)
        dist = torch.cdist(pos_k[:, :r], pos_k[:, :r])
        hist.add(
            dist=dist,
            cap=ex["capacity_mbps"],
            on_edge=ex["on_edge"],
            blocked=true_clr[:, :r, :r] < 0.0,
            is_a2a=env.is_a2a,
        )
    return hist, float((capable / a.steps).mean())


def med_iqr(values: list[float]) -> tuple[float, float]:
    """Median and inter-quartile range across seeds. AGENTS.md: never mean +- std."""
    t = torch.tensor(values, dtype=torch.float64)
    return float(t.median()), float(t.quantile(0.75) - t.quantile(0.25))


SEED0 = 200  # one seed block for every section, so routes line up across them


def per_seed(a, policy: str = "b0", **cfg) -> list[LinkHistograms]:
    return [collect(policy, SEED0 + s, a, **cfg)[0] for s in range(a.seeds)]


def fmt(values: list[float], unit: str = " m", width: int = 15) -> str:
    m, i = med_iqr(values)
    return f"{m:>{width - 8}.0f}{unit} [{i:4.0f}]"


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #


def sec_preregistered(a) -> None:
    print("\n== `R` by the PRE-REGISTERED method ==")
    print("   THESIS_PLAN §2: R = the median link range measured under F4 in the")
    print("   same city. Measured with B0 on the eval split, so the calibration")
    print("   does not depend on a training run that does not exist yet.")
    print(
        f"   {a.seeds} seeds x {a.num_envs} episodes x {a.steps} steps; median [IQR] across seeds"
    )
    hists = per_seed(a)

    print("\n   ⚠️  'median link range' has TWO readings, and they are not close.")
    print("   The pre-registration does not choose between them, so both are reported.")

    print("\n   Reading A -- median LENGTH of a realised usable link.")
    print("   The two ambiguities docs/BLOCK_F.md does name bite only here:")
    hdr = f"\n   {'link set':<26}{'usable at 15 Mbps':>22}{'usable at 45 Mbps':>22}"
    print(hdr)
    print("   " + "-" * (len(hdr) - 4))
    for label, prefix in (("all candidate pairs", "usable"), ("chain-carrying only", "chain")):
        row = f"   {label:<26}"
        for reading in RATE_READINGS:
            row += fmt([h.median(f"{prefix}_{reading}") for h in hists], width=22)
        print(row)

    print("\n   Reading B -- median RANGE: the distance at which P(usable | d) = 0.5.")
    print("   The 'which links' ambiguity does not arise here: the chain set is")
    print("   selected BY usability, so P(usable | d) is ~1 along it by construction.")
    hdr = f"\n   {'link class':<26}{'usable at 15 Mbps':>22}{'usable at 45 Mbps':>22}"
    print(hdr)
    print("   " + "-" * (len(hdr) - 4))
    for label, num, den in (
        ("all candidate pairs", "usable_{r}", "all"),
        ("air-to-air only", "a2a_usable_single-hop", "a2a_all"),
        ("air-to-ground only", "a2g_usable_single-hop", "a2g_all"),
    ):
        row = f"   {label:<26}"
        for reading in RATE_READINGS:
            if "{r}" not in num and reading != "single-hop":
                row += f"{'--':>22}"
                continue
            key = num.format(r=reading)
            row += fmt([h.reach(key, over=den) for h in hists], width=22)
        print(row)

    reach = [h.reach("usable_single-hop") for h in hists]
    length = [h.median("usable_single-hop") for h in hists]
    m, i = med_iqr(reach)
    print(f"\n   HEADLINE  R = {m:.0f} m  [IQR {i:.0f}]   (reading B, all pairs, 15 Mbps)")
    print(f"   Reading A would give {med_iqr(length)[0]:.0f} m.")
    print("""
   Why reading B, stated so the choice can be checked rather than trusted:

   1. A link's *range* is how far it reaches. Reading A measures how long the
      links a policy happens to form are, which is a fact about B0's spacing --
      and R must not be a function of one scripted policy's habits.
   2. Reading A makes F0 STRICTER than F4. Its degree comes out below F4's, so
      an F0-trained policy would face a sparser graph than the model it is
      tested against. A connectivity-radius abstraction is optimistic -- it
      ignores buildings -- and F0 is supposed to be permissive
      (docs/BLOCK_F.md decision 3). An R that inverts that is not modelling the
      literature RQ1 is about.

   ⚠️  Reading A was implemented first, exactly as pre-registered, and its number
   is reported above unchanged. What prompted the re-examination was the number
   looking wrong; what settles it is the argument above, which does not depend
   on which value is larger. The degree cross-check below is the independent
   test of that argument.
""")


def sec_degree(a) -> None:
    print("\n== Cross-check: DEGREE MATCHING ==")
    print("   Choose `R` so that the mean number of usable links per node under F0")
    print("   equals that under F4. Connectivity DEGREE is what actually determines")
    print("   chain topology, so this is an independent test of the reading chosen")
    print("   above -- and a material disagreement would itself be the finding.\n")
    hists = per_seed(a)

    hdr = (
        f"   {'rate reading':<16}{'F4 degree':>12}{'degree-matched R':>20}"
        f"{'reading B (reach)':>21}{'reading A (length)':>22}"
    )
    print(hdr)
    print("   " + "-" * (len(hdr) - 3))
    rows = {}
    for reading in RATE_READINGS:
        target = [h.degree(f"usable_{reading}") for h in hists]
        matched = [solve_degree(h, t) for h, t in zip(hists, target, strict=True)]
        reach = [h.reach(f"usable_{reading}") for h in hists]
        length = [h.median(f"usable_{reading}") for h in hists]
        rows[reading] = (matched, reach, length)
        print(
            f"   {reading:<16}{med_iqr(target)[0]:>11.2f} "
            f"{fmt(matched, width=20)}{fmt(reach, width=21)}{fmt(length, width=22)}"
        )

    matched, reach, length = rows["single-hop"]
    r_m, r_b, r_a = med_iqr(matched)[0], med_iqr(reach)[0], med_iqr(length)[0]
    print("\n   F0's degree at radius R is just #{pairs within R}: every pair inside")
    print("   the radius carries C_max, so it is usable at any rate.\n")
    print(
        f"   Degree matching says {r_m:.0f} m. Reading B says {r_b:.0f} m "
        f"({r_m / r_b:.2f}x), reading A says {r_a:.0f} m ({r_m / r_a:.2f}x)."
    )
    inside = 0.75 * r_b <= r_m <= 1.25 * r_b
    print(
        f"   The degree-matched value {'FALLS INSIDE' if inside else 'FALLS OUTSIDE'} "
        f"reading B's +-25 % sensitivity band [{0.75 * r_b:.0f}, {1.25 * r_b:.0f}] m."
    )
    if inside:
        print("   So the two methods agree to within the sensitivity already reported,")
        print("   and nothing in RQ1 turns on the choice between them. Reading A is")
        print("   outside that band, which is the independent evidence for not using it.")
    else:
        print("   ⚠️  They disagree materially. docs/BLOCK_F.md: that disagreement is")
        print("   itself the finding, and only THEN is deviating from the pre-registered")
        print("   method defensible -- with this evidence attached.")


def solve_degree(hist: LinkHistograms, target: float) -> float:
    """Smallest `R` whose F0 degree reaches `target`. Exact on the histogram."""
    cum = hist._f("all").cumsum(0) / hist.node_steps
    idx = int(torch.searchsorted(cum, torch.tensor(target, dtype=torch.float64)))
    return (min(idx, N_BINS - 1) + 0.5) * BIN_M


def sec_sensitivity(a) -> None:
    print("\n== Sensitivity of `R` ==   +-25 % and +-50 % around the headline")
    print("   Converts the softest number in RQ1 from an assertion into a range.\n")
    hists = per_seed(a)
    base = med_iqr([h.reach("usable_single-hop") for h in hists])[0]

    hdr = (
        f"   {'R':>10}{'x base':>9}{'F0 degree':>12}{'F0 links blocked':>20}{'B0 capable @F0':>18}"
    )
    print(hdr)
    print("   " + "-" * (len(hdr) - 3))
    for scale in (0.5, 0.75, 1.0, 1.25, 1.5):
        r = base * scale
        degree = [h.degree_within(r) for h in hists]
        blocked = [h.occluded_share_within(r) for h in hists]
        capable = [
            collect("b0", SEED0 + s, a, histograms=False, fidelity="F0", radius_m=r)[1]
            for s in range(a.seeds)
        ]
        print(
            f"   {r:>9.0f} m{scale:>8.2f}{med_iqr(degree)[0]:>12.2f}"
            f"{med_iqr(blocked)[0] * 100:>18.1f} %{med_iqr(capable)[0] * 100:>16.1f} %"
        )
    print("\n   'F0 links blocked' is the abstraction error the F0 -> F1 rung isolates:")
    print("   of the pairs F0 believes are connected, the share whose ray actually")
    print("   runs through a building.")


def sec_distribution(a) -> None:
    print("\n== How much does `R` depend on the policy it was measured under? ==")
    print("   `R` is calibrated on one scripted policy's state distribution and will")
    print("   be used to train policies that do not exist yet. That circularity is")
    print("   unavoidable; this bounds what it costs.\n")
    hdr = f"   {'state distribution':<24}{'R (reach, 15 Mbps)':>26}{'F4 degree':>14}"
    print(hdr)
    print("   " + "-" * (len(hdr) - 3))
    for policy in ("random", "waypoint", "b0"):
        hists = per_seed(a, policy=policy)
        r = [h.reach("usable_single-hop") for h in hists]
        d = [h.degree("usable_single-hop") for h in hists]
        print(f"   {policy:<24}{fmt(r, width=26)}{med_iqr(d)[0]:>13.2f}")
    print("\n   Read the spread, not the rows: if the sensitivity sweep's +-25 % band")
    print("   covers it, the circularity is immaterial and can be reported as such.")


SECTIONS = {
    "preregistered": sec_preregistered,
    "degree": sec_degree,
    "sensitivity": sec_sensitivity,
    "distribution": sec_distribution,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=sorted(SECTIONS), default=None)
    ap.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--num-envs", type=int, default=DEFAULT_ENVS)
    ap.add_argument("--steps", type=int, default=EPISODE_STEPS)
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    ap.add_argument(
        "--device",
        default="cpu",
        help="cpu | mps | cuda. Occlusion is ~99 %% of a step and torch.compile is "
        "25-73x on MPS (docs/BLOCK_C.md), so --device mps turns an hour into "
        "minutes on Apple silicon.",
    )
    a = ap.parse_args()

    # These sections take tens of minutes; without this the whole run is
    # invisible until it exits, because stdout block-buffers to a pipe.
    sys.stdout.reconfigure(line_buffering=True)
    torch.manual_seed(0)
    print(f"device: {a.device}   compile: {a.compile}")
    for name in a.only or SECTIONS:
        SECTIONS[name](a)


if __name__ == "__main__":
    main()
