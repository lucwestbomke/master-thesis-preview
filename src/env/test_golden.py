"""`fidelity="F4"` must reproduce the pre-Block-F environment, exactly.

This is the one test in Block F that, if it fails, invalidates every number in
`docs/BLOCK_D.md` and `docs/BLOCK_E.md` -- because `F4` is not a new condition,
it *is* the environment those blocks measured (`docs/BLOCK_F.md`, decision 5).

The artefact was captured before the ladder existed. See
`scripts/capture_f4_golden.py` for why re-capturing it is not a way to make this
test pass.

Two rollout passes, no more: the module-scoped fixtures are what keep this file
at ~12 s rather than ~40 s. Running the default config and the explicitly-named
rung as two separate executions, both against the same frozen trace, is also
what makes the determinism check below cheap -- if the env read a global RNG,
the two passes could not both match.
"""

from __future__ import annotations

import platform

import pytest
import torch

from . import golden
from .golden import GOLDEN_SCENARIOS, load_golden, run_all, run_scenario

#: The CPU architecture the artefact was captured on (Apple silicon, 5ce0a2f).
#: `golden.FORCED_CFG` pins `device="cpu"`, so the *device* is already
#: controlled -- but the **instruction set is not**, and float32 is not
#: associative across it.
#:
#: Measured on 2026-08-24, first run on x86-64: identical code, identical seeds,
#: `device="cpu"` on both, and the traces diverge. `reset` (180 steps) by
#: 1.4e-6, `offn_eval` by 3.0e-6, `design` (300 steps) by 2.4e-3 -- ULP-level at
#: the start, amplified by a closed loop where the action depends on the state it
#: just produced. The suite passes exactly on arm64 with the same commit, which
#: is what rules out an environment change.
#:
#: So exactness is asserted where it is meaningful, and a weaker but honest
#: check runs elsewhere. ⛔ Do NOT "fix" this by re-capturing on x86: that
#: discards the only record of the pre-Block-F env, and the divergence is real
#: rather than a defect.
GOLDEN_ARCH = "arm64"
ON_GOLDEN_ARCH = platform.machine() == GOLDEN_ARCH
OFF_ARCH_REASON = (
    f"the frozen trace was captured on {GOLDEN_ARCH}; float32 is not associative "
    f"across instruction sets, so exact equality is only meaningful there "
    f"(this machine: {platform.machine()})"
)


@pytest.fixture(scope="module")
def frozen() -> dict[str, dict[str, torch.Tensor]]:
    if not golden.GOLDEN_PATH.exists():
        pytest.fail(
            f"{golden.GOLDEN_PATH} is missing. It is committed on purpose -- it is "
            "the only record of what the env did before Block F."
        )
    return load_golden()


@pytest.fixture(scope="module")
def as_built() -> dict[str, dict[str, torch.Tensor]]:
    """The env as `EnvConfig` constructs it by default."""
    return run_all()


@pytest.fixture(scope="module")
def as_f4() -> dict[str, dict[str, torch.Tensor]]:
    """The env with the top rung asked for by name."""
    return run_all(fidelity="F4")


def assert_trace_equal(got: dict[str, torch.Tensor], want: dict[str, torch.Tensor], what: str):
    """Element for element, with a message that localises the first divergence.

    Exact equality, not a tolerance. Every arithmetic path `F4` takes is meant to
    be the one that was already there, so a last-bit difference means an
    operation was reordered -- which is a real answer to "did the environment
    change?" and should be looked at rather than absorbed by an `atol`.
    """
    missing = sorted(set(want) - set(got))
    assert not missing, f"{what}: keys vanished from the env output: {missing}"

    for key in sorted(want):
        a, b = got[key], want[key]
        assert a.shape == b.shape, f"{what}/{key}: shape {tuple(a.shape)} != {tuple(b.shape)}"
        if torch.equal(a, b):
            continue
        differing = a != b
        first = int(differing.flatten(1).any(dim=1).to(torch.uint8).argmax())
        delta = (a.float() - b.float()).abs().max()
        pytest.fail(
            f"{what}/{key}: {int(differing.sum())} of {a.numel()} elements differ, "
            f"first at recorded index {first}, max |delta| {float(delta):.6g}.\n"
            "The environment changed. Block D and Block E were measured under the "
            "frozen trace, so the question this raises is which of their numbers "
            "moved -- re-capturing the golden does not answer it."
        )


@pytest.mark.skipif(not ON_GOLDEN_ARCH, reason=OFF_ARCH_REASON)
@pytest.mark.parametrize("scenario", GOLDEN_SCENARIOS, ids=lambda s: s.name)
def test_default_config_reproduces_the_frozen_trace(scenario, as_built, frozen):
    """The env as constructed by default. This guards the whole `EnvConfig`
    surface: if Block F moved a default, it shows here."""
    assert_trace_equal(as_built[scenario.name], frozen[scenario.name], scenario.name)


@pytest.mark.skipif(not ON_GOLDEN_ARCH, reason=OFF_ARCH_REASON)
@pytest.mark.parametrize("scenario", GOLDEN_SCENARIOS, ids=lambda s: s.name)
def test_explicit_f4_reproduces_the_frozen_trace(scenario, as_f4, frozen):
    """`fidelity="F4"` asked for by name. The default and the top rung must be
    the same environment, or the ladder's reference condition is not what Blocks
    D and E measured."""
    assert_trace_equal(as_f4[scenario.name], frozen[scenario.name], scenario.name)


def test_the_rollout_is_deterministic():
    """Two executions of the same scenario agree.

    Without this the comparisons above prove nothing: a non-deterministic env
    would make them a coin toss, and the failure would read as a Block F
    regression rather than as the RNG-discipline problem it would actually be.
    Run on the cheapest scenario, since the two passes above already cover the
    expensive ones from opposite directions.
    """
    small = min(GOLDEN_SCENARIOS, key=lambda s: s.steps * s.cfg.get("num_envs", 1))
    assert_trace_equal(run_scenario(small), run_scenario(small), f"{small.name} (rerun)")


def test_the_env_output_contract_is_unchanged(as_built, frozen):
    """No *new* keys either.

    Separate from the equality tests on purpose. A new `extras` key does not move
    a trajectory and so does not invalidate Block D or E -- but every downstream
    consumer sees it, so it should be a decision rather than a side effect.
    """
    for name, trace in as_built.items():
        new = sorted(set(trace) - set(frozen[name]))
        assert not new, (
            f"{name}: the env now emits {new}, which the frozen trace does not have. "
            "Not a regression, but a deliberate widening of the output contract: "
            "bump golden.GOLDEN_FORMAT and re-capture only once the equality tests "
            "above pass on the unwidened output."
        )


def test_the_frozen_trace_actually_exercises_the_physics(frozen):
    """The golden is only as good as its coverage.

    A trace over a degenerate rollout would pass every test above while pinning
    nothing -- and the first attempt at this artefact was exactly that: it never
    produced a multi-hop chain with the jammer on, so it would have covered
    neither the `F3` nor the `F4` path. These are the properties that make the
    comparison load-bearing, asserted so they cannot quietly rot.
    """
    design = frozen["design"]
    hops = torch.bincount(design["extras/hop_count"].flatten().long(), minlength=6)
    assert (hops[:6] > 0).all(), f"design must cover every hop count 0-5, got {hops.tolist()}"
    assert design["env/jammer_on"].min() > 0.5, "design must run with the jammer on"
    assert design["extras/chain_occluded"].float().mean() > 0.1, "occluded chains must occur"
    # The relay chain binding, not just the sensor: `mission_capable` strictly
    # below `sees_any` is the 15 Mbps regime every Block E number lives in.
    assert design["extras/mission_capable"].float().mean() < (
        design["extras/sees_any"].float().mean() - 0.05
    ), "design must reach the regime where the chain, not the sensor, binds"

    alt = design["env/drone_pos"][..., 2]
    assert alt.min() <= 40.0 and alt.max() >= 80.0, "both altitude clamps must be hit"
    assert frozen["reset"]["truncated"].any(), "the reset scenario must cross a truncation"
    assert frozen["reset"]["env/jammer_on"].max() < 0.5, "reset must cover the jammer-off branch"


@pytest.mark.skipif(ON_GOLDEN_ARCH, reason=f"exact equality is asserted on {GOLDEN_ARCH}")
@pytest.mark.parametrize("scenario", GOLDEN_SCENARIOS, ids=lambda s: s.name)
def test_off_arch_the_behaviour_the_numbers_were_measured_from_is_unchanged(
    scenario, as_built, frozen
):
    """The weaker check, for machines the artefact was not captured on.

    Bitwise equality cannot hold off-architecture (see `GOLDEN_ARCH`), but the
    question the golden exists to answer still can be: **are Block D's and Block
    E's numbers still valid?** Those numbers are *aggregates* -- mission-capable
    fraction, observed fraction, mean hop count -- so that is what is asserted
    here, at a tolerance far tighter than any effect either block reports.

    A real environment change moves these. Last-bit divergence amplified through
    a 300-step closed loop does not: it reshuffles which individual steps are
    capable without moving the rate.

    ⚠️ This is a genuinely weaker test and it is not a substitute for the exact
    one. CI for this project should run on `arm64`.
    """
    got, want = as_built[scenario.name], frozen[scenario.name]
    for key, tol in (
        ("extras/mission_capable", 0.02),
        ("extras/sees_any", 0.02),
        ("extras/hop_count", 0.05),
    ):
        a = got[key].float().mean()
        b = want[key].float().mean()
        assert abs(float(a - b)) <= tol, (
            f"{scenario.name}/{key}: rate moved {float(a):.4f} -> {float(b):.4f}, "
            f"more than {tol}. Off-architecture float divergence reshuffles which "
            "steps are capable; it does not move the rate. This looks like a real "
            "environment change -- check it on arm64, where the exact test runs."
        )
