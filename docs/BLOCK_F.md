# Block F — the fidelity ladder F0–F4

**Goal:** one environment that can be constructed at five channel fidelities, so
that RQ1 — *which physical effects must a channel model include for learned
policies to transfer?* — becomes an experiment rather than an argument.

Consumes Block A (`channel`, `routing`), Block C (`occlusion`), Block D
(`core.py`) and Block E (`eval_baseline.py`, B0). Produces a `fidelity` seam on
`EnvConfig`, a calibrated `R`, and the measurements that justify it.

**This block is small in code and large in consequence.** The rungs are a handful
of flags; what makes it hard is that three of the decisions below, if taken
carelessly, destroy RQ1's attribution *silently* — the runs complete, the numbers
look plausible, and the primary result means something other than what it says.

Every number in this file must be regenerable by a committed script, in the
[`measure_envelope.py`](../scripts/measure_envelope.py) /
[`eval_baseline.py`](../scripts/eval_baseline.py) pattern.

---

## The ladder, from THESIS_PLAN §2

| Level | Link capacity is… | Rung isolates |
|---|---|---|
| **F0** | `C_max` if `distance < R` else 0 | the standard abstraction |
| **F1** | + requires an unoccluded ray | cost of ignoring **buildings** |
| **F2** | continuous: path loss → SINR → Shannon with modulation cap | cost of **binary** connectivity |
| **F3** | + jammer in the SINR denominator | cost of ignoring the **threat** |
| **F4** | + multi-hop rate division `min(Cᵢ)/min(n,3)` | cost of ignoring **relay cost** |

Rungs are **cumulative**. Train one policy per rung; **evaluate all of them under
F4**. The gaps attribute the answer instead of merely demonstrating one.

**Hypothesis: occlusion dominates** — a radius model lets the policy believe it
is connected straight through a building, so it learns geometry that cannot work.

> ⚠️ **Block E revised one prediction.** At the old 5 Mbps rate requirement the
> F3 → F4 rung was **inert** (Δ = +0.0 pp: the chain carried 8× the bar, so
> dividing by 1, 2 or 3 landed on the same side of it). At **15 Mbps** it is a
> **large** effect — Δ = **+26.5 pp**, flipping 26.6 % of chain-steps. Expect F4
> to matter. [`BLOCK_E.md`](BLOCK_E.md) §6.

---

## Decisions to settle before writing code

### 1. ⚠️ Fidelity is a property of the **channel**, not of the world

**The single most consequential decision in this block, and the current code gets
it wrong.**

`EnvConfig.use_occlusion` was left by Block D as "the F0 seam". It is not one.
`_clearance` returns `FREE_CLEARANCE_M` for **every node pair** when it is false —
which also switches off:

- the **sensor** (`sees = clearance ≥ 0`), so every drone sees the HVT anywhere
  within 830 m;
- the **jammer's** line of sight;
- the `chain_occluded` **diagnostic** (§2).

So as it stands, "F0" would not be a channel abstraction — it would be *a city
with no buildings*, and the F0→F1 gap would conflate **sensor** occlusion with
**link** occlusion. RQ1's primary result would then be uninterpretable in exactly
the dimension it exists to interpret.

**Decision: the sensor uses true occlusion at every rung. Fidelity gates only the
channel.** Reasons, in order of weight:

1. **RQ1 is literally about a channel model** — "which physical effects must a
   *channel model* include". A sensor is not part of the channel model.
2. **It keeps the attribution clean.** The F0→F1 gap becomes purely the cost of
   ignoring buildings *in the radio*, which is the claim being tested.
3. **Block E showed the two halves are not comparable in size.** Observation is
   ~93 % solved by geometry alone while the chain binds; letting the sensor vary
   across rungs would let the larger, uninteresting effect swamp the smaller,
   interesting one.

**What to build:** replace `use_occlusion` with `channel_occlusion`, and make
`_clearance` always compute the real thing. The rung decides only whether the
*capacity* computation consults it.

> **Consequence to accept and state:** occlusion runs at every rung, so **all
> five rungs cost the same** to simulate. F0 is not cheaper. That is good for the
> budget (THESIS_PLAN §3's 45 runs are unaffected) and good for comparability —
> no rung is advantaged by running more steps per GPU-hour.

**Record the alternative rather than pretending it does not exist.** A reviewer
may argue that papers using a radius channel typically model no buildings at all,
so F0 *should* be a building-free world. That is a defensible reading, and if it
is wanted it belongs as a **separate, explicitly named rung** (`F0-nogeo`), not
folded into F0. Do not make it the default: it confounds the primary result.

### 2. ⚠️ The RQ1 diagnostic must always use **true** occlusion

`chain_occluded` — *"fraction of steps where the intended chain passes through an
occluded link"* — is
[`THESIS_PLAN.md`](THESIS_PLAN.md) §4's failure-attribution metric and **the
direct signature of a radius-trained policy**. It is the number that turns RQ1
from a table into an explanation.

If it is computed from the *fidelity-gated* clearance, then **under F0 it reads
0.0 % by construction** — the F0 policy routes straight through buildings and the
metric reports it never happens. The headline diagnostic would be destroyed in
the one condition it exists to expose.

**Compute every diagnostic from the true geometry, always, independent of the
rung.** Decision 1 makes this free: the real clearance is computed anyway.

Same rule applies to `on_edge` × true-clearance, hop counts, and anything else
reported in `extras`. **The rung changes what the agent's world does, never what
the instrumentation sees.**

### 3. `C_max` for F0/F1 — unspecified in THESIS_PLAN, decide it deliberately

F0 is "`C_max` if `distance < R` else 0" and `C_max` is never given a value.

**Recommendation: the modulation cap** — `7.4 b/s/Hz × 10 MHz = 74 Mbps`, i.e.
`channel.capacity_mbps`'s own ceiling. Two reasons: it is the honest reading of a
connectivity-radius model (a link is *connected* or it is not, and a connected
link runs at full rate), and it is not a new free parameter — it is a quantity
the channel module already defines.

**Consequence, and it is the point:** under F0 a chain that exists geometrically
delivers 74 Mbps against a 15 Mbps requirement, so `mission_capable` reduces to
*"someone sees the HVT and a chain of ≤R hops exists"*. F0 is meant to be
permissive. That is the abstraction under test.

### 4. F3's jammer switch must **not** be `jammer_on`

Already flagged in [`DECISIONS.md`](DECISIONS.md) and on `EnvConfig` itself, and
repeated here because it is the trap most likely to be walked into.

`self.jammer_on` is the **curriculum's** jammer axis, sampled per episode from
the stage table. [`ENVIRONMENT.md`](ENVIRONMENT.md) requires the curriculum ramp
to run **identically in every fidelity condition**, with the fidelity level
deciding whether it *does* anything. Driving F3 from `jammer_on` confounds RQ1's
jammer rung with the curriculum ramp, and the two cannot be separated afterwards.

**F3 needs its own construction-time flag**, multiplied in alongside the
curriculum tensor:

```python
jam_mw = ... * self.jammer_on.unsqueeze(-1) * float(cfg.channel_jammer)
```

### 5. One composed enum, not five independent flags

Expose **`fidelity: Literal["F0","F1","F2","F3","F4"]`** and derive the flags from
it. Do not let a caller set them independently.

A condition that can be half-specified is a confound waiting to happen, and the
failure is silent — `channel_occlusion=False, channel_jammer=True` is not any rung
on the ladder, but it would run happily and produce a number that goes in a table.

| rung | `channel_occlusion` | `capacity_model` | `channel_jammer` | `reuse_limit` |
|---|---|---|---|---|
| F0 | ✗ | binary | ✗ | 1 |
| F1 | ✓ | binary | ✗ | 1 |
| F2 | ✓ | continuous | ✗ | 1 |
| F3 | ✓ | continuous | ✓ | 1 |
| F4 | ✓ | continuous | ✓ | **3** |

**`F4` must reproduce today's environment exactly** — it *is* the current
environment. Assert it: same seed, same actions, identical trajectories and
identical `extras`, against `fidelity="F4"` and against the pre-Block-F code
path. If that test does not pass, every Block D and Block E number is invalidated.

---

## The research task: calibrating `R`

THESIS_PLAN §2 is explicit that this is **"the first thing an examiner will
probe"**, and that an arbitrary `R` makes the comparison meaningless. It is also
the only part of Block F that is not plumbing.

**The pre-registered method is: `R` = the median link range measured under F4 in
the same city.** Implement that. Do **not** quietly substitute a method you like
better after seeing the number — that is the move this file exists to prevent.

Two ambiguities the pre-registration does not resolve, which must be settled and
*stated*:

- **Which links?** Those *actually carrying* chosen chains, or all candidate
  pairs? Report both. The chain-carrying set is the operational meaning of
  "link"; the all-pairs set is less policy-dependent.
- **Usable at what rate?** A single hop needs 15 Mbps; a hop in a 3-hop chain
  needs 45. Report `R` under both readings and say which the headline uses.

**Cross-check with degree matching**, reported alongside rather than instead:
choose `R` such that the mean number of usable links per node under F0 equals
that under F4. Connectivity *degree* is what actually determines chain topology,
so if the two methods disagree materially, that disagreement is itself worth
reporting — and only *then* is deviating from the pre-registered method
defensible, with the evidence attached.

**Run a sensitivity analysis over `R`** (±25 % and ±50 %) and report it. That is
the same move `routing.py` already makes with `reuse_limit`, and it converts the
softest number in RQ1 from an assertion into a measured range.

Measure it with **B0** — it is a fixed, tuned, non-learned policy, so the
calibration does not depend on a training run that does not exist yet.

---

## What to build

```
src/env/core.py        `fidelity` on EnvConfig; `channel_occlusion` replacing
                       `use_occlusion`; `channel_jammer`; binary capacity model;
                       diagnostics forced onto true geometry
src/env/test_core.py   F4 == today's env, byte for byte; each rung runs; the
                       reward is identical across rungs
scripts/calibrate_r.py `R` by both methods, both link sets, both rate readings,
                       plus the sensitivity sweep
docs/BLOCK_F.md        this file, with the measured numbers filled in
```

**Not a new module.** The rungs are a property of the existing env, and a
parallel `channel_f0.py` would drift from the real one. One code path, gated.

**`configs/` stays empty for now.** Per-condition YAML belongs with the training
entrypoints in Block G; F fixes the *seam*, not the experiment harness.

---

## Correctness

- **F4 is today's environment.** Identical trajectories and `extras` for a fixed
  seed and action sequence. The one test that, if it fails, invalidates Blocks D
  and E.
- **The reward is byte-identical across rungs.** Only the physics feeding it
  changes ([`REWARD.md`](REWARD.md)). Assert `RewardWeights` and the reward
  function are untouched by fidelity.
- **The sensor is identical across rungs.** `sees_hvt` for a fixed state must not
  depend on `fidelity` — decision 1, made enforceable.
- **The diagnostics are identical across rungs.** `chain_occluded` computed for a
  fixed geometry must not depend on `fidelity` — decision 2, made enforceable.
- **Monotonicity of permissiveness:** for a fixed state, `e2e_capacity_mbps`
  under F0 ≥ F1 ≥ F2 is *not* guaranteed (F2 is continuous, not a restriction of
  F1), but F3 ≥ F4 and F2 ≥ F3 are. Assert the two that hold and document why the
  others do not.
- **The curriculum runs identically in every condition.** Same stage schedule,
  same `jammer_on` draws for a fixed seed, regardless of rung. This is what
  decision 4 protects; test it directly.
- **Throughput is rung-independent** (decision 1's consequence). Spot-check it,
  because a rung that runs faster would quietly get more samples per GPU-hour.

---

## Expect, and report

- **F1 to be the big rung.** The hypothesis is that occlusion dominates.
- **F3 → F4 to be large** (+26.5 pp at fixed geometry under B0), reversing the
  null that the 5 Mbps requirement would have produced.
- **F0 to be very permissive** — mission-capable close to `observed` — which is
  the whole point of the abstraction being tested.
- **Φ_link to be degenerate under F0/F1.** Capacity is binary there, so
  `sigmoid((C − 15)/6)` saturates and the shaping term carries no gradient.
  [`REWARD.md`](REWARD.md) already predicts this: *"that is part of what training
  under a simplified channel means, not a bug."* Do not fix it — fixing it would
  make the rungs differ in their reward.

---

## What Block F does **not** build

- **Actors, critics, MAPPO, curriculum schedules** — Block G.
- **The training runs themselves.** F makes the conditions constructible; the
  45-run matrix is executed after the March 2027 freeze.
- **`interference_mode`** (`"scheduled"` vs `"concurrent"`). Block D specified the
  seam and it was never added. It is a *duplexing robustness check*, not a
  fidelity rung — leave it out unless PHYSICS.md's robustness table is being
  built, and do not confuse it with F4.
- **A second city** — Block B-shaped work, unowned, and it must be baked before
  the freeze if RQ2 keeps cross-morphology transfer. Flag it; do not start it
  here.

---

## Definition of done

- [ ] `fidelity` enum on `EnvConfig`, flags derived from it and not settable
      independently
- [ ] `use_occlusion` replaced by `channel_occlusion`; the sensor and all
      diagnostics run on true geometry at every rung
- [ ] `channel_jammer` separate from the curriculum's `jammer_on`, with a test
      that the curriculum ramp is identical across rungs
- [ ] `C_max` decided, justified in this file, and traceable to
      `channel.capacity_mbps`'s own ceiling rather than asserted
- [ ] **F4 reproduces today's environment exactly**, asserted
- [ ] `R` calibrated by the **pre-registered** method, cross-checked by degree
      matching, both link sets and both rate readings reported, sensitivity swept
- [ ] All five rungs run at `num_envs = 1024` without NaNs
- [ ] B0 evaluated under each rung as an env sanity check — not an RQ1 result,
      since B0 cannot be "trained under F0"
- [ ] Throughput spot-checked as rung-independent
- [ ] `ROADMAP.md`, `AGENTS.md`, `DECISIONS.md` updated

---

## Watch out for

- **`use_occlusion` as the F0 seam.** It is an all-geometry switch and it
  silently disables the sensor and the RQ1 diagnostic. Decisions 1 and 2.
- **Driving F3 from `jammer_on`.** Confounds the jammer rung with the curriculum
  ramp, unrecoverably. Decision 4.
- **A diagnostic that reads zero because the rung told it to.** Under F0,
  `chain_occluded` reading 0.0 % is a bug, not a finding.
- **Independent fidelity flags.** `channel_occlusion=False, channel_jammer=True`
  is not on the ladder, and nothing would stop it running.
- **Substituting the `R` calibration method after seeing the number.** Report
  both, deviate only with evidence.
- **Quoting a 5 Mbps-era number.** Everything in `BLOCK_D.md` predates Block E's
  rate change; that file carries a banner.
- **"Fixing" the degenerate `Φ_link` under F0.** The reward must be identical
  across rungs, so a fix would break the comparison it is meant to help.
