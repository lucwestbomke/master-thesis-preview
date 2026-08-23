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

## ✅ Built and measured

`fidelity` is a five-value enum on `EnvConfig` with every flag derived from it;
`F4` is the default and **reproduces the pre-Block-F environment element for
element** against a frozen trace. `R` is measured, not chosen. All five rungs run
at `num_envs = 1024` without a single non-finite value, and within 1.05× of each
other's throughput.

Regenerate everything below with:

```bash
uv run python scripts/calibrate_r.py    --seeds 8 --num-envs 64 --device mps
uv run python scripts/eval_fidelity.py  --seeds 5 --num-envs 64 --device mps
uv run python scripts/render_episode.py --policy b0 --route 12 --compare-fidelity
```

> **`--device mps` is a ~17× speedup on Apple silicon** (5 min against ~1.5 h)
> and the physics is identical — the occlusion kernel is bit-identical across
> CPU/MPS/compiled and capacity agrees to 1.9e-5 Mbps on a fixed state. But
> `torch.Generator` streams differ per device, so **the same seed draws different
> episodes**, and a number measured on one device must not be compared with one
> measured on another. [`DECISIONS.md`](DECISIONS.md).

### The ladder under B0

⚠️ **Not an RQ1 result.** RQ1 trains one policy per rung and evaluates all of
them under F4; B0 is scripted and fixed, so it cannot be "trained under F0".
These rows describe **five environments**, not five policies.

`eval_fidelity.py --only ladder`, 5 seeds × 64 eval episodes, median [IQR]:

| rung | capable | observed | chain occluded | chain bottleneck | return |
|---|---|---|---|---|---|
| **F0** | 92.0 % [1.8] | 92.0 % [1.8] | **85.8 %** [2.0] | 72.8 [1.2] | 433.3 |
| *F0-nogeo* | *100.0 %* [0.0] | *100.0 %* [0.0] | *0.0 %* | *74.0* | *490.1* |
| **F1** | **27.9 %** [0.1] | 92.0 % [1.7] | 0.0 % | 69.4 [1.2] | 51.0 |
| **F2** | 92.0 % [1.7] | 92.0 % [1.7] | 65.5 % [1.5] | 46.5 [1.9] | 435.1 |
| **F3** | 83.1 % [1.2] | 92.0 % [1.7] | 62.0 % [4.2] | 40.6 [1.2] | 382.1 |
| **F4** | 56.0 % [5.2] | 92.0 % [1.7] | 64.3 % [6.0] | 40.1 [2.9] | 220.1 |

**Read the `observed` column first.** It is **92.0 % at every rung on the
ladder** and moves only for the world-level `F0-nogeo` variant. That is
decision 1 holding: the sensor runs on true geometry everywhere, so the F0→F1
gap is the cost of ignoring buildings *in the radio* and nothing else. A rung
that moved this column would make the primary result uninterpretable.

**Then read `chain occluded` at F0: 85.8 %.** Decision 2 holding. Computed from
the fidelity-gated clearance it would read 0.0 % by construction, and the
headline failure-attribution metric would be destroyed in the one condition it
exists to expose. F1's 0.0 % is legitimate and is a fact about the *physics*,
not the metric: F1 is the only rung where occlusion is a hard veto, so a chosen
chain cannot contain a blocked link.

**F4 = 56.0 % [5.2] against Block E's B0 = 57.2 % [3.5].** An independent,
statistical confirmation of what `test_golden.py` asserts element-wise: the
Block F seam did not move the environment Blocks D and E measured.

#### Three things the table says that the spec did not predict

1. **F1 is the harshest rung on the ladder, not an intermediate one.** 27.9 %
   against F4's 56.0 %. The ladder is *not* monotone in difficulty. F1 vetoes
   any blocked link outright, where F4 charges blockage loss and lets the link
   carry anyway, and F1 additionally drops everything beyond `R` — so B0 has no
   chain at all on **72.5 %** of steps. This does not make F1 wrong: "requires
   an unoccluded ray" is the literal and only reading of THESIS_PLAN's rung. But
   it changes what to expect from RQ1, because an F1-trained policy will have
   learned a world *harder* than the one it is tested in, and that is a different
   transfer problem from F0's.
2. **F0, F2 and F3 all collapse `mission_capable` onto `observed`** (92.0 %,
   92.0 %, 83.1 % against a 92.0 % ceiling). This is the 5 Mbps pathology Block E
   diagnosed, reappearing structurally: F0–F3 carry `reuse_limit = 1`, so a
   chain delivers its bottleneck undivided, and the bottleneck sits far above
   15 Mbps. **It is not a defect — it is what the cumulative ladder means.** The
   divisor is the rung that makes the mission hard, so every rung below it is
   permissive by construction. It does mean the *within-rung* numbers above
   understate what RQ1 will measure, because RQ1 evaluates every rung's policy
   **under F4**.
3. **The jammer rung is not inert** — F2 → F3 costs 8.9 pp of mission success
   and 5.9 Mbps of chain bottleneck at fixed geometry. An early small-sample
   probe suggested it was inert; at 5 seeds × 64 episodes it is not. Worth
   recording because the analogous claim about the *divisor* was true at 5 Mbps
   and had to be fixed by moving the rate requirement.

### Chain topology per rung

`eval_fidelity.py --only hops`, pooled across seeds. Shares conditioned on a
chain existing, per [`BLOCK_E.md`](BLOCK_E.md) §6:

| rung | 0 hop | 1 | 2 | 3 | 4 | 5+ | multi-hop | divisor saturated |
|---|---|---|---|---|---|---|---|---|
| F0 | 7.5 % | 18.7 % | 27.4 % | 31.1 % | 4.4 % | 10.9 % | 79.8 % | 50.2 % |
| *F0-nogeo* | *0.0 %* | *82.3 %* | *5.3 %* | *12.4 %* | — | — | *17.7 %* | *12.4 %* |
| F1 | **72.5 %** | 5.7 % | 12.4 % | 7.2 % | 1.2 % | 1.0 % | 79.2 % | 34.1 % |
| F2 | 7.5 % | 6.7 % | 37.5 % | 26.5 % | 17.1 % | 4.7 % | 92.7 % | 52.2 % |
| F3 | 7.6 % | 9.0 % | 39.5 % | 26.0 % | 14.9 % | 3.0 % | 90.3 % | 47.5 % |
| F4 | 7.5 % | 17.9 % | 44.0 % | 14.5 % | 13.1 % | 2.9 % | 80.6 % | 33.0 % |

**The rungs below F4 build longer chains than F4 does**, and the reason is the
divisor itself: with `reuse_limit = 1` there is no cost to an extra hop, so the
widest-path DP takes free hops whenever they raise the bottleneck. F4's `min(n,3)`
is what creates pressure toward short chains. That is the rung stated in units of
behaviour rather than of capacity.

**`F0-nogeo` is 82.3 % single-hop and 100 % mission-capable.** Deleting the
buildings deletes the relay problem outright — which is the measured reason it
must stay a separately-named condition and never be folded into F0.

### Throughput is rung-independent

`eval_fidelity.py --only throughput`, batch 256, MPS, compiled:

| rung | F0 | F1 | F2 | F3 | F4 | *F0-nogeo* |
|---|---|---|---|---|---|---|
| env-steps/s | 32,563 | 32,553 | 32,169 | 32,022 | 30,848 | *57,676* |

**Spread across the five ladder rungs: 1.06×.** Decision 1's stated consequence,
confirmed: occlusion runs at every rung, so no rung is cheaper and none gets more
samples per GPU-hour. `F0-nogeo` is 1.87× — it genuinely skips the geometry,
which is a third reason it is a separate condition.

> ⚠️ **This benchmark needs a batch large enough that occlusion dominates**, and
> it takes its own (`--throughput-envs`, default 256) rather than `--num-envs`.
> Measured at 64 the step is dominated by per-call overhead, the spread inflates
> to 1.16×, and `F0-nogeo` — which genuinely skips the geometry — comes out
> **0.85×, i.e. slower than F4**. Both runs are correct measurements of different
> things; only the occlusion-dominated one answers the question.

All six conditions ran at `num_envs = 1024` for five steps with **zero
non-finite values** in any observation, reward or `extras` tensor
(`--only scale`).

### What the pictures show

`render_episode.py --policy b0 --route 12 --compare-fidelity` draws the same
policy on the same route at every rung. Side by side, F0 against F4:

- **F0**: end-to-end rate pinned at `C_max` off the top of the axis, hop count
  mostly **1** — one drone reaches the MCV directly — and the "chosen chain
  crosses a building" band covering a continuous ~60 s stretch.
- **F4**: a real 1–3 hop chain delivering 27–40 Mbps, varying continuously, with
  essentially no occluded steps.

The drone tracks differ too, and that is the observation-gating decision below
made visible: under F0 the per-edge clearance feature is saturated, so B0's link
repair correctly has nothing to climb.

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


> **Decisions 6 and 7 were not in the original spec.** They were forced by
> building it, and both are recorded here because each is a way to break RQ1
> quietly, in the same family as 1, 2 and 5.

### 6. ⚠️ The **observation's** channel features follow the rung — decisions 1 and 2's third sibling

Decision 1 settles the sensor and decision 2 settles the diagnostics. **Neither
settles the observation, and the observation *is* the agent's world** — so the
same question arises there, and it is the one place where "the rung changes what
the agent's world does" actually has teeth.

Six of the 108 dims are channel state rather than sensing. Three already follow
the rung for free, being computed from its capacity matrix (`on_path`, e2e
capacity, per-edge capacity). Three did not: the **measured noise floor**, the
**clearance margin to the MCV**, and the **per-edge clearance margin**.

**Decision: they follow the rung.** *Sensor features report the sensor, channel
features report the channel model in force, diagnostics report the truth.*

Ungated, F0's observation is internally contradictory — an edge reporting
74 Mbps beside a clearance feature reading −150 m — and that contradiction is
learnable in exactly the direction that would **understate** the F0 → F1 gap RQ1
exists to measure. It is not hypothetical: B0's `_update_repair` hill-climbs on
those two features and is, in its own docstring, *"the only part of B0 aimed at
`chain_occluded`"*. Gated, it correctly goes inert under F0 — there is nothing to
repair in a radius world — and the difference is visible in the rendered tracks.

The ego **clearance to the HVT** stays on true geometry: it is the sensor's own
ray, the one `sees_hvt` is computed from, and gating it would put the soft flag
and the hard gate into disagreement. The split is between *sensing the target*
and *sensing the link*, not between geometry and not-geometry. Full argument in
[`DECISIONS.md`](DECISIONS.md).

### 7. `no_buildings` is a world-level flag and is **not** on the ladder

`use_occlusion` could not simply be renamed, because the test suite depends on it
for speed: it appears at eight sites as `FAST`, and occlusion is **37×** the rest
of the step on CPU (20 steps at `num_envs=8`: 0.031 s without, 1.154 s with).

So it split in two. `fidelity` decides whether occlusion costs the **link**
anything; `no_buildings` removes buildings from the **world**. The second is
documented as not-a-rung, and it is also exactly the `F0-nogeo` variant
decision 1 records as the defensible alternative reading of F0 — constructed as
`fidelity="F0", no_buildings=True` and always reported under that name.

**The measurements vindicate keeping them apart.** `F0-nogeo` scores 100 %
mission-capable, is 82.3 % single-hop, and runs 1.97× faster. Folding it into F0
would have deleted the relay problem and given one rung twice the samples per
GPU-hour.

A third flag went the same way: **`reuse_limit` was a settable field** and is now
derived from the rung, because `fidelity="F0", reuse_limit=3` is not on the
ladder and nothing would have stopped it running. `PHYSICS.md`'s duplexing
robustness check survives as `duplexing_override`, which raises anywhere except
`F4`.
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

Measure it with **B0** — a fixed, tuned, non-learned policy, so the calibration
does not depend on a training run that does not exist yet.

### ✅ Result: `R` = 524 m

`scripts/calibrate_r.py`, 8 seeds × 64 eval episodes under B0, median [IQR]
across seeds.

#### ⚠️ A third ambiguity the pre-registration does not name, and it is the big one

"The median link **range**" has two readings, and they differ by 2×:

| reading | what it measures | all pairs @15 | all pairs @45 | chain-carrying @15 |
|---|---|---|---|---|
| **A — median LENGTH** of a realised usable link | how long the links a policy forms are | 266 m [9] | 242 m [10] | 318 m [12] |
| **B — median RANGE**: where `P(usable │ d)` crosses 0.5 | how far a link reaches | **524 m** [22] | 233 m [10] | — |

Reading A is the literal "median over observed links"; reading B is the literal
"median of the *range* of a link". The pre-registration says both and neither.

**The headline is reading B**, on two arguments neither of which is "it is the
bigger number":

1. **A link's *range* is how far it reaches.** Reading A measures how far apart
   B0 happens to park its drones — a fact about one scripted policy, and `R` will
   be used to train policies that do not exist yet.
2. **Reading A makes F0 *stricter* than F4.** At 266 m, F0's mean degree is 1.63
   against F4's 2.72, so an F0-trained policy would face a *sparser* graph than
   the model it is tested against. A connectivity-radius abstraction is
   optimistic — it ignores buildings — and decision 3 above states that F0 must
   be permissive. An `R` that inverts that is not modelling the literature RQ1 is
   about.

> **Reading A was implemented first, exactly as pre-registered, and its number is
> reported unchanged.** What prompted the re-examination was the number looking
> wrong; what settles it is the argument above, which does not depend on which
> value is larger. The degree cross-check below is the independent test.

The two ambiguities the spec *does* name turn out to be the smaller ones. "Which
links" only bites under reading A — under reading B the chain-carrying set has no
reading at all, because it is selected *by* usability, so `P(usable │ d) ≈ 1`
along it by construction. And **link class is immaterial**: A2A reaches 487 m
[96] against A2G's 511 m [13], so a single blended `R` — which is what the
abstraction under test actually uses — costs nothing.

#### Cross-check: degree matching agrees

| rate reading | F4 degree | degree-matched `R` | reading B | reading A |
|---|---|---|---|---|
| single-hop (15 Mbps) | 2.72 | **418 m** [7] | 524 m [22] | 266 m [9] |
| chain-hop (45 Mbps) | 1.82 | 282 m [8] | 233 m [10] | 242 m [10] |

**418 m is 0.80× the headline and falls inside its ±25 % band [393, 656].** So
the two methods — a quantile of the usability curve, and solving for equal mean
node degree — **agree to within the sensitivity that is swept and reported
anyway, and nothing in RQ1 turns on the choice between them.** Reading A's 266 m
is 0.51× and falls outside that band, which is the independent evidence against
it. Pinned by `test_fidelity.py`, so a future change to `R` has to come with a
re-measurement.

#### Sensitivity: the softest number in RQ1 barely matters

| `R` | × base | F0 mean degree | F0 links actually blocked | B0 mission-capable @F0 |
|---|---|---|---|---|
| 262 m | 0.50 | 1.60 | 20.4 % | 68.3 % |
| 393 m | 0.75 | 2.63 | 27.0 % | 93.3 % |
| **524 m** | **1.00** | **3.33** | **32.0 %** | **93.4 %** |
| 656 m | 1.25 | 3.87 | 36.7 % | 93.4 % |
| 787 m | 1.50 | 4.25 | 40.0 % | 93.4 % |

**B0's mission success under F0 is flat at 93.4 % from 0.75× to 1.5×.** The F0
arm is insensitive to `R` across the entire plausible range; only halving it
breaks the saturation. This is the answer to the examiner's question, and it is
stronger than defending a point estimate: *the parameter moves and the conclusion
does not*.

**"F0 links actually blocked" = 32.0 %** is the abstraction error the F0 → F1
rung isolates, in one number: **a third of the links F0 believes in run straight
through a building.** It agrees closely with Block B/E's independent 31.2 % of
A2A links blocked at 80 m.

#### How much does `R` depend on the policy it was measured under?

| state distribution | `R` (reading B, 15 Mbps) | F4 degree |
|---|---|---|
| random | 706 m [35] | 2.77 |
| waypoint | 590 m [27] | 3.54 |
| **B0** | **524 m** [22] | 2.72 |

**Report this honestly: the estimator *is* policy-dependent**, and the spread
(1.35× from B0 to random) is *not* fully covered by the ±25 % band. `R` is
calibrated on one scripted policy's state distribution and will be used to train
policies that do not exist yet; that circularity is unavoidable and cannot be
argued away.

What defuses it is the sensitivity table above rather than the spread itself:
across 262 → 787 m — a range that comfortably contains every policy's estimate —
B0's F0 mission success moves by 0.1 pp above 0.75×. **The parameter is uncertain
to about ±35 %; the conclusion is not sensitive to ±50 %.**

#### Cross-device replication

The whole calibration was run twice, on different hardware and different sample
sizes, which — because `torch.Generator` streams differ per device — means two
disjoint draws of episodes:

| | reading B | reading A | degree-matched | F4 degree |
|---|---|---|---|---|
| MPS, 8 seeds × 64 | 524 m [22] | 266 m [9] | 418 m [7] | 2.72 |
| CPU, 5 seeds × 32 | 536 m [39] | 266 m [9] | 412 m [12] | 2.74 |

Agreement to 2 %, with reading A identical. The estimator is stable, and the
device is not doing any work.

## What to build

```
src/env/core.py           `fidelity` enum + LADDER table; flags derived, not
                          settable; `no_buildings` as the world-level escape
                          hatch; `_clearance` returns (true, channel); binary
                          capacity; diagnostics and sensor forced onto true
                          geometry; `F0_RADIUS_M` = the measured `R`
src/env/golden.py         the frozen pre-Block-F trace: scenarios, the driver,
                          save/load. Shared by the capture and the check so the
                          two provably run the same code
src/env/test_golden.py    ✅ F4 == the pre-Block-F env, element for element
src/env/test_fidelity.py  ✅ 38 tests: the five decisions, made enforceable
scripts/capture_f4_golden.py   freezes the trace (and refuses to overwrite it)
scripts/calibrate_r.py    `R` by both methods, both link sets, both rate
                          readings, the sensitivity sweep and the
                          state-distribution robustness column
scripts/eval_fidelity.py  the ladder under B0, chain topology, throughput,
                          and the `num_envs = 1024` finiteness check
src/viz/episode.py        `fly(..., fidelity=)`, so the same policy on the same
                          route can be drawn at every rung
data/f4_golden.pt.gz      the frozen trace itself, 2.0 MB, committed on purpose
```

**Two scripts, not one.** The spec named `calibrate_r.py`; that file does `R` and
only `R`. The rung sanity tables, the throughput comparison and the scale check
are properties of the *rungs* rather than of `R`, so they live in
`eval_fidelity.py`. Both follow the `--only` section pattern.

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
- **Monotonicity of permissiveness:** ⚠️ **this spec understated what holds.**
  `F0 ≥ F1` *is* guaranteed — F1's capacity is F0's times an unoccluded mask,
  elementwise, and `best_relay_path` is monotone in the capacity matrix at a
  fixed `reuse_limit`. So **three** orderings are asserted end-to-end, not two:
  `F0 ≥ F1`, `F2 ≥ F3`, `F3 ≥ F4`.

  Only `F1 ↔ F2` is genuinely unordered, and only one direction is reliable.
  `F2 > F1` happens constantly (F2 drops the radius cutoff and turns occlusion
  into a penalty rather than a veto). `F1 > F2` needs a link inside `R`,
  unoccluded, and *still* below the modulation cap — and a clear link at
  `R = 524 m` runs ~30 dB above what 7.4 b/s/Hz needs, so at the calibrated `R`
  it may not occur in a rollout at all. The test asserts that direction at a
  deliberately wide `R`, so it measures the mechanism instead of the sample and
  does not start failing when `calibrate_r.py` moves `R`.
- **The curriculum runs identically in every condition.** Same stage schedule,
  same `jammer_on` draws for a fixed seed, regardless of rung. This is what
  decision 4 protects; test it directly.
- **Throughput is rung-independent** (decision 1's consequence). Spot-check it,
  because a rung that runs faster would quietly get more samples per GPU-hour.

---

## Expect, and report — ✅ scored

| prediction | outcome |
|---|---|
| **F1 to be the big rung** | ✅ and then some. At fixed geometry F0 → F1 is **−64.1 pp**, the largest gap on the ladder — but see below: F1 is harsher than F4, which the spec did not anticipate. |
| **F3 → F4 to be large** (+26.5 pp under B0) | ✅ **−27.1 pp** measured independently here (83.1 → 56.0), reproducing Block E's +26.5 pp. |
| **F0 to be very permissive**, capable close to `observed` | ✅ exactly equal: **92.0 % against 92.0 %**. |
| **Φ_link degenerate under F0/F1** | ✅ capacity is binary there, so `sigmoid((C−15)/6)` saturates at 1.000 or 0.076. Left alone, as required — the reward must be identical across rungs. |

**What the spec did not predict, and should have:**

- **The ladder is not monotone in difficulty.** F1 (27.9 %) is *harder* than F4
  (56.0 %), because it vetoes blocked links outright where F4 charges blockage
  loss and lets them carry. An F1-trained policy will have learned a world harder
  than the one it is evaluated in — a different transfer problem from F0's, and
  worth saying so in Chapter 6 rather than discovering it in the results.
- **F0, F2 and F3 all collapse `mission_capable` onto `observed`**, because
  `reuse_limit = 1` below F4. Structural, not a defect, but it means the
  within-rung numbers understate what RQ1 measures — RQ1 evaluates every rung's
  policy **under F4**.
- **The rungs below F4 build *longer* chains** (F0 reaches 5+ hops on 10.9 % of
  steps against F4's 2.9 %), because without the divisor an extra hop is free.

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

- [x] `fidelity` enum on `EnvConfig`, flags derived from it and not settable
      independently — and `reuse_limit`, which was a settable field, demoted with
      it
- [x] `use_occlusion` replaced; the sensor and all diagnostics run on true
      geometry at every rung (`observed` = 92.0 % at all five; `chain_occluded`
      = 85.8 % at F0)
- [x] `channel_jammer` separate from the curriculum's `jammer_on`, with a test
      that the curriculum ramp is bit-identical across rungs
- [x] `C_max` decided and traceable to `channel.capacity_mbps`'s own ceiling
      (74.0 Mbps), asserted rather than written as a literal
- [x] **F4 reproduces the pre-Block-F environment exactly**, asserted against a
      committed trace — and independently corroborated statistically (56.0 %
      against Block E's 57.2 %)
- [x] `R` calibrated by the pre-registered method (**524 m**), cross-checked by
      degree matching (418 m, 0.80×, inside the ±25 % band), both link sets and
      both rate readings reported, sensitivity swept, and a **third** ambiguity
      the pre-registration did not name surfaced and resolved
- [x] All five rungs run at `num_envs = 1024` without NaNs
- [x] B0 evaluated under each rung as an env sanity check — reported as five
      environments, not five policies
- [x] Throughput spot-checked as rung-independent (**1.06×** across the ladder)
- [x] `ROADMAP.md`, `AGENTS.md`, `DECISIONS.md` updated

**Left for Block G, deliberately:** the `configs/` per-condition YAML, which
belongs with the training entrypoints. Block F fixes the *seam*, not the
experiment harness.

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
- **Reading a Block F number against a Block E one without checking the device.**
  `torch.Generator` streams differ per device, so the same seed draws *different
  episodes* on MPS than on CPU. The physics is identical (bit-identical
  occlusion; 1.9e-5 Mbps on capacity for a fixed state), but the sample is not.
  Block F's tables are MPS; Block E's are CPU. Where they are compared here it is
  on purpose and the agreement is the point.
- **Quoting the throughput comparison at a small batch.** Below ~128 envs the
  step is overhead-dominated and the comparison inverts — `F0-nogeo` measures
  *slower* than F4 despite skipping the geometry entirely.
- **Re-capturing `data/f4_golden.pt.gz` to make a test pass.** It is the only
  record of what the environment did before the ladder existed; re-capturing
  compares the new code against itself. `scripts/capture_f4_golden.py` refuses to
  overwrite without `--force` and says why.
- **Treating `F1 > F0` as "a bit more realism".** F1 is the *harshest* rung
  measured (27.9 % against F4's 56.0 %), because occlusion is a hard veto there
  rather than a graded penalty. The ladder is cumulative in *effects*, not
  monotone in *difficulty*.
