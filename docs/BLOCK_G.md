# Block G — MAPPO, the curriculum, and making something learn

**Goal:** a training entrypoint that takes any fidelity rung and any of the three
architectures and produces a policy that beats B0. Everything before this block
built an environment nobody has learned in yet.

Consumes Block A (reward), Block D (`core.py`, `skrl_wrapper.py`), Block E (B0 as
the floor, `evaluate.py` as the metrics harness) and Block F (the `fidelity`
seam). Produces `src/models/`, `src/training/`, `configs/`, and the answer to
whether this project has a thesis.

---

## Why this block is different from every block before it

A, B, C, D, E and F were **engineering with a checkable answer**. A test either
passed or it did not; a number was either measured or it was not. Block G is the
first block where the deliverable is *emergent* — you cannot unit-test your way
to a policy that learns, and the failure mode is not a crash but a flat reward
curve that could be caused by any of thirty things.

`ROADMAP.md` has said since it was written that this is **the place projects of
this shape stall.** Two consequences for how it is built:

1. **Sequence matters more than completeness.** The build order below is chosen
   so that the riskiest question is answered first and cheapest, not so that the
   most code is written first.
2. **Instrument before you tune.** Every hour spent staring at a flat return
   curve without per-term reward logging is an hour wasted. §4 is not optional
   polish.

---

## The gate

> **One toy run beats a random policy.** Then: a full run beats **B0 = 57.2 %**.

`ROADMAP.md` sets the first; `MODELS.md` sets the second and calls it a sanity
floor — *"any architecture must beat a random policy and at least match B0.
Failing that is a bug, not a finding."*

| | mission-capable, eval split, 5 seeds |
|---|---|
| random | 10.9 % [1.1] |
| `B0-geodesic` | 47.1 % [3.1] |
| **B0 — the floor** | **57.2 % [3.5]** |
| *sensor-only ceiling* | *93.0 %* |

**~36 points of headroom, all of it relay geometry.** That is what a learned
policy has to close.

---

## Build order — riskiest question first

### G1. Throughput, end to end, on the real hardware
Block D's outstanding item, and it gates the budget rather than the science.

The occlusion kernel clears the gate by ~3170× on a 5090, but the number the
thesis actually reports is **wall-clock for a 10 M-step run end-to-end including
the learner, target ≤3 h** — and that has never been measured, on any device,
with a learner attached. `THESIS_PLAN.md` §3's whole 120 GPU-hour budget rests on
it.

Do this in the first GPU session, before building anything on top:

- re-run `scripts/bench_env.py` on CUDA (Block D's pending re-run);
- then the same at `fidelity="F0"` and `"F4"`, to confirm Block F's rung-
  independence result (1.06 × spread, measured on MPS at batch 256) survives at
  CUDA scale with the learner competing for the device;
- report `num_envs` chosen on **learning** grounds, not throughput ones — Block C
  settled that the env is not the bottleneck.

**If ≤3 h/run does not hold, the 45-run matrix shrinks and RQ2 is the first thing
cut.** Better to know in the first week than in April 2027.

### G2. The smallest thing that can learn
**Do not build three architectures first.** Build one — the **flat MLP**, because
it has no relational machinery to debug — and get it to beat random on the
*easiest* curriculum stage, stage 1: stationary HVT, jammer off, 3× battery,
150 steps, exact cue.

Stage 1 is deliberately a different problem from the mission: fly out, form a
chain, hold station. If MAPPO cannot solve *that*, nothing downstream matters and
the fault is in the plumbing, the reward scale, or the learner config — all of
which are cheap to find at this size.

Success criterion: return curve rises, `mission_capable` beats random's 10.9 %.
Wall-clock target: minutes, not hours.

### G3. The learner config that is not skrl's default
Three things must be set, and two of them fail **silently**:

| setting | value | why |
|---|---|---|
| `time_limit_bootstrap` | `True` | skrl defaults to `False`. Every truncation would be treated as a genuine terminal state, the critic learns the world ends at 600 steps, and at γ = 0.997 that bias is large and invisible. `ENVIRONMENT.md` omits a time feature on exactly this reasoning (Pardo et al. 2018) |
| `discount_factor` | `core.GAMMA` = 0.997 | skrl defaults to 0.99, which AGENTS.md rules out — horizon 100 steps is blind to the hard end of the episode. **Import it, never retype it**: the env's PBRS shaping uses the same constant and the invariance proof requires the two to be identical |
| `value_preprocessor` | `RunningStandardScaler` | ⬅️ **the open item.** Returns are of order **300** at γ = 0.997 and the critic has to fit that scale. Flagged in `DECISIONS.md` and left as a `TODO(Block G)` in `skrl_wrapper.py` because the preprocessor needs the state width |

The first two are already applied by `training.skrl_wrapper.mappo_cfg()`. **Always
build the config through that function** — it also works around a skrl 2.1.0 bug
where `MAPPO_CFG` cannot be constructed with its own defaults.

`gae_lambda` is left alone: skrl already defaults it to 0.95.

### G4. The curriculum callback
`ENVIRONMENT.md` specifies the four stages and `core.STAGES` implements them;
what does not exist is the thing that moves `stage_weights` during a run.

Two rules from `ENVIRONMENT.md`, and both protect the results rather than the
learning:

1. **Fixed schedule by step count in the reported runs**, not adaptive
   advancement. Adaptive advancement lets easier fidelity rungs progress faster
   and hands them more experience at the final stage — which confounds RQ1
   directly. *Use adaptive advancement during development to find the schedule,
   then freeze it and use the same one everywhere.*
2. **Mix in earlier stages (~20 % of episodes)** rather than hard-switching, or
   the policy forgets the opening it still has to execute every episode. This is
   why `stage_weights` is a weight vector and not a stage index.

> ⛔ **Never use channel fidelity as a curriculum axis.** It is RQ1's independent
> variable. Same reasoning forbids ramping building density.

**The test that protects RQ1:** the schedule must produce an identical sequence
of `stage_weights` for a given seed regardless of `fidelity`. Block F already
asserts the env's *draws* are rung-independent; this asserts the *schedule* is.

### G5. The three architectures
Only now, and all three at once so the comparison is built in rather than
retrofitted. `MODELS.md` settles the layer choice — a custom MPNN on PyG's
`MessagePassing`, `message(x_i, x_j, e_ij) = MLP([x_i, x_j, e_ij])` — and the
reason is that **the DeepSets rung is then the identical layer with `e_ij`
zeroed**: same code path, same parameter count, same optimiser, one input masked.
No confound is possible.

Non-negotiable, all from `MODELS.md`:

- ⛔ **Never `SAGEConv`.** It cannot ingest edge features, so it silently collapses
  the GNN rung into DeepSets and RQ2 measures nothing. It is the layer people
  reach for by default.
- **All three consume `obs["flat"]` and unpack it with `core.unpack_flat()`**, so
  max-N padding is identical by construction rather than by discipline.
- **Equal hyperparameter budget** across the three, and say so in the
  methodology. Tuning the GNN harder is the single most likely way this result
  gets dismissed.
- **Match parameter counts to within ~20 %.**
- **2 message-passing layers is the ceiling** the graph justifies — it is softly
  fully connected at N ≤ 8, so diameter 1; a third layer propagates nothing new
  and causes over-smoothing.
- **The critic is identical across all three conditions**, so RQ2 isolates the
  actor. `skrl_wrapper.state()` already shares one global state across agents for
  this reason.

### G6. Retune `τ_c` and `τ_l`
Deferred here deliberately from Block E. They live in the **potential**, so by
the PBRS invariance proof they cannot move the optimum — only learning *speed*,
which could not be measured until a learner existed.

Block E supplies the empirical distributions the retune needs (`clearance_best`,
`C_e2e` under B0). Retune **once**, against learning speed, and then freeze.

> **The scale of `Φ` is likewise free** and is the one quantity in the reward
> tunable purely for learning speed with zero methodological consequence. Every
> other weight changes the objective — ⛔ **sweep nothing but `λ`.**

---

## Decisions to settle before writing code

### 1. ⚠️ Does the reward teach relaying below F4? — new, from Block F

Block F measured B0 under every rung and found that **F0, F2 and F3 all collapse
`mission_capable` onto `observed`** (92.0 %, 92.0 %, 83.1 % against a 92.0 %
sensor ceiling). The cause is structural: `reuse_limit = 1` below F4, so a chain
delivers its bottleneck undivided and the bottleneck sits far above 15 Mbps.

The mission term is the **dominant** reward term and it is binary. If it is on
92 % of the time, it is nearly constant, and there is little gradient toward
relay geometry. `Φ_link = sigmoid((C_e2e − 15)/6)` does not rescue it either — at
F2's median bottleneck of 46.5 Mbps that sigmoid reads 0.995, i.e. saturated.

**This is most likely the mechanism of RQ1's finding rather than a problem**:
*train in a permissive world → little pressure to learn relay geometry → the
policy does not → it fails under F4.* That is the hypothesis with a causal story
attached.

**But there is a version that would hurt.** If policies trained at F0, F2 and F3
all learn the same thing ("go look at the car") and then fail under F4 in the same
way, the *headline* result survives — the abstraction costs you X — while the
**attribution between rungs** does not. RQ1 has five rungs precisely so effects
can be separated; this would blur four of them together.

**Decide how it gets checked, and check it on the first pilots rather than in
April 2027.** Cheap diagnostics, all available from `evaluate.py` today:

- do F0-, F2- and F3-trained policies differ in **hop count** and **mean hop
  distance**, even when their mission scores under F4 coincide?
- does `chain_occluded` under F4 separate them? It is RQ1's designated
  failure-attribution metric and it is a *behavioural* signature, not a score.
- does the **5th-percentile capacity** separate them where the mean does not?

⛔ **Do not "fix" this by changing the reward per rung.** The reward must be
byte-identical across rungs or the comparison is destroyed — Block F asserts it.

### 2. Which N, and it is already settled — do not re-open it

**Train at `N = 5` only.** Evaluate zero-shot at `N ∈ {3,5,8}`.

Training at more than one N turns RQ2's transfer columns into in-distribution
tests, costs +15 runs against a budgeted 45, and — at N = 3, where every drone is
load-bearing — removes the slack RQ3 needs. Measured and settled in
`DECISIONS.md`; it has been proposed twice.

**Put the analytical weight at N = 8**, where better control is worth **+25.9 pp**
against +3.2 pp at N = 3. Hardness and headroom move in opposite directions.

### 3. What "equal hyperparameter budget" means operationally

`MODELS.md` requires it and calls unequal tuning the most likely way the result
gets dismissed. It needs an operational definition *before* tuning starts, or it
becomes unfalsifiable afterwards.

Suggested and cheap: **the same search space, the same number of trials, the same
selection rule, for each of the three architectures** — recorded in `configs/`
and reported in the methodology as a number of trials, not as a claim of
fairness.

### 4. Seeds, and what counts as a run

`AGENTS.md`: **≥5 seeds for anything reported as a finding. Median + IQR, never
mean ± std.** RL returns are not normally distributed. Never report single runs.

Note the asymmetry Block E established and Block F followed: **means across
episodes *within* a seed, median + IQR *across* seeds.** A median within a seed
reports 0.0 % for every rare-event metric.

### 5. Evaluation runs on the eval route split, and under F4

256 held-out routes (`eval_routes=True`). This is the only generalisation check
that survives the second city being cut, so it is load-bearing now in a way it
was not before — [`DECISIONS.md`](DECISIONS.md).

**Every RQ1 number is measured under `fidelity="F4"`**, whatever the policy was
trained under. That is the entire design of the ladder.

---

## What to build

```
src/models/          shared trunk + the three actor heads; one critic
src/models/test_*.py parameter-count parity; the DeepSets ablation is the GNN
                     with e_ij zeroed; off-N forward passes at N in {3,5,8}
src/training/train.py       the entrypoint: fidelity x architecture x seed
src/training/curriculum.py  the stage_weights callback + its schedule
src/training/test_*.py      schedule is fidelity-independent for a fixed seed
configs/             one YAML per condition. Block F deliberately left this here
scripts/eval_policy.py      checkpoints -> the same RolloutMetrics B0 produces
```

**Reuse `src/baselines/evaluate.py`.** It already produces every metric the
thesis reports — mission-capable, `chain_occluded`, hop distribution, the RQ3
anticipation lead, the rate-division counterfactual — and B0's numbers came out
of it. A learned policy scored by a *different* harness is not comparable to B0,
and that comparison is the point of B0 existing.

---

## Correctness

- **The curriculum schedule is identical across fidelity conditions** for a fixed
  seed. Test it directly; it is what decision 1 of Block F protects at the env
  level and this protects at the training level.
- **γ agrees between env and learner.** `skrl_wrapper` imports `core.GAMMA`; a
  test already asserts it. The PBRS invariance proof fails silently otherwise.
- **`terminated` and `truncated` stay distinct**, and the learner bootstraps at
  truncation. Wrappers routinely collapse the two; Block D asserts it in the smoke
  test and `time_limit_bootstrap=True` is what acts on it.
- **The DeepSets rung is the GNN with `e_ij` zeroed**, asserted by running both
  and checking parameter counts and shapes match.
- **All three architectures accept N ∈ {3,5,8}** without retraining or reshaping.
- **The edge capacity feature actually varies.** `MODELS.md`: informative share is
  93.7 % after `CAPACITY_CLAMP` moved to 5.0. ⚠️ **Check this before trusting any
  RQ2 null** — a GNN cannot weight messages by a constant, and the resulting null
  would look like a finding about relational structure.

---

## Watch out for

- **Tuning against the eval split.** B0 was tuned on the training split for
  exactly this reason and the cost was measured at 0.6 pp. Do the same.
- **Adaptive curriculum advancement in a reported run.** It hands easier fidelity
  rungs more experience at the final stage and confounds RQ1 unrecoverably.
- **`SAGEConv`.** ☠️
- **Reporting a single run**, or mean ± std.
- **Changing the reward to make something learn.** The reward is the objective;
  changing it changes what "success" means and invalidates every earlier number.
  `Φ`'s scale and `τ_c`/`τ_l` are the *only* safe knobs, because PBRS proves they
  cannot move the optimum.
- **Quoting a 5 Mbps-era number.** Everything in `BLOCK_D.md` predates Block E's
  rate change; that file carries a banner.
- **Comparing numbers measured on different devices.** `torch.Generator` streams
  differ per device, so the same seed draws different episodes. Block F's tables
  are MPS; Block E's are CPU.

---

## Definition of done

- [ ] `bench_env.py` re-run on CUDA; **wall-clock for 10 M steps end-to-end
      reported**, and the ≤3 h target either met or the matrix re-planned
- [ ] one toy run beats random — **the gate**
- [ ] `value_preprocessor` wired, closing `skrl_wrapper.py`'s `TODO(Block G)`
- [ ] curriculum callback, with a fixed step-count schedule and a test that it is
      fidelity-independent
- [ ] all three architectures, equal budget, parameter counts within 20 %
- [ ] a full F4 run **beats B0's 57.2 %** on the eval split, ≥5 seeds
- [ ] `τ_c` / `τ_l` retuned once against learning speed, then frozen
- [ ] pilot check on Block F's open question: do F0/F2/F3-trained policies
      separate on hop count, `chain_occluded` or p5 capacity under F4?
- [ ] `configs/` per condition; `ROADMAP.md`, `AGENTS.md`, `DECISIONS.md` updated

---

## What Block G does **not** build

- **The 45 reported runs.** Those execute after the March 2027 freeze. G makes
  them possible and finds the curriculum; it does not spend the budget.
- **A second city.** ⛔ Cut — [`DECISIONS.md`](DECISIONS.md).
- **Sionna validation.** Block H, fully parallel, optional.
- **New physics of any kind.** The environment is frozen at the end of March and
  Block F was the last block permitted to touch it.
