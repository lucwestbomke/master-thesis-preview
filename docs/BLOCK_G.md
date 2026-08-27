# Block G — MAPPO, the curriculum, and making something learn

**Goal:** a training entrypoint that takes any fidelity rung and any of the three
architectures and produces a policy that beats B0. Everything before this block
built an environment nobody has learned in yet.

Consumes Block A (reward), Block D (`core.py`, `skrl_wrapper.py`), Block E (B0 as
the floor, `evaluate.py` as the metrics harness) and Block F (the `fidelity`
seam). Produces `src/models/`, `src/training/`, `configs/`, and the answer to
whether this project has a thesis.

---

## 📏 The Block G results table

Everything below is **eval split, F4, stage 4, CUDA, 5 seeds, median [IQR]**,
scored through `evaluate.py` — one device, one split, one harness, so these rows
are comparable with each other and with nothing else in this repo.

| policy | mission capable | observed | observer tenure | hop mean | p5 capacity |
|---|---|---|---|---|---|
| random | 10.7 % [0.2] | 21.9 % | 16.3 | 0.4 | 0.0 |
| MAPPO **MLP** | 31.2 % [1.2] | 53.8 % | 34.1 | 1.00 | 0.0 |
| MAPPO **DeepSets** | 38.1 % [1.0] | 65.3 % | 41.6 | 1.23 | 0.0 |
| MAPPO **GNN** | **41.2 % [3.8]** | 66.5 % | 47.2 | 1.27 | 0.0 |
| **B0 — the floor** | **57.3 % [3.9]** | **92.8 %** | **294.7** | **2.1** | **4.5** |

**The gate is not met: 41.2 % against 57.3 %, a 16.1 pp gap.** The learned policy
closes 65 % of the distance from random to B0 and stops.

✅ B0's **57.3 %** here reproduces Block E's **57.2 %** measured on CPU. Device
changes which episodes are drawn, not the physics, and at 5 × 128 episodes the
aggregate is stable — which cross-validates the harness across devices even
though ⛔ individual runs still must not be compared across them.

### 📏 The gap is `observed`, and nothing else

| conditioned on holding a sightline | random | MLP | DeepSets | GNN | B0 |
|---|---|---|---|---|---|
| `capable / observed` | 0.489 | 0.580 | 0.583 | **0.620** | **0.617** |
| `hop_mean / observed` | 1.83 | 1.86 | 1.88 | **1.91** | **2.26** |

**Given a sightline, the GNN converts it into mission capability exactly as well
as B0 does — 0.620 against 0.617.** The entire 16.1 pp gap is the first row of
the table above: `observed` at 66.5 % against 92.8 %.

⚠️ And the second row is the finding nobody was looking for. **Conditioned on
observing, every learned policy's chain is indistinguishable from a random
policy's** — 1.86 / 1.88 / 1.91 against random's **1.83** — while B0 sits at
2.26. The routing is `best_relay_path`, computed by the env from whatever
geometry the policy produces, so 1.8 hops is simply what scattered drones give
you. **The swarm learned to fly at the target and learned nothing at all about
relaying.**

### 🔍 One mechanism explains both rows

B0 parks its observer **79 m** from the HVT — nearly overhead, where the sightline
survives the target moving down a street — and *therefore* needs a 2.1-hop chain
to reach the MCV. The learned policies loiter at **291 m**, keep a 1.27-hop chain,
and hold the sightline 42 % of the time instead of 94 %.

**Going in close is only survivable if teammates relay behind you.** A drone that
closes to 79 m loses its own link to the MCV; the move pays off only when the
swarm has already differentiated into an observer and a chain. With a shared team
reward and no per-drone role signal, **no drone can afford to be the one that
goes in** — so none do, and the swarm settles at the stand-off distance where
every drone can do a mediocre version of both jobs.

That is a **coordination trap**, and it reframes the deficit: not observation
persistence, not chain-building, but **role emergence**, of which those two are
the symptoms.

## ✅ Built and measured — G0–G4

The gate is met. **MAPPO beats random on curriculum stage 1**, scored
deterministically through `evaluate.py` — the same harness B0's 57.2 % came from
— on 5 seeds, median [IQR]:

| stage-1 policy | mission-capable | observed | chain occluded | p5 capacity |
|---|---|---|---|---|
| random | 35.1 % [4.2] | 36.9 % [4.5] | 14.3 % [1.2] | 0.8 |
| **MAPPO, MLP, 4 M steps** | **74.8 % [12.7]** | 74.8 % [13.4] | 33.7 % [9.6] | 9.5 |
| B0 (for scale) | 87.5 % [0.8] | 87.5 % [0.8] | 29.0 % [1.2] | 14.5 |

**Five training seeds**, each scored on the same 128 held-out-of-tuning episodes,
median [IQR] across seeds — `eval_policy.py --group`. That flag exists because
the obvious reading of `--seeds` is the wrong one: it varies the *evaluation
episodes of one policy*, which says nothing about whether a second training run
would land anywhere near the first. Per seed: **75.4 / 60.0 / 78.3 / 62.6 /
74.8 %**.

> ⚠️ **Stage 1, train split, MPS.** Not comparable with anything in BLOCK_E.md or
> BLOCK_F.md: different stage, different route split, different device. The
> numbers this table exists to compare are the first two rows of it.

**Three things this table says, and two of them are warnings.**

1. **The gate holds on every seed.** The worst run, 60.0 %, is still 25 pp above
   random. That is the claim Block G's first gate makes and it survives.
2. ⚠️ **The seed spread is large — 12.7 pp IQR, 60–78 % range.** Two of five runs
   land ~15 pp below the other three. Something bimodal is happening in training
   and it is not yet diagnosed. Whatever tuning happens next should be judged on
   the *worst* seed, not the median, or it will be fitting the lucky ones.
3. ⚠️ **B0 is better here, and `chain_occluded` says why.** 87.5 % against
   74.8 %, with the learned policy's chosen chain crossing a building **33.7 %**
   of the time against B0's 29.0 %. B0 hill-climbs on the clearance feature
   explicitly; the learner has not discovered that. Two caveats on the
   comparison, neither of which rescues it: stage 1 is B0's best case (stationary
   target, 150 steps, exact cue — "fly at it and space out" is what B0 *is*), and
   `mission_capable` == `observed` for both, so at stage 1 the metric is the
   sensor and not the relay chain. **Stage 1 does not test the coordination
   problem MARL is supposed to earn its keep on.** B0 = 57.2 % on the full
   mission is still the gate that matters.

Regenerate:

```bash
for s in 0 1 2 3 4; do uv run python -m src.training.train --stage 1 \
    --num-envs 256 --env-steps 4000000 --device mps --seed $s --name g2-mlp-s$s; done
uv run python scripts/eval_policy.py runs/g2-mlp-s*/checkpoint.pt \
    --group "MAPPO mlp" --policy random b0 --stage 1 --num-envs 128 \
    --device mps --train-routes
```

### ☠️ The bug that made everything anti-learn, and how it was found

**Symptom.** Every configuration made the policy monotonically *worse*. Under a
reward whose only non-zero term was `mission_capable` — where the return IS the
metric — the policy fell from 30 % to 4.6 %. Learning rate, KL-adaptive
scheduling and rollout length changed only how fast it collapsed.

**Cause.** skrl's `GaussianMixin(clip_actions=True)` clamps the sampled action
to the action space and then evaluates its log-probability under the *unclamped*
Normal. Every tail draw is recorded as though it had landed exactly on ±1, so
empirical mass piles on the boundary that the density in the PPO ratio does not
account for. The policy is then pushed toward the corners, the action standard
deviation rises with **no entropy bonus anywhere in the config**, and every
reward term degrades together.

**Fix.** `clip_actions=False`. `core._advance_drones` already opens with
`actions.clamp(-1.0, 1.0)`, so the bound is enforced regardless and only the
density changes. Pinned by `src/models/test_actor.py`.

**What found it, and it is the reusable part.** Three probes, in order:

1. **A reward with a known optimum** — every objective term zeroed except
   `-w_effort·‖a‖²`, whose optimum is `a = 0`. PPO improved it monotonically
   (−0.91 → −0.50), which cleared the whole loop: wrapper, flattening,
   bootstrap, GAE, optimiser.
2. **One reward term at a time.** `mission`-only collapsed, `-‖a‖²` did not.
   The difference is that the first depends on the *state* and the second on the
   *action* — which pointed the search at the action distribution rather than at
   the task.
3. **The rising standard deviation.** With `entropy_loss_scale = 0` the only
   thing that can inflate σ is the policy gradient, so σ rising monotonically
   for millions of steps is a statement that the ratio is systematically wrong.

⚠️ **Instrument before you tune** is not advice — the per-term reward log
(`EnvConfig.training_extras`) and skrl's own loss/σ tracking are what made this
findable at all. The aggregate return said only "flat, then falling".

### ✅ G1a / G1b — measured on CUDA, and the budget assumption was wildly conservative

RTX 5090, torch 2.13.0+cu130, 2026-08-24. `scripts/cuda_session.sh` reproduces
the whole session.

| | measured | target |
|---|---|---|
| **G1b: 10 M steps end-to-end, learner attached** | **2.2 min (75,252 env-steps/s)** | **≤3 h** |
| G1a: env only, `num_envs = 256` | 37,083 env-steps/s | ≥1000 |
| rung spread F0–F4 | **1.09x** | no rung cheaper |

**The gate is met by 75x.** THESIS_PLAN §3 budgets 45 runs at ~2.8 h each ≈ 120
GPU-hours; the measured cost is **~2 GPU-hours for the entire matrix**. Compute
is not a constraint on this project and never was — which means the equal-budget
hyperparameter search `MODELS.md` requires is affordable, and so is any number of
seeds. ⚠️ Update THESIS_PLAN's budget rather than quoting 120 GPU-hours again.

**Utilisation was 33 % at 3 GiB of 32 GiB**, so `num_envs = 1024` leaves most of
the card idle. Two consequences: `num_envs` can rise a long way on *learning*
grounds, and several seeds fit concurrently on one GPU.

Block F's rung-independence result survives at CUDA scale: **1.09x** spread
across F0–F4 (1.06x on MPS), so no rung gets more samples per GPU-hour.

**The CUDA baseline reproduces the MPS one**, which is the useful part — the
diagnosis transfers to real hardware:

| | CUDA (3 seeds) | MPS (3 seeds) |
|---|---|---|
| GNN + floor | 38.1 % [3.5] | 37.6 % [1.7] |
| B0 | 57.5 % [1.4] | 58.0 % [2.2] |
| observer tenure, GNN vs B0 | 41.5 vs **270.3** | 35.6 vs 264.6 |

### ☠️ What the first CUDA run found — three bugs, one per never-executed test

**1. A host synchronisation on every step.** `core._advance_drones` built its
position limits with `torch.tensor([...], device=...)` from a Python list *inside
the step*, which copies host memory and stalls the pipeline every tick. This
violates `AGENTS.md`'s device rule and had been there since Block D. It was
caught by `test_step_never_syncs_to_the_host`, which is CUDA-gated and had
**never executed in the project's history** — the test worked the first time it
ran. Fixed by hoisting the two tensors into `__init__`; the golden trace still
passes exactly on arm64, so no number moved.

**2. The value preprocessor landed on the wrong device.** `mappo_cfg(device=None)`
let skrl resolve the scaler's device to the *global default*, which is `cuda` on
a GPU box even when the env and models are on CPU. `mappo_cfg`/`ppo_cfg` now
refuse `device=None` when scaling is on.

**3. `bench_env.py --breakdown` had been broken since Block F** — `_clearance`
returns `(true, channel)` now and the breakdown still passed the tuple to
`_capacity`. It aborted step 2 of the session, which is why only the 256-env row
of G1a exists. Re-run `--envs 1024 4096` to complete the table.

### Three decisions the spec did not settle

**1. G1 is split into G1a and G1b, because the spec's build order is circular.**
G1 asks for "wall-clock for a 10 M-step run end-to-end **including the learner**"
*before* building anything on top — but there is no learner until G2/G3, and
`bench_env.py` measures the env alone. **G1a** is Block D's pending env-only CUDA
re-run; **G1b** is the end-to-end wall-clock, which needs the trainer and runs in
the same GPU session. No CUDA is available yet, so both are queued.

Provisional, from the trainer on **MPS** at `num_envs = 256`, learner attached:
**20,519 env-steps/s**, i.e. **0.14 h per 10 M-step run**. That is a laptop lower
bound on the wrong device and settles nothing — but it is 20× the ≤3 h target,
so the budget risk G1 exists to expose is not currently visible.

**2. The swarm is one parameter-shared agent, not `N` skrl agents.**
`SwarmMultiAgentWrapper` keys tensors per drone, which is the natural reading of
skrl's API and is what the contract smoke tests use. It cannot be what a reported
run uses: five per-drone policies **cannot be evaluated at N = 8**, so RQ2's
zero-shot columns would not exist, and per-drone networks would assign roles by
identity, which is the opposite of the "roles emerge" claim.

Handing skrl the *same* `Model` under five agent ids is a trap — it builds five
Adam optimizers over the same parameters and runs five sequential PPO updates per
rollout, four of them against stale log-probabilities. `SharedPolicyWrapper`
collapses the drones into the batch dimension instead: one optimizer, one update,
correct ratios. This is MAPPO as published (Yu et al., 2022), not a weakening of
it — decentralized actors, one centralized critic on the shared state, parameters
shared across homogeneous agents.

**3. One message-passing layer, not two.** `MODELS.md` reasons from a graph of
`N` drone nodes and allows two layers. **The actor does not hold that graph**: its
observation is its own ego block plus 7 neighbour slots, i.e. a *star* centred on
itself. A second layer over the true swarm graph would give drone `i` access to
`j`'s aggregate of `k` — information `i` can only get by exchanging embeddings —
and would hand the GNN rung strictly more information than the MLP and DeepSets
rungs, confounding RQ2's contrast with an information difference. Depth goes into
the message and update MLPs instead, where it is capacity rather than reach.
MODELS.md's own diameter-1 argument already says one layer reaches everyone.

### Two additive env changes, both opt-in

The environment is frozen except for bugs, so both are behind
`EnvConfig.training_extras = False` and the default build is byte-identical to
the one `test_golden.py` pins — **no re-capture, and the contract test is
untouched**.

* **`extras["final_state"]` — a correctness fix, not instrumentation.** skrl
  bootstraps a truncation as `gamma * V(next_observations, next_states)`, and
  with `auto_reset=True` the tensors `step()` returns are already a fresh
  episode's opening. Without the pre-reset pair the learner values an unrelated
  state at every truncation — silently, at gamma = 0.997 on returns of order
  300, cancelling `time_limit_bootstrap=True` while leaving the flag looking
  correct. Block D's smoke test asserted the *flag*; nothing asserted the
  *state*. The critic's own state was being computed and thrown away.
* **`extras["reward/<term>"]`** — the six terms separately, per `REWARD.md`.

`BatchedSwarmEnv.set_stage_weights()` is the third addition and is not gated: it
moves the sampling distribution over `STAGES` and touches nothing else.
`EnvConfig` is frozen, so the curriculum needs a seam.

### G6 — `tau_c` and `tau_l` were measured, and both stay

The retune BLOCK_E deferred to here. `Phi_observe = sigmoid(clearance_best/tau_c)`
and `Phi_link = sigmoid((C_e2e - 15)/tau_l)`, so the question is whether either
sigmoid is saturated across the range the policy actually operates in. Measured
on B0 and on random over 600-step episodes on the eval split (CPU, 24 envs — a
tuning distribution, not a reported number):

| | `tau` | saturated | spread (std of the term) |
|---|---|---|---|
| `Phi_observe`, blocked regime (random) | **15 m** | 12.3 % | **0.265** |
| | 40 m | 8.5 % | 0.192 |
| | 80 m | 8.4 % | 0.164 |
| `Phi_link` (B0) | **6 Mbps** | 11.1 % | **0.324** |
| | 15 Mbps | 5.2 % | 0.201 |

**Both shipped values win, and widening either makes things worse** — a wider
sigmoid saturates less often but carries less signal per step, and the spread is
what the gradient sees. `tau_c = 15`, `tau_l = 6`: **frozen**, on measurement.

One structural fact this exposed, and it matters for the diagnosis below:
`occlusion` returns **1e4** for "nothing in the way", so `clearance_best` is 1e4
whenever *any* drone holds a clear ray and `Phi_observe` is then pinned at 1.0.
The observe potential is effectively binary at the top — it rewards *having* a
sightline and says nothing about having a better one.

### ⚠️ The full mission does not clear B0 yet — and the failure is specific

`b0` and `random` re-measured through `eval_policy.py` on **this device**, stage 4,
F4, eval split, 5 seeds — they reproduce Block E and Block F, which is the
harness cross-validating:

| | mission-capable | observed | hop mean |
|---|---|---|---|
| random | 10.6 % [1.9] | 21.4 % [0.6] | 0.4 |
| **B0 — the floor** | **58.5 % [5.9]** | 92.2 % [1.4] | 2.1 |
| first full-mission pilot (10 M) | 24.3 % | 35.4 % | 0.6 |

(Block E: B0 57.2 % on CPU. Block F: 56.0 % on MPS. This harness: 58.5 % on MPS.)

**The failure is not acquisition, and it is not navigation.** Measured over 600-step
eval episodes:

| | ever acquires | first acquisition | **observed after acquiring** | **nearest drone to HVT** | drone–MCV range |
|---|---|---|---|---|---|
| B0 | 100 % | step 18 | **94.5 %** | **79 m** | 568 m |
| pilot | 100 % | step 30 | **42.4 %** | **291 m** | 561 m |
| random | 100 % | step 30 | 26.6 % | 328 m | 764 m |

Every policy finds the target, which is what `DECISIONS.md`'s uncued-fan
measurement predicts. The learned policy also flies out to the right *radius* —
561 m from the MCV against B0's 568 m. What it never does is **close the last
200 m**: B0 parks a drone 79 m from the HVT (i.e. nearly overhead, at 40–80 m
altitude) and holds the sightline 94.5 % of the time; the pilot loiters at 291 m,
where a city sightline is intermittent, and holds 42.4 %.

The pull toward closing is weak by construction: `Phi_approach` normalises by
`d_ref_m` = 1500 m (the map diagonal), so 8 m of closing — one step of travel —
pays **0.013**, while `Phi_observe` is already saturated at 1.0 the moment any ray
is clear. `d_ref_m` also lives inside `Phi` and so is optimum-preserving; it is
now exposed as `--d-ref` alongside `--potential-scale` for exactly this test.

### The exploration floor — why the full mission stalls, mechanically

Two 20 M-step controls, both against the 10 M pilot:

| run | change | stage-4 plateau | final action std |
|---|---|---|---|
| pilot 1 | — (10 M) | 30–33 % | 0.180 |
| **c1** | **2x the steps** (20 M) | **34 %** | **0.061** |
| — | entropy_loss_scale 0.01 | abandoned | **1.11 and rising** |

**Doubling the budget bought nothing**, and the reason is visible in the last
column: with `entropy_loss_scale = 0` the Gaussian's standard deviation shrinks
monotonically as the policy grows confident. By 13 M steps it is 0.117 and by
20 M it is **0.061** — the policy is deterministic, has stopped exploring, and is
locked into whatever it found early. More steps at that point are more steps of
a policy that cannot change.

The obvious counter is an entropy bonus, and 0.01 fails in the opposite
direction: the deviation rose 0.64 → **1.11**, i.e. the entropy gradient
dominated the policy gradient and the actions became noise. That run was killed
rather than finished.

**The cleaner instrument is `min_log_std`**, now exposed as `--min-std`. It
floors the deviation by bounding the policy *class* rather than adding a term to
the objective, so it cannot trade reward for entropy the way a bonus does. skrl's
default (-20) is effectively no floor at all.

⚠️ **Do not read "20 M is enough" out of c1.** What c1 shows is that 20 M steps of
a policy whose exploration has collapsed is worth no more than 10 M. The budget
question is not settled until a run keeps exploring to the end.

### 🔍 The diagnosis: nobody commits to the observer role

The renderer now takes a checkpoint path as a policy
(`--policy runs/<name>/checkpoint.pt`), which is the one line `_make_policy` was
always going to need. Flying a learned policy and B0 down the **same route at the
same seed** is what turned an aggregate into a mechanism.

**Route 12, F4, N = 5** — B0 97.2 % capable, the learned policy 60.2 %. The
bottom panel of the figure is the finding: B0's observer is **drone 3 for 228 of
240 s**, one handoff. The learned policy's observer jumps 0 → 1 → 2 → 3 → 2 → 4,
with ~90 s of the episode where **nobody is observing at all**.

Measured across 5 seeds x 64 episodes on the train split (stage 4, F4):

| policy | capable | observed | handoffs / episode | **observer tenure** |
|---|---|---|---|---|
| **B0** | **58.0 % [2.2]** | **92.8 % [1.1]** | **1.1 [0.1]** | **264.6 steps [11.1]** |
| MLP + floor (d1) | 32.3 % [1.8] | 50.0 % [0.6] | 10.0 [0.7] | 27.6 [1.5] |
| MLP + floor + `d_ref` (d2) | 35.0 % [1.1] | 52.5 % [4.2] | 9.8 [0.4] | ~29 |
| **GNN + floor (d3)** | **37.1 % [2.1]** | **54.4 % [1.2]** | **8.4 [0.5]** | **35.3 [0.2]** |
| *GNN + floor, 3 training seeds* | *37.6 % [1.7]* | *56.0 % [2.3]* | *8.3 [0.6]* | *35.6 [1.1]* |
| random | 11.0 % [0.3] | 23.2 % [0.8] | 6.8 [0.7] | 17.5 [2.4] |

**Observer tenure — mean steps one drone holds the role — separates B0 from every
learned policy by ~8x.** B0 commits for 106 s at a stretch; the learned policies
manage 11–14 s. Note the learned policies hand over *more often than random
does*: random's lower count is an artefact of observing so rarely that there are
few handovers to make, which is why tenure is the honest statistic and raw
handoff count is not.

**The mechanism.** Every drone runs the same feedforward function of its own
current observation, and `hvt_rel` is **zeroed when the target is not seen**
(`docs/ENVIRONMENT.md` -> Observations). So nothing in the actor can represent
*"I am the observer, I hold station"*: whichever drone is nearest acquires, drifts
(the dynamics are a double integrator with no drag), loses line of sight, goes
blind, and wanders until another drone stumbles onto the target. B0's advantage
on this route is not better flying — it is **commitment**, which it gets from a
carried belief and an explicit role assignment.

**⚠️ A correction this forced.** The `hop_count` ~ 0.9 statistic (against B0's
2.1) was read earlier as a second, independent failure: "the swarm never builds a
chain". The rate panel says otherwise — the learned policy's e2e trace is
**binary, either >=60 Mbps or exactly 0**, and the zero stretches coincide with
the unobserved gaps. No observer means no source, hence no chain and
`hop_count = 0`. That is precisely the denominator effect
[`BLOCK_E.md`](BLOCK_E.md) §6 warned about. Conditioned on a chain existing the
learned policy runs 1–2 hops against B0's 2. **There is one primary failure —
observation persistence — and the chain statistics are largely its shadow.**

The encouraging half: when a chain exists it carries **>=60 Mbps against a 15 Mbps
requirement**, 4x the bar. The physics is nowhere near binding, and on route 12
`observed` = 62 % against `capable` = 60 %, so converting unobserved steps into
observed ones converts almost 1:1 into mission success.

**The GNN is ahead on every axis of this table** — capable, observed, fewest
handoffs, longest tenure — and reached d2's 11.5 M-step score by 6.9 M. That is
RQ2's claim appearing early, in the place the mechanism predicts it should: the
relational channel is where a drone can read its neighbours' `sees_hvt` and
`on_path` bits and decline to duplicate a role somebody already holds. It is one
seed and it does not yet clear B0, so it is a lead, not a result.

### ✅ Recurrence: it trains. The GRU was never the problem — skrl's `PPO_RNN` was

Superseded 2026-08-25. The previous version of this section concluded "the fault
is localised to sequence replay" and recommended `--seq-len 1` as the working
fallback and the bisection point. **Both were wrong**, and the error is the
instructive part: every probe had been aimed at the component under suspicion
(the GRU) and none at the *vehicle carrying it*.

**The bisection that settled it.** `PPO_RNN_Aligned` was run with **feedforward**
models — no GRU anywhere, so skrl leaves `_rnn = False` and the class degrades to
plain PPO on the same code path. Stage 1, seed 0, 4 M steps, `--min-std 0.2`:

| vehicle, identical MLP actor + MLP critic | peak | final |
|---|---|---|
| **MAPPO** — what every reported number used | **82.7 %** | **76.2 %** |
| **`PPO_RNN`** | 52.5 % | **3.8 %** |

The collapse reproduces with no recurrence in the run at all. Everything
attributed to the GRU for a week belonged to the agent class.

**The cause, and it is one line.** `skrl/agents/torch/ppo/ppo_rnn.py` in 2.1.0 is
an **un-migrated copy of an older PPO**: it ships its own private `compute_gae`
and its own private `record_transition`, and skrl's truncation rework landed in
`ppo.py` and `mappo.py` without ever being propagated to it.

```python
not_terminated = terminated.logical_not()  # ppo_rnn.py:45
not_done = (
    (terminated | truncated)
    if time_limit_bootstrap  # mappo.py:49
    else terminated
).logical_not()
```

GAE recurses backwards as
`A_i = r_i − V_i + γ · not_done_i · (V_{i+1} + λ·A_{i+1})`. At a truncation
`terminated` is False, so `not_terminated_i` is **True** and the recursion keeps
going — but `auto_reset` has already put a fresh episode at `i+1`. The truncation
step therefore receives `γ·(V_{i+1} + λ·A_{i+1})` where `V_{i+1}` is the **next
episode's opening value** and `A_{i+1}` is the **next episode's advantage**, on
top of the bootstrap already folded into `r_i`. The bootstrap is double-counted
*and* the next episode's advantage stream leaks backwards through the reset, then
propagates to every earlier step of the rollout with weight `(γλ)^k` = 0.947 per
step at γ = 0.997, λ = 0.95.

⚠️ **That is why it presented as a slow poison rather than a crash.** Only
rollouts spanning a reset are contaminated — ~21 % at stage 1 (32-step rollouts,
150-step episodes) — so the curve rises normally for ~600 k steps before
degrading. And because the envs reset in lockstep, when it hits it hits all 256
at once.

**A second, smaller divergence, fixed at the same time.** `PPO_RNN` bootstraps
with `V(observations, states)` — the state being *left* — where `MAPPO` uses
`V(next_observations, next_states)`. So `PPO_RNN` never reads `next_observations`
at all, and Block D's `final_observations()` / `final_states()` seam is dead code
on this path.

**Which fix did the work — isolated, because two fixes shipped together.** Same
vehicle, feedforward models, seed 0:

| | peak | final |
|---|---|---|
| neither fix | 52.5 % | 3.8 % |
| **A** — bootstrap off the next state | 68.9 % | 11.3 % |
| **B** — the GAE mask | 82.1 % | **69.7 %** |
| both (what ships) | 73.8 % | 68.5 % |

**B is the entire effect; A alone changes nothing.** A is kept because it is what
`MAPPO` does and it costs nothing, but ⚠️ **do not record A as the fix.** B alone
against both (69.7 vs 68.5) is one seed each and not distinguishable; A alone
against B alone is far outside any seed spread measured here.

> ☠️ **The root cause is the file, not the line.** Both this bug and the
> hidden-state aliasing bug below come from `ppo_rnn.py` being a stale fork.
> Treat anything else inherited from it as suspect, and diff against `mappo.py`
> before trusting it.

**✅ No reported number is invalidated.** The bug lives only in `PPO_RNN`.
BLOCK_G's tables, the 81-run sweep and every feedforward run went through
`MAPPO`, which is correct on both counts.

#### The gate, and what it does and does not show

Stage 1, 5 training seeds, scored through `evaluate.py` on the train split, MPS —
the same harness and split as the table at the top of this file:

| stage-1 policy | mission capable | observer tenure | handoffs |
|---|---|---|---|
| B0 | 87.5 % [0.8] | 101.8 [3.1] | 0.3 |
| MAPPO MLP feedforward | 74.8 % [12.7] | 50.3 [10.8] | 1.0 |
| **MAPPO MLP recurrent** | **76.7 % [13.1]** | **53.5 [2.5]** | 1.1 |

**The gate is met: recurrence reaches feedforward parity.** It is no longer
broken. That is all it shows.

⚠️ **This is a null, and it is the expected null.** +1.9 pp on a 13 pp IQR, tenure
+3.2 on a 10.8 IQR. Stage 1 has a stationary HVT and `mission_capable ==
observed`, so there is no history-dependence to exploit and memory *should* buy
nothing. **Do not read this table as evidence for or against recurrence.** The
one suggestive number is tenure's seed spread collapsing 10.8 → 2.5.

The feedforward row reproduces this file's own 74.8 % [12.7] exactly, which
cross-checks the harness — `--min-std 0.2` was inert at stage 1, because σ never
fell below 0.357 and the floor never bound.

#### ⛔ The recurrent-critic hypothesis: built, tested, not the mechanism

The previous "where to resume" note argued the critic was feedforward while the
policy was recurrent, so `V(s)` averaged over a hidden state it could not see and
every advantage was biased for exactly the history-dependent behaviour the GRU
exists to produce (Yu et al. 2022 make both recurrent). It was built —
`models.critic.SwarmCriticRNN`, no `architecture` argument, identical in all
three rungs — and measured against the feedforward critic on the *unfixed* path:

| stage 1, recurrent actor | peak | final |
|---|---|---|
| feedforward critic | 42.7 % | 2.9 % |
| recurrent critic | 54.9 % | 2.9 % |

It raised the peak and delayed the collapse by ~600 k steps. It did not stop it.
**A real improvement, not the mechanism** — and it is what motivated the
bisection that found the real one. It ships, because it is the published MAPPO
configuration and it is now the cheaper half of a question worth answering.

The second hypothesis in that note — `grad_norm_clip = 0.5` applied jointly to
policy and value parameters (`ppo_rnn.py:557`) — **is real and remains untested.**
It never needed to be invoked.

#### Still true from the earlier investigation

1. **The gradient is not inverted.** The known-optimum probe (`-w_effort·‖a‖²`,
   optimum `a = 0`) improves through the recurrent path. ⚠️ It also stayed
   positive throughout the collapse, which is *why* it did not localise this bug:
   a pure per-step action cost has almost no cross-episode structure, so the
   boundary leak costs it almost nothing. **The probe clears the loop; it does
   not clear the credit assignment.**
2. **The model is exact.** Sequence-mode replay reproduces step-mode collection
   to 1.19e-7 given the same hidden state, and the epoch-0 log-probability
   identity holds at `--seq-len` 1 and 4. Both are pinned by
   `models/test_recurrence.py` and `training/test_recurrent.py`.
3. **☠️ The hidden-state aliasing bug is real and stays fixed.**
   `PPO_RNN.record_transition` ends `self._rnn_initial_states =
   self._rnn_final_states`, binding both names to the same dict, so from the next
   step on the transition records a state one step ahead of the one that produced
   its action. Fixed by `PPO_RNN_Aligned`, pinned directly. It was never the
   cause of the collapse, and it is still worth having fixed.
4. **It was never a train/deploy context mismatch**, and `--seq-len 1` was never
   a diagnosis — that run was simply at a different point on the same doomed
   curve.

#### What is pinned now

`models/recurrence.py` holds one GRU driver shared by actor and critic, so the
sequence logic exists once. Tests cover: sequence replay reproduces step-mode
collection; the episode-boundary state zeroing; the gradient surviving the
boundary split; the critic's own epoch-0 value identity; the critic being
identical across all three rungs (⛔ MODELS.md); and **skrl's `Memory` ordering
sequence rows env-major and time-contiguous**, which is the layout `view(-1, L)`
silently assumes and which nothing had asserted.

Two latent bugs were found while factoring it out: the boundary zeroing was an
in-place write into a tensor autograd saves (now `masked_fill`), and
`terminated=None, truncated=<tensor>` raised. Neither had fired.

#### ⚠️ Open: is recurrence actually worth keeping?

**Undecided, and stage 1 cannot decide it.** The mechanism argument is a stage-4
one: B0 holds the observer 264.6 steps, the sweep's best feedforward policy 47.4,
and the 81-run grid moved that by 12 steps out of a 218-step deficit. A stateless
function cannot represent "I am the observer and I am holding station"; that
argument is untouched by anything measured here.

The decision experiment, ~1 GPU-hour, **with the rule declared before it runs**:

> Full mission (stage 4, F4, curriculum), `deep` cadence, recurrent vs
> feedforward, 5 seeds each, train split. Primary metric **observer tenure**,
> secondary `mission_capable`.
> * tenure ≥ ~95 (2× feedforward) **and** capable ≥ 45.1 % → keep, and the matrix
>   runs recurrent
> * tenure moves < 20 % **and** capable within IQR → drop it, record the negative
>   result, and the tenure deficit needs a different attack

Recurrence is a *shared component*, not a per-rung hyperparameter — the GRU is
identical in all three rungs and the critic has no `architecture` argument — so
deciding it once on one architecture and applying it uniformly satisfies
MODELS.md rule 2. Say so in the methodology.

Cost, if kept: ~1.6× wall-clock (13.2 k against 20.6 k env-steps/s on MPS).
Compute is not a constraint (G1b: ~2 GPU-hours for the entire 45-run matrix).

### 📏 G8 — Gate 1: recurrence and `w_hold`, crossed. Both fail.

The 2×2 `BLOCK_G_PLAN.md` §4 declared, run 2026-08-25. GNN, `deep` cadence,
`shipped` shaping, F4, stage 4, **train** split (a tuning decision, so not eval),
CUDA, 5 seeds. Rules were fixed before the runs.

| cell | capable | IQR | worst seed | tenure |
|---|---|---|---|---|
| ff + shipped | 40.7 % | 3.2 | 29.6 | 43.2 |
| ff + `w_hold` | 40.8 % | 5.9 | 35.6 | 39.0 |
| rnn + shipped | 35.3 % | 5.0 | 30.8 | 36.8 |
| **rnn + `w_hold`** | **41.4 %** | 2.5 | 27.3 | 37.1 |
| B0 (train split) | 59.6 % | 2.0 | 57.4 | 272.7 |

**⛔ Recurrence — dropped.** Keep required tenure ≥ 95 and capable ≥ 45.1 %;
measured **36.8** and **39.7**. The drop rule is met outright: tenure moves
**−9.3 %** (rule: < 20 %) and capable **−1.1 pp** on a 4.7 IQR (rule: within IQR).
Pooled over 10 runs per level, recurrence is **−1.05 pp** and *widens* the seed
IQR 4.7 → 6.9.

⚠️ **This is not the same result as "recurrence does not train".** It trains fine
now and reaches feedforward parity at stage 1. It simply does not help on the full
mission. The `PPO_RNN` fix was still worth having — it removed a real bug and a
false diagnosis — but the hypothesis it unblocked did not pay.

**⛔ `w_hold` — null.** Keep required +≥3 pp on the median **and** the worst seed
improving. Under ff: median **+0.08** ✗, worst **+6.0** ✓. Under rnn: median
**+6.1** ✓, worst **−3.5** ✗. Passing a different half in each arm is what noise
looks like; the pooled main effect is **+1.65 pp** on a 6.8 IQR. Ships at
`w_hold = 0`.

⚠️ **The sweep's 45.1 % headline does not reproduce.** `shipped` at 5 seeds gives
**40.7 %**, matching stage A's own `gnn/deep/shipped` cell (42.9 % at 3 seeds).
The 45.1 % was `dref400_k30` at 3 seeds — the shaping axis is noise, so that cell
was the winner's curse, exactly as the stage-A analysis predicted. **Block G's
real best-known number was ~41 %, not 45 %.**

📏 **The bimodality got worse.** Per-seed `episode_return` runs
4.8 / 83.8 / 90.5 / 96.8 / 108.9 in one cell and −17.8 / 75.0 / 87.5 / 95.2 / 117.1
in another — roughly one catastrophic seed in five, not a smooth spread. Five
seeds is barely enough to place a median. Diagnosing this is now high priority.

### 📏 G9 / Gate 2 — the credit channel opened, the behaviour did not move

The intervention `scripts/probe_credit.py` motivated. GNN, `deep`, F4, stage 4,
train split, 5 seeds, 12 M steps; control is `g8-ff-shipped`, same cadence and
shaping, so `--w-relay` is the only variable.

| arm | capable | worst seed | observed | **hop \| obs** | tenure |
|---|---|---|---|---|---|
| control | 40.7 % [3.2] | 29.6 | 64.9 % | **1.91** | 43.2 |
| `w_relay 0.2` | 42.4 % [1.9] | 25.2 | 65.8 % | **1.93** | 44.5 |
| `w_relay 0.5` | 39.7 % [1.8] | 34.3 | 61.5 % | **1.88** | 39.7 |
| `w_relay 0.5` + agent-specific critic | 34.1 % [5.9] | **0.3** | 55.9 % | **1.89** | 37.6 |
| B0 | 57.3 % [3.9] | | 92.8 % | **2.26** | 294.7 |

⛔ **All three arms fail.** The rule was `hop|obs ≥ 2.0` and `capable ≥ 40.7 %`;
`hop|obs` measured **1.88–1.93 against the control's 1.91**. The per-drone
advantage signal was raised **71×** (0.00041 → 0.02931) and the behaviour did not
move at all.

Two secondary readings. **More `w_relay` is worse** (42.4 → 39.7), so the term is
degrading the objective's learning signal before it changes the policy. And the
**agent-specific critic actively hurt** — 39.7 → 34.1 with one seed collapsing to
0.3 % and a return of −278. ⛔ Drop it: `probe_credit.py` already said it was
variance reduction rather than credit, and it bought instability instead.

✅ **The kill condition did not fire.** `standoff_gap_m` is 322–366 m, so the
swarm did **not** cluster onto the HVT. `REWARD.md`'s objection to per-drone
potentials is not what happened here — this is a clean null, not the predicted
failure.

### ☠️ `hop | observed` was the wrong primary readout, and this is why

Every configuration this block has measured, on the same statistic:

| | hop \| obs | | hop \| obs |
|---|---|---|---|
| random | **1.83** | + `w_hold` | 1.87 |
| MLP | 1.86 | + `w_relay 0.2` | 1.93 |
| DeepSets | 1.88 | + `w_relay 0.5` | 1.88 |
| GNN control | 1.91 | + `w_relay` + asc | 1.89 |
| + recurrence | 1.89 | **B0** | **2.26** |

**Six interventions, a 1.86–1.93 range, and random sits at 1.83.** A statistic
that invariant is not measuring the thing it was chosen to measure.

⚠️ **A chain's hop count is set by where the OBSERVER stands, not by whether
anyone takes a relay role.** Against the measured `R` = 524 m, with the HVT ~1 km
out late in an episode:

* B0's observer sits 79 m from the HVT, so **~920 m from the MCV** → ~2–3 hops.
* The learned observer sits 291 m from the HVT, so **~710 m from the MCV** → ~2 hops.

The learned swarm builds exactly the chain its observer position requires. There
is **no separate relay-role failure to fix** — the hop deficit is a *shadow* of
the stand-off deficit, and `w_relay` was aimed at a role that was never missing.

**So the diagnosis collapses to one failure, not two: the observer does not
close.** `observed` (66.5 % vs 92.8 %), tenure (47 vs 295) and hop count
(1.91 vs 2.26) are three views of it.

### 🔍 Why closing is a coordination trap, and why no single-agent gradient escapes it

To hold a sightline from 79 m the observer must be ~920 m from the MCV, which is
beyond its own link range — so closing **only pays if the rest of the swarm has
already extended the chain outward to meet it**. But a relay drone gains nothing
by moving out until the observer is out: its `on_path` bit is already satisfied
by the shorter chain it is currently carrying.

**Every unilateral deviation is worse than the joint move.** That is a local
optimum requiring *coordinated* exploration, and it explains the whole pattern of
nulls in this block: recurrence, `w_hold`, `w_relay` and the agent-specific
critic are all instruments that change one agent's incentive or capacity, and
none of them can move a formation that has to move together.

⚠️ It also predicts **where** the failure lives: the chain only needs to extend as
the HVT drives out, so the swarm should be fine early in an episode and fail
late. `MODELS.md` already records `capable` decaying 84 % → 35 % across an
episode and asks for the second half to be reported separately — **and nothing in
this block has ever reported it.** That measurement gap is the next thing to
close, before any fifth intervention.

### 📏 G10 — the behavioural profile, and three corrections it forces

Free measurement: the **existing** g8/g9 checkpoints re-scored with
`observer_range_m` and the late-episode split. No training. Stage 4, F4, train
split, CUDA, 5 seeds.

| policy | capable | last third | ratio | observer range | late | tenure | role entropy | handoffs |
|---|---|---|---|---|---|---|---|---|
| **B0** | 59.6 % | 50 % | **0.84** | **88.8 [1.2]** | **79.8** | 272.7 | **0.10** | 1.0 |
| GNN control | 40.7 % | 40 % | **0.98** | **184.0 [20.7]** | 169.2 | 43.2 | **0.50** | 8.0 |
| + `w_relay 0.2` | 42.4 % | 40 % | 0.94 | 189.5 | 181.6 | 44.5 | 0.50 | 8.0 |
| + `w_relay 0.5` | 39.7 % | 30 % | 0.76 | 208.3 | 213.1 | 39.7 | 0.50 | 8.7 |
| random | 11.1 % | 10 % | 0.90 | 327.0 | 324.7 | 16.8 | 0.60 | 7.1 |

**1. ⛔ "The swarm fails late" is falsified.** Last-third over overall: **B0 0.84**
— *it* degrades most — against the learned policies' **0.94–0.98** and random's
0.90. The learned deficit is **uniform across the episode**, not concentrated at
the hard end. The coordination-trap prediction in the section above made exactly
the opposite call and is wrong.

**2. ⚠️ The 291 m stand-off figure was stale.** It came from the *first
full-mission pilot* and has been quoted throughout this block as though current.
The GNN policies sit at **184 m**, B0 at **88.8 m** — a **2.1×** gap, not 3.7×.
Every place 291 m appears above should be read as historical.

**3. 📏 And Block B already measured why 184 m fails.** Along-street sightline
median **127 m** (p90 387); across-street envelope at 80 m median **43 m**.
**B0's 89 m sits inside the median sightline and the learned policy's 184 m sits
outside it** — so the learned observer's view is intermittent *by construction of
the city*, not by bad flying. That is the whole `observed` gap (64.9 % vs 91.6 %)
in one line, and it is a **geometry threshold**, not a smooth cost.

### 🔍 The finding: role differentiation emerges at stage 1 and collapses at stage 4

| | role entropy | handoffs | tenure |
|---|---|---|---|
| B0 | **0.10** | 1.0 | 272.7 |
| GNN, **stage 4** | **0.50** | 8.0 | 43.2 |
| random | 0.60 | 7.1 | 16.8 |
| *GNN, **stage 1*** | ***0.20*** | *0.9* | *51.4* |

**The same architecture measures 0.20 at stage 1 and 0.50 at stage 4** — against
random's 0.60. On every *role* statistic the full-mission policy is close to
random; on every *performance* statistic it is far above it (40.7 % vs 11.1 %).

**The swarm learns to fly at the target and does not learn to organise.** And it
*can* organise — it does at stage 1, where the HVT is stationary. What breaks is
holding a role while the target moves, which is the one thing B0 gets from an
explicit belief plus hysteresis (a seeing drone outranks a non-seeing one, so the
incumbent keeps the role through a momentary occlusion).

⚠️ **This re-opens recurrence, on a technicality worth taking seriously.**
Recurrence was dropped on `hop | observed` — since shown to measure geometry —
and on tenure. `role_entropy` did not exist then, so **the recurrent runs have
never been scored on the statistic that now defines the failure.** The
checkpoints exist; re-scoring them is free and must happen before recurrence
stays dropped.

### 🔍 G11 — what route 12 actually looks like, and it is not what the aggregates said

☠️ **First, a correction to how this was ever diagnosed.** `render_episode.py`'s
`--compare` discarded `--policy`, so the command this file recommends for
"turning an aggregate into a mechanism" had been rendering the five scripted
baselines and never the checkpoint. Fixed; `src/viz/test_render.py` (previously an
empty file) pins it.

**Route 12, F4, N = 5, seed 0 — same route, same seed, side by side:**

| | capable | observed | hops | chain occluded | handoffs | e2e p5 |
|---|---|---|---|---|---|---|
| B0 | 97.2 % | 97.2 % | med 2, max 3 | 0.2 % | **1** | 26.9 |
| `waypoint` (oracle-fed) | 40.5 % | 41.8 % | med 2 | 1.3 % | 2 | 0.0 |
| **GNN, g8-ff-shipped-s0** | **80.8 %** | **82.5 %** | **med 2, max 3** | **2.2 %** | **9** | **0.0** |
| random | 13.2 % | 35.5 % | med 1 | 24.3 % | 0 | 0.0 |

⚠️ **The learned policy scores 80.8 % here against a 40.7 % aggregate**, with
B0-like chain structure — median 2 hops, max 3, 2.2 % occluded. So the flat
statement "the swarm never builds a chain" is **wrong**: on this route it builds
exactly the right one. ⚠️ Route 12 is B0's *strong* route (97.2 %), so read this
as "route difficulty varies a lot", not as proof of bimodality — B0's own
`capable_share_high` is only 25 %.

**And the figure shows something no aggregate in this block could.**

📏 **B0's drone tracks are confined to the HVT corridor.** The learned policy's
**sprawl across the entire map** — large loops out past x = −700 m, into the
western half the HVT never enters. Five drones, and several are executing long
excursions far from any part of the mission.

The observer panel says the same in a different way: B0's is a flat purple line
(drone 3, from t = 12 s to the end, one handoff). The learned policy's is mostly
drone 2 — it *does* commit on this route — but broken nine times, and the grey
"can see" bars show long stretches where only one drone has the target at all.

**The hypothesis this suggests, and it is new.** The non-observer drones are not
holding station. `core` integrates a double integrator **with no drag**, so
"output nothing" does not mean "stop" — holding position is an active control
problem the policy has not solved. Idle drones therefore drift into long loops,
and a drone 1 km off-axis cannot be recruited when the chain needs to extend.
That would explain why the *aggregate* chain looks random while a good route
looks like B0.

**Measured immediately, as `off_axis_m`** — mean distance from a drone to the
MCV–HVT segment, i.e. *is the swarm where the mission is*. Stage 4 reference
(MPS, 2 seeds — a reference, not a reported number):

| | off-axis | capable > 80 % | capable < 20 % |
|---|---|---|---|
| B0 | **104.4 m [1.1]** | 25.0 % | 9.4 % |
| random | **396.8 m [8.0]** | 0.0 % | 87.5 % |

⚠️ The learned policies' `off_axis_m` is **not yet measured** — it is one free
re-score of the existing checkpoints, and it is the direct test of the hypothesis
above. If it sits near random's 397 m rather than B0's 104 m, station-keeping is
the deficit and it is a *control* problem, not a coordination or credit one.

### 📏 G5 / stage B — RQ2's first eval-split answer

Each architecture at **its own** equal-budget winner, eval split, 5 seeds:

| arch | winner | capable | tenure | hops | train → eval |
|---|---|---|---|---|---|
| MLP | `base`/`dref400` | 31.2 % [1.2] | 34.1 | 1.00 | 35.6 → 31.2 (−4.4) |
| DeepSets | `deep`/`shipped` | 38.1 % [1.0] | 41.6 | 1.23 | 42.5 → 38.1 (−4.4) |
| GNN | `deep`/`dref400_k30` | 41.2 % [3.8] | 47.2 | 1.27 | 45.1 → 41.2 (−3.9) |

**The winner's curse is uniform — −3.9 / −4.4 / −4.4 pp.** The absolute numbers
were optimistic by ~4 pp, but the bias applies equally, so the *ordering* is
trustworthy. That is the equal-budget protocol doing its job.

**Tenure tracks capable exactly** (34.1 / 41.6 / 47.2 against 31.2 / 38.1 / 41.2),
so RQ2's effect runs through the mechanism the diagnosis names — a stronger
result than a bare score ordering.

⚠️ **But DeepSets → GNN is confounded with shaping.** Stage B compares best cells
and the shapings differ. Hold cadence *and* shaping fixed:

| `deep` / `shipped`, train, 3 seeds | capable |
|---|---|
| GNN | 42.9 % [2.1] |
| DeepSets | 42.5 % [4.0] |

**+0.4 pp — a null**, exactly as `MODELS.md` predicts for the in-distribution
rung at N = 5. Report **MLP → DeepSets (+6.9 pp, robust)** as the finding and the
relational rung as a null at N = 5. RQ2's informative test is **N = 8 zero-shot**
and it is still untouched.

### ☠️ `chain_occluded` confounds with hop count — RQ1 cannot use it as it stands

📏 Measured on the eval split: B0 **61.5 %** against every learned policy's
**34–40 %** — the *opposite* ordering to the stage-1 table above, and
`corr(hop_mean, chain_occluded) = 0.963` across the five policies. More hops means
more edges means more chances one crosses a building.

`THESIS_PLAN.md` designates `chain_occluded` as **RQ1's failure-attribution
metric**, and F4's rate division changes chain length — so comparing it across
fidelity rungs would compare hop counts wearing an occlusion label. ⛔ **Fix this
before RQ1 uses it**: report the *per-edge* occlusion rate, which is
hop-count-invariant, and keep the per-chain figure only as a descriptive
statistic.

### ⚠️ Two claims in this file were wrong and are corrected here

1. **"The learned policy converts observation into capability better than B0
   (0.67 vs 0.62)."** Same-device, same-split: **0.620 against 0.617** — they are
   identical. The earlier figure mixed provenance.
2. **"The chain statistics are largely the shadow of the observation failure."**
   Partly. Conditioned on observing, learned policies run 1.86–1.91 hops against
   **random's 1.83** and B0's 2.26 — so the chain deficit survives conditioning
   and is a failure in its own right.

### ⚠️ "The curriculum is hurting" — proposed on one seed, refuted on three

Recorded because the *error* is the useful part, and because somebody looking at
a single no-curriculum run will propose this again.

A GNN trained on **stage 4 only** (no curriculum, same 12 M budget) scored
**42.7 %** against the curriculum-trained GNN's 37.1 % — better on every axis
including observer tenure (51 vs 35). The mechanism looked obvious: with
boundaries at 0.10/0.20/0.35 and a 20 % mix, roughly **half** the budget goes to
stages other than the one being measured, and each transition shows a visible
drop with incomplete recovery.

**Three seeds per condition say the opposite:**

| GNN, 12 M | per seed | median [IQR] |
|---|---|---|
| **with** curriculum | 35.2 / 37.6 / 38.6 | **37.6 % [1.7]** |
| without curriculum | 27.8 / 33.2 / **43.1** | **33.2 % [7.7]** |

**The curriculum is worth +4.4 pp on the median and cuts the seed spread 4.5x.**
The 42.7 % was the best seed of a high-variance condition, run first. The
curriculum's value here is as much in *variance reduction* as in mean
performance — which is exactly what `docs/ENVIRONMENT.md` claims for it, and it
survives.

Two things to carry:

1. **≥5 seeds is not bureaucracy.** Three were enough to flip the sign of this
   result. Every tuning decision in this block must be judged on the seed
   *distribution*, and preferably on its worst member.
2. **Dropping the curriculum would not have endangered RQ1**, which is worth
   knowing for its own sake: BLOCK_G requires the schedule to be *identical
   across fidelity conditions*, not non-trivial. "Stage 4 from step 0" satisfies
   that as well as any schedule. The reason to keep the curriculum is that it
   works, not that RQ1 needs it.

### The equal-budget sweep — `scripts/sweep.py`

`MODELS.md` rule 2 has been owed since the block opened: *equal hyperparameter
budget across all three architectures, and say so in the methodology*. BLOCK_G
decision 3 gives it an operational definition — same search space, same number of
trials, same selection rule, per architecture — and this executes and records it.

```bash
uv run python scripts/sweep.py --device cuda            # stage A, 81 runs
uv run python scripts/sweep.py --device cuda --stage-b  # winners, 5 seeds, eval split
uv run python scripts/sweep.py --report-only            # the tables, any time
```

Results append to **`results/sweep_summary.jsonl`**, which is tracked rather than
living under the gitignored `runs/`: the summary is a *result*, the checkpoints
are regenerable. One 30 KB file carries every number the sweep produced, so the
pod that ran it pushes and the laptop that analyses it pulls — no scp, and the
provenance is versioned with the code that made it.

**Two stages, and the eval split is touched exactly once.** Stage A runs the full
grid on the **train** split at 3 seeds per cell and selects on median
`mission_capable` scored through `evaluate.py`, ties broken by smaller IQR —
declared before any result was seen. Stage B re-runs each architecture's winner
at 5 seeds and reports on the eval split.

**The grid: 3 architectures x 3 cadences x 3 shapings x 3 seeds = 81 runs.** At
the measured CUDA throughput that is roughly an hour.

| cadence | num_envs | rollouts | mini_batches | grad steps / M env-steps |
|---|---|---|---|---|
| `base` | 1024 | 32 | 4 | 488 |
| `wide` | **4096** | 32 | **16** | 488 |
| `deep` | **4096** | **64** | **32** | 488 |

**Batch size and update cadence cannot be swept independently** — raising
`num_envs` at fixed `rollouts` divides the optimizer steps by the same factor.
Each preset therefore holds gradient density *constant at 488* and varies what
the extra samples buy: `wide` spends them on a bigger batch (less gradient noise,
aimed at the seed spread), `deep` on a longer GAE horizon (γ = 0.997 has an
effective horizon of 333 steps and the rollout currently sees 32). The first CUDA
session is what makes this free: `ms/call` is **flat from 256 to 4096
environments**, so a 4x batch costs nothing.

Shapings are `shipped`, `d_ref=400`, and `d_ref=400 + potential_scale=30` — the
only reward knobs the design permits, all inside `Phi` and so optimum-preserving
by the PBRS proof.

⛔ **Not swept, deliberately:** the learning rate (fixed at 3e-4 so cadence is not
confounded with it — if a winner sits at a boundary, sweep it separately and say
so), the curriculum schedule (measured: the shipped one wins), and **fidelity**,
which is RQ1's independent variable and never a tuning axis.

⚠️ **The search runs per architecture.** Tuning on the GNN and applying its winner
to the other two is exactly the unequal budget the rule forbids, and it is the
tempting shortcut because the GNN is currently ahead.

#### Stage A: measured, 81 runs, RTX 5090

Median [IQR] `mission_capable` across 3 seeds, stage 4, F4, train split, 12 M
steps. Reference on the same harness and split: **B0 = 57.5 % [1.4]**.

| arch | winner | per seed | median [IQR] | tenure |
|---|---|---|---|---|
| MLP | `base` / `dref400` | 36.1 / 34.8 / 35.6 | **35.6 % [0.6]** | 34 |
| DeepSets | `deep` / `shipped` | 38.4 / 46.4 / 42.5 | **42.5 % [4.0]** | 42 |
| **GNN** | `deep` / `dref400_k30` | 45.1 / 45.3 / 39.3 | **45.1 % [3.0]** | 47 |

**1. `deep` won, `wide` failed — and `wide` never tested its own hypothesis.**
Pooled within-cell seed *range*, the quantity `wide` existed to shrink:

| cadence | median within-cell range | median of cell-medians |
|---|---|---|
| `base` | 5.0 pp | 36.1 % |
| `deep` | 6.1 pp | **42.2 %** |
| `wide` | **21.6 pp** | 22.5 % |

☠️ **`wide` did the opposite of its design intent: 4× the seed spread, −14 pp.**
The reason is arithmetic. Holding gradient density at 488 forces `mini_batches`
up by the same factor as `num_envs`, so **the minibatch the optimizer sees is
40,960 rows in all three cadences** — gradient noise per step is unchanged and
the preset's stated rationale ("a bigger batch, less gradient noise, aimed
straight at the seed spread") is not what it varies. What it actually varies is
*staleness*: 16 / 64 / 128 sequential grad steps on one collected batch, over
366 / 92 / 46 update rounds.

`deep` is staler still and wins anyway, so the winning axis is the one thing
left: **rollout length 32 → 64**, against γ = 0.997's 333-step effective horizon.
That was the axis the script claimed had "a real reason behind it"; it does.

⚠️ **But `deep` confounds `num_envs` with `rollouts`** — see "What is still open".

**2. MLP → DeepSets is a large clean effect; DeepSets → GNN is a null.**
Like-for-like on `deep`, 9 runs per architecture:

| arch | min | med | max |
|---|---|---|---|
| mlp | 22.7 | **26.1** | 32.3 |
| deepsets | 38.4 | **42.2** | 46.4 |
| gnn | 35.6 | **43.4** | 48.1 |

Permutation invariance is worth **+16 pp with zero range overlap**. The
relational rung — RQ2's actual claim — is **+1.2 pp with fully overlapping
distributions**, which is the in-distribution null `MODELS.md` predicts. ⚠️ Report
the *best-cell* contrast (35.6 / 42.5 / 45.1) rather than the deep-only one: the
MLP is the only rung that prefers `base`, so the 16 pp figure carries an
arch × cadence interaction. RQ2's informative column is N = 8 zero-shot and this
sweep does not touch it.

**3. Shaping is a null, and the winners are selecting on it.** On `deep`, pooled:
`shipped` 39.4, `dref400` 38.8, `dref400_k30` 40.0 — a 1.2 pp spread against 6 pp
of within-cell seed range, and the three per-architecture winners land on three
*different* shapings, which is the signature of ranking noise. ⛔ **The shaping
label on each winner carries no information**, and this file's earlier
"`d_ref_m = 400` measured +3 pp on MPS" does not reproduce at 81 runs on CUDA.

**4. The tuning did not touch the deficit.** Best-cell observer tenure is **47.4**
against B0's **264.6**; the best single run of all 81 is 54.8. The entire grid
bought ~12 steps out of a 218-step gap, while `corr(capable, tenure) = 0.875` and
`corr(capable, observed) = 0.966` across all 81 runs. The best cell also converts
observation into capability *better* than B0 (0.67 against 0.62). **The remaining
gap is observation persistence, not chain-building and not tuning.**

### What is still open

Ordered by what blocks the thesis, not by build order. The *sequenced* version --
commands, decision gates and the 2026-12-31 stopping rule -- is
[`BLOCK_G_PLAN.md`](BLOCK_G_PLAN.md).

* **⛔ THE GATE: the full mission does not clear B0.** Best measured **45.1 %
  [3.0]** (GNN, `deep`, `dref400_k30`, sweep stage A) against B0's **57.5 %** on
  the train split. Everything else on this list is secondary to closing that
  12 pp. The diagnosis — observer tenure, 47.4 against 264.6 — is above.
* **Is recurrence worth keeping?** It trains now and reaches feedforward parity
  at stage 1, which is a null by construction. The decision experiment and its
  pre-declared rule are in the recurrence section. ~1 GPU-hour.
* **Sweep stage B** — the three winners at 5 seeds on the **eval** split, closing
  the equal-budget claim `MODELS.md` rule 2 has owed since the block opened.
  Decided 2026-08-25: run as declared, next GPU session. ~1 GPU-hour.
* **G7** — Block F's open question: do F0/F2/F3-trained policies separate under
  F4 on hop count, `chain_occluded` or p5 capacity? One pilot per rung, and it
  de-risks RQ1's *attribution*, which is the primary research question. Cheap
  enough that there is no reason to defer it to April 2027.
* **`Φ_observe`'s hold factor — built 2026-08-25, `w_hold = 0`, unmeasured.**
  The flat-success problem is written up in [`REWARD.md`](REWARD.md): while the
  swarm is succeeding *every* reward term is flat, so nothing distinguishes
  holding a sightline from drifting out of it. 📏 The sweep already refuted the
  earlier reading ("the pull to close is too weak") — `d_ref 400` and
  `potential_scale 30` were both nulls, because a zero gradient cannot be fixed
  by multiplying it. `--w-hold 0.4 --d-hold 400` is the arm to run; shipped `Φ`
  is the control. Closing 291 m → 79 m is worth **0.000** shipped and **0.74** at
  `w_hold = 0.4`.
* 🔧 **`Φ`'s component weights `w_a` / `w_o` / `w_l` have never been moved.**
  `REWARD.md` used to read as though they were locked; they are inside `Φ` and
  as free as `k` (corrected 2026-08-25). Given the deficit is observation
  persistence, `w_observe` against `w_link` is an obvious untried lever.
* **`Φ_link` has the same disease and is untouched.** A formed chain carries
  ~60 Mbps against a 15 Mbps bar, so `sigmoid((C−15)/6)` reads 0.999 whenever a
  chain exists — while 📏 `chain_occluded` runs 33–41 %. The analogous fix grades
  it by chain clearance. Left alone deliberately so the first experiment has one
  variable.
* **The cadence grid confounds two axes.** `deep` changes `num_envs` (1024 →
  4096) *and* `rollouts` (32 → 64) together, and `wide` shows `num_envs` alone is
  harmful. The isolating cell — `1024 × 64 × 8 mini-batches`, same 40,960
  minibatch, same 488 grad steps/M — is missing. 9 runs, and it must be labelled
  a follow-up rather than folded into the equal-budget claim.
* **G5** — the three architectures are built, parameter-matched to **2.3 %** and
  tested. Compared on the sweep at N = 5 only; RQ2's informative column is the
  **N = 8 zero-shot** one and it is untouched.
* **The curriculum schedule is provisional.** `(0.15, 0.35, 0.60)` with a 20 %
  mix is a starting point, not a measured schedule. Find it, then freeze it, then
  record the freeze in `DECISIONS.md`.
* **The seed spread is undiagnosed.** 60–78 % over five runs at stage 1, and the
  sweep did not shrink it: `wide` was built to and made it **4× worse**. Judge
  every tuning decision on the worst seed.
* **`grad_norm_clip` is applied to policy and value parameters jointly**
  (`ppo_rnn.py:557`, and `MAPPO` does the same). Real, plausible under a GRU,
  never tested.
* **G1a is incomplete** — `bench_env.py --envs 1024 4096` still owed; the first
  CUDA session aborted on a `--breakdown` bug and only the 256-env row exists.

---

## G13 — the `Φ` audit, and the rebuild (2026-08-27)

Five interventions had now been tried against the 16.1 pp gap and all five were
nulls. This one started from the other end: instead of proposing a sixth
mechanism, **measure what the potential is worth in the states the swarm actually
occupies.** `scripts/measure_potential.py` banks those states off a real rollout
and scores any `RewardWeights` over them, so a candidate is judged before a run.

### 📏 What the audit found — `Φ` is off, and in two different ways

**1. No gradient along the closing axis.** Sweep the observer 250 m → 60 m with
the ray clear and a chain carrying 25 Mbps — the decision "close to B0's 89 m or
stand off at 184 m", with everything else held where the policy already has it:

| | swing over the band | per 8 m step | vs the energy term's 0.0544 |
|---|---|---|---|
| shipped `Φ` | **+0.320** | 0.0133 | **0.25×** |
| `Φ v2` | +1.717 | 0.0774 | **1.42×** |

**2. `Φ` is exactly constant in four drones out of five.** Every shipped
component is a hard `min` (`nearest_dist_m`), a hard `max` (`best_clearance_m`)
or the router's chosen path (`e2e_capacity_mbps`). A drone that is not currently
the nearest, the clearest or on the chain can fly **anywhere** without moving `Φ`
by one bit — and those are the drones that have to pre-position for the relay and
the handoff. With four drones on the axis and the fifth stranded to one side, one
8 m step home is worth:

| | 200 m off-axis | 500 m | 800 m |
|---|---|---|---|
| shipped | **0.0000** | **0.0000** | **0.0000** |
| `Φ v2`, per step | 0.0078 | 0.0010 | 0.0003 |
| `Φ v2`, whole trip home | **+0.339** | **+0.446** | **+0.465** |

⚠️ **This retro-explains the *shape* of every null in this block.** `w_hold`,
`w_relay`, `d_ref 400` and `potential_scale 30` all scaled a potential whose
gradient in the operating regime was 0.013–0.03 per step. The reason they failed
is arithmetic, not mechanism.

### ☠️ Two claims this block has been reasoning from are wrong

**The learned policy is not collecting the energy bonus.** Measured per drone per
step on the eval split at stage 4:

| | speed p50 | steps > 24 m/s | energy term | steps at the map wall | mean \|a_z\| |
|---|---|---|---|---|---|
| B0 | **5.81 m/s** | 3.1 % | **−0.1250** | **0.9 %** | 0.005 |
| GNN | **24.71 m/s** | **56.7 %** | **−0.1333** | **23.1 %** | **0.821** |
| MLP | — | — | — | 15.6 % | 0.626 |
| random | 17.13 m/s | 13.9 % | −0.1158 | — | — |

The policy flies at the **25 m/s dash cap on 57 % of steps**, where
`P/P_hover ≈ 0.99`, and pays **more** energy than B0. It is nowhere near the
13.3 m/s minimum-power airspeed. 0.0544/step stays the right **bar to size `Φ`
against** — it is the largest per-step force the objective can apply — but it is
not the mechanism behind the 184 m stand-off. `DECISIONS.md`.

**`Φ` is loud, not quiet.** Its per-step magnitude is `|ΔΦ|` p90 = **0.365** for
the GNN and 0.052 for B0 — the learned policy receives *seven times more* shaping
than B0 while doing worse. The variance is the two binary terms flickering
(a sightline acquired and lost, a chain forming and breaking); the **directional**
content along the closing axis is 0.0133. "Switched off" is right about the
gradient and wrong about the amplitude, and the distinction matters because
scaling `k` amplifies the flicker and not the direction.

### 🔍 And one render, which the aggregates could not show

`render_episode.py --policy runs/full-d3/checkpoint.pt --compare --route 12`:
B0's tracks stay inside the HVT corridor with one observer held for the whole
episode. The learned policy's are **long sweeping arcs across the entire map**,
including the western half the HVT never enters, with straight segments pinned
along the box boundary where the position clamp zeroes the velocity. The e2e
trace is *good* — 25–30 Mbps — in long stretches and then drops to zero: the
failure is intermittent loss from drifting out, not a chain that is chronically
too weak.

### The rebuild — `reward.PHI_V2`, off by default

Full derivation, sizing rule and per-component argument in
[`REWARD.md`](REWARD.md). In brief:

```
Φ = k · [ 0.05·Φ_approach + 0.20·Φ_observe + 0.20·Φ_standoff
                          + 0.15·Φ_link    + 0.40·Φ_cover ]        k = 10
```

* **`Φ_standoff`** — the closing decision, a logistic centred on Block B's
  measured **127 m** sightline median, gated on `observed`. ⚠️ Not `w_hold`
  again: additive with its own budget rather than a factor multiplied *into*
  `Φ_observe`, and 0.077/step against `w_hold`'s 0.03.
* **`Φ_cover`** — coverage of the MCV→HVT axis, the only component that is not
  blind to four drones out of five. Soft-OR over drones plus a per-drone muster
  half, Cauchy kernel so the far field is never numerically zero.
* 🔒 **The five weights sum to 1.0 and `k` stays 10.** `Φ` is redistributed, never
  inflated: PBRS pays `(γ−1)·Φ` per step for *holding* a state, a drag
  proportional to `Φ` and therefore largest for the best policy (B0's mean
  shaping is **−0.018/step**). That is the most likely reason
  `potential_scale = 30` was a null rather than an improvement.

### The offline test that could have separated the four nulls

⚠️ **The ideal `Φ` is `V*`** — the shaped problem's value function is `V − Φ`, so
`Φ = V*` makes every advantage immediate. So score a candidate against the
discounted future `mission_capable` return of banked states, with no training run:

| pooled over both banks | shipped | `Φ v2` |
|---|---|---|
| corr(`Φ`, discounted future capable) | +0.270 | **+0.301** |
| mean `Φ`(B0) − mean `Φ`(learned) | +2.101 | **+3.584** |

The correlation gain is modest and is reported as such. The **separation** is what
moved, by 71 %. Costs a minute; would have been available for every one of the
five nulls.

⚠️ **Provenance.** All of the above is MPS, eval split, stage 4, F4, `num_envs`
32, seed 0, against `runs/full-d3` (a 12 M-step feedforward GNN). It is a
*design* measurement, not a result: no `Φ v2` policy has been trained yet, and
the gate for that is `BLOCK_G_PLAN.md` § Gate 3.

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
