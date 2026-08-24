# Block G — MAPPO, the curriculum, and making something learn

**Goal:** a training entrypoint that takes any fidelity rung and any of the three
architectures and produces a policy that beats B0. Everything before this block
built an environment nobody has learned in yet.

Consumes Block A (reward), Block D (`core.py`, `skrl_wrapper.py`), Block E (B0 as
the floor, `evaluate.py` as the metrics harness) and Block F (the `fidelity`
seam). Produces `src/models/`, `src/training/`, `configs/`, and the answer to
whether this project has a thesis.

---

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

### ⛔ Recurrence: implemented, unit-tested, and it does not train — parked

The diagnosis above says the actor cannot represent *"I am the observer"*, so a
GRU was added between the trunk and the head, **identical in all three rungs** so
RQ2 still isolates the trunk. It is built, it is tested, and **it loses badly to
the feedforward baseline it was meant to beat.** Recorded in full because the
diagnosis that motivated it still stands and somebody will propose this again.

| run (12 M, fast schedule, `--min-std 0.2`) | stage-4 plateau |
|---|---|
| feedforward GNN (d3) | **37.1 %** |
| feedforward MLP (d1) | 35.9 % |
| **recurrent GNN** | **1.6 %** |
| **recurrent MLP** | **0.7 %** |
| *random, for scale* | *11.0 %* |

**Below random**, and monotonically decreasing — the same shape as the
`clip_actions` bug. What is known, in the order it was established:

1. **The gradient is not inverted.** The known-optimum probe (`-w_effort·‖a‖²`,
   optimum `a = 0`) improves through the recurrent path, just ~2x slower and
   noisier than feedforward (−0.91 → −0.71 against −0.91 → −0.38).
2. **The model is exact.** Sequence-mode replay reproduces step-mode collection
   to **1.19e-7** given the same hidden state — a standalone check, no skrl
   involved. The GRU code, the `view(layers, N_seq, L, H)[:, :, 0, :]` initial-state
   selection and the episode-boundary chunking are all correct.
3. **The replayed hidden states are nevertheless wrong.** An epoch-0 identity
   check — before any parameter moves, the recomputed log-probability of the
   stored action must equal the stored one — gives mean |Δlog p| = **1.8e-3**,
   concentrated at **sequence position 0** and decaying along the sequence, with
   no episode boundary in the rollout. Given (2), that residual can only come
   from the stored/replayed states themselves.
4. **It is not a train/deploy context mismatch.** The obvious hypothesis was that
   training on 16-step sequences while collection integrates over 600 steps puts
   the GRU outside its trained state distribution. **Falsified:** the recurrent
   policy also collapses at **stage 1**, where episodes are 150 steps and the
   feedforward policy reaches 75–79 %.

5. **✅ A real skrl bug found and fixed — and it was NOT the cause.**
   `PPO_RNN.record_transition` ends with
   `self._rnn_initial_states = self._rnn_final_states`, binding both names to the
   **same dict**. From the next step on, `act()` writes the post-step state into
   `_rnn_final_states` -- which *is* `_rnn_initial_states` -- so the transition
   records a hidden state one step ahead of the one that produced its action.
   Measured directly: `stored[t] == h_in[t+1]` for every `t >= 1`, exact only at
   `t = 0`, before the aliasing happens. Every importance ratio in every
   recurrent update was computed against the wrong state.

   Fixed by `training/recurrent_ppo.PPO_RNN_Aligned`, which snapshots the states
   in `act()` before skrl can overwrite them. `test_recurrent.py` pins the root
   cause directly (the memory must hold the state the action was taken from) and
   the epoch-0 identity at both `--seq-len 1` and `--seq-len 4`.

   ⚠️ **Fixing it changed nothing about the training.** Stage 1 still collapses
   (37.4 % -> 2.2 %, against 35 % -> 3 % before the fix), and the known-optimum
   probe still improves ~2x more slowly than feedforward (−0.913 -> −0.712
   against −0.912 -> −0.416). The bug was real, is worth having fixed, and was
   not the mechanism. **Recurrence remains unresolved.**

6. **The fault is localised to sequence replay** — before and after the fix. `--seq-len 1` makes skrl take
   its non-sequence sampling path (no reordering, no sequence grouping) and
   changes nothing else — same GRU, same agent, same config. It **does not
   collapse**: at stage 1 it dips to 13 % and then recovers to **39.6 % and
   still rising**, against `--seq-len 16`'s collapse to 3 %.

⚠️ The epoch-0 ratio of 0.9999 is far inside PPO's 0.2 clip and cannot by itself
explain the collapse, so the 1.8e-3 residual is a **symptom** rather than the
mechanism — but it points the same way as (5): it is concentrated at sequence
position 0, which is exactly where the replayed initial state enters. The precise
defect in the sequence path is **not yet identified**; what is established is
which path contains it.

**Implementation note for whoever picks this up.** skrl 2.1.0 ships `PPO_RNN` but
**no recurrent MAPPO** (`skrl/multi_agents/` contains no RNN handling at all).
Since `SharedPolicyWrapper` already presents the swarm as one parameter-shared
agent, skrl's MAPPO at a single agent id *is* PPO with a centralized state-based
critic, so `PPO_RNN` was used as the vehicle — same algorithm, different class.
**Where to resume, and the strongest untested hypothesis first.** The critic is
**feedforward while the policy is recurrent**. The policy's behaviour then
depends on a hidden state the value function cannot see, so `V(s)` is an average
over hidden states and every advantage is biased for exactly the
history-dependent behaviour the GRU exists to produce. Yu et al. (2022) make
both recurrent. That is one flag away and has never been tried.

Second: `grad_norm_clip = 0.5` is applied to the policy and value parameters
**jointly**, and a GRU's gradient norm is much larger than an MLP's -- so the
clip may be squashing the critic's update whenever the actor's is big.

`--seq-len 1` is the working fallback and the bisection point: it bypasses
sequence replay, so a recurrent policy still carries hidden state through
collection and evaluation — which is what role persistence needs — but gets no
backpropagation through time, so the GRU can only learn myopic uses of its state.
Resume by diffing the stored `rnn_policy_0` tensor against a hand-stepped
reference across an episode boundary, at `mini_batches=1` first.

**Recommendation: park it, but it is no longer a dead end.** The best result
remains the **feedforward GNN at 37.1 %**, which was still improving at 12 M. Recurrence is a real lead — B0's
264-step observer tenure against 27–35 is not going to be closed by a stateless
policy — but it should be resumed with the CUDA budget, not debugged further on
a laptop.

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

### What is still open

* **G1a / G1b** — blocked on CUDA.
* **G5** — the three architectures are built, parameter-matched to **2.3 %** and
  tested (permutation invariance, off-N at 3/5/8, DeepSets == GNN with `e_ij`
  zeroed). Not yet *compared* on a training run.
* **G6** — `tau_c` / `tau_l` untouched. Retune once, against learning speed.
* **G7** — Block F's open question. Needs one pilot per rung.
* **The curriculum schedule is provisional.** `(0.15, 0.35, 0.60)` with a 20 %
  mix is a starting point, not a measured schedule. Find it, then freeze it, then
  record the freeze in `DECISIONS.md`.
* **The full mission does not clear B0.** Best pilot 37.1 % against 58.0 % on the
  train split. The gate is not met and the diagnosis above is where to attack it.
* **The seed spread is undiagnosed.** 60–78 % over five runs at stage 1. Judge
  every tuning decision on the worst seed.

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
