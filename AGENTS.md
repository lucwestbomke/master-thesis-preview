# AGENTS.md — UAV Swarm MARL Thesis

Entry point for any agent or human working in this repo. Kept deliberately short;
detail lives in `docs/` and is read **on demand**.

## How to read these documents

Every claim in `docs/` and here is one of three kinds, and they are **not
equally binding**. Conflating them is how a wrong guess ends up carrying the
authority of a measurement — which cost this project a week when a misattributed
diagnosis sat in `BLOCK_G.md` looking exactly like a result.

| tag | means | how to change it |
|---|---|---|
| 📏 **MEASURED** | a number that came out of a script, with device and seed count | measure it better, then overwrite. Data is not a rule |
| 🔒 **CONSTRAINT** | protects an RQ's interpretability, not the model's score | change it **deliberately**, in `DECISIONS.md`, and re-derive what it invalidates |
| 🔧 **PROVISIONAL** | a default nobody has tested | change it freely, on evidence, no ceremony |

The asymmetry that justifies the middle row: **you can always retrain a model;
you cannot retroactively un-confound an experiment.** Breaking a 🔒 does not make
the policy worse — it makes the resulting number uninterpretable, and the runs
have to be redone. A 🔧 costs twenty minutes.

⚠️ **Untagged prose is 🔧 by default.** If it mattered, someone would have
measured it or written down what it protects.

---

## Mission
A swarm of `N` UAVs must simultaneously **observe** a moving ground High-Value
Target (HVT) in Frankfurt, **relay** the sensor feed to a Mobile Command Vehicle
(MCV) over a multi-hop chain at ≥5 Mbps end-to-end, and **survive** on finite
batteries while a jammer riding the HVT degrades links near it. Buildings block
line of sight, so the relay chain is geometrically necessary.

- **RQ1 (primary):** which physical effects must a channel model include for
  learned policies to transfer? Train one policy per fidelity rung F0–F4,
  evaluate all under F4. Hypothesis: **occlusion** dominates.
- **RQ2:** MLP → DeepSets → GNN; zero-shot transfer across `N ∈ {3,5,8}`.
  ⛔ The **second city is cut** (2026-08-23) — LoD2 is a Hessen-only service, so
  it was a full Block B rebuild, not "one extra OSM extract". Analytical weight
  sits at **N = 8** — [`docs/DECISIONS.md`](docs/DECISIONS.md).
- **RQ3:** does the observer role **hand off** as sightlines change, and is the
  handoff coordinated and anticipatory?

---

## Where the project is

| Block | What | State |
|---|---|---|
| **A** | Channel, routing, energy, reward — all pure, batched, tested | ✅ **done**, 103 tests |
| **B** | Frankfurt LoD2/OSM pipeline → buildings + road graph as tensors | ✅ **done**, `data/frankfurt_box.npz`, 27 tests |
| **C** | Occlusion: batched torch segment-vs-**oriented**-box (slab method) | ✅ **done**, 29 tests; `torch.compile` required |
| **D** | Batched env core + PettingZoo adapter + skrl wrapper | ✅ **built**, 46 tests; gate met with ~3170× margin. Awaiting the CUDA re-run of the *full env* (D3) — [`docs/BLOCK_D.md`](docs/BLOCK_D.md) |
| **E** | Presentation renderer + B0 scripted baseline | ✅ **done**, 27 tests; B0 = **57.2 %** mission-capable, and the rate requirement moved 5 → **15 Mbps** — [`docs/BLOCK_E.md`](docs/BLOCK_E.md) |
| **F** | Fidelity levels F0–F4 as config flags — RQ1's independent variable | ✅ **done**, 38 + 9 tests; `R` = **524 m** measured; F4 == the pre-Block-F env — [`docs/BLOCK_F.md`](docs/BLOCK_F.md) |
| **G** | MAPPO integration + curriculum | 🔨 **in progress**, 370 tests. Equal-budget sweep + stage B done. **Gate not met**: GNN **41.2 % [3.8]** against B0's **57.3 %**, eval split, CUDA, 5 seeds. `Φ` rebuilt 2026-08-27, untrained — [`docs/BLOCK_G.md`](docs/BLOCK_G.md), plan in [`docs/BLOCK_G_PLAN.md`](docs/BLOCK_G_PLAN.md) |
| H | Sionna offline validation of the closed-form channel | not started |

Phase 0 (prep) runs to Feb 2027; the thesis window is Mar–Aug 2027. **Freeze the
environment end of March 2027** — results before are pilots, after are thesis
material. Full timeline in [`docs/THESIS_PLAN.md`](docs/THESIS_PLAN.md).

Block B is done; [`docs/BLOCK_B.md`](docs/BLOCK_B.md) records what was measured
and decided, and is the reference for the artefact's contents. Block C is
specified in [`docs/BLOCK_C.md`](docs/BLOCK_C.md), Block D in
[`docs/BLOCK_D.md`](docs/BLOCK_D.md), Block E in
[`docs/BLOCK_E.md`](docs/BLOCK_E.md), Block F in
[`docs/BLOCK_F.md`](docs/BLOCK_F.md), Block G in
[`docs/BLOCK_G.md`](docs/BLOCK_G.md). Why each block exists, what it gates and
which thesis chapter it feeds: [`docs/ROADMAP.md`](docs/ROADMAP.md).

🔨 **Block G is in progress and the full-mission gate is NOT met.** Eval split,
F4, CUDA, 5 seeds, one harness — random **10.7 %**, MLP **31.2 %**, DeepSets
**38.1 %**, GNN **41.2 % [3.8]**, **B0 57.3 % [3.9]**. A **16.1 pp** gap, and the
learned policy closes 65 % of the random→B0 distance and stops.

📏 **The whole gap is `observed` — 66.5 % against 92.8 %.** Conditioned on holding
a sightline the GNN converts it exactly as well as B0 (0.620 vs 0.617). And
conditioned on observing, every learned policy's chain is indistinguishable from
**random's** (1.86–1.91 hops vs random 1.83, B0 2.26): the swarm learned to fly at
the target and nothing about relaying.

🔍 **One mechanism fits both.** B0 parks its observer at 79 m and *therefore* needs
2.1 hops; the learned policies loiter at 291 m on 1.27 hops. Going in close is only
survivable if teammates relay behind you — so with a shared team reward and no
per-drone role signal, **no drone can afford to be the one that goes in.** A
coordination trap: the deficit is **role emergence**, and observer tenure
(47 vs 295) and the missing chain are its two symptoms.

⛔ **Four interventions have now been tested against this and all four are
nulls** — recurrence (−1.05 pp), `w_hold`, the per-drone `w_relay` potential, and
Yu et al.'s agent-specific critic (which actively hurt). Each was pre-declared and
run at 5 seeds; all are recorded in [`docs/DECISIONS.md`](docs/DECISIONS.md).

☠️ **`Φ` was audited on 2026-08-27 and it is switched off where it matters**, which
retro-explains the *shape* of all four. Measured by
[`scripts/measure_potential.py`](scripts/measure_potential.py) on states a real
policy visits — two defects, different in kind:

1. **No gradient along the closing axis.** Observer 250 → 60 m with the ray clear
   and a chain at 25 Mbps, the exact decision the diagnosis turns on: the shipped
   `Φ` moves **0.320 in total**, **0.0133 per 8 m step**, against the **0.0544**
   per step the energy term can pay. **0.25×.**
2. **`Φ` is exactly constant in four drones out of five.** Every component is a
   hard `min`, a hard `max` or the router's chosen path, so a drone that is not
   the nearest, the clearest or on the chain can fly anywhere without moving `Φ`
   by one bit. Moving a stranded drone 8 m home is worth **0.0000**, at any
   distance. 📏 Learned policies sit against the map boundary on **15–23 %** of
   steps; B0 on **0.9 %**.

⚙️ **The rebuilt potential ships as `--phi v2`, off by default and bitwise
identical to the shipped `Φ` when off.** Two new components — `Φ_standoff` (the
closing decision, a logistic on Block B's measured 127 m sightline threshold,
**0.0774/step = 1.42× the bar**) and `Φ_cover` (axis coverage, the only component
that is not blind to four drones out of five). 🔒 The five component weights sum
to 1.0 and `k` stays 10: **`Φ` is redistributed, never inflated**, because PBRS
charges `(γ−1)·Φ` per step for *holding* a state and that drag is largest for the
best policy. [`docs/REWARD.md`](docs/REWARD.md); the gate is
[`docs/BLOCK_G_PLAN.md`](docs/BLOCK_G_PLAN.md) § Gate 3 and **no `Φ v2` policy has
been trained yet**.

⚠️ **Two claims this project was reasoning from are measured wrong** — both in
[`docs/DECISIONS.md`](docs/DECISIONS.md):

* **The learned policy is not collecting the energy bonus.** It flies at the
  **25 m/s dash cap on 57 % of steps** (`P/P_hover ≈ 0.99`) and pays **−0.1333**
  per step against B0's **−0.1250** — *more*, not less. 0.0544/step stays the
  right **bar to size `Φ` against**, but it is not the mechanism behind the 184 m
  stand-off, and over an episode it is ~2 % of that policy's return.
* **`Φ` is loud, not quiet.** `|ΔΦ|` p90 is **0.365** for the GNN against B0's
  0.052 — the learned policy receives 7× more shaping while doing worse. The
  amplitude is the two binary terms flickering; only the *direction* is missing.
  Scaling `k` amplifies the flicker, which is the likeliest reason
  `potential_scale = 30` was a null.

📏 **The failure, measured.** The learned observer stands at **184 m**; B0 stands
at **88.8 m [1.2]**. Block B measured the along-street sightline median at
**127 m** — so B0 sits *inside* it and the learned policy sits *outside* it, and
its view is intermittent by construction of the city. That is the whole
`observed` gap (64.9 % against 91.6 %), and it is a **geometry threshold**, not a
smooth cost.

🔍 **Role differentiation emerges at stage 1 and collapses at stage 4.**
`role_entropy` (0 = one drone owns the observer role, 1 = all equal): B0 **0.10**,
the same GNN architecture **0.20 at stage 1** and **0.50 at stage 4**, random
**0.60**. On every *role* statistic the full-mission policy is near random; on
every *performance* statistic it is far above it. **The swarm learns to fly at the
target and does not learn to organise** — and it can organise when the target
holds still, so what breaks is holding a role while the target moves.

⚠️ Two corrections this forced: the long-quoted **291 m** stand-off was the *first
pilot*, not current; and **`hop | observed` measures geometry, not behaviour**
(random 1.83, every learned policy 1.86–1.93, B0 2.26 — hop count follows from
where the observer stands). Three interventions were judged on it.

Sweep findings worth carrying: `deep` (rollout 64) wins and `wide` **quadrupled**
the seed spread it was built to shrink; **MLP → DeepSets is +6.9 pp and robust**
while DeepSets → GNN is **+0.4 pp — a null** once shaping is held fixed; and the
shaping axis is noise, so a winner's shaping label means nothing. ⚠️ The old
"45.1 %" headline was that noise: `shipped` at 5 seeds gives 40.7 %.

☠️ **`skrl`'s `PPO_RNN` is an un-migrated fork of an older PPO** and its stale
`compute_gae` masks GAE on `terminated` alone, so at every truncation the
recursion runs *through* the reset and pulls the next episode's value and
advantage backwards. Recurrent training collapsed for a week on this, and the GRU
was blamed for it: the collapse reproduces with **feedforward** models on the same
path (3.8 % against MAPPO's 76.2 %). Fixed in
`training/recurrent_ppo.PPO_RNN_Aligned`, isolated to that one line, and
recurrence now reaches feedforward parity. ⚠️ **No reported number is affected** —
everything measured went through `MAPPO`, which is correct. Anything else
inherited from `ppo_rnn.py` should be diffed against `mappo.py` before it is
trusted.

Earlier stage-1 record, unchanged: MAPPO reaches **74.8 % [12.7]**
mission-capable on curriculum stage 1 against random's **35.1 %** (5 **training**
seeds, deterministic, scored through `evaluate.py` on the train split, MPS).
The spread is wide (60–78 %) and B0 still wins that stage at 87.5 %. Three
things to carry before touching the trainer — [`docs/BLOCK_G.md`](docs/BLOCK_G.md):

1. ☠️ **Never set `clip_actions=True` on the actor.** skrl clamps the sampled
   action and then evaluates its log-probability under the *unclamped* Normal,
   which inverts learning: under a reward whose only term is `mission_capable`,
   the policy fell 30 % → 4.6 %. The tell is the action standard deviation rising
   with `entropy_loss_scale = 0`. `core._advance_drones` clamps actions itself.
2. **Training uses `SharedPolicyWrapper`, not `SwarmMultiAgentWrapper`.** The
   swarm is ONE parameter-shared agent over `num_envs * N` rows. Per-drone
   policies cannot be evaluated at `N = 8`, so RQ2's zero-shot columns would not
   exist — and handing skrl one shared `Model` under `N` agent ids builds `N`
   optimizers over the same parameters and runs `N` stale sequential updates.
3. **The learner must be handed `env.final_observations()` / `final_states()`**,
   not what `step()` returns. With `auto_reset` the returned tensors are already a
   fresh episode's opening, so `time_limit_bootstrap=True` would bootstrap an
   unrelated state — silently, while the flag still looks correct. Needs
   `EnvConfig(training_extras=True)`, which is off by default so the golden trace
   is untouched.

⚙️ **Block F built RQ1's independent variable.** `EnvConfig.fidelity` is a
five-value enum (`"F0"`…`"F4"`, default `"F4"`) and every flag derives from it —
`channel_occlusion`, `binary_capacity`, `channel_jammer`, `reuse_limit`. Read
[`docs/BLOCK_F.md`](docs/BLOCK_F.md) before touching the env, and carry three
things:

1. **Fidelity gates the CHANNEL only.** The sensor, the reward's clearance term
   and every diagnostic run on **true geometry at every rung** — measured:
   `observed` is 92.0 % at all five, and `chain_occluded` reads **85.8 % at F0**
   rather than the 0.0 % a gated diagnostic would report. The observation's
   *channel* features (noise floor, clearance-to-MCV, per-edge clearance) do
   follow the rung; its *sensor* features do not.
2. **`R` = 524 m, measured** (`scripts/calibrate_r.py`), cross-checked by degree
   matching at 418 m. `"median link range"` has two readings differing 2× — see
   BLOCK_F.md before quoting it.
3. **The ladder is cumulative in effects, not monotone in difficulty.** F1
   (27.9 % under B0) is *harder* than F4 (56.0 %). F0, F2 and F3 all collapse
   `mission_capable` onto `observed`, because `reuse_limit = 1` below F4.

⚠️ **Block E changed a settled parameter and several downstream expectations.**
Read [`docs/BLOCK_E.md`](docs/BLOCK_E.md) before planning an experiment:

1. **The rate requirement is 15 Mbps, not 5.** At 5 the radio link never bound —
   the chain carried 8× the bar, `mission_capable` collapsed to `observed`, and a
   *scripted* baseline reached 93 % with the metric saturated. At 15 the binding
   constraint moves from the sensor to the relay chain, which is the swarm
   problem this project studies. **Block D's numbers are all at 5 Mbps and are
   not comparable.** Ledger of what was re-derived:
   [`docs/DECISIONS.md`](docs/DECISIONS.md).
2. **The altitude ceiling is no longer *derived* from W1.** At 15 Mbps a
   best-placed solo drone fails at every altitude (0.4 % at 80 m, 0.8 % at
   120 m), so W1 holds everywhere and no longer selects a ceiling. The band stays
   40–80 m on the **A2A-occlusion** ground — 31.2 % of A2A links blocked at 80 m
   against 24.6 % at 120 m — which is the effect RQ1 measures. Do not raise the
   ceiling on the strength of the new W1 numbers.
3. **F3 → F4 is a large effect** (Δ = +26.5 pp), reversing the null predicted at
   5 Mbps. The divisor flips 26.6 % of B0's chain-steps.
4. **The escalation worry is closed.** Conditioned on a chain existing, B0 is
   multi-hop on 95.1 % of last-third steps with the divisor saturated on 50.0 %.
   The old "4.2 % 3-hop" figure was diluted by chainless steps and excluded 4-
   and 5-hop chains.
5. **RQ2's off-N weight belongs at N = 8, not N = 3.** Control is worth +25.9 pp
   at N = 8 against +3.2 pp at N = 3 — the hardest condition is the least
   informative one, because three drones on a three-hop chain have one viable
   arrangement. **Do not train at more than one N**: it confounds the zero-shot
   transfer test RQ2 exists to run.
6. **RQ3 is re-pointed** from observer handoff (~0.9 per episode) to **relay-chain
   reconfiguration** (~52 per episode), which is driven by occlusion changing
   link quality rather than sensor occlusion. E3a's ablation now zeroes the
   `on_path` bit and edge features, not `sees_hvt`.

✅ **G1b measured on CUDA (2026-08-24): a 10 M-step run costs 2.2 minutes**
end-to-end with the learner attached (75,252 env-steps/s, RTX 5090, `num_envs`
1024) against a ≤3 h target. The 45-run matrix is **~2 GPU-hours, not 120**.
Compute is not a constraint — [`docs/BLOCK_G.md`](docs/BLOCK_G.md). GPU
utilisation was 33 % at 3 GiB of 32 GiB, so `num_envs` has room to grow on
learning grounds.

⚠️ **The frozen trace is `arm64`-specific.** `golden.py` pins `device="cpu"`, but
float32 is not associative across instruction sets: the same commit diverges by
2.4e-3 over 300 steps on x86-64 and passes exactly on Apple silicon.
`test_golden.py` asserts bitwise equality only on arm64 and checks aggregate
rates elsewhere. **Run the suite on arm64 before believing a golden failure** —
[`docs/DECISIONS.md`](docs/DECISIONS.md).

✅ **The throughput gate is met.** RTX 5090, 2026-08-12: compiled occlusion runs
**3.17 M env-steps/s** at `num_envs = 1024` — ~3170× the gate — and occlusion is
~99 % of the step, so the env is not the bottleneck. `num_envs` is now chosen on
*learning* grounds, not throughput. Full table and provenance in
[`docs/BLOCK_C.md`](docs/BLOCK_C.md).

**Use `torch.compile`** — it is 110–150× on CUDA and free. But note the earlier
"required, not an optimisation" framing was an MPS artefact: on CUDA *eager also
clears the gate* (28.9 k env-steps/s, 29× over). And the 17.6 GB figure is memory
*traffic* per call, **not** live allocation — chunking keeps peak VRAM under 4 GB
in both paths. Do not reason about VRAM pressure from it.

---

## Read before you change things

| File | Read it when |
|---|---|
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | **start here** — the red thread: claim → experiment → block → chapter |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | **always, second** — every entry was proposed then killed on evidence |
| [`docs/THESIS_PLAN.md`](docs/THESIS_PLAN.md) | making a research decision: RQs, conditions, metrics, timeline |
| [`docs/PHYSICS.md`](docs/PHYSICS.md) | touching channel / routing / energy / scenario parameters |
| [`docs/REWARD.md`](docs/REWARD.md) | touching the reward or its weights |
| [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) | building the env: episodes, cue, curriculum, observations |
| [`docs/MODELS.md`](docs/MODELS.md) | building actors/critics |
| [`docs/NEGATIVE_RESULTS.md`](docs/NEGATIVE_RESULTS.md) | before proposing adaptive transmit power |
| [`docs/BLOCK_B.md`](docs/BLOCK_B.md) | consuming `data/frankfurt_box.npz`, or touching geometry/routes |
| [`docs/BLOCK_C.md`](docs/BLOCK_C.md) | touching occlusion, or the geometry it consumes |
| [`docs/BLOCK_D.md`](docs/BLOCK_D.md) | building the env core, or touching altitude / cue / sensor / the throughput gate |
| [`docs/BLOCK_F.md`](docs/BLOCK_F.md) | touching the fidelity ladder, `R`, or anything RQ1 reports |
| [`docs/BLOCK_G.md`](docs/BLOCK_G.md) | **building models, the trainer or the curriculum** — read the anti-learning bug first |
| [`docs/BLOCK_G_PLAN.md`](docs/BLOCK_G_PLAN.md) | **starting a work session** — what happens next, in order, with the gate rules declared in advance and a dated stopping rule |

---

## Hard rules

**Device.** Training tensors live on `cuda:0`. **Never call `.cpu()`, `.numpy()`
or `.item()` inside env `step()` or the training hot loop** — `.item()` forces a
GPU sync and is the easy one to miss. Local dev is Apple Silicon (CPU/MPS), toy
configs only. Guard device selection; never silently degrade a real run to CPU.

**Throughput.** ≥1000 **env-steps/s**, where one env-step is *one environment
advancing one tick*, summed across the batch — **not** one batched `step()` call.
The two readings differ by 1000× and both appeared in this repo; the transition
reading is the one THESIS_PLAN §3's budget is written in (10 M steps ÷ 1000/s
≈ 2.8 h/run × 45 runs ≈ 120 GPU-h), and cost is what the gate protects. Settled
in [`docs/BLOCK_D.md`](docs/BLOCK_D.md) — do not re-open it.

Under that reading the 1000/s floor clears easily, so the number actually
reported is **wall-clock for a 10 M-step run, end-to-end including the learner,
target ≤3 h**. A **gate, not an aspiration** — measure it before building
anything on top of the env.

**Batched core, thin adapter.** The env core carries a leading `num_envs`
dimension. The PettingZoo adapter exists for API-compliance tests and visual
debugging only; training uses a custom skrl multi-agent wrapper. (skrl's
`PettingZooWrapper` round-trips through NumPy every step at `num_envs == 1`.)

**Geometry offline.** `osmnx`/`shapely` are CPU-and-NumPy-only — used **only** in
`scripts/prep_osm.py` to bake buildings into a tensor. Runtime occlusion is
vectorized segment-vs-**oriented**-box (slab method) in pure torch: rotate the
segment into each box's local frame, then run the standard branch-free slab
test. Axis-aligned boxes were measured and rejected — they fill 94 % of the
Frankfurt box ([`docs/DECISIONS.md`](docs/DECISIONS.md)). Buildings are 2.5D:
check the segment's altitude across the 2D intersection interval, not just a
planar crossing.

**B0 sees only `obs["flat"]`.** The scripted baseline is a pure function of the
same `(B, N, 108)` tensor the actors consume, plus its own carried state — it
never reads `env.hvt_pos` or any other env attribute. That is what makes "does
MARL earn its keep?" a question about *control* rather than about information,
and it is asserted by a test that runs a decoy scenario underneath a replay
(`src/baselines/test_b0.py`). The `oracle` variant is the deliberate exception
and takes ground truth through an explicit argument. Do not relax this to make a
number look better; the measured cost of the restriction is **0.6 pp**.

**Formulas are traceable.** Do not change path-loss / SINR / capacity / energy
formulas without updating the hand-computed tests and checking the cited
standard. They appear in the methodology chapter.

**Multi-seed.** ≥5 seeds for anything reported as a finding. Median + IQR, never
mean ± std — RL returns are not normally distributed. Never report single runs.

---

## 🔧 What you may change freely, right now

The 🔒 list below is short on purpose. Everything here is untested default and
carries **no** methodological weight — change it on evidence without ceremony,
and do not write a `DECISIONS.md` entry unless the answer is interesting.

| | current | note |
|---|---|---|
| recurrence: GRU vs LSTM, `rnn_hidden`, `--seq-len` | GRU, 128, 16 | GRU is the MARL default (Yu et al. 2022), never compared here |
| trunk widths | 232 (MLP) / 128 (relational) / 256 (critic) | only the ±20 % parity between rungs is 🔒 |
| learning rate, `learning_epochs`, `gae_lambda` | 3e-4, 4, 0.95 | LR is fixed so cadence is not confounded with it, not because 3e-4 is right |
| `min_log_std`, `entropy_loss_scale` | 0.2, 0.0 | the floor was inert at stage 1 — σ never reached it |
| `num_envs` / `rollouts` / `mini_batches` | the sweep's `deep` | 📏 `deep` won, but the grid confounds two axes |
| curriculum boundaries and mix | (0.15, 0.35, 0.60), 20 % | explicitly provisional; find it, freeze it, then it becomes 🔒 |
| **everything inside `Φ`** | see below | PBRS proves none of it can move the optimum. 📏 Now **measured** rather than guessed — `scripts/measure_potential.py` scores a candidate before a training run |

### ⚠️ `Φ`'s internal weights are free, and nobody has touched them

`REWARD.md` says *"the scale of `Φ` is free… every other weight changes the
objective — ⛔ sweep nothing but `λ`"*, which reads as though `w_approach`,
`w_observe` and `w_link` were locked. **They are not.** They sit *inside* the
potential:

```
Φ = k · [ w_a·Φ_approach + w_o·Φ_observe + w_l·Φ_link ]
```

and Ng, Harada & Russell's proof holds for **any** `Φ`, so any setting of them is
optimum-preserving for exactly the same reason `k` is. `REWARD.md` itself calls
`0.25 / 0.35 / 0.40` *"suggested"*. The ⛔ applies to the **objective** weights
(`mission`, `idle`, `energy`, `battery_variance`, `effort`), where only `λ` is
sweepable.

So the free reward surface is larger than the docs imply: `k`, `d_ref_m`,
`tau_clearance_m`, `tau_capacity_mbps`, `w_approach`, `w_observe`, `w_link`,
`w_hold`, `d_hold_m`, `w_relay`, and (since 2026-08-27) `w_standoff`,
`d_standoff_m`, `tau_standoff_m`, `w_cover`, `r_cover_m`, `n_cover_samples` —
none of which can change what "success" means. Every one has a `--flag`, and
`test_train.py` derives that requirement from `RewardWeights` rather than
hand-listing it, because a knob nothing can set is dead code and this repo has
shipped that twice.

⛔ **Do not tune any of them by argument.** `scripts/measure_potential.py` banks
the states a policy actually visits and scores a candidate `Φ` over them — swing,
per-step gradient against the 0.0544 bar, recall for a drone holding no role, and
correlation against the discounted future `mission_capable` return, which is the
quantity the ideal `Φ` (`= V*`) would equal exactly. It costs a minute and it
would have separated all four of this block's nulls from a real candidate.

---

## Never do these

These are not all the same kind of rule, and the difference decides what happens
if you break one. **📏 entries are measured** — someone ran the script, and the
number is the argument; overturn them by measuring better. **🔒 entries protect a
research question** — breaking one does not make the policy worse, it makes the
resulting number uninterpretable, and the affected runs have to be redone after
the freeze, when there is no time. Read the tag before you argue with the entry.

- ⛔ 📏 **Reintroduce transmit power as an action.** Three framings, three nulls —
  [`docs/NEGATIVE_RESULTS.md`](docs/NEGATIVE_RESULTS.md). Action space is motion
  only (3-dim); Ptx is fixed at 30 dBm. E4 reproduces the null deliberately.
- ⛔ 📏 **Raise the Ptx ceiling.** At 40 dBm a *blocked* A2A link carries 15 Mbps
  over 2.8 km — one drone spans the map and the relay chain becomes pointless.
- ⛔ 🔒 **Use channel fidelity as a curriculum axis.** It is RQ1's independent
  variable. Same reasoning forbids ramping building density.
- ⛔ 🔒 **Use `SAGEConv`** for the GNN rung. It cannot take edge features, so it
  silently collapses the GNN into DeepSets and RQ2 measures nothing.
- ⛔ 📏 **Terminate the episode on mission failure.** The policy learns never to
  acquire, and a random initial policy never reaches the tracking phase.
- ⛔ 🔒 **Sweep more than `λ`.** Other weights are pinned by behavioural orderings
  in [`docs/REWARD.md`](docs/REWARD.md).
- ⛔ 📏 **Raise the altitude ceiling above 80 m.** It is not a comfort margin — it
  is where the scenario stops being a swarm problem. A best-placed *single* drone
  is mission-capable 3.3 % of the time at 80 m, 23 % at 100 m and 57 % at 120 m,
  so raising it falsifies W1 ("one drone cannot do this"). Raising it also *weakens*
  RQ1: A2A blockage falls 31 % → 25 % → 10 % at 80 / 120 / 180 m. Measured:
  [`scripts/measure_envelope.py`](scripts/measure_envelope.py).
- ⛔ 🔒 **Move to mmWave.** It makes RQ1 trivial (mmWave is textbook
  blockage-limited, so "occlusion matters" stops being a finding), needs
  beamforming and beam-pointing modelling that couples to the motion policy, is
  the wrong band for the tactical MANET radios Ptx is derived from, and would
  invalidate Block A, PHYSICS.md and Chapter 3 before the freeze. It belongs in
  future work, where it strengthens the discussion for free.
- ⛔ 📏 **Lower the rate requirement back toward 5 Mbps.** It was 5, and at 5 the
  link never binds: the chain carries 8× the bar, `mission_capable` becomes
  identical to `observed` for every policy, the N-scaling flattens to ~92 % at
  N = 3/5/8, and F4's rate-division rung does nothing. Raising it to 15 is what
  made the relay chain the binding constraint. [`docs/DECISIONS.md`](docs/DECISIONS.md).
- ⛔ 📏 **Raise the altitude ceiling because W1 now permits it.** At 15 Mbps W1
  holds at every altitude, so it no longer pins the ceiling — but A2A occlusion
  still does, and that is the constraint RQ1's F1 rung rests on.
- ⛔ 🔒 **Promote the `measure_envelope.py` waypoint policy to B0.** It reads
  `env.hvt_pos` off the env, so it is oracle-fed, and it is untuned. It exists
  only so Block D's numbers had a regenerable ceiling. B0 is `src/baselines/b0.py`.
- ⛔ 🔒 **Set a fidelity flag directly.** `channel_occlusion`, `binary_capacity`,
  `channel_jammer` and `reuse_limit` are derived from `fidelity` and are not
  fields. `channel_occlusion=False, channel_jammer=True` is not on the ladder and
  nothing else would stop its number reaching a table.
- ⛔ 📏 **Use `no_buildings` as "F0".** It removes buildings from the **world** —
  sensor and diagnostics included — which is `F0-nogeo`, a separate named
  condition. Measured: it scores 100 % mission-capable, is 82.3 % single-hop and
  runs 1.9× faster. Folding it into F0 deletes the relay problem.
- ⛔ 🔒 **Re-capture `data/f4_golden.pt.gz` to make a test pass.** It is the only
  record of what the env did before the ladder existed; re-capturing compares the
  new code against itself. A failure means the environment changed, and the
  question is which Block D or E number moved.
- ⛔ 🔒 **Compare a number measured on one device with one measured on another.**
  `torch.Generator` streams differ per device, so the same seed draws *different
  episodes* on MPS than on CPU. The physics is identical; the sample is not.
- ⛔ 🔒 **Add heavy dependencies** (sim engines, RL frameworks) without flagging.
- ⛔ 🔒 **Train through `SwarmMultiAgentWrapper`.** It is the API-contract wrapper.
  Reported runs go through `SharedPolicyWrapper` — see the Block G note above.
- ⛔ 🔒 **Turn `training_extras` on for a run that reports a number without saying
  so.** It widens the `extras` contract `test_golden.py` pins. Training needs it;
  measurement scripts do not.
- ⛔ 🔒 **Use adaptive curriculum advancement in a reported run.** It hands the
  easier fidelity rungs more experience at the final stage and confounds RQ1
  unrecoverably. `curriculum.weights()` is a pure function of training progress
  so that it *cannot* see the rung.
- ⛔ 🔒 **Cite constants an AI produced.** `TODO(verify)` markers in `channel.py`
  and `energy.py` mean exactly that — and now also the 120 m altitude ceiling.
  📏 **TR 36.777 was worked on 2026-08-26 and is 3/4 closed**: the LoS intercept
  and slope and the NLoS slope are corroborated; **the NLoS intercept `32.4` is
  not**, and it happens to equal TR 38.901's *terrestrial* UMi LoS intercept,
  which is the neighbouring constant a transcription slip lands on. One human
  reading of the UMi-AV table closes it — [`docs/PHYSICS.md`](docs/PHYSICS.md).
  📏 A2A's `blockage_db = 20.0` was worked the same day and is **closed by
  sensitivity, not by citation**: the physically-motivated value is ~40 dB
  (median occluded ray passes 60.5 m *inside* a Frankfurt tower), but B0's
  headline moves 59.7 → 59.5 % across a 20–40 dB sweep, because 83 % of
  occluded chain edges are air-to-ground. `scripts/verify_blockage.py`.
  📏 And TR 36.777 is still current: **TR 38.901 adds no aerial UE heights
  through Rel-19 (June 2025)**, so there is nothing newer to migrate to.

---

## Settled parameters

| | Value | Basis |
|---|---|---|
| City / area | Frankfurt, **1500 m** box, centre **50.11200 N, 8.67040 E** (UTM 32N) | heterogeneous: low fabric gives a workable observation envelope, towers block A2A. Centre chosen by sweep — [`docs/BLOCK_B.md`](docs/BLOCK_B.md) |
| Buildings | Hessen **LoD2** via INSPIRE WFS, **oriented** boxes | 100 % height coverage; AABBs fill 94 % of the box and were rejected |
| HVT roads | all surface streets, speed capped **13.9 m/s** | the constraint is speed, not road class |
| Ptx | **30 dBm fixed** | UAV tactical MANET radios are 0.5–2 W |
| Jammer | 30 dBm in-band, rides the HVT | vehicle C-UAS barrage emitter |
| Carrier / bandwidth | 3.5 GHz / **10 MHz** | so the 5 Mbps target actually binds |
| Rate target | **15 Mbps** end-to-end | dual EO/IR feed at low latency. Raised from 5 in Block E: at 5 the link never bound, `mission_capable` collapsed to `observed`, and a script scored 93 %. Defined once in `reward.py` — [`docs/DECISIONS.md`](docs/DECISIONS.md) |
| Flight altitude | band **40–80 m**, ceiling = nominal | Both ends are *derived*, not chosen. **Floor**: model validity — below 40 m, 8–37 % of positions sit inside a building box (where occlusion's endpoint convention lets a drone see through its own building) and TR 36.777 stops at 22.5 m. **Ceiling**: scenario validity — above it a best-placed *single* drone can do the mission (3.3 % at 80 m vs 57.4 % at 120 m), which dissolves W1 and with it the reason for a swarm. [`docs/BLOCK_D.md`](docs/BLOCK_D.md). **Ceiling re-justified in Block E**: W1 no longer binds at any altitude, but the ceiling is the primary control on how much A2A occlusion — RQ1's independent variable — exists at all. B0 scores 56.6 % at 80 m and 74.5 % at 120 m while `observed` barely moves, so raising it deletes the effect under study |
| Drone speed | 20 m/s cruise, 25 m/s dash | **1.4×** against the fastest permitted road class (13.9 m/s), **~3.4×** against the bank's realised median (5.8 m/s). Quote both — tracking is not the binding constraint ([`docs/DECISIONS.md`](docs/DECISIONS.md)) |
| HVT | 300–500 m from MCV, drives away | chain escalates 1 → 2 → 3 hops |
| Episode | **600 steps × 0.4 s** = 240 s | covers the escalation to 3 hops. **Do not shorten**: at 120 s the HVT reaches only ~1000 m and *no* route enters the 3-hop regime (0.0 % vs 36.8 %). A route step is a fixed *displacement*, so changing `dt` also changes HVT speed and needs a re-bake of the frozen artefact |
| Swarm | `N = 5` trained; 3/5/8 evaluated | **Trained at one N only** — training off-N would turn RQ2's zero-shot transfer columns into in-distribution tests. Control headroom is largest at N = 8 (+25.9 pp) and smallest at N = 3 (+3.2 pp) |
| Discount | **γ ≈ 0.997–0.999** | default 0.99 is blind to the hard end of the episode |

Regenerate the sizing with [`scripts/scenario_design.py`](scripts/scenario_design.py)
and [`scripts/link_budget_check.py`](scripts/link_budget_check.py);
`tests/test_scenario_sizing.py` pins the trade-off table.

**Measured in Block B, replacing assumptions** ([`docs/BLOCK_B.md`](docs/BLOCK_B.md)):
street width median **21 m** (assumed 20) · canyon ratio `H_b/W` median **0.93**
(assumed 1.10) · across-street envelope at 80 m median **43 m**, p10–p90 24–88
(assumed a flat 36) · along-street sightline median **127 m**, p90 387.
The **830 m sensor range never binds** — 99.8 % of sightlines are shorter — so
occlusion is the constraint everywhere, which is what keeps RQ1 measuring channel
physics rather than sensor specification.

---

## Stack (do not substitute without asking)
Python 3.12 · PyTorch · **uv** (`uv.lock` authoritative) · skrl (MAPPO, CTDE) ·
PyTorch Geometric · osmnx + shapely (offline only) · Weights & Biases ·
Hydra or plain YAML.

**Rejected:** Isaac Sim/Lab (too heavy; occlusion needs ray/polygon, not rigid
bodies) · Sionna *in the training loop* (TF boundary every step; offline
validation only) · stable-baselines3, Ray/RLlib, OmniDrones, SUMO, NS-3 · EW
detectability modelling.

---

## Layout & conventions
```
src/env/       channel, routing, energy, reward, occlusion, batched core
src/baselines/ B0 scripted control + the rollout/metrics harness
src/viz/       shared scene drawing + presentation figures and videos
src/models/    GNN / DeepSets / MLP actors + one shared critic
src/training/  skrl wrappers, curriculum, train.py entrypoint
scripts/       offline data prep + scenario tooling
configs/       YAML per experiment condition
data/          baked artefacts — frankfurt_box.npz IS the frozen environment
               f4_golden.pt.gz IS the pre-Block-F behavioural trace
tests/         cross-module only — unit tests are CO-LOCATED
docs/          reference, read on demand
```
`data/frankfurt_box.npz` is **committed on purpose**, not a build product: OSM and
the LoD2 service both change, so re-running `prep_osm.py` in 2027 would silently
produce a different map. The file in git is the environment; the script only
documents how it was made.
- Unit tests sit next to their module: `src/env/test_channel.py`.
- Naming: `hvt`, `mcv_base`, `sinr_db`, `capacity_mbps`, `edge_weight`,
  `ptx_dbm`. Keep tactical/telecom terms consistent.
- Observations: **24**-dim ego (21 + a persistent 3-dim vector to the cue), 9-dim
  per neighbour, 2-dim per edge; packed flat at `N_max = 8` to **108** dims for
  skrl's rollout storage. Actor stays **agent-local**; global state belongs to
  the critic. **No time feature** — truncation is handled by bootstrapping, not
  by observing the clock ([`docs/BLOCK_D.md`](docs/BLOCK_D.md)).

## Build / test
```bash
uv sync --extra dev                              # `dev` is an EXTRA -- plain
                                                 # `uv sync` gives you neither
                                                 # pytest nor ruff
uv run pytest                                    # 370 tests (+7 CUDA-only skips)
uv run ruff check . && uv run ruff format .
```
Offline data prep (needs network; the artefact is committed, so this is only for
regenerating it deliberately):
```bash
uv run python scripts/prep_osm.py --plot         # bake data/frankfurt_box.npz
uv run python scripts/measure_sightlines.py --plot
```
Offline, no network — regenerates every number Block D's design rests on
(altitude band, sensor envelope, cue staleness, uncued search, link budget):
```bash
uv run python scripts/measure_envelope.py
uv run python scripts/measure_envelope.py --only a2a inside
```
Look at the map before trusting it — every geometry bug so far was found by
happening to compute the right statistic, and the viewer finds the next one in
seconds:
```bash
uv run python scripts/view_episode.py --worst --polygons   # boxes vs source footprints
uv run python scripts/view_episode.py --route 0 --zoom --save ep.mp4
uv run python scripts/bench_occlusion.py                   # re-run this on CUDA
```
Block E. `eval_baseline.py` regenerates every number in `docs/BLOCK_E.md`;
`render_episode.py` is the *presentation* renderer (thesis figures, wandb videos)
as against `view_episode.py`, which is the inspection tool:
```bash
uv run python scripts/eval_baseline.py                     # all sections, 5 seeds
uv run python scripts/eval_baseline.py --only ladder hops
uv run python scripts/render_episode.py --compare --route 12 --video
```
Block F. `calibrate_r.py` produces `R`; `eval_fidelity.py` produces every ladder
table in `docs/BLOCK_F.md`; `--compare-fidelity` draws the same policy at every
rung, which is how the F0 chain running through a tower becomes visible:
```bash
uv run python scripts/calibrate_r.py    --seeds 8 --num-envs 64 --device mps
uv run python scripts/eval_fidelity.py  --seeds 5 --num-envs 64 --device mps
uv run python scripts/render_episode.py --policy b0 --route 12 --compare-fidelity
```
Block G. `measure_potential.py` scores a candidate `Φ` against the states a real
policy visits — run it **before** proposing a reward change, not after the run
comes back a null. `train.py` is one `(fidelity, architecture, seed)` cell;
`eval_policy.py` scores a checkpoint through the *same* `evaluate.py` B0 was
scored with, so the numbers are comparable:
```bash
uv run python scripts/measure_potential.py --policy b0 runs/<name>/checkpoint.pt \
    --device mps --bank-dir .banks     # banks are cached; every candidate reuses them
```
```bash
uv run python -m src.training.train --fidelity F4 --arch mlp --seed 0 \
    --env-steps 10000000 --num-envs 1024 --device cuda
uv run python -m src.training.train --stage 1 --env-steps 4000000 --device mps  # the toy gate
uv run python scripts/eval_policy.py runs/F4-mlp-s0/checkpoint.pt \
    --policy random b0 --n 3 5 8 --seeds 5 --device cuda
uv run python -m src.training.train --phi v2 --fidelity F4 --arch gnn --seed 0 \
    --env-steps 12000000 --num-envs 4096 --device cuda   # the rebuilt potential
```
**`--device mps` is ~17× on Apple silicon** (5 min against ~1.5 h) and the
physics is identical — but it draws *different episodes* for the same seed, so
never mix devices within a comparison. `scripts/capture_f4_golden.py` re-freezes
the pre-Block-F trace and refuses to run without `--force`; read its docstring
first.
