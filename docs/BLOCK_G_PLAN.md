# Block G — the plan to the freeze

**Drafted 2026-08-25** on `block-g-mappo` @ `950ef35`. Presentation version:
<https://claude.ai/code/artifact/71e14261-f31b-4f9b-a2ba-2d5c354f2f8a>

`BLOCK_G.md` records *what happened*. This file records *what happens next*, in
order, with the decision rules declared **before** the runs that resolve them —
because two claims in this block were already overturned by reading a single run,
and a rule invented after the fact is not a rule.

Tags follow `AGENTS.md` → *How to read these documents*: 📏 measured ·
🔒 constraint · 🔧 provisional.

---

## 1. Position

**Updated 2026-08-25 after Gate 1 and sweep stage B.** 📏 Eval split, F4, stage 4,
CUDA, 5 seeds, one harness — the first fully comparable table this block has had:

| policy | capable | observed | tenure | hops |
|---|---|---|---|---|
| random | 10.7 % [0.2] | 21.9 % | 16.3 | 0.4 |
| MLP | 31.2 % [1.2] | 53.8 % | 34.1 | 1.00 |
| DeepSets | 38.1 % [1.0] | 65.3 % | 41.6 | 1.23 |
| **GNN** | **41.2 % [3.8]** | 66.5 % | 47.2 | 1.27 |
| **B0** | **57.3 % [3.9]** | **92.8 %** | **294.7** | **2.1** |

**Gap 16.1 pp.** The learned policy closes 65 % of the random→B0 distance and
stops. ⚠️ The old 45.1 % headline was the winner's curse on a noise axis;
`shipped` at 5 seeds gives 40.7 %.

### The diagnosis, reframed by Gate 1

| conditioned on holding a sightline | random | GNN | B0 |
|---|---|---|---|
| `capable / observed` | 0.489 | **0.620** | **0.617** |
| `hop_mean / observed` | **1.83** | **1.91** | 2.26 |

📏 **Given a sightline, the GNN converts it exactly as well as B0.** The whole gap
is `observed`. And conditioned on observing, learned chain structure is
**indistinguishable from random** — the swarm learned to fly at the target and
nothing about relaying.

🔍 **One mechanism fits both.** B0 parks its observer at **79 m** and *therefore*
needs 2.1 hops; the learned policies loiter at **291 m** on 1.27 hops. Going in
close is only survivable if teammates relay behind you, so with a shared team
reward and no per-drone role signal **no drone can afford to be the one that goes
in.** A coordination trap. The deficit is **role emergence**; observer tenure and
the missing chain are its symptoms.

⛔ **Four interventions, four nulls** — recurrence, `w_hold`, the per-drone
`w_relay` potential, and the agent-specific critic (which actively hurt). All
pre-declared, all at 5 seeds, all in `DECISIONS.md`.

☠️ **And `hop | observed`, the statistic three of them were judged on, measures
geometry.** Random 1.83, every learned policy 1.86–1.93, B0 2.26 — hop count is
set by where the *observer* stands against `R` = 524 m. **There is no separate
relay-role failure.** `observed`, tenure and hop count are three views of one
thing: the observer does not close.

🔍 **Closing is a coordination trap.** Holding a sightline from 79 m puts the
observer ~920 m from the MCV, beyond its own link range — so it only pays once
the rest of the swarm has extended the chain to meet it, and a relay gains
nothing by moving out first. **Every unilateral deviation is worse than the joint
move**, which is why four instruments that each change *one* agent's incentive or
capacity all returned nulls.

📏 **Newly measured (stage 4, B0/random reference):** B0's observer holds
**87.3 m [0.5]** — and **80.2 m** in the last third, so it closes and *stays*
closed. Random sits at 319 m. Even B0's `capable` falls to **40 %** in the final
third against 56.4 % overall, confirming `MODELS.md`'s "difficulty is
concentrated late" — a split this block had never reported.

⛔ The original Gate 1 line, kept for the record: both candidates failed their
pre-declared rules — recurrence
**dropped** (−1.05 pp, tenure 36.8 against a required 95, seed IQR *widened*
4.7 → 6.9), `w_hold` **null** (+1.65 pp on a 6.8 IQR). Recorded in
`DECISIONS.md`.

## 2. Destinations

**D1 — Block G's gate.** A full-mission F4 policy beats **B0 = 57.2 %** on the
**eval** split at ≥5 seeds. The eval-split number for a learned policy does not
yet exist; that is sweep stage B.

**D2 — the freeze, 2027-03-31.** Env, curriculum and reward frozen; every
`TODO(verify)` closed; `configs/` complete; the freeze recorded in
`DECISIONS.md`.

**D3 — the thesis, Mar–Aug 2027.** The 45 reported runs answer RQ1/RQ2/RQ3.

### ⚠️ D1 is **not** required for D2 or D3

RQ1 measures relative degradation across F0–F4 under F4 evaluation; RQ2 compares
three architectures and zero-shot swarm sizes; RQ3 measures relay-chain
reconfiguration. **All three are relative comparisons across conditions.** None
requires exceeding B0.

If D1 fails, B0 is re-scoped from *"the floor to match"* to *"the scripted
reference the learned policies are measured against"*, and the tenure diagnosis
becomes a reported finding rather than a failure.

> 🔒 `MODELS.md` rule 4 currently reads *"any architecture must beat a random
> policy and at least match B0. **Failing that is a bug, not a finding.**"* That
> line predates the diagnosis. Using the fallback means **revising it
> deliberately and recording why** — not drifting into it in April 2027 when
> there is no time to argue the point. Hence §5.

---

## 3. ~~Track A~~ — COMPLETE 2026-08-25

Kept for reproduction. A0–A5 all ran; results in §1 and `BLOCK_G.md`.
⚠️ A4 exposed a real bug — stage B appended eval rows unconditionally, so an
interrupted-and-resumed stage B double-counted seeds (`n = 9` where 5 were
asked for). Guarded now, and `scripts/dedupe_summary.py` cleans an affected
file. 📏 On this occasion no reported number moved: the MLP row reads
31.2 % [1.2] before and after the dedupe.

### Original commands

### A0. Sync, and run the suite **on the GPU box**

```bash
git pull && uv sync --extra dev
uv run pytest -q                      # expect 339 passed, 0 skipped
```

⚠️ Not a formality: 7 tests are CUDA-gated and skip on arm64. 📏 The first time
they ran on CUDA they found three bugs, one live since Block D — and new tests
have just landed on that path.

🔒 If `test_golden.py` fails here, **do not act on it.** The frozen trace is
arm64-only; float32 is not associative across instruction sets. Re-run on the Mac
before believing a golden failure.

### A1. Close G1a

```bash
uv run python scripts/bench_env.py --envs 1024 4096 --device cuda
```

The first CUDA session aborted on a `--breakdown` bug, so only the 256-env row of
G1a exists.

### A2. The 2×2 — recurrence × reward, 5 seeds

```bash
# GNN + `deep`: 📏 the sweep's real cadence finding (+6 pp, clean).
# SHIPPED shaping, NOT the winner's `dref400_k30`: 📏 the shaping axis measured
# as noise (1.2 pp spread against 6 pp of within-cell seed range, and the three
# architectures picked three different winners). Carrying it forward would be
# fitting noise AND would stop `--w-hold` being the only reward variable.
BASE="--fidelity F4 --arch gnn --num-envs 4096 --rollouts 64 --mini-batches 32 \
      --env-steps 12000000 --boundaries 0.10 0.20 0.35 --min-std 0.2 --device cuda"
HOLD="--w-hold 0.4 --d-hold 400"

for s in 0 1 2 3 4; do
  uv run python -m src.training.train $BASE --seed $s                   --name g8-ff-shipped-s$s
  uv run python -m src.training.train $BASE --seed $s $HOLD             --name g8-ff-hold-s$s
  uv run python -m src.training.train $BASE --seed $s --recurrent       --name g8-rnn-shipped-s$s
  uv run python -m src.training.train $BASE --seed $s --recurrent $HOLD --name g8-rnn-hold-s$s
done
```

20 runs, ≈70 min. **Crossed rather than run separately** because the two fixes
attack the same deficit by different routes and could interact — and the
interaction is itself informative (see §4).

### A3. Score through the harness B0 went through

```bash
OUT="--device cuda --train-routes --num-envs 128 --out results/g8_gate1.jsonl"

for c in ff-shipped ff-hold rnn-shipped rnn-hold; do
  uv run python scripts/eval_policy.py runs/g8-$c-s*/checkpoint.pt --group "$c" $OUT
done
uv run python scripts/eval_policy.py --policy b0 $OUT
```

🔒 `--train-routes` on purpose: this is a **tuning** decision and the eval split
is not for tuning. B0 paid a measured 0.6 pp for the same restriction.

### A5. Get the results off the box

```bash
git add results/g8_gate1.jsonl && git commit -m "Gate 1 results" && git push
```

`results/` is tracked and `runs/` is gitignored, on purpose: **the summary is a
result, the checkpoints are regenerable.** The pod pushes, the laptop pulls, and
the numbers are versioned with the commit that produced them — no file juggling,
and the provenance cannot drift from the code.

⚠️ `--out` records **per-seed** values, not just `median [IQR]`. Every gate rule
in §4 is declared on the *worst* seed, and a median alone cannot be judged
against one.

If the training curves are wanted too — they are what diagnosed the last three
failures — `runs/*/log.jsonl` is small and gitignored, so copy it deliberately:

```bash
mkdir -p results/g8_curves
for d in runs/g8-*/; do cp "$d/log.jsonl" "results/g8_curves/$(basename $d).jsonl"; done
```

### A4. Sweep stage B

```bash
uv run python scripts/sweep.py --device cuda --stage-b
uv run python scripts/sweep.py --report-only
```

Reuses surviving stage-A checkpoints, so ≈6 new runs rather than 15. The eval
split is touched here **once, for confirmation, never for selection**.

---

## 4. ~~Gate 1~~ — RESOLVED 2026-08-25: both candidates dropped

Kept for the record; the rules were declared before the runs and applied
unchanged. Full tables in `BLOCK_G.md` § *G8*.

| factor | rule | measured | verdict |
|---|---|---|---|
| recurrence | keep if tenure ≥ 95 **and** capable ≥ 45.1 % | 36.8 / 39.7 | ⛔ **drop** |
| `w_hold` | keep if capable +≥3 pp **and** worst seed improves | passed a different half in each arm | ⛔ **null** |

Two things the negative bought, and they are worth more than the gate:

1. 📏 **Conditioned on observing, learned chain structure is random.** 1.86–1.91
   hops against random's 1.83. The relay half of the mission was never learned by
   *any* architecture, with or without memory or shaping.
2. ☠️ **`chain_occluded` confounds with hop count** (`corr = 0.963`). It is RQ1's
   designated failure-attribution metric and it is not usable as defined — see
   `DECISIONS.md`.

## 4b. Gate 2 — role emergence, declared 2026-08-25 before the runs

### What changed the design

The Gate 1 negative said the deficit was role emergence. `scripts/probe_credit.py`
then measured **why nothing had touched it**: the critic is handed one global
state repeated per drone, so `max |V_i - V_j|` is exactly `0.000e+00`, the reward
is team-dominated, and 📏 **0.015-0.06 % of advantage variance distinguishes one
drone from another**. Every drone's gradient is `grad log pi(a_i|o_i) * A` with
the *same* `A`. Role differentiation cannot be learned from a signal that is
constant across the agents it would differentiate — which retro-explains every
null in this block.

Two interventions were built against it, and the probe already separated them:

| | value between-drone | **advantage between-drone** |
|---|---|---|
| shipped | 0.00000 | **0.00041** |
| `--agent-specific-critic` | **0.17527** | 0.00042 |
| `--w-relay 0.2` | 0.00000 | **0.00527** (13x) |
| `--w-relay 0.5` | 0.00000 | **0.02931** (71x) |

⚠️ **The agent-specific critic opens the value channel and does not reach the
gradient.** `V_i(s_t)` is a baseline — it does not depend on `a_t` — and the `V`
half of `delta_i - delta_j` is a potential in `(V_i - V_j)` that telescopes away.
That is exactly why COMA uses a *counterfactual* baseline rather than a per-agent
value function. It ships, because it may become credit once trained through the
bootstrap term `gamma*V_i(s_{t+1})`, but it is the **weaker** arm.

`w_relay` reaches it, because `Phi_i` enters `r_i` and `s'` depends on `a_i`.
📏 Note the GAE attenuation: reward 0.264 → advantage 0.029, ~9x, consistent with
`lambda = 0.95`'s ~19 effective steps. **"PBRS-safe so any scale works" is true of
the optimum and false of the learning signal.**

### The runs

```bash
BASE="--fidelity F4 --arch gnn --num-envs 4096 --rollouts 64 --mini-batches 32 \
      --env-steps 12000000 --boundaries 0.10 0.20 0.35 --min-std 0.2 --device cuda"

for s in 0 1 2 3 4; do
  uv run python -m src.training.train $BASE --seed $s --w-relay 0.2 --name g9-relay02-s$s
  uv run python -m src.training.train $BASE --seed $s --w-relay 0.5 --name g9-relay05-s$s
  uv run python -m src.training.train $BASE --seed $s --w-relay 0.5 --agent-specific-critic \
      --name g9-relay05-asc-s$s
done

OUT="--device cuda --train-routes --num-envs 128 --out results/g9_gate2.jsonl"
for c in relay02 relay05 relay05-asc; do
  uv run python scripts/eval_policy.py runs/g9-$c-s*/checkpoint.pt --group "$c" $OUT
done
```

Control is **`g8-ff-shipped` at 40.7 %**, already run — same cadence, same
shaping, same seeds, so `--w-relay` is the only variable.

### The rule

**Primary readout is `hop_mean | observed`**, not `mission_capable`. 📏 The
measured deficit is that conditioned on observing, learned chain structure is
indistinguishable from random's — **1.86-1.91 hops against random's 1.83**, with
B0 at **2.26**. That is the number the intervention is aimed at, and
`mission_capable` is downstream of it.

| | keep | kill |
|---|---|---|
| `w_relay` | `hop_mean \| observed` ≥ **2.0** *and* `mission_capable` ≥ **40.7 %** (no worse than control) | see the clustering condition below |
| more `w_relay` | 0.5 beats 0.2 on the primary readout | 0.5 is worse — the term is overwhelming the objective's *learning signal*, not its optimum |
| `--agent-specific-critic` | adds ≥ 3 pp on top of the same `w_relay` | within IQR — drop it, it was variance reduction |

⛔ **The kill condition, and it is `REWARD.md`'s own prediction.** If
`nearest_dist_m` falls while `hop_mean | observed` does **not** rise, the swarm
has clustered onto the HVT — which is precisely what `REWARD.md` says per-drone
potentials do. That would mean the objection was right and `on_path` is not the
exception it was argued to be. Record it as such rather than re-tuning around it.

Also report, because they are what a role *looks like* and nothing else in this
block could see one:

* `role_entropy` — 0 = one drone owns the observer role, 1 = all equal.
  📏 B0 **0.0**, learned **0.2**, random **0.5**.
* `relay_entropy` — the same for chain membership. **This is the one that should
  move**; the observer role already partly emerges and the relay role does not.
* `standoff_gap_m` — ⚠️ never read alone. 📏 B0 **88.6**, learned **34.1**,
  random **163.7**: a scattered policy scores the *largest* gap by accident.

⚠️ Judge on the **worst seed**, as always in this block, and on `>= 5` seeds.

## 4c. Gate 3 — the `Φ` rebuild, declared 2026-08-27 **before** the runs

### What changed the design

Gates 1 and 2 proposed mechanisms. This one measured the instrument first.
`scripts/measure_potential.py` banks the states a policy actually visits and
scores any `RewardWeights` over them, and it found the shipped potential moves
**0.320 in total** across the closing band — 0.0133 per 8 m step against the
0.0544 the objective can pay — and is **exactly constant in four drones out of
five**, because every component is a `min`/`max`/routing reduction. Full audit in
`BLOCK_G.md` § G13 and `REWARD.md`.

⚠️ **Two claims this plan was reasoning from are now measured wrong**, and both
are in `DECISIONS.md`: the learned policy is *not* collecting the energy bonus
(it flies at the 25 m/s dash cap on 57 % of steps and pays **more** than B0), and
`Φ` is *loud* rather than quiet (|ΔΦ| p90 is 0.365 for the GNN against B0's
0.052) — its problem is direction, not amplitude.

`Φ v2` is one preset, `--phi v2`, off by default and bitwise-identical to the
shipped potential when off.

### The runs

**Step 1 — the fast loop, one seed, and LOOK at it.** 📏 Four hypotheses drawn
from aggregate statistics went 0-for-4 in this block and one render found what
they all missed, so this step is not optional and its output is a *picture*.

```bash
BASE="--fidelity F4 --arch gnn --num-envs 4096 --rollouts 64 --mini-batches 32 \
      --env-steps 12000000 --boundaries 0.10 0.20 0.35 --min-std 0.2 --device cuda"

uv run python -m src.training.train $BASE --seed 0 --phi v2 --name g13-v2-s0
uv run python scripts/render_episode.py --policy runs/g13-v2-s0/checkpoint.pt \
    --compare --route 12
```

⛔ **Do not proceed to step 2 on the checkpoint's `mission_capable` alone.** The
question the render answers is whether the drones are *in the corridor* — B0's
tracks stay inside it, the shipped policy's sweep the whole map and pin against
the box boundary. If the arcs and the boundary-pinning are unchanged, `Φ_cover`
did not reach the policy and the weights are the thing to move, not the seeds.

**Step 2 — 5 seeds, with the control already in hand.**

```bash
for s in 0 1 2 3 4; do
  uv run python -m src.training.train $BASE --seed $s --phi v2 --name g13-v2-s$s
done

OUT="--device cuda --train-routes --num-envs 128 --out results/g13_gate3.jsonl"
uv run python scripts/eval_policy.py runs/g13-v2-s*/checkpoint.pt --group v2 $OUT
```

Control is **`g8-ff-shipped` at 40.7 %**, already run — same architecture, same
cadence, same seeds, so `--phi` is the only variable.

### The rule

**Primary readout is `observer_range_m`.** 📏 That is what `Φ_standoff` is aimed
at, `mission_capable` is downstream of it, and this block has twice judged an
intervention on a statistic that turned out to measure geometry. Secondary is
`off_axis_m`, which is what `Φ_cover` is aimed at.

⛔ **Not `hop_mean | observed`** (it measures geometry — random 1.83, every
learned policy 1.86–1.93) and ⛔ **not `chain_occluded`** (it confounds with hop
count, `corr = 0.963`). Both are in `DECISIONS.md`.

| | keep | kill |
|---|---|---|
| `Φ v2` | `observer_range_m` ≤ **140 m** *and* `mission_capable` ≥ **40.7 %** (no worse than control) | `observer_range_m` > 160 m, or capable below control on the **worst seed** |
| `Φ_standoff` vs `Φ_cover` | if kept, isolate them: `--phi v2 --w-cover 0` and `--phi v2 --w-standoff 0`, 5 seeds each | — |
| the weights | — | if `observer_range_m` moves and `off_axis_m` does not, `Φ_cover` is the term that failed, not the design |

**Why 140 m.** 📏 The learned observer stands at **184 m** and B0 at **88.8 m**;
Block B measured the along-street sightline median at **127 m**, and the whole
`observed` gap is that B0 sits inside it and the learned policy outside it. 140 m
is the far edge of "inside the threshold" and is the smallest move that could
plausibly convert the gap. A `Φ` term worth 1.42× the objective's strongest
per-step force that cannot move the number it directly grades has been refuted,
not under-tuned.

⚠️ Judge on the **worst seed**, at ≥5 seeds, and report `role_entropy` alongside —
📏 B0 **0.10**, the GNN **0.50** at stage 4 against random's 0.60. `Φ_cover`'s
marginal value is *how uncovered a point is by everybody else*, so it is the first
term in this block with a mechanism by which `role_entropy` could move at all.

⛔ **The pre-declared failure reading.** If `observer_range_m` moves inside the
threshold and `mission_capable` does **not**, then the stand-off was never the
binding constraint and the `observed` gap has another cause. Record that as the
result — it is a stronger finding than a sixth null, and it would retire the
geometry-threshold diagnosis rather than inviting a seventh intervention.

---

## 5. ⛔ Stopping rule — 2026-12-31

> If recurrence, `w_hold`, the `Φ_link` term **and the `Φ` v2 rebuild (§4c)**
> have **each** been measured at ≥5 seeds and the best full-mission result is
> still below B0: **stop optimising.** Freeze on schedule, re-scope B0 per §2,
> and promote the tenure diagnosis to a reported finding.

⚠️ `Φ` v2 is added to this list rather than restarting the clock. It is the last
*reward-side* candidate: the audit in §4c measured what the potential is worth
and rebuilt it to the bar, so a null here is evidence that shaping is not the
binding constraint, not an invitation to a third potential.

Three months of slack before the freeze, and it converts an open-ended question
into a bounded one — which is what stops it consuming the thesis window.

---

## 6. Phase 2 — Sep–Dec 2026

None of these depend on Gate 1.

| task | why it cannot wait | when |
|---|---|---|
| ⚠️ **Close both `TODO(verify)` sets** | see §7 — everything downstream re-derives if wrong | Sep |
| Diagnose the seed spread | 📏 60–78 % over five stage-1 runs, bimodal, undiagnosed. Every tuning decision is read through it | Sep |
| **Gate 2 — `w_relay`, the per-drone relay potential** | 📏 the only intervention that reaches the per-drone gradient (§4b). Ready to run | **now** |
| `Φ_link` chain-clearance term | Gate 1 said yes — 📏 conditioned on observing, learned chains are *random* | Sep |
| Fix `chain_occluded` to a per-edge rate | ☠️ it confounds with hop count and RQ1 depends on it | Sep |
| Measure, then 🔒 **freeze**, the curriculum schedule | 🔧 `(0.15, 0.35, 0.60)` + 20 % mix is provisional; RQ1 needs it identical across rungs | Oct |
| **G7** — fidelity attribution pilot | do F0/F2/F3-trained policies separate under F4 on hop count, `chain_occluded`, p5 capacity? If not, RQ1's headline survives but its *attribution* does not | Oct |
| Port RNN onto skrl's **current** `PPO` | only if recurrence is kept. Removes "our algorithm runs on a patched stale fork" from the viva | Nov |
| Cadence isolation cell — `1024 × 64 × 8` | `deep` confounds `num_envs` with `rollouts`; `wide` shows `num_envs` alone is harmful. Label it a follow-up, **not** part of the equal-budget claim | Nov |
| `configs/` — one YAML per condition | the matrix must be reproducible from config, not shell history | Nov |

---

## 7. ⚠️ The largest unpriced risk

`channel.py:114` carries a live marker:

> *"TODO(verify): these coefficients must be checked against the actual 3GPP
> TR 36.777 document (UMi-AV table) before anything derived from them appears in
> the methodology chapter."*

They were AI-produced. They are self-consistent and physically sensible. **Nobody
has opened the spec.** 🔒 `AGENTS.md` has a ⛔ for exactly this.

Every number in this project — B0's 57.2 %, the whole fidelity ladder, the
altitude band, `R` = 524 m — sits downstream of that table. If a coefficient is
wrong, `data/f4_golden.pt.gz` legitimately re-captures (the one sanctioned reason)
and **every measured number re-derives.**

Which is why it happens in September and not in March. Cost if right: an
afternoon. Cost if wrong and found late: the thesis. `energy.py` has four more
markers — lower stakes, since energy sets the observer's *cost* rather than the
mission metric, but the same afternoon.

---

## 8. Phases 3 and 4

**Jan 2027** — run the whole matrix end-to-end at **one** seed. Not for results:
to find the plumbing failures that only appear at 45-run scale, while there is
still time to fix them.

**Feb 2027** — Block H (Sionna offline validation) if §7 raised any doubt about
the channel model. Fully parallel, genuinely optional, much more valuable if the
coefficients needed correcting.

**Mar 2027** — 🔒 **freeze.** Record it in `DECISIONS.md` with the date and scope.
Re-capture `f4_golden.pt.gz` *if and only if* a verified constant changed — never
to make a test pass.

**Apr–Aug 2027** — the 45 reported runs (~3 GPU-hours), then RQ2's zero-shot
transfer at `N ∈ {3,5,8}` (no new training; analytical weight at N = 8, where
📏 better control is worth +25.9 pp against +3.2 pp at N = 3), then RQ3's
reconfiguration analysis, then writing.

---

## 9. Risk register

| risk | blast radius | mitigation, and when |
|---|---|---|
| **TR 36.777 coefficients wrong** | total — every number re-derives | verify in September (§7) |
| nothing ever beats B0 | contained — the RQs are relative and survive | dated stopping rule (§5) |
| F0/F2/F3 do not separate under F4 | RQ1's headline survives, its attribution does not | G7 pilot in October surfaces it 6 months early |
| seed spread swamps every effect | all tuning decisions become unreadable | diagnose in September; judge on the worst seed meanwhile |
| more stale-fork bugs in `ppo_rnn.py` | silent, and that file has produced two already | port to current `PPO`; diff against `mappo.py` before trusting anything inherited from it |
| **calendar, not capability** | the actual binding constraint | 📏 compute is free (~2 GPU-h for the entire matrix) — protect thinking time, not GPU time |

---

## The one-sentence version

Run the 2×2 and read observer tenure; verify the 3GPP table this month because
everything rests on it; and keep the date on §5 so "will it ever beat B0" cannot
quietly consume the thesis window.
