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

⛔ Both Gate 1 candidates failed their pre-declared rules — recurrence
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

### Gate 2 — the role-emergence probe

The reframe above is a hypothesis with one cheap, PBRS-safe test. `REWARD.md`
rules out per-drone potentials on the grounds that they cluster the swarm onto the
HVT — but that was **reasoned, not measured**, and Devlin & Kudenko (2011) extend
the PBRS invariance to the multi-agent case, so a per-agent `Φ_i` cannot move the
equilibrium either.

Declare the rule before running it, as always. Draft — to be fixed before the
first run:

> Per-drone `Φ_i` giving each drone its *own* approach/observe potential, crossed
> against the shipped team `Φ`, GNN, `deep`, 5 seeds, train split.
> * **keep** if `hop_mean | observed` ≥ 2.1 (i.e. the chain becomes non-random)
>   **and** capable ≥ 41.2 %
> * **kill** if the swarm clusters — `nearest_dist_m` falls while `hop_mean` does
>   not rise, which is exactly the failure `REWARD.md` predicts
> * ⚠️ report `observer_dist_m` spread across drones: role emergence should show
>   as **one** drone close and the rest back, not five at the same radius

## 5. ⛔ Stopping rule — 2026-12-31

> If recurrence, `w_hold` and the `Φ_link` term have **each** been measured at
> ≥5 seeds and the best full-mission result is still below B0: **stop
> optimising.** Freeze on schedule, re-scope B0 per §2, and promote the tenure
> diagnosis to a reported finding.

Three months of slack before the freeze, and it converts an open-ended question
into a bounded one — which is what stops it consuming the thesis window.

---

## 6. Phase 2 — Sep–Dec 2026

None of these depend on Gate 1.

| task | why it cannot wait | when |
|---|---|---|
| ⚠️ **Close both `TODO(verify)` sets** | see §7 — everything downstream re-derives if wrong | Sep |
| Diagnose the seed spread | 📏 60–78 % over five stage-1 runs, bimodal, undiagnosed. Every tuning decision is read through it | Sep |
| **Gate 2 — the per-drone `Φ_i` probe** | Gate 1's reframe says the deficit is role emergence; this is its cheapest direct test (§4) | **Sep** |
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
