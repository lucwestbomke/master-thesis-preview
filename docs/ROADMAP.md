# Roadmap — the red thread

One page that connects **why the thesis exists** → **what must be proven** →
**which experiment proves it** → **what software that needs** → **which block
builds it** → **which chapter it becomes**.

If you are working on a task and cannot trace it upward through this chain, stop
and ask why you are doing it.

Research design detail: [`THESIS_PLAN.md`](THESIS_PLAN.md).
Current state and rules: [`../AGENTS.md`](../AGENTS.md).

---

## The claim

> **Multi-agent RL for UAV communication swarms is usually trained against a
> connectivity-radius abstraction. Policies learned that way fail under a
> physically realistic channel — and occlusion is the effect responsible.**

Falsifiable both ways. If radius-trained policies transfer fine, that is also a
result, and a useful one: it tells the field the abstraction is safe.

### Why it is worth making
The gap is real and checkable in the literature: swarm-comms MARL abstracts the
channel, while joint trajectory-and-power work is single-UAV or centralised
optimisation without buildings. Nobody has measured **what the abstraction
costs**. The answer is directly actionable — it tells people which physics their
simulator must include.

---

## The chain

| The claim needs… | …which requires… | …delivered by |
|---|---|---|
| A reference channel nobody can dismiss | standards-based path loss, SINR, rate, routing, energy — all tested | **A** ✅ |
| Occlusion that is real, not a parameter | actual Frankfurt footprints and heights | **B** ✅ |
| Occlusion computed fast enough to train against | batched torch ray-vs-box, 2.5D | **C** ✅ |
| Enough samples to make 45 runs affordable | batched env at ≥1000 steps/s | **D** |
| Proof MARL earns its keep | a scripted geometric baseline (B0) | **E** ✅ |
| The independent variable itself | F0–F4 as config flags on one env | **F** |
| Policies to compare | MAPPO + a curriculum that actually learns | **G** |
| A channel model a telecoms examiner accepts | offline Sionna agreement plot | **H** |

```mermaid
flowchart TD
    A["A · physics<br/>channel · routing<br/>energy · reward"]
    B["B · geometry<br/>buildings · roads"]
    C["C · occlusion<br/>batched slab"]
    D["D · env core<br/>GATE: 1000 steps/s"]
    E["E · renderer<br/>+ B0 baseline"]
    F["F · fidelity F0-F4"]
    G["G · MAPPO<br/>+ curriculum"]
    H["H · Sionna check"]
    R1["RQ1 · fidelity ladder"]
    R2["RQ2 · architectures"]
    R3["RQ3 · handoff"]

    A --> D
    B --> C
    B --> D
    C --> D
    D --> E
    D --> F
    F --> G
    E --> R1
    G --> R1
    G --> R2
    G --> R3
    A -.-> H
```

**Critical path: B → C → D → F → G.** Everything else hangs off it.
**E** can run alongside D once the env steps. **H** is fully parallel and touches
only the methodology chapter.

---

## Blocks

Each gets a full spec when it becomes *next* — written just-in-time, because a
spec written six months early goes stale. [`BLOCK_B.md`](BLOCK_B.md),
[`BLOCK_C.md`](BLOCK_C.md) and [`BLOCK_D.md`](BLOCK_D.md) are the record of what
was measured and decided.

| Block | Delivers | Serves | Gate | Fails if |
|---|---|---|---|---|
| **A** ✅ | channel, routing, energy, reward — pure, batched, tested | all | 103 tests, hand-computed | — |
| **B** ✅ | Frankfurt buildings + road graph as tensors; route sampler | RQ1 (occlusion is the hypothesis), RQ2 (2nd city), RQ3 (sightlines cause handoff) | height coverage verified, not assumed → **LoD2, 100 %** | heights are missing → the map is useless and the scenario is unfounded |
| **C** ✅ | batched segment-vs-**oriented**-box occlusion, 2.5D | RQ1 (the F1 rung *is* occlusion) | matches a slow shapely reference on random geometry ✅ | too slow → blows D's throughput gate. Fusion via `torch.compile` is what makes it viable |
| **D** ⬅️ | batched env core + PettingZoo adapter | everything | **≥1000 env-steps/s on GPU** (transitions, not batched calls) and **≤3 h per 10 M-step run end-to-end** | below gate → 45 runs unaffordable, matrix must shrink |
| **E** ✅ | renderer + B0 scripted heuristic | sanity floor for every RQ; all figures and videos | B0 completes an episode on video | no B0 → cannot answer "is MARL needed at all?" |
| **F** | F0–F4 as config flags on one env | **RQ1 directly** | all five run; `R` calibrated under F4 | uncalibrated `R` → RQ1 comparison is meaningless |
| **G** | MAPPO + curriculum | everything | one toy run beats random | nothing learns → the usual place projects stall |
| **H** | offline Sionna agreement plot | methodology credibility | plot exists | — (optional, cut freely) |

**Hand to an agent:** B, C, E — well-specified and testable.
**Do yourself:** A, D, G — judgement calls and the throughput gate.

---

## From work to chapters

The final deliverable is a written document. Mapping blocks and experiments onto
chapters shows what can be written *early*:

| Chapter | Fed by | Writable |
|---|---|---|
| 1 Introduction | the claim above | last |
| 2 Background & Related Work | literature | **now** — independent of all code |
| 3 System Model | **A, B, C, H** + `PHYSICS.md` | **now, once B lands** |
| 4 Method | **F, G** + `REWARD.md`, `MODELS.md`, `ENVIRONMENT.md` | after the env freeze |
| 5 Experimental Design | `THESIS_PLAN.md` §3–4 | after the env freeze |
| 6 Results | E1 / E2 / E3 / E4 | Apr–Jun 2027 |
| 7 Discussion | + `NEGATIVE_RESULTS.md` | Jun–Jul 2027 |
| 8 Conclusion & Future Work | — | last |

> **Chapters 2 and 3 are ~40 % of the page count and neither depends on a single
> training run.** The physics is frozen and tested; `PHYSICS.md` and
> `NEGATIVE_RESULTS.md` are already most of Chapter 3's substance and a chunk of
> Chapter 7. Writing them during Phase 0, while builds and pilots run, is the
> single biggest scheduling win available — and it is why the environment freeze
> at end of March matters so much: **after the freeze, Chapters 3–5 cannot change.**

---

## Where we are

```
A ████████████████████  done   physics, tested, frozen
B ████████████████████  done   geometry baked → data/frankfurt_box.npz
C ████████████████████  done   occlusion, validated + benchmarked
D ██████████████████░░  built  env core + adapter + skrl seam; gate met 3170x
                              (CUDA re-run of the full env still pending)
E ████████████████████  done   B0 = 57.2 %; renderer; rate target 5 -> 15 Mbps
F ░░░░░░░░░░░░░░░░░░░░  next   F3->F4 is a LARGE effect (+26.5 pp), not a null
G ░░░░░░░░░░░░░░░░░░░░         ← the usual place projects of this shape stall
H ░░░░░░░░░░░░░░░░░░░░
```

**Chapter 3 is now writable.** `PHYSICS.md` was already most of its substance;
Block B replaced its remaining assumptions with measurements (canyon ratio,
sightline distribution, observation envelope) and produced the box figure. That
is the single biggest scheduling win available before the March 2027 freeze.

**Block E changed a settled parameter, and it changed the shape of the whole
experiment.** B0 — the like-for-like scripted control, on the same observation
the actors get — exposed that at the original **5 Mbps** requirement the radio
link never bound: the chain carried 8× the bar, `mission_capable` was identical
to `observed` for every policy, and a script scored 93 % with the metric
saturated. **The rate requirement is now 15 Mbps** and the experiment is properly
shaped. Detail in [`BLOCK_E.md`](BLOCK_E.md), routed through
[`DECISIONS.md`](DECISIONS.md):

1. **The relay chain is now the binding constraint.** B0 reaches 57.2 %
   mission-capable against a 93.0 % *sensor* ceiling — a 36-point gap that is
   pure relay geometry, and did not exist at 5 Mbps.
2. **Difficulty is now at the end of the episode**, where the escalation lives
   and where γ = 0.997 reaches: capable peaks at 84 % around t = 40 s and decays
   to 35 % by t = 240 s, while observed holds at 98 %.
3. **N-scaling is monotone** — 36.4 / 57.2 / 74.3 % at N = 3/5/8, all of it the
   chain. RQ2's off-N columns now measure something — and **the weight belongs at
   N = 8, not N = 3**: control is worth +25.9 pp there against +3.2 pp at N = 3.
   Hardness is not headroom.
4. **RQ3 is re-pointed** from observer handoff (~0.9 per episode, too thin) to
   **relay-chain reconfiguration** (~52 per episode), which is driven by the
   occlusion physics RQ1 studies. E3a's ablation changes with it.
5. **F3 → F4 is a large effect** (+26.5 pp), not the null predicted at 5 Mbps.
6. **The altitude ceiling is no longer derived from W1** — W1 now holds at every
   altitude. 80 m stands on A2A-occlusion grounds instead. Chapter 3 must say so.
7. **Control beats information decisively.** `geodesic` → `B0` is +10.1 pp;
   `B0` → `B0-oracle` is −0.4 pp, i.e. perfect target knowledge is worth nothing.

⚠️ **Every Block D number is at 5 Mbps and is not comparable to a Block E one.**

**Now → Feb 2027:** Phase 0. Build B–H. Write Chapters 2 and 3 in parallel.
**End Mar 2027:** environment freeze. Pilots before, thesis material after.
**Apr–Aug 2027:** run the 45 reported experiments, write 4–8.

---

## Two things that decide whether this works

**D's throughput gate.** At ≥1000 **env-steps/s** — one environment advancing one
tick, summed over the batch, *not* one batched call — the 45-run matrix costs
~120 GPU-hours and everything in `THESIS_PLAN.md` §3 is affordable. Below it, the
matrix has to shrink and RQ2 is the first thing cut. Measure it before building
on top of the env — not after.

The unit was ambiguous in this repo and is now settled in
[`BLOCK_D.md`](BLOCK_D.md). Under it the floor clears easily, so the number
actually reported is **wall-clock for a 10 M-step run end-to-end including the
learner, target ≤3 h**. `torch.compile` is mandatory regardless.

**G's curriculum.** Getting MAPPO to learn anything at all is the classic failure
point for projects of this shape. Budget real calendar time, and remember the
curriculum must be *identical across all fidelity conditions* or RQ1 is
confounded ([`ENVIRONMENT.md`](ENVIRONMENT.md) → Curriculum).

If either slips badly, the honest fallback is to cut RQ2 and RQ3 and defend RQ1
alone. RQ1 is the contribution; the other two are supporting evidence.
