# Thesis Plan

**Working title:** Interference-Aware Joint Motion and Transmit-Power Control for
Multi-Hop UAV Relay Swarms in Contested Urban Environments

**Degree:** MSc Artificial Intelligence
**Official window:** March 2027 – August 2027 (~5 months)
**Preparation window:** now – March 2027 (unpressured, part-time)

---

## 1. The mission — what the swarm is actually for

A ground **High-Value Target (HVT)** vehicle drives through a dense city along the
real road network. A **Mobile Command Vehicle (MCV)** sits at a fixed position
somewhere else in the city. Buildings block line-of-sight between them, so the MCV
cannot see or hear the HVT directly.

A swarm of `N` quadrotors must, continuously and simultaneously:

1. **Observe** — at least one drone must hold the HVT inside its sensor envelope:
   within range `D_max`, inside a downward Field-of-View cone `θ_max`, and with an
   unoccluded ray (no building between drone and vehicle).
2. **Relay** — the resulting sensor feed (an EO/IR video stream) must reach the MCV
   at **≥ 5 Mbps end-to-end**. No single drone can usually see both the HVT and the
   MCV, so this requires a **multi-hop chain**: observer → relay → relay → MCV.
3. **Survive** — each drone has a finite battery. The HVT carries a roof-mounted
   jammer that travels with it, raising the noise floor for any drone near it.

The episode ends in failure if the mission link drops below 5 Mbps for more than 5
consecutive steps, or if any drone's battery reaches zero.

### Why this is genuinely hard

Three tensions make it a real coordination problem rather than a formation-flying
exercise:

**Tension 1 — observe vs. survive.** To observe the HVT you must be close to it.
Close to the HVT is exactly where the jammer is loudest. The best observation
position is the worst communication position.

**Tension 2 — reach vs. relay quality.** The observer is deep in an urban canyon,
far from the MCV. Its data must be carried by others. Adding hops extends reach but
costs throughput: a half-duplex chain of `n` hops delivers at most
`min_i(C_i) / n`. So the swarm must find the *shortest chain that still clears the
buildings* — a geometric problem defined by the actual OSM footprints.

**Tension 3 — loud vs. quiet (the core of RQ1).** Each drone chooses its own
transmit power. Transmitting louder improves *your* link but raises the interference
floor for *every other hop in the chain*. Power is therefore not a private energy
dial — it is a shared resource that must be allocated spatially across the swarm.
This is what makes joint power control a multi-agent problem and not a per-agent
optimization.

### The behaviour we expect to emerge

All drones run the **same** policy (homogeneous, CTDE). Nothing in the architecture
assigns roles. But the task rewards differentiation: one drone burning energy chasing
the HVT, others holding cheap station-keeping positions at street intersections where
they have LoS to both neighbours. The battery-variance penalty `-λ·Var(B)` then makes
a *fixed* division of labour unsustainable — the tracker drains faster, variance
grows, and the swarm is pushed to **rotate roles**, handing tracking duty to a fresher
drone. Demonstrating that this rotation emerges from a homogeneous policy, and that it
is caused by the variance penalty, is RQ3.

---

## 2. Research questions

### RQ1 (primary) — Does joint motion + power control beat motion-only control?

> Under an equal total energy budget, does a policy that jointly controls kinematics
> and transmit power outperform a motion-only policy with fixed transmit power, in a
> multi-hop relay task with intra-swarm interference?

**Hypothesis:** Yes — and the mechanism is **interference management**, not energy
saving. The joint policy will learn to attenuate transmitters whose signal reaches
receivers they are not serving, raising end-to-end capacity at equal energy cost.

This mechanistic claim is what elevates the thesis above a benchmark table. It is
falsifiable and separately measurable (see §4, mechanism metrics). Note the
consequence: **if the joint policy wins but shows no interference-reduction
signature, the hypothesis is wrong even though the headline number is favourable.**
Report that honestly if it happens.

**Why the framing changed.** The original plan framed transmit power purely as an
energy term. At the original 30 dBm ceiling that term is ~1.6 % of the power budget
(hover dominates by ~60×) — too small to resolve against seed variance. Two fixes,
both applied: the ceiling rises to 40 dBm (10 W, realistic for a tactical MANET
radio, ~14–17 % of draw), **and** intra-swarm interference is modelled, which makes
power control matter for reasons independent of the energy arithmetic.

### RQ2 (secondary) — Does relational structure help, and does it transfer across swarm size?

> Does a GNN actor/critic outperform permutation-invariant but non-relational
> architectures, and does it enable zero-shot transfer to unseen swarm sizes?

Three architectures form a ladder that isolates **one factor at a time**:

| Architecture | Permutation-invariant | Size-agnostic | Uses link structure |
|---|---|---|---|
| Flat MLP (max-N padded + masked) | ✗ | ✗ (padding only) | ✗ |
| DeepSets (mean-pool over neighbours) | ✓ | ✓ | ✗ |
| **GNN (capacity-weighted edges)** | ✓ | ✓ | ✓ |

MLP → DeepSets isolates permutation invariance. DeepSets → GNN isolates *who can
talk to whom* — the actual research claim. Comparing a GNN only against a flat MLP
conflates the two and is the weaker experiment.

Trained at `N=5`, evaluated zero-shot at `N ∈ {3, 5, 8}`. The MLP requires max-N
padding and masking to be evaluable at all off-N; without this the comparison is
rigged in the GNN's favour and an examiner will say so.

**Honest expectation:** at N=5 the *in-distribution* gap between DeepSets and GNN may
be within seed noise. That is fine — the interesting result lives in the off-N
transfer column, and a null in-distribution result reported cleanly is still a
contribution.

### RQ3 (tertiary) — Does role specialization and rotation emerge, and what causes it?

> Does a homogeneous policy spontaneously differentiate into tracker/relay roles, and
> is the battery-variance penalty necessary for role *rotation*?

Ablation over `λ ∈ {0, λ*}`. Measured by role-switch frequency, terminal battery
variance, and episode length. This is the qualitative/behavioural chapter and the
source of the best figures. **First thing to cut if time runs short.**

---

## 3. Conditions and baselines

RQ1 stands or falls on baseline fairness. The most common way this kind of result
gets destroyed in a defence is "you compared against a straw man."

| # | Condition | Purpose |
|---|---|---|
| B0 | **Scripted geometric heuristic** — relays evenly spaced on the MCV→HVT geodesic, one tracker, fixed Ptx | Non-learned control. Answers "is MARL earning its keep at all?" Cheap to build, disproportionately valuable. |
| B1 | **Motion-only MARL, fixed Ptx**, swept over `{20, 25, 30, 35, 40}` dBm | The RQ1 baseline. **Report the best-performing fixed value**, not an arbitrary one. |
| E1 | **Joint motion + power MARL** | RQ1 treatment. |
| E2 | E1 × {MLP, DeepSets, GNN} × N ∈ {3,5,8} | RQ2. |
| E3 | E1 with `λ = 0` | RQ3 ablation. |

**Equal energy budget** is enforced by giving every condition the same initial battery
and the same energy model — conditions differ only in what the policy may control.

**Compute estimate:** B1 needs 5 Ptx values × 3 seeds = 15 short runs (sweep, not full
length). E1/E2/E3 need 5 seeds each. Total ≈ 45–60 runs. At ~10 M steps and a target
of ≥1000 env-steps/s batched, that is roughly 150–250 GPU-hours — comfortably within a
few hundred dollars of RunPod. **The throughput target is the gate; measure it early.**

---

## 4. Metrics — pre-registered before any results are seen

Fixing these now protects against unconsciously selecting the metric that flatters the
outcome.

**Primary (RQ1 headline):**
- **Energy per successful tracking-second** — joules consumed per second during which
  the HVT is observed *and* the mission link is ≥ 5 Mbps. Lower is better. This single
  number captures the whole trade-off.

**Mission outcome:**
- Episode length / survival rate
- Link-alive fraction (share of steps with end-to-end ≥ 5 Mbps)
- Tracking coverage fraction (share of steps with a valid HVT observation)
- Mean and 5th-percentile end-to-end capacity

**Mechanism (these test RQ1's *causal* claim, not just its outcome):**
- Total intra-swarm interference power received across the chain
- Correlation between a drone's chosen Ptx and its distance to the nearest
  non-served neighbour — the signature of learned spatial power allocation
- Mean chain hop count and mean per-hop margin above threshold

**Behavioural (RQ3):**
- Role-switch count per episode (role = argmax observation quality)
- Terminal battery variance across the swarm

**Reporting:** ≥5 seeds per condition; report median and interquartile range, not
mean ± std (RL returns are not normally distributed). Show per-seed learning curves
in the appendix, never just the aggregate.

---

## 5. Model specification — what changed and why

Full formal spec lives in [`AGENTS.md`](../AGENTS.md). Summary of corrections made
against the original plan:

| Item | Was | Now | Why |
|---|---|---|---|
| SINR | `P_sig − (P_jam + N0)` in dB | Linear-domain sum of interference + noise, then convert | Adding dBm values is a product, not a sum. Original produced ~+100 dB SINR — physically impossible, and it silently deletes the jammer from the experiment. |
| Path loss | 3GPP TR 38.901 UMi for everything | TR 36.777 UMi-AV for air-to-ground; FSPL + blockage for air-to-air | 38.901 UMi is specified for UE heights 1.5–22.5 m. It is not valid for aerial nodes, and it is doubly invalid for drone-to-drone links above rooftop. |
| Interference | Jammer only | Jammer + all concurrent friendly transmitters | Five drones at up to 10 W on a shared band interfere with each other far more than the jammer does. Without this, RQ1 has no mechanism. |
| Multi-hop capacity | Undefined | `min_i(C_i) / n_hops`, path chosen by hop-limited widest-path DP | The entire premise is a relay chain; it was never specified how the path is chosen or how end-to-end rate is computed. |
| Ptx range | 0–30 dBm | 0–40 dBm | At 30 dBm the telecom term is 1.6 % of power draw — unmeasurable. |
| Noise floor | Hardcoded −100 dBm | `−174 + 10log10(B) + NF` | Must track bandwidth. At B=10 MHz, NF=7 dB → −97 dBm. |
| Capacity | Unbounded Shannon | `min(0.75·log2(1+SINR), 7.4)` b/s/Hz | Shannon is an upper bound; real NR caps at 256QAM. Unbounded Shannon reports fantasy throughput at high SINR. |
| Bandwidth | 20 MHz | 10 MHz | At 20 MHz, 5 Mbps needs only −7.2 dB SINR — the constraint never binds and the jammer becomes decorative. At 10 MHz over a 3-hop chain it needs ≈ +4.8 dB per hop. Properly contested. |
| Energy | `P_hover + α‖v‖² + β‖a‖²` | Rotary-wing model (Zeng et al. 2019) + explicit control-effort term | Quadratic-in-speed says hovering is cheapest. Real rotary-wing power is U-shaped with a minimum near 10–15 m/s. RQ1 is an energy claim; it cannot rest on an energy model that rewards hovering when reality doesn't. |

> ⚠️ **Verify before citing.** The TR 36.777 UMi-AV coefficients in
> [`src/env/channel.py`](../src/env/channel.py) were written from memory and are
> marked `TODO(verify)`. Pull the actual 3GPP document and check them against
> Table B-2 before any of this reaches the methodology chapter. Same for the
> Zeng et al. rotary-wing constants. Do not cite numbers an AI gave you.

---

## 6. Architecture decision: batched env from day one

Verified against the installed stack: **skrl's `PettingZooWrapper` round-trips every
action and observation through NumPy on every step** (`untensorize_space` /
`tensorize_space`), and exposes `num_envs == 1`. skrl's vectorized paths are Isaac
Lab-only. So the PettingZoo dict API directly contradicts the project's own
"everything stays in VRAM" rule and caps throughput at single-env Python speed.

**Decision:** the env core is written as a **batched tensor env with a leading
`num_envs` dimension**, with:
- a thin PettingZoo `ParallelEnv` adapter on top, used only for API-compliance tests
  and single-env visual debugging;
- a small custom skrl multi-agent wrapper written against the batched core for
  training, bypassing `PettingZooWrapper` entirely.

Same rule for geometry: `osmnx`/`shapely` are **offline-only**, used in
`scripts/prep_osm.py` to bake buildings into a tensor of boxes. Runtime occlusion is
vectorized segment-vs-box intersection (slab method) in pure torch.

---

## 7. Timeline

The preparation window is the single biggest advantage available. Goal: **enter March
2027 with a finished, tested, benchmarked simulator**, so the official five months are
experiments and writing only.

### Phase 0 — Preparation (now → Feb 2027, part-time)

| Block | Deliverable | Done when |
|---|---|---|
| A | Channel model + routing, unit-tested | Hand-computed link budgets pass as assertions ✅ *(done — see §8)* |
| B | OSM pipeline (`scripts/prep_osm.py`) | Buildings + road graph cached as tensors for one real city district |
| C | Occlusion (batched torch slab method) | Matches a slow shapely reference implementation on random geometry |
| D | Batched env core + PettingZoo adapter | Random policy runs; **≥1000 env-steps/s measured on GPU** |
| E | Renderer + B0 scripted heuristic baseline | Video of the heuristic completing an episode |
| F | MAPPO integration + curriculum | One toy run learns *something* above random |
| G | Sionna offline validation of the closed-form channel | Agreement plot for the methodology chapter |

Blocks B/C/E are the ones to hand to an AI agent in small chunks — they are
well-specified and testable. Blocks A/D/F deserve your own attention.

### Phase 1 — Official thesis (Mar → Aug 2027)

| Month | Focus |
|---|---|
| Mar | Curriculum tuning until E1 trains reliably. Freeze the env — **no model changes after this point.** |
| Apr | B1 Ptx sweep + E1 at 5 seeds. First RQ1 answer. |
| May | E2 architecture ladder + N-transfer. Write Methodology chapter (the model is frozen, so this is safe to write now). |
| Jun | E3 ablation, mechanism analysis, figures. Write Results. |
| Jul | Discussion, related work, introduction. Buffer for reruns. |
| Aug | Revisions, defence prep. |

**Hard rule:** freeze the environment at the end of March. Every result before the
freeze is a pilot; every result after is thesis material. Mixing them is how these
projects lose a month re-running everything.

---

## 8. Risks and mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| **Env throughput too low** — the project's main failure mode | High | Batched design from day one. Benchmark in Block D; if <1000 steps/s, cut N, cut building count, or profile before proceeding. Do not start Phase 1 without hitting this. |
| **Nothing learns** — sparse reward + harsh termination | High | Curriculum: stationary HVT → slow HVT → jammer off → jammer on → full speed. Budget real time for this; it is the usual place these projects stall. |
| **RQ1 effect is null** | Medium | Already mitigated (Ptx ceiling + interference). If still null, the mechanism metrics let you write an informative negative result rather than a hole. |
| **Variance penalty has a degenerate optimum** (all drones hover, variance = 0) | Medium | Verify tracking/capacity terms dominate; consider potential-based shaping so the penalty doesn't distort the optimal policy. |
| **Reward-weight sweep explodes** (6 free weights) | Medium | Do not sweep 6 weights. Fix α, β, ω, γ, threshold from physical reasoning and document the choice. Sweep λ only. |
| **Scope creep** | Medium | RQ3 is the designated cut. EW/EMCON detectability is explicitly **out of scope** — noted as future work only. |

---

## 9. Deliberately out of scope

- **EW detectability / EMCON** (probability of adversary geolocation as a function of
  Ptx). Genuinely interesting and a natural extension, but it widens the mission
  objective and adds a modelling assumption that needs its own defence. Future work.
- Adversarial/learning jammer — the jammer is a fixed-policy scripted threat.
- Rigid-body flight dynamics — kinematic point-mass model with acceleration limits.
- Sionna in the training loop — offline validation only.
