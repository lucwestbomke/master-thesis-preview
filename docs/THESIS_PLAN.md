# Thesis Plan

**Working title:** How Much Channel Realism Does a Learned UAV Relay Swarm Need?
Multi-Agent Reinforcement Learning for Urban Relay Chains under Jamming

**Degree:** MSc Artificial Intelligence
**Official window:** March 2027 – August 2027 (~5 months)
**Preparation window:** now – March 2027 (unpressured, part-time)

---

## 1. The mission — what the swarm is for

A ground **High-Value Target (HVT)** drives through Frankfurt along the real road
network. A **Mobile Command Vehicle (MCV)** is parked elsewhere in the city.
Buildings block line of sight between them.

A swarm of `N` quadrotors must, continuously and simultaneously:

1. **Observe** — at least one drone must hold the HVT in an unoccluded ray,
   within sensor range.
2. **Relay** — the resulting video feed must reach the MCV at **≥5 Mbps
   end-to-end**, which needs a multi-hop chain because no single drone can
   usually manage both jobs at once.
3. **Survive** — finite batteries, and a jammer riding on the HVT that degrades
   links near it.

Episode fails if the mission link drops below 5 Mbps for >5 consecutive steps,
or any drone's battery reaches zero.

### The observation envelope — an angle constraint, not a distance one

To see a vehicle in a street canyon the ray must clear the roofline. That fixes
an *elevation angle* (~66° for Frankfurt's 20 m streets and 22 m fabric), not a
range. Consequences:

- **Across the street**, the drone must stay within `(W/2)·h/H_b` horizontally —
  36 m at 80 m altitude, 91 m at 200 m. Flying higher buys lateral freedom.
- **Along the street**, the roofline never blocks; the limit is the sensor.
  ~830 m to recognise a vehicle type, ~2.8 km merely to detect one.

So the real envelope is a long wedge down the street plus an overhead cone, and
real OSM footprints make it richer — intersections open it in four directions,
squares open it entirely. This is why occlusion must be computed from actual
geometry rather than a radius.

### Why it is hard

- **Observing fights relaying.** The observer is pinned near the HVT, deep in
  the street network and in the jammer's line of sight. Its data must be carried
  by others.
- **Occlusion is the binding constraint.** Frankfurt's tower cluster blocks
  air-to-air links; the low-rise fabric does not. Where a relay can usefully sit
  is decided by building geometry, not by range.
- **The target moves.** The chain must reconfigure continuously as the HVT turns
  through the street network and sightlines open and close.
- **Batteries deplete.** The observer role is the expensive one — it chases, and
  it sits in the jammer's face. A fixed division of labour is unsustainable.

All drones run the same policy (homogeneous, CTDE). Nothing assigns roles.

---

## 2. Research questions

### RQ1 (primary) — Which physical effects must a channel model include for learned policies to transfer?

> Most MARL work on UAV swarm communication abstracts the channel to a
> connectivity radius — two agents are "connected" if they are within *R*. Does
> that abstraction produce policies that fail under a physically realistic
> channel, and which specific physics is responsible?

Train one policy per **fidelity level**, evaluate **all of them under F3**:

| Level | Channel model | Isolates |
|---|---|---|
| **F0** | Connectivity radius, `R` calibrated to the median link range observed under F3 | the standard abstraction |
| **F1** | F0 + geometric occlusion from real footprints | cost of ignoring buildings |
| **F2** | F1 + jammer | cost of ignoring the threat |
| **F3** | Full SINR, Shannon rate with modulation cap, multi-hop rate division | reference model |

The gaps decompose the answer: F0→F1 is the price of ignoring occlusion, F1→F2
of ignoring the jammer, F2→F3 of ignoring rate and multi-hop division.

**Hypothesis:** the gap is dominated by **occlusion**. A radius model lets the
policy believe it is connected straight through a building, so it learns
geometry that cannot work. The jammer contributes a smaller, spatially localised
penalty, and rate division mainly shifts preferred chain length rather than
breaking the policy.

**Fairness requirement:** `R` in F0 must be *calibrated*, not guessed — set it to
the median link range measured under F3 in the same city. An arbitrary `R` makes
the comparison meaningless, and it is the first thing an examiner will probe.

**Why this question:** it is falsifiable either way, it reuses every line of the
channel work, and its answer is directly useful — it tells the field which
physics a swarm-communication simulator may safely omit.

### RQ2 (secondary) — Does relational structure help, and does it transfer?

Architecture ladder, isolating one factor per rung — full spec and the rules that
keep the comparison honest are in [`AGENTS.md`](../AGENTS.md#model-architectures):

| Architecture | Permutation-invariant | Size-agnostic | Uses link structure |
|---|---|---|---|
| Flat MLP (max-N padded + masked) | ✗ | ✗ | ✗ |
| DeepSets — `ρ(Σᵢ φ(xᵢ))` | ✓ | ✓ | ✗ |
| **GNN (capacity-weighted edges)** | ✓ | ✓ | ✓ |

Trained at `N=5`, evaluated zero-shot at `N ∈ {3,5,8}` **and on a second city**
with different morphology. Transfer across urban form is a stronger
generalisation claim than transfer across swarm size alone, and costs one extra
OSM extract, not extra training.

**Why not simply use the GNN.** A GNN that works shows only that a GNN works —
you cannot claim the graph structure helped without removing it and measuring the
loss. DeepSets is that control: identical except it ignores edges. Note also that
you must choose and justify an architecture regardless, so RQ2 is largely writing
down a decision the project forces anyway.

**Honest expectation:** at `N=5` the graph is tiny and GNN ≈ DeepSets is
plausible. The interesting result is in the transfer columns. RQ2 is the most
conventional part of the thesis and the first place to shrink (to DeepSets vs
GNN) if scope tightens — RQ1 is the contribution.

### RQ3 (tertiary) — Does role rotation emerge, and what causes it?

Does a homogeneous policy differentiate into observer / relay / resting roles,
and rotate them? Two candidate drivers, ablated separately so they do not
confound: the battery-variance penalty `-λ·Var(B)`, and the fact that the
observer role is also the jammer-exposed one. **First thing to cut if time runs
short.**

---

## 3. Conditions and baselines

| # | Condition | Purpose |
|---|---|---|
| B0 | **Scripted geometric heuristic** — relays placed on the MCV→HVT geodesic, one observer, fixed Ptx | Non-learned control. Answers "is MARL earning its keep?" Cheap, disproportionately valuable. |
| E1 | Learned policy trained at each of F0–F3, all evaluated under F3 | RQ1 |
| E2 | F3-trained × {MLP, DeepSets, GNN} × N ∈ {3,5,8} × 2 cities | RQ2 |
| E3 | F3-trained with `λ = 0` vs `λ = λ*` | RQ3 — see below |
| E4 | F3-trained with a 4-dim action (motion **+** transmit power) | RQ-power null, see below |

**E3, the `λ=0` ablation.** `λ` weights the battery-variance term `−λ·Var(B)`. If
one drone does all the observing it drains while the others idle, battery levels
spread, variance rises, penalty grows — so the term is *supposed* to cause role
rotation. Training with and without it tests whether it actually does. Rotation
only at `λ>0` gives a clean causal claim; rotation in both means something else
drives it (likely batteries simply running out and forcing a swap), which is
arguably the more interesting outcome; rotation in neither means the mechanism
does not work, and that gets reported too.

**E4, the power null check.** Hands the policy back the fourth action dimension
and shows it changes nothing. The oracle analysis in §6 already proves this
analytically; E4 makes it empirical as well. Two independent kinds of evidence
for ~$20 of GPU time, and it forecloses the obvious defence question *"but did
you actually try it?"* Cuttable under time pressure, but cheap insurance.

≥5 seeds per condition. Median and IQR, never mean ± std.

### Compute budget — final runs are a small fraction of the total

| Reported (final) runs | Count |
|---|---|
| F0, F1, F2 trained, 5 seeds each | 15 |
| F3 × {MLP, DeepSets, GNN}, 5 seeds each (the GNN run doubles as RQ1's F3 arm) | 15 |
| E3 `λ=0` ablation | 5 |
| E4 motion+power null check | 5 |
| **Total reported** | **40** |

At 10 M steps and the ≥1000 env-steps/s gate that is ~3 h per run, so **~120
GPU-hours** for everything that appears in the thesis. RQ2's transfer evaluation
adds no training — the policies already exist; evaluating them at `N ∈ {3,8}` and
on the second city is minutes.

**But development dominates.** A realistic project total:

| | Rough count | Note |
|---|---|---|
| Debugging / smoke runs | 50–100 | mostly killed within minutes |
| Curriculum tuning | 20–50 | the big unknown; where projects of this shape stall |
| Hyperparameter search (equal budget × 3 architectures) | ~30 | short or early-stopped |
| Crashes, reruns, mistakes | +25 % | always |

**≈300–500 GPU-hours in total, roughly $300–1000 on RunPod.**

The calendar consequence matters more than the money: essentially all of that is
*development*, and it belongs in Phase 0 before the March 2027 freeze. After the
freeze only the 40 reported runs execute. This is precisely what the preparation
window is for.

---

## 4. Metrics — pre-registered before any results are seen

**Primary (RQ1):** mission success rate under F3 — fraction of episodes
completed without link or battery failure.

**Mission outcome:** episode length; link-alive fraction; tracking coverage
fraction; mean and 5th-percentile end-to-end capacity.

**Failure attribution (this is what makes RQ1 explanatory rather than a table):**
- fraction of steps where the policy's intended chain passes through an occluded
  link — the direct signature of a radius-trained policy
- mean chain hop count and mean hop distance
- fraction of failures caused by observation loss vs link loss vs battery

**Behavioural (RQ3):** role-switch count per episode; terminal battery variance;
time each drone spends in the jammer's line of sight.

---

## 5. Scenario parameters — derived, not chosen

Fixed from sources outside this project, then the operating area solved for.

| Parameter | Value | Basis |
|---|---|---|
| City | **Frankfurt**, ~1500 m box over Bankenviertel + surrounding fabric | see below |
| Ptx | 30 dBm (1 W), fixed | UAV tactical MANET radios (Silvus SC4200, Doodle Labs Helix, TrellisWare TW-950) are 0.5–2 W |
| Jammer, in-band | 30 dBm | vehicle C-UAS barrage emitter, tens of watts over several hundred MHz |
| Carrier / bandwidth | 3.5 GHz / 10 MHz | S/C-band tactical allocation |
| Flight altitude | 80 m nominal | above the fabric, below the towers; inside TR 36.777's 22.5–300 m band |
| Rate target | 5 Mbps end-to-end | compressed HD EO/IR feed |
| Operating area | **1500 m** | single drone manages only ~1.7 Mbps at that range (fails); the swarm reaches ~24 Mbps (feasible) |

**Why Frankfurt.** The canyon ratio `H_b/W` decides everything:

| Morphology | H_b/W | Across-street LoS @80 m | Verdict |
|---|---|---|---|
| Manhattan Midtown | 8.3 | 4.8 m | Observation nearly degenerate off-axis |
| Chicago Loop | 5.5 | 7.3 m | Same |
| **Frankfurt fabric** | **1.1** | **36 m** | **Workable** |
| Frankfurt Bankenviertel | 9.0 | 4.4 m | Towers block A2A — the useful part |
| Paris Haussmann | 0.8 | 50 m | Above every roof, no A2A blocking → 2-hop chains |
| Barcelona Eixample | 1.1 | 36 m | Same, and no towers |

Frankfurt wins because it is **heterogeneous**: low-rise fabric gives a workable
observation envelope, while the tower cluster genuinely blocks air-to-air links
and forces chains to route around it. Uniform-tall cities are uniformly extreme;
uniform-low cities have no occlusion problem at all. Data quality is also good —
Hessen publishes open LoD2 3D building models, so heights are exact rather than
dependent on OSM tag coverage.

> **Phase 0 check:** verify LoD2 / OSM `building:levels` coverage for the chosen
> box before committing. Fall back to a height prior from footprint area and
> land use if coverage is patchy.

Regenerate the sizing with [`scripts/scenario_design.py`](../scripts/scenario_design.py);
the trade-off table is pinned by `tests/test_scenario_sizing.py`.

---

## 6. Reported negative result: transmit power control

Three independent framings for adaptive transmit power were tested numerically
against fair baselines. All three came out null, for three different structural
reasons. This is documented in [`NEGATIVE_RESULTS.md`](NEGATIVE_RESULTS.md) and
reported in the thesis rather than buried — it saves the next person the same
weeks, and E4 confirms it empirically alongside the analysis.

Summary: at a realistic 30 dBm ceiling the telecom term is ~1.6 % of power draw;
with a single flow and ordinary routing-aware medium access a ≤3-hop chain never
runs two transmitters concurrently, so a centralized oracle with full state
yields **0.0 %** over fixed power; and an emission-detectability cost saturates,
because the observer is unavoidably exposed while every other drone is already
below a −100 dBm ESM floor (**0.1–1.1 %**).

Consequence: the action space is **motion only** (3-dim) for the main
experiments. Transmit power is fixed at 30 dBm.

---

## 7. Architecture decision: batched env from day one

Verified against the installed stack: skrl's `PettingZooWrapper` round-trips
every action and observation through NumPy on each step, and exposes
`num_envs == 1`; its vectorized paths are Isaac Lab-only. That contradicts the
project's own stay-in-VRAM rule and caps throughput at single-env Python speed.

So: the env core is **batched with a leading `num_envs` dimension**; a thin
PettingZoo adapter sits on top for API-compliance tests and visual debugging
only; training uses a **custom skrl multi-agent wrapper** against the batched
core. `osmnx`/`shapely` are offline-only, baking buildings into a tensor of
boxes; runtime occlusion is vectorized segment-vs-box intersection in pure torch.

---

## 8. Timeline

Goal: enter March 2027 with a finished, tested, benchmarked simulator, so the
official five months are experiments and writing only.

### Phase 0 — Preparation (now → Feb 2027, part-time)

| Block | Deliverable | Done when |
|---|---|---|
| A | Channel model + routing, unit-tested | ✅ done — 53 tests, hand-computed |
| B | OSM/LoD2 pipeline for Frankfurt | Buildings + road graph cached as tensors; height coverage verified |
| C | Occlusion (batched torch slab method) | Matches a slow shapely reference on random geometry |
| D | Batched env core + PettingZoo adapter | Random policy runs; **≥1000 env-steps/s on GPU** |
| E | Renderer + B0 scripted heuristic | Video of the heuristic completing an episode |
| F | Fidelity levels F0–F3 as config flags | All four run; `R` calibration measured under F3 |
| G | MAPPO integration + curriculum | One toy run learns above random |
| H | Sionna offline validation of the closed-form channel | Agreement plot for the methodology chapter |

Blocks B/C/E are well-specified enough to hand to an agent in chunks. A/D/G
deserve your own attention.

### Phase 1 — Official thesis (Mar → Aug 2027)

| Month | Focus |
|---|---|
| Mar | Curriculum tuning until F3 trains reliably. **Freeze the environment.** |
| Apr | E1 fidelity ladder at 5 seeds. First RQ1 answer. |
| May | E2 architecture ladder + transfer. Write Methodology (model is frozen). |
| Jun | E3, E4, failure attribution, figures. Write Results. |
| Jul | Discussion, related work, introduction. Buffer for reruns. |
| Aug | Revisions, defence prep. |

**Hard rule:** freeze the environment end of March 2027. Everything before is a
pilot; everything after is thesis material.

---

## 9. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Env throughput too low | High | Batched design from day one; benchmark in Block D. Do not start Phase 1 without ≥1000 steps/s. |
| Nothing learns | High | Curriculum: stationary HVT → slow → jammer off → jammer on → full speed. Budget real time; this is where such projects stall. |
| RQ1 gap is trivially large ("wrong model gives wrong policy") | Medium | Calibrate `R` fairly, and make the *attribution* the contribution — which physics matters, and by how much — not the existence of a gap. |
| Building height data patchy | Medium | Hessen LoD2 as primary; footprint-area prior as fallback. Check in Block B. |
| Scope creep | Medium | RQ3 is the designated cut. Multiple concurrent flows and adaptive jammer are future work. |

---

## 10. Deliberately out of scope

- Adaptive/learning jammer — fixed scripted threat.
- Multiple concurrent sensor flows — the one untested route to making transmit
  power matter; noted as future work.
- Rigid-body flight dynamics — kinematic point-mass with acceleration limits.
- Sionna in the training loop — offline validation only.
- Sim-to-real. Stated plainly in limitations.
