# Decisions and reversals

Every entry here was **proposed, then rejected on evidence**. The point of this
file is that neither a human nor an AI session re-proposes them six months from
now, having forgotten why they died.

If you are about to suggest something on this list, read the row first. If you
still think it is right, the evidence is reproducible — go and disprove it.

Detailed numbers for the transmit-power nulls are in
[`NEGATIVE_RESULTS.md`](NEGATIVE_RESULTS.md). Full reasoning for anything else is
in the git log; commit messages are long on purpose.

---

## Errors in the original spec (corrected, tests pin them)

| Was | Now | Why it mattered |
|---|---|---|
| `SINR_dB = P_sig − (P_jam + N0)` | linear-domain sum, then convert | Adding two dBm values is a *product*. Returned ~**+100 dB** SINR for realistic urban links, silently deleting the jammer from every experiment. |
| TR 38.901 UMi for all links | TR 36.777 UMi-AV for A2G; FSPL + blockage for A2A | 38.901 UMi is specified for UE heights **1.5–22.5 m** and is not valid for aerial nodes. Doubly invalid for drone-to-drone above rooftop. |
| Noise floor hardcoded at −100 dBm | `−174 + 10log10(B) + NF` | Must track bandwidth. At 10 MHz / 7 dB NF it is −97 dBm. |
| Unbounded Shannon | `min(0.75·log2(1+SINR), 7.4)` b/s/Hz | Shannon is an upper bound; real NR caps at 256QAM. Unbounded reports fantasy throughput. |
| `B = 20 MHz` | `B = 10 MHz` | At 20 MHz, 5 Mbps needs only **−7.2 dB** SINR — satisfied by accident, jammer decorative. |
| `P_hover + α‖v‖²` | rotary-wing model (U-shaped) | Asserted hovering is cheapest. It is not: cruise at 13.3 m/s costs **0.64** of hover draw. Energy sets the observer's cost, so the sign error would have inverted the intended behaviour. |
| Multi-hop capacity undefined | `min(Cᵢ)/min(n,3)` + widest-path DP | The relay chain was the premise but was never specified. |
| `/n_hops` **and** full concurrent interference | `min(n,3)`, with `tx_mask` carrying the MAC | Double-counted the half-duplex cost. `/n` *is* the pure-TDMA schedule, which has no intra-chain interference to charge. Made a feasible 3-hop chain look infeasible. |
| Terminate on link loss > 5 steps | fixed-length episodes; failure is per-step | Two failures: the policy learns never to acquire so it can never fail, **and** a random initial policy dies by step 6 and never reaches the tracking phase at all. |

---

## Design directions abandoned

### ☠️ "The energy term pays drones to keep moving" — the arithmetic is right, the diagnosis is not

**Proposed** on 2026-08-26 and written into `REWARD.md` and `AGENTS.md`: the
rotary-wing power curve is U-shaped, so at `w_energy = 0.15` cruising at 13.3 m/s
pays **+0.0544/step** over holding station — 102× the shipped potential's whole
reachable swing — and *that* is why the observer stands at 184 m rather than
closing to 89 m.

📏 **Measured 2026-08-27, and the learned policy is not taking the money.**
`scripts/measure_potential.py`, eval split, stage 4, F4, MPS, per drone per step:

| | speed p50 | steps > 24 m/s | energy term | steps at the map wall | mean \|a_z\| |
|---|---|---|---|---|---|
| B0 | **5.81 m/s** | 3.1 % | **−0.1250** | **0.9 %** | 0.005 |
| GNN | **24.71 m/s** | **56.7 %** | **−0.1333** | **23.1 %** | **0.821** |
| MLP | — | — | — | 15.6 % | 0.626 |
| random | 17.13 m/s | 13.9 % | −0.1158 | — | — |

The learned policy flies at the **25 m/s dash cap** on more than half of all
steps, where `P/P_hover ≈ 0.99`. It pays **0.0083/step more** than B0, not less.
The minimum-power airspeed is 13.3 m/s and it is nowhere near it.

**What survives.** 0.0544/step is the largest per-step force the objective can
exert on the motion decision, so it remains the right **bar to size `Φ` against**
— and `REWARD.md`'s conclusion ("the lever is `Φ`; do not cut `w_energy`") is
unchanged.

**What does not.** Energy is **not the mechanism** behind the 184 m stand-off.
Over the episode the energy term is worth ~2 % of this policy's return
(0.0083 × 600 = 5.0 against ~240 of mission reward), which is noise to it. ⚠️ Any
future intervention argued from "the objective pays them to cruise" is arguing
from a force the policy is measurably declining to collect.

🔍 **And the replacement is more actionable.** Half the steps at the dash cap,
23 % pressed against the map boundary, and near-full vertical acceleration
commanded while parked at the 80 m ceiling, is not an energy trade — it is a
policy with no gradient telling it where to be. Which is exactly what the
`Φ` audit found: every shipped component is a `min`/`max`/routing reduction, so
`Φ` is **exactly constant in four drones out of five**. See `REWARD.md`.

⚠️ **Method note, and it is the same one this block keeps re-learning.** The
0.0544 was computed from the power curve and never checked against what the
policies fly. One render of route 12 and one histogram of speed cost ten minutes
and inverted the reading.


### ⛔ `r_cover = R/2` for `Φ_cover` — the derivation was clean and the sweep rejected it

**Proposed** as the obviously principled choice: 📏 `R` = 524 m is Block F's
measured median link range, so a coverage radius of 262 m makes two drones whose
discs touch exactly one hop apart. It is the link criterion, stated in geometry.

**Killed by measuring what it does to the learning signal.** A potential is not a
feasibility test — it has to still have gradient *at the behaviour it rewards*,
and at 262 m the term is saturated for B0. Swept on real state banks
(`scripts/measure_potential.py`), B0 against a learned policy, `w_cover = 0.30`:

| `r_cover` | 120 | 180 | 262 | 400 m |
|---|---|---|---|---|
| B0 − learned separation | **1.247** | 1.114 | 0.872 | 0.575 |
| B0's p1, of a 3.0 maximum | **1.428** | 2.000 | 2.446 | 2.750 |
| whole trip home from 500 m off-axis | **0.335** | 0.286 | 0.242 | 0.184 |

120 m wins on all three. It is *corroborated* by Block B's 127 m along-street
sightline median — but that is a coincidence noticed afterwards, not the reason,
and it is recorded that way on purpose.

**What it rules out.** "Derived from a measured constant" is not by itself an
argument for a value **inside `Φ`**, where the only thing that matters is where
the gradient lands. The optimum is protected by PBRS; the learning signal is not.


### ⛔ Per-drone potentials as the fix for role emergence — the channel opened, nothing moved

**Proposed** on a measured deficit rather than a hunch. `scripts/probe_credit.py`
showed the learner had **no per-drone credit channel at all**: the critic is
handed one global state repeated per drone, so `max |V_i - V_j| = 0.000e+00`, and
0.015–0.06 % of advantage variance distinguished one drone from another. Every
drone's gradient was `grad log pi(a_i|o_i) * A` with the same `A`.

`w_relay` — a per-drone potential on `on_path` — raised that **71×**
(0.00041 → 0.02931). The behaviour did not move: `hop | observed` measured
**1.88–1.93 against a control of 1.91**, at 5 seeds. More of the term was worse
(42.4 % → 39.7 %), and the agent-specific critic paired with it was worse still
(34.1 %, one seed collapsing to 0.3 %).

✅ Both stay in the code, off by default (`w_relay = 0.0`,
`agent_specific_state=False`), because the probe that motivated them is the
reusable part and the flags are how it gets re-run.

**What it rules out, and the ruling-out is the value.** Missing per-drone credit
was real and is now supplied. Roles still do not emerge, so credit assignment was
**necessary-looking but not sufficient**, and the search moves elsewhere.

### ☠️ `hop | observed` as a measure of relay-role emergence — it measures geometry

Used as Gate 2's primary readout, and that was a mistake worth recording because
the statistic looks behavioural and is not.

📏 Measured across every configuration in Block G: random **1.83**, MLP 1.86,
DeepSets 1.88, GNN 1.91, +recurrence 1.89, +`w_hold` 1.87, +`w_relay` 1.88–1.93,
B0 **2.26**. Six interventions inside a 1.86–1.93 band, with random at the bottom
of it.

**A chain's hop count is set by where the observer stands.** Against `R` = 524 m
with the HVT ~1 km out: B0's observer at 79 m from the HVT is ~920 m from the MCV
(~2–3 hops); the learned observer at 291 m is ~710 m from the MCV (~2 hops). The
swarm builds exactly the chain its observer position requires.

⚠️ **Consequence for the diagnosis.** There is no separate relay-role failure.
`observed`, observer tenure and hop count are three views of **one** failure: the
observer does not close. `BLOCK_G.md`'s earlier framing of "two roles, neither
emerging" is corrected there.

⚠️ **Consequence for RQ3.** Hop count and chain composition are reported as
behavioural signatures. They are **confounded with observer stand-off**, and must
be reported conditioned on it — or reported as what they are, a consequence of
where the sensor sits.


### ⛔ Recurrence as the fix for observer tenure — killed on its own pre-declared rule

**Proposed** on a mechanism that still stands: a stateless policy provably cannot
represent B0, which carries state, so the 5.6× observer-tenure deficit could not
be closed by a feedforward actor. Recurrence was built, unit-tested, blocked for
a month by a skrl bug (see the entry above), fixed, and reached feedforward parity
at stage 1.

**Killed 2026-08-25 on the full mission.** A 2×2 (recurrence × `w_hold`), GNN,
`deep`, 5 seeds, train split, with the rule declared before the runs — keep if
tenure ≥ 95 **and** capable ≥ 45.1 %:

| pooled, 10 runs per level | capable | seed IQR | tenure |
|---|---|---|---|
| feedforward | 40.7 % | 4.7 | 40.5 |
| recurrent | **39.7 %** | **6.9** | **36.8** |

Tenure **36.8** against a required 95, capable **−1.05 pp**, and the seed spread
*widened*. The drop rule is met outright.

⚠️ **Distinguish this from "recurrence does not train".** It trains. It reaches
feedforward parity at stage 1. It simply does not help on the full mission, and
the `PPO_RNN` fix that unblocked it is still worth having — it removed a real bug
and a false diagnosis.

**What it rules out, and this is the useful part.** Memory was the strongest
representational hypothesis for the tenure deficit. Its failure says the deficit
is **not** that a drone cannot remember it was the observer. Combined with the
chain evidence below, it points at role emergence instead — see `BLOCK_G.md` §
*One mechanism explains both rows*. ⛔ Do not re-propose recurrence for tenure
without a new mechanism; propose it for partial observability of the *cue* if at
all.

### ☠️ `chain_occluded` as RQ1's failure-attribution metric — not usable as defined

📏 Measured on the eval split, CUDA, 5 seeds: B0 **61.5 %** against every learned
policy's **34–40 %**, and `corr(hop_mean, chain_occluded) = 0.963` across the five
policies. It is a **per-chain** statistic, so it rises with the number of edges in
the chain: more hops, more chances one crosses a building.

`THESIS_PLAN.md` designates it as RQ1's failure-attribution metric, and F4's rate
division changes chain length — so comparing it across fidelity rungs would
compare **hop counts wearing an occlusion label**, and the resulting "F4 chains
are more occluded" would be arithmetic, not physics.

**Fix before RQ1 uses it:** report the **per-edge** occlusion rate, which is
hop-count-invariant. Keep the per-chain figure as a descriptive statistic only,
and never compare it across policies or rungs with different `hop_mean`.

⚠️ This also invalidates the reading in `BLOCK_G.md`'s stage-1 table that the
learned policy's higher `chain_occluded` showed it "had not discovered the
clearance feature". At stage 4 the ordering reverses, and hop count explains both.


### ⚠️ "Recurrence does not train" — believed for a week, and the GRU was innocent

**Proposed and acted on:** the recurrent actor collapsed at stage 1 (37 % → 2 %)
where the feedforward actor reached 75–79 %. Three sessions of probes localised
the fault to *sequence replay*, on the strength of `--seq-len 1` not collapsing
and an epoch-0 log-probability residual concentrated at sequence position 0.
`BLOCK_G.md` recommended parking recurrence and resuming by "diffing the stored
`rnn_policy_0` tensor against a hand-stepped reference".

**Killed 2026-08-25 by one control that had never been run:** `PPO_RNN` with
**feedforward** models — no GRU anywhere, `_rnn = False`, so the class degrades to
plain PPO on the same code path. It collapses identically (peak 52.5 %, final
3.8 %) where the same models under `MAPPO` reach 76.2 %.

**Cause:** `skrl/agents/torch/ppo/ppo_rnn.py` in 2.1.0 is an un-migrated copy of
an older PPO, carrying its own stale `compute_gae`:

```python
not_terminated = terminated.logical_not()  # ppo_rnn.py:45
not_done = (
    (terminated | truncated)
    if time_limit_bootstrap  # mappo.py:49
    else terminated
).logical_not()
```

At a truncation `not_terminated` is True, so GAE recurses *through* the reset and
the step collects `γ·(V_{i+1} + λ·A_{i+1})` from the **next episode** — on top of
a bootstrap already folded into the reward. Double-counted, and the next
episode's advantage propagates backwards at `(γλ)^k` = 0.947/step. Fixed in
`training.recurrent_ppo.PPO_RNN_Aligned`; isolated against the second fix
(bootstrapping off the next state), which on its own changes nothing: 3.8 % →
11.3 % for that one alone, 69.7 % for the GAE mask alone.

**✅ No reported number is affected.** Only `PPO_RNN` was ever wrong; every
measured result went through `MAPPO`.

**Two things to carry, and they are about method rather than about skrl.**

1. ⚠️ **The known-optimum probe cleared the loop and hid this bug.** Under
   `-w_effort·‖a‖²` PPO improved throughout the collapse, because a pure
   per-step action cost has almost no cross-episode structure and the boundary
   leak costs it nothing. **The probe clears the plumbing; it does not clear
   credit assignment.** Add a probe with a known optimum that *spans* the
   episode before trusting it again.
2. ☠️ **Every probe was aimed at the component under suspicion and none at the
   vehicle carrying it.** The decisive control — run the suspect *path* with the
   suspect *component removed* — cost four minutes and was available from day
   one. Do that before a third session of instrumenting the component.


### Adaptive transmit power — three framings, three nulls
**Do not reintroduce Ptx as an action.** Evidence in `NEGATIVE_RESULTS.md`.

| Framing | Result | Reason |
|---|---|---|
| Energy saving | ~1.6 % of power draw | Raising the ceiling to fix it destroys the mission instead — at 40 dBm one drone spans any simulable map. |
| Interference management | **0.0 %** vs a fair baseline | One flow + routing-aware MAC + ≤3 hops ⇒ the reuse schedule never runs two transmitters at once. Nothing to manage. |
| Detectability / EMCON | 0.1–1.1 % | Exposure saturates: the observer must sit in the threat's LoS and is always detected; everyone else is already below a −100 dBm ESM floor. |

Condition **E4** reproduces the null empirically alongside the analysis.
The one untested route is **multiple concurrent flows**, which would create real
contention — listed as future work because it turns the project into a
distributed link-scheduling study.

### Ptx ceiling of 40 dBm
Briefly specified to enlarge the energy term. A *blocked* A2A link at 10 W still
carries 15 Mbps over 2.8 km, so a single drone spans any map up to 2 km and the
relay chain — the entire premise — becomes unnecessary. **Range grows with power
far faster than the mission area can absorb.** Fixed at 30 dBm.

### Refreshed cue ("external ISR asset updates every 10 s")
Incoherent: a sensor that can persistently track the HVT through a city makes the
swarm redundant. It was a mechanism invented to fix cue staleness with a
justification attached afterwards. **The correct fix was geometric** — start the
HVT 300–500 m away so transit is short enough that one cue survives it.

### Blind search for the HVT — ⚠️ the stated reason was wrong; the conclusion stands for a different reason
This entry used to read: *"a sparse 'found it' reward over 1500 m² would swamp
the learning signal."* **Measured in Block D, that is false**, and it argues
against a scenario this project does not have. The HVT starts in a 300–500 m
annulus the drones launch *inside*, and the sensor reaches 830 m. Five drones on
a radial fan over 512 real routes
([`../scripts/measure_envelope.py`](../scripts/measure_envelope.py)):

| strategy | ever found | t50 | t90 |
|---|---|---|---|
| **no cue**, 5-way fan | **100 %** | **8 s** | 22 s |
| cue σ=150 m, narrow fan | 99.8 % | 10 s | 22 s |
| no cue, all five on one bearing | 59.0 % | 13 s | 46 s |

Uncued is *faster*, and the reward is never sparse — acquisition takes ~20 of 600
steps, and the PBRS `approach` potential is dense throughout regardless.

**The cue survives on a narrower argument.** What the third row shows is that
what matters is **spreading out**, not knowing the direction — and spreading
requires homogeneous agents to break symmetry off the neighbour channel. That is
a coordination problem no RQ asks about, sitting in Block G, the acknowledged
place projects of this shape stall. The cue buys that risk away for 3 observation
dims. It is variance reduction on a phase that is not the subject of study, not a
fix for exploration.

Two consequences: the no-cue condition becomes a **cheap ablation rather than an
optional final curriculum stage**, and the result leans on the 360° sensor
assumption ([`BLOCK_D.md`](BLOCK_D.md)) — optimistic for search, not for
tracking. Say so when reporting it.

### Manhattan (and any uniform-tall city) as the map
Canyon ratio `H_b/W ≈ 8.3` gives a **4.8 m** across-street observation envelope —
the drone would have to hover within five metres of a moving vehicle, so the
observation task collapses into "be exactly overhead" with no spatial decision
left. Uniform-*low* cities (Paris, Barcelona) have the opposite problem: the drone
is above every roof, air-to-air never blocks, chains collapse to two hops.
**Frankfurt wins because it is heterogeneous** — low fabric gives a workable 36 m
envelope while the tower cluster still blocks A2A.

### Energy-driven role rotation as RQ3
A realistic airframe burns only **7 %** of a 548 Wh pack over a 240 s episode.
Batteries never bind, `Var(B)` stays tiny, λ has nothing to act on. That risks the
worst kind of null — *"there was nothing to explain"* rather than *"λ does not
explain it"* — which is uninterpretable. Replaced by **geometric handoff**, which
is forced by the environment rather than contingent on a reward parameter.
λ survives as a secondary ablation, made informative by randomising initial charge
in `[0.3, 1.0]`.

### skrl's `PettingZooWrapper` for training
Verified against the installed version: it round-trips every action and
observation through NumPy on each step (`untensorize_space` / `tensorize_space`)
and exposes `num_envs == 1`; its vectorized paths are Isaac Lab-only. That
contradicts the stay-in-VRAM rule and caps throughput at single-env Python speed.
The adapter is kept for API-compliance tests and visual debugging only.

### OSM `height` / `building:levels` as the height source
Measured, not assumed (`scripts/check_height_coverage.py`, 1500 m box, three
candidate centres). **Area-weighted coverage 57–59 %**; raw coverage 41–44 %.
Only 3–5 % of footprints carry an explicit `height` tag — the rest of the
coverage comes from `building:levels` and its 3.2 m/storey assumption.

The 42 % of built area with no height at all is **not** sheds: it includes Die
Welle, the Börse, the Bundesbank headquarters and Triton House — the 20–60 m
mid-rise blocks that set the canyon ratio, which is half of why Frankfurt was
chosen. Where OSM *does* have a tag it is often wrong: it puts the
Deutsche-Bank-Hochhaus at 22 m (actual ~155 m) and the Main Tower at 170 m
(actual ~200 m).

**Superseded by Hessen LoD2** (below). OSM is still the source for the road
graph — that is well mapped and heights are irrelevant to it.

### One axis-aligned box (AABB) per building part
The original Block B spec said axis-aligned boxes, splitting rotated or concave
footprints as needed. Measured on the chosen Frankfurt box (4351 LoD2 parts):
one AABB per part inflates built area **+134 %**, from 0.901 km² to 2.109 km²,
filling **94 % of the box**. The city becomes effectively solid, every link is
blocked, and occlusion no longer separates the fidelity rungs — RQ1 measures
nothing.

Cause: LoD2 parts are rectangles but **rotated**. Only 18 % are within 10° of
axis-aligned; median long-axis orientation is 38°. An AABB around a 45°-rotated
rectangle doubles its area (measured median ratio 1.90).

**Replaced by oriented boxes (OBB):** median ratio **1.07**, +38 % total, 55 %
fill. The slab method is unchanged — rotate the segment into the box frame
first, with `cos θ`/`sin θ` baked in offline. `M` stays at 4351; matching OBB
fidelity with AABBs would need tens of thousands of boxes.

### Flattening the MCV spawn quadrant bias
Raised as a concern, then **investigated and dropped** — the fix would have been
worse than the thing it fixed.

The observation: all 121 MCV spawn points sit at `r > 500 m` from the box centre,
and route counts split SW 34 % / NE 28 % / SE 25 % / NW 13 %.

Three measurements killed it:

| quadrant | eligible spots | routes | **routes/spot** |
|---|---|---|---|
| NE | 30 | 571 | 19.0 |
| NW | 15 | 269 | **17.9** |
| SE | 28 | 510 | 18.2 |
| SW | 48 | 698 | **14.5** |

1. **There is no sampling bias.** Every eligible junction is drawn about equally;
   NW spots are sampled slightly *more* than SW ones. The route-count split is
   entirely because NW has 15 eligible junctions and SW has 48 — a property of
   Frankfurt's layout under the reach requirement, not of the sampler.
2. **Flattening would concentrate repetition where it hurts most.** Forcing 512
   routes per quadrant gives NW's 15 spots 34 routes each — **1.9× more**
   repetition than now — on the smallest and most geometrically peculiar subset,
   while SW drops to 10.7. That trades a weak concern for a stronger one.
3. **The actor cannot see which quadrant it is in.** The 24 ego features contain
   no absolute position except own altitude ([`ENVIRONMENT.md`](ENVIRONMENT.md) →
   Observations). Everything else is relative or local sensing, so "the MCV is
   usually south-west" is not representable. The only residual channel is the
   pattern of clearance margins — and a policy responding to local building
   geometry is doing the right thing, not cheating.

   > Re-checked when Block D added the 3-dim cue vector. It does not break this:
   > the cue is relative to the drone's own position and its location is
   > randomised per episode, so combining it with the relative vector to the MCV
   > yields only the *initial HVT bearing from the MCV* — still no absolute
   > position, still no quadrant identity.

The **periphery** constraint is separate and is arithmetic, not a choice: the box
half-diagonal is 1060 m, so a centrally-parked MCV cannot reach the 1400 m the
escalation needs. Only a larger box or less escalation would change it, and a
command vehicle staging at the perimeter is the realistic reading anyway.

**If diversity ever does bind, add positions rather than redistribute them.** MCV
placement is currently restricted to graph *junctions*; nothing requires a vehicle
to park at an intersection. Sampling from all densified road points gives **858
eligible spots instead of 121** — a 7× increase, dropping per-spot repetition from
17 to 2.4 — and barely moves the quadrant split, confirming the skew is the map.
That is a one-line change in `sample_routes` plus a re-bake. First thing to try if
RQ2 transfer ever looks like map memorisation.

### Rotating the whole map to rescue AABBs
The obvious follow-up once AABBs fail. Swept map rotations 0–74°: the best is
**+96 % at 30°**, still far worse than OBB's +38 %, and the fill never drops
below 78 %. Frankfurt has no single dominant street orientation — part
orientations run 22–74° interquartile — so any global rotation that helps one
district hurts another. Dead.

### An unbounded (or high) altitude ceiling
Nothing in the model charges for altitude — `energy.propulsion_power_w` is a
function of speed only, and a full 150 m climb costs 0.55 % of the pack. Both
physical effects then point straight up, so **the ceiling is the entire altitude
policy** and `a_z` will saturate there. Measured
([`../scripts/measure_envelope.py`](../scripts/measure_envelope.py)):

| altitude | A2A links blocked | HVT visible at 100–200 m offset |
|---|---|---|
| 80 m | 31.2 % | 38.2 % |
| **120 m** | **24.6 %** | **48.1 %** |
| 180 m | 10.2 % | 55.9 % |
| 230 m | **0.0 %** | — |

Above ~180 m the tower cluster stops blocking anything, F1's A2A component
disappears, and RQ1's primary result changes **silently**. Band fixed at
**40–80 m** (see the W1 entry below, which tightened it further).

The floor is a *model-validity* limit, not a flight rule: at 10 m altitude 37 %
of positions sit inside a building box, where `occlusion.py`'s
`ignore_endpoint_boxes` convention — chosen for a 1 % case — lets a drone see
through the building it is standing in; and `pathloss_a2g_umi_av_db` clamps `h`
to 22.5 m, silently substituting a different altitude. At 40 m containment is
3.3 %.

Rejected alongside: **an altitude energy penalty**. Physically correct to add
(`W·v_z/η`, and it is being added), but it cannot bind — the climb is a one-off
0.55 % of the battery. Only the ceiling controls this.

### A 120 m altitude ceiling — it falsifies W1
Proposed in Block D on the A2A-occlusion constraint alone: keep the tower cluster
blocking air-to-air links so RQ1's F1 rung has an A2A component. That ruled out
anything above ~150 m and 120 m looked safe. **The scenario constraint was not
consulted, and it is tighter.**

W1 — "a single drone cannot do the mission" — is what makes this a swarm problem
at all. Measured on real geometry by placing one drone in the most favourable
position available to it (hovering directly over the HVT, an upper bound on solo
capability under *any* policy):

| ceiling | solo mission-capable at 1336 m |
|---|---|
| 80 m | **3.3 %** |
| 100 m | 23.2 % |
| 120 m | **57.4 %** |

At 120 m a perfectly-placed single drone does the mission most of the time and
the swarm becomes an optimisation, not a necessity. **Band fixed at 40–80 m.**
Both constraints point the same way, so nothing is traded: A2A blockage is 31 %
at 80 m against 25 % at 120 m, so RQ1 gets *stronger*.

Until this was measured, W1 rested on `scenario_design.py`'s analytic canyon rule
(ground LoS within 0.625×altitude), which the Block D A2G measurement showed is
more conservative than the real map. **Any future change to the altitude band
must re-run `measure_envelope.py --only solo`.**

Side benefit: the ceiling is now *derived* from the project's own scenario
requirement rather than needing a civil-UAS citation, so that `TODO(verify)` is
discharged. Regulation becomes corroboration.

### Raising the HVT speed (`CONGESTION_FACTOR`) — considered, not done
Asked in Block E: the target's realised median is **5.8 m/s (21 km/h)** after
`CONGESTION_FACTOR = 0.70`, against a 20 m/s drone — a 3.4× margin that makes
keeping up trivial. Is the congestion factor too strong?

Measured with `speed_scale`, scored over the first 400 steps so the route running
out cannot confound it:

| `speed_scale` | ~m/s | mission-capable | observed |
|---|---|---|---|
| 1.00 | 5.8 | 65.2 % | 90.0 % |
| 1.25 | 7.2 | 59.3 % | 89.2 % |
| 1.50 | 8.7 | 52.4 % | 87.7 % |

**A faster target barely affects tracking** — `observed` falls 2.3 points across
a 50 % speed increase, because a 20 m/s drone keeps up with an 8.7 m/s car
easily. It costs 12.8 points of *mission* success, and it does so by covering
more ground, opening the separation faster, and stressing the **chain**.

So it is the same lever as the rate requirement, pulled less defensibly:

1. **It buys chain stress, which is already bought.** The 5 → 15 Mbps change
   did that, at no cost to the frozen artefact.
2. **It requires a re-bake.** A route step is a fixed *displacement*, so changing
   the speed means regenerating `data/frankfurt_box.npz` and re-validating
   BLOCK_B's joint calibration of `CONGESTION_FACTOR` against `MCV_MIN_REACH_M`.
3. **It is less realistic, not more.** 21 km/h is a normal average speed for a
   vehicle crossing a dense European city centre with intersections and signals.
   31 km/h (`speed_scale = 1.5`) is not.
4. **It does not fix RQ3.** Observation stays ~88 % solved, so the observer role
   still hands over rarely; the thin-handoff problem was solved by re-pointing
   RQ3, not by speed.

**Keep `CONGESTION_FACTOR = 0.70`.**

> ⚠️ **One inconsistency this exposed, worth fixing in the write-up.** AGENTS.md
> justifies the drone speed as a "**1.4–1.8× margin over the HVT**", computed
> against the *free-flow class cap* of 13.9 m/s. The baked bank's realised median
> is 5.8 m/s, so the **actual** margin is ~3.4×. Neither number is wrong — the cap
> is the cap — but quoting 1.4× implies the target typically moves at 13.9 m/s,
> and it does not. Say "1.4× against the fastest permitted road class, ~3.4×
> against the realised median".

### Promoting N=3 to a training condition — proposed, then killed by measurement
Raised in Block E on the reasonable-sounding grounds that N = 3 is the hardest
condition (B0 36.4 % against 57.2 % at N = 5) and therefore the place a learned
policy has most to prove. **Measured, it is the place a learned policy has
*least* to prove.**

What matters is not difficulty but *headroom for control*, which the
`B0-geodesic` → `B0` gap measures directly (`eval_baseline.py --only transfer`,
5 seeds, eval split):

| N | `B0-geodesic` | `B0` | **control is worth** | fail-with-target-in-sight |
|---|---|---|---|---|
| 3 | 33.3 % | 36.4 % | **+3.2 pp** | 55.0 % |
| 5 | 47.1 % | 57.2 % | **+10.1 pp** | 35.3 % |
| 8 | 48.4 % | 74.3 % | **+25.9 pp** | 19.1 % |

**Hardness and headroom move in opposite directions.** Three drones on a
three-hop chain have essentially one viable arrangement, so cleverness buys 3
points. Eight drones have many, and it buys 26. Look at `geodesic` along the row:
33 → 47 → **48** — it barely improves from N=5 to N=8, because evenly spacing
eight drones on a line is no better than evenly spacing five. `B0` does improve,
because something is deciding where to put them. **That difference is the
coordination problem**, and it is largest at N = 8.

**Two things follow:**

1. **Put the analytical weight on N = 8, not N = 3.** It is already in the matrix
   as a zero-shot transfer column, so this costs nothing. It is also the max-N
   padding boundary (`N_MAX = 8`), where a flat MLP should struggle and the
   size-agnostic rungs should not — exactly RQ2's claim.
2. **Do not train at more than one N.** It would destroy what RQ2 measures: the
   off-N columns stop being zero-shot transfer and become in-distribution. It
   also costs +15 runs against a budgeted 45, and `N = 5 trained; 3/5/8
   evaluated` is a settled parameter. N = 3 stays valuable as the hard end of the
   *evaluation* range, where the relay premise binds hardest.

**A third reason not to train at N = 3, worth recording separately:** with three
drones and a three-hop chain, every drone is load-bearing and none can be spared.
That removes the slack any role-reconfiguration question needs, so N = 3 would
also have damaged RQ3.

### RQ3 re-pointed: relay reconfiguration, not observer handoff
Not a rejection — a **redirection**, made on measurement, and recorded here
because the RQ's headline changed.

RQ3 asked whether the **observer** role hands off. It does, and the detector
built in Block E works and discriminates. But the phenomenon is **rare**: a drone
parked over the target seldom loses it, so a competent policy hands over ~once
per 240 s episode. Per episode under B0:

| | observer role | relay chain |
|---|---|---|
| role changes | **0.9** | **52.2** |
| drones entering/leaving | — | 100.6 |
| distinct role assignments | — | 16.6 |

Relay reconfiguration is ~58× more frequent, and it is **driven by occlusion
changing link quality** — RQ1's subject — where observer handoff is driven by
sensor occlusion, the smaller half of the mission since the rate requirement
moved to 15 Mbps.

**The measurement hazard, and the metric that clears it.** `on_path` changes both
because the routing DP re-selects over stationary drones (the algorithm) and
because drones move into position (the policy). RQ3 must isolate the second. The
metric that does is a drone's **link-viable run** — consecutive steps sat off the
chain while holding a usable link to it — sampled when it is recruited. It orders
policies correctly, which is the evidence that it measures behaviour:

| | random | waypoint | `B0-geodesic` | `B0` |
|---|---|---|---|---|
| relay anticipation lead (steps) | 1.6 | 6.3 | 19.7 | 18.4 |

Caveat to report with it: a stationary drone accumulates viable time by luck, so
read it **across** policies, not absolutely.

**Consequence for E3a:** the ablation used to zero neighbours' `sees_hvt`, which
is the channel for *observer* handoff. The channel for relay reconfiguration is
the `on_path` bit plus the per-edge capacity and clearance, so that is what it
must zero now — which incidentally makes E3a the DeepSets↔GNN contrast applied to
behaviour. Full framing in [`THESIS_PLAN.md`](THESIS_PLAN.md) RQ3.

**Observer handoff is kept as a secondary result.** The detector exists, it costs
nothing, and reporting the rare phenomenon beside the abundant one is what
justifies having chosen between them.

### ⚠️ The 5 Mbps rate requirement — raised to 15 in Block E
Not a rejected direction: a **change**, made deliberately and recorded here
because it moves a constant that nine documents quote and that two other
decisions were derived from.

**Why.** At 5 Mbps the radio link never binds. Block E measured the chain's
bottleneck at a median **37.6 Mbps — 8× the bar** — so `mission_capable`
collapsed to `observed`, the mission reduced to "put one drone over the car",
and a scripted baseline reached **93.2 %** with the metric saturated. The relay
chain was *necessary* (a solo drone still fails) but never *difficult*, which is
the wrong shape for a thesis about relay geometry. Measured for B0 at N=5:

| requirement | 5 | 10 | **15** | 20 | 30 | 40 Mbps |
|---|---|---|---|---|---|---|
| B0 mission-capable | 93.3 % | 69.6 % | **54.7 %** | 44.3 % | 19.4 % | 11.2 % |
| sensor-only ceiling | 93.4 % | — | — | — | — | — |

At 15 the binding constraint moves from the **sensor** to the **relay chain** —
the drone can see the car and still cannot get the video home — leaving ~39
points of headroom that are pure relay geometry. It also revives F4's
rate-division rung, which was inert at 5 (37.6/3 = 12.5 still cleared it).

**Defensible independently of the measurement**: 5 Mbps is one compressed HD
stream; 15 is a dual EO/IR feed at low latency, which is what a tracking ISR
sortie actually carries. The measurement chose *among* defensible values rather
than inventing one.

**What it did NOT touch:** `data/frankfurt_box.npz`, the geometry bake,
occlusion, the throughput gate. The expensive frozen work is untouched.

**What it did touch**, all re-run: `τ_l` 2 → 6 Mbps (still 40 % of the
threshold); the three duplicate copies of the constant in `scripts/` now import
it; `test_reward.py`'s stub capacities, which were magic numbers that silently
crossed the bar and inverted three tests — they now express intent (`GOOD`,
`OK`, `POOR`) relative to the constant; and every Block D/E measurement.

### ⚠️ Consequence: the altitude ceiling is no longer *derived* from W1
Block D discharged the ceiling's `TODO(verify)` by deriving 80 m from W1 — above
it, a best-placed solo drone could do the mission and the swarm premise
dissolved. **At 15 Mbps that argument no longer selects an altitude**, because
W1 now holds everywhere:

| ceiling | solo mission-capable at 1336 m, 5 Mbps | at **15 Mbps** |
|---|---|---|
| 80 m | 3.3 % | **0.4 %** |
| 100 m | 23.2 % | **0.4 %** |
| 120 m | **57.4 %** | **0.8 %** |

This is good news about the *scenario* and awkward about the *justification*.
The swarm premise is now robust rather than balanced on the altitude choice — at
5 Mbps W1 broke at 120 m, and now it does not break anywhere in or above the
band.

**The band stays 40–80 m, on the reason that was always the other half of the
argument:** A2A occlusion is what RQ1's F1 rung measures, and it is *strongest*
at the bottom of the band — 31.2 % of air-to-air links blocked at 80 m against
24.6 % at 120 m and 10.2 % at 180 m. Above ~180 m it disappears entirely and
RQ1's primary result would change silently. The floor is unchanged and is still
a hard model-validity limit (3.3 % of positions inside a building box at 40 m,
and TR 36.777 stops at 22.5 m).

**Say it accurately in the write-up.** The ceiling is now *chosen* to maximise
the A2A blockage RQ1 studies, within TR 36.777's validity, and corroborated by
the EU open-category 120 m AGL limit — **not** derived from W1. That is a weaker
form of justification than Block D claimed, and pretending otherwise is the kind
of thing an examiner checks. W1 itself is now far more robust than it was, which
is the compensating gain.

**Do not raise the ceiling on the strength of the new W1 numbers.** They permit
it; A2A occlusion still forbids it — and that is now measured end to end rather
than inferred from a link statistic.

#### The replacement argument, and it is stronger than the one it replaces

`eval_baseline.py --only ceiling`, B0 at 15 Mbps, 6 seeds on the eval split:

| ceiling | mission-capable | observed | A2A links blocked |
|---|---|---|---|
| **80 m** | **56.6 %** | 93.0 % | **31.2 %** |
| 100 m | 64.2 % | 93.3 % | ~28 % |
| 120 m | 74.5 % | 94.6 % | 24.6 % |
| 150 m | 80.8 % | 95.1 % | ~17 % |

**`observed` barely moves — 93.0 → 95.1 across 70 m of climb.** Climbing does
almost nothing for the sensor, because a drone over the target already sees it.
Nearly the entire 24-point gain in mission success is **air-to-air**: higher
relays have clearer links to each other.

Which is to say: **the altitude ceiling is the primary control on how much of
RQ1's independent variable exists.** Raising it to 120 m hands back ~18 points of
mission success by deleting the occlusion under study. That is a *construct
validity* argument, and it beats the W1 one it replaces in three ways — it is
measured on end-to-end policy performance rather than on a blockage statistic; it
does not depend on a scenario premise that could shift again; and it points at
the thesis's own independent variable rather than at a side condition.

Neither external limit binds: **TR 36.777 is valid to 300 m** and the EU open
category caps at **120 m AGL**, so the research constraint is tighter than both.
That is the right shape — the number is set by what the experiment needs, and
regulation merely fails to contradict it.

**State this in the methodology.** B0's 57 % is partly a consequence of the
altitude choice; the same controller scores 75 % at 120 m. Reporting the number
without the band is meaningless, and an examiner who asks "why 80 m?" now gets a
measured answer instead of a regulatory one.

### Shorter, coarser episodes (`dt = 0.5 s`, 240 steps, 120 s)
Proposed so that a standard `γ = 0.99, λ = 0.95` would fit the horizon. Rejected
on three separate grounds; full tables in [`BLOCK_D.md`](BLOCK_D.md).

1. **120 s truncates the escalation before it starts.** Routes reaching the
   3-hop regime (≥1400 m): **36.8 % at 240 s, 3.7 % at 120 s @ 0.4 s, 0.0 % at
   120 s @ 0.5 s.** 3-hop chains are already only ~4 % of steps at 240 s; at
   120 s F4's multi-hop rate-division rung would have nothing to act on.
2. **Changing `dt` means re-baking the frozen artefact.** A route step is a fixed
   *displacement*: at `dt = 0.5` without a re-bake the HVT slows from 5.8 to
   4.6 m/s, and *with* a re-bake `data/frankfurt_box.npz` changes and the joint
   calibration of `CONGESTION_FACTOR` / `MCV_MIN_REACH_M` against the escalation
   table is invalidated.
3. **It saves nothing.** PPO's rollout length is independent of episode length,
   so 10 M samples cost the same however they are partitioned. Shortening
   episodes changes the task, not the cost.

**The legitimate part of the proposal was γ**, and the fix is γ alone: **0.997**
(horizon 333 steps, 55 % of the episode) rather than 0.999 (horizon 1000), which
halves the value scale the critic must fit. Still inside the band AGENTS.md pins,
so it is a choice within the range, not a change to it. λ = 0.95 is already
skrl's `gae_lambda` default.

### mmWave instead of 3.5 GHz
Raised as a way to make blockage matter more. It would do the opposite of what
the thesis needs:

- **RQ1 becomes trivial.** mmWave is textbook blockage-limited; "occlusion
  matters at 28 GHz" is a lecture slide, not a finding. The result is interesting
  at sub-6 *because* the radius abstraction might plausibly have been safe.
- **Wrong radio for the platform.** Ptx = 30 dBm is justified from real tactical
  UAV MANET radios (Silvus, Doodle Labs, TrellisWare), all sub-6. mmWave is
  *less* realistic here, not more.
- **Different project.** At 28 GHz, FSPL at 1400 m is ~18 dB worse than at
  3.5 GHz, so nothing closes without beamforming array gain — which means
  modelling arrays and beam pointing, and beam alignment couples to the motion
  policy. There is also no aerial mmWave model with TR 36.777's standing.
- **Calendar.** It invalidates Block A's 103 tests, `PHYSICS.md`, the scenario
  sizing and Chapter 3 — which is writable *now* — before a freeze whose purpose
  is preventing exactly this.

Belongs in Chapter 7, where it strengthens the discussion for free: *if occlusion
dominates at 3.5 GHz, where diffraction still partly rescues blocked links, the
abstraction must be even less safe at mmWave* — a testable prediction.

### A minimum sensor depression angle
Would make the no-cue ablation airtight rather than caveated, and is cheap to
compute. Rejected: it fixes one unsourced constant (the 830 m range) by adding a
second, and a *binding* sensor parameter is precisely what
[`BLOCK_B.md`](BLOCK_B.md) identifies as confounding RQ1's fidelity ladder with
sensor specification. If that ablation ever needs hardening, run it as a
**sensitivity analysis** over two or three angles — stronger evidence than any
asserted value, and the same move `routing.py` already makes with `reuse_limit`.

A fixed downward camera cone is rejected separately: it models the wrong hardware
(the payload is gimballed) and duplicates the roofline-clearance constraint.

### A time / remaining-horizon feature in the observation
Proposed to fix value aliasing under the fixed 600-step truncation. Rejected
after separating the two cases Pardo et al. (2018) distinguish: time-awareness is
required for *time-limited* tasks, where the horizon is part of the problem, but
this mission is *time-unlimited* — 240 s covers the hop escalation, nothing about
the mission ends there. The correct treatment is partial-episode bootstrapping,
which Block A already chose (`reward.shaping`: *"truncation is not terminal —
bootstrap the value there instead"*). Observing the clock would let the policy
condition on an artificial horizon.

What replaces it is a **requirement on the wrapper**: keep `terminated` and
`truncated` distinct and bootstrap at truncation. Wrappers routinely collapse the
two, so Block D asserts it in the skrl smoke test.

### "1000 steps/s" as batched calls per second
Not a design direction so much as an ambiguity that had to be killed. The repo
stated the gate in two units differing by 1000×: `bench_occlusion.py` and
[`BLOCK_C.md`](BLOCK_C.md) print batched calls/s, while
[`THESIS_PLAN.md`](THESIS_PLAN.md) §3's budget is written in transitions
(10 M ÷ 1000/s ≈ 2.8 h/run × 45 ≈ 120 GPU-h). **The transition reading wins** —
it is the one the affordability argument is made in, and 10 M batched steps at
`num_envs = 1024` would be 10.2 billion samples per run, which nobody budgeted.

Consequence to accept: the 1000/s floor then clears even in eager mode
(1.8 × 1024 = 1843 env-steps/s), so the reported number becomes **wall-clock per
10 M-step run, end-to-end including the learner**. `torch.compile` stays
mandatory regardless — 73× applies to the 300–500 GPU-hours of development, and
unfused the slab chain holds ~8.8 GB of live intermediates that compete with the
learner for VRAM.

### Block F: gating the observation's channel features — the third sibling of decisions 1 and 2

[`BLOCK_F.md`](BLOCK_F.md) settles that fidelity gates the **channel** and never
the sensor (decision 1) or the diagnostics (decision 2). It does not say what the
**observation** does, and the observation *is* the agent's world — so the same
question arises there and it was answered the same way.

Six of the 108 observation dims are channel state rather than sensing. Three of
them already follow the rung for free, because they are computed from its
capacity matrix (`on_path`, e2e capacity, per-edge capacity). The other three did
not: the **measured noise floor**, the **clearance margin to the MCV**, and the
**per-edge clearance margin**.

**Decision: they follow the rung.** The rule, in one line: *sensor features
report the sensor, channel features report the channel model in force,
diagnostics report the truth.*

Three reasons, in order of weight:

1. **A radius simulator has no building data to put in an observation.** Under
   F0 the clearance features have no referent; reporting the true value reports a
   quantity that model does not possess.
2. **Ungated, F0's observation is internally contradictory** — an edge reporting
   74 Mbps beside a clearance feature reading −150 m. That contradiction is
   *learnable*, and in exactly the direction that would **understate** the
   F0 → F1 gap RQ1 exists to measure. A silent bias toward the null on the
   primary result is the worst available failure.
3. **It is not hypothetical.** `b0.py`'s `_update_repair` hill-climbs on
   `edge_clr` and `edge_cap` and is, in its own docstring, *"the only part of B0
   aimed at `chain_occluded`"*. Gated, that mechanism correctly goes inert under
   F0 — there is nothing to repair in a radius world. Ungated, it would keep
   repairing against buildings the channel never charges for.

**The alternative reading, recorded rather than hidden.** Buildings exist in the
world at every rung, so one could argue the drone has a *terrain* sensor and only
the radio is abstracted. That is coherent, and it is why the ego **clearance to
the HVT** stays on true geometry — it is the sensor's own ray, the one `sees_hvt`
is computed from, and gating it would put the soft flag and the hard gate into
disagreement. The split is between *sensing the target* and *sensing the link*,
not between "geometry" and "not geometry".

### Block F: `use_occlusion` was one flag doing two incompatible jobs

Block D left `EnvConfig.use_occlusion` labelled "the F0 seam". It is not one —
[`BLOCK_F.md`](BLOCK_F.md) decision 1 already establishes that it is an
all-geometry switch that also disables the sensor and the RQ1 diagnostic. What
the spec did not anticipate is that **the test suite depends on it for speed**:
it appears at eight sites as `FAST`, and occlusion is **37×** the rest of the
step on CPU (20 steps at `num_envs=8`: 0.031 s without, 1.154 s with). Deleting
it outright would have turned a 599-step energy test from 0.9 s into 35 s.

So it split in two rather than being renamed:

- **`fidelity`** decides whether occlusion costs the **link** anything. The
  sensor and the diagnostics always see the real geometry.
- **`no_buildings`** removes buildings from the **world**. Documented as
  not-a-rung, used by tests that are not about geometry, and it is also exactly
  the `F0-nogeo` variant decision 1 records as the defensible alternative
  reading of F0 — constructed as `fidelity="F0", no_buildings=True` and reported
  under that name.

The two are orthogonal on purpose. Folding `F0-nogeo` into F0 would confound the
primary result; leaving the fast path out entirely would have cost minutes of
suite time at every commit.

### Block F: `reuse_limit` demoted from a settable field to a derived property

`EnvConfig.reuse_limit` was a plain field defaulting to 3. Under the ladder it is
the F4 rung's defining flag, so leaving it settable would have made
`fidelity="F0", reuse_limit=3` constructible — a condition that is not on the
ladder, which nothing would stop running, and whose number would go into a table.
That is precisely what [`BLOCK_F.md`](BLOCK_F.md) decision 5 forbids.

It is now derived from `fidelity` (1 at F0–F3, 3 at F4). Both existing readers
(`evaluate.py`, `bench_env.py`) read `cfg.reuse_limit` and were unaffected.

`PHYSICS.md` still wants the main result under more than one duplexing
assumption, and that survives as **`duplexing_override`**, which raises anywhere
except `fidelity="F4"`. A duplexing robustness check is a statement about the
full model; at F0–F3 the rung already pins the divisor.

### Block F: `F0 ≥ F1` end to end — the spec understated what can be asserted

[`BLOCK_F.md`](BLOCK_F.md)'s correctness section groups `F0 ≥ F1 ≥ F2` together
and calls the pair "not guaranteed". The **first half is guaranteed**: F1's
capacity is F0's times an unoccluded mask, elementwise, and `best_relay_path` is
monotone in the capacity matrix at a fixed `reuse_limit` (the DP is a max of mins
and the answer a max over hop counts of monotone terms). So three orderings hold
end-to-end and are asserted, not two: **F0 ≥ F1**, F2 ≥ F3, F3 ≥ F4.

Only `F1 ↔ F2` is genuinely unordered, and only in one direction reliably: F2
drops the radius cutoff and turns occlusion from a hard veto into a blockage
penalty, so **F2 > F1** happens constantly. **F1 > F2** needs a link that is
inside `R`, unoccluded, and still below the modulation cap — and a clear link at
`R ≈ 500 m` runs ~30 dB above what 7.4 b/s/Hz needs, so at the calibrated `R` it
may not occur in a given rollout at all. The test therefore asserts the
mechanism at a deliberately wide `R` rather than asserting the sample.

### Device-independent episode sampling — considered, rejected

Found while moving Block F's measurements onto MPS. **`torch.Generator` produces
a different stream on each device**, so `_sample_episode` draws *different routes,
different initial charges and different cue noise* on MPS than on CPU **for the
same seed**. The physics is not the problem — the occlusion kernel is bit-identical
across CPU/MPS/compiled (`bench_occlusion.py`: 0.00e+00 m) and capacity agrees to
1.9e-5 Mbps on an identical state, with every discrete output (`sees_hvt`,
`on_edge`, `hop_count`, `chain_occluded`) exactly equal. The two devices simply
sample different episodes.

The obvious fix — draw the episode randomness on the host and copy it over — was
rejected: with `auto_reset=True`, `_sample_episode` runs on **every step**, so it
would put a host transfer and a sync in the training hot loop, which is exactly
what AGENTS.md's device rule and Block D's throughput gate exist to prevent.

**What follows instead is a reporting rule: a device is part of a measurement's
provenance.** Numbers from different devices are different route samples and must
not be compared as if they were the same experiment. `golden.py` already forces
`device="cpu"` for this reason.

### The second city for RQ2 — ⛔ cut, and the original costing was wrong

THESIS_PLAN specified RQ2's cross-morphology column as costing *"one extra OSM
extract, not extra training"*. **That costing was wrong, and it is the reason the
column is cut rather than merely deferred.**

The road graph would indeed be one extra OSM extract. The *buildings* would not.
Block B's whole height story rests on **Hessen's LoD2 INSPIRE WFS**, which is a
state service and covers no city outside Hessen. A second city therefore needs:

- a new height source, with the coverage gate re-run from scratch — and the
  OSM-tag fallback was already measured and **rejected** at 57–59 % area-weighted
  coverage, putting the Deutsche-Bank-Hochhaus at 22 m;
- the OBB fitting re-validated (the AABB-vs-OBB result is a property of
  *Frankfurt's* 38° median part orientation, not a universal one);
- route sampling, `MCV_MIN_REACH_M` and `CONGESTION_FACTOR` re-calibrated against
  the new box, since the escalation table is what pins them;
- a re-run of the occlusion map tests against a new artefact.

That is a **full Block B rebuild** for one evaluation column, with a March 2027
freeze in the way and Block G — the acknowledged stall point — not started.

**What is lost, and it should be stated rather than glossed.** Transfer across
*urban form* was the stronger of RQ2's two generalisation claims: a network could
plausibly memorise Frankfurt's layout in a way that off-`N` transfer would not
expose. What survives is transfer across *swarm size*, and Block E made that a
real test rather than a formality — B0 scores 36.4 / 57.2 / 74.3 % at N = 3/5/8
while `observed` stays flat at ~93 %, so the off-`N` columns measure **relay
scaling**, which is the thing RQ2 is about.

Two mitigations already exist and cost nothing, so quote them instead of the
missing column:

1. **The actor cannot see absolute position.** The 24 ego features contain no
   absolute coordinate except own altitude; everything else is relative or local
   sensing, so "the MCV is usually south-west" is not representable. Re-checked
   when Block D added the cue vector — see the MCV-quadrant entry above.
2. **The route bank is held out.** 256 of the routes are an eval split the
   policy never trains on, so within-city generalisation is already measured.

**Belongs in Chapter 7 as future work**, where it is a one-paragraph, genuinely
open question. Do **not** write RQ2 as if the architecture ladder had been tested
across cities.

### Block G: `GaussianMixin(clip_actions=True)` — ☠️ it inverts learning

Not a design proposal: a **library default that silently destroys the result**,
recorded here because it cost a day and nothing about it is visible in a return
curve.

skrl clamps the sampled action to the action space and then evaluates its
log-probability under the *unclamped* Normal, so every tail draw is recorded as
though it had landed exactly on ±1. The PPO ratio is then computed against a
density that does not describe how the actions were actually generated, the
policy is pushed to the corners of the action box, and the action standard
deviation rises **with `entropy_loss_scale = 0`** — which is impossible under a
correct gradient, and is the tell.

Measured: under a reward whose only non-zero weight was `w_mission`, so that the
return **is** the headline metric, mission-capable fell from 30 % to 4.6 % over
600 k steps. Learning rate, KL-adaptive scheduling and rollout length changed
only the speed of the collapse.

**`clip_actions=False`.** `core._advance_drones` already opens with
`actions.clamp(-1.0, 1.0)`, so the bound is enforced either way and only the
density changes. With the fix the same configuration reaches **74.8 % [12.7] against
random's 35.1 %** on stage 1 over five training seeds. Pinned by `src/models/test_actor.py`, which also
asserts the recorded log-probability is the one belonging to the action returned.

**The method is the reusable part**, and it generalises to the next time
something in Block G does not learn: build a reward whose optimum you know
(`-w_effort·‖a‖²`, optimum `a = 0`) and check PPO improves *it*. That cleared the
wrapper, the flattening, the bootstrap, GAE and the optimiser in one run, and
left the action distribution as the only suspect.

### Block G: the swarm trains as ONE parameter-shared agent, not `N` skrl agents

`SwarmMultiAgentWrapper` (Block D) keys every tensor per drone. That is the
natural reading of skrl's multi-agent API and it stays, because the observation
contract smoke tests are written against it — but it cannot be what a reported
run uses, for two structural reasons:

1. **RQ2's zero-shot transfer needs one policy.** The matrix trains at `N = 5`
   and evaluates at `N ∈ {3,5,8}`. Five per-drone policies cannot be evaluated at
   eight drones at all, so the transfer columns would not exist.
2. **Homogeneity is a claim this project makes.** `REWARD.md` requires the reward
   not to depend on agent index or "roles emerge rather than being assigned"
   collapses. A distinct network per drone breaks it from the other side.

The obvious route is a trap: handing skrl the *same* `Model` object under five
agent ids builds **five Adam optimizers over the same parameters** and runs five
sequential PPO updates per rollout, four of them computing ratios against
log-probabilities collected under a policy that has already moved.

`SharedPolicyWrapper` collapses the drones into the batch dimension — one
optimizer, one update, correct ratios. This is MAPPO **as published** (Yu et al.,
2022), not a weakened form of it: decentralized agent-local actors, one
centralized critic on the shared global state, parameters shared across
homogeneous agents. CTDE is unaffected; the actor still reads `observations` and
never `states`.

### Block G: one message-passing layer for the actor, not the two MODELS.md allows

`MODELS.md` argues depth from graph diameter over a graph of `N` drone nodes and
sets the ceiling at two layers. **The actor never holds that graph.** Its
observation is `(B, 108)` — its own ego block plus 7 neighbour slots — which is a
*star* centred on itself.

A second layer over the true swarm graph would give drone `i` access to `j`'s
aggregate of `k`: information `i` does not possess and could obtain only by
exchanging embeddings with its neighbours. That hands the GNN rung **strictly
more information** than the MLP and DeepSets rungs get, so RQ2's contrast would
confound architecture with information — the exact confound the zeroed-`e_ij`
design exists to rule out. Extra layers over the *local star* add no information
at all, only depth.

**One message-passing layer, with depth in the message and update MLPs.**
MODELS.md's own diameter-1 argument already says one layer reaches every drone;
the second layer it allows for is *unavailable* to an agent-local actor, not
merely unnecessary. Two-layer-with-communication belongs in Chapter 7, where it
is a different claim — about the control plane, not about relational structure.

### Block G: `time_limit_bootstrap=True` was bootstrapping the wrong state

A bug in the training seam, found by reading skrl 2.1.0's `record_transition`
rather than by a failing test — which is the point of recording it.

skrl acts on the flag by computing `gamma * V(next_observations, next_states)`
and adding it to the reward at truncation. With `auto_reset=True` the tensors
`step()` returns are already the **next episode's opening**, so the learner would
have valued an unrelated state at every truncation, at gamma = 0.997 on returns
of order 300. `time_limit_bootstrap=True` would have looked correct in the config
and in Block D's smoke test — which asserts the *flag*, not the *state* — while
doing the opposite of what it is set for.

`extras["final_observation"]` already existed; the critic's `final_state` was
being computed inside `step()` and thrown away. Both are now emitted under
`EnvConfig.training_extras`, **off by default**, so the golden trace's output
contract is untouched and no re-capture was needed. The training loop passes them
to `record_transition` instead of the step return, and
`SharedPolicyWrapper.final_states()` refuses with a message naming the flag
rather than silently handing back the wrong tensor.

### Block G: G1 split into G1a and G1b — the spec's build order was circular

`BLOCK_G.md` puts G1 first and asks for "wall-clock for a 10 M-step run
end-to-end **including the learner**" before building anything on top. There is
no learner until G2/G3, and `bench_env.py` measures the env alone — so the
headline number G1 exists to produce cannot be measured in G1.

**G1a** is Block D's pending env-only CUDA re-run and has no dependencies.
**G1b** is the end-to-end wall-clock and runs in the same GPU session, after the
trainer exists. The risk argument the ordering protects is unaffected: G1b is
still the first thing that happens on real hardware.

Provisional and on the wrong device — MPS, `num_envs = 256`, learner attached:
**20,519 env-steps/s → 0.14 h per 10 M-step run** against a ≤3 h target. A laptop
lower bound settles nothing, but the budget risk is not currently visible.

### Block G: the golden trace is architecture-specific, not just device-specific

`golden.FORCED_CFG` pins `device="cpu"` precisely so the frozen trace is not at
the mercy of a device's RNG stream. **That controls the device and not the
instruction set**, and float32 is not associative across instruction sets.

Measured 2026-08-24, the first time the suite ran on x86-64: identical commit,
identical seeds, `device="cpu"` on both machines, and the traces diverge —
`reset` (180 steps) by **1.4e-6**, `offn_eval` by **3.0e-6**, `design` (300
steps) by **2.4e-3**. ULP-level at the start, amplified by a closed loop in which
the recorded action depends on the state it just produced. The same commit passes
**exactly** on arm64, which is what rules out an environment change.

**Consequence, and it extends an existing rule.** `DECISIONS.md` already records
that *a device is part of a measurement's provenance*. So is the **architecture**.
`test_golden.py` now asserts bitwise equality only on `arm64` (where the artefact
was captured) and runs a weaker aggregate check elsewhere: mission-capable,
observed and hop-count *rates* within a tolerance far tighter than any effect
Block D or E reports. Last-bit divergence reshuffles which individual steps are
capable; it does not move the rate.

⛔ **Do not re-capture on x86 to make the test green.** That discards the only
record of the pre-Block-F environment, and the divergence is a real property of
float arithmetic rather than a defect. Run CI on arm64.

### `SAGEConv` for the GNN rung
☠️ **Never.** It cannot ingest edge features at all, so it would silently collapse
the GNN rung into the DeepSets rung and leave RQ2 measuring nothing — and it is
the layer people reach for by default.

---

## Still open

| Question | Blocked on |
|---|---|
| `τ_c`, `τ_l` retuning | **Block G**, deliberately not Block E. B0 now supplies the distributions the retune needs, but the thresholds live in the potential, so they can only affect *learning speed* — which cannot be measured until a learner exists. Retune once, in G, against the thing they act on |
| Local height raster | whether the policy is visibly blind without it. **Block G** — the clearance margins shipped in Block D are the cheap half, and the raster is only worth building if a learned policy demonstrably cannot anticipate without it ([`ENVIRONMENT.md`](ENVIRONMENT.md) → Terrain) |
| ~~Second city for cross-morphology transfer~~ | ⛔ **CUT 2026-08-23** — see the entry below. Not deferred: dropped, and RQ2's generalisation claim is now swarm size only |
| `830 m` recognition / `2.8 km` detection range | **unverified — no derivation exists in this repo.** Measured to be non-binding (99.8 % of sightlines are shorter), so results are insensitive to it; if a defensible number is ever needed, derive it from a stated camera rather than assert it. Same standing as the `TODO(verify)` constants |
| MCV spawn diversity — **investigated, no action** | see below |
| ~~Why one route lingered 333 steps on a ~240 m bridge~~ | ✅ **closed in Block D.** Measured over the whole bank (`measure_envelope.py --only route`): longest near-stationary run is **1 step**, p90 1, no route stalls >50 steps, slowest route still averages 5.77 m/s. `grow_outward` does not stall — the 333 steps were the bridge decks, and those are gone |
| ~~F3's jammer switch must NOT be `jammer_on`~~ | ✅ **CLOSED in Block F.** `channel_jammer` is derived from the `fidelity` enum and multiplied in *alongside* the curriculum tensor, never instead of it. `test_fidelity.py` asserts the curriculum's `jammer_on` / `speed_scale` / `episode_len` / `route_id` draws are bit-identical across all five rungs at a fixed seed, and that jam power is identically zero below F3 and strictly positive at or above it |
| ~~Value preprocessing for the critic~~ | ✅ **CLOSED in Block G.** `value_preprocessor=RunningStandardScaler` is applied by `training.skrl_wrapper.mappo_cfg()`, `size=1` (it normalises the value *output*; the critic's global state is already unit-scaled by `core._critic_state`, so no `state_preprocessor` is set). One wrinkle: skrl keeps the scaler's running moments in **float64**, which MPS cannot allocate, so `Float32RunningStandardScaler` is selected by device — CUDA keeps float64, because over a 10 M-step run the parallel-variance update adds `delta * count / total` with `total` near 1e8, where a float32 increment can vanish |
| `tau_c`, `tau_l` and Phi's scale — the retune | **still Block G**, and now possible: a learner exists. Untouched so far, deliberately — the anti-learning bug had to be cleared first, or the retune would have been fitted to a broken gradient |
| ~~Is the 3-hop regime under-exercised?~~ | ✅ **CLOSED in Block E — no.** The framing was wrong twice over. (1) Chains of **4 and 5 hops were never counted**, and `routing.py`'s divisor is `min(n, 3)`, so they are charged exactly like 3-hop ones — the regime that matters is ≥3 hops, not exactly 3. (2) The denominator included the 59 % of steps where *nobody is observing*, so no chain exists at all. Under B0 on the eval split, conditioned on a chain existing: **multi-hop 80.5 % overall and 95.6 % in the last third, with the divisor saturated at 3 on 54.2 % of late chain-steps.** Against the 4.2 % that caused the alarm that is an order of magnitude. No change to the box or the escalation — [`BLOCK_E.md`](BLOCK_E.md) §6 |
| ~~Expect F3 → F4 to be a null~~ | ✅ **SUPERSEDED, twice.** The null was a 5 Mbps-era prediction; Block E re-measured it at 15 Mbps as **+26.5 pp**, and Block F reproduced it independently under the ladder at **−27.1 pp** (F3 83.1 % → F4 56.0 %). The divisor is the rung that makes the mission hard. **Do not carry the null prediction forward** — [`BLOCK_F.md`](BLOCK_F.md) |
| ~~Mission success saturates at N=5 under F4~~ | ✅ **SUPERSEDED by the rate change.** The 93.2 %-with-7 pp-headroom figure is 5 Mbps-era. At 15 Mbps B0 reaches **57.2 %** against a 93.0 % sensor ceiling, so the headline metric has ~36 points of headroom and it is all relay geometry. The companion metrics named there (time-to-first-capable, 5th-percentile capacity, N = 3) remain worth reporting on their own merits |
| Where should RQ2's off-N weight go? | ✅ **N = 8, not N = 3** — see the entry above. The relay premise binds hardest at N = 3 (55 % of in-sight steps fail), but *control* is worth only +3.2 pp there against +25.9 pp at N = 8. Hardness is not headroom |
| ~~Is RQ3's handoff phenomenon frequent enough to study?~~ | ✅ **RESOLVED in Block E — by changing which phenomenon RQ3 studies.** Observer handoff is ~0.9/episode and too thin; relay-chain reconfiguration is ~52/episode and is driven by the occlusion physics RQ1 is about. RQ3 re-pointed; E3a's ablation changed with it. See the entry above and [`THESIS_PLAN.md`](THESIS_PLAN.md) |
| Verifying TR 36.777 and rotorcraft constants against primary sources | you, with the actual documents — **do not cite numbers an AI produced** |
| ~~The altitude ceiling's citation~~ | ✅ **discharged in Block D.** The ceiling is 80 m and *derived* from W1 — above it a best-placed single drone can do the mission alone, which dissolves the swarm premise. No regulatory citation is load-bearing any more; civil UAS limits are corroboration. See [`BLOCK_D.md`](BLOCK_D.md) |
