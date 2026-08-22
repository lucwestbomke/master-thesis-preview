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

### `SAGEConv` for the GNN rung
☠️ **Never.** It cannot ingest edge features at all, so it would silently collapse
the GNN rung into the DeepSets rung and leave RQ2 measuring nothing — and it is
the layer people reach for by default.

---

## Still open

| Question | Blocked on |
|---|---|
| `τ_c`, `τ_l` retuning | a running env (Block D) — safe to change, they live in the potential |
| Local height raster | whether the policy is visibly blind without it (Block D/G) |
| Second city for cross-morphology transfer | candidates London City (similar structure, different topology) or Barcelona (maximum contrast). Note LoD2 is a *Hessen* service — a second city needs its own height source, and the coverage gate must be re-run |
| `830 m` recognition / `2.8 km` detection range | **unverified — no derivation exists in this repo.** Measured to be non-binding (99.8 % of sightlines are shorter), so results are insensitive to it; if a defensible number is ever needed, derive it from a stated camera rather than assert it. Same standing as the `TODO(verify)` constants |
| MCV spawn diversity — **investigated, no action** | see below |
| ~~Why one route lingered 333 steps on a ~240 m bridge~~ | ✅ **closed in Block D.** Measured over the whole bank (`measure_envelope.py --only route`): longest near-stationary run is **1 step**, p90 1, no route stalls >50 steps, slowest route still averages 5.77 m/s. `grow_outward` does not stall — the 333 steps were the bridge decks, and those are gone |
| Verifying TR 36.777 and rotorcraft constants against primary sources | you, with the actual documents — **do not cite numbers an AI produced** |
| **The 120 m altitude ceiling's citation** | same standing. The band is fixed on measurement (above), but the ceiling wants an external basis the way Ptx and bandwidth have one. The civil UAS operating limit is the natural source and is *believed* to be 120 m AGL — **no regulation text has been read for this repo.** `TODO(verify)` |
