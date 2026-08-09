# AGENTS.md — UAV Swarm MARL Thesis

## Mission
A swarm of `N` UAVs must simultaneously **observe** a moving ground High-Value
Target (HVT) in a real city, **relay** the resulting sensor feed back to a fixed
Mobile Command Vehicle (MCV) over a multi-hop chain at ≥5 Mbps end-to-end, and
**survive** on finite batteries while a jammer mounted on the HVT raises the
noise floor around it. Buildings block LoS, so the relay chain is geometrically
necessary — no single drone can usually see both the HVT and the MCV.

The full research design (research questions, hypotheses, baselines, metrics,
timeline) lives in [`docs/THESIS_PLAN.md`](docs/THESIS_PLAN.md). **Read it before
making design decisions.** Short version:

- **RQ1 (primary):** which physical effects must a channel model include for
  learned policies to transfer? Train at fidelity **F0** (connectivity radius,
  `R` calibrated to F3's median link range) → **F1** (+occlusion) → **F2**
  (+jammer) → **F3** (full SINR/rate/multi-hop division); evaluate *all* under
  F3. Hypothesis: the gap is dominated by **occlusion**.
- **RQ2:** MLP → DeepSets → GNN ladder; zero-shot transfer across `N ∈ {3,5,8}`
  **and** across city morphology.
- **RQ3:** does observer/relay role rotation emerge? Two drivers ablated
  separately: `-λ·Var(B)` and jammer exposure.

> **Action space is motion only (3-dim).** Transmit power is fixed at 30 dBm.
> Three independent justifications for adaptive Ptx — energy, interference,
> detectability — were each tested numerically against fair baselines and each
> came out null. Do not reintroduce it as a control dimension without reading
> [`docs/NEGATIVE_RESULTS.md`](docs/NEGATIVE_RESULTS.md); condition E4 keeps it
> only to reproduce the null empirically.

## Stack (do not substitute without asking)
- Python 3.12, PyTorch. Dependency management via **uv** (`uv.lock` is authoritative).
- Custom **batched tensor env** with a PettingZoo `ParallelEnv` adapter — NOT
  Isaac Sim / Isaac Lab (deliberately rejected: too heavy for the timeline,
  slower iteration, and occlusion needs geometric ray/polygon checks rather than
  rigid-body physics).
- skrl for MAPPO (CTDE: centralized critic, decentralized actors), driven through
  a **custom multi-agent wrapper**, not `PettingZooWrapper` — see Device rules.
- PyTorch Geometric (PyG) for GNN actor/critic, dynamic per-timestep graph
- osmnx + shapely for OSM building footprints and road network — **offline only**
- Weights & Biases for run tracking, incl. periodic rendered eval videos
- Hydra or plain YAML for configs

## Explicitly rejected — do not reintroduce without discussion
- **NVIDIA Sionna** in the training loop: TensorFlow-based; crossing framework
  boundaries every env step breaks the stay-in-VRAM design. Closed-form PyTorch
  path loss is used instead. Sionna may be used **offline** to validate the
  closed-form model — never inside `step()`.
- **stable-baselines3, Ray/RLlib, OmniDrones, SUMO, NS-3:** wrong paradigm or
  redundant with the above.
- **EW detectability / EMCON modelling:** interesting, but widens the mission
  objective past what a 5-month thesis can defend. Future work only.

---

## Physics / math — implemented and unit-tested

Implemented in [`src/env/channel.py`](src/env/channel.py) and
[`src/env/routing.py`](src/env/routing.py), with hand-computed assertions in the
co-located test files. **Do not change these formulas without updating the tests
and checking against the cited standard** — they appear in the methodology
chapter and must stay traceable.

### Link classes — one model does not fit all
| Link | Model | Why |
|---|---|---|
| Drone ↔ drone (A2A) | FSPL + 20 dB blockage penalty when occluded | Both endpoints are above rooftop; a ground street-canyon model does not describe this at all. |
| Drone ↔ HVT / MCV (A2G) | **3GPP TR 36.777 UMi-AV** | TR 38.901 UMi is specified for UE heights 1.5–22.5 m and is **not valid for aerial nodes**. |
| Jammer → drone | Same A2G UMi-AV | Jammer is ground-mounted on the HVT. |

> ⚠️ The TR 36.777 coefficients in `channel.py` are marked `TODO(verify)`. Check
> them against the actual 3GPP document before citing. Same for the rotary-wing
> energy constants.

### SINR — linear domain, with intra-swarm interference
```
SINR_lin(i→j) = P_rx(i→j) / ( Σ_{k∉{i,j}, k active} P_rx(k→j) + P_jam(j) + N0 )
SINR_dB       = 10·log10(SINR_lin)
```
Interference and noise sum in the **linear** domain. The earlier spec had
`SINR_dB = P_sig − (P_jam + N0)`, which adds two dBm quantities — a product, not
a sum — and returned ~+100 dB for realistic urban links, silently deleting the
jammer from every experiment. A regression test pins this.

Node `j`'s own transmission is excluded via a zeroed diagonal — half-duplex, it
does not receive its own emission.

**`tx_mask` carries the MAC assumption — set it deliberately.** The routing
divisor `min(n_hops, 3)` presumes a spatial-reuse TDMA schedule, under which a
≤3-hop chain never has two hops active at once. So when evaluating a link, the
mask must contain only the transmitters active *in that slot* — for short chains,
one node, and SINR reduces to signal over jammer-plus-noise. Passing every node
while also applying the divisor double-counts the half-duplex cost, and made a
feasible 3-hop chain look infeasible during scenario design. The
uncoordinated-access mode (all nodes concurrent, no divisor) stays available for
worst-case analysis. Pinned by tests.

### Noise floor — derived, never hardcoded
```
N0_dBm = -174 + 10·log10(B_Hz) + NF_dB        # B=10 MHz, NF=7 dB → -97.0 dBm
```

### Rate — Shannon with implementation loss and a modulation cap
```
SE     = min( 0.75 · log2(1 + SINR_lin), 7.4 )   b/s/Hz
C_Mbps = B_Hz · SE / 1e6
```
Unbounded Shannon reports throughput no real radio delivers.

### Multi-hop end-to-end capacity and routing
```
C_e2e = min_i(C_i) / min(n_hops, 3)            # half-duplex with spatial reuse
```
Half-duplex relays on one channel must be scheduled, but hops far enough apart
transmit concurrently, so a linear chain saturates near **1/3** of single-link
capacity rather than degrading as `1/n` (Li et al., MobiCom 2001; cf. Gupta &
Kumar 1999).

> A `/n_hops` divisor **plus** full concurrent interference double-counts: `/n`
> is the pure-TDMA schedule, in which only one hop is active and there is no
> intra-chain interference to charge. The two cannot both be true. `min(n, 3)`
> is the form consistent with the interference model in `channel.py`.

Short chains are still preferred — that pressure now comes from physics rather
than an arbitrary factor: every extra hop is another concurrent transmitter
raising everyone's noise floor, and must itself clear the SINR bar.

`reuse_limit` is a **parameter, not a constant** (`=max_hops` recovers strict
TDMA, `=1` removes the penalty). Report the headline result under at least two
duplexing settings — it converts a soft modelling assumption into a robustness
check.

Path selection maximises `min_i(C_i)/min(n,3)` via a hop-limited widest-path DP:
```
W[h][j] = max_i min( W[h-1][i], C[i][j] )      # answer: max_h W[h][dst]/min(h,3)
```
Sources are all drones currently holding a valid HVT observation; if none, mission
capacity is 0. Fully batched, exact, no per-env Python loop.

### Bandwidth and threshold — chosen so the constraint actually binds
`B = 10 MHz`, threshold `5 Mbps` end-to-end. A 3-hop chain then needs **+4.8 dB
SINR per hop**. At the originally-specified 20 MHz a single hop needed only
−7.2 dB, which a swarm satisfies by accident and which makes the jammer
decorative.

### Scenario — derived, not chosen
Every parameter is fixed from an external source, and the operating area is then
*solved for* so that a single drone fails while the swarm succeeds. Regenerate
with [`scripts/scenario_design.py`](scripts/scenario_design.py) and
[`scripts/link_budget_check.py`](scripts/link_budget_check.py);
`tests/test_scenario_sizing.py` pins the trade-off table.

| Parameter | Value | Basis |
|---|---|---|
| City | **Frankfurt**, 1500 m box over Bankenviertel + fabric | heterogeneous: low-rise gives a workable observation envelope, towers block A2A |
| Operating area | **1500 m** | solo drone manages ~1.7 Mbps (fails); swarm ~24 Mbps (feasible) |
| Ptx | **30 dBm, fixed** | UAV tactical MANET radios are 0.5–2 W |
| Jammer, in-band | 30 dBm | vehicle C-UAS barrage emitter |
| Flight altitude | 80 m nominal | above fabric, below towers; inside TR 36.777's 22.5–300 m band |

> ⚠️ **Never raise Ptx to make the energy term measurable.** At 40 dBm a
> *blocked* A2A link still carries 15 Mbps over 2.8 km, so one drone spans any
> simulable map and the relay chain becomes unnecessary. Range grows with power
> far faster than the mission area can absorb.

### Observation envelope — an angle constraint, not a distance one
The ray must clear the roofline, which fixes an elevation angle (~66° for
Frankfurt), not a range:
- **across-street:** within `(W/2)·h/H_b` — 36 m at 80 m altitude, 91 m at 200 m.
  Flying higher buys lateral freedom.
- **along-street:** the roofline never blocks; the sensor limits instead
  (~830 m to recognise a vehicle, ~2.8 km to detect one).

So the envelope is a wedge down the street plus an overhead cone — **not a
36 m disc**. Compute it from real footprints, never from a radius.

### Energy
```
P_total = P_flight(‖v‖) + κ·‖a‖² + P_tx_DC
P_tx_DC = 10^(Ptx_dBm/10)/1000 / η_PA + P_circuit          # η_PA ≈ 0.25
```
`P_flight` is the **rotary-wing model of Zeng, Xu & Zhang (2019)**, U-shaped with
a minimum near 10–15 m/s. The earlier `α‖v‖²` form claimed hovering is cheapest,
which is false for rotary-wing UAVs — and energy drives the role-rotation result
in RQ3, so it cannot rest on a model that rewards hovering when reality does not.
`κ‖a‖²` is an explicit **control-effort heuristic**, not physics; present it as
such.

`P_tx_DC` is a **constant** (Ptx is fixed at 30 dBm), so it shifts the budget by
~1.6 % and does not vary with the action. Keep it in the accounting for
completeness — including `P_circuit`, the always-on radio front end at ~2–5 W —
but flight energy is what the policy actually controls.

### Graph and reward
- GNN edge weight (continuous, no hard cutoff — avoids gradient cliffs):
  `E_ij = sigmoid((C_ij − 5.0) · gamma)`
- Reward: tracking quality + capacity − energy − `λ·Var(B_1..B_N)`
  **− idle penalty per step with no HVT observation** (see below). The variance
  term forces role **rotation** rather than a fixed division of labour. Watch for
  its degenerate optimum (all drones hover ⇒ variance 0) — the tracking and
  capacity terms must dominate.
- Jammer is mounted on the HVT and moves with it.
- HVT route: randomized valid path over the real OSM road graph, resampled per
  episode. Speed ~10 m/s. Fixed routes only for early debugging.

### Episode structure — launch, cue, termination

**Launch.** Drones start parked on the MCV and fly out to deploy. The relay chain
forms during transit; this is a phase of the mission, not a preamble to it. It
also creates the energy tension — flying out costs battery, so the swarm cannot
send everyone everywhere.

**Cue.** The HVT's position is *cued* with error, not known and not searched for
blind. `σ ≈ 150 m`, refreshed every ~10 s until the swarm acquires it directly —
justified as an intermittent external ISR source. The 150 m is set against the
sensor envelope: from the cued point the target is within along-street detection
range with high probability, so acquisition is likely rather than lucky.

> **Do not make this a blind search.** Exploration is RL's weakest point; a sparse
> "found it" reward over 1500 m² would dominate the learning signal and swamp
> everything the thesis is actually about. Acquisition difficulty arises for free
> anyway: transit takes ~100 s and the HVT covers up to a kilometre in that time,
> so the cue is stale on arrival.

Curriculum axis, free of charge: `σ` 0 → 50 → 150 m, refresh continuous → 10 s
→ 30 s.

**Termination — mission failure must NOT terminate the episode.** Two independent
failure modes if it does:

1. *Termination hacking.* If the link requirement starts only at acquisition, the
   optimal policy is to never acquire, never fail, and loiter.
2. *An undesigned curriculum.* Under the old rule (`C_e2e < 5 Mbps` for >5
   consecutive steps), a random initial policy dies around step 6 and the agent
   only ever experiences the first six steps. It cannot learn to track because it
   never reaches the tracking phase.

So:
- **Fixed-length episodes** (truncation), ~600–1200 steps at `dt = 0.25–0.5 s`.
- **Battery exhaustion still terminates** — physical, and unhackable, since
  hovering at the MCV burns power too.
- **Mission failure is a per-step condition**, feeding reward and metrics. The
  chain may drop and re-form, which is what real missions do.
- **Per-step idle penalty** whenever the HVT is unobserved, so loitering accrues
  unbounded negative reward and "never acquire" is strictly worse than trying.

Primary metric becomes **fraction of steps mission-capable** rather than survival
time — richer signal, and it cannot be gamed by refusing to start.

---

## Observations

**Rule: the actor may only see what a real drone could sense or receive.** Global
state belongs to the critic. Violating this quietly turns decentralized execution
into centralized execution and invalidates the whole CTDE framing.

### Actor — ego features (21)
| Feature | Dims | Realizable from |
|---|---|---|
| own velocity | 3 | INS |
| own altitude | 1 | absolute — LoS geometry depends on it |
| battery | 1 | |
| sees HVT (soft flag) | 1 | own sensor |
| relative vector to HVT | 3 | own sensor; zeroed when not seen |
| **relative velocity of HVT** | 3 | own sensor — without this the drone cannot anticipate |
| relative vector to MCV | 3 | MCV position is fixed and briefed |
| measured noise floor | 1 | **how the drone senses the jammer** |
| clearance margin to HVT | 1 | signed metres the ray clears the roofline |
| clearance margin to MCV | 1 | ditto |
| on active relay path | 1 | routing layer |
| current e2e capacity | 1 | reported back down the chain |
| steps since link last OK | 1 | proximity to episode failure |

### Actor — per-neighbour features (9 × N−1)
Relative position (3), relative velocity (3), their battery (1), whether they see
the HVT (1), whether they are on the path (1). All standard MANET position
reporting.

### Edge features (2)
Link capacity `C_ij` and the ray's clearance margin. **This is the only input the
GNN has and DeepSets does not** — it is precisely the rung RQ2 tests.

### How many neighbours — all of them, softly gated
`N−1 ≤ 7`. The graph is **fully connected in the tensor**, with influence scaled
by `E_ij = sigmoid((C_ij − 5.0)·γ)`. A neighbour behind a tower gets weight ≈0 and
its message is suppressed.

Not top-K, not a hard link-quality cutoff: a hard cutoff creates a gradient cliff
when a neighbour flickers across the threshold, and changes tensor shape per
timestep, which wrecks batching. Soft weights give the same effect with a smooth
gradient and a fixed shape.

### Terrain — clearance margins first, raster only if needed
Nothing above tells the drone a tower is *in the way* before a link degrades, so
it can react but never anticipate.

**Clearance margins (already listed) are the cheap half.** Signed metres by which
a ray clears the roofline — negative is blocked, positive is clear with margin.
Free from the slab-intersection code, and smooth where a boolean is a cliff.

**Local height raster is the optional half.** 24×24 cells at 20 m (a 480 m box),
into a 2–3 layer CNN → 64-dim embedding. Two things make this work:

- **Encode height relative to own altitude**, clipped: a cell reading `+100`
  means "something 100 m above me — I cannot see through it". This makes the
  representation **altitude-invariant**, which is a strong inductive bias and
  should help cross-city transfer by preventing the network from memorising
  Frankfurt's absolute heights.
- **Precompute one global grid** (1500 m / 20 m = 75×75) offline in
  `prep_osm.py`; at runtime each drone's patch is a batched **crop/gather**. No
  per-drone rasterization, no shapely, stays on GPU.

**Build order: margins first, raster only if the policy is visibly blind.** This
defers real work and yields a free ablation — *does spatial awareness of buildings
help, or do local sightline measurements suffice?*

### Critic — centralized, training-only
Sees global state: all drone states, HVT position and velocity, the full link
matrix. Two consequences:

- It **does not need to be size-agnostic**. Zero-shot transfer to `N ∈ {3,8}` runs
  the actor alone; the critic is discarded at evaluation. A plain MLP over
  concatenated global state is fine.
- Keep the critic **identical across all three architecture conditions**. If only
  the actor varies, RQ2 isolates the actor. If both vary, it is confounded.

---

## Model architectures

> Layer choice is now settled (custom MPNN — see below). Widths and depths remain
> hyperparameters for the equal-budget search, not findings.

### The ladder isolates one factor per rung
| | Neighbours read as | Permutation-invariant | Size-agnostic | Uses link quality |
|---|---|---|---|---|
| Flat MLP | concatenated vector, max-N padded + masked | ✗ | ✗ | ✗ |
| DeepSets | `ρ(Σᵢ φ(xᵢ))` — shared embed, then pool | ✓ | ✓ | ✗ |
| GNN | same, messages weighted by `edge_weight` | ✓ | ✓ | ✓ |

MLP → DeepSets isolates permutation invariance. DeepSets → GNN isolates the
*relational* part, which is RQ2's actual claim. Comparing a GNN only against a
flat MLP conflates the two and is the weaker experiment.

The MLP needs **max-N padding plus masking** or it cannot be evaluated off-N at
all, which would rig the transfer comparison toward the GNN.

### Layer choice — the edge features are the whole point
RQ2's GNN rung exists **only** to test whether link quality should modulate who a
drone listens to. If the layer cannot ingest edge features, the GNN rung silently
becomes the DeepSets rung and RQ2 measures nothing.

| PyG layer | Edge features | Verdict |
|---|---|---|
| `SAGEConv` (GraphSAGE) | **none** | ☠️ **Never use here.** Collapses GNN into DeepSets. This is the default people reach for. |
| `GCNConv` | scalar weight, degree-normalised | Poor fit — the normalisation assumes a different graph structure |
| `GATv2Conv` | ✓ via `edge_dim` — enters the attention weights | Good fit |
| `NNConv` | ✓ — edge features generate the message weight matrix | Expressive but the hypernetwork emits 256×256 values. Expensive. |
| `GINEConv` | ✓ additive only (`x_j + e_ij`) | Cheap, blunt |
| `TransformerConv` | ✓ | Heavier than this graph needs |

**Decision: a custom layer on PyG's `MessagePassing` base**, with
`message(x_i, x_j, e_ij) = MLP([x_i, x_j, e_ij])`.

This is *not* inventing an architecture — it is the standard MPNN formulation of
Gilmer et al. (2017), ~20 lines on top of PyG, and fully citable. It is preferred
here because **it makes the ablation exact**: the DeepSets rung is the identical
layer with `e_ij` zeroed. Same code path, same parameter count, same optimiser,
one input masked. No confound is possible. Two differently-named layers would
always invite "maybe GATv2 is just a better layer."

Fallback if an off-the-shelf named layer is preferred: **`GATv2Conv` with
`edge_dim=2`**. Attention fits conceptually — "how much should I listen to this
neighbour" is exactly what link capacity says — and GATv2 (Brody et al., 2022)
fixed the static-attention flaw in the original GAT, so it is the right citation.

### Rules that keep the comparison honest
1. **Do not invent an architecture.** Either the MPNN formulation above or a
   citable PyG layer. Designing a novel GNN is a different thesis.
2. **Equal hyperparameter budget** across all three, and say so in the
   methodology. Tuning the GNN harder than the baselines is the single most
   likely way this result gets dismissed.
3. **Match parameter counts** to within ~20 %, so the comparison is not
   capacity-vs-capacity.
4. **Sanity floor:** any architecture must beat a random policy and at least
   match the B0 scripted heuristic. Failing that is a bug, not a finding.

### Depth follows graph diameter — and "layer" means two different things
Do not confuse these:

- **Message-passing layers** = how far information travels across the graph. One
  layer reaches direct neighbours; two reaches neighbours-of-neighbours. Nothing
  to do with capacity.
- **MLP hidden layers** = ordinary network depth, inside each message-passing
  layer and in the heads. This is where capacity lives.

The graph is softly fully connected at `N ≤ 8`, so its diameter is **1**: after
one message-passing layer every drone has already heard every other. A second
layer buys two-hop relational structure. A third propagates nothing new and
causes **over-smoothing**, where all node representations converge — a documented
GNN failure mode, not a rule of thumb.

So **2 message-passing layers** is the ceiling the graph justifies, while width
stays normal. A reasonable build:

| Component | Shape | Params |
|---|---|---|
| Ego encoder | 21 → 256 → 256 | ~70k |
| Message function φ (×2 layers) | (256+256+2) → 256 → 256 | ~400k |
| Policy head | 256 → 256 → 6 | ~67k |
| **Total actor** | | **~550k** |

Width is a hyperparameter and belongs in the equal-budget search; 256 is the
starting point, not a finding.

### Expect a null on the in-distribution rung
At `N=5` the graph is tiny and GNN ≈ DeepSets is a plausible outcome. The
interesting result lives in the **off-N and cross-city transfer** columns. A
clean null, reported as such, is still a contribution.

---

## Device / performance rules

Training tensors live on `cuda:0`. **Never call `.cpu()`, `.numpy()`, or
`.item()` inside the env `step()` or the training hot loop** — `.item()` forces a
GPU sync and is the easy one to miss.

Local dev is Apple Silicon (CPU/MPS), toy configs only (2–3 agents, tiny building
set), for correctness debugging. Real training runs on a rented CUDA GPU (RunPod).
Guard device selection; never silently degrade a real training run to CPU.

**Two verified consequences that shape the architecture:**

1. **skrl's `PettingZooWrapper` round-trips every action and observation through
   NumPy on each step** (`untensorize_space` / `tensorize_space`) and exposes
   `num_envs == 1`; its vectorized paths are Isaac Lab-only. So: the env core is
   **batched with a leading `num_envs` dimension**, a thin PettingZoo adapter sits
   on top for API-compliance tests and single-env visual debugging only, and
   training uses a **custom skrl multi-agent wrapper** written against the batched
   core.

2. **osmnx / shapely are CPU-and-NumPy-only.** They are used **offline** in
   `scripts/prep_osm.py` to bake buildings into a tensor of boxes. Runtime
   occlusion is vectorized segment-vs-box intersection (slab method) in pure
   torch. Buildings are 2.5D — check the segment's altitude across the 2D
   intersection interval, not just a planar crossing.

**Throughput target: ≥1000 env-steps/s batched on GPU.** This is a gate, not an
aspiration — measure it before building anything on top of the env.

---

## Structure & conventions
- `src/env/` — batched env, occlusion geometry, channel model, routing
- `src/models/` — GNN / DeepSets / MLP actor-critic (PyG)
- `src/training/` — skrl wrappers, training entrypoints
- `configs/` — YAML per experiment condition
- `scripts/` — one-off data prep (OSM ingestion/caching)
- `tests/` — **cross-module/integration only**. Unit tests are co-located: the
  test for `src/env/occlusion.py` is `src/env/test_occlusion.py`.
- Naming: `hvt`, `mcv_base`, `tracker`, `relay`, `sinr_db`, `capacity_mbps`,
  `edge_weight`, `ptx_dbm` — keep tactical/telecom terms consistent.
- Observations: 21-dim ego, 9-dim per neighbour, 2-dim per edge — full breakdown
  in the "Observations" section above. Keep the actor **agent-local**; global
  state belongs to the critic, not the actor.

## Build / test
```bash
uv sync                                          # uv.lock is authoritative
uv run pytest                                    # tests
uv run ruff check . && uv run ruff format .      # lint / format
```

## Boundaries
- **No new heavy dependencies** (sim engines, RL frameworks) without flagging
  first — the stack was chosen deliberately for the timeline.
- **Do not change path-loss / SINR / capacity formulas** without checking against
  the cited standard and updating the hand-computed tests. These are cited in the
  methodology chapter.
- **Multi-seed runs (≥5 seeds per condition) for anything reported as a finding.**
  Report median + IQR, not mean ± std — RL returns are not normally distributed.
  Never report single-run numbers.
- **Freeze the environment at the end of March 2027.** Results before the freeze
  are pilots; results after are thesis material. Do not mix them.
- **Do not sweep six reward weights.** Fix α, β, ω, γ and the threshold from
  physical reasoning and document the choice; sweep `λ` only.
