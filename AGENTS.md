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

- **RQ1 (primary):** does joint control of motion *and* transmit power beat
  motion-only control with fixed Ptx, at equal energy budget? Hypothesised
  mechanism: **interference management**, not energy saving.
- **RQ2:** MLP → DeepSets → GNN ladder, plus zero-shot transfer across `N ∈ {3,5,8}`.
- **RQ3:** does tracker/relay role rotation emerge, and does `-λ·Var(B)` cause it?

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

All active drones share one band (full spatial reuse, worst case). Node `j`'s own
transmission is excluded via a zeroed diagonal — half-duplex, it does not receive
its own emission.

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

### Operating area and Ptx ceiling — must be co-designed
**The relay chain has to be geometrically necessary.** If one drone can observe
the HVT and still reach the MCV across the whole map, there is no multi-hop
problem and no thesis. Ptx and map size therefore cannot be chosen independently.

Run [`scripts/link_budget_check.py`](scripts/link_budget_check.py) before
committing to either; `tests/test_scenario_sizing.py` pins the resulting table.
Current verdicts (fc 3.5 GHz, B 10 MHz, 3-hop, 5 Mbps):

| Map | Ptx 10 | Ptx 20 | Ptx 30 | Ptx 40 |
|---|---|---|---|---|
| 300 m | infeasible | **trivial** | **trivial** | **trivial** |
| 600 m | infeasible | contested | **trivial** | **trivial** |
| 1200 m | infeasible | infeasible | contested | **trivial** |
| 2000 m | infeasible | infeasible | contested | **trivial** |

> ⚠️ **40 dBm is unusable at any simulable scale** — a *blocked* A2A link at
> 10 W still carries 15 Mbps over 2.8 km and 5 Mbps over 6.3 km. The viable band
> is roughly **600 m @ 20 dBm** or **1200–2000 m @ 30 dBm**.

**Consequence for RQ1:** the telecom *energy* term cannot be made large by
raising Ptx — range grows with power far faster than the mission area can absorb.
RQ1's claim is therefore about **interference management and throughput**, not
about saving watts on the radio. Energy stays in the reward and stays equalised
across conditions, but it is the constraint, not the claim. Flight energy (>90 %
of the budget) still drives the tracker/relay rotation story in RQ3.

### Energy
```
P_total = P_flight(‖v‖) + κ·‖a‖² + P_tx_DC
P_tx_DC = 10^(Ptx_dBm/10)/1000 / η_PA + P_circuit          # η_PA ≈ 0.25
```
`P_flight` is the **rotary-wing model of Zeng, Xu & Zhang (2019)**, which is
U-shaped with a minimum near 10–15 m/s. The earlier `α‖v‖²` form claimed hovering
is cheapest, which is false for rotary-wing UAVs — and RQ1 is an energy claim, so
it cannot rest on a model that rewards hovering when reality does not. `κ‖a‖²` is
an explicit **control-effort heuristic**, not physics; present it as such.

**`Ptx` ceiling is set by the operating area, not by what makes the energy term
look good** — see "Operating area and Ptx ceiling" above. A 40 dBm ceiling was
briefly specified to enlarge the telecom energy share; it makes the mission
trivially satisfiable by a single drone at every simulable map size and must not
be reintroduced. Include `P_circuit` (always-on radio front end, ~2–5 W) so the
radio *subsystem* — not just the PA — is what appears in the energy accounting.

### Graph, reward, termination
- GNN edge weight (continuous, no hard cutoff — avoids gradient cliffs):
  `E_ij = sigmoid((C_ij − 5.0) · gamma)`
- Reward: tracking quality + capacity − energy − `λ·Var(B_1..B_N)`. The variance
  term forces role **rotation** rather than a fixed division of labour. Watch for
  its degenerate optimum (all drones hover ⇒ variance 0) — the tracking and
  capacity terms must dominate.
- Termination: `C_e2e ≥ 5.0 Mbps` false for >5 consecutive steps, **or** any drone
  hits `B_t == 0`.
- Jammer is mounted on the HVT and moves with it.
- HVT route: randomized valid path over the real OSM road graph, resampled per
  episode. Fixed routes only for early debugging.

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
- 13-dim per-node observation: local kinematics, battery, tracking metrics,
  ambient noise floor. Keep it **agent-local** — global state belongs to the
  critic, not the actor.

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
