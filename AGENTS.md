# AGENTS.md — UAV Swarm MARL Thesis

Entry point for any agent or human working in this repo. Kept deliberately short;
detail lives in `docs/` and is read **on demand**.

## Mission
A swarm of `N` UAVs must simultaneously **observe** a moving ground High-Value
Target (HVT) in Frankfurt, **relay** the sensor feed to a Mobile Command Vehicle
(MCV) over a multi-hop chain at ≥5 Mbps end-to-end, and **survive** on finite
batteries while a jammer riding the HVT degrades links near it. Buildings block
line of sight, so the relay chain is geometrically necessary.

- **RQ1 (primary):** which physical effects must a channel model include for
  learned policies to transfer? Train one policy per fidelity rung F0–F4,
  evaluate all under F4. Hypothesis: **occlusion** dominates.
- **RQ2:** MLP → DeepSets → GNN; zero-shot transfer across `N ∈ {3,5,8}` and
  across city morphology.
- **RQ3:** does the observer role **hand off** as sightlines change, and is the
  handoff coordinated and anticipatory?

---

## Where the project is

| Block | What | State |
|---|---|---|
| **A** | Channel, routing, energy, reward — all pure, batched, tested | ✅ **done**, 103 tests |
| **B** | Frankfurt OSM/LoD2 pipeline → buildings + road graph as tensors | ⬅️ **next** |
| C | Occlusion: batched torch segment-vs-box (slab method) | not started |
| D | Batched env core + PettingZoo adapter; **≥1000 steps/s gate** | not started |
| E | Renderer + B0 scripted heuristic baseline | not started |
| F | Fidelity levels F0–F4 as config flags | not started |
| G | MAPPO integration + curriculum | not started |
| H | Sionna offline validation of the closed-form channel | not started |

Phase 0 (prep) runs to Feb 2027; the thesis window is Mar–Aug 2027. **Freeze the
environment end of March 2027** — results before are pilots, after are thesis
material. Full timeline in [`docs/THESIS_PLAN.md`](docs/THESIS_PLAN.md).

Block B is specified in [`docs/BLOCK_B.md`](docs/BLOCK_B.md). Why each block
exists, what it gates and which thesis chapter it feeds:
[`docs/ROADMAP.md`](docs/ROADMAP.md).

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
| [`docs/BLOCK_B.md`](docs/BLOCK_B.md) | the current task |

---

## Hard rules

**Device.** Training tensors live on `cuda:0`. **Never call `.cpu()`, `.numpy()`
or `.item()` inside env `step()` or the training hot loop** — `.item()` forces a
GPU sync and is the easy one to miss. Local dev is Apple Silicon (CPU/MPS), toy
configs only. Guard device selection; never silently degrade a real run to CPU.

**Throughput.** ≥1000 env-steps/s batched on GPU. A **gate, not an aspiration** —
measure it before building anything on top of the env.

**Batched core, thin adapter.** The env core carries a leading `num_envs`
dimension. The PettingZoo adapter exists for API-compliance tests and visual
debugging only; training uses a custom skrl multi-agent wrapper. (skrl's
`PettingZooWrapper` round-trips through NumPy every step at `num_envs == 1`.)

**Geometry offline.** `osmnx`/`shapely` are CPU-and-NumPy-only — used **only** in
`scripts/prep_osm.py` to bake buildings into a tensor. Runtime occlusion is
vectorized segment-vs-box (slab method) in pure torch. Buildings are 2.5D: check
the segment's altitude across the 2D intersection interval, not just a planar
crossing.

**Formulas are traceable.** Do not change path-loss / SINR / capacity / energy
formulas without updating the hand-computed tests and checking the cited
standard. They appear in the methodology chapter.

**Multi-seed.** ≥5 seeds for anything reported as a finding. Median + IQR, never
mean ± std — RL returns are not normally distributed. Never report single runs.

---

## Never do these

- ⛔ **Reintroduce transmit power as an action.** Three framings, three nulls —
  [`docs/NEGATIVE_RESULTS.md`](docs/NEGATIVE_RESULTS.md). Action space is motion
  only (3-dim); Ptx is fixed at 30 dBm. E4 reproduces the null deliberately.
- ⛔ **Raise the Ptx ceiling.** At 40 dBm a *blocked* A2A link carries 15 Mbps
  over 2.8 km — one drone spans the map and the relay chain becomes pointless.
- ⛔ **Use channel fidelity as a curriculum axis.** It is RQ1's independent
  variable. Same reasoning forbids ramping building density.
- ⛔ **Use `SAGEConv`** for the GNN rung. It cannot take edge features, so it
  silently collapses the GNN into DeepSets and RQ2 measures nothing.
- ⛔ **Terminate the episode on mission failure.** The policy learns never to
  acquire, and a random initial policy never reaches the tracking phase.
- ⛔ **Sweep more than `λ`.** Other weights are pinned by behavioural orderings
  in [`docs/REWARD.md`](docs/REWARD.md).
- ⛔ **Add heavy dependencies** (sim engines, RL frameworks) without flagging.
- ⛔ **Cite constants an AI produced.** `TODO(verify)` markers in `channel.py`
  and `energy.py` mean exactly that.

---

## Settled parameters

| | Value | Basis |
|---|---|---|
| City / area | Frankfurt, **1500 m** box | heterogeneous: low fabric gives a workable observation envelope, towers block A2A |
| Ptx | **30 dBm fixed** | UAV tactical MANET radios are 0.5–2 W |
| Jammer | 30 dBm in-band, rides the HVT | vehicle C-UAS barrage emitter |
| Carrier / bandwidth | 3.5 GHz / **10 MHz** | so the 5 Mbps target actually binds |
| Rate target | **5 Mbps** end-to-end | compressed HD EO/IR feed |
| Flight altitude | 80 m nominal | above fabric, below towers; inside TR 36.777's 22.5–300 m band |
| Drone speed | 20 m/s cruise, 25 m/s dash | 1.4–1.8× margin over the HVT |
| HVT | 300–500 m from MCV, drives away | chain escalates 1 → 2 → 3 hops |
| Episode | **600 steps × 0.4 s** = 240 s | covers the escalation to 3 hops |
| Swarm | `N = 5` trained; 3/5/8 evaluated | |
| Discount | **γ ≈ 0.997–0.999** | default 0.99 is blind to the hard end of the episode |

Regenerate the sizing with [`scripts/scenario_design.py`](scripts/scenario_design.py)
and [`scripts/link_budget_check.py`](scripts/link_budget_check.py);
`tests/test_scenario_sizing.py` pins the trade-off table.

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
src/env/       channel, routing, energy, reward  (built)
               occlusion, batched core           (to build)
src/models/    GNN / DeepSets / MLP actor-critic
src/training/  skrl wrappers, entrypoints
scripts/       offline data prep + scenario tooling
configs/       YAML per experiment condition
tests/         cross-module only — unit tests are CO-LOCATED
docs/          reference, read on demand
```
- Unit tests sit next to their module: `src/env/test_channel.py`.
- Naming: `hvt`, `mcv_base`, `sinr_db`, `capacity_mbps`, `edge_weight`,
  `ptx_dbm`. Keep tactical/telecom terms consistent.
- Observations: 21-dim ego, 9-dim per neighbour, 2-dim per edge. Actor stays
  **agent-local**; global state belongs to the critic.

## Build / test
```bash
uv sync
uv run pytest                                    # 103 tests
uv run ruff check . && uv run ruff format .
```
