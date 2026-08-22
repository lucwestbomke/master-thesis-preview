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
| **B** | Frankfurt LoD2/OSM pipeline → buildings + road graph as tensors | ✅ **done**, `data/frankfurt_box.npz`, 27 tests |
| **C** | Occlusion: batched torch segment-vs-**oriented**-box (slab method) | ✅ **done**, 29 tests; `torch.compile` required |
| **D** | Batched env core + PettingZoo adapter + skrl wrapper | ✅ **built**, 46 tests; gate met with ~3170× margin. Awaiting the CUDA re-run of the *full env* (D3) — [`docs/BLOCK_D.md`](docs/BLOCK_D.md) |
| E | Renderer + B0 scripted heuristic baseline | ⬅️ **next** |
| F | Fidelity levels F0–F4 as config flags | not started |
| G | MAPPO integration + curriculum | not started |
| H | Sionna offline validation of the closed-form channel | not started |

Phase 0 (prep) runs to Feb 2027; the thesis window is Mar–Aug 2027. **Freeze the
environment end of March 2027** — results before are pilots, after are thesis
material. Full timeline in [`docs/THESIS_PLAN.md`](docs/THESIS_PLAN.md).

Block B is done; [`docs/BLOCK_B.md`](docs/BLOCK_B.md) records what was measured
and decided, and is the reference for the artefact's contents. Block C is
specified in [`docs/BLOCK_C.md`](docs/BLOCK_C.md), Block D in
[`docs/BLOCK_D.md`](docs/BLOCK_D.md). Why each block exists, what it gates and
which thesis chapter it feeds: [`docs/ROADMAP.md`](docs/ROADMAP.md).

✅ **The throughput gate is met.** RTX 5090, 2026-08-12: compiled occlusion runs
**3.17 M env-steps/s** at `num_envs = 1024` — ~3170× the gate — and occlusion is
~99 % of the step, so the env is not the bottleneck. `num_envs` is now chosen on
*learning* grounds, not throughput. Full table and provenance in
[`docs/BLOCK_C.md`](docs/BLOCK_C.md).

**Use `torch.compile`** — it is 110–150× on CUDA and free. But note the earlier
"required, not an optimisation" framing was an MPS artefact: on CUDA *eager also
clears the gate* (28.9 k env-steps/s, 29× over). And the 17.6 GB figure is memory
*traffic* per call, **not** live allocation — chunking keeps peak VRAM under 4 GB
in both paths. Do not reason about VRAM pressure from it.

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
| [`docs/BLOCK_B.md`](docs/BLOCK_B.md) | consuming `data/frankfurt_box.npz`, or touching geometry/routes |
| [`docs/BLOCK_C.md`](docs/BLOCK_C.md) | touching occlusion, or the geometry it consumes |
| [`docs/BLOCK_D.md`](docs/BLOCK_D.md) | building the env core, or touching altitude / cue / sensor / the throughput gate |

---

## Hard rules

**Device.** Training tensors live on `cuda:0`. **Never call `.cpu()`, `.numpy()`
or `.item()` inside env `step()` or the training hot loop** — `.item()` forces a
GPU sync and is the easy one to miss. Local dev is Apple Silicon (CPU/MPS), toy
configs only. Guard device selection; never silently degrade a real run to CPU.

**Throughput.** ≥1000 **env-steps/s**, where one env-step is *one environment
advancing one tick*, summed across the batch — **not** one batched `step()` call.
The two readings differ by 1000× and both appeared in this repo; the transition
reading is the one THESIS_PLAN §3's budget is written in (10 M steps ÷ 1000/s
≈ 2.8 h/run × 45 runs ≈ 120 GPU-h), and cost is what the gate protects. Settled
in [`docs/BLOCK_D.md`](docs/BLOCK_D.md) — do not re-open it.

Under that reading the 1000/s floor clears easily, so the number actually
reported is **wall-clock for a 10 M-step run, end-to-end including the learner,
target ≤3 h**. A **gate, not an aspiration** — measure it before building
anything on top of the env.

**Batched core, thin adapter.** The env core carries a leading `num_envs`
dimension. The PettingZoo adapter exists for API-compliance tests and visual
debugging only; training uses a custom skrl multi-agent wrapper. (skrl's
`PettingZooWrapper` round-trips through NumPy every step at `num_envs == 1`.)

**Geometry offline.** `osmnx`/`shapely` are CPU-and-NumPy-only — used **only** in
`scripts/prep_osm.py` to bake buildings into a tensor. Runtime occlusion is
vectorized segment-vs-**oriented**-box (slab method) in pure torch: rotate the
segment into each box's local frame, then run the standard branch-free slab
test. Axis-aligned boxes were measured and rejected — they fill 94 % of the
Frankfurt box ([`docs/DECISIONS.md`](docs/DECISIONS.md)). Buildings are 2.5D:
check the segment's altitude across the 2D intersection interval, not just a
planar crossing.

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
- ⛔ **Raise the altitude ceiling above 80 m.** It is not a comfort margin — it
  is where the scenario stops being a swarm problem. A best-placed *single* drone
  is mission-capable 3.3 % of the time at 80 m, 23 % at 100 m and 57 % at 120 m,
  so raising it falsifies W1 ("one drone cannot do this"). Raising it also *weakens*
  RQ1: A2A blockage falls 31 % → 25 % → 10 % at 80 / 120 / 180 m. Measured:
  [`scripts/measure_envelope.py`](scripts/measure_envelope.py).
- ⛔ **Move to mmWave.** It makes RQ1 trivial (mmWave is textbook
  blockage-limited, so "occlusion matters" stops being a finding), needs
  beamforming and beam-pointing modelling that couples to the motion policy, is
  the wrong band for the tactical MANET radios Ptx is derived from, and would
  invalidate Block A, PHYSICS.md and Chapter 3 before the freeze. It belongs in
  future work, where it strengthens the discussion for free.
- ⛔ **Add heavy dependencies** (sim engines, RL frameworks) without flagging.
- ⛔ **Cite constants an AI produced.** `TODO(verify)` markers in `channel.py`
  and `energy.py` mean exactly that — and now also the 120 m altitude ceiling.

---

## Settled parameters

| | Value | Basis |
|---|---|---|
| City / area | Frankfurt, **1500 m** box, centre **50.11200 N, 8.67040 E** (UTM 32N) | heterogeneous: low fabric gives a workable observation envelope, towers block A2A. Centre chosen by sweep — [`docs/BLOCK_B.md`](docs/BLOCK_B.md) |
| Buildings | Hessen **LoD2** via INSPIRE WFS, **oriented** boxes | 100 % height coverage; AABBs fill 94 % of the box and were rejected |
| HVT roads | all surface streets, speed capped **13.9 m/s** | the constraint is speed, not road class |
| Ptx | **30 dBm fixed** | UAV tactical MANET radios are 0.5–2 W |
| Jammer | 30 dBm in-band, rides the HVT | vehicle C-UAS barrage emitter |
| Carrier / bandwidth | 3.5 GHz / **10 MHz** | so the 5 Mbps target actually binds |
| Rate target | **5 Mbps** end-to-end | compressed HD EO/IR feed |
| Flight altitude | band **40–80 m**, ceiling = nominal | Both ends are *derived*, not chosen. **Floor**: model validity — below 40 m, 8–37 % of positions sit inside a building box (where occlusion's endpoint convention lets a drone see through its own building) and TR 36.777 stops at 22.5 m. **Ceiling**: scenario validity — above it a best-placed *single* drone can do the mission (3.3 % at 80 m vs 57.4 % at 120 m), which dissolves W1 and with it the reason for a swarm. [`docs/BLOCK_D.md`](docs/BLOCK_D.md) |
| Drone speed | 20 m/s cruise, 25 m/s dash | 1.4–1.8× margin over the HVT |
| HVT | 300–500 m from MCV, drives away | chain escalates 1 → 2 → 3 hops |
| Episode | **600 steps × 0.4 s** = 240 s | covers the escalation to 3 hops. **Do not shorten**: at 120 s the HVT reaches only ~1000 m and *no* route enters the 3-hop regime (0.0 % vs 36.8 %). A route step is a fixed *displacement*, so changing `dt` also changes HVT speed and needs a re-bake of the frozen artefact |
| Swarm | `N = 5` trained; 3/5/8 evaluated | |
| Discount | **γ ≈ 0.997–0.999** | default 0.99 is blind to the hard end of the episode |

Regenerate the sizing with [`scripts/scenario_design.py`](scripts/scenario_design.py)
and [`scripts/link_budget_check.py`](scripts/link_budget_check.py);
`tests/test_scenario_sizing.py` pins the trade-off table.

**Measured in Block B, replacing assumptions** ([`docs/BLOCK_B.md`](docs/BLOCK_B.md)):
street width median **21 m** (assumed 20) · canyon ratio `H_b/W` median **0.93**
(assumed 1.10) · across-street envelope at 80 m median **43 m**, p10–p90 24–88
(assumed a flat 36) · along-street sightline median **127 m**, p90 387.
The **830 m sensor range never binds** — 99.8 % of sightlines are shorter — so
occlusion is the constraint everywhere, which is what keeps RQ1 measuring channel
physics rather than sensor specification.

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
data/          baked artefacts — frankfurt_box.npz IS the frozen environment
tests/         cross-module only — unit tests are CO-LOCATED
docs/          reference, read on demand
```
`data/frankfurt_box.npz` is **committed on purpose**, not a build product: OSM and
the LoD2 service both change, so re-running `prep_osm.py` in 2027 would silently
produce a different map. The file in git is the environment; the script only
documents how it was made.
- Unit tests sit next to their module: `src/env/test_channel.py`.
- Naming: `hvt`, `mcv_base`, `sinr_db`, `capacity_mbps`, `edge_weight`,
  `ptx_dbm`. Keep tactical/telecom terms consistent.
- Observations: **24**-dim ego (21 + a persistent 3-dim vector to the cue), 9-dim
  per neighbour, 2-dim per edge; packed flat at `N_max = 8` to **108** dims for
  skrl's rollout storage. Actor stays **agent-local**; global state belongs to
  the critic. **No time feature** — truncation is handled by bootstrapping, not
  by observing the clock ([`docs/BLOCK_D.md`](docs/BLOCK_D.md)).

## Build / test
```bash
uv sync --extra dev                              # `dev` is an EXTRA -- plain
                                                 # `uv sync` gives you neither
                                                 # pytest nor ruff
uv run pytest                                    # 207 tests (+4 CUDA-only skips)
uv run ruff check . && uv run ruff format .
```
Offline data prep (needs network; the artefact is committed, so this is only for
regenerating it deliberately):
```bash
uv run python scripts/prep_osm.py --plot         # bake data/frankfurt_box.npz
uv run python scripts/measure_sightlines.py --plot
```
Offline, no network — regenerates every number Block D's design rests on
(altitude band, sensor envelope, cue staleness, uncued search, link budget):
```bash
uv run python scripts/measure_envelope.py
uv run python scripts/measure_envelope.py --only a2a inside
```
Look at the map before trusting it — every geometry bug so far was found by
happening to compute the right statistic, and the viewer finds the next one in
seconds:
```bash
uv run python scripts/view_episode.py --worst --polygons   # boxes vs source footprints
uv run python scripts/view_episode.py --route 0 --zoom --save ep.mp4
uv run python scripts/bench_occlusion.py                   # re-run this on CUDA
```
