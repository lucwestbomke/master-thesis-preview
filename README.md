# UAV Swarm MARL Thesis

MARL (MAPPO + GNN, CTDE) for a UAV swarm that tracks a moving ground target
through Frankfurt while relaying a ≥5 Mbps feed back to a command vehicle, under
jamming co-located with the target.

**Research question:** most MARL work on swarm communication abstracts the channel
to a connectivity radius. Does that abstraction produce policies that fail under
realistic physics, and *which* physics is responsible?

## Where to start

| | |
|---|---|
| [`AGENTS.md`](AGENTS.md) | entry point — current state, hard rules, settled parameters |
| [`docs/THESIS_PLAN.md`](docs/THESIS_PLAN.md) | research design: questions, conditions, metrics, timeline |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | what was tried and rejected, with evidence |
| [`docs/BLOCK_B.md`](docs/BLOCK_B.md) | the current task |

Reference docs read on demand: [`PHYSICS`](docs/PHYSICS.md) ·
[`REWARD`](docs/REWARD.md) · [`ENVIRONMENT`](docs/ENVIRONMENT.md) ·
[`MODELS`](docs/MODELS.md) · [`NEGATIVE_RESULTS`](docs/NEGATIVE_RESULTS.md)

## Setup
```bash
uv sync
uv run pytest                                    # 103 tests
uv run ruff check . && uv run ruff format .
```

Local dev is Mac (CPU/MPS), toy configs only. Real training runs on a rented CUDA
GPU — see AGENTS.md for the device-split rationale.

## Status

**Block A complete.** Channel model, relay routing, rotary-wing energy and the
reward function are implemented as pure batched torch, with hand-computed test
assertions. Not yet built: OSM pipeline, occlusion, the batched env core,
models, training.

Scenario is derived rather than guessed: Frankfurt, 1500 m operating area,
30 dBm fixed transmit power, 10 MHz at 3.5 GHz, 5 Mbps end-to-end target,
600-step (240 s) episodes.
