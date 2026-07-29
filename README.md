# UAV Swarm MARL Thesis

MARL (MAPPO + GNN, CTDE) for a UAV swarm that tracks a moving ground target
through a real city while relaying a connection back to a fixed base station,
under jamming co-located with the target.

- **Research design** — questions, hypotheses, baselines, metrics, timeline:
  [`docs/THESIS_PLAN.md`](./docs/THESIS_PLAN.md)
- **Technical context** — stack decisions, physics, conventions, for any AI
  coding agent or human working in this repo: [`AGENTS.md`](./AGENTS.md)

## Setup
```bash
uv sync                                          # uv.lock is authoritative
uv run pytest                                    # tests
uv run ruff check . && uv run ruff format .      # lint / format
```

Local dev is Mac (CPU/MPS), toy configs only. Real training runs on a rented
CUDA GPU (RunPod) — see AGENTS.md for the device-split rationale.

## Status
Pre-implementation. Channel model and relay routing are implemented and
unit-tested; the batched environment core, occlusion geometry, OSM pipeline, and
training entrypoints are not yet built. See the Phase 0 table in the thesis plan
for the build order.
