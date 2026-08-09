# UAV Swarm MARL Thesis

MARL (MAPPO + GNN, CTDE) for a UAV swarm that tracks a moving ground target
through a real city while relaying a connection back to a fixed base station,
under jamming co-located with the target.

- **Research design** — questions, hypotheses, baselines, metrics, timeline:
  [`docs/THESIS_PLAN.md`](./docs/THESIS_PLAN.md)
- **Negative results** — why transmit-power control is *not* the research
  question, with the numbers: [`docs/NEGATIVE_RESULTS.md`](./docs/NEGATIVE_RESULTS.md)
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
Pre-implementation. Channel model, relay routing and the scenario-sizing tools
are built and unit-tested (55 tests). The batched environment core, occlusion
geometry, OSM/LoD2 pipeline and training entrypoints are not. See the Phase 0
table in the thesis plan for build order.

Scenario is settled and derived rather than guessed: **Frankfurt, 1500 m
operating area, 30 dBm fixed transmit power, 10 MHz at 3.5 GHz, 5 Mbps
end-to-end target, jammer riding the HVT.**
