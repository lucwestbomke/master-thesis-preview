#!/usr/bin/env bash
# Block G, Phase 1 -- the CUDA session, in order of what it settles.
#
# Everything here is FIXED work: no decisions, no tuning. Run it, keep
# cuda_session.log, and bring the log home. Each step prints what it settles.
#
#   bash scripts/cuda_session.sh 2>&1 | tee cuda_session.log
#
# ⚠️ Every number this produces is a CUDA number. `torch.Generator` streams
# differ per device, so NONE of it is comparable with the MPS numbers in
# docs/BLOCK_G.md -- that is why step 5 re-measures the baseline here rather
# than reusing the laptop's 37.6 %.
set -u  # not -e: a failing step should not hide the steps after it

say() { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }

say "0. Machine"
nvidia-smi || echo "!! no nvidia-smi -- is this actually a GPU box?"
uv run python - <<'PY'
import torch
print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"device: {p.name}  {p.total_memory / 1e9:.0f} GB  sm_{p.major}{p.minor}")
else:
    raise SystemExit("CUDA not available -- stop here and fix the install")
PY

say "1. Test suite -- the 4 CUDA-only tests have NEVER run"
# src/env/test_device_parity.py has a CUDA-gated block: it checks the occlusion
# kernel and capacity agree across devices. It has been skipped on every run of
# this project so far. If it fails, that is a real finding about the physics.
uv run pytest -q

say "2. G1a -- env-only throughput (Block D's pending item)"
# Gate: >=1000 env-steps/s (transitions, not batched calls). BLOCK_C measured
# 3.17M env-steps/s for the occlusion kernel alone on a 5090.
uv run python scripts/bench_env.py --device cuda --envs 256 1024 4096 --breakdown

say "3. Throughput is rung-independent (Block F's result, at CUDA scale)"
# BLOCK_F measured a 1.06x spread across F0-F4 on MPS. If a rung is cheaper on
# CUDA it gets more samples per GPU-hour and RQ1 is confounded.
uv run python scripts/eval_fidelity.py --only throughput --device cuda

say "4. G1b -- THE number: 10M steps end-to-end, learner attached"
# This has never been measured on any device. THESIS_PLAN's 120 GPU-hour budget
# rests on it. Target <=3 h. The run prints env-steps/s and a 10M projection.
uv run python -m src.training.train --fidelity F4 --arch gnn --seed 0 \
    --num-envs 1024 --env-steps 10000000 --device cuda \
    --boundaries 0.10 0.20 0.35 --min-std 0.2 --log-every 500 \
    --checkpoint-every 2000 --name cuda-g1b

say "5. CUDA baseline -- 3 seeds of the best known config"
# The laptop's 37.6 % is an MPS number and cannot be carried over.
for s in 0 1 2; do
  uv run python -m src.training.train --fidelity F4 --arch gnn --seed "$s" \
      --num-envs 1024 --env-steps 12000000 --device cuda \
      --boundaries 0.10 0.20 0.35 --min-std 0.2 --log-every 2000 \
      --name cuda-base-s"$s"
done

say "6. Score them against B0 on the SAME device"
uv run python scripts/eval_policy.py runs/cuda-base-s*/checkpoint.pt \
    --group "GNN (CUDA, 3 seeds)" --policy b0 random \
    --stage 4 --seeds 5 --num-envs 256 --device cuda --train-routes

say "DONE -- keep cuda_session.log and the runs/ directory"
