"""Cross-device tests: does the GPU compute what the CPU computes?

Every other test file in this repo runs on the default device, so invoking
`pytest` on a GPU box exercises the CPU and proves nothing about CUDA. This file
is the exception: it parameterises over every backend actually available and
compares against CPU, which is the property at risk -- a device-specific
numerical divergence, or a kernel that silently produces different results.

Two reasons it is a separate file rather than a fixture threaded through the
whole suite: the numerical paths are what matter (not the bookkeeping), and
parameterising 200 tests over three backends would triple a suite that already
takes half a minute.

`test_step_never_syncs_to_the_host` is CUDA-only and is the enforcement arm of
AGENTS.md's device rule -- `.item()` in the hot loop is the easiest way to
destroy throughput and the hardest to spot in review.
"""

from __future__ import annotations

import pytest
import torch

from .core import BatchedSwarmEnv, EnvConfig
from .occlusion import pairwise_clearance
from .routing import best_relay_path

STATE_TENSORS = (
    "drone_pos",
    "drone_vel",
    "last_accel",
    "battery",
    "mcv_pos",
    "hvt_pos",
    "hvt_vel",
    "cue",
    "route_id",
    "t",
    "steps_since_link",
    "episode_len",
    "speed_scale",
    "jammer_on",
    "battery_scale",
)


def available_devices() -> list[str]:
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.append("cuda")
    if torch.backends.mps.is_available():
        devs.append("mps")
    return devs


@pytest.fixture(params=available_devices())
def device(request) -> str:
    return request.param


def _mirror(src: BatchedSwarmEnv, dst: BatchedSwarmEnv) -> None:
    """Copy `src`'s episode state onto `dst`, which may be on another device.

    Needed because the per-device RNG streams differ, so two envs with the same
    seed do not draw the same episodes. Mirroring isolates the arithmetic, which
    is what this file is about.
    """
    for name in STATE_TENSORS:
        setattr(dst, name, getattr(src, name).to(dst.device).clone())
    dst.snap, _ = dst._evaluate()


def test_occlusion_matches_cpu(device):
    """The kernel that is 99.7 % of the step, and the one most likely to differ:
    it leans on +/-inf sentinels and a branch-free slab test."""
    if device == "cpu":
        pytest.skip("cpu is the reference")
    art = BatchedSwarmEnv(EnvConfig(num_envs=1, compile_occlusion=False))
    torch.manual_seed(0)
    pos = torch.empty(32, 7, 3)
    pos[..., 0].uniform_(-700, 700)
    pos[..., 1].uniform_(-700, 700)
    pos[..., 2].uniform_(1.5, 120.0)

    ref = pairwise_clearance(pos, art.boxes, art.heights)
    got = pairwise_clearance(pos.to(device), art.boxes.to(device), art.heights.to(device))
    assert torch.allclose(ref, got.cpu(), atol=1e-2), (ref - got.cpu()).abs().max()


def test_routing_matches_cpu(device):
    if device == "cpu":
        pytest.skip("cpu is the reference")
    torch.manual_seed(1)
    cap = torch.rand(64, 6, 6) * 40.0
    src = torch.rand(64, 6) < 0.4

    e_ref, path_ref, edge_ref, hop_ref = best_relay_path(cap, src, 5, 5)
    e_got, path_got, edge_got, hop_got = best_relay_path(cap.to(device), src.to(device), 5, 5)
    assert torch.allclose(e_ref, e_got.cpu(), atol=1e-4)
    assert torch.equal(path_ref, path_got.cpu())
    assert torch.equal(edge_ref, edge_got.cpu())
    assert torch.equal(hop_ref, hop_got.cpu())


def test_full_step_matches_cpu(device):
    """End to end: same state, same actions, same observations and rewards."""
    if device == "cpu":
        pytest.skip("cpu is the reference")
    kw = {"num_envs": 8, "num_drones": 5, "seed": 5, "compile_occlusion": False}
    ref = BatchedSwarmEnv(EnvConfig(device="cpu", **kw))
    got = BatchedSwarmEnv(EnvConfig(device=device, **kw))
    ref.reset()
    got.reset()
    _mirror(ref, got)

    torch.manual_seed(2)
    for step in range(20):
        act = torch.empty(8, 5, 3).uniform_(-1, 1)
        o_ref, r_ref, t_ref, u_ref, x_ref = ref.step(act)
        o_got, r_got, t_got, u_got, x_got = got.step(act.to(device))

        assert torch.allclose(r_ref, r_got.cpu(), atol=1e-3), f"reward, step {step}"
        assert torch.allclose(o_ref["flat"], o_got["flat"].cpu(), atol=1e-3), f"obs, step {step}"
        assert torch.allclose(o_ref["state"], o_got["state"].cpu(), atol=1e-3)
        assert torch.equal(t_ref, t_got.cpu()) and torch.equal(u_ref, u_got.cpu())
        assert torch.equal(x_ref["sees_any"], x_got["sees_any"].cpu())
        assert torch.equal(x_ref["hop_count"], x_got["hop_count"].cpu())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_step_never_syncs_to_the_host():
    """Turns AGENTS.md's device rule into something enforced rather than hoped.

    `set_sync_debug_mode("error")` raises on any operation that forces a
    device-to-host synchronisation -- a stray `.item()`, an `if` on a tensor, a
    `.cpu()`. Each costs a full pipeline stall per step, and none of them looks
    wrong in a diff.
    """
    env = BatchedSwarmEnv(EnvConfig(num_envs=64, device="cuda", compile_occlusion=False))
    env.reset()
    act = torch.zeros(64, 5, 3, device="cuda")
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for _ in range(5):
            env.step(act)
    finally:
        torch.cuda.set_sync_debug_mode("default")
