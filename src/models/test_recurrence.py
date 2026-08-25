"""The GRU driver, and the skrl memory layout it silently depends on.

`docs/BLOCK_G.md` records that the recurrent path collapsed for a week while
every component tested clean, so the invariants here are deliberately about the
*seams* rather than the parts: sequence replay must reproduce step-mode
collection, and skrl's row ordering must be the one `view(-1, L)` assumes.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .recurrence import run_gru

F, H, LAYERS = 7, 5, 1


def _gru() -> nn.GRU:
    torch.manual_seed(0)
    return nn.GRU(input_size=F, hidden_size=H, num_layers=LAYERS, batch_first=True)


def _step_mode(gru: nn.GRU, features: torch.Tensor, n_seq: int, length: int):
    """Collection, hand-rolled: one transition at a time, hidden carried.

    Returns the per-row outputs and the per-row *input* hidden states, laid out
    sequence-major exactly as skrl's memory stores them.
    """
    seq = features.view(n_seq, length, F)
    h = torch.zeros(LAYERS, n_seq, H)
    outs, states = [], []
    for t in range(length):
        states.append(h.clone())
        out, h = gru(seq[:, t : t + 1, :], h)
        outs.append(out[:, 0, :])
    # (L, n_seq, ...) -> (n_seq, L, ...) -> rows
    stacked_out = torch.stack(outs, dim=1).reshape(n_seq * length, H)
    stacked_h = torch.stack(states, dim=2).reshape(LAYERS, n_seq * length, H)
    return stacked_out, stacked_h


def test_collection_mode_is_one_step_per_row():
    gru = _gru()
    features = torch.randn(12, F)
    hidden = torch.randn(LAYERS, 12, H)
    out, new_hidden = run_gru(
        gru, features, hidden, training=False, sequence_length=16, num_layers=LAYERS
    )
    expected_out, expected_h = gru(features.unsqueeze(1), hidden)
    assert torch.allclose(out, expected_out[:, 0, :], atol=1e-6)
    assert torch.allclose(new_hidden, expected_h, atol=1e-6)


def test_sequence_replay_reproduces_step_mode_collection():
    """The invariant the whole recurrent path rests on.

    If replaying `L`-step sequences from each sequence's stored first hidden
    state does not reproduce what collection computed one step at a time, every
    PPO importance ratio is taken against a policy that was never run.
    """
    gru, n_seq, length = _gru(), 4, 8
    features = torch.randn(n_seq * length, F)
    expected, stored_hidden = _step_mode(gru, features, n_seq, length)

    replayed, _ = run_gru(
        gru,
        features,
        stored_hidden,
        training=True,
        sequence_length=length,
        num_layers=LAYERS,
    )
    assert torch.allclose(replayed, expected, atol=1e-6), (
        f"max |delta| = {float((replayed - expected).abs().max()):.3e}"
    )


def test_memory_is_zeroed_at_an_episode_boundary_inside_a_sequence():
    """A sequence that spans a reset must not carry belief across it."""
    gru, n_seq, length = _gru(), 3, 6
    features = torch.randn(n_seq * length, F)
    hidden = torch.zeros(LAYERS, n_seq * length, H)

    clean, _ = run_gru(
        gru, features, hidden, training=True, sequence_length=length, num_layers=LAYERS
    )
    # Sequence 1 ends its episode at t = 2; nothing else does.
    done = torch.zeros(n_seq, length, dtype=torch.bool)
    done[1, 2] = True
    cut, _ = run_gru(
        gru,
        features,
        hidden,
        training=True,
        sequence_length=length,
        num_layers=LAYERS,
        terminated=done.reshape(-1, 1),
    )
    clean, cut = clean.view(n_seq, length, H), cut.view(n_seq, length, H)

    assert torch.allclose(clean[0], cut[0], atol=1e-6), "an unrelated sequence was disturbed"
    assert torch.allclose(clean[1, :3], cut[1, :3], atol=1e-6), "the pre-boundary part moved"
    assert not torch.allclose(clean[1, 3:], cut[1, 3:], atol=1e-5), (
        "memory leaked across the episode boundary"
    )


def test_the_gradient_survives_the_boundary_split():
    """The zeroing must not be an in-place write into a tensor autograd saved."""
    gru, n_seq, length = _gru(), 2, 4
    features = torch.randn(n_seq * length, F, requires_grad=True)
    done = torch.zeros(n_seq, length, dtype=torch.bool)
    done[0, 1] = True
    out, _ = run_gru(
        gru,
        features,
        torch.zeros(LAYERS, n_seq * length, H),
        training=True,
        sequence_length=length,
        num_layers=LAYERS,
        truncated=done.reshape(-1, 1),
    )
    out.sum().backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()


def test_skrl_orders_sequence_rows_env_major_and_time_contiguous():
    """The layout `view(-1, L)` assumes, pinned against the installed skrl.

    ⚠️ Not a test of our code. `Memory.all_sequence_indexes` is what makes a
    reshape into `(N_seq, L)` mean "L consecutive timesteps of one environment";
    a skrl that reordered it would scramble time against environment and produce
    a silent collapse rather than an error.
    """
    from skrl.memories.torch import RandomMemory

    steps, envs = 6, 3
    memory = RandomMemory(memory_size=steps, num_envs=envs, device="cpu")
    memory.create_tensor(name="probe", size=2, dtype=torch.float32)
    for t in range(steps):
        # probe[:, 0] = timestep, probe[:, 1] = environment
        rows = torch.stack([torch.full((envs,), float(t)), torch.arange(envs, dtype=torch.float32)])
        memory.add_samples(probe=rows.T)

    sampled = memory.sample_all(names=["probe"], mini_batches=1, sequence_length=2)[0][0]
    ordered = sampled.view(envs, steps, 2)
    assert torch.equal(
        ordered[:, :, 1], torch.arange(envs, dtype=torch.float32).view(envs, 1).expand(envs, steps)
    ), "rows are not env-major"
    assert torch.equal(
        ordered[:, :, 0],
        torch.arange(steps, dtype=torch.float32).view(1, steps).expand(envs, steps),
    ), "timesteps within an environment are not contiguous or not in order"
    assert np.array_equal(
        memory.all_sequence_indexes,
        np.concatenate([np.arange(i, steps * envs + i, envs) for i in range(envs)]),
    )
