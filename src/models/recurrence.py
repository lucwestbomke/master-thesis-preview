"""One GRU driver, shared by the recurrent actor and the recurrent critic.

The sequence handling is the part of the recurrent path that has already cost
this project a week (`docs/BLOCK_G.md`), so it exists **once** rather than being
written a second time in `critic.py`. Two modes, and skrl decides which by the
model's `nn.Module.training` flag -- it puts every model in eval mode for
collection (`ppo_rnn.py:170`) and in train mode for the update
(`ppo_rnn.py:396`), so `training` really does mean "this batch is sequences":

* **collection** (`training=False`): one transition per row, `(rows, 1, F)`.
* **update** (`training=True`): `(N_seq * L, F)` reshaped to `(N_seq, L, F)`,
  driven from each sequence's *first* stored hidden state, with the carried
  state zeroed wherever an episode ended inside the sequence.

⚠️ The `(N_seq, L)` reshape is only correct because skrl's
`Memory.all_sequence_indexes` orders rows **env-major, time-contiguous** --
`np.concatenate([arange(i, T*E + i, E) for i in range(E)])`, i.e. all of env 0's
timesteps, then all of env 1's. `test_recurrence.py` pins that layout rather
than trusting it, because a future skrl that reordered it would scramble time
against environment and produce exactly the silent collapse this path already
had once.
"""

from __future__ import annotations

import itertools

import torch
from torch import Tensor, nn


def run_gru(
    gru: nn.GRU,
    features: Tensor,
    hidden: Tensor,
    *,
    training: bool,
    sequence_length: int,
    num_layers: int,
    terminated: Tensor | None = None,
    truncated: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Drive `gru` over `features`, returning `(outputs, final_hidden)`.

    `features` is `(rows, F)` and the returned outputs are `(rows, H)` -- the
    same row order in, the same row order out, in both modes.
    """
    if not training:
        out, hidden = gru(features.view(-1, 1, features.shape[-1]), hidden)
        return out.flatten(start_dim=0, end_dim=1), hidden

    # (N_seq * L, F) -> (N_seq, L, F), and take each sequence's FIRST hidden
    # state rather than all L copies of it.
    seq = features.view(-1, sequence_length, features.shape[-1])
    hidden = hidden.view(num_layers, -1, sequence_length, hidden.shape[-1])[:, :, 0, :].contiguous()

    # Either may be absent -- skrl always passes both, but the model is also
    # driven directly by the probes and tests.
    done = (
        terminated
        if truncated is None
        else truncated
        if terminated is None
        else (terminated | truncated)
    )
    if done is None or not torch.any(done):
        out, hidden = gru(seq, hidden)
        return out.flatten(start_dim=0, end_dim=1), hidden

    # Split the sequence at every step where some episode ended and zero the
    # carried state there. Without this, memory leaks across an episode boundary
    # and the policy conditions on a previous episode's target.
    done = done.view(-1, sequence_length)
    cuts = (
        [0] + (done[:, :-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist() + [sequence_length]
    )
    chunks = []
    for lo, hi in itertools.pairwise(cuts):
        out, hidden = gru(seq[:, lo:hi, :], hidden)
        # ⚠️ Out-of-place. An in-place `hidden[:, mask] = 0` writes into the
        # tensor `gru` returned, which autograd saves for the backward pass.
        hidden = hidden.masked_fill(done[:, hi - 1].view(1, -1, 1), 0.0)
        chunks.append(out)
    return torch.cat(chunks, dim=1).flatten(start_dim=0, end_dim=1), hidden


def rnn_specification(
    sequence_length: int, num_layers: int, num_envs: int, hidden_size: int
) -> dict:
    """The `get_specification()` payload skrl reads to size its RNN memory.

    skrl takes `sequence_length` from the **policy** only, so a recurrent critic
    has to be given the same one or its stored states are reshaped against a
    length nothing produced them at.
    """
    return {
        "rnn": {"sequence_length": sequence_length, "sizes": [(num_layers, num_envs, hidden_size)]}
    }
