"""The centralized critic -- one design, identical in all three RQ2 conditions.

`docs/MODELS.md`: "Keep the critic identical across all three architecture
conditions. If only the actor varies, RQ2 isolates the actor. If both vary, it
is confounded." So this file has no `architecture` argument, deliberately.

It need not be size-agnostic either: zero-shot transfer to `N in {3, 8}` runs
the actor alone and the critic is discarded at evaluation, so a plain MLP over
the concatenated global state is the right shape. The state is already unit-
scaled by `core._critic_state` (positions MCV-relative and divided by the box
half-width, velocities by the dash speed, capacity in threshold units), which is
why the training config sets a `value_preprocessor` but no `state_preprocessor`.
"""

from __future__ import annotations

from typing import Any

import torch
from skrl.models.torch import DeterministicMixin, Model
from torch import Tensor, nn

from .actor import _mlp
from .recurrence import rnn_specification, run_gru

DEFAULT_CRITIC_HIDDEN = 256


class SwarmCritic(DeterministicMixin, Model):
    """Value of the global state. Training only."""

    def __init__(
        self,
        state_space: Any,
        action_space: Any,
        device: Any,
        hidden: int = DEFAULT_CRITIC_HIDDEN,
    ):
        Model.__init__(
            self, observation_space=state_space, action_space=action_space, device=device
        )
        DeterministicMixin.__init__(self, clip_actions=False)
        self.net = _mlp([state_space.shape[0], hidden, hidden, 1])

    def compute(self, inputs: dict[str, Tensor], role: str = ""):
        return self.net(inputs["states"]), {}


class SwarmCriticRNN(SwarmCritic):
    """The centralized critic, with the same memory the actor has.

    ## Why, and it is a correctness argument rather than a tuning one

    A recurrent policy's behaviour depends on a hidden state `h`. If the value
    function sees only `s`, it necessarily fits `E_h[V(s, h)]` -- the average
    over whatever hidden states the policy happened to occupy at `s` -- so every
    advantage carries a `V(s, h) - E_h[V(s, h)]` error term that is *exactly*
    the history-dependent part the GRU exists to produce. The advantage is then
    biased hardest for the behaviour the memory was added to learn, and PPO
    optimises the bias. Yu et al. (2022) make both networks recurrent for this
    reason; skrl's `PPO_RNN` already stores `rnn_value_*` tensors for it, so the
    machinery existed and only the model did not.

    ## What it does not change

    ⛔ `docs/MODELS.md`: "the critic is identical across all three architecture
    conditions", so RQ2 isolates the actor. This class has no `architecture`
    argument for the same reason `SwarmCritic` has none -- a recurrent critic for
    the GNN rung alone would confound actor with critic and delete RQ2. It reads
    `states` (the global state) and never `observations`, so CTDE is unaffected.

    ⚠️ `sequence_length` must match the actor's. skrl reads the sequence length
    from the **policy's** specification only (`ppo_rnn.py:202`) and applies it to
    every RNN tensor in memory, so a critic built with a different one would have
    its stored states reshaped against a length nothing produced them at.
    """

    def __init__(
        self,
        state_space: Any,
        action_space: Any,
        device: Any,
        hidden: int = DEFAULT_CRITIC_HIDDEN,
        num_envs: int = 1,
        rnn_hidden: int = 128,
        rnn_layers: int = 1,
        sequence_length: int = 16,
    ):
        super().__init__(state_space, action_space, device, hidden=hidden)
        self.num_envs = num_envs
        self.rnn_hidden = rnn_hidden
        self.rnn_layers = rnn_layers
        self.sequence_length = sequence_length
        # The feedforward critic is state -> hidden -> hidden -> 1. Here the
        # last layer becomes the GRU's input, so the encoder keeps both hidden
        # layers and the value head reads the recurrent state instead.
        self.net = _mlp([state_space.shape[0], hidden, hidden])
        self.gru = nn.GRU(
            input_size=hidden,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
        )
        self.head = nn.Linear(rnn_hidden, 1)

    def get_specification(self) -> dict[str, Any]:
        return rnn_specification(
            self.sequence_length, self.rnn_layers, self.num_envs, self.rnn_hidden
        )

    def compute(self, inputs: dict[str, Tensor], role: str = ""):
        rnn_out, hidden = run_gru(
            self.gru,
            torch.tanh(self.net(inputs["states"])),
            inputs["rnn"][0],
            training=self.training,
            sequence_length=self.sequence_length,
            num_layers=self.rnn_layers,
            terminated=inputs.get("terminated"),
            truncated=inputs.get("truncated"),
        )
        return self.head(rnn_out), {"rnn": [hidden]}
