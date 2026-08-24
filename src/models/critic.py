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

from skrl.models.torch import DeterministicMixin, Model
from torch import Tensor

from .actor import _mlp

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
