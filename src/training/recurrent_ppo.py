"""skrl's `PPO_RNN`, with the hidden state it stores actually being the one used.

## The bug, in skrl 2.1.0

`PPO_RNN.record_transition` ends with

    self._rnn_initial_states = self._rnn_final_states

which binds the two names to the **same dict object**. From the next step
onwards, `act()` does

    self._rnn_final_states["policy"] = outputs["rnn"]     # the POST-step state

and because the dicts are the same object that also overwrites
`_rnn_initial_states["policy"]`. `record_transition` then packages
`_rnn_initial_states` into memory as the state the action was taken from -- but
it now holds the state the action *produced*.

**Every transition after the first stores a hidden state one step ahead of the
one that generated its action.** Measured directly (`test_recurrent.py`):
`stored[t] == h_in[t+1]` for every `t >= 1`, and exact only at `t = 0`, which is
the one step where the aliasing has not yet happened.

## Why it presents the way it does

Every PPO importance ratio in a recurrent update is then computed against the
wrong state, silently:

* at `--seq-len 1` each row is individually off by one step. A GRU state moves
  slowly, so the policy still limps upward -- this project measured 39.6 % at
  stage 1 where the feedforward actor reaches 75-79 %.
* at `--seq-len 16` the wrong `h0` is fed into a 16-step replay and compounds
  down the sequence, which is why the error profile peaks at sequence position 0
  and decays, and why training collapses outright.

## The fix

Snapshot the states in `act()`, before skrl can overwrite them, and hand that
snapshot to `record_transition`. Nothing else changes: the forward update of the
recurrent state, the episode-boundary zeroing and the memory layout are all
skrl's, untouched.

⚠️ Remove this class only after checking the installed skrl. The invariant it
restores is asserted by `test_recurrent.py`, which will fail loudly if a future
version fixes the aliasing and this workaround starts double-correcting.
"""

from __future__ import annotations

from typing import Any

import torch
from skrl.agents.torch.ppo import PPO_RNN


def _deep_copy_states(states: dict[str, list[torch.Tensor]]) -> dict[str, list[torch.Tensor]]:
    return {key: [s.clone() for s in value] for key, value in states.items()}


class PPO_RNN_Aligned(PPO_RNN):
    """`PPO_RNN` that records the hidden state the action was actually taken from."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._pre_step_rnn: dict[str, list[torch.Tensor]] | None = None

    def act(self, observations, states, *, timestep: int, timesteps: int):
        # Taken BEFORE super().act(), which overwrites `_rnn_final_states` -- and
        # after the first step that is the same object as `_rnn_initial_states`.
        if self._rnn:
            self._pre_step_rnn = _deep_copy_states(self._rnn_initial_states)
        return super().act(observations, states, timestep=timestep, timesteps=timesteps)

    def record_transition(self, **kwargs: Any) -> None:
        if not (self._rnn and self._pre_step_rnn is not None):
            super().record_transition(**kwargs)
            return
        # Swap the corrupted "initial" states for the snapshot, let skrl store
        # them, and let its own trailing `_rnn_initial_states = _rnn_final_states`
        # advance the state for the next step exactly as it does today.
        corrupted = self._rnn_initial_states
        self._rnn_initial_states = self._pre_step_rnn
        try:
            super().record_transition(**kwargs)
        finally:
            if self._rnn_initial_states is self._pre_step_rnn:
                self._rnn_initial_states = corrupted  # super() did not advance it
