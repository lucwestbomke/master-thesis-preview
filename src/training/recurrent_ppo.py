"""skrl's `PPO_RNN`, corrected: two bugs, both from the same cause.

`skrl/agents/torch/ppo/ppo_rnn.py` in 2.1.0 is an **un-migrated copy of an
older PPO**. It carries its own stale `compute_gae` and its own stale
`record_transition`, neither of which was updated when skrl reworked
truncation handling in `ppo.py` and `mappo.py`. Diffed directly against
`MAPPO` -- the class every working number in this project came from -- the
two paths differ in three places, and this class closes all three.

## Bug 1: the stored hidden state is one step ahead

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

## Bug 2: the truncation bootstrap, and it is the one that mattered

Measured (`docs/BLOCK_G.md`): with **feedforward** models on the `PPO_RNN` code
path -- `_rnn = False`, i.e. plain PPO -- stage 1 peaks at 52 % and collapses to
4 %, while the identical models under `MAPPO` reach 79 %. The GRU was never
involved. Two differences against `MAPPO`, and they compound:

| | `MAPPO` | `PPO_RNN` 2.1.0 |
|---|---|---|
| bootstrap value at truncation | `V(next_observations, next_states)` | `V(observations, states)` -- the **current** state |
| GAE mask | `~(terminated OR truncated)` | `~terminated` |

The first means `PPO_RNN` never reads `next_observations` at all, so the
`final_observations()` / `final_states()` seam Block D added -- precisely so the
learner would not bootstrap off a fresh episode's opening -- is silently
discarded on this path.

The second is the severe one. `not_terminated` is **True at a truncation**, so
GAE keeps recursing across the episode boundary: the truncation step gets
`gamma * (values[i+1] + lambda * advantage)` where `values[i+1]` is the *next*
episode's opening value and `advantage` is the *next* episode's advantage. The
bootstrap is therefore counted twice and the next episode's credit is propagated
backwards through the reset, with `gamma = 0.997` and `lambda = 0.95` carrying it
a long way. Every rollout that spans a reset is contaminated, and because the
envs reset in lockstep, every env is contaminated on the same step.

## The fix

Do exactly what `MAPPO` does, without reimplementing `update()`:

1. bootstrap with `V(next_observations, next_states)`, computed here, with
   skrl's own (wrong) bootstrap switched off for the duration;
2. fold `truncated` into the memory's `terminated` tensor before `update()`
   runs, so the stale `compute_gae`'s `not_terminated` *is* `MAPPO`'s
   `not_done`. `terminated` is read nowhere else in `update` except the RNN
   boundary zeroing, which takes `terminated | truncated` regardless.

⚠️ Remove this class only after checking the installed skrl. The invariants it
restores are asserted by `test_recurrent.py`, which will fail loudly if a future
version fixes these and this workaround starts double-correcting.
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

    # --- bug 2a: bootstrap off the NEXT state, as MAPPO does ---------------- #

    def _bootstrap(self, kwargs: dict[str, Any]) -> None:
        """Add `gamma * V(next_observations, next_states) * truncated` to rewards.

        `MAPPO.record_transition`, transcribed. skrl's own `PPO_RNN` version uses
        `V(observations, states)` -- the state the agent is leaving rather than
        the one it is truncated at -- and so never reads `next_observations`.
        """
        truncated = kwargs["truncated"]
        if not truncated.any():
            return
        inputs = {
            "observations": self._observation_preprocessor(kwargs["next_observations"]),
            "states": self._state_preprocessor(kwargs["next_states"]),
        }
        if self._rnn:
            # The pre-step value state IS the right conditioning for the next
            # observation: the GRU consumes s_{t+1} from h_t, which is exactly
            # `V(s_{t+1} | history up to t)`. `value.act` is pure, so reading the
            # agent's state here does not perturb skrl's own bookkeeping.
            inputs["rnn"] = self._rnn_initial_states["value"]
        with torch.no_grad():
            next_values, _ = self.value.act(inputs, role="value")
            next_values = self._value_preprocessor(next_values, inverse=True)
        kwargs["rewards"] = kwargs["rewards"] + self.cfg.discount_factor * next_values * truncated

    def record_transition(self, **kwargs: Any) -> None:
        bootstrap = self.training and self.memory is not None and self.cfg.time_limit_bootstrap
        if bootstrap:
            self._bootstrap(kwargs)

        corrupted = None
        if self._rnn and self._pre_step_rnn is not None:
            # Swap the corrupted "initial" states for the snapshot, let skrl
            # store them, and let its own trailing
            # `_rnn_initial_states = _rnn_final_states` advance the state for the
            # next step exactly as it does today.
            corrupted = self._rnn_initial_states
            self._rnn_initial_states = self._pre_step_rnn

        # skrl would otherwise add its own bootstrap off the CURRENT state on
        # top of the one just added.
        flag = self.cfg.time_limit_bootstrap
        self.cfg.time_limit_bootstrap = False
        try:
            super().record_transition(**kwargs)
        finally:
            self.cfg.time_limit_bootstrap = flag
            if corrupted is not None and self._rnn_initial_states is self._pre_step_rnn:
                self._rnn_initial_states = corrupted  # super() did not advance it

    # --- bug 2b: stop GAE recursing through the reset ------------------------ #

    def update(self, *, timestep: int, timesteps: int) -> None:
        """Fold `truncated` into `terminated`, then run skrl's update unchanged.

        `PPO_RNN`'s stale `compute_gae` masks on `terminated` alone, so a
        truncation lets the recursion carry the next episode's value and
        advantage backwards across the reset. `MAPPO` masks on
        `terminated | truncated`; writing the OR into the tensor `compute_gae`
        reads is the same thing, and `terminated` is used nowhere else in
        `update` except the RNN boundary zeroing, which ORs them anyway.
        """
        if self.cfg.time_limit_bootstrap and self.memory is not None:
            terminated = self.memory.get_tensor_by_name("terminated")
            truncated = self.memory.get_tensor_by_name("truncated")
            self.memory.set_tensor_by_name("terminated", terminated | truncated)
        super().update(timestep=timestep, timesteps=timesteps)
