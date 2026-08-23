"""Non-learned controls -- Block E.

`b0.py` is the scripted geometric baseline THESIS_PLAN §3 calls B0: the control
that answers *"is MARL earning its keep?"*. `docs/MODELS.md` makes it a hard
requirement rather than a curiosity -- every architecture must at least match it,
so a weak B0 flatters every result in Chapter 6.

`evaluate.py` is the rollout harness. It takes any callable of the observation,
so Block G evaluates checkpoints through the same code path and the numbers stay
comparable.

Design and the measurements behind it: `docs/BLOCK_E.md`.
"""

from .b0 import VARIANTS, B0Config, B0Policy
from .evaluate import RolloutMetrics, rollout

__all__ = ["VARIANTS", "B0Config", "B0Policy", "RolloutMetrics", "rollout"]
