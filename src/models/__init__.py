"""Actor/critic architectures -- Block G's RQ2 ladder.

`docs/MODELS.md` is the specification. Three actor rungs isolating one factor
each (MLP -> DeepSets -> GNN), one critic shared by all three so RQ2 measures
the actor and nothing else.
"""

from .actor import ARCHITECTURES, SwarmActor, SwarmActorRNN, parameter_count
from .critic import SwarmCritic, SwarmCriticRNN

__all__ = [
    "ARCHITECTURES",
    "SwarmActor",
    "SwarmActorRNN",
    "SwarmCritic",
    "SwarmCriticRNN",
    "parameter_count",
]
