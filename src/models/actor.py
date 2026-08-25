"""The three actor rungs of RQ2's ladder, on one observation contract.

    | rung     | neighbours read as        | perm-invariant | size-agnostic | edges |
    | mlp      | concatenated, max-N padded| no             | no            | no    |
    | deepsets | shared embed, then pooled | yes            | yes           | no    |
    | gnn      | same, messages see e_ij   | yes            | yes           | yes   |

MLP -> DeepSets isolates permutation invariance; DeepSets -> GNN isolates the
*relational* part, which is RQ2's actual claim (`docs/MODELS.md`).

**DeepSets is the GNN with `e_ij` zeroed** -- literally the same class, the same
code path, the same parameter count, the same optimiser, one input masked. That
is what makes the ablation exact and is why the layer is a custom MPNN
(Gilmer et al., 2017: `message(x_i, x_j, e_ij) = MLP([x_i, x_j, e_ij])`) rather
than two differently-named PyG layers, which would always invite "maybe GATv2 is
just a better layer".

⛔ `SAGEConv` is never an option here: it cannot ingest edge features at all, so
it would silently collapse the GNN rung into the DeepSets rung and RQ2 would
measure nothing. It is also the layer people reach for by default.

## One message-passing layer, not two -- and why that is not a shortcut

`docs/MODELS.md` says two message-passing layers is the ceiling the graph
justifies, reasoning from a graph of `N` drone nodes. **The actor does not hold
that graph.** Its observation is `(B, 108)`: its own ego block plus 7 neighbour
slots -- a *star* centred on itself. A second layer over the true swarm graph
would give drone `i` access to `j`'s aggregate of `k`, which `i` does not
possess and could only obtain by exchanging embeddings with its neighbours. That
would hand the GNN rung strictly more information than the MLP and DeepSets
rungs get, and RQ2's comparison would confound architecture with information --
the exact confound the zeroed-`e_ij` design exists to rule out.

So: one message-passing layer over the local star, with depth living in the
message and update MLPs, where it is capacity rather than reach. MODELS.md's own
diameter-1 argument already says one layer reaches every drone; the second layer
it allows for is unavailable to an agent-local actor, not merely unnecessary.
Two-layer-with-communication belongs in future work, where it is a different
claim about the control plane rather than about relational structure.

Every rung consumes `obs["flat"]` and unpacks it with `core.unpack_flat`, so the
max-N padding is identical across rungs by construction rather than by
discipline, and all three accept `N` in {3, 5, 8} without reshaping.
"""

from __future__ import annotations

from typing import Any

import torch
from skrl.models.torch import GaussianMixin, Model
from torch import Tensor, nn

from ..env.core import ACTION_DIM, EDGE_DIM, EGO_DIM, FLAT_DIM, NEIGHBOUR_DIM, unpack_flat
from .recurrence import rnn_specification, run_gru

ARCHITECTURES = ("mlp", "deepsets", "gnn")

#: Widths chosen so the three rungs land within 20 % of each other on parameter
#: count (`docs/MODELS.md` rule 3: the comparison must not be capacity-vs-
#: capacity). `test_actor.py` asserts the spread, so these cannot drift apart
#: silently. They are a starting point for the equal-budget search, not findings.
DEFAULT_HIDDEN = 128
DEFAULT_MLP_HIDDEN = 232


def _mlp(sizes: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class FlatTrunk(nn.Module):
    """The MLP rung. Reads the padded 108-vector as one flat block.

    Not permutation-invariant and not size-agnostic: neighbour slot 3 is a
    different set of weights from slot 4, so relabelling the swarm changes the
    output. That is the property RQ2's first contrast measures, so it is a
    faithful implementation rather than a weak one -- and the max-N padding plus
    validity bits are what let it be *evaluated* off-N at all.
    """

    def __init__(self, hidden: int = DEFAULT_MLP_HIDDEN):
        super().__init__()
        self.net = _mlp([FLAT_DIM, hidden, hidden, hidden])
        self.out_dim = hidden

    def forward(self, flat: Tensor) -> Tensor:
        return torch.tanh(self.net(flat))


class RelationalTrunk(nn.Module):
    """The DeepSets and GNN rungs. One class, one flag, one masked input.

    `use_edges=False` zeroes `e_ij` on the way into the message function and
    changes nothing else -- the weights that would multiply it still exist and
    are still trained, so the two rungs have identical parameter counts by
    construction. Pooling is a **mean over valid neighbours**, not a sum: a sum
    scales with `N` and would make the off-N transfer columns measure a
    normalisation artefact instead of the architecture.
    """

    def __init__(self, hidden: int = DEFAULT_HIDDEN, use_edges: bool = True):
        super().__init__()
        self.use_edges = use_edges
        self.ego = _mlp([EGO_DIM, hidden, hidden])
        self.neighbour = _mlp([NEIGHBOUR_DIM, hidden, hidden])
        self.message = _mlp([2 * hidden + EDGE_DIM, hidden, hidden])
        self.update = _mlp([2 * hidden, hidden, hidden])
        self.out_dim = hidden

    def forward(self, flat: Tensor) -> Tensor:
        parts = unpack_flat(flat)
        valid = parts["valid"].unsqueeze(-1)  # (..., 7, 1)

        h_i = torch.tanh(self.ego(parts["ego"]))  # (..., H)
        h_j = torch.tanh(self.neighbour(parts["neighbour"]))  # (..., 7, H)

        edge = parts["edge"]
        if not self.use_edges:
            edge = torch.zeros_like(edge)

        pair = torch.cat([h_i.unsqueeze(-2).expand_as(h_j), h_j, edge], dim=-1)
        msg = torch.tanh(self.message(pair)) * valid
        # Mean over *valid* neighbours; `clamp_min(1)` keeps N=1 finite rather
        # than dividing by zero, and the padded slots contribute nothing.
        agg = msg.sum(dim=-2) / valid.sum(dim=-2).clamp_min(1.0)

        return torch.tanh(self.update(torch.cat([h_i, agg], dim=-1)))


def build_trunk(architecture: str, hidden: int | None = None) -> nn.Module:
    if architecture == "mlp":
        return FlatTrunk(hidden or DEFAULT_MLP_HIDDEN)
    if architecture == "deepsets":
        return RelationalTrunk(hidden or DEFAULT_HIDDEN, use_edges=False)
    if architecture == "gnn":
        return RelationalTrunk(hidden or DEFAULT_HIDDEN, use_edges=True)
    raise ValueError(f"architecture must be one of {ARCHITECTURES}, got {architecture!r}")


class SwarmActor(GaussianMixin, Model):
    """Agent-local stochastic actor. Reads `observations`, never `states`.

    Reading `states` here would quietly turn CTDE into centralized execution and
    invalidate the whole framing (`docs/ENVIRONMENT.md` -> Observations), so the
    separation is structural: this class has no access to the critic's state.

    The action is motion only -- 3-dim. ⛔ Transmit power is not an action and
    never becomes one: three framings, three nulls (`docs/NEGATIVE_RESULTS.md`).
    """

    def __init__(
        self,
        observation_space: Any,
        action_space: Any,
        device: Any,
        architecture: str = "mlp",
        hidden: int | None = None,
        initial_log_std: float = -0.5,
        min_log_std: float = -20.0,
    ):
        Model.__init__(
            self, observation_space=observation_space, action_space=action_space, device=device
        )
        # ⚠️ `clip_actions=False` on purpose, and it is not a detail.
        #
        # skrl's GaussianMixin with `clip_actions=True` clamps the sampled
        # action to the action space AND then evaluates the log-probability of
        # that clamped value under the *unclamped* Normal. Every sample from the
        # tails is therefore recorded as if it had landed exactly on the
        # boundary, which piles empirical mass on +-1 that the density used in
        # the PPO ratio does not account for. Measured here: the action standard
        # deviation rises monotonically with no entropy bonus anywhere in the
        # config, actions saturate at the corners, and mission-capable falls from
        # ~30 % to ~5 % on a reward whose only term IS mission-capable.
        #
        # The standard treatment is to let the distribution stay unbounded and
        # clip in the environment, so the log-probability belongs to the action
        # that was actually sampled. `core._advance_drones` already opens with
        # `actions.clamp(-1.0, 1.0)`, so the bound is enforced either way and
        # only the density changes.
        #
        # `min_log_std` floors exploration. skrl's default (-20) is effectively no
        # floor, and with `entropy_loss_scale = 0` the standard deviation then
        # shrinks monotonically as the policy grows confident: measured on the
        # full mission it reached **0.061** by 13 M steps, at which point the
        # policy is deterministic, stops exploring, and sits in whatever local
        # behaviour it found. A floor is preferable to an entropy bonus for that
        # job because it bounds the policy CLASS instead of adding a term to the
        # objective -- an entropy bonus large enough to matter here (0.01) instead
        # drove the deviation UP, to 1.11, which is its own failure.
        GaussianMixin.__init__(self, clip_actions=False, min_log_std=min_log_std)
        self.architecture = architecture
        self.trunk = build_trunk(architecture, hidden)
        self.head = nn.Linear(self.trunk.out_dim, ACTION_DIM)
        # One state-independent log-std per action dimension, the PPO standard.
        # -0.5 (sigma ~ 0.61) against a tanh-bounded mean in [-1, 1]: exploratory
        # without spending most of its mass on the clip boundary.
        self.log_std = nn.Parameter(torch.full((ACTION_DIM,), initial_log_std))

    def compute(self, inputs: dict[str, Tensor], role: str = ""):
        features = self.trunk(inputs["observations"])
        return torch.tanh(self.head(features)), {"log_std": self.log_std}


def parameter_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


class SwarmActorRNN(SwarmActor):
    """The same three rungs, with one GRU between the trunk and the head.

    ## Why memory, and why it is not a tuning knob

    Measured on the full mission (`docs/BLOCK_G.md`): B0 holds the observer role
    for a mean of **264 steps**; every feedforward policy manages **28–35**, and
    hands the role over *more often than a random policy does*. The cause is
    representational rather than a matter of tuning. The actor is a pure function
    of its current observation, and `hvt_rel` is **zeroed when the target is not
    seen** -- so nothing in the network can express *"I am the observer and I am
    holding station"*, and a drone that loses line of sight is instantly blind
    with no way to recover the target's last known position. B0 has both: a
    carried belief and an explicit role. No value of `d_ref`, entropy or training
    budget can add state to a stateless function.

    Recurrent actors are the *published* MAPPO configuration -- Yu et al. (2022)
    use them for most benchmarks -- so this is the standard setting rather than
    an invention.

    ## What it does NOT change

    The GRU sits between the trunk and the policy head and is **identical in all
    three rungs**, so RQ2 still isolates the trunk: MLP vs DeepSets vs GNN, same
    memory, same parameter budget, `e_ij` masked for DeepSets exactly as before.
    The critic stays feedforward and stays identical across conditions.

    The sequence handling follows skrl's own GRU pattern: during collection the
    model steps one transition at a time; during the update it is fed
    `(N_sequences, L)` blocks with the hidden state of each sequence's first step,
    and the state is **zeroed wherever an episode ended inside the sequence** so
    memory never leaks across an episode boundary.
    """

    def __init__(
        self,
        observation_space: Any,
        action_space: Any,
        device: Any,
        architecture: str = "mlp",
        hidden: int | None = None,
        initial_log_std: float = -0.5,
        min_log_std: float = -20.0,
        num_envs: int = 1,
        rnn_hidden: int = 128,
        rnn_layers: int = 1,
        sequence_length: int = 16,
    ):
        super().__init__(
            observation_space,
            action_space,
            device,
            architecture=architecture,
            hidden=hidden,
            initial_log_std=initial_log_std,
            min_log_std=min_log_std,
        )
        self.num_envs = num_envs
        self.rnn_hidden = rnn_hidden
        self.rnn_layers = rnn_layers
        self.sequence_length = sequence_length
        self.gru = nn.GRU(
            input_size=self.trunk.out_dim,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
        )
        self.head = nn.Linear(rnn_hidden, ACTION_DIM)

    def get_specification(self) -> dict[str, Any]:
        return rnn_specification(
            self.sequence_length, self.rnn_layers, self.num_envs, self.rnn_hidden
        )

    def compute(self, inputs: dict[str, Tensor], role: str = ""):
        rnn_out, hidden = run_gru(
            self.gru,
            self.trunk(inputs["observations"]),
            inputs["rnn"][0],
            training=self.training,
            sequence_length=self.sequence_length,
            num_layers=self.rnn_layers,
            terminated=inputs.get("terminated"),
            truncated=inputs.get("truncated"),
        )
        return torch.tanh(self.head(rnn_out)), {"log_std": self.log_std, "rnn": [hidden]}
