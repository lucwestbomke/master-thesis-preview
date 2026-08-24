"""What keeps RQ2's architecture comparison honest.

Each test here pins one of `docs/MODELS.md`'s rules. They are cheap and they all
guard silent failures: an architecture that is secretly another architecture, a
comparison that is capacity-vs-capacity, or a rung that cannot be evaluated
off-`N` at all.
"""

from __future__ import annotations

import gymnasium
import numpy as np
import pytest
import torch

from ..env.core import ACTION_DIM, EDGE_DIM, EGO_DIM, FLAT_DIM, N_MAX, NEIGHBOUR_DIM
from .actor import ARCHITECTURES, RelationalTrunk, SwarmActor, build_trunk, parameter_count
from .critic import SwarmCritic

OBS_SPACE = gymnasium.spaces.Box(-np.inf, np.inf, shape=(FLAT_DIM,), dtype=np.float32)
ACT_SPACE = gymnasium.spaces.Box(-1.0, 1.0, shape=(ACTION_DIM,), dtype=np.float32)


def flat_batch(rows: int = 16, num_drones: int = 5, seed: int = 0) -> torch.Tensor:
    """A `(rows, 108)` observation with `num_drones - 1` valid neighbour slots."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, FLAT_DIM, generator=g)
    k = N_MAX - 1
    real = num_drones - 1
    valid = torch.zeros(rows, k)
    valid[:, :real] = 1.0
    x[..., FLAT_DIM - k :] = valid
    # padded slots carry zeros, exactly as `_pack` writes them
    nb = x[..., EGO_DIM : EGO_DIM + k * NEIGHBOUR_DIM].unflatten(-1, (k, NEIGHBOUR_DIM))
    eg = x[..., EGO_DIM + k * NEIGHBOUR_DIM : FLAT_DIM - k].unflatten(-1, (k, EDGE_DIM))
    nb[:, real:] = 0.0
    eg[:, real:] = 0.0
    return x


def permute_neighbours(flat: torch.Tensor, order: list[int]) -> torch.Tensor:
    k = N_MAX - 1
    a, b = EGO_DIM, EGO_DIM + k * NEIGHBOUR_DIM
    c = b + k * EDGE_DIM
    out = flat.clone()
    idx = torch.tensor(order)
    out[..., a:b] = flat[..., a:b].unflatten(-1, (k, NEIGHBOUR_DIM))[:, idx].flatten(-2)
    out[..., b:c] = flat[..., b:c].unflatten(-1, (k, EDGE_DIM))[:, idx].flatten(-2)
    out[..., c:] = flat[..., c:][:, idx]
    return out


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_every_architecture_accepts_every_swarm_size(architecture):
    """RQ2 trains at N=5 and evaluates zero-shot at N in {3,5,8}. A rung that
    cannot take the off-N observation cannot be in the comparison at all."""
    actor = SwarmActor(OBS_SPACE, ACT_SPACE, "cpu", architecture=architecture)
    for n in (3, 5, 8):
        mean, outputs = actor.compute({"observations": flat_batch(num_drones=n)})
        assert mean.shape == (16, ACTION_DIM)
        assert torch.isfinite(mean).all()
        assert outputs["log_std"].shape == (ACTION_DIM,)


def test_parameter_counts_are_within_twenty_percent():
    """docs/MODELS.md rule 3. Otherwise the comparison is capacity-vs-capacity
    and the result says nothing about relational structure."""
    counts = {a: parameter_count(build_trunk(a)) for a in ARCHITECTURES}
    spread = max(counts.values()) / min(counts.values())
    assert spread <= 1.2, counts


def test_deepsets_is_the_gnn_with_the_edge_features_zeroed():
    """The ablation has to be exact: same class, same code path, same parameter
    count, one input masked. Two differently-named layers would always invite
    'maybe the GNN layer is just better'."""
    gnn = build_trunk("gnn")
    deepsets = build_trunk("deepsets")
    assert isinstance(gnn, RelationalTrunk) and isinstance(deepsets, RelationalTrunk)
    assert parameter_count(gnn) == parameter_count(deepsets)
    assert gnn.use_edges and not deepsets.use_edges

    deepsets.load_state_dict(gnn.state_dict())
    x = flat_batch()
    k = N_MAX - 1
    b = EGO_DIM + k * NEIGHBOUR_DIM
    zeroed = x.clone()
    zeroed[..., b : b + k * EDGE_DIM] = 0.0

    assert torch.allclose(deepsets(x), gnn(zeroed), atol=1e-6)
    assert not torch.allclose(gnn(x), gnn(zeroed), atol=1e-4), "edges must actually matter"


def test_the_relational_rungs_are_permutation_invariant_and_the_mlp_is_not():
    """The first contrast of the ladder, MLP -> DeepSets, *is* this property."""
    x = flat_batch(num_drones=5)
    order = [3, 0, 2, 1, 4, 5, 6]  # the four valid slots shuffled among themselves
    y = permute_neighbours(x, order)

    for architecture in ("deepsets", "gnn"):
        trunk = build_trunk(architecture)
        assert torch.allclose(trunk(x), trunk(y), atol=1e-6), architecture

    mlp = build_trunk("mlp")
    assert not torch.allclose(mlp(x), mlp(y), atol=1e-4)


def test_padded_neighbour_slots_do_not_reach_the_pooling():
    """Off-N transfer is only meaningful if the padding is inert. A padded slot
    that leaked into the mean would make N=3 a different task, not a smaller
    one."""
    for architecture in ("deepsets", "gnn"):
        trunk = build_trunk(architecture)
        x = flat_batch(num_drones=3)
        polluted = x.clone()
        k = N_MAX - 1
        a, b = EGO_DIM, EGO_DIM + k * NEIGHBOUR_DIM
        nb = polluted[..., a:b].unflatten(-1, (k, NEIGHBOUR_DIM))
        nb[:, 2:] = 7.0  # garbage in every invalid slot
        polluted[..., a:b] = nb.flatten(-2)
        assert torch.allclose(trunk(x), trunk(polluted), atol=1e-6), architecture


def test_the_actor_never_reads_the_critic_state():
    """CTDE is a claim this project makes, and a wrapper that quietly handed the
    actor the global state would invalidate it silently."""
    actor = SwarmActor(OBS_SPACE, ACT_SPACE, "cpu")
    mean, _ = actor.compute({"observations": flat_batch(), "states": torch.randn(16, 54)})
    other, _ = actor.compute({"observations": flat_batch(), "states": torch.randn(16, 54) * 9})
    assert torch.equal(mean, other)


def test_actions_are_not_clipped_inside_the_distribution():
    """⚠️ Regression pin, and the failure it guards is silent and severe.

    skrl's `clip_actions=True` clamps the sample to the action space and then
    evaluates its log-probability under the *unclamped* Normal, so every tail
    draw is recorded as if it landed exactly on the boundary. Measured on this
    task: the action standard deviation rises monotonically with no entropy
    bonus anywhere in the config, actions saturate at the corners, and
    mission-capable falls from ~30 % to ~5 % under a reward whose only term is
    mission-capable. `core._advance_drones` clamps actions itself, so the bound
    is enforced either way and only the density changes.
    """
    actor = SwarmActor(OBS_SPACE, ACT_SPACE, "cpu")
    assert actor._g_clip_actions is False

    # and the recorded log-probability must be the one of the action returned
    actions, outputs = actor.act({"observations": flat_batch(rows=4096, seed=1)})
    replayed = actor.act({"observations": flat_batch(rows=4096, seed=1), "taken_actions": actions})
    assert torch.allclose(outputs["log_prob"], replayed[1]["log_prob"], atol=1e-5)
    assert (actions.abs() > 1.0).any(), "an unclipped Gaussian must leave the box sometimes"


def test_the_critic_is_one_design_with_no_architecture_argument():
    """docs/MODELS.md: if both the actor and the critic vary, RQ2 is confounded."""
    import inspect

    assert "architecture" not in inspect.signature(SwarmCritic.__init__).parameters
    state_space = gymnasium.spaces.Box(-np.inf, np.inf, shape=(54,), dtype=np.float32)
    critic = SwarmCritic(state_space, ACT_SPACE, "cpu")
    value, _ = critic.compute({"states": torch.randn(16, 54)})
    assert value.shape == (16, 1)


# --------------------------------------------------------------------------- #
# The recurrent variant
# --------------------------------------------------------------------------- #


def make_rnn(architecture: str = "mlp", num_envs: int = 4, sequence_length: int = 4):
    from .actor import SwarmActorRNN

    return SwarmActorRNN(
        OBS_SPACE,
        ACT_SPACE,
        "cpu",
        architecture=architecture,
        num_envs=num_envs,
        sequence_length=sequence_length,
    )


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_the_gru_is_identical_in_every_rung(architecture):
    """RQ2 must still isolate the trunk. If memory differed between rungs the
    comparison would be memory-vs-architecture."""
    model = make_rnn(architecture)
    assert model.gru.input_size == model.trunk.out_dim
    assert model.gru.hidden_size == model.rnn_hidden
    spec = model.get_specification()["rnn"]
    assert spec["sizes"] == [(model.rnn_layers, 4, model.rnn_hidden)]
    counts = {a: parameter_count(make_rnn(a)) for a in ARCHITECTURES}
    assert max(counts.values()) / min(counts.values()) <= 1.2, counts


def test_the_hidden_state_actually_changes_the_action():
    """A GRU wired in but ignored would look exactly like a working one on every
    shape test, and would quietly reproduce the memoryless failure it was added
    to fix."""
    model = make_rnn("gnn").eval()
    obs = flat_batch(rows=4)
    zero = [torch.zeros(model.rnn_layers, 4, model.rnn_hidden)]
    other = [torch.randn(model.rnn_layers, 4, model.rnn_hidden)]

    a, out_a = model.compute({"observations": obs, "rnn": zero})
    b, _ = model.compute({"observations": obs, "rnn": other})
    assert not torch.allclose(a, b, atol=1e-5), "the action ignores the hidden state"
    assert not torch.allclose(out_a["rnn"][0], zero[0]), "the hidden state never advances"


def test_memory_does_not_leak_across_an_episode_boundary():
    """The one correctness rule of a recurrent policy on truncated episodes: a
    new episode must not condition on the previous episode's target."""
    model = make_rnn("mlp", num_envs=2, sequence_length=4).train()
    obs = flat_batch(rows=8)  # 2 sequences x 4 steps
    hidden = [torch.randn(model.rnn_layers, 8, model.rnn_hidden)]

    done = torch.zeros(8, 1, dtype=torch.bool)
    done[1] = True  # episode ends inside the first sequence, at step 1
    no_done = torch.zeros(8, 1, dtype=torch.bool)

    _, ended = model.compute(
        {"observations": obs, "rnn": hidden, "terminated": done, "truncated": no_done}
    )
    _, intact = model.compute(
        {"observations": obs, "rnn": hidden, "terminated": no_done, "truncated": no_done}
    )
    assert not torch.allclose(ended["rnn"][0], intact["rnn"][0], atol=1e-6), (
        "the carried state is identical whether or not an episode ended -- the reset is not firing"
    )


def test_training_and_collection_shapes_agree():
    """Collection steps one transition at a time; the update replays sequences.
    Both must produce one action per row or the ratio is computed against the
    wrong actions."""
    model = make_rnn("deepsets", num_envs=3, sequence_length=5)
    model.eval()
    act, _ = model.compute({"observations": flat_batch(rows=3), "rnn": [torch.zeros(1, 3, 128)]})
    assert act.shape == (3, ACTION_DIM)

    model.train()
    act, extra = model.compute(
        {
            "observations": flat_batch(rows=15),  # 3 sequences x 5
            "rnn": [torch.zeros(1, 15, 128)],
            "terminated": torch.zeros(15, 1, dtype=torch.bool),
            "truncated": torch.zeros(15, 1, dtype=torch.bool),
        }
    )
    assert act.shape == (15, ACTION_DIM)
    assert extra["rnn"][0].shape == (1, 3, 128)
