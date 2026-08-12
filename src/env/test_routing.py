"""
Relay-routing tests. Node layout in these fixtures: 0..M-2 are drones, M-1 is
the MCV (the destination).
"""

import pytest
import torch

from .routing import best_relay_capacity, best_relay_path, link_alive


def _chain(caps: dict, m: int = 3, b: int = 1) -> torch.Tensor:
    cap = torch.zeros(b, m, m)
    for (i, j), v in caps.items():
        cap[:, i, j] = v
    return cap


def test_two_hop_chain_pays_the_halfduplex_divisor():
    # 0 -> 1 -> 2, both hops 30 Mbps. Two hops => 30 / 2 = 15 Mbps end to end.
    cap = _chain({(0, 1): 30.0, (1, 2): 30.0})
    src = torch.tensor([[True, False, False]])
    got = best_relay_capacity(cap, src, dst_index=2, max_hops=2)
    assert got.item() == pytest.approx(15.0, abs=1e-4)


def test_bottleneck_hop_dominates():
    # min(30, 6) = 6, over two hops => 3 Mbps.
    cap = _chain({(0, 1): 30.0, (1, 2): 6.0})
    src = torch.tensor([[True, False, False]])
    got = best_relay_capacity(cap, src, dst_index=2, max_hops=2)
    assert got.item() == pytest.approx(3.0, abs=1e-4)


def test_direct_link_wins_when_relaying_is_not_worth_it():
    # Direct 20 Mbps beats the 30/2 = 15 Mbps two-hop route.
    cap = _chain({(0, 1): 30.0, (1, 2): 30.0, (0, 2): 20.0})
    src = torch.tensor([[True, False, False]])
    got = best_relay_capacity(cap, src, dst_index=2, max_hops=2)
    assert got.item() == pytest.approx(20.0, abs=1e-4)


def test_relaying_wins_when_the_direct_link_is_weak():
    # Direct 10 Mbps loses to the 30/2 = 15 Mbps two-hop route.
    cap = _chain({(0, 1): 30.0, (1, 2): 30.0, (0, 2): 10.0})
    src = torch.tensor([[True, False, False]])
    got = best_relay_capacity(cap, src, dst_index=2, max_hops=2)
    assert got.item() == pytest.approx(15.0, abs=1e-4)


def test_extra_hops_must_earn_their_place():
    """A 3-hop detour only wins if its bottleneck exceeds 1.5x the 2-hop one.

    This is exactly the pressure that stops the swarm from learning long
    daisy-chains, so it is worth pinning explicitly.
    """
    # 0->1->3 gives 20/2 = 10.  0->1->2->3 gives 24/3 = 8.  Two hops wins.
    cap = _chain({(0, 1): 40.0, (1, 3): 20.0, (1, 2): 24.0, (2, 3): 24.0}, m=4)
    src = torch.tensor([[True, False, False, False]])
    assert best_relay_capacity(cap, src, 3, max_hops=3).item() == pytest.approx(10.0, abs=1e-4)

    # Raise the 3-hop bottleneck to 36: 36/3 = 12 > 10. Now three hops wins.
    cap = _chain({(0, 1): 40.0, (1, 3): 20.0, (1, 2): 36.0, (2, 3): 36.0}, m=4)
    assert best_relay_capacity(cap, src, 3, max_hops=3).item() == pytest.approx(12.0, abs=1e-4)


def test_spatial_reuse_caps_the_divisor_at_three():
    """A 4-hop chain pays the same penalty as a 3-hop one.

    Non-adjacent hops transmit concurrently, so a linear chain saturates near
    1/3 of single-link capacity rather than degrading as 1/n. Pressure toward
    short chains then comes from interference and per-hop SINR, not from an
    arbitrary divisor.
    """
    cap = _chain({(0, 1): 30.0, (1, 2): 30.0, (2, 3): 30.0, (3, 4): 30.0}, m=5)
    src = torch.tensor([[True, False, False, False, False]])
    got = best_relay_capacity(cap, src, dst_index=4, max_hops=4)
    assert got.item() == pytest.approx(10.0, abs=1e-4)  # 30 / min(4, 3)


def test_strict_tdma_recovers_the_per_hop_divisor():
    cap = _chain({(0, 1): 30.0, (1, 2): 30.0, (2, 3): 30.0, (3, 4): 30.0}, m=5)
    src = torch.tensor([[True, False, False, False, False]])
    got = best_relay_capacity(cap, src, 4, max_hops=4, reuse_limit=4)
    assert got.item() == pytest.approx(7.5, abs=1e-4)  # 30 / 4


def test_reuse_limit_one_disables_the_halfduplex_penalty():
    cap = _chain({(0, 1): 30.0, (1, 2): 30.0})
    src = torch.tensor([[True, False, False]])
    got = best_relay_capacity(cap, src, 2, max_hops=2, reuse_limit=1)
    assert got.item() == pytest.approx(30.0, abs=1e-4)


def test_no_observer_means_no_mission_capacity():
    # Perfect radio links, but nobody is looking at the HVT -> nothing to relay.
    cap = _chain({(0, 1): 100.0, (1, 2): 100.0, (0, 2): 100.0})
    src = torch.tensor([[False, False, False]])
    got = best_relay_capacity(cap, src, dst_index=2, max_hops=2)
    assert got.item() == pytest.approx(0.0, abs=1e-9)


def test_best_observer_is_selected_automatically():
    # Two drones observe the HVT; the router must pick the better-connected one.
    cap = _chain({(0, 2): 4.0, (1, 2): 18.0}, m=3)
    src = torch.tensor([[True, True, False]])
    got = best_relay_capacity(cap, src, dst_index=2, max_hops=2)
    assert got.item() == pytest.approx(18.0, abs=1e-4)


def test_disconnected_destination_yields_zero():
    cap = _chain({(0, 1): 50.0})
    src = torch.tensor([[True, False, False]])
    got = best_relay_capacity(cap, src, dst_index=2, max_hops=2)
    assert got.item() == pytest.approx(0.0, abs=1e-9)


def test_source_sentinel_never_leaks_into_the_result():
    """Guards the 1e9 sentinel used to seed source nodes."""
    cap = _chain({(0, 1): 7.0, (1, 2): 7.0})
    src = torch.tensor([[True, True, True]])  # MCV wrongly flagged as a source
    got = best_relay_capacity(cap, src, dst_index=2, max_hops=3)
    assert got.item() < 1e3


def test_batched_independence():
    cap = torch.zeros(3, 3, 3)
    cap[0, 0, 1], cap[0, 1, 2] = 30.0, 30.0  # -> 15
    cap[1, 0, 2] = 9.0  # -> 9
    # env 2 left disconnected                 # -> 0
    src = torch.tensor([[True, False, False]] * 3)
    got = best_relay_capacity(cap, src, dst_index=2, max_hops=2)
    assert torch.allclose(got, torch.tensor([15.0, 9.0, 0.0]), atol=1e-4)


def test_cycles_are_never_beneficial():
    # A tempting cycle 0->1->0->2 can only lower the bottleneck and raise hops.
    cap = _chain({(0, 1): 99.0, (1, 0): 99.0, (0, 2): 12.0}, m=3)
    src = torch.tensor([[True, False, False]])
    got = best_relay_capacity(cap, src, dst_index=2, max_hops=4)
    assert got.item() == pytest.approx(12.0, abs=1e-4)


def test_link_alive_threshold():
    caps = torch.tensor([4.99, 5.0, 5.01])
    assert torch.equal(link_alive(caps, 5.0), torch.tensor([False, True, True]))


def test_scales_to_full_swarm_shape():
    b, m = 16, 9  # 8 drones + MCV, the largest configuration in the study
    cap = torch.rand(b, m, m) * 40.0
    src = torch.zeros(b, m, dtype=torch.bool)
    src[:, 0] = True
    got = best_relay_capacity(cap, src, dst_index=m - 1, max_hops=m - 1)
    assert got.shape == (b,)
    assert torch.isfinite(got).all()
    assert torch.all(got >= 0.0) and torch.all(got <= 40.0)


# --------------------------------------------------------------------------- #
# Path extraction -- who actually carries the chain
# --------------------------------------------------------------------------- #


def test_path_capacity_matches_the_capacity_only_dp():
    """The property that matters: adding back-pointers must not change the answer.

    Random dense graphs, so this covers far more topologies than the hand-built
    fixtures above. If these two ever disagree, the DP and the observation are
    describing different chains.
    """
    torch.manual_seed(0)
    for m in (3, 6, 9):
        cap = torch.rand(64, m, m) * 40.0
        src = torch.rand(64, m) < 0.4
        cap_only = best_relay_capacity(cap, src, dst_index=m - 1, max_hops=m - 1)
        cap_path, _, _, _ = best_relay_path(cap, src, dst_index=m - 1, max_hops=m - 1)
        assert torch.allclose(cap_only, cap_path, atol=1e-5)


def test_path_membership_reproduces_the_reported_capacity():
    """Walk the returned chain by hand and recompute min(C_i)/min(n,3)."""
    # 0 -> 1 -> 3 is the only route to the MCV; 2 is a decoy with no link out.
    cap = _chain({(0, 1): 30.0, (1, 3): 12.0, (0, 2): 99.0}, m=4)
    src = torch.tensor([[True, False, False, False]])
    capacity, on_path, _, hops = best_relay_path(cap, src, dst_index=3, max_hops=3)

    assert hops.item() == 2
    assert on_path.tolist() == [[True, True, False, True]]
    assert capacity.item() == pytest.approx(min(30.0, 12.0) / 2, abs=1e-4)


def test_single_hop_path_is_source_plus_destination():
    cap = _chain({(0, 2): 20.0}, m=3)
    src = torch.tensor([[True, False, False]])
    capacity, on_path, _, hops = best_relay_path(cap, src, dst_index=2, max_hops=3)
    assert hops.item() == 1
    assert on_path.tolist() == [[True, False, True]]
    assert capacity.item() == pytest.approx(20.0, abs=1e-4)


def test_no_observer_means_no_path():
    """No source is a mission failure, not a routing failure -- and it must not
    leave a phantom chain in the observation."""
    cap = _chain({(0, 1): 30.0, (1, 2): 30.0})
    src = torch.zeros(1, 3, dtype=torch.bool)
    capacity, on_path, _, hops = best_relay_path(cap, src, dst_index=2, max_hops=2)
    assert capacity.item() == 0.0
    assert hops.item() == 0
    assert not on_path.any()


def test_disconnected_destination_yields_no_path():
    cap = _chain({(0, 1): 30.0}, m=3)  # nothing reaches node 2
    src = torch.tensor([[True, False, False]])
    capacity, on_path, _, hops = best_relay_path(cap, src, dst_index=2, max_hops=2)
    assert capacity.item() == 0.0
    assert hops.item() == 0
    assert not on_path.any()


def test_path_is_independent_across_the_batch():
    """Two environments with different answers must not contaminate each other --
    the back-walk indexes per-env, and a reduction bug here is invisible."""
    a = _chain({(0, 1): 30.0, (1, 3): 12.0}, m=4)
    b = _chain({(0, 3): 8.0}, m=4)
    cap = torch.cat([a, b], dim=0)
    src = torch.tensor([[True, False, False, False], [True, False, False, False]])
    capacity, on_path, _, hops = best_relay_path(cap, src, dst_index=3, max_hops=3)

    assert hops.tolist() == [2, 1]
    assert on_path.tolist() == [
        [True, True, False, True],
        [True, False, False, True],
    ]
    assert capacity[0].item() == pytest.approx(6.0, abs=1e-4)
    assert capacity[1].item() == pytest.approx(8.0, abs=1e-4)


def test_path_never_reports_more_hops_than_nodes_on_it():
    """A back-walk that loops would mark fewer nodes than it claims hops."""
    torch.manual_seed(1)
    m = 7
    cap = torch.rand(128, m, m) * 40.0
    src = torch.rand(128, m) < 0.3
    _, on_path, _, hops = best_relay_path(cap, src, dst_index=m - 1, max_hops=m - 1)
    # a chain of n hops touches n+1 distinct nodes, destination included
    assert torch.all(on_path.sum(dim=-1) == torch.where(hops > 0, hops + 1, hops))


def test_path_edges_are_the_hops_actually_used():
    """Node membership cannot say which pairs carried the feed; edges can.

    RQ1's failure attribution counts steps where the chain crosses an occluded
    link, which is an edge property -- so this is the output that metric reads.
    """
    cap = _chain({(0, 1): 30.0, (1, 3): 12.0, (0, 2): 99.0}, m=4)
    src = torch.tensor([[True, False, False, False]])
    _, on_path, on_edge, hops = best_relay_path(cap, src, dst_index=3, max_hops=3)

    assert hops.item() == 2
    assert on_edge[0].nonzero().tolist() == [[0, 1], [1, 3]]
    # every edge endpoint must be a node on the path, and there are `hops` edges
    assert on_edge.sum().item() == hops.item()
    ends = on_edge[0].any(0) | on_edge[0].any(1)
    assert torch.equal(ends, on_path[0])


def test_edge_count_equals_hop_count_on_random_graphs():
    torch.manual_seed(2)
    m = 7
    cap = torch.rand(128, m, m) * 40.0
    src = torch.rand(128, m) < 0.3
    _, on_path, on_edge, hops = best_relay_path(cap, src, dst_index=m - 1, max_hops=m - 1)
    assert torch.all(on_edge.sum(dim=(-2, -1)) == hops)
    # edges only ever connect nodes the walk marked
    ends = on_edge.any(-2) | on_edge.any(-1)
    assert torch.all(ends == on_path)
