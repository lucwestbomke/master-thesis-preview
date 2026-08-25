"""
Link-budget tests. Every expected value here is hand-computed from the closed
form, not captured from a previous run -- these are the numbers that back the
methodology chapter, so a regression must fail loudly rather than re-baseline.
"""

import math

import pytest
import torch

from .channel import (
    capacity_mbps,
    dbm_to_mw,
    fspl_db,
    mw_to_dbm,
    noise_floor_dbm,
    pairwise_distance_m,
    pathloss_a2a_db,
    pathloss_a2g_umi_av_db,
    received_power_dbm,
    sinr_db,
)

TOL = 1e-3


def test_dbm_mw_roundtrip():
    x = torch.tensor([-100.0, -30.0, 0.0, 30.0, 40.0])
    assert torch.allclose(mw_to_dbm(dbm_to_mw(x)), x, atol=1e-5)


def test_dbm_to_mw_known_points():
    # 0 dBm = 1 mW; 30 dBm = 1 W = 1000 mW; 40 dBm = 10 W
    got = dbm_to_mw(torch.tensor([0.0, 30.0, 40.0]))
    assert torch.allclose(got, torch.tensor([1.0, 1000.0, 10000.0]), rtol=1e-6)


def test_noise_floor_tracks_bandwidth():
    # -174 + 10log10(10e6) + 7 = -174 + 70 + 7 = -97
    assert noise_floor_dbm(10e6, 7.0) == pytest.approx(-97.0, abs=TOL)
    # -174 + 10log10(20e6) + 7 = -174 + 73.010 + 7 = -93.99
    assert noise_floor_dbm(20e6, 7.0) == pytest.approx(-93.990, abs=TOL)
    # The original spec hardcoded -100 dBm regardless of bandwidth. Confirm the
    # two genuinely differ, so the hardcode cannot creep back unnoticed.
    assert abs(noise_floor_dbm(10e6, 7.0) - (-100.0)) > 2.0


def test_fspl_known_point():
    # FSPL(100 m, 3.5 GHz) = 20log10(100) + 20log10(3.5) + 32.44
    #                      = 40 + 10.8814 + 32.44 = 83.3214 dB
    got = fspl_db(torch.tensor([100.0]), fc_ghz=3.5)
    assert got.item() == pytest.approx(83.3214, abs=1e-3)


def test_fspl_doubles_distance_adds_6db():
    d = torch.tensor([100.0, 200.0])
    pl = fspl_db(d, 3.5)
    assert (pl[1] - pl[0]).item() == pytest.approx(6.0206, abs=1e-3)


def test_a2a_blockage_penalty_is_additive():
    d = torch.tensor([300.0, 300.0])
    occ = torch.tensor([False, True])
    pl = pathloss_a2a_db(d, occ, fc_ghz=3.5, blockage_db=20.0)
    assert (pl[1] - pl[0]).item() == pytest.approx(20.0, abs=TOL)
    assert pl[0].item() == pytest.approx(fspl_db(d[:1], 3.5).item(), abs=TOL)


def test_umi_av_los_hand_computed():
    # 30.9 + (22.25 - 0.5*log10(100)) * log10(200) + 20*log10(3.5)
    #   = 30.9 + 21.25 * 2.301030 + 10.881361
    #   = 30.9 + 48.896887 + 10.881361 = 90.678248
    d = torch.tensor([200.0])
    h = torch.tensor([100.0])
    got = pathloss_a2g_umi_av_db(d, h, los=torch.tensor([True]), fc_ghz=3.5)
    assert got.item() == pytest.approx(90.6782, abs=1e-3)


def test_umi_av_nlos_hand_computed():
    # 32.4 + (43.2 - 7.6*2) * 2.301030 + 10.881361
    #   = 32.4 + 28.0 * 2.301030 + 10.881361 = 107.710201
    d = torch.tensor([200.0])
    h = torch.tensor([100.0])
    got = pathloss_a2g_umi_av_db(d, h, los=torch.tensor([False]), fc_ghz=3.5)
    assert got.item() == pytest.approx(107.7102, abs=1e-3)


def test_umi_av_nlos_never_below_los():
    d = torch.logspace(0, 3.5, 40).unsqueeze(0)
    h = torch.full_like(d, 80.0)
    los = pathloss_a2g_umi_av_db(d, h, torch.ones_like(d, dtype=torch.bool))
    nlos = pathloss_a2g_umi_av_db(d, h, torch.zeros_like(d, dtype=torch.bool))
    assert torch.all(nlos >= los - 1e-6)


def test_umi_av_los_is_near_free_space_at_altitude():
    # Sanity anchor: a clear air-to-ground ray should sit within a few dB of
    # free space. If a coefficient is mistyped this check breaks immediately.
    d = torch.tensor([50.0, 200.0, 800.0])
    h = torch.full_like(d, 100.0)
    los = pathloss_a2g_umi_av_db(d, h, torch.ones_like(d, dtype=torch.bool))
    excess = los - fspl_db(d, 3.5)
    assert torch.all(excess > -1.0) and torch.all(excess < 8.0)


# --------------------------------------------------------------------------- #
# SINR
# --------------------------------------------------------------------------- #


def _two_node_setup(prx_01_dbm: float, n0: float, jam: float = -300.0):
    """One transmitter (node 0) and one receiver (node 1)."""
    prx = torch.full((1, 2, 2), -300.0)
    prx[0, 0, 1] = prx_01_dbm
    jam_t = torch.full((1, 2), jam)
    tx = torch.tensor([[True, False]])
    return sinr_db(prx, jam_t, n0, tx)


def test_sinr_noise_limited_is_prx_minus_noise():
    # With no interference and no jammer, SINR reduces to Prx - N0.
    s = _two_node_setup(prx_01_dbm=-60.0, n0=-97.0)
    assert s[0, 0, 1].item() == pytest.approx(37.0, abs=1e-3)


def test_sinr_equal_interferer_gives_zero_db():
    # Two transmitters delivering identical power to receiver 2. Signal equals
    # interference, so SINR -> 0 dB once noise is negligible.
    prx = torch.full((1, 3, 3), -300.0)
    prx[0, 0, 2] = -60.0
    prx[0, 1, 2] = -60.0
    jam = torch.full((1, 3), -300.0)
    tx = torch.tensor([[True, True, False]])
    s = sinr_db(prx, jam, n0_dbm=-140.0, tx_mask=tx)
    assert s[0, 0, 2].item() == pytest.approx(0.0, abs=1e-2)


def test_silenced_interferer_stops_interfering():
    # Same geometry, but node 1 is not transmitting -> link 0->2 recovers fully.
    prx = torch.full((1, 3, 3), -300.0)
    prx[0, 0, 2] = -60.0
    prx[0, 1, 2] = -60.0
    jam = torch.full((1, 3), -300.0)
    s_loud = sinr_db(prx, jam, -140.0, torch.tensor([[True, True, False]]))
    s_quiet = sinr_db(prx, jam, -140.0, torch.tensor([[True, False, False]]))
    assert s_loud[0, 0, 2].item() == pytest.approx(0.0, abs=1e-2)
    assert s_quiet[0, 0, 2].item() == pytest.approx(80.0, abs=1e-2)


def test_node_does_not_self_interfere():
    # Receiver 2 is itself transmitting. Its own signal must not appear in its
    # interference term (half-duplex: it is not receiving its own emission).
    prx = torch.full((1, 3, 3), -300.0)
    prx[0, 0, 2] = -60.0
    prx[0, 2, 2] = 40.0  # would obliterate the link if counted
    jam = torch.full((1, 3), -300.0)
    tx = torch.tensor([[True, False, True]])
    s = sinr_db(prx, jam, -140.0, tx)
    assert s[0, 0, 2].item() == pytest.approx(80.0, abs=1e-2)


def test_jammer_degrades_sinr():
    clean = _two_node_setup(-60.0, n0=-97.0, jam=-300.0)
    jammed = _two_node_setup(-60.0, n0=-97.0, jam=-60.0)
    # Jammer at signal strength -> SINR collapses to ~0 dB.
    assert clean[0, 0, 1].item() == pytest.approx(37.0, abs=1e-3)
    assert jammed[0, 0, 1].item() == pytest.approx(0.0, abs=1e-2)


def test_regression_original_spec_sinr_formula_was_broken():
    """Pins the bug the project spec originally contained.

    The spec defined SINR_dB = P_signal - (P_jam + N0), adding two dBm
    quantities. In the linear domain that is a product, not a sum. For a
    realistic urban link it returns ~+100 dB -- physically impossible, and it
    would have silently removed the jammer from every experiment.
    """
    p_sig, p_jam, n0 = -61.6, -61.6, -100.0

    broken = p_sig - (p_jam + n0)
    assert broken > 90.0, "the historical formula really did produce absurd SINR"

    correct = _two_node_setup(p_sig, n0=n0, jam=p_jam)[0, 0, 1].item()
    assert correct == pytest.approx(0.0, abs=1e-2)
    assert broken - correct > 90.0


def test_tx_mask_carries_the_mac_assumption():
    """The scheduled and uncoordinated MACs must give different answers.

    Under the spatial-reuse TDMA schedule routing.py assumes, a chain of <=3
    hops never has two hops active at once, so only the slot's transmitter
    belongs in the mask and SINR reduces to signal over jammer-plus-noise.
    Passing every node instead -- while still dividing end-to-end rate by
    min(n, 3) -- double-counts the half-duplex cost. That combination made a
    feasible 3-hop chain look infeasible during scenario design, so pin it.
    """
    # Chain 0 -> 1 -> 2 -> 3. Node 2's emission also lands on node 1, so it
    # competes with hop 0->1 whenever the two are scheduled together.
    prx = torch.full((1, 4, 4), -300.0)
    prx[0, 0, 1] = prx[0, 1, 2] = prx[0, 2, 3] = -70.0
    prx[0, 2, 1] = -70.0
    jam = torch.full((1, 4), -300.0)

    scheduled = sinr_db(prx, jam, -97.0, torch.tensor([[True, False, False, False]]))
    uncoordinated = sinr_db(prx, jam, -97.0, torch.tensor([[True, False, True, False]]))

    assert scheduled[0, 0, 1].item() == pytest.approx(27.0, abs=1e-2)  # noise only
    assert uncoordinated[0, 0, 1].item() == pytest.approx(0.0, abs=1e-2)  # equal interferer
    assert (scheduled[0, 0, 1] - uncoordinated[0, 0, 1]).item() > 25.0


def test_scheduled_mac_reduces_to_jammer_plus_noise():
    """With one transmitter in the mask, SINR is signal over (jammer + noise)."""
    prx = torch.full((1, 3, 3), -300.0)
    prx[0, 0, 1] = -70.0
    jam = torch.full((1, 3), -300.0)
    jam[0, 1] = -100.0  # jammer equal to the noise floor at the receiver

    s = sinr_db(prx, jam, -100.0, torch.tensor([[True, False, False]]))
    # denominator doubles -> exactly 3 dB below the noise-only case
    assert s[0, 0, 1].item() == pytest.approx(30.0 - 3.0103, abs=1e-3)


# --------------------------------------------------------------------------- #
# Rate
# --------------------------------------------------------------------------- #


def test_capacity_hand_computed():
    # SINR_lin = 3 -> log2(4) = 2 b/s/Hz; x0.75 impl loss = 1.5; x10 MHz = 15 Mbps
    s = torch.tensor([10.0 * math.log10(3.0)])
    got = capacity_mbps(s, bandwidth_hz=10e6, impl_loss=0.75, se_cap=7.4)
    assert got.item() == pytest.approx(15.0, abs=1e-3)


def test_capacity_respects_modulation_cap():
    got = capacity_mbps(torch.tensor([80.0]), bandwidth_hz=10e6)
    assert got.item() == pytest.approx(74.0, abs=1e-6)  # 7.4 b/s/Hz x 10 MHz


def test_capacity_monotone_in_sinr():
    s = torch.linspace(-20.0, 30.0, 60)
    c = capacity_mbps(s, 10e6)
    assert torch.all(torch.diff(c) >= -1e-9)


def test_threshold_binds_on_a_three_hop_chain():
    """The bandwidth choice must make the 5 Mbps mission target contested.

    A 3-hop chain delivers min(C_i)/3, so each hop needs 15 Mbps, i.e.
    1.5 b/s/Hz after implementation loss -> log2(1+SINR) = 2 -> SINR = +4.77 dB.
    At the originally-specified 20 MHz the same target needed about -7 dB, which
    a swarm satisfies by accident and which makes the jammer decorative.
    """
    per_hop_needed = 5.0 * 3
    sinr_grid = torch.linspace(-20.0, 20.0, 4001)
    cap10 = capacity_mbps(sinr_grid, 10e6)
    required_10mhz = sinr_grid[cap10 >= per_hop_needed][0].item()
    assert required_10mhz == pytest.approx(4.77, abs=0.05)

    cap20 = capacity_mbps(sinr_grid, 20e6)
    required_20mhz = sinr_grid[cap20 >= 5.0][0].item()
    assert required_20mhz < -5.0  # single hop at 20 MHz: trivially satisfied


# --------------------------------------------------------------------------- #
# Shapes / device hygiene
# --------------------------------------------------------------------------- #


def test_pairwise_distance_shape_and_values():
    pos = torch.tensor([[[0.0, 0.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 12.0]]])
    d = pairwise_distance_m(pos)
    assert d.shape == (1, 3, 3)
    assert d[0, 0, 1].item() == pytest.approx(5.0, abs=1e-4)
    assert d[0, 0, 2].item() == pytest.approx(12.0, abs=1e-4)
    assert torch.allclose(d, d.transpose(1, 2), atol=1e-4)


def test_batched_pipeline_end_to_end():
    b, m = 8, 6
    pos = torch.rand(b, m, 3) * 400.0
    pos[..., 2] = 80.0
    d = pairwise_distance_m(pos)
    occ = torch.rand(b, m, m) > 0.7
    pl = pathloss_a2a_db(d, occ)
    ptx = torch.full((b, m), 30.0)
    prx = received_power_dbm(ptx, pl)
    assert prx.shape == (b, m, m)
    s = sinr_db(
        prx, torch.full((b, m), -90.0), noise_floor_dbm(10e6), torch.ones(b, m, dtype=torch.bool)
    )
    c = capacity_mbps(s, 10e6)
    assert c.shape == (b, m, m)
    assert torch.isfinite(c).all()
    assert torch.all(c >= 0.0)


def test_umi_av_los_has_the_free_space_floor_the_standard_specifies():
    """TR 36.777's UMi-AV LoS is `max(FSPL, ...)`, and the max must be real.

    At very short range the canyon expression drops below free space, which is
    unphysical -- no propagation beats vacuum. The floor was missing until the
    2026-08-26 verification pass; it is asserted here so it cannot quietly go
    again.
    """
    d = torch.tensor([2.0, 5.0])  # well inside the crossover
    h = torch.full_like(d, 60.0)
    los = pathloss_a2g_umi_av_db(d, h, torch.ones_like(d, dtype=torch.bool))
    assert torch.all(los >= fspl_db(d, 3.5) - 1e-6), "LoS dipped below free space"


def test_the_free_space_floor_is_vacuous_above_the_altitude_floor():
    """⚠️ Pins *why* adding the floor moved no measured number.

    The floor binds only below ~9.5-18.7 m of 3-D separation, and `ALT_MIN_M`
    puts every drone above that height, so a drone<->MCV ray is at least
    `ALT_MIN_M` long by construction. If the altitude band is ever lowered this
    test fails, which is the signal that the golden trace legitimately moves.
    """
    from .core import ALT_MAX_M, ALT_MIN_M

    d = torch.linspace(ALT_MIN_M, 1500.0, 200)
    for h in (ALT_MIN_M, ALT_MAX_M):
        hh = torch.full_like(d, float(h))
        los = pathloss_a2g_umi_av_db(d, hh, torch.ones_like(d, dtype=torch.bool))
        canyon = 30.9 + (22.25 - 0.5 * math.log10(h)) * torch.log10(d) + 20.0 * math.log10(3.5)
        assert torch.allclose(los, canyon, atol=1e-4), (
            f"the FSPL floor binds at h={h} m -- the altitude band no longer "
            "guarantees it is vacuous, and every path-loss number must be re-derived"
        )


def test_a2a_and_a2g_are_genuinely_different_models():
    """⛔ TR 36.777 is air-to-ground ONLY.

    Applying a street-canyon model to a drone<->drone ray that never enters the
    canyon would be a category error, so the two paths must not converge. A
    clear A2A link is free space; a clear A2G link carries the canyon excess.
    """
    d = torch.tensor([500.0])
    clear = torch.tensor([False])
    a2a = pathloss_a2a_db(d, clear, fc_ghz=3.5)
    a2g = pathloss_a2g_umi_av_db(d, torch.tensor([60.0]), ~clear, fc_ghz=3.5)
    assert a2a.item() == pytest.approx(fspl_db(d, 3.5).item(), abs=1e-6)
    assert a2g.item() > a2a.item() + 1.0, "A2G must not collapse onto free space"
