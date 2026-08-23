"""
Pins the scenario design space.

The relay chain has to be *geometrically necessary*: if one drone can observe the
HVT and still reach the MCV across the whole operating area, there is no
multi-hop problem and the thesis has no subject. That is a link-budget question,
so it is asserted here rather than assumed.

These tests encode the trade-off table produced by
`scripts/link_budget_check.py`. They are not a claim that any one design point is
correct -- they document which regions of (map size, Ptx ceiling) are trivial,
contested, or infeasible, so a later change to the channel model, bandwidth or
carrier cannot silently move the chosen operating point into the trivial region.

If you change fc, bandwidth, the rate target or the blockage penalty, re-run the
script and update this table deliberately.

⚠️ **Re-pinned in Block E when the rate target moved 5 -> 15 Mbps.** Read the
verdicts below with the model's conservatism in mind, because it is now large:

`classify` asks whether a chain of THREE hops, EVERY ONE OF THEM BLOCKED, clears
`threshold * min(hops, 3)` per hop -- 45 Mbps per hop at a 15 Mbps target. That
was a reasonable screen at 15 Mbps per hop; at 45 it is a genuine worst case that
almost never occurs. A2A links are blocked ~31 % of the time at 80 m, so all
three hops blocked happens on ~3 % of chains, and a clear 500 m hop carries
~74 Mbps against a blocked one's ~25.

**Consequence to know about: the analytic screen now calls the CHOSEN operating
point (1500 m box, 30 dBm) INFEASIBLE, and the measured environment does not.**
B0 reaches 57.2 % mission-capable there (docs/BLOCK_E.md). The measurement is the
authority; this file is a screen for silent drift into the TRIVIAL region, which
is the boundary it can still see clearly. Do not quote an INFEASIBLE verdict from
this table as evidence about the real environment.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from link_budget_check import classify, max_range_m

TRIVIAL = "TRIVIAL"
CONTESTED = "CONTESTED"
INFEASIBLE = "INFEASIBLE"


@pytest.mark.parametrize(
    ("map_size_m", "ptx_dbm", "expected"),
    [
        # A 300 m area is trivial at 30 dBm and above. This is the concrete
        # reason the Ptx ceiling cannot simply be raised to make the telecom
        # energy term measurable -- it destroys the mission instead.
        (300, 30, TRIVIAL),
        (300, 40, TRIVIAL),
        # 40 dBm (10 W) is trivial out to 1200 m even at the 15 Mbps target. It
        # is not a usable ceiling for any operating area this thesis can
        # plausibly simulate -- and a *blocked* A2A link at 40 dBm still reaches
        # 2789 m at the mission rate, which is the sharper form of the argument
        # (see test_ptx_ceiling_of_40dbm_is_unusable).
        (600, 40, TRIVIAL),
        (1200, 40, TRIVIAL),
        # Too little power for the area, under the model's all-hops-blocked
        # worst case. At the 5 Mbps target (600, 20), (1200, 30), (2000, 30) and
        # (2000, 40) read CONTESTED/TRIVIAL; at 15 Mbps the per-hop requirement
        # triples to 45 Mbps and the whole middle band collapses. That is the
        # screen becoming conservative, NOT the environment becoming infeasible
        # -- see the module docstring.
        (300, 20, INFEASIBLE),
        (600, 20, INFEASIBLE),
        (600, 10, INFEASIBLE),
        (1200, 20, INFEASIBLE),
        (1200, 30, INFEASIBLE),
        (2000, 10, INFEASIBLE),
        (2000, 30, INFEASIBLE),
        (2000, 40, INFEASIBLE),
    ],
)
def test_scenario_classification(map_size_m, ptx_dbm, expected):
    verdict, _single, _relayed = classify(map_size_m, ptx_dbm)
    assert verdict.startswith(expected), f"{map_size_m} m @ {ptx_dbm} dBm -> {verdict}"


def test_ptx_ceiling_of_40dbm_is_unusable():
    """10 W reaches far past any simulable urban operating area.

    Sanity anchor for the correction that produced this file: a *blocked*
    air-to-air link at 40 dBm still carries 15 Mbps over ~2.8 km, and 5 Mbps
    over ~6.3 km. A clear air-to-ground link reaches tens of kilometres.
    """
    assert max_range_m(40, 15, "a2a_blocked") > 2500.0
    assert max_range_m(40, 5, "a2a_blocked") > 6000.0
    assert max_range_m(40, 5, "a2g_los") > 20000.0


def test_thirty_dbm_keeps_the_chain_bounded():
    """1 W leaves the chain hop-limited at kilometre scale, which is the point."""
    assert 300.0 < max_range_m(30, 15, "a2g_nlos") < 1000.0
    assert max_range_m(30, 5, "a2g_nlos") < 1500.0


def test_range_is_monotone_in_power_and_target():
    for kind in ("a2g_nlos", "a2g_los", "a2a_los", "a2a_blocked"):
        assert max_range_m(30, 5, kind) > max_range_m(20, 5, kind)
        assert max_range_m(30, 5, kind) > max_range_m(30, 15, kind)


def test_the_chosen_operating_point_is_not_trivial():
    """The one thing this file exists to protect.

    If a single drone can span the 1500 m Frankfurt box and still deliver the
    mission rate, the relay chain is unnecessary and the thesis has no subject.
    Asserted on the single-hop A2G capacity, which is the quantity that would
    make it true -- and separately on the measured environment, where a solo
    drone hovering directly over the HVT at 1336 m is mission-capable 0.4 % of
    the time (`measure_envelope.py --only solo`).
    """
    verdict, single, _relayed = classify(1500, 30)
    assert not verdict.startswith(TRIVIAL), verdict
    assert single < 15.0, f"a single 1500 m A2G hop carries {single:.1f} Mbps"


def test_thirty_dbm_leaves_blocked_links_hop_limited():
    """At the 15 Mbps mission rate, 30 dBm reaches ~900 m A2A and ~560 m A2G
    through blockage -- under the 1336 m the episode opens up, which is what
    forces the chain to grow. This is the positive statement that the
    over-conservative INFEASIBLE verdicts above cannot make."""
    assert 700.0 < max_range_m(30, 15, "a2a_blocked") < 1100.0
    assert 400.0 < max_range_m(30, 15, "a2g_nlos") < 700.0
