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
        # A 300 m area is trivial at any transmit power worth having. This is the
        # concrete reason the Ptx ceiling cannot simply be raised to make the
        # telecom energy term measurable -- it destroys the mission instead.
        (300, 20, TRIVIAL),
        (300, 30, TRIVIAL),
        (300, 40, TRIVIAL),
        # 40 dBm (10 W) is trivial out to 2 km. It is not a usable ceiling for
        # any operating area this thesis can plausibly simulate.
        (600, 40, TRIVIAL),
        (1200, 40, TRIVIAL),
        (2000, 40, TRIVIAL),
        # The contested band -- these are the viable design points.
        (600, 20, CONTESTED),
        (1200, 30, CONTESTED),
        (2000, 30, CONTESTED),
        # Too little power for the area: even the chain cannot close.
        (600, 10, INFEASIBLE),
        (1200, 20, INFEASIBLE),
        (2000, 10, INFEASIBLE),
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
