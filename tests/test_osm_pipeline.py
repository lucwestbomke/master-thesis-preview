"""Block B: the baked Frankfurt artefact must be loadable and sane.

Every assertion here is about `data/frankfurt_box.npz` as consumed by the env,
so the test deliberately imports **only NumPy**. If this file ever needs
`osmnx` or `shapely`, the offline/runtime split has been broken.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ARTEFACT = Path(__file__).resolve().parent.parent / "data" / "frankfurt_box.npz"

BOX_SIZE_M = 1500.0
HALF = BOX_SIZE_M / 2.0
EPISODE_STEPS = 600
DT_S = 0.4

# docs/ENVIRONMENT.md escalation table: step -> HVT-to-MCV separation, metres
ESCALATION = {0: 400.0, 150: 700.0, 300: 1000.0, 599: 1400.0}
ESCALATION_TOL_M = 150.0


@pytest.fixture(scope="module")
def art():
    if not ARTEFACT.exists():
        pytest.skip(f"{ARTEFACT} not built; run scripts/prep_osm.py")
    return np.load(ARTEFACT)


# --------------------------------------------------------------------------
# artefact shape and provenance
# --------------------------------------------------------------------------


def test_all_fields_present(art):
    expected = {
        "building_boxes",
        "building_heights",
        "height_grid",
        "road_nodes",
        "road_edges",
        "road_speeds",
        "road_route_ok",
        "route_mcv",
        "route_xy",
        "origin_lonlat",
        "box_size_m",
    }
    assert expected <= set(art.files)


def test_no_nans_anywhere(art):
    for key in art.files:
        a = art[key]
        if a.dtype.kind == "f":
            assert np.isfinite(a).all(), f"{key} holds NaN or inf"


def test_provenance_is_the_frozen_box(art):
    lon, lat = art["origin_lonlat"]
    assert lat == pytest.approx(50.11200, abs=1e-5)
    assert lon == pytest.approx(8.67040, abs=1e-5)
    assert float(art["box_size_m"]) == pytest.approx(BOX_SIZE_M)


# --------------------------------------------------------------------------
# buildings
# --------------------------------------------------------------------------


def test_building_boxes_are_inside_the_box(art):
    b = art["building_boxes"]
    assert b.ndim == 2 and b.shape[1] == 6
    assert np.abs(b[:, 0]).max() <= HALF + 1e-3
    assert np.abs(b[:, 1]).max() <= HALF + 1e-3


def test_building_extents_are_positive(art):
    b = art["building_boxes"]
    assert (b[:, 2] > 0).all(), "half_w must be positive"
    assert (b[:, 3] > 0).all(), "half_h must be positive"


def test_orientation_is_a_unit_vector(art):
    """cos/sin are stored precomputed; if they drift the slab test is wrong."""
    b = art["building_boxes"]
    norm = np.hypot(b[:, 4], b[:, 5])
    assert np.abs(norm - 1.0).max() < 1e-5


def test_heights_are_positive_and_plausible(art):
    h = art["building_heights"]
    assert len(h) == len(art["building_boxes"])
    assert (h > 0).all()
    # Commerzbank Tower is the tallest thing in Frankfurt at ~260 m
    assert h.max() < 300.0
    assert h.max() > 150.0, "the tower cluster must be in the box"


def test_both_height_regimes_present(art):
    """Frankfurt was chosen for heterogeneity -- see docs/DECISIONS.md."""
    h = art["building_heights"]
    assert (h < 40.0).mean() > 0.5, "low fabric missing"
    assert (h >= 100.0).sum() >= 10, "tower cluster missing"


def test_buildings_do_not_fill_the_box(art):
    """The AABB failure mode: boxes so inflated the city becomes solid."""
    b = art["building_boxes"]
    area = float(np.sum(4.0 * b[:, 2] * b[:, 3]))
    fill = area / BOX_SIZE_M**2
    assert fill < 0.70, f"boxes cover {fill:.0%} of the box; occlusion will not discriminate"


def test_height_grid(art):
    g = art["height_grid"]
    assert g.shape == (75, 75)
    assert (g >= 0).all()
    assert g.max() == pytest.approx(art["building_heights"].max(), rel=1e-3)


# --------------------------------------------------------------------------
# road graph
# --------------------------------------------------------------------------


def test_road_nodes_inside_box(art):
    n = art["road_nodes"]
    assert n.ndim == 2 and n.shape[1] == 2
    assert np.abs(n).max() <= HALF + 1e-3


def test_road_edges_index_valid_nodes(art):
    e, n = art["road_edges"], art["road_nodes"]
    assert e.ndim == 2 and e.shape[1] == 2
    assert e.min() >= 0 and e.max() < len(n)
    assert (e[:, 0] != e[:, 1]).all(), "self-loop in the road graph"


def test_road_graph_is_connected(art):
    """A route sampled in an island could never reach the rest of the map."""
    e, n = art["road_edges"], art["road_nodes"]
    adj: dict[int, list[int]] = {i: [] for i in range(len(n))}
    for u, v in e:
        adj[int(u)].append(int(v))
        adj[int(v)].append(int(u))
    seen = {0}
    stack = [0]
    while stack:
        for nb in adj[stack.pop()]:
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)
    assert len(seen) == len(n), f"{len(n) - len(seen)} road nodes unreachable"


def test_road_speeds_respect_the_cap(art):
    s = art["road_speeds"]
    assert (s > 0).all()
    # 13.9 m/s = 50 km/h, above which the drone loses its speed margin
    assert s.max() <= 13.9 + 1e-6


# --------------------------------------------------------------------------
# route bank -- the five conditions in docs/BLOCK_B.md
# --------------------------------------------------------------------------


def test_route_shapes(art):
    mcv, xy = art["route_mcv"], art["route_xy"]
    assert xy.ndim == 3 and xy.shape[1] == EPISODE_STEPS and xy.shape[2] == 2
    assert mcv.shape == (xy.shape[0], 2)
    assert len(xy) > 100, "route bank too small to avoid memorisation"


def test_routes_start_300_to_500_m_from_mcv(art):
    d0 = np.linalg.norm(art["route_xy"][:, 0] - art["route_mcv"], axis=1)
    assert d0.min() >= 300.0 - 1.0
    assert d0.max() <= 500.0 + 1.0


def test_routes_move_outward(art):
    mcv, xy = art["route_mcv"], art["route_xy"]
    d = np.linalg.norm(xy - mcv[:, None, :], axis=2)
    gain = d[:, -1] - d[:, 0]
    assert gain.min() >= 500.0, "a route failed to escalate away from the MCV"
    # and it should be a trend, not a late dash
    away = (np.diff(d, axis=1) > 0).mean(axis=1)
    assert np.median(away) > 0.6


def test_routes_stay_inside_the_box(art):
    assert np.abs(art["route_xy"]).max() <= HALF + 1.0


def test_routes_last_the_whole_episode(art):
    """The HVT must keep moving; a stalled tail is a trivially easy episode."""
    xy = art["route_xy"]
    step = np.linalg.norm(np.diff(xy, axis=1), axis=2)
    # no route may sit still for the final quarter of the episode
    assert step[:, -EPISODE_STEPS // 4 :].sum(axis=1).min() > 50.0


def test_hvt_never_outruns_the_drone(art):
    """Drone cruise is 20 m/s; the HVT needs to stay comfortably below it."""
    step = np.linalg.norm(np.diff(art["route_xy"], axis=1), axis=2)
    speed = step / DT_S
    assert speed.max() <= 13.9 + 0.5


def test_route_separation_tracks_the_escalation_table(art):
    """The premise is a chain escalating 1 -> 2 -> 3 hops (docs/ENVIRONMENT.md)."""
    mcv, xy = art["route_mcv"], art["route_xy"]
    d = np.linalg.norm(xy - mcv[:, None, :], axis=2)
    for step, want in ESCALATION.items():
        got = float(np.median(d[:, step]))
        assert abs(got - want) < ESCALATION_TOL_M, (
            f"step {step}: median separation {got:.0f} m, table says {want:.0f} m"
        )


def test_route_bank_is_diverse(art):
    """Randomised MCV and route, so the policy cannot memorise one layout."""
    mcv, xy = art["route_mcv"], art["route_xy"]
    assert len(np.unique(mcv, axis=0)) > 50
    flat = xy.reshape(len(xy), -1)
    assert len(np.unique(flat, axis=0)) > 0.5 * len(xy)


def _inside_any_box(pts, boxes, chunk=600):
    """Is each 2D point inside any oriented footprint? (NumPy only, on purpose.)"""
    cx, cy, hw, hh, ca, sa = boxes.astype(np.float64).T
    out = np.zeros(len(pts), dtype=bool)
    for i in range(0, len(boxes), chunk):
        s = slice(i, i + chunk)
        dx = pts[:, None, 0] - cx[s]
        dy = pts[:, None, 1] - cy[s]
        lx = dx * ca[s] + dy * sa[s]
        ly = -dx * sa[s] + dy * ca[s]
        out |= ((np.abs(lx) <= hw[s]) & (np.abs(ly) <= hh[s])).any(axis=1)
    return out


def test_mcv_never_spawns_inside_a_building(art):
    """The MCV never moves, so a spawn inside a footprint kills every link."""
    mcv = np.unique(art["route_mcv"], axis=0).astype(np.float64)
    assert _inside_any_box(mcv, art["building_boxes"]).sum() == 0


def test_hvt_rarely_passes_through_a_building(art):
    """A point inside an obstacle is blocked by construction.

    Some overlap is legitimate -- roads really do run under podiums and through
    arcades, and brief unobservability is the handoff pressure RQ3 studies. But
    it was 6.3 % before bridge decks were dropped and footprints split, with one
    route spending 333 of 600 steps inside a single box. See docs/BLOCK_C.md.
    """
    xy = art["route_xy"].astype(np.float64)
    sub = xy[::8]
    ins = _inside_any_box(sub.reshape(-1, 2), art["building_boxes"])
    assert ins.mean() < 0.03, f"{ins.mean():.1%} of route points are inside a building"


def test_no_route_spends_long_inside_a_building(art):
    xy = art["route_xy"].astype(np.float64)
    sub = xy[::8]
    per = _inside_any_box(sub.reshape(-1, 2), art["building_boxes"]).reshape(len(sub), -1)
    worst = per.mean(axis=1).max()
    assert worst <= 0.06, f"worst route is inside a building for {worst:.0%} of the episode"


def test_boxes_do_not_swallow_the_road_network(art):
    frac = _inside_any_box(art["road_nodes"].astype(np.float64), art["building_boxes"]).mean()
    assert frac < 0.03, f"{frac:.1%} of road nodes are inside a building box"


def test_routes_use_more_than_the_arterials(art):
    """Shortest-time routing once put every HVT on the 13.9 m/s roads only."""
    speed = np.linalg.norm(np.diff(art["route_xy"], axis=1), axis=2) / DT_S
    fast = 13.9 * 0.70
    assert (speed < fast - 0.5).mean() > 0.05, "routes never leave the fast roads"
