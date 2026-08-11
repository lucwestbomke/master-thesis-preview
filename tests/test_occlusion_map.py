"""Cross-module: the occlusion kernel against the real Frankfurt map.

`src/env/test_occlusion.py` proves the kernel matches a shapely reference on
random geometry. This file proves the *pair* -- kernel plus baked artefact --
reproduces a property of Frankfurt that was measured independently, offline,
by a completely different code path (`scripts/measure_sightlines.py`, which
ray-casts against the source LoD2 polygons via shapely).

If the two disagree, either the artefact or the kernel is wrong, and neither
unit-test suite would catch it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.env.occlusion import is_occluded, segment_clearance

ARTEFACT = Path(__file__).resolve().parent.parent / "data" / "frankfurt_box.npz"

RAY_HEIGHT_M = 2.0  # vehicle height, matching measure_sightlines.py
MAX_RAY_M = 1500.0

# Measured offline against the source polygons (docs/BLOCK_B.md)
REFERENCE_MEDIAN_M = 127.0
REFERENCE_P90_M = 387.0


@pytest.fixture(scope="module")
def art():
    if not ARTEFACT.exists():
        pytest.skip(f"{ARTEFACT} not built; run scripts/prep_osm.py")
    return np.load(ARTEFACT)


def _sightline_lengths(origins, dirs, boxes, heights, iters=22):
    """Distance to the first blocker along each ray, by bisection on clearance.

    Uses the production kernel as the oracle: grow/shrink the segment until the
    longest unoccluded length is found. `iters` of bisection over [0, 1500] m
    resolves to well under a metre.
    """
    n = len(origins)
    lo = torch.zeros(n, dtype=torch.float64)
    hi = torch.full((n,), MAX_RAY_M, dtype=torch.float64)
    z = torch.full((n, 1), RAY_HEIGHT_M, dtype=torch.float64)
    p0 = torch.cat([origins, z], dim=1)

    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        p1 = torch.cat([origins + dirs * mid[:, None], z], dim=1)
        blocked = is_occluded(segment_clearance(p0, p1, boxes, heights))
        hi = torch.where(blocked, mid, hi)
        lo = torch.where(blocked, lo, mid)
    return lo


@pytest.fixture(scope="module")
def sightlines(art):
    rng = np.random.default_rng(0)
    nodes = art["road_nodes"].astype(np.float64)
    edges = art["road_edges"]

    # sample edges, take the street axis from the edge itself
    pick = rng.choice(len(edges), size=min(700, len(edges)), replace=False)
    a = nodes[edges[pick, 0]]
    b = nodes[edges[pick, 1]]
    seg = b - a
    length = np.hypot(seg[:, 0], seg[:, 1])
    keep = length > 1.0
    a, seg, length = a[keep], seg[keep], length[keep]
    axis = seg / length[:, None]

    # midpoint of each edge, looking both ways
    mid = a + 0.5 * seg
    origins = np.concatenate([mid, mid])
    dirs = np.concatenate([axis, -axis])

    boxes = torch.from_numpy(art["building_boxes"]).double()
    heights = torch.from_numpy(art["building_heights"]).double()
    out = _sightline_lengths(torch.from_numpy(origins), torch.from_numpy(dirs), boxes, heights)
    los = out.numpy()

    # A ray that leaves the box stops being a measurement -- there are no
    # buildings beyond the artefact's edge. Same right-censoring treatment as
    # scripts/measure_sightlines.py, without which the two are not comparable.
    edge = _dist_to_box_edge(origins, dirs)
    censored = los >= np.minimum(edge, MAX_RAY_M) - 1.0
    return los, censored


def _dist_to_box_edge(origins, dirs, half=750.0):
    """Ray parameter at which each ray exits the 1500 m box (2D slab method)."""
    t = np.full(len(origins), MAX_RAY_M)
    for ax in (0, 1):
        o, d = origins[:, ax], dirs[:, ax]
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (-half - o) / d
            t2 = (half - o) / d
        hit = np.maximum(t1, t2)
        hit = np.where(np.abs(d) < 1e-12, MAX_RAY_M, hit)
        t = np.minimum(t, hit)
    return np.maximum(t, 0.0)


def test_sightlines_reproduce_the_offline_measurement(sightlines):
    """Median sightline must match the independently measured 127 m."""
    los, censored = sightlines
    measured = los[~censored]
    assert len(measured) > 200, "too few uncensored rays to compare"

    median = float(np.median(measured))
    assert median == pytest.approx(REFERENCE_MEDIAN_M, rel=0.45), (
        f"kernel median sightline {median:.0f} m vs {REFERENCE_MEDIAN_M:.0f} m measured offline"
    )


def test_sightlines_are_short_enough_that_the_sensor_never_binds(sightlines):
    """RQ1 rests on occlusion, not on sensor range (docs/BLOCK_B.md).

    Offline, 0.2 % of uncensored rays exceeded 830 m. Censored rays must be
    excluded here too or the comparison is against a different quantity.
    """
    los, censored = sightlines
    assert np.mean(los[~censored] > 830.0) < 0.05


def test_censoring_rate_is_plausible(sightlines):
    """Offline the same treatment censored 16 % of rays."""
    _, censored = sightlines
    assert 0.02 < censored.mean() < 0.45


def test_a_high_drone_sees_much_further_than_a_ground_ray(art):
    """Sanity: flying above the fabric must buy line of sight."""
    boxes = torch.from_numpy(art["building_boxes"]).double()
    heights = torch.from_numpy(art["building_heights"]).double()
    rng = np.random.default_rng(1)
    xy = torch.from_numpy(rng.uniform(-600, 600, (400, 2)))

    far = xy + torch.from_numpy(rng.normal(0, 250, (400, 2)))
    low0 = torch.cat([xy, torch.full((400, 1), 2.0, dtype=torch.float64)], dim=1)
    low1 = torch.cat([far, torch.full((400, 1), 2.0, dtype=torch.float64)], dim=1)
    high0 = torch.cat([xy, torch.full((400, 1), 80.0, dtype=torch.float64)], dim=1)
    high1 = torch.cat([far, torch.full((400, 1), 80.0, dtype=torch.float64)], dim=1)

    low_ok = (~is_occluded(segment_clearance(low0, low1, boxes, heights))).double().mean()
    high_ok = (~is_occluded(segment_clearance(high0, high1, boxes, heights))).double().mean()
    assert high_ok > low_ok + 0.2, f"80 m: {high_ok:.2f} clear vs ground {low_ok:.2f}"
