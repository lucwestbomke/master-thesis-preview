"""Occlusion correctness: hand-computed cases plus a slow shapely reference.

The random-geometry agreement test is the real one -- it is what catches sign
errors, frame-rotation mistakes and slab edge cases that hand-written examples
never reach. The explicit cases pin the situations a random sampler produces too
rarely to be trusted: grazing a corner, running parallel to a face, passing
exactly at roof height.

`shapely` appears here and nowhere under `src/` outside this file. It is an
offline tool (AGENTS.md); importing it in the module under test would break the
offline/runtime split.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from .occlusion import (
    FREE_CLEARANCE_M,
    is_occluded,
    pairwise_clearance,
    segment_clearance,
)


def make_box(cx, cy, w, h, theta_deg):
    t = math.radians(theta_deg)
    return [cx, cy, w / 2.0, h / 2.0, math.cos(t), math.sin(t)]


def T(x):
    return torch.tensor(x, dtype=torch.float64)


# --------------------------------------------------------------------------
# slow reference
# --------------------------------------------------------------------------


def reference_clearance(p0, p1, boxes, heights, ignore_endpoint_boxes=True):
    """Obviously-correct clearance for one segment, via shapely 2D geometry."""
    from shapely.affinity import rotate, translate
    from shapely.geometry import LineString, Point
    from shapely.geometry import box as sbox

    best = FREE_CLEARANCE_M
    for (cx, cy, hw, hh, ca, sa), H in zip(boxes, heights, strict=True):
        theta = math.degrees(math.atan2(sa, ca))
        rect = sbox(-hw, -hh, hw, hh)
        rect = rotate(rect, theta, origin=(0, 0), use_radians=False)
        rect = translate(rect, cx, cy)

        if ignore_endpoint_boxes:
            in0 = rect.covers(Point(p0[0], p0[1])) and p0[2] <= H
            in1 = rect.covers(Point(p1[0], p1[1])) and p1[2] <= H
            if in0 or in1:
                continue

        seg = LineString([(p0[0], p0[1]), (p1[0], p1[1])])
        inter = seg.intersection(rect)
        if inter.is_empty:
            continue

        # parameter range of the intersection along the segment
        total = seg.length
        if total < 1e-12:
            ts = [0.0, 0.0]
        else:
            pts = []
            geoms = inter.geoms if hasattr(inter, "geoms") else [inter]
            for g in geoms:
                pts.extend(list(g.coords))
            if not pts:
                continue
            ts = [seg.project(Point(q)) / total for q in pts]

        t_lo, t_hi = max(0.0, min(ts)), min(1.0, max(ts))
        if t_lo > t_hi:
            continue
        z_lo = p0[2] + t_lo * (p1[2] - p0[2])
        z_hi = p0[2] + t_hi * (p1[2] - p0[2])
        best = min(best, min(z_lo, z_hi) - H)
    return best


# --------------------------------------------------------------------------
# hand-computed cases
# --------------------------------------------------------------------------


def test_ray_well_above_the_roof_is_clear():
    boxes = T([make_box(0, 0, 40, 40, 0)])
    h = T([20.0])
    c = segment_clearance(T([-100.0, 0, 80]), T([100.0, 0, 80]), boxes, h)
    assert c.item() == pytest.approx(60.0)  # 80 m ray over a 20 m roof


def test_ray_through_the_middle_is_blocked():
    boxes = T([make_box(0, 0, 40, 40, 0)])
    h = T([20.0])
    c = segment_clearance(T([-100.0, 0, 5]), T([100.0, 0, 5]), boxes, h)
    assert c.item() == pytest.approx(-15.0)
    assert bool(is_occluded(c))


def test_ray_missing_the_footprint_is_free():
    boxes = T([make_box(0, 0, 40, 40, 0)])
    h = T([20.0])
    c = segment_clearance(T([-100.0, 500.0, 5]), T([100.0, 500.0, 5]), boxes, h)
    assert c.item() == pytest.approx(FREE_CLEARANCE_M)


def test_rotation_is_actually_applied():
    """A 45-degree box is missed by a ray that a same-size AABB would catch."""
    boxes = T([make_box(0, 0, 40, 10, 45)])
    h = T([20.0])
    # travels along y at x = +17: inside the axis-aligned hull, outside the
    # rotated rectangle
    c = segment_clearance(T([17.0, -100.0, 5]), T([17.0, 100.0, 5]), boxes, h)
    ref = reference_clearance([17.0, -100.0, 5], [17.0, 100.0, 5], boxes.tolist(), h.tolist())
    assert c.item() == pytest.approx(ref, abs=1e-6)


def test_descending_ray_uses_the_minimum_altitude_over_the_crossing():
    """2.5D: the altitude must be checked across the intersection interval."""
    boxes = T([make_box(0, 0, 40, 40, 0)])
    h = T([20.0])
    # from 100 m down to 0 m across x in [-100, 100]; over the box it spans
    # x in [-20, 20], i.e. t in [0.4, 0.6], so z runs 60 -> 40. min is 40.
    c = segment_clearance(T([-100.0, 0, 100.0]), T([100.0, 0, 0.0]), boxes, h)
    assert c.item() == pytest.approx(20.0)  # 40 - 20


def test_grazing_the_roof_is_near_zero():
    boxes = T([make_box(0, 0, 40, 40, 0)])
    h = T([20.0])
    c = segment_clearance(T([-100.0, 0, 20.0]), T([100.0, 0, 20.0]), boxes, h)
    assert abs(c.item()) < 1e-6


def test_parallel_to_a_face_does_not_divide_by_zero():
    boxes = T([make_box(0, 0, 40, 40, 0)])
    h = T([20.0])
    # exactly along the face x = 20
    c = segment_clearance(T([20.0, -100.0, 5.0]), T([20.0, 100.0, 5.0]), boxes, h)
    assert torch.isfinite(c).all()


def test_zero_length_segment_is_finite():
    boxes = T([make_box(0, 0, 40, 40, 0)])
    h = T([20.0])
    c = segment_clearance(T([500.0, 500.0, 5.0]), T([500.0, 500.0, 5.0]), boxes, h)
    assert torch.isfinite(c).all()


def test_vertical_segment_is_finite():
    boxes = T([make_box(0, 0, 40, 40, 0)])
    h = T([20.0])
    c = segment_clearance(T([0.0, 0.0, 100.0]), T([0.0, 0.0, 1.0]), boxes, h)
    assert torch.isfinite(c).all()


def test_endpoint_inside_a_box_is_ignored_by_default():
    """A node standing in an over-approximated footprint must not self-blind."""
    boxes = T([make_box(0, 0, 40, 40, 0)])
    h = T([20.0])
    p0, p1 = T([0.0, 0.0, 2.0]), T([200.0, 0.0, 80.0])
    assert segment_clearance(p0, p1, boxes, h).item() == pytest.approx(FREE_CLEARANCE_M)
    blocked = segment_clearance(p0, p1, boxes, h, ignore_endpoint_boxes=False)
    assert blocked.item() < 0.0


def test_a_drone_above_a_box_is_not_inside_it():
    """Containment is 3D -- 80 m over a 22 m block is above it, not inside it.

    The drone starts directly over the footprint, so a 2D containment test would
    wave this box through as an "endpoint box" and report clear. In 3D it is not
    inside, and the steep descent really does clip the roof:
    the ray leaves the footprint at x = 20, i.e. t = 0.8, where z = 16 m, so the
    clearance is 16 - 22 = -6 m.
    """
    boxes = T([make_box(0, 0, 40, 40, 0)])
    h = T([22.0])
    p0, p1 = T([0.0, 0.0, 80.0]), T([25.0, 0.0, 0.0])
    c = segment_clearance(p0, p1, boxes, h)
    assert c.item() == pytest.approx(-6.0)
    ref = reference_clearance(p0.tolist(), p1.tolist(), boxes.tolist(), h.tolist())
    assert c.item() == pytest.approx(ref, abs=1e-6)


def test_a_shallow_ray_over_a_box_stays_clear():
    """The counterpart: same start, far shallow endpoint, roof never reached."""
    boxes = T([make_box(0, 0, 40, 40, 0)])
    h = T([22.0])
    # leaves the footprint at x = 20 (t = 1/15) while still at ~74.8 m
    c = segment_clearance(T([0.0, 0.0, 80.0]), T([300.0, 0.0, 1.5]), boxes, h)
    assert c.item() == pytest.approx(74.7667 - 22.0, abs=1e-3)


# --------------------------------------------------------------------------
# the real test: random geometry against the reference
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_matches_shapely_reference_on_random_geometry(seed):
    rng = np.random.default_rng(seed)
    n_boxes, n_seg = 40, 60

    boxes = np.stack(
        [
            make_box(
                rng.uniform(-300, 300),
                rng.uniform(-300, 300),
                rng.uniform(5, 90),
                rng.uniform(5, 90),
                rng.uniform(0, 180),
            )
            for _ in range(n_boxes)
        ]
    )
    heights = rng.uniform(3, 120, size=n_boxes)

    p0 = np.stack(
        [rng.uniform(-400, 400, n_seg), rng.uniform(-400, 400, n_seg), rng.uniform(0, 150, n_seg)],
        axis=1,
    )
    p1 = np.stack(
        [rng.uniform(-400, 400, n_seg), rng.uniform(-400, 400, n_seg), rng.uniform(0, 150, n_seg)],
        axis=1,
    )

    got = segment_clearance(T(p0), T(p1), T(boxes), T(heights)).numpy()
    want = np.array(
        [reference_clearance(p0[i], p1[i], boxes.tolist(), heights.tolist()) for i in range(n_seg)]
    )
    np.testing.assert_allclose(got, want, atol=1e-4)


@pytest.mark.parametrize("seed", [7, 8])
def test_matches_reference_with_endpoints_inside_boxes(seed):
    """Segments deliberately starting inside footprints -- pins the convention."""
    rng = np.random.default_rng(seed)
    boxes = np.stack([make_box(0, 0, 120, 120, rng.uniform(0, 180)) for _ in range(3)])
    heights = rng.uniform(10, 60, size=3)
    p0 = np.stack(
        [rng.uniform(-40, 40, 40), rng.uniform(-40, 40, 40), rng.uniform(0, 30, 40)], axis=1
    )
    p1 = np.stack(
        [rng.uniform(-400, 400, 40), rng.uniform(-400, 400, 40), rng.uniform(0, 150, 40)], axis=1
    )
    got = segment_clearance(T(p0), T(p1), T(boxes), T(heights)).numpy()
    want = np.array(
        [reference_clearance(p0[i], p1[i], boxes.tolist(), heights.tolist()) for i in range(40)]
    )
    np.testing.assert_allclose(got, want, atol=1e-4)


# --------------------------------------------------------------------------
# batching, shapes, invariants
# --------------------------------------------------------------------------


def test_chunking_does_not_change_the_answer():
    rng = np.random.default_rng(3)
    boxes = np.stack(
        [make_box(*rng.uniform(-200, 200, 2), 40, 25, rng.uniform(0, 180)) for _ in range(97)]
    )
    heights = rng.uniform(5, 80, 97)
    p0 = rng.uniform(-300, 300, (25, 3))
    p1 = rng.uniform(-300, 300, (25, 3))
    a = segment_clearance(T(p0), T(p1), T(boxes), T(heights), chunk=7)
    b = segment_clearance(T(p0), T(p1), T(boxes), T(heights), chunk=1000)
    torch.testing.assert_close(a, b)


def test_pairwise_is_symmetric_with_free_diagonal():
    rng = np.random.default_rng(5)
    boxes = T(np.stack([make_box(0, 0, 60, 40, 30)]))
    heights = T([25.0])
    pos = T(rng.uniform(-200, 200, (4, 6, 3)))
    c = pairwise_clearance(pos, boxes, heights)
    assert c.shape == (4, 6, 6)
    torch.testing.assert_close(c, c.transpose(1, 2))
    assert torch.all(torch.diagonal(c, dim1=1, dim2=2) == FREE_CLEARANCE_M)


def test_pairwise_agrees_with_segment_clearance():
    rng = np.random.default_rng(11)
    boxes = T(np.stack([make_box(10, -5, 80, 30, 15), make_box(-60, 40, 50, 50, 70)]))
    heights = T([30.0, 55.0])
    pos = T(rng.uniform(-200, 200, (3, 5, 3)))
    c = pairwise_clearance(pos, boxes, heights)
    for b in range(3):
        for i in range(5):
            for j in range(i + 1, 5):
                one = segment_clearance(pos[b, i], pos[b, j], boxes, heights)
                assert c[b, i, j].item() == pytest.approx(one.item(), abs=1e-6)


def test_leading_dimensions_are_preserved():
    boxes = T([make_box(0, 0, 40, 40, 0)])
    h = T([20.0])
    p0 = torch.zeros(2, 7, 3, dtype=torch.float64)
    p1 = torch.ones(2, 7, 3, dtype=torch.float64) * 100.0
    assert segment_clearance(p0, p1, boxes, h).shape == (2, 7)


def test_float32_matches_float64_closely():
    rng = np.random.default_rng(17)
    boxes = np.stack(
        [make_box(*rng.uniform(-300, 300, 2), 50, 30, rng.uniform(0, 180)) for _ in range(30)]
    )
    heights = rng.uniform(5, 100, 30)
    p0 = rng.uniform(-500, 500, (50, 3))
    p1 = rng.uniform(-500, 500, (50, 3))
    c64 = segment_clearance(T(p0), T(p1), T(boxes), T(heights))
    c32 = segment_clearance(T(p0).float(), T(p1).float(), T(boxes).float(), T(heights).float())
    # fp32 is the production dtype; coordinates span +-750 m so it is plenty
    np.testing.assert_allclose(c32.numpy(), c64.numpy(), atol=1e-2)
