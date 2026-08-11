"""Batched line-of-sight against 2.5D oriented building boxes.

This is the **F1 rung of RQ1**. Occlusion is the hypothesis of the whole thesis:
that it is the effect a channel abstraction cannot afford to omit. So this module
is not a utility, it is the independent variable.

A building is a vertical prism: an oriented rectangle extruded from ``z = 0`` to
``z = H``. The test is the standard slab method, run in each box's own frame so
that rotation costs one 2x2 multiply rather than a different algorithm --
axis-aligned boxes were measured and rejected, they fill 94 % of the Frankfurt
box (``docs/DECISIONS.md``).

**Returns signed clearance in metres, not a boolean.** Positive means the ray
passes that far above the roofline, negative means it is blocked by that depth.
Three consumers need exactly this: ``channel.pathloss_a2a_db(occluded=...)`` and
``pathloss_a2g_umi_av_db(los=...)`` take the sign, and the actor observation
carries "clearance margin -- signed metres the ray clears the roofline"
(``docs/ENVIRONMENT.md``). The margin is also the better learning signal: it
tells the policy *how close* it is to losing a link, which is what makes RQ3's
anticipation metric measurable at all.

Pure and batched. No ``.item()``, ``.cpu()`` or ``.numpy()`` -- see AGENTS.md.
"""

from __future__ import annotations

import torch
from torch import Tensor

# Reported clearance when nothing blocks. Finite on purpose: +inf poisons any
# downstream mean/normalisation, and an observation of +inf is unusable.
FREE_CLEARANCE_M = 1.0e4


def _slab_interval(o: Tensor, d: Tensor, half: Tensor, big: float = 1.0e9) -> tuple[Tensor, Tensor]:
    """Entry/exit ray parameters for one slab ``|o + t*d| <= half``.

    Branch-free. Where the ray is parallel to the slab (``d ~ 0``) the interval
    is the whole line if the origin is inside it and empty otherwise, expressed
    with +/-``big`` sentinels so the caller's min/max still composes.
    """
    safe_d = torch.where(d.abs() < 1e-9, torch.full_like(d, 1.0), d)
    t1 = (-half - o) / safe_d
    t2 = (half - o) / safe_d
    lo = torch.minimum(t1, t2)
    hi = torch.maximum(t1, t2)

    parallel = d.abs() < 1e-9
    inside = o.abs() <= half
    lo = torch.where(parallel, torch.where(inside, -big, big), lo)
    hi = torch.where(parallel, torch.where(inside, big, -big), hi)
    return lo, hi


def segment_clearance(
    p0: Tensor,
    p1: Tensor,
    boxes: Tensor,
    heights: Tensor,
    *,
    ignore_endpoint_boxes: bool = True,
    chunk: int = 512,
) -> Tensor:
    """Signed clearance in metres for a batch of segments against all boxes.

    Args:
        p0, p1: ``(..., 3)`` segment endpoints, local metres, z up.
        boxes:  ``(M, 6)`` ``cx, cy, half_w, half_h, cos(theta), sin(theta)``.
        heights: ``(M,)`` building heights in metres.
        ignore_endpoint_boxes: skip any box that *contains* an endpoint. Boxes
            over-approximate real footprints, so a node standing in one would
            otherwise blind itself permanently. Containment is 3D: a drone at
            80 m over a 22 m block is not inside it.
        chunk: boxes processed per iteration. Memory, not arithmetic, is the
            binding constraint here -- see the module benchmark.

    Returns:
        ``(...)`` signed metres. ``> 0`` clears the roofline, ``< 0`` blocked.
    """
    lead = p0.shape[:-1]
    a = p0.reshape(-1, 3)
    b = p1.reshape(-1, 3)
    n = a.shape[0]

    best = torch.full((n,), FREE_CLEARANCE_M, dtype=a.dtype, device=a.device)

    for start in range(0, boxes.shape[0], chunk):
        bx = boxes[start : start + chunk]
        hh = heights[start : start + chunk]

        cx, cy = bx[:, 0], bx[:, 1]
        half_w, half_h = bx[:, 2], bx[:, 3]
        ca, sa = bx[:, 4], bx[:, 5]

        # --- into each box's frame: translate, then rotate by -theta ---
        dx0 = a[:, None, 0] - cx
        dy0 = a[:, None, 1] - cy
        dx1 = b[:, None, 0] - cx
        dy1 = b[:, None, 1] - cy

        lx0 = dx0 * ca + dy0 * sa
        ly0 = -dx0 * sa + dy0 * ca
        lx1 = dx1 * ca + dy1 * sa
        ly1 = -dx1 * sa + dy1 * ca

        dirx = lx1 - lx0
        diry = ly1 - ly0

        # --- 2D slab test -> parameter interval, clamped to the segment ---
        lo_x, hi_x = _slab_interval(lx0, dirx, half_w)
        lo_y, hi_y = _slab_interval(ly0, diry, half_h)

        t_enter = torch.maximum(torch.maximum(lo_x, lo_y), torch.zeros_like(lo_x))
        t_exit = torch.minimum(torch.minimum(hi_x, hi_y), torch.ones_like(hi_x))
        overlaps = t_enter <= t_exit

        # --- 2.5D: altitude across that interval, not just a planar crossing ---
        z0 = a[:, None, 2]
        dz = b[:, None, 2] - z0
        z_at_enter = z0 + t_enter * dz
        z_at_exit = z0 + t_exit * dz
        z_min = torch.minimum(z_at_enter, z_at_exit)  # z is linear in t

        clearance = z_min - hh

        if ignore_endpoint_boxes:
            in0 = (lx0.abs() <= half_w) & (ly0.abs() <= half_h) & (a[:, None, 2] <= hh)
            in1 = (lx1.abs() <= half_w) & (ly1.abs() <= half_h) & (b[:, None, 2] <= hh)
            overlaps = overlaps & ~(in0 | in1)

        clearance = torch.where(overlaps, clearance, torch.full_like(clearance, FREE_CLEARANCE_M))
        best = torch.minimum(best, clearance.min(dim=1).values)

    return best.reshape(lead)


def pairwise_clearance(
    pos: Tensor,
    boxes: Tensor,
    heights: Tensor,
    *,
    ignore_endpoint_boxes: bool = True,
    chunk: int = 512,
) -> Tensor:
    """All-pairs line-of-sight clearance for a batch of environments.

    Args:
        pos: ``(B, K, 3)`` node positions -- drones, MCV and HVT together.

    Returns:
        ``(B, K, K)`` symmetric signed metres. The diagonal is
        ``FREE_CLEARANCE_M``: a node has line of sight to itself, and the value
        must never be mistaken for a real link by the routing layer.

    Only the upper triangle is computed; the lower half is a transpose. Doing
    both halves doubles the cost for nothing.
    """
    bsz, k, _ = pos.shape
    iu, ju = torch.triu_indices(k, k, offset=1, device=pos.device)

    p0 = pos[:, iu, :]  # (B, P, 3)
    p1 = pos[:, ju, :]
    flat = segment_clearance(
        p0, p1, boxes, heights, ignore_endpoint_boxes=ignore_endpoint_boxes, chunk=chunk
    )

    out = torch.full((bsz, k, k), FREE_CLEARANCE_M, dtype=pos.dtype, device=pos.device)
    out[:, iu, ju] = flat
    out[:, ju, iu] = flat
    return out


def is_occluded(clearance: Tensor) -> Tensor:
    """Boolean blockage from signed clearance, for `channel.py`'s flags."""
    return clearance < 0.0
