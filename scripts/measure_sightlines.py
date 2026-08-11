"""Block B: measure what the scenario design left as assumptions.

Two empirical questions, both flagged in docs/BLOCK_B.md as "measure, do not
assume", and both feeding Chapter 3:

1. **How long is a real sightline down a Frankfurt street?**
   `ENVIRONMENT.md` says the 830 m recognition range "holds only down a clear
   straight street" and guesses 100-400 m is typical. That guess currently sets
   acquisition difficulty and nobody has checked it.

2. **What is the true canyon ratio `H_b/W`?**
   `PHYSICS.md` assumes 20 m streets and 22 m fabric, giving a 36 m
   across-street observation envelope at 80 m altitude. Measure the spread
   rather than trusting the single assumed value.

Method. Sample points along the road graph, take the local street axis from the
edge geometry, and cast a ray each way at the HVT's own height until it meets a
building. That is a **2D ground-level corridor length** -- how far one can see
along the street. The full 3D drone-to-target case belongs to Block C's occlusion
test; this bounds it and is what "clear straight street" actually means.

Offline diagnostic. Nothing here is imported by `src/`.

Usage:
    uv run python scripts/measure_sightlines.py --plot
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import osmnx as ox
from prep_osm import BOX_SIZE_M, UTM32N, fetch_buildings, fetch_roads, local_origin_utm
from shapely.geometry import LineString, Point
from shapely.geometry import box as shapely_box
from shapely.strtree import STRtree

# A vehicle-height ray: what a sensor looking at the HVT along the street sees.
RAY_HEIGHT_M = 2.0
MAX_RAY_M = 1500.0  # box diagonal bounds anything longer anyway
SAMPLE_EVERY_M = 25.0

# PHYSICS.md assumptions under test
ASSUMED_STREET_W = 20.0
ASSUMED_FABRIC_H = 22.0
DRONE_ALT_M = 80.0

CACHE = Path(__file__).resolve().parent.parent / ".cache" / "sightlines"


def sample_road_points(g, ox_, oy_, half):
    """Points every SAMPLE_EVERY_M along each road, with the local street axis."""
    pts, dirs, classes = [], [], []
    for u, v, k, d in g.edges(keys=True, data=True):
        hw = d.get("highway")
        hw = hw if isinstance(hw, str) else (hw or ["?"])[0]
        geom = d.get("geometry")
        if geom is not None and hasattr(geom, "coords"):
            line = np.asarray(geom.coords)[:, :2]
        else:
            line = np.array(
                [[g.nodes[u]["x"], g.nodes[u]["y"]], [g.nodes[v]["x"], g.nodes[v]["y"]]]
            )
        for a, b in itertools.pairwise(line):
            seg = b - a
            L = float(np.hypot(*seg))
            if L < 1e-6:
                continue
            axis = seg / L
            for t in np.arange(0.0, L, SAMPLE_EVERY_M):
                p = a + axis * t
                if abs(p[0] - ox_) <= half and abs(p[1] - oy_) <= half:
                    pts.append(p)
                    dirs.append(axis)
                    classes.append(hw)
    return np.array(pts), np.array(dirs), np.array(classes)


def dist_to_box_edge(p, ax, half, ox_, oy_):
    """How far the ray travels before it leaves the box (slab method, 2D)."""
    t = MAX_RAY_M
    for c, a, lo, hi in (
        (p[0], ax[0], ox_ - half, ox_ + half),
        (p[1], ax[1], oy_ - half, oy_ + half),
    ):
        if abs(a) < 1e-12:
            continue
        t = min(t, max((lo - c) / a, (hi - c) / a))
    return max(t, 0.0)


def cast(pts, dirs, tree, geoms, half, ox_, oy_):
    """Unobstructed distance along +axis and -axis, with censoring flags.

    Buildings only exist inside the fetched area, so a ray that leaves the box
    stops being a measurement -- it flies through empty space and would report a
    spuriously long sightline. Those samples are **right-censored**: the true
    value is *at least* the distance to the box edge. Reporting them as if
    measured put 28 % of points beyond 830 m, which is an artefact of the data
    boundary, not of Frankfurt.
    """
    out = np.full((len(pts), 2), MAX_RAY_M, dtype=float)
    censored = np.zeros((len(pts), 2), dtype=bool)
    for i, (p, ax) in enumerate(zip(pts, dirs, strict=True)):
        for j, sign in enumerate((1.0, -1.0)):
            d_ax = sign * ax
            edge = dist_to_box_edge(p, d_ax, half, ox_, oy_)
            ray = LineString([p, p + d_ax * MAX_RAY_M])
            hit = MAX_RAY_M
            for idx in tree.query(ray):
                inter = geoms[idx].intersection(ray)
                if inter.is_empty:
                    continue
                hit = min(hit, Point(p).distance(inter))
            if hit > edge:
                out[i, j] = edge
                censored[i, j] = True
            else:
                out[i, j] = hit
    return out, censored


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    ox_, oy_ = local_origin_utm()
    half = BOX_SIZE_M / 2.0
    sq = shapely_box(ox_ - half, oy_ - half, ox_ + half, oy_ + half)

    b = fetch_buildings(False).to_crs(UTM32N)
    b = b[b["height"].notna() & (b["height"] > 0)]
    b = b[b.geometry.intersects(sq)].copy()
    # only buildings tall enough to block a ray at vehicle height
    blockers = b[b["height"] >= RAY_HEIGHT_M]
    geoms = list(blockers.geometry)
    tree = STRtree(geoms)

    g = ox.projection.project_graph(fetch_roads(False), to_crs=UTM32N)
    pts, dirs, classes = sample_road_points(g, ox_, oy_, half)
    print(f"sampling {len(pts)} road points every {SAMPLE_EVERY_M:.0f} m ...")

    los, cens = cast(pts, dirs, tree, geoms, half, ox_, oy_)

    # Treat each direction as one observation. A ray that left the box tells us
    # only a lower bound, so quote the uncensored distribution and report the
    # censoring rate alongside it rather than mixing the two.
    flat, flat_c = los.ravel(), cens.ravel()
    meas = flat[~flat_c]

    print("\n=== SIGHTLINE ALONG THE STREET (ground level) ===")
    print(f"  {len(flat)} rays, {100 * flat_c.mean():.0f}% left the box (right-censored)")
    q = np.percentile(meas, [10, 25, 50, 75, 90, 99])
    print(
        f"  measured rays:  p10 {q[0]:5.0f}  p25 {q[1]:5.0f}  p50 {q[2]:5.0f}  "
        f"p75 {q[3]:5.0f}  p90 {q[4]:5.0f}  p99 {q[5]:5.0f}  max {meas.max():5.0f} m"
    )
    # Conservative bound over ALL rays: censored ones counted at their lower
    # bound. This under-states long sightlines, so it brackets the truth.
    ql = np.percentile(flat, [50, 75, 90])
    print(
        f"  lower bound over all rays:      p50 {ql[0]:5.0f}  p75 {ql[1]:5.0f}  p90 {ql[2]:5.0f} m"
    )

    print("\n  ENVIRONMENT.md guesses 100-400 m is typical (measured rays):")
    print(f"    within 100-400 m : {100 * np.mean((meas >= 100) & (meas <= 400)):.0f}%")
    print(f"    below 100 m      : {100 * np.mean(meas < 100):.0f}%")
    print(f"    beyond 830 m     : {100 * np.mean(meas > 830):.1f}%")

    print("\n  by road class (measured rays):")
    cls_per_ray = np.repeat(classes, 2)
    cls_meas = cls_per_ray[~flat_c]
    rows = []
    for c in sorted(set(classes)):
        m = cls_meas == c
        if m.sum() < 20:
            continue
        rows.append(
            (
                c,
                m.sum(),
                np.median(meas[m]),
                np.percentile(meas[m], 90),
                100 * flat_c[cls_per_ray == c].mean(),
            )
        )
    for c, n, p50, p90, pc in sorted(rows, key=lambda r: -r[2]):
        print(f"    {c:16s} n={n:5d}  p50 {p50:5.0f}  p90 {p90:5.0f} m  censored {pc:3.0f}%")

    best = meas

    # ---- canyon ratio -------------------------------------------------
    print("\n=== CANYON RATIO H_b / W ===")
    bu_tree = STRtree(geoms)
    heights = blockers["height"].to_numpy()
    widths, ratios, local_h = [], [], []
    for p in pts:
        pt = Point(p)
        near = bu_tree.query(pt.buffer(60.0))
        if len(near) == 0:
            continue
        d = min(geoms[i].distance(pt) for i in near)
        w = 2.0 * d  # centreline to facade, both sides
        if w < 2.0 or w > 120.0:
            continue
        h = float(np.median(heights[near]))
        widths.append(w)
        ratios.append(h / w)
        local_h.append(h)
    widths = np.array(widths)
    ratios = np.array(ratios)
    local_h = np.array(local_h)
    qw = np.percentile(widths, [10, 25, 50, 75, 90])
    qr = np.percentile(ratios, [10, 25, 50, 75, 90])
    print(
        f"  street width W (m): p10 {qw[0]:.0f}  p25 {qw[1]:.0f}  p50 {qw[2]:.0f}  p75 {qw[3]:.0f}  p90 {qw[4]:.0f}"
    )
    print(
        f"  ratio H_b/W:        p10 {qr[0]:.2f}  p25 {qr[1]:.2f}  p50 {qr[2]:.2f}  p75 {qr[3]:.2f}  p90 {qr[4]:.2f}"
    )
    print(
        f"  PHYSICS.md assumes W={ASSUMED_STREET_W:.0f} m, H_b={ASSUMED_FABRIC_H:.0f} m -> ratio {ASSUMED_FABRIC_H / ASSUMED_STREET_W:.2f}"
    )

    # across-street envelope (PHYSICS.md): (W/2)*h/H_b at drone altitude
    env = (widths / 2.0) * DRONE_ALT_M / np.maximum(local_h, 1e-6)
    qe = np.percentile(env, [10, 25, 50, 75, 90])
    print(f"\n  across-street envelope at {DRONE_ALT_M:.0f} m (m):")
    print(
        f"    p10 {qe[0]:.0f}  p25 {qe[1]:.0f}  p50 {qe[2]:.0f}  p75 {qe[3]:.0f}  p90 {qe[4]:.0f}   (PHYSICS.md assumes 36)"
    )

    np.savez(
        CACHE / "sightlines.npz",
        sightline_measured=meas,
        sightline_all=flat,
        censored=flat_c,
        widths=widths,
        ratios=ratios,
        envelope=env,
    )

    if args.plot:
        plot(best, ratios, widths)


def plot(best, ratios, widths) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].hist(best, bins=60, color="#2c3e50")
    axes[0].axvspan(100, 400, color="#e08a3c", alpha=0.25, label="ENVIRONMENT.md guess")
    axes[0].axvline(830, color="#c0392b", lw=2, label="830 m sensor limit")
    axes[0].set_xlabel("unobstructed sightline along street (m)")
    axes[0].set_ylabel("road points")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Sightline distribution", fontsize=11)

    axes[1].hist(widths, bins=50, color="#2c3e50")
    axes[1].axvline(ASSUMED_STREET_W, color="#c0392b", lw=2, label="assumed 20 m")
    axes[1].set_xlabel("street width W (m)")
    axes[1].legend(fontsize=8)
    axes[1].set_title("Street width", fontsize=11)

    axes[2].hist(ratios, bins=60, range=(0, 6), color="#2c3e50")
    axes[2].axvline(ASSUMED_FABRIC_H / ASSUMED_STREET_W, color="#c0392b", lw=2, label="assumed 1.1")
    axes[2].set_xlabel("canyon ratio $H_b/W$")
    axes[2].legend(fontsize=8)
    axes[2].set_title("Canyon ratio", fontsize=11)

    fig.suptitle("Frankfurt box — measured, not assumed", fontsize=12)
    fig.tight_layout()
    out = CACHE / "sightlines.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nfigure -> {out}")


if __name__ == "__main__":
    main()
