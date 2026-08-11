"""Block B: choose the exact 1500 m box by sweeping candidate centres.

The height-coverage gate is settled (LoD2, 100 %) and does not discriminate
between placements. What does discriminate is *scenario geometry*, so this
script scores a grid of candidate centres on four measurable criteria and
prints a ranked table. The point is that the final box is chosen from evidence
and can be defended in the thesis, not hand-placed.

Criteria, in the priority order argued in docs/BLOCK_B.md:

1. **Usable area** -- the Main costs up to 6 % of some boxes, dead for both
   buildings and route sampling.
2. **Tower centrality** -- the towers are the mechanism for RQ1. They must sit
   between the MCV and the HVT's later positions to block A2A; pushed into a
   corner they stop intercepting links.
3. **MCV placements that admit the full escalation** -- ENVIRONMENT.md randomises
   the MCV per episode and puts the HVT 1400 m away at t=240 s, so a usable box
   is one where *many* MCV positions have a 1400 m-distant road point reachable
   within the same drivable component.
4. **Both height regimes present** -- low fabric for the observation envelope,
   towers for A2A blocking. A box with only one reproduces the Manhattan or the
   Paris failure in docs/DECISIONS.md.

Network access happens once, over a superset area covering every candidate box;
scoring is then pure local geometry.

Offline diagnostic. Nothing here is imported by `src/`.

Usage:
    uv run python scripts/choose_box.py
    uv run python scripts/choose_box.py --step 50 --extent 800
    uv run python scripts/choose_box.py --plot
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
from check_lod2_coverage import PAGE, bbox_deg, fetch_page, parse_members
from shapely.geometry import box as shapely_box

UTM32N = "EPSG:25832"
BOX_SIZE_M = 1500.0

# Centre of the sweep: the Bankenviertel. Candidates are offsets from here.
BASE = (50.1120, 8.6690)

# The binding constraint is the HVT's *speed*, not the road's class: the drone
# needs a 1.4-1.8x margin, which it keeps up to 50 km/h. So the HVT may use every
# surface street, capped at SPEED_CAP_MS. Excluding `primary` outright -- the
# earlier reading of docs/BLOCK_B.md -- also stranded the residential streets by
# the Hauptbahnhof, whose only link to the network runs along Mainzer Landstrasse.
SPEED_CAP_MS = 13.9  # 50 km/h; drone cruise 20 m/s => 1.44x margin

ROUTE_SPEEDS = {
    "living_street": 5.6,
    "residential": 8.3,
    "unclassified": 8.3,
    "tertiary": 13.9,
    "tertiary_link": 13.9,
    "secondary": 13.9,
    "secondary_link": 13.9,
    "primary": SPEED_CAP_MS,
    "primary_link": SPEED_CAP_MS,
}

# Grade-separated roads stay out: they carry no urban canyon, and an HVT on one
# leaves the box in under a minute. (None occur in the Frankfurt box, so this is
# a guard for the second city rather than an active filter here.)
EXCLUDED_CLASSES = {"motorway", "motorway_link", "trunk", "trunk_link"}

DRIVABLE = set(ROUTE_SPEEDS)

# HVT-to-MCV separation at t=240 s (docs/ENVIRONMENT.md escalation table).
ESCALATION_M = 1400.0

# Height regimes, metres.
FABRIC_MAX = 40.0
TOWER_MIN = 100.0

# Criterion 4 (both regimes present) is a **filter, not a score**. A real city is
# ~93 % fabric by built area, so grading a box on how close it gets to an even
# split just penalises every candidate by about the same amount and discriminates
# nothing. What matters is only that neither regime is absent.
MIN_TOWERS = 10
MIN_FABRIC_FRAC = 0.50

# Composite weights. A convenience for ranking only -- the per-criterion columns
# are what the decision should actually rest on, and re-weighting is a one-line
# change here. `central` leads because the tower cluster is the *mechanism* for
# RQ1: if it does not sit between the MCV and the HVT's later positions, the
# fidelity ladder has nothing to bite on.
WEIGHTS = {"usable": 0.30, "central": 0.40, "mcv": 0.30}

CACHE = Path(__file__).resolve().parent.parent / ".cache" / "choose_box"


# --------------------------------------------------------------------------
# one-time fetch over the superset area
# --------------------------------------------------------------------------


def superset_radius(extent_m: float) -> float:
    """Half-side of the area that must be fetched to cover every candidate."""
    return extent_m + BOX_SIZE_M / 2.0


def fetch_buildings(radius_m: float, refresh: bool) -> gpd.GeoDataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"lod2_r{int(radius_m)}.gpkg"
    if path.exists() and not refresh:
        return gpd.read_file(path)

    bbox = bbox_deg(*BASE, 2 * radius_m)
    print(f"fetching LoD2 over {2 * radius_m:.0f} m superset box ...")
    rows: list[tuple[object, float | None]] = []
    start = 0
    while True:
        page, n_members = parse_members(fetch_page(bbox, start))
        rows.extend(page)
        print(f"  startIndex={start:5d}  server {n_members:4d}  total {len(rows)}", flush=True)
        if n_members < PAGE:
            break
        start += PAGE
        time.sleep(1.0)

    gdf = gpd.GeoDataFrame(
        {"height": [h for _, h in rows]},
        geometry=[g for g, _ in rows],
        crs="EPSG:4326",
    )
    gdf.to_file(path, driver="GPKG")
    return gdf


def fetch_roads(radius_m: float, refresh: bool) -> nx.MultiDiGraph:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"roads_r{int(radius_m)}.graphml"
    if path.exists() and not refresh:
        return ox.io.load_graphml(path)
    print(f"fetching road graph, radius {radius_m:.0f} m ...")
    g = ox.graph_from_point(BASE, dist=radius_m, network_type="drive")
    ox.io.save_graphml(g, path)
    return g


def fetch_water(radius_m: float, refresh: bool) -> gpd.GeoDataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"water_r{int(radius_m)}.gpkg"
    if path.exists() and not refresh:
        return gpd.read_file(path)
    print(f"fetching water, radius {radius_m:.0f} m ...")
    frames = []
    for tags in ({"natural": "water"}, {"waterway": "riverbank"}):
        try:
            w = ox.features.features_from_point(BASE, tags=tags, dist=radius_m)
        except Exception as exc:  # noqa: BLE001 - diagnostic script
            print(f"  [warn] water query {tags} failed: {exc}")
            continue
        w = w[w.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
        if len(w):
            frames.append(w[["geometry"]])
    if not frames:
        return gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326")
    out = gpd.GeoDataFrame(gpd.pd.concat(frames, ignore_index=True), crs=frames[0].crs)
    out.to_file(path, driver="GPKG")
    return out


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------


def edge_speed(d: dict) -> float | None:
    """HVT speed for an edge, or None if it may not use it at all.

    Multi-class edges take the *slowest* matching class -- the conservative
    reading, and the one that keeps the drone's speed margin.
    """
    hw = d.get("highway")
    hw = {hw} if isinstance(hw, str) else set(hw or [])
    if hw & EXCLUDED_CLASSES:
        return None
    speeds = [ROUTE_SPEEDS[c] for c in hw if c in ROUTE_SPEEDS]
    if not speeds:
        return None
    return min(min(speeds), SPEED_CAP_MS)


def drivable_projected(g: nx.MultiDiGraph) -> nx.Graph:
    """Project to UTM and keep every edge the HVT may drive, speed-capped."""
    gp = ox.projection.project_graph(g, to_crs=UTM32N)
    keep = []
    for u, v, k, d in gp.edges(keys=True, data=True):
        if edge_speed(d) is not None:
            keep.append((u, v, k))
    return gp.edge_subgraph(keep).to_undirected()


def score_box(
    cx: float,
    cy: float,
    bld: gpd.GeoDataFrame,
    roads: nx.Graph,
    node_xy: dict,
    water: gpd.GeoDataFrame,
) -> dict | None:
    half = BOX_SIZE_M / 2.0
    sq = shapely_box(cx - half, cy - half, cx + half, cy + half)
    box_area = BOX_SIZE_M**2

    # --- 1. usable area -------------------------------------------------
    water_area = float(water.geometry.intersection(sq).area.sum()) if len(water) else 0.0
    water_frac = water_area / box_area

    # --- 2 & 4. buildings ------------------------------------------------
    b = bld[bld.geometry.intersects(sq)]
    if not len(b):
        return None
    b_area = b.geometry.intersection(sq).area
    total_built = float(b_area.sum())
    if total_built <= 0:
        return None

    h = b["height"].to_numpy(dtype=float)
    fabric_frac = float(b_area[h < FABRIC_MAX].sum()) / total_built
    towers = b[h >= TOWER_MIN]
    n_towers = len(towers)
    if n_towers:
        tc = towers.geometry.centroid
        offset = float(np.hypot(tc.x.mean() - cx, tc.y.mean() - cy))
    else:
        offset = float("inf")

    # --- 3. MCV placements admitting the 1400 m escalation ---------------
    inside = [n for n in roads.nodes if sq.contains_properly(node_xy[n])]
    sub = roads.subgraph(inside)
    if sub.number_of_nodes() < 2:
        return None
    comps = sorted(nx.connected_components(sub), key=len, reverse=True)
    big = sub.subgraph(comps[0])
    pts = np.array([[node_xy[n].x, node_xy[n].y] for n in big.nodes])
    # An MCV placement works if some road point in the SAME component is far
    # enough away to host the t=240 s position.
    d = np.linalg.norm(pts[:, None] - pts[None, :], axis=-1)
    mcv_frac = float(np.mean(d.max(axis=1) >= ESCALATION_M))

    total_len = sum(e.get("length", 0.0) for *_, e in sub.edges(data=True))
    big_len = sum(e.get("length", 0.0) for *_, e in big.edges(data=True))
    conn = big_len / max(total_len, 1e-9)

    # --- criterion 4 as a filter -----------------------------------------
    if n_towers < MIN_TOWERS or fabric_frac < MIN_FABRIC_FRAC:
        return None

    # --- normalise and combine -------------------------------------------
    s_usable = 1.0 - water_frac
    s_central = max(0.0, 1.0 - offset / half)
    s_mcv = mcv_frac

    score = WEIGHTS["usable"] * s_usable + WEIGHTS["central"] * s_central + WEIGHTS["mcv"] * s_mcv

    return {
        "cx": cx,
        "cy": cy,
        "water_pct": 100 * water_frac,
        "n_towers": n_towers,
        "tower_off_m": offset,
        "fabric_pct": 100 * fabric_frac,
        "mcv_pct": 100 * mcv_frac,
        "road_km": total_len / 1000.0,
        "conn_pct": 100 * conn,
        "score": score,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=float, default=100.0, help="sweep step, metres")
    ap.add_argument("--extent", type=float, default=600.0, help="max offset from BASE, metres")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--plot", action="store_true", help="render the winning box")
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    radius = superset_radius(args.extent)
    bld = fetch_buildings(radius, args.refresh).to_crs(UTM32N)
    bld = bld[bld.geometry.area >= 5.0]  # drop LoD2 wall slivers
    bld = bld[bld["height"].notna() & (bld["height"] > 0)]
    roads_raw = fetch_roads(radius, args.refresh)
    water = fetch_water(radius, args.refresh).to_crs(UTM32N)

    roads = drivable_projected(roads_raw)
    node_xy = {n: gpd.points_from_xy([d["x"]], [d["y"]])[0] for n, d in roads.nodes(data=True)}

    bx, by = (
        gpd.GeoSeries(gpd.points_from_xy([BASE[1]], [BASE[0]]), crs="EPSG:4326")
        .to_crs(UTM32N)
        .iloc[0]
        .coords[0]
    )

    offsets = np.arange(-args.extent, args.extent + 1e-9, args.step)
    print(
        f"\nscoring {len(offsets) ** 2} candidate centres "
        f"({args.step:.0f} m steps, +/-{args.extent:.0f} m) ..."
    )
    rows = []
    for dy in offsets:
        for dx in offsets:
            r = score_box(bx + dx, by + dy, bld, roads, node_xy, water)
            if r is not None:
                r["dx"], r["dy"] = dx, dy
                rows.append(r)

    df = gpd.pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

    # back to lon/lat for the report -- the pipeline stores origin_lonlat
    ll = (
        gpd.GeoSeries(gpd.points_from_xy(df["cx"], df["cy"]), crs=UTM32N)
        .to_crs("EPSG:4326")
        .reset_index(drop=True)
    )
    df["lat"], df["lon"] = ll.y, ll.x

    print(
        f"\n{'lat':>9} {'lon':>8} {'score':>6} {'water%':>7} {'towers':>7} "
        f"{'twr_off':>8} {'fabric%':>8} {'mcv%':>6} {'road_km':>8} {'conn%':>6}"
    )
    for _, r in df.head(args.top).iterrows():
        print(
            f"{r['lat']:9.5f} {r['lon']:8.5f} {r['score']:6.3f} {r['water_pct']:7.1f} "
            f"{int(r['n_towers']):7d} {r['tower_off_m']:8.0f} {r['fabric_pct']:8.1f} "
            f"{r['mcv_pct']:6.0f} {r['road_km']:8.1f} {r['conn_pct']:6.0f}"
        )

    best = df.iloc[0]
    print(
        f"\nweights {WEIGHTS}"
        f"\ntop centre: lat {best['lat']:.5f}, lon {best['lon']:.5f}"
        f"\n  water {best['water_pct']:.1f}% | {int(best['n_towers'])} towers, centroid "
        f"{best['tower_off_m']:.0f} m off centre | fabric {best['fabric_pct']:.0f}% of built area"
        f"\n  {best['mcv_pct']:.0f}% of road nodes admit the {ESCALATION_M:.0f} m escalation"
        f" | {best['road_km']:.1f} km drivable, {best['conn_pct']:.0f}% in one component"
    )
    print(
        "\nThe columns, not the composite, are the decision. Re-weight in WEIGHTS "
        "if you disagree with the ranking."
    )

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nfull table -> {args.csv}")

    if args.plot:
        # Render every road, including any the HVT may not use, so the figure
        # shows the real corridor structure of the box.
        roads_all = ox.projection.project_graph(roads_raw, to_crs=UTM32N).to_undirected()
        plot_box(best, bld, roads_all, node_xy, water)


def plot_box(best, bld, roads_all, node_xy, water) -> None:
    import matplotlib.pyplot as plt

    half = BOX_SIZE_M / 2.0
    cx, cy = best["cx"], best["cy"]
    sq = shapely_box(cx - half, cy - half, cx + half, cy + half)

    fig, ax = plt.subplots(figsize=(9, 9))
    if len(water):
        water.geometry.intersection(sq).plot(ax=ax, color="#a8c8e0", zorder=0)

    b = bld[bld.geometry.intersects(sq)]
    h = b["height"].to_numpy(dtype=float)
    b[h < FABRIC_MAX].plot(ax=ax, color="#c9c9c9", edgecolor="none", zorder=1)
    b[(h >= FABRIC_MAX) & (h < TOWER_MIN)].plot(ax=ax, color="#8a8a8a", zorder=2)
    b[h >= TOWER_MIN].plot(ax=ax, color="#c0392b", zorder=3)

    # Follow each edge's real geometry. Drawing straight chords between
    # intersection nodes instead invents long diagonals across whole blocks
    # wherever intersections are sparse, which makes the network look far
    # thinner than it is.
    def edge_xy(u, v, d):
        geom = d.get("geometry")
        if geom is not None and hasattr(geom, "xy"):
            return geom.xy
        if u in node_xy and v in node_xy:
            return ([node_xy[u].x, node_xy[v].x], [node_xy[u].y, node_xy[v].y])
        return None

    # Colour by the speed the HVT may drive, not by whether it may drive at all
    # -- every surface street is routable now, capped at SPEED_CAP_MS.
    for u, v, d in roads_all.edges(data=True):
        xy = edge_xy(u, v, d)
        if xy is None:
            continue
        spd = edge_speed(d)
        if spd is None:
            ax.plot(*xy, color="#b0b0b0", lw=0.8, ls=":", alpha=0.8, zorder=4)
        elif spd >= SPEED_CAP_MS:
            ax.plot(*xy, color="#e08a3c", lw=1.5, alpha=0.9, zorder=5)
        else:
            ax.plot(*xy, color="#2c3e50", lw=0.7, alpha=0.8, zorder=4)

    ax.plot(*sq.exterior.xy, color="k", lw=2, zorder=6)
    ax.set_xlim(cx - half - 100, cx + half + 100)
    ax.set_ylim(cy - half - 100, cy + half + 100)
    ax.set_aspect("equal")
    ax.set_title(
        f"candidate box  lat {best['lat']:.5f}  lon {best['lon']:.5f}\n"
        f"red = towers >{TOWER_MIN:.0f} m, dark grey = 40-100 m, light grey = fabric\n"
        f"orange = HVT at {SPEED_CAP_MS:.1f} m/s cap, dark blue = slower streets",
        fontsize=9,
    )
    out = CACHE / "chosen_box.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"figure -> {out}")


if __name__ == "__main__":
    main()
