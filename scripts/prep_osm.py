"""Block B: bake the Frankfurt box into a tensor artefact.

Runs offline, once. Everything geospatial happens here so that nothing
`osmnx`- or `shapely`-shaped ever runs inside `step()`. The output is a single
`.npz` loadable with NumPy alone.

Sources, both settled in docs/BLOCK_B.md:
  * buildings -- Hessen LoD2 via the INSPIRE WFS (100 % measured heights)
  * roads     -- OpenStreetMap via osmnx

Produces `data/frankfurt_box.npz`:

    building_boxes   (M, 6)  cx, cy, half_w, half_h, cos(theta), sin(theta)
    building_heights (M,)    metres above ground
    height_grid      (75,75) max building height per 20 m cell
    road_nodes       (K, 2)  local metres
    road_edges       (E, 2)  node index pairs
    road_speeds      (E,)    m/s, capped
    road_route_ok    (E,)    bool -- HVT may drive this edge
    route_mcv        (R, 2)  MCV position per pre-sampled route
    route_xy         (R, T, 2) HVT position at each of T steps
    origin_lonlat    (2,)    provenance
    box_size_m       ()      1500.0

Coordinates are local metres with the **box centre at the origin**, so every
in-box coordinate lies in [-750, +750].

Usage:
    uv run python scripts/prep_osm.py
    uv run python scripts/prep_osm.py --plot --refresh
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
from check_lod2_coverage import PAGE, bbox_deg, fetch_page, parse_members
from choose_box import EXCLUDED_CLASSES, ROUTE_SPEEDS, SPEED_CAP_MS
from shapely.geometry import LineString
from shapely.geometry import box as shapely_box
from shapely.ops import split as shapely_split
from shapely.ops import unary_union

# --------------------------------------------------------------------------
# frozen scenario parameters -- see AGENTS.md / docs/BLOCK_B.md
# --------------------------------------------------------------------------

ORIGIN_LATLON = (50.11200, 8.67040)  # FROZEN 2026-08-11
BOX_SIZE_M = 1500.0
UTM32N = "EPSG:25832"

GRID_CELL_M = 20.0
GRID_N = int(BOX_SIZE_M // GRID_CELL_M)  # 75

MIN_PART_AREA_M2 = 5.0  # LoD2 wall slivers below this are noise
MIN_PART_HEIGHT_M = 2.0  # sub-2 m "buildings" are LoD2 artefacts, not obstacles
MIN_PART_HALF_M = 0.5  # and neither is a box thinner than a metre

# A LoD2 part lying this much on an OSM bridge is a bridge deck, not a building.
BRIDGE_OVERLAP = 0.5
BRIDGE_HALF_WIDTH_M = 12.0  # OSM bridge ways are lines; give them a deck width

# Split a footprint into several oriented boxes while its single-OBB fit is
# worse than this. 1.5 / 4 parts takes over-approximation from +37 % to +21 %
# for 1.21x the box count -- see docs/BLOCK_C.md.
OBB_SPLIT_RATIO = 1.5
OBB_MAX_PARTS = 4

# Episode: 600 steps x 0.4 s = 240 s (AGENTS.md)
EPISODE_STEPS = 600
DT_S = 0.4

# HVT-to-MCV separation profile (docs/ENVIRONMENT.md escalation table)
HVT_START_MIN_M = 300.0
HVT_START_MAX_M = 500.0
HVT_END_MIN_M = 1200.0

# The MCV must sit somewhere the escalation is geometrically possible. In a
# 1500 m box only 66 % of road nodes have another node 1400 m away at all, so
# sampling the MCV uniformly caps the achievable separation well below target.
# Calibrated jointly with CONGESTION_FACTOR against the escalation table.
MCV_MIN_REACH_M = 1500.0

# Free-flow class speeds are what a road *permits*; a vehicle crossing a city
# centre averages far less once junctions, lights and turns are counted. The
# escalation table implies ~1000 m of radial gain over 240 s. This factor scales
# class speed to an achievable mean; it is a scenario assumption, stated here and
# flagged in the thesis.
CONGESTION_FACTOR = 0.70

# Exponent on the outward gain when choosing the next junction. 0 = unbiased
# random walk, large = always take the most outward edge (and so always the same
# route). ~2 keeps routes varied while still escalating the hop count.
OUTWARD_BIAS = 2.0
MAX_ROUTE_NODES = 400

# Reject a route that spends more than this fraction of the episode inside a
# building footprint -- it would be unobservable and the episode unwinnable.
MAX_ROUTE_INSIDE_FRAC = 0.05

N_ROUTES = 2048

CACHE = Path(__file__).resolve().parent.parent / ".cache" / "prep_osm"
OUT = Path(__file__).resolve().parent.parent / "data" / "frankfurt_box.npz"


# --------------------------------------------------------------------------
# fetch
# --------------------------------------------------------------------------


def local_origin_utm() -> tuple[float, float]:
    p = (
        gpd.GeoSeries(gpd.points_from_xy([ORIGIN_LATLON[1]], [ORIGIN_LATLON[0]]), crs="EPSG:4326")
        .to_crs(UTM32N)
        .iloc[0]
    )
    return float(p.x), float(p.y)


def fetch_buildings(refresh: bool) -> gpd.GeoDataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "lod2.gpkg"
    if path.exists() and not refresh:
        return gpd.read_file(path)

    # Fetch a slightly larger area than the box: a building straddling the edge
    # must still be clipped, not dropped.
    bbox = bbox_deg(*ORIGIN_LATLON, BOX_SIZE_M + 200.0)
    print("fetching LoD2 buildings ...")
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


def fetch_bridges(refresh: bool):
    """Union of OSM bridge footprints, in UTM metres, or None if there are none.

    Used to reject LoD2 parts that are bridge decks rather than buildings.
    Bridge *ways* are lines, so they are buffered to a plausible deck half-width.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "bridges.gpkg"
    if path.exists() and not refresh:
        br = gpd.read_file(path)
    else:
        print("fetching bridges ...")
        parts = []
        for tags in ({"man_made": "bridge"}, {"bridge": True}):
            try:
                x = ox.features.features_from_point(
                    ORIGIN_LATLON, tags=tags, dist=BOX_SIZE_M / 2 + 100.0
                )
            except Exception as exc:  # noqa: BLE001 - no bridges is a valid answer
                print(f"  [warn] bridge query {tags} failed: {exc}")
                continue
            if len(x):
                parts.append(x[["geometry"]])
        if not parts:
            return None
        br = gpd.GeoDataFrame(gpd.pd.concat(parts, ignore_index=True), crs=parts[0].crs)
        br.to_file(path, driver="GPKG")

    br = br.to_crs(UTM32N)
    polys = [g for g in br.geometry if g.geom_type in ("Polygon", "MultiPolygon")]
    lines = [
        g.buffer(BRIDGE_HALF_WIDTH_M)
        for g in br.geometry
        if g.geom_type in ("LineString", "MultiLineString")
    ]
    if not polys and not lines:
        return None
    return unary_union(polys + lines)


def fetch_roads(refresh: bool) -> nx.MultiDiGraph:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "roads.graphml"
    if path.exists() and not refresh:
        return ox.io.load_graphml(path)
    print("fetching road graph ...")
    g = ox.graph_from_point(ORIGIN_LATLON, dist=BOX_SIZE_M / 2 + 200.0, network_type="drive")
    ox.io.save_graphml(g, path)
    return g


# --------------------------------------------------------------------------
# buildings -> oriented boxes
# --------------------------------------------------------------------------


def oriented_box(poly) -> tuple[float, float, float, float, float, float] | None:
    """Minimum-area rotated rectangle as (cx, cy, half_w, half_h, cos, sin).

    `theta` is the orientation of the half_w axis. Storing cos/sin rather than
    the angle keeps trigonometry out of the runtime occlusion test.
    """
    mrr = poly.minimum_rotated_rectangle
    if mrr.is_empty or mrr.geom_type != "Polygon":
        return None
    xs, ys = mrr.exterior.coords.xy
    pts = np.array(list(zip(xs[:4], ys[:4], strict=True)))
    e0 = pts[1] - pts[0]
    e1 = pts[2] - pts[1]
    l0, l1 = float(np.hypot(*e0)), float(np.hypot(*e1))
    if l0 <= 0 or l1 <= 0:
        return None
    # let the *longer* edge define the local x axis, purely for consistency
    if l0 >= l1:
        axis, half_w, half_h = e0 / l0, l0 / 2.0, l1 / 2.0
    else:
        axis, half_w, half_h = e1 / l1, l1 / 2.0, l0 / 2.0
    cx, cy = pts.mean(axis=0)
    return float(cx), float(cy), float(half_w), float(half_h), float(axis[0]), float(axis[1])


def _long_axis(mrr):
    """Unit vector along the rectangle's longer side, its length, and centre."""
    c = np.asarray(mrr.exterior.coords)[:4]
    edges = [c[1] - c[0], c[2] - c[1]]
    lens = [float(np.hypot(*e)) for e in edges]
    i = int(np.argmax(lens))
    return edges[i] / max(lens[i], 1e-12), lens[i], np.asarray(mrr.centroid.coords[0])


def split_to_boxes(poly, ratio_thresh: float, max_parts: int) -> list:
    """Approximate a footprint by one or more oriented rectangles.

    A single OBB around an L-shaped or curved block covers the courtyard and
    often the street beside it -- measured, plain OBBs over-approximate built
    area by +37 % and swallow road network. Cutting the polygon in half across
    its long axis and recursing wherever the fit is still poor brings that to
    +21 % for 1.21x the box count, which is the right trade when Block C's cost
    is dominated by `M`.
    """
    out, stack = [], [(poly, max_parts)]
    while stack:
        p, budget = stack.pop()
        if p.is_empty or p.area < MIN_PART_AREA_M2:
            continue
        mrr = p.minimum_rotated_rectangle
        if mrr.is_empty or mrr.geom_type != "Polygon":
            continue
        if budget <= 1 or mrr.area / p.area <= ratio_thresh:
            out.append(mrr)
            continue
        axis, length, centre = _long_axis(mrr)
        normal = np.array([-axis[1], axis[0]])
        cut = LineString([centre - normal * length * 2.0, centre + normal * length * 2.0])
        try:
            pieces = list(shapely_split(p, cut).geoms)
        except Exception:  # noqa: BLE001 - degenerate geometry, keep the whole box
            pieces = []
        if len(pieces) < 2:
            out.append(mrr)
            continue
        for q in pieces:
            stack.append((q, max(budget // 2, 1)))
    return out


def build_buildings(gdf, bridges, ox_, oy_) -> tuple[np.ndarray, np.ndarray]:
    g = gdf.to_crs(UTM32N)
    g = g[g["height"].notna() & (g["height"] >= MIN_PART_HEIGHT_M)]
    half = BOX_SIZE_M / 2.0
    sq = shapely_box(ox_ - half, oy_ - half, ox_ + half, oy_ + half)
    g = g[g.geometry.intersects(sq)].copy()
    # clip so a building straddling the border contributes only its in-box part
    g["geometry"] = g.geometry.intersection(sq)
    g = g[g.geometry.area >= MIN_PART_AREA_M2]

    boxes, heights = [], []
    n_bridge = 0
    for geom, h in zip(g.geometry, g["height"], strict=True):
        parts = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
        for p in parts:
            if p.area < MIN_PART_AREA_M2:
                continue
            # LoD2 ships bridge decks as buildings. Keeping them puts a solid
            # slab along the road surface: the Untermainbruecke alone accounted
            # for 63 % of all HVT-route-inside-a-building events.
            if bridges is not None and p.intersection(bridges).area / p.area > BRIDGE_OVERLAP:
                n_bridge += 1
                continue
            for mrr in split_to_boxes(p, OBB_SPLIT_RATIO, OBB_MAX_PARTS):
                ob = oriented_box(mrr)
                if ob is None:
                    continue
                cx, cy, hw, hh, ca, sa = ob
                if hh < MIN_PART_HALF_M:
                    continue
                boxes.append((cx - ox_, cy - oy_, hw, hh, ca, sa))
                heights.append(float(h))

    print(f"  dropped {n_bridge} bridge-deck parts")
    return np.asarray(boxes, dtype=np.float32), np.asarray(heights, dtype=np.float32)


def build_height_grid(boxes: np.ndarray, heights: np.ndarray) -> np.ndarray:
    """Max building height per 20 m cell.

    Rasterised from the oriented boxes by testing cell centres, which is what
    the runtime raster would see. A cell whose centre falls in no building is 0.
    """
    half = BOX_SIZE_M / 2.0
    centres = (np.arange(GRID_N) + 0.5) * GRID_CELL_M - half
    gx, gy = np.meshgrid(centres, centres, indexing="xy")
    grid = np.zeros((GRID_N, GRID_N), dtype=np.float32)

    # process box-by-box; M ~ 4400 and the grid is small, so this is cheap
    for (cx, cy, hw, hh, ca, sa), h in zip(boxes, heights, strict=True):
        dx, dy = gx - cx, gy - cy
        # rotate the cell centre into the box frame -- same transform the
        # runtime occlusion test uses
        lx = dx * ca + dy * sa
        ly = -dx * sa + dy * ca
        hit = (np.abs(lx) <= hw) & (np.abs(ly) <= hh)
        np.maximum(grid, np.where(hit, h, 0.0), out=grid)
    return grid


# --------------------------------------------------------------------------
# roads
# --------------------------------------------------------------------------


def edge_speed(d: dict) -> float | None:
    hw = d.get("highway")
    hw = {hw} if isinstance(hw, str) else set(hw or [])
    if hw & EXCLUDED_CLASSES:
        return None
    speeds = [ROUTE_SPEEDS[c] for c in hw if c in ROUTE_SPEEDS]
    if not speeds:
        return None
    return min(min(speeds), SPEED_CAP_MS)


def build_roads(g: nx.MultiDiGraph, ox_, oy_):
    """Clip the drivable graph to the box and return arrays + a networkx view."""
    gp = ox.projection.project_graph(g, to_crs=UTM32N)
    half = BOX_SIZE_M / 2.0

    keep = [(u, v, k) for u, v, k, d in gp.edges(keys=True, data=True) if edge_speed(d)]
    H = gp.edge_subgraph(keep).to_undirected()

    inside = [
        n for n, d in H.nodes(data=True) if abs(d["x"] - ox_) <= half and abs(d["y"] - oy_) <= half
    ]
    sub = H.subgraph(inside)
    # keep only the largest component -- a route sampled in an island could
    # never reach the rest of the map
    big = sub.subgraph(max(nx.connected_components(sub), key=len)).copy()

    node_ids = sorted(big.nodes)
    idx = {n: i for i, n in enumerate(node_ids)}
    nodes = np.array(
        [[big.nodes[n]["x"] - ox_, big.nodes[n]["y"] - oy_] for n in node_ids],
        dtype=np.float32,
    )

    edges, speeds, route_ok = [], [], []
    G = nx.Graph()
    G.add_nodes_from(range(len(node_ids)))
    for u, v, d in big.edges(data=True):
        s = edge_speed(d)
        if s is None:
            continue
        iu, iv = idx[u], idx[v]

        # osmnx simplifies away interstitial nodes and keeps the real shape on
        # the edge. Walking node-to-node instead would cut corners straight
        # through blocks, putting the HVT inside buildings.
        geom = d.get("geometry")
        if geom is not None and hasattr(geom, "coords"):
            pts = np.asarray(geom.coords, dtype=np.float64)[:, :2]
            pts = pts - np.array([ox_, oy_])
            # orient u -> v
            if np.hypot(*(pts[0] - nodes[iu])) > np.hypot(*(pts[-1] - nodes[iu])):
                pts = pts[::-1]
        else:
            pts = np.array([nodes[iu], nodes[iv]], dtype=np.float64)

        length = float(d.get("length", 0.0)) or float(
            np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
        )
        edges.append((iu, iv, pts))
        speeds.append(s)
        route_ok.append(True)
        if not G.has_edge(iu, iv) or G[iu][iv]["length"] > length:
            G.add_edge(iu, iv, length=length, speed=s, time=length / s, pts=pts)

    dense_nodes, dense_edges, dense_speeds, dense_ok = densify(nodes, edges, speeds, route_ok)
    # `G` and `nodes` share the simplified indexing that route sampling needs;
    # the dense arrays are what gets stored, and carry no index relationship to G.
    return dense_nodes, dense_edges, dense_speeds, dense_ok, G, nodes


def densify(nodes, edges, speeds, route_ok):
    """Expand each edge's polyline into extra nodes and short straight segments.

    The artefact stores only `(K,2)` nodes and `(E,2)` index pairs, so an edge is
    implicitly a straight line. osmnx removes interstitial nodes, which would
    make those lines cut through blocks -- wrong for the renderer and for
    anything measuring sightlines down a street. Densifying keeps the flat array
    format while making every stored segment follow the real road.
    """
    out_nodes = [np.asarray(n, dtype=np.float64) for n in nodes]
    out_edges, out_speeds, out_ok = [], [], []

    for (iu, iv, pts), s, ok in zip(edges, speeds, route_ok, strict=True):
        interior = pts[1:-1]
        if len(interior) == 0:
            out_edges.append((iu, iv))
            out_speeds.append(s)
            out_ok.append(ok)
            continue
        chain = [iu]
        for p in interior:
            out_nodes.append(np.asarray(p, dtype=np.float64))
            chain.append(len(out_nodes) - 1)
        chain.append(iv)
        for a, b in itertools.pairwise(chain):
            out_edges.append((a, b))
            out_speeds.append(s)
            out_ok.append(ok)

    return (
        np.asarray(out_nodes, dtype=np.float32),
        np.asarray(out_edges, dtype=np.int32),
        np.asarray(out_speeds, dtype=np.float32),
        np.asarray(out_ok, dtype=bool),
    )


# --------------------------------------------------------------------------
# route bank
# --------------------------------------------------------------------------


def inside_any_box(pts: np.ndarray, boxes: np.ndarray, chunk: int = 600) -> np.ndarray:
    """(P,) bool — is each 2D point inside any oriented box footprint?

    Same transform the runtime occlusion test uses, so "inside" means the same
    thing here as it will there.
    """
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


def sample_routes(nodes: np.ndarray, G: nx.Graph, n_routes: int, seed: int, boxes=None):
    """Pre-sample HVT routes as (mcv_xy, trajectory) pairs.

    Sampling offline means `reset()` only has to index a tensor: no graph search
    in the hot loop. Each route satisfies the five conditions in BLOCK_B.md --
    start 300-500 m from the MCV, move outward, stay in the box by construction,
    last the full episode, and be reproducible from `seed`.
    """
    rng = np.random.default_rng(seed)
    d_all = np.linalg.norm(nodes[:, None] - nodes[None, :], axis=-1)

    # Only MCV positions from which the escalation is geometrically reachable.
    ok = d_all.max(axis=1) >= MCV_MIN_REACH_M
    # ...and that are not inside a building. The MCV never moves, so a spawn
    # inside a footprint means every one of its links is dead for the whole
    # episode -- strictly worse than the moving HVT briefly passing through one.
    if boxes is not None:
        ok &= ~inside_any_box(nodes.astype(np.float64), boxes)
    mcv_pool = np.flatnonzero(ok)
    if not len(mcv_pool):
        raise RuntimeError("no MCV position can reach MCV_MIN_REACH_M in this box")

    mcvs, trajs = [], []
    attempts = 0
    max_attempts = n_routes * 60

    while len(mcvs) < n_routes and attempts < max_attempts:
        attempts += 1
        m = int(rng.choice(mcv_pool))
        d_from_m = d_all[m]

        starts = np.flatnonzero((d_from_m >= HVT_START_MIN_M) & (d_from_m <= HVT_START_MAX_M))
        if not len(starts):
            continue

        s = int(rng.choice(starts))
        path = grow_outward(nodes, G, m, s, rng)
        if path is None:
            continue

        traj = walk_path(nodes, G, path)
        if traj is None:
            continue
        # outward: the HVT must end materially farther from the MCV than it began
        d0 = float(np.linalg.norm(traj[0] - nodes[m]))
        d1 = float(np.linalg.norm(traj[-1] - nodes[m]))
        if d1 - d0 < 500.0:
            continue

        # A road genuinely running under a podium or through an arcade is real,
        # and brief unobservability is legitimate difficulty -- it is the handoff
        # pressure RQ3 studies. Half an episode underneath one is not.
        if boxes is not None:
            frac = float(inside_any_box(traj.astype(np.float64), boxes).mean())
            if frac > MAX_ROUTE_INSIDE_FRAC:
                continue

        mcvs.append(nodes[m])
        trajs.append(traj)

    if len(mcvs) < n_routes:
        print(f"  [warn] only {len(mcvs)}/{n_routes} routes after {attempts} attempts")
    return (
        np.asarray(mcvs, dtype=np.float32),
        np.asarray(trajs, dtype=np.float32),
    )


def grow_outward(
    nodes: np.ndarray, G: nx.Graph, mcv: int, start: int, rng: np.random.Generator
) -> list[int] | None:
    """Grow a route that keeps moving away from the MCV for a full episode.

    Shortest-pathing to a distant target does not work: a 1200 m path takes far
    less than 240 s, so the only routes surviving the duration filter are ones
    that wander -- and wandering routes do not escalate the hop count. Measured,
    that gave a median separation of 1180 m at t=240 s against a 1400 m target,
    and *raising* the speed made the mid-episode profile worse, not better.

    Growing the path instead makes outwardness a property of construction: at
    each junction prefer neighbours that increase the distance from the MCV,
    weighted by how much, and stop once the episode's worth of travel time is
    banked.
    """
    need_t = EPISODE_STEPS * DT_S
    mcv_xy = nodes[mcv]
    path = [start]
    t = 0.0
    prev = None

    while t < need_t:
        cur = path[-1]
        nbrs = [n for n in G.neighbors(cur) if n != prev]
        if not nbrs:
            nbrs = list(G.neighbors(cur))  # dead end: allow the U-turn
            if not nbrs:
                return None

        d_cur = float(np.linalg.norm(nodes[cur] - mcv_xy))
        gain = np.array([float(np.linalg.norm(nodes[n] - mcv_xy)) - d_cur for n in nbrs])

        outward = np.flatnonzero(gain > 0)
        if len(outward):
            cand = [nbrs[i] for i in outward]
            w = gain[outward] ** OUTWARD_BIAS
        else:
            # nothing leads outward from here -- take any onward edge rather
            # than abandon the route, and let the outwardness filter judge it
            cand = nbrs
            w = np.ones(len(nbrs))

        nxt = int(rng.choice(cand, p=w / w.sum()))
        t += G[cur][nxt]["length"] / (G[cur][nxt]["speed"] * CONGESTION_FACTOR)
        prev = cur
        path.append(nxt)

        if len(path) > MAX_ROUTE_NODES:
            return None

    return path


def walk_path(nodes: np.ndarray, G: nx.Graph, path: list[int]) -> np.ndarray | None:
    """Resample a node path at DT_S intervals for EPISODE_STEPS steps.

    Follows each edge's stored geometry, not the straight line between its
    endpoints: osmnx removes interstitial nodes, so chords would cut corners
    through whole blocks and drive the HVT through buildings.

    Speeds are the class speed scaled by CONGESTION_FACTOR. If the path runs out
    before the episode does, the route is rejected -- an HVT that stops moving
    would make the tail of the episode trivially easy.
    """
    chunks, spd_chunks = [], []
    for u, v in itertools.pairwise(path):
        d = G[u][v]
        pts = np.asarray(d["pts"], dtype=np.float64)
        # each stored polyline runs u -> v; flip if this traversal is v -> u
        if np.hypot(*(pts[0] - nodes[u])) > np.hypot(*(pts[-1] - nodes[u])):
            pts = pts[::-1]
        if chunks:
            pts = pts[1:]  # drop the duplicated shared node
        if len(pts) == 0:
            continue
        chunks.append(pts)
        spd_chunks.append(np.full(len(pts), d["speed"] * CONGESTION_FACTOR))

    if not chunks:
        return None
    pts = np.concatenate(chunks, axis=0)
    if len(pts) < 2:
        return None
    spd = np.concatenate(spd_chunks)[1:]  # speed applies to each segment

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    seg_t = seg / np.maximum(spd, 1e-6)
    total_t = seg_t.sum()
    need_t = EPISODE_STEPS * DT_S
    if total_t < need_t:
        return None

    cum_t = np.concatenate([[0.0], np.cumsum(seg_t)])
    want = np.arange(EPISODE_STEPS) * DT_S
    i = np.clip(np.searchsorted(cum_t, want, side="right") - 1, 0, len(seg) - 1)
    frac = (want - cum_t[i]) / np.maximum(seg_t[i], 1e-9)
    return pts[i] + (pts[i + 1] - pts[i]) * frac[:, None]


# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--routes", type=int, default=N_ROUTES)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    ox_, oy_ = local_origin_utm()
    print(f"origin {ORIGIN_LATLON} -> UTM32N ({ox_:.1f}, {oy_:.1f})")

    boxes, heights = build_buildings(
        fetch_buildings(args.refresh), fetch_bridges(args.refresh), ox_, oy_
    )
    print(f"buildings: {len(boxes)} oriented boxes")

    grid = build_height_grid(boxes, heights)
    print(
        f"height grid: {grid.shape}, max {grid.max():.1f} m, {100 * (grid > 0).mean():.0f}% filled"
    )

    nodes, edges, speeds, route_ok, G, junctions = build_roads(fetch_roads(args.refresh), ox_, oy_)
    print(
        f"roads: {len(junctions)} junctions -> {len(nodes)} densified nodes, "
        f"{len(edges)} segments, connected={nx.is_connected(G)}"
    )

    print(f"sampling {args.routes} routes ...")
    route_mcv, route_xy = sample_routes(junctions, G, args.routes, args.seed, boxes)
    print(f"routes: {len(route_mcv)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        building_boxes=boxes,
        building_heights=heights,
        height_grid=grid,
        road_nodes=nodes,
        road_edges=edges,
        road_speeds=speeds,
        road_route_ok=route_ok,
        route_mcv=route_mcv,
        route_xy=route_xy,
        origin_lonlat=np.array([ORIGIN_LATLON[1], ORIGIN_LATLON[0]], dtype=np.float64),
        box_size_m=np.float32(BOX_SIZE_M),
        congestion_factor=np.float32(CONGESTION_FACTOR),
    )
    print(f"\nwrote {OUT}  ({OUT.stat().st_size / 1e6:.1f} MB)")

    if args.plot:
        plot_artefact(boxes, heights, nodes, edges, route_mcv, route_xy)


def plot_artefact(boxes, heights, nodes, edges, route_mcv, route_xy) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPoly

    fig, ax = plt.subplots(figsize=(10, 10))
    half = BOX_SIZE_M / 2.0

    for (cx, cy, hw, hh, ca, sa), h in zip(boxes, heights, strict=True):
        R = np.array([[ca, -sa], [sa, ca]])
        corners = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]]) @ R.T + [cx, cy]
        col = "#c0392b" if h >= 100 else ("#8a8a8a" if h >= 40 else "#cccccc")
        ax.add_patch(MplPoly(corners, closed=True, facecolor=col, edgecolor="none"))

    for a, b in edges:
        ax.plot(*zip(nodes[a], nodes[b], strict=True), color="#2c3e50", lw=0.5, alpha=0.6)

    for i in range(min(3, len(route_xy))):
        ax.plot(route_xy[i][:, 0], route_xy[i][:, 1], lw=2.5, alpha=0.9)
        ax.plot(*route_mcv[i], "k*", ms=16)
        ax.plot(*route_xy[i][0], "o", ms=8, color="w", mec="k")

    ax.set_xlim(-half, half)
    ax.set_ylim(-half, half)
    ax.set_aspect("equal")
    ax.set_title(
        "frankfurt_box.npz — oriented building boxes, road graph, 3 sample routes\n"
        "star = MCV, white dot = HVT start, red = towers >100 m",
        fontsize=10,
    )
    out = CACHE / "artefact.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"figure -> {out}")


if __name__ == "__main__":
    main()
