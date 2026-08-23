"""Static scene drawing shared by the inspection and presentation front ends.

Extracted verbatim from `scripts/view_episode.py`; behaviour is unchanged. The
one rule this file exists to protect: **draw the oriented boxes the env
consumes, never the source polygons**, so what is on screen is what occlusion
tests against. `source_footprints` is the deliberate exception and is drawn as
an overlay, for judging the approximation rather than for trusting it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ARTEFACT = Path(__file__).resolve().parents[2] / "data" / "frankfurt_box.npz"

HALF_M = 750.0
TOWER_M = 100.0
MIDRISE_M = 40.0

# One palette, so a figure in the thesis and a video in wandb are the same
# picture. Chosen to stay legible in greyscale print.
COLOURS = {
    "tower": "#c0392b",
    "midrise": "#8a8a8a",
    "low": "#d8d8d8",
    "road": "#3498db",
    "road_muted": "#b8c6d1",  # context in a presentation figure, not content
    "route": "#f39c12",
    "trail": "#e67e22",
    "hvt": "#e74c3c",
    "drone": "#2980b9",
    "track": "#7d3c98",  # drone trails -- must not read as road, which is blue
    "chain": "#27ae60",
    "blocked": "#c0392b",
    "clear": "#27ae60",
    "observer": "#8e44ad",
    "source": "#16a085",
}


def load_artefact(path: Path = ARTEFACT) -> dict:
    """The frozen environment, as plain arrays. `data/frankfurt_box.npz` IS the
    environment (AGENTS.md), so both front ends read exactly this."""
    art = np.load(path)
    return {
        "boxes": art["building_boxes"].astype(np.float64),
        "heights": art["building_heights"].astype(np.float64),
        "nodes": art["road_nodes"].astype(np.float64),
        "edges": art["road_edges"],
        "mcvs": art["route_mcv"].astype(np.float64),
        "routes": art["route_xy"].astype(np.float64),
    }


def box_corners(b: np.ndarray) -> np.ndarray:
    """(M,6) oriented boxes -> (M,4,2) corner polygons."""
    cx, cy, hw, hh, ca, sa = b.T
    local = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
    out = np.empty((len(b), 4, 2))
    for i, (dx, dy) in enumerate(local):
        x, y = dx * hw, dy * hh
        out[:, i, 0] = cx + x * ca - y * sa
        out[:, i, 1] = cy + x * sa + y * ca
    return out


def inside_any_box(pts: np.ndarray, boxes: np.ndarray, chunk: int = 600) -> np.ndarray:
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


def source_footprints():
    """Original LoD2 polygons in local metres, for visual comparison only.

    Reads the offline cache, so this is the one place in the drawing code that
    needs `geopandas`. The env never sees polygons -- it only ever sees boxes.
    """
    import sys

    import geopandas as gpd

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from prep_osm import BOX_SIZE_M, UTM32N, local_origin_utm

    cache = Path(__file__).resolve().parents[2] / ".cache" / "prep_osm" / "lod2.gpkg"
    if not cache.exists():
        print(f"[warn] {cache} missing; run scripts/prep_osm.py --refresh")
        return
    ox_, oy_ = local_origin_utm()
    half = BOX_SIZE_M / 2.0
    g = gpd.read_file(cache).to_crs(UTM32N)
    g = g.cx[ox_ - half : ox_ + half, oy_ - half : oy_ + half]
    for geom in g.geometry:
        for part in geom.geoms if geom.geom_type == "MultiPolygon" else [geom]:
            xs, ys = part.exterior.coords.xy
            yield np.asarray(xs) - ox_, np.asarray(ys) - oy_


def draw_static_scene(
    ax, art: dict, roads: bool = True, polygons: bool = False, road_colour: str | None = None
) -> None:
    """Buildings (by height class) and the road graph. Everything else animates.

    `road_colour` exists because the road graph is *content* in the inspection
    tool (it is what the HVT drives on, and box-vs-road overlap is a real bug
    class) but *context* in a presentation figure, where drone tracks are the
    subject and share the same blue. The presentation renderer mutes it; the
    inspection tool keeps the default.
    """
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon as MplPoly

    heights = art["heights"]
    colours = np.where(
        heights >= TOWER_M,
        COLOURS["tower"],
        np.where(heights >= MIDRISE_M, COLOURS["midrise"], COLOURS["low"]),
    )
    ax.add_collection(
        PatchCollection(
            [MplPoly(c, closed=True) for c in box_corners(art["boxes"])],
            facecolors=colours,
            edgecolors="#00000018",
            linewidths=0.3,
        )
    )
    if polygons:
        for xs, ys in source_footprints():
            ax.plot(xs, ys, color=COLOURS["source"], lw=0.8, alpha=0.9, zorder=2.5)
        ax.plot([], [], color=COLOURS["source"], lw=0.8, label="source LoD2 footprint")
    if roads:
        nodes = art["nodes"]
        for a, b in art["edges"]:
            ax.plot(
                *zip(nodes[a], nodes[b], strict=True),
                color=road_colour or COLOURS["road"],
                lw=0.5,
                alpha=0.55,
                zorder=2,
            )
