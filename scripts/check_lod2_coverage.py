"""Block B gate, part 2: does Hessen LoD2 cover the Frankfurt box?

`check_height_coverage.py` shows OSM fails the gate. This script checks the
fallback source before committing to it: pull every LoD2 building in the same
1500 m box from the Hessen INSPIRE WFS and measure

  * how many footprints carry a measured height (expected: all of them)
  * built area vs the OSM answer, i.e. whether LoD2 also finds the footprints
    OSM has but leaves untagged
  * the height distribution, to confirm the heterogeneity the scenario needs

The service returns INSPIRE `bu-core3d:Building`: a 2D footprint MultiSurface
plus `bu-base:heightAboveGround`. That is precisely the `(M,4)` boxes and
`(M,)` heights docs/BLOCK_B.md asks for -- the LoD2 roof solids are not needed.

Licence: Datenlizenz Deutschland Zero 2.0 (attribution-free, but cite anyway).

Offline diagnostic. Nothing here is imported by `src/`.

Usage:
    uv run python scripts/check_lod2_coverage.py
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from xml.etree import ElementTree as ET

import geopandas as gpd
import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

WFS = "https://inspire-hessen.de/ows/services/org.2.ef07833e-78a6-4c2c-a895-e31de788aac3_wfs"
TYPENAME = "bu-core3d:Building"
PAGE = 1000

NS = {
    "wfs": "http://www.opengis.net/wfs/2.0",
    "gml": "http://www.opengis.net/gml/3.2",
    "bu-core3d": "http://inspire.ec.europa.eu/schemas/bu-core3d/4.0",
    "bu-base": "http://inspire.ec.europa.eu/schemas/bu-base/4.0",
}

UTM32N = "EPSG:25832"
BOX_SIZE_M = 1500.0
CENTRE = (50.1109, 8.6721)

CACHE = Path(__file__).resolve().parent.parent / ".cache" / "lod2_probe"


def bbox_deg(lat: float, lon: float, size_m: float) -> tuple[float, float, float, float]:
    """Lat/lon bbox for a metric square. Good enough for a probe at this scale."""
    half = size_m / 2.0
    dlat = half / 111_320.0
    dlon = half / (111_320.0 * math.cos(math.radians(lat)))
    return (lat - dlat, lon - dlon, lat + dlat, lon + dlon)


def fetch_page(bbox: tuple[float, float, float, float], start: int) -> bytes:
    import urllib.parse
    import urllib.request

    q = urllib.parse.urlencode(
        {
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "typeNames": TYPENAME,
            "count": PAGE,
            "startIndex": start,
            # urn CRS => lat,lon axis order
            "bbox": "{:.6f},{:.6f},{:.6f},{:.6f},urn:ogc:def:crs:EPSG::4326".format(*bbox),
        }
    )
    req = urllib.request.Request(f"{WFS}?{q}", headers={"User-Agent": "uav-swarm-marl/0.1"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()


def parse_members(xml: bytes) -> tuple[list[tuple[Polygon | MultiPolygon, float | None]], int]:
    """Return (parsed buildings, number of wfs:member elements the server sent).

    The two counts differ when a member fails to yield usable geometry, so
    paging must key off the second one -- otherwise a few unparseable features
    make a full page look short and the loop stops early.

    `geometryMultiSurface` holds *every* surface of the LoD2 solid (ground,
    walls, roof) projected to 2D, so a wall face degenerates to a sliver. The
    building outline is their union, not the first one.
    """
    root = ET.fromstring(xml)
    members = root.findall("wfs:member", NS)
    out = []
    for member in members:
        b = member.find("bu-core3d:Building", NS)
        if b is None:
            continue

        h = None
        hv = b.find(".//bu-base:HeightAboveGround/bu-base:value", NS)
        if hv is not None and hv.text:
            try:
                h = float(hv.text)
            except ValueError:
                h = None

        polys = []
        for poly in b.findall(".//bu-core3d:geometryMultiSurface//gml:Polygon", NS):
            ext = poly.find(".//gml:exterior//gml:posList", NS)
            if ext is None or not ext.text:
                continue
            vals = [float(v) for v in ext.text.split()]
            # service emits lat lon pairs; shapely wants (x=lon, y=lat)
            ring = [(vals[i + 1], vals[i]) for i in range(0, len(vals) - 1, 2)]
            if len(ring) >= 4:
                p = Polygon(ring)
                if not p.is_valid:
                    p = p.buffer(0)
                if not p.is_empty and p.area > 0:
                    polys.append(p)
        if not polys:
            continue
        geom = polys[0] if len(polys) == 1 else unary_union(polys)
        if geom.is_empty or geom.area <= 0:
            continue
        out.append((geom, h))
    return out, len(members)


def load(refresh: bool = False) -> gpd.GeoDataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "lod2_frankfurt.gpkg"
    if path.exists() and not refresh:
        return gpd.read_file(path)

    bbox = bbox_deg(*CENTRE, BOX_SIZE_M)
    print(f"bbox (lat,lon): {bbox}")
    rows: list[tuple[object, float | None]] = []
    start = 0
    while True:
        print(f"  fetching startIndex={start} ...", flush=True)
        page, n_members = parse_members(fetch_page(bbox, start))
        rows.extend(page)
        print(f"    server returned {n_members}, parsed {len(page)} (total {len(rows)})")
        if n_members < PAGE:
            break
        start += PAGE
        time.sleep(1.0)  # be polite to a public service

    gdf = gpd.GeoDataFrame(
        {"height": [h for _, h in rows]},
        geometry=[g for g, _ in rows],
        crs="EPSG:4326",
    )
    gdf.to_file(path, driver="GPKG")
    return gdf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    gdf = load(refresh=args.refresh).to_crs(UTM32N)

    # clip to the true square, same as the OSM probe
    cx, cy = (
        gpd.GeoSeries(gpd.points_from_xy([CENTRE[1]], [CENTRE[0]]), crs="EPSG:4326")
        .to_crs(UTM32N)
        .iloc[0]
        .coords[0]
    )
    half = BOX_SIZE_M / 2.0
    gdf = gdf.cx[cx - half : cx + half, cy - half : cy + half].copy()
    gdf["area"] = gdf.geometry.area

    n = len(gdf)
    has = gdf["height"].notna() & (gdf["height"] > 0)
    total_area = float(gdf["area"].sum())

    print(f"\n=== Hessen LoD2, {BOX_SIZE_M:.0f} m box at {CENTRE} ===")
    print(f"  footprints:  {n}")
    print(f"  built area:  {total_area / 1e6:.3f} km²")
    print(f"  with measured height: {int(has.sum())}  ({100 * has.mean():.1f}%)")
    print(
        f"  area-weighted coverage: {100 * gdf.loc[has, 'area'].sum() / max(total_area, 1e-9):.1f}%"
    )

    h = gdf.loc[has, "height"].to_numpy(dtype=float)
    qs = np.percentile(h, [10, 25, 50, 75, 90, 99])
    print(
        "\n  height dist (m): p10 {:.1f}  p25 {:.1f}  p50 {:.1f}  p75 {:.1f}  p90 {:.1f}  p99 {:.1f}  max {:.1f}".format(
            *qs, h.max()
        )
    )
    print(
        f"  regimes: <15 m {np.mean(h < 15) * 100:.0f}%   "
        f"15-40 m {np.mean((h >= 15) & (h < 40)) * 100:.0f}%   "
        f">=100 m {np.mean(h >= 100) * 100:.0f}%  ({int(np.sum(h >= 100))} towers)"
    )
    print("\n  tallest:")
    for _, r in gdf.loc[has].nlargest(8, "height").iterrows():
        print(f"    {r['height']:6.1f} m  {r['area']:8,.0f} m²")


if __name__ == "__main__":
    main()
