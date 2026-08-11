"""Block B gate: measure OSM building-height coverage for the Frankfurt box.

Answers one question before any pipeline code is written: does OSM carry usable
heights for the 1500 m box, or do we need the Hessen LoD2 CityGML models?

Reports, for each candidate box centre:
  * fraction of footprints with a usable height, raw and area-weighted
  * the same split by source (`height` tag vs `building:levels` vs nothing)
  * the height distribution, to confirm the box is actually heterogeneous
    (tower cluster + low fabric, see docs/DECISIONS.md -> Manhattan)

Offline diagnostic. Nothing here is imported by `src/`.

Usage:
    uv run python scripts/check_height_coverage.py
    uv run python scripts/check_height_coverage.py --lat 50.1130 --lon 8.6650
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import osmnx as ox

# Frankfurt is UTM 32N. Project once, work in metres thereafter.
UTM32N = "EPSG:25832"

# Metres per storey when only `building:levels` is tagged. German offices and
# residential blocks sit around 3.0-3.5 m floor-to-floor; 3.2 m is the midpoint.
# Stated here because it is a thesis-visible assumption (docs/BLOCK_B.md).
METRES_PER_LEVEL = 3.2

BOX_SIZE_M = 1500.0

# Candidate centres for the 1500 m box. All must contain both the tower cluster
# and low-rise fabric.
CANDIDATES = {
    "bankenviertel_core": (50.1109, 8.6721),
    "bankenviertel_west": (50.1130, 8.6650),
    "bankenviertel_mid": (50.1120, 8.6690),
}

CACHE = Path(__file__).resolve().parent.parent / ".cache" / "osm_height_probe"


# --------------------------------------------------------------------------
# tag parsing
# --------------------------------------------------------------------------

_NUM = re.compile(r"[-+]?\d*[.,]?\d+")


def parse_metres(value) -> float | None:
    """Parse an OSM height-like tag into metres.

    Handles '155', '155 m', '155.5', '112,5', "48'" (feet), and multi-values
    like '12;15' (takes the max, the conservative choice for an obstacle).
    Returns None if nothing sane can be extracted.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip().lower()
    if not text:
        return None

    best = None
    for part in text.split(";"):
        m = _NUM.search(part)
        if m is None:
            continue
        try:
            num = float(m.group().replace(",", "."))
        except ValueError:
            continue
        if "'" in part or "ft" in part or "feet" in part:
            num *= 0.3048
        if num <= 0 or num > 500:  # taller than any German building
            continue
        best = num if best is None else max(best, num)
    return best


def parse_levels(value) -> float | None:
    """Parse `building:levels` into a storey count."""
    lv = parse_metres(value)  # same numeric handling, no unit conversion needed
    if lv is None or lv <= 0 or lv > 120:
        return None
    return lv


# --------------------------------------------------------------------------
# fetch
# --------------------------------------------------------------------------


def fetch(lat: float, lon: float, name: str, refresh: bool = False) -> gpd.GeoDataFrame:
    """Download building footprints (incl. 3D building parts) for the box."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{name}.gpkg"
    if path.exists() and not refresh:
        return gpd.read_file(path)

    half = BOX_SIZE_M / 2.0
    # osmnx `dist` is a half-side for bbox_from_point, so this is a 1500 m box.
    parts = []
    for tags in ({"building": True}, {"building:part": True}):
        try:
            gdf = ox.features.features_from_point((lat, lon), tags=tags, dist=half)
        except Exception as exc:  # noqa: BLE001 - diagnostic script
            print(f"  [warn] query {tags} failed: {exc}")
            continue
        if len(gdf):
            gdf = gdf.copy()
            gdf["_query"] = "part" if "building:part" in tags else "building"
            parts.append(gdf)

    combined = gpd.GeoDataFrame(
        # keep all columns; the tag sets differ between the two queries
        gpd.pd.concat(parts, ignore_index=False, sort=False),
        crs=parts[0].crs,
    )
    combined = combined.reset_index()
    combined.to_file(path, driver="GPKG")
    return combined


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------


@dataclass
class Coverage:
    name: str
    centre: tuple[float, float]
    n_footprints: int
    total_area_m2: float
    # counts
    n_height: int
    n_levels_only: int
    n_none: int
    # area
    a_height: float
    a_levels_only: float
    a_none: float
    # heterogeneity
    heights: np.ndarray
    tallest: list[tuple[str, float, float]]

    @property
    def raw_cov(self) -> float:
        return (self.n_height + self.n_levels_only) / max(self.n_footprints, 1)

    @property
    def area_cov(self) -> float:
        return (self.a_height + self.a_levels_only) / max(self.total_area_m2, 1e-9)


def analyse(gdf: gpd.GeoDataFrame, name: str, centre: tuple[float, float]) -> Coverage:
    g = gdf.to_crs(UTM32N)

    # Clip to the actual square box; osmnx returns everything intersecting the
    # bbox, and a partially-outside footprint would skew the area weighting.
    cx, cy = (
        gpd.GeoSeries([gpd.points_from_xy([centre[1]], [centre[0]])[0]], crs="EPSG:4326")
        .to_crs(UTM32N)
        .iloc[0]
        .coords[0]
    )
    half = BOX_SIZE_M / 2.0
    g = g.cx[cx - half : cx + half, cy - half : cy + half]

    # Only real footprints. Points/lines tagged `building` are mapping noise.
    g = g[g.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()

    # Building *parts* are 3D refinements nested inside a footprint. Counting
    # them as separate obstacles double-counts area, so they are excluded from
    # the coverage statistic -- but their heights are harvested onto the parent
    # below, because on towers the height often lives only on the part.
    parts = g[g["_query"] == "part"].copy()
    g = g[g["_query"] == "building"].copy()

    def best_height(row) -> tuple[float | None, str]:
        h = parse_metres(row.get("height"))
        if h is None:
            h = parse_metres(row.get("building:height"))
        if h is not None:
            return h, "height"
        lv = parse_levels(row.get("building:levels"))
        if lv is not None:
            return lv * METRES_PER_LEVEL, "levels"
        return None, "none"

    g["_h"], g["_src"] = zip(*g.apply(best_height, axis=1), strict=True)

    # Harvest part heights onto untagged parents: a part sitting inside an
    # untagged footprint tells us how tall that footprint is.
    if len(parts):
        parts["_h"], _ = zip(*parts.apply(best_height, axis=1), strict=True)
        parts = parts[parts["_h"].notna()]
        if len(parts):
            untagged = g[g["_h"].isna()]
            if len(untagged):
                joined = gpd.sjoin(
                    parts[["geometry", "_h"]],
                    untagged[["geometry"]].reset_index(names="_parent"),
                    how="inner",
                    predicate="intersects",
                )
                if len(joined):
                    maxh = joined.groupby("_parent")["_h"].max()
                    g.loc[maxh.index, "_h"] = maxh.values
                    g.loc[maxh.index, "_src"] = "part"

    g["_area"] = g.geometry.area
    total_area = float(g["_area"].sum())

    has_h = g["_src"].isin(["height", "part"])
    only_lv = g["_src"] == "levels"
    none = g["_h"].isna()

    heights = g.loc[~none, "_h"].to_numpy(dtype=float)

    name_col = g["name"] if "name" in g.columns else gpd.pd.Series(index=g.index, dtype=object)
    tall = g.loc[~none].nlargest(8, "_h")
    tallest = [
        (str(name_col.get(i, "") or "(unnamed)"), float(r["_h"]), float(r["_area"]))
        for i, r in tall.iterrows()
    ]

    return Coverage(
        name=name,
        centre=centre,
        n_footprints=len(g),
        total_area_m2=total_area,
        n_height=int(has_h.sum()),
        n_levels_only=int(only_lv.sum()),
        n_none=int(none.sum()),
        a_height=float(g.loc[has_h, "_area"].sum()),
        a_levels_only=float(g.loc[only_lv, "_area"].sum()),
        a_none=float(g.loc[none, "_area"].sum()),
        heights=heights,
        tallest=tallest,
    )


def report(c: Coverage) -> None:
    pct = lambda x: f"{100 * x:5.1f}%"
    print(f"\n=== {c.name}  ({c.centre[0]:.4f}, {c.centre[1]:.4f})  {BOX_SIZE_M:.0f} m box ===")
    print(f"  footprints: {c.n_footprints}   built area: {c.total_area_m2 / 1e6:.3f} km²")
    print(f"  {'source':<16}{'count':>8}{'  ':2}{'of n':>7}{'area m²':>12}{'of area':>9}")
    rows = [
        ("height tag", c.n_height, c.a_height),
        ("levels only", c.n_levels_only, c.a_levels_only),
        ("no height", c.n_none, c.a_none),
    ]
    for label, n, a in rows:
        print(
            f"  {label:<16}{n:>8}  {pct(n / max(c.n_footprints, 1)):>7}"
            f"{a:>12,.0f}{pct(a / max(c.total_area_m2, 1e-9)):>9}"
        )
    print(f"  --> raw coverage          {pct(c.raw_cov)}")
    print(f"  --> AREA-WEIGHTED coverage{pct(c.area_cov)}   <-- the number that decides")

    h = c.heights
    if len(h):
        qs = np.percentile(h, [10, 25, 50, 75, 90, 99])
        print(
            "  height dist (m): p10 {:.1f}  p25 {:.1f}  p50 {:.1f}  p75 {:.1f}  p90 {:.1f}  p99 {:.1f}  max {:.1f}".format(
                *qs, h.max()
            )
        )
        print(
            f"  regimes: <15 m {np.mean(h < 15) * 100:.0f}%   "
            f"15-40 m {np.mean((h >= 15) & (h < 40)) * 100:.0f}%   "
            f">=100 m {np.mean(h >= 100) * 100:.0f}%  ({int(np.sum(h >= 100))} towers)"
        )
    print("  tallest tagged:")
    for nm, hh, ar in c.tallest:
        print(f"    {hh:6.1f} m  {ar:8,.0f} m²  {nm}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, default=None)
    ap.add_argument("--lon", type=float, default=None)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    if args.lat is not None and args.lon is not None:
        cands = {"custom": (args.lat, args.lon)}
    else:
        cands = CANDIDATES

    out = []
    for name, (lat, lon) in cands.items():
        gdf = fetch(lat, lon, name, refresh=args.refresh)
        cov = analyse(gdf, name, (lat, lon))
        report(cov)
        out.append(
            {
                "name": cov.name,
                "centre": cov.centre,
                "n_footprints": cov.n_footprints,
                "raw_coverage": cov.raw_cov,
                "area_weighted_coverage": cov.area_cov,
                "metres_per_level": METRES_PER_LEVEL,
            }
        )

    print(
        "\nDecision rule (docs/BLOCK_B.md): if area-weighted coverage is poor, go to "
        "Hessen LoD2. Heights derived from `building:levels` carry the "
        f"{METRES_PER_LEVEL} m/storey assumption and must be flagged in the thesis."
    )
    if args.json:
        args.json.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
