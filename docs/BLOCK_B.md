# Block B — Frankfurt geometry pipeline

**Goal:** turn a real Frankfurt district into tensors the env can use at runtime,
offline and once, so nothing geospatial ever runs inside `step()`.

Produces `scripts/prep_osm.py` and a cached artefact. Everything downstream
(occlusion in Block C, the env in Block D) depends on this.

---

## The one thing to check before writing any code

**Building height coverage.** The whole scenario rests on heights — the canyon
ratio `H_b/W` sets the observation envelope, and the tower cluster is what blocks
air-to-air links. If heights are missing for most footprints, the map is useless.

Two candidate sources:

| Source | Coverage | Effort |
|---|---|---|
| **Hessen open LoD2 3D building models** | authoritative, every building | CityGML parsing — more work |
| **OSM `height` / `building:levels`** | patchy, city-dependent | `osmnx` one-liner |

**Measure the OSM coverage first** for the chosen box. Report: fraction of
footprints with usable height, and the same fraction weighted by footprint area
(a few large towers matter more than many small sheds). If area-weighted coverage
is poor, go to LoD2.

Fallback if both disappoint: a height prior from footprint area and land-use tag,
applied only to untagged buildings, and **flagged in the thesis as an
assumption**. Do not silently impute.

---

## What to produce

A single cached artefact (`.pt` or `.npz`) holding:

| Field | Shape | Notes |
|---|---|---|
| `building_boxes` | `(M, 4)` | axis-aligned `xmin, ymin, xmax, ymax` in local metres |
| `building_heights` | `(M,)` | metres above ground |
| `height_grid` | `(75, 75)` | max building height per 20 m cell, for the optional local raster |
| `road_nodes` | `(K, 2)` | local metric coordinates |
| `road_edges` | `(E, 2)` | node index pairs |
| `road_speeds` | `(E,)` | m/s from OSM class — see below |
| `origin_lonlat`, `box_size_m` | | provenance, so the projection is reproducible |

**Axis-aligned boxes, not polygons.** The runtime occlusion test is
segment-vs-AABB (slab method), which is branch-free and batches perfectly.
Concave or rotated footprints should be split into several boxes rather than kept
as polygons. Over-approximating a footprint slightly is acceptable; make the
choice explicit and note it.

**Local metric CRS.** Project once (UTM 32N for Frankfurt) and store metres with
a recorded origin. Never carry lat/lon into the env.

---

## Parameters already fixed

From [`AGENTS.md`](../AGENTS.md); do not re-derive:

- **1500 m box** over the Bankenviertel **plus surrounding low-rise fabric** —
  both regimes are needed. The towers block A2A; the low fabric is what makes the
  observation envelope workable. A box containing only towers reproduces the
  Manhattan failure documented in [`DECISIONS.md`](DECISIONS.md).
- Road speeds from OSM class: `residential` 8.3 m/s, `secondary` 13.9 m/s.
  **Exclude `primary`, `trunk`, `motorway` from route sampling** — the drone
  needs a 1.4–1.8× speed margin and loses it above 50 km/h.
- Grid resolution 20 m ⇒ 75×75 cells.

---

## Route sampling

The HVT route is **pre-sampled at reset** as a path on the in-box road graph, not
chosen randomly at junctions. Random turning doubles back, stalls in cul-de-sacs
and leaves the map; preventing that amounts to writing a route sampler by
accident. Pre-sampling gives the map border for free.

The sampler must produce a path that:
1. starts on a road **300–500 m** from the MCV,
2. **moves outward**, so the chain requirement escalates 1 → 2 → 3 hops,
3. stays inside the box by construction,
4. lasts ≥240 s at class speeds,
5. is reproducible from a seed.

Belongs in Block B (it is graph work) even though it is called at reset.

---

## Measure this while you are here

Two empirical questions the design deliberately left open:

1. **How often does a long straight sightline actually occur?** The 830 m
   recognition range only holds down a clear street, and Frankfurt's streets
   bend. Sample points on the road graph, cast rays along the street axis, report
   the distribution of unobstructed length. This sets real acquisition difficulty
   and belongs in the thesis.
2. **What is the true canyon ratio distribution?** `PHYSICS.md` assumes 20 m
   streets and 22 m fabric giving a 36 m across-street envelope. Measure it and
   report the spread rather than the single assumed value.

---

## Definition of done

- [ ] Height coverage measured and reported (raw and area-weighted); source chosen
- [ ] `scripts/prep_osm.py` runs offline and caches the artefact
- [ ] Loading the artefact needs **no** `osmnx`/`shapely` import
- [ ] Route sampler produces valid outward routes, reproducible from a seed
- [ ] `tests/test_osm_pipeline.py`: box bounds, no NaNs, heights positive, road
      graph connected, sampled routes satisfy all five conditions above
- [ ] A rendered figure of the box (footprints + road graph + a sample route) —
      you will want it for the thesis anyway, and it is the fastest way to see
      that the projection is right
- [ ] Sightline and canyon-ratio distributions measured

## Watch out for

- **Do not import `osmnx` or `shapely` anywhere under `src/`.** They are offline
  tools. The env must load a tensor and nothing else.
- OSM buildings come as polygons in lon/lat — project *before* computing boxes,
  or the metres are wrong by a latitude-dependent factor.
- `building:levels` needs a metres-per-level assumption (~3–3.5 m). State it.
- Some footprints are courtyards or building *parts*; check for duplicate and
  nested geometry before treating each as an obstacle.
