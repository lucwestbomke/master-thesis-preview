# Block B — Frankfurt geometry pipeline

**Goal:** turn a real Frankfurt district into tensors the env can use at runtime,
offline and once, so nothing geospatial ever runs inside `step()`.

Produces `scripts/prep_osm.py` and a cached artefact. Everything downstream
(occlusion in Block C, the env in Block D) depends on this.

---

## The height-coverage gate — RESOLVED: use Hessen LoD2

Measured, not assumed. Two probe scripts, both offline diagnostics:
`scripts/check_height_coverage.py` (OSM) and `scripts/check_lod2_coverage.py`
(LoD2). Both operate on the same 1500 m box.

| Source | footprints | raw cov. | **area-weighted cov.** |
|---|---|---|---|
| OSM `height` / `building:levels` | 1908 | 41 % | **58 %** ❌ |
| **Hessen LoD2 (INSPIRE WFS)** | 4789 | **100 %** | **100 %** ✅ |

**OSM fails.** Only 3–5 % of footprints have an explicit `height`; the rest of
its coverage rides on `building:levels` × 3.2 m. The missing 42 % of built area
is mid-rise, not sheds — Die Welle, the Börse, the Bundesbank HQ, Triton House —
i.e. exactly the 20–60 m blocks that set the canyon ratio. And its tags are
unreliable where they exist (Deutsche-Bank-Hochhaus tagged 22 m, actually ~155 m).

**LoD2 wins on every axis** and needs no CityGML parsing: the INSPIRE WFS serves
`bu-core3d:Building` as a 2D footprint plus a *measured* `bu-base:heightAboveGround`
— already the `(M,4)` + `(M,)` this block wants. Licence **dl-de/zero-2.0**.

```
WFS  https://inspire-hessen.de/ows/services/org.2.ef07833e-78a6-4c2c-a895-e31de788aac3_wfs
     typeNames=bu-core3d:Building, bbox in urn:ogc:def:crs:EPSG::4326 (lat,lon)
     page with startIndex; server caps at 1000 features per request
```

**Cross-validation.** On the 34 buildings where OSM carries an explicit height,
LoD2 − OSM has median **+0.2 m**, IQR **−2.5 … +3.5 m**. Every large
disagreement inspected resolves in LoD2's favour except church steeples
(Glockenturm: LoD2 24 m vs actual ~48 m), which LoD2 truncates to the roof.
Irrelevant here — a steeple is a thin obstacle and the fabric height is what
matters. LoD2 heights are to the roof structure and **exclude antenna masts**,
which is the correct convention for occlusion.

No imputation is needed anywhere. The height-prior fallback is dropped.

### What this changes downstream

LoD2 decomposes a building into stepped **parts** — 4789 features for ~1900 OSM
footprints; 67 features above 100 m resolve into **21 distinct towers**. Parts
tile without overlapping (union 0.894 km² vs sum-of-parts 0.901 km²).

- **Good:** the parts are already near-box-shaped, so "split concave footprints
  into several AABBs" is mostly done by the data. A stepped tower is also
  represented more faithfully than one extruded prism.
- **Watch:** 551 features are under 5 m² (degenerate LoD2 wall slivers, 0.0014 km²
  in total). Drop them on an area threshold.
- ⚠️ **Flag for Block C:** `M ≈ 5000` boxes, not the few hundred implied by OSM
  footprints. A naive all-links × all-boxes occlusion test is ~10⁸ segment–box
  tests per batched step and **will not meet Block D's 1000 steps/s gate**.
  Block C needs a broad phase — a uniform grid or per-segment 2D bbox cull —
  designed in from the start, not bolted on.

---

## What to produce

A single cached artefact (`.pt` or `.npz`) holding:

| Field | Shape | Notes |
|---|---|---|
| `building_boxes` | `(M, 6)` | **oriented** `cx, cy, half_w, half_h, cos θ, sin θ` in local metres |
| `building_heights` | `(M,)` | metres above ground |
| `height_grid` | `(75, 75)` | max building height per 20 m cell, for the optional local raster |
| `road_nodes` | `(K, 2)` | local metric coordinates |
| `road_edges` | `(E, 2)` | node index pairs |
| `road_speeds` | `(E,)` | m/s from OSM class — see below |
| `road_route_ok` | `(E,)` bool | may the HVT drive this edge? — see below |
| `origin_lonlat`, `box_size_m` | | provenance, so the projection is reproducible |

**Cap the HVT's speed; do not delete road classes.** The binding constraint is
that the drone keeps a 1.4–1.8× speed margin, which it holds up to 50 km/h. That
is a statement about **speed**, not about road class. So the HVT may use every
surface street, with `road_speeds` capped at **13.9 m/s**; `road_route_ok` exists
only to exclude grade-separated roads (`motorway`, `trunk` and their links),
which carry no urban canyon and would take the HVT out of the box in under a
minute. *None occur in the Frankfurt box* — the mask is a guard for the second
city.

The earlier reading — drop `primary` entirely — was wrong on three counts:

- **It stranded part of the map.** The residential streets by the Hauptbahnhof
  (Poststraße, Ottostraße, Niddastraße) reach the network only via Mainzer
  Landstraße and Am Hauptbahnhof, both `primary`. Removing those left an 11-node
  orphan the route sampler would have had to special-case.
- **It deleted the widest open corridors.** Reuterweg, Taunusanlage and Mainzer
  Landstraße trace the Wallanlagen belt straight past the tower cluster, and are
  exactly where long unobstructed sightlines occur. Dropping them would bias the
  sightline distribution this block is supposed to measure.
- **A vehicle that may not speed along an arterial can still cross it.** Deleting
  the edges risks severing the network at dual carriageways.

Measured effect of the fix on the chosen box: connectivity **95 % → 100 %** in
one component, drivable length **27.3 → 31.6 km**, MCV placements admitting the
1400 m escalation **64 % → 70 %**. Every candidate centre in the sweep now
reaches 100 % connectivity, so the orphan was an artefact of the exclusion rule
and not a property of the map.

Speeds by class, all capped at 13.9 m/s: `living_street` 5.6, `residential` and
`unclassified` 8.3, `tertiary`/`secondary`/`primary` 13.9. Multi-class edges take
the slowest match.

**Oriented boxes, not axis-aligned ones.** Measured on the chosen box, one AABB
per part is **not usable**. (The comparison below is over the 4351 parts passing
the 5 m² area filter alone. The shipped artefact holds 5120 after the height and
thickness filters *and* Block C's footprint splitting, which took
over-approximation further, to +21 %. The ratios below are unaffected.)

| | median ratio | p90 | total area | share of box filled |
|---|---|---|---|---|
| AABB (axis-aligned) | 1.90 | 3.20 | 2.109 km² (**+134 %**) | **94 %** |
| **OBB (oriented)** | **1.07** | 1.58 | 1.240 km² (+38 %) | 55 % |

At 94 % fill the city is effectively solid, every link is blocked, and occlusion
stops discriminating between fidelity rungs — which destroys RQ1. The cause is
that LoD2 parts *are* rectangles but **rotated**: only 18 % lie within 10° of
axis-aligned and the median long-axis orientation is 38°. An AABB around a
45°-rotated rectangle doubles its area, and 1.90 is exactly that signature.

The **slab method survives unchanged**. Segment-vs-OBB is: translate the segment
by `−(cx, cy)`, rotate it by `−θ`, then run the identical branch-free slab test
in the box's local frame. Cost is one 2×2 rotation per (segment, box) pair, and
`cos θ`/`sin θ` are baked in offline so no trig runs in the hot loop. `M` does
**not** grow — which matters, because splitting each part into enough AABBs to
match OBB fidelity would push `M` from 4351 to the high tens of thousands, and
Block C is already tight (see the flag above). OBB is better on fidelity *and*
cheaper.

Genuinely non-rectangular parts (p90 ratio 1.58) may still be split into two or
three OBBs. Do that only if Block C shows it matters; measure first.

**Local metric CRS.** Project once (UTM 32N for Frankfurt) and store metres with
a recorded origin. Never carry lat/lon into the env.

**Commit the artefact.** `data/frankfurt_box.npz` is 9.7 MB and is *not* treated
as a regenerable build product. OSM changes continuously and the LoD2 service
will be revised, so re-running `prep_osm.py` in 2027 would silently produce a
different map — which is exactly what the end-of-March environment freeze exists
to prevent. The file in git **is** the frozen environment; the script only
documents how it was made.

---

## Choosing the exact box — `scripts/choose_box.py`

Height coverage is 100 % everywhere in the district, so it cannot pick the
square. The box is chosen on **scenario geometry** instead: the script fetches
once over a superset area, then scores 169 candidate centres (100 m steps,
±600 m) offline on

1. **usable area** — the Main costs up to 6 % of some boxes;
2. **tower centrality** — distance from box centre to the >100 m cluster
   centroid. Weighted highest: the towers are RQ1's mechanism, and pushed to a
   corner they stop intercepting A2A links;
3. **MCV placements admitting the escalation** — fraction of drivable road nodes
   with a ≥1400 m-distant node in the same component (`ENVIRONMENT.md` puts the
   HVT 1400 m out at t=240 s and randomises the MCV per episode). Note this is
   the *node* extent, ~1476 m across, not the box: the outermost junctions sit a
   few metres inside the 1500 m edge;
4. **both regimes present** — a *filter*, not a score. A real city is ~93 %
   fabric by built area, so grading boxes on how close they get to an even split
   penalises all of them equally and discriminates nothing.

**FROZEN centre: `50.11200 N, 8.67040 E`** (2026-08-11). Local origin is the box
centre, so in-box coordinates run −750 … +750 m on both axes.

| | value |
|---|---|
| water in box | 3.9 % — clips the **SE corner**, does not cut the box |
| towers >100 m | 62 parts, centroid **48 m** from box centre |
| fabric | 93 % of built area |
| MCV placements admitting 1400 m escalation | 66 % of road nodes |
| drivable road | 31.3 km, **100 %** in one connected component |

Under the speed-cap policy the top two centres are tied within 0.001 —
`50.11110 N, 8.66901 E` scores marginally higher on MCV placements (70 % vs
66 %) but sits 64 m off the tower centroid against 48 m. They are 100 m apart and
describe practically the same box; the recommendation stays with the more central
one, since centrality is the criterion tied directly to RQ1's mechanism and the
MCV difference has no practical consequence.

The real trade-off in the table is against a western cluster near `8.6606`,
which has essentially no river (0.3 %) but puts the tower centroid 141–239 m off
centre with only 51 tower parts. **Centrality wins** — 4 % dead area in a corner
costs little, whereas off-centre towers weaken the effect RQ1 is trying to
measure. The Main also isn't purely dead: it gives long unobstructed sightlines,
which is signal for the sightline distribution below.

The per-criterion columns are the decision; `WEIGHTS` in the script is a ranking
convenience and re-weighting is a one-line change.

## Parameters already fixed

From [`AGENTS.md`](../AGENTS.md); do not re-derive:

- **1500 m box** over the Bankenviertel **plus surrounding low-rise fabric** —
  both regimes are needed. The towers block A2A; the low fabric is what makes the
  observation envelope workable. A box containing only towers reproduces the
  Manhattan failure documented in [`DECISIONS.md`](DECISIONS.md).
- Road speeds from OSM class: `residential` 8.3 m/s, `secondary` 13.9 m/s,
  **all capped at 13.9 m/s (50 km/h)** so the drone keeps its 1.4–1.8× margin.
  Only `motorway`/`trunk` are excluded outright; see "Cap the HVT's speed" above.
- Grid resolution 20 m ⇒ 75×75 cells.

---

## Route sampling — how it actually works

Routes are pre-sampled **offline** into a bank of 2048 `(MCV, trajectory)` pairs,
so `reset()` only indexes a tensor and no graph search ever runs in the hot loop.

Three things had to be got right, each of which failed first:

- **Grow the path outward; do not shortest-path to a distant target.** A 1200 m
  shortest path takes far less than the 240 s episode, so the only routes
  surviving a duration filter were ones that wandered — and wandering routes do
  not escalate the hop count. Measured: median separation 1180 m at t=240 s
  against a 1400 m target, and *raising* the speed made the mid-episode profile
  worse. Growing the path (at each junction, prefer neighbours that increase MCV
  distance, weighted by gain) makes outwardness a property of construction.
- **Walk the edge geometry, not node-to-node chords.** osmnx removes
  interstitial nodes; chords cut corners through blocks and put the HVT inside
  buildings. The stored graph is densified for the same reason.
- **Perturb the routing weights.** Pure shortest-time routing put *every*
  trajectory on the 13.9 m/s arterials — measured as a single constant speed
  across the whole bank, with residential streets never used.

Two calibrated constants, both scenario assumptions to state in the thesis:

| | value | why |
|---|---|---|
| `CONGESTION_FACTOR` | **0.70** | free-flow class speed is what a road *permits*; crossing a city centre averages less. Gives ~9.7 m/s on arterials, 5.8 m/s on residential |
| `MCV_MIN_REACH_M` | **1500 m** | only 66 % of road nodes have another node 1400 m away *at all*, so a uniform MCV caps the achievable separation below target |

Jointly calibrated against the `ENVIRONMENT.md` escalation table:

| t | 0 s | 60 s | 120 s | 240 s |
|---|---|---|---|---|
| table | 400 | 700 | 1000 | 1400 |
| achieved (median) | 404 | 712 | 1015 | **1334** |

The −66 m at t=240 s is a **box-size limit, not a tuning failure**: the farthest
road node from a median MCV position is 1490 m, so 1400 m of separation is at the
edge of what a 1500 m box supports. Worth revisiting if the 3-hop regime turns
out to be under-exercised in Block D.

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

### Measured — `scripts/measure_sightlines.py`

3383 road points every 25 m, a ray cast each way along the street axis at
vehicle height, 6766 rays. 16 % leave the box before hitting anything and are
treated as **right-censored** (a lower bound), not as measurements — counting
them at face value put 28 % of sightlines beyond 830 m, which was an artefact of
the data boundary rather than a fact about Frankfurt.

**Sightline along the street** (measured rays):

| p10 | p25 | p50 | p75 | p90 | p99 | max |
|---|---|---|---|---|---|---|
| 32 | 64 | **127 m** | 239 | 387 | 643 | 957 |

Counting censored rays at their lower bound gives p50 126 / p90 396 — the same
picture, so the conclusion does not rest on how censoring is handled.

- `ENVIRONMENT.md`'s guess of "100–400 m typical" holds for **51 %** of rays;
  another **40 % are below 100 m**, so the guess was if anything optimistic.
- **Only 0.2 % of sightlines exceed 830 m.** The sensor range therefore
  essentially never binds — see below.
- By class, medians track street openness: primary 190 m, secondary 151 m,
  residential 117 m, tertiary 112 m.

**Canyon ratio — `PHYSICS.md`'s assumptions hold up:**

| | p10 | p25 | p50 | p75 | p90 | assumed |
|---|---|---|---|---|---|---|
| street width `W` | 12 | 16 | **21 m** | 30 | 45 | 20 m |
| ratio `H_b/W` | 0.45 | 0.63 | **0.93** | 1.27 | 1.64 | 1.10 |
| across-street envelope @ 80 m | 24 | 32 | **43 m** | 63 | 88 | 36 m |

The measured city is marginally *more* open than assumed (median ratio 0.93 vs
1.10, envelope 43 m vs 36 m), and the spread is wide — a factor of ~3.7 between
p10 and p90 on the envelope. Report the distribution in the thesis, not the
single number.

### The 830 m recognition range

**It has no derivation anywhere in this repo.** It appears in `PHYSICS.md`,
`ENVIRONMENT.md` and `THESIS_PLAN.md` as a bare assertion with no sensor model,
no source, and no `TODO(verify)` marker — exactly the class of number
[`AGENTS.md`](../AGENTS.md) forbids citing.

The measurement makes that mostly moot: **99.8 % of street sightlines are
shorter than 830 m**, and the across-street envelope (43 m median) is shorter
still. Occlusion cuts first, everywhere. So the value should be documented as a
**non-binding ceiling**, not defended as a tuned parameter:

> Sensor recognition range is set to 830 m. Measured sightlines in the operating
> area exceed this in 0.2 % of cases, so the observation constraint is geometric
> throughout and results are insensitive to this value.

That is a stronger claim than any justification of 830 specifically, and it
protects RQ1: a *binding* sensor range would confound the fidelity ladder with
sensor specification. If a defensible number is wanted anyway, derive it from a
stated camera (Johnson criteria: ~4 line pairs across a ~2.3 m vehicle) rather
than assert it — and put the camera spec in the methodology chapter.

⚠️ This is a **2D ground-level** measure. A drone at 80 m sees a wedge down the
street plus an overhead cone; the true 3D envelope is Block C's job. The
along-street number bounds it, which is why 0.2 % is safe as a conclusion.

---

## Definition of done

- [x] Height coverage measured and reported (raw and area-weighted); source chosen
      → **Hessen LoD2 via INSPIRE WFS**, 100 % coverage. See the gate section above.
- [x] `scripts/prep_osm.py` runs offline and caches the artefact
      → `data/frankfurt_box.npz`, 9.1 MB: 5120 oriented boxes (46 % fill;
      4220 before Block C's bridge-and-split fix),
      1757 densified road nodes / 1882 segments, 2048 pre-sampled routes
- [x] Loading the artefact needs **no** `osmnx`/`shapely` import — `.npz`,
      and `tests/test_osm_pipeline.py` imports only NumPy to prove it
- [x] Route sampler produces valid outward routes, reproducible from a seed
- [x] `tests/test_osm_pipeline.py`: box bounds, no NaNs, heights positive, road
      graph connected, sampled routes satisfy all five conditions above — 23 tests
- [x] A rendered figure of the box (footprints + road graph + a sample route) —
      you will want it for the thesis anyway, and it is the fastest way to see
      that the projection is right
- [x] Sightline and canyon-ratio distributions measured →
      `scripts/measure_sightlines.py`; results in "Measure this while you are
      here" above. Sightline p50 **127 m**; canyon ratio p50 **0.93**;
      830 m sensor range shown to be non-binding (0.2 % of rays)

## Watch out for

- **Do not import `osmnx` or `shapely` anywhere under `src/`.** They are offline
  tools. The env must load a tensor and nothing else.
- Both sources hand back polygons in lat/lon — project *before* computing boxes,
  or the metres are wrong by a latitude-dependent factor. The LoD2 WFS emits
  **lat lon** order (urn-style CRS), not lon/lat; swapping them silently puts
  Frankfurt in the Indian Ocean.
- The LoD2 `geometryMultiSurface` holds *every* surface of the solid (ground,
  walls, roof) projected flat. Take the **union** of them for the outline —
  the first one is an arbitrary wall face and gives ~1 m² footprints on towers.
- Page the WFS on the count of `wfs:member` elements the **server** returned,
  not on how many you successfully parsed; a couple of unparseable features
  otherwise make a full page look short and the loop stops early.
- Some footprints are courtyards or building *parts*; check for duplicate and
  nested geometry before treating each as an obstacle.
