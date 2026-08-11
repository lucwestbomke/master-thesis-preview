# Block C — batched occlusion

**Goal:** decide, for every link in every parallel environment, whether the ray
is blocked by a building — fast enough that Block D's throughput gate survives.

This is the **F1 rung of RQ1**. Occlusion is the hypothesis: the claim is that it
is the effect a channel abstraction cannot afford to omit. So this module is not
a utility, it is the independent variable, and it has to be right *and* cheap.

Consumes `data/frankfurt_box.npz` from [`BLOCK_B.md`](BLOCK_B.md). Produces
`src/env/occlusion.py`.

---

## Fix this before writing any code

**35 building boxes swallow the road network.** Measured on the shipped artefact:

| | |
|---|---|
| HVT route points inside a building box | **6.34 %** |
| …inside a real LoD2 *polygon* (so not an OBB artefact) | **4.43 %** |
| routes with ≥1 point inside a box | 224 / 256 sampled |
| worst single route | **333 of 600 steps** inside one box |
| MCV spawn points inside a box | 4 / 121 |

A point inside an obstacle is blocked by construction, so those episodes are
unwinnable — the target can never be observed and the reward is unreachable
through no fault of the policy. That would quietly poison training and be
extremely hard to diagnose later.

**It is concentrated, so the fix is targeted, not systemic:**

| boxes | share of all intrusions |
|---|---|
| top 5 | **84.7 %** |
| top 20 | 97.9 % |
| all 35 | 100 % |

The worst offender is a **242 × 22 m box only 10.6 m tall** at local
`(+452, −615)`, which puts one route inside it for 333 consecutive steps. A
242 m long, 22 m wide, 10.6 m tall "building" running alongside a road near the
Main is a **bridge deck or canopy**, not a building.

Only ~30 % of the overlap is OBB over-approximation; the rest is a genuine
disagreement between two independent sources — OSM road centrelines and Hessen
LoD2 footprints. Real causes to expect: structures spanning streets
(*Durchfahrten*, arcades), roads passing under buildings, bridge and station
decks, and centreline-vs-carriageway offset.

**Investigate those 35 boxes before anything else.** Likely resolution is a
combination of:

1. **Drop or reclassify non-buildings** — bridges and canopies are in LoD2 but
   are not the urban canyon the scenario models. Check what the top 5 actually
   are before deleting anything.
2. **Split high-ratio parts into several OBBs** — this is the deferred split
   policy, and road intrusion is now a concrete reason to answer it. Applies to
   the ~30 % that is over-approximation.
3. **Down-weight affected road edges in route sampling** — a road genuinely
   running under a building is real, but the HVT should not spend half an episode
   there.

A residual few percent is legitimate difficulty: a target passing under an
arcade *is* briefly unobservable, and that is exactly the handoff pressure RQ3
studies. Half an episode is not.

> This is a `prep_osm.py` fix, i.e. a Block B amendment. It is written here
> because looking at Block C is what surfaced it. Re-bake the artefact and
> re-run `tests/test_osm_pipeline.py` afterwards, and add a test pinning the
> road-intrusion rate so it cannot regress.

---

## What to produce

One function, pure and batched, in `src/env/occlusion.py`:

```python
def clearance_m(
    pos: Tensor,            # (B, K, 3) node positions, local metres, z up
    boxes: Tensor,          # (M, 6) cx, cy, half_w, half_h, cos, sin
    heights: Tensor,        # (M,)
) -> Tensor:                # (B, K, K) signed metres, symmetric, diag = +inf
```

**Return signed clearance, not a boolean.** Positive means the ray passes that
many metres above the roofline; negative means it is blocked by that depth. Three
consumers already need exactly this:

| consumer | needs |
|---|---|
| `channel.pathloss_a2a_db(occluded=…)` | `clearance < 0` |
| `channel.pathloss_a2g_umi_av_db(los=…)` | `clearance >= 0` |
| `ENVIRONMENT.md` observations | *"clearance margin to HVT — signed metres the ray clears the roofline"* (and the same to the MCV) |

A boolean would force the observation to be recomputed separately, and a soft
margin is a far better learning signal than a hard flag — it tells the policy
*how close* it is to losing the link, which is what makes anticipation possible
at all (RQ3's anticipation-lead-time metric depends on it).

**Occlusion must be switchable off.** F0 is a pure connectivity radius with no
occlusion; F1 is F0 *plus* this. Do not bake the call into the channel — the
fidelity ladder in Block F turns it on and off as a config flag.

---

## The maths

A building is a vertical prism: an oriented rectangle extruded from `z = 0` to
`z = H`. Buildings are **2.5D** — check the segment's altitude *across the 2D
intersection interval*, not just whether it crosses the footprint in plan.

For segment `P0 → P1` and box `(cx, cy, hw, hh, cosθ, sinθ)` with height `H`:

**1. Into the box frame.** For each endpoint, translate then rotate by `−θ`:

```
dx, dy = px - cx, py - cy
lx     =  dx*cosθ + dy*sinθ
ly     = -dx*sinθ + dy*cosθ
```

`cos`/`sin` are baked into the artefact, so no trigonometry runs here.

**2. 2D slab test** on `|lx| ≤ hw`, `|ly| ≤ hh`, giving the parameter interval
`[t_enter, t_exit]` along the segment, clamped to `[0, 1]`. The ray misses the
footprint iff `t_enter > t_exit`.

Keep it branch-free: when the direction component is ~0, substitute `±∞`
sentinels rather than testing, so the min/max still works.

**3. Altitude across that interval.** `z(t)` is linear, so its minimum over
`[t_enter, t_exit]` is at one end:

```
z_min       = min(z(t_enter), z(t_exit))
clearance_i = z_min - H          # this box only; +inf if no 2D overlap
```

**4. Reduce.** `clearance = min over all boxes`, then `occluded = clearance < 0`.

**Endpoint-inside-a-box convention.** Decide it explicitly and test it. The
recommended rule is to **ignore any box containing an endpoint**: a node sitting
inside an over-approximated footprint should not blind itself, and after the fix
above this should be rare. Document whichever you choose — it changes results.

---

## Throughput is the binding constraint

Block D needs **≥1000 env-steps/s**. The arithmetic at `num_envs = 1024`,
`K = 7` nodes (5 drones + MCV + HVT), `M = 4220` boxes:

```
1024 envs x 21 unordered pairs x 4220 boxes  =  91 M segment-box tests per step
```

**Memory, not flops, is what bites.** ~10 flops per test is ~0.9 TFLOP/s at the
gate — comfortable on any training GPU. But materialising a `(1024, 21, 4220)`
fp32 intermediate is 363 MB, and the slab test needs several at once, so the
naive fully-vectorised form is multi-gigabyte and will OOM or thrash.

Two approaches, in order of effort:

1. **Stream over boxes in chunks**, accumulating a running `min`. A chunk of 512
   holds `(1024, 21, 512)` ≈ 44 MB per intermediate — 9 iterations, no culling,
   no data structure. **Try this first**; it may already clear the gate.
2. **Broad-phase cull** if streaming is not enough. A uniform grid over the box
   (the 20 m `height_grid` resolution is a natural starting point) with per-cell
   box index lists; gather only the cells a segment's 2D span touches. Note that
   links are long and thin, so a swept-AABB cull is weak — grid traversal is the
   better shape.

**Measure before optimising, and measure on the real `M`.** A micro-benchmark on
100 boxes proves nothing.

---

## Correctness

**Reference implementation.** A slow, obviously-correct `shapely` version:
intersect the 2D segment with the footprint polygon, take the intersection's
parameter range, interpolate `z` at both ends, subtract `H`. Compare against the
torch version on **random geometry** — random boxes at random orientations,
random segments, including deliberate edge cases:

- segment entirely inside / entirely outside a footprint
- segment exactly grazing a corner or an edge
- segment parallel to a slab axis (the division-by-zero path)
- segment passing exactly at roof height (`clearance ≈ 0`)
- zero-length segment
- vertical segment
- one endpoint inside a box (pins the convention above)

Agreement to ~1e-4 m. Random-geometry agreement is the real test; hand-computed
cases pin the ones a random sampler will rarely produce.

**Sanity against Block B.** The measured along-street sightline distribution
(median 127 m, p90 387 m — [`BLOCK_B.md`](BLOCK_B.md)) is an independent check:
horizontal rays at vehicle height down a street should reproduce it. If this
module says sightlines are much longer, something is wrong.

---

## Definition of done

- [ ] The 35 road-swallowing boxes investigated and resolved; artefact re-baked;
      a test pins the road-intrusion rate
- [ ] `src/env/occlusion.py`: pure, batched, no `.item()` / `.cpu()` / `.numpy()`
- [ ] Returns **signed clearance in metres**, not a boolean
- [ ] Matches a slow `shapely` reference on random geometry to ~1e-4 m
- [ ] Edge cases above are covered by co-located unit tests
      (`src/env/test_occlusion.py`)
- [ ] Reproduces Block B's measured sightline distribution
- [ ] Benchmarked at realistic `num_envs` and the real `M = 4220`, with the
      number written down — Block D's gate depends on it
- [ ] Works on CPU (local dev) and `cuda:0`, no silent device downgrade

## Watch out for

- **`.item()` anywhere in this file** forces a GPU sync. It is the easiest way
  to destroy the throughput gate and the hardest to spot.
- **Division by zero** when a segment is parallel to a slab axis. Use `±∞`
  sentinels, not branches — branches serialise on GPU.
- **The diagonal.** `clearance[b, i, i]` is meaningless; set it to `+inf` and
  never let it reach the routing layer.
- **Symmetry.** Compute the upper triangle and mirror it; computing both halves
  doubles the cost for nothing.
- **fp32 is fine** — coordinates span ±750 m and heights ±250 m, so there is no
  precision problem. Do not reach for fp64 out of caution; it halves throughput.
- **Do not import `shapely` in `src/`.** The reference implementation lives in
  the test file, which is the only place it may appear.
