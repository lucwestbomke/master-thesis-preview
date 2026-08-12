# Block C — batched occlusion

**Goal:** decide, for every link in every parallel environment, whether the ray
is blocked by a building — fast enough that Block D's throughput gate survives.

This is the **F1 rung of RQ1**. Occlusion is the hypothesis: the claim is that it
is the effect a channel abstraction cannot afford to omit. So this module is not
a utility, it is the independent variable, and it has to be right *and* cheap.

Consumes `data/frankfurt_box.npz` from [`BLOCK_B.md`](BLOCK_B.md). Produces
`src/env/occlusion.py`.

---

> **Status: done.** `src/env/occlusion.py` + `src/env/test_occlusion.py` (24
> tests) + `tests/test_occlusion_map.py` (5) + `scripts/bench_occlusion.py`.
> The artefact was re-baked to fix the road-swallowing problem below. What was
> found and measured is recorded at the end of this file; the sections in
> between are the spec as written, kept because the reasoning still applies.

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
`K = 7` nodes (5 drones + MCV + HVT), `M` boxes (4220 when this was written,
5120 after the road-swallowing fix below):

```
1024 envs x 21 unordered pairs x 5120 boxes  =  110 M segment-box tests per step
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

- [x] The 35 road-swallowing boxes investigated and resolved; artefact re-baked;
      a test pins the road-intrusion rate — see "What the fix was" below
- [x] `src/env/occlusion.py`: pure, batched, no `.item()` / `.cpu()` / `.numpy()`
- [x] Returns **signed clearance in metres**, not a boolean
- [x] Matches a slow `shapely` reference on random geometry to ~1e-4 m
      (7 seeds, including segments starting inside footprints)
- [x] Edge cases covered by co-located unit tests (`src/env/test_occlusion.py`,
      24 tests)
- [x] Reproduces Block B's measured sightline distribution
      (`tests/test_occlusion_map.py`) — see below
- [x] Benchmarked at realistic `num_envs` and the real `M`
      (`scripts/bench_occlusion.py`) — ✅ **re-run on CUDA 2026-08-12, gate met
      with ~3170× margin**; see "CUDA — the verdict" below
- [x] Works on CPU, MPS and CUDA with no silent downgrade. ⚠️ CUDA correctness
      rests on `bench_occlusion.py`'s parity check, **not** on the test suite —
      no test in this repo sets a device, so all 158 run on CPU wherever they are
      invoked. Parameterising them over device is a Block D deliverable

---

## What the fix was

Three changes to `prep_osm.py`, then a re-bake:

1. **Drop LoD2 bridge decks.** Cross-referenced against OSM `man_made=bridge`
   and `bridge=*`; a part lying >50 % on a bridge is not a building. Two parts
   dropped, removing **62.6 %** of all route intrusions on their own.
2. **Split badly-fitting footprints** into up to 4 oriented boxes, recursing
   while the single-OBB fit is worse than 1.5×. Over-approximation **+37 % →
   +21 %** for 1.21× the box count.
3. **Filter route sampling**: the MCV may not spawn inside a footprint (it never
   moves, so that would kill every link for the whole episode), and a route is
   rejected if it spends >5 % of the episode inside one.

| | before | after |
|---|---|---|
| boxes `M` | 4220 | **5120** |
| OBB fill of the box | 52 % | 46 % |
| HVT route points inside a building | 6.34 % | **1.12 %** |
| worst route, steps inside | 333 / 600 | **29 / 600** |
| MCV spawns inside | 4 | **0** |
| road nodes inside | 36 | 25 |

The escalation profile is unchanged (404 / 709 / 1011 / 1333 m). The residual
~1 % is roads genuinely running under podiums and through arcades — legitimate,
and exactly the brief unobservability RQ3 studies.

## Validation against the map

`tests/test_occlusion_map.py` re-derives the sightline distribution using the
production kernel on the baked boxes, and compares it with the offline
measurement made by a completely different code path (shapely against the source
LoD2 polygons). Same right-censoring treatment in both.

| | kernel | offline |
|---|---|---|
| median sightline | 118 m | 127 m |
| p90 | 379 m | 387 m |
| censored | 13 % | 16 % |
| beyond 830 m | 0.1 % | 0.2 % |

The kernel reads slightly *shorter*, which is the expected direction — oriented
boxes still over-approximate real footprints by ~21 %.

## Measured throughput

`M = 5120`, `K = 7` nodes ⇒ 21 links/env. Apple M-series **MPS**, fp32:

| num_envs | tests/step | eager | **`torch.compile`** | speedup |
|---|---|---|---|---|
| 64 | 6.9 M | 35 st/s | **885 st/s** | 25× |
| 256 | 27.5 M | 8 st/s | **503 st/s** | 64× |
| 1024 | 110 M | 1.8 st/s | **130 st/s** | 73× |
| 2048 | 220 M | 0.9 st/s | **53 st/s** | 60× |

**Fusion is the whole story, and it confirms the diagnosis in this spec.**
Arithmetic was never the wall — 110 M tests/step is 3.3 GFLOP, i.e. 3.3 TFLOP/s
at the gate against ~19.5 TFLOP/s fp32 on an A100. The wall was memory: unfused,
the elementwise slab chain writes ~20 intermediates of `(links × M)`, about
17.6 GB/step at `num_envs = 1024`, which would need 17 600 TB/s to hit the gate.
`torch.compile` keeps those intermediates in registers and the traffic collapses.

### ✅ CUDA — the verdict (2026-08-12)

RTX 5090 32 GB · torch 2.13.0+cu130 · CUDA 13.0 · Triton 3.7.1 · Python 3.12.3 ·
`chunk=512`, fp32. Parity green before timing: `cuda vs cpu` 0.00e+00 m,
`compiled vs eager` 1.53e-05 m.

| num_envs | calls/s eager | calls/s compiled | speedup | **env-steps/s compiled** | 10 M-step run |
|---|---|---|---|---|---|
| 64 | 291.2 | 19529.0 | 67× | 1.25 M | 8.0 s |
| 256 | 147.0 | 9325.6 | 63× | 2.39 M | 4.2 s |
| **1024** | 28.2 | **3094.1** | **110×** | **3.17 M** | **3.2 s** |
| 2048 | 10.9 | 1636.4 | 150× | 3.35 M | 3.0 s |

**The gate is met with ~3170× of margin at `num_envs = 1024`**, and occlusion is
~99 % of the step, so the env cannot plausibly be the bottleneck. Throughput
saturates near **3.3 M env-steps/s** by 2048, so `num_envs` is now free to choose
on *learning* grounds (rollout batch size) rather than throughput grounds.

Two things the MPS numbers got wrong, both worth recording:

1. **On CUDA, eager also clears the gate** — 28.2 × 1024 = 28.9 k env-steps/s,
   29× over. `torch.compile` is a 110–150× improvement on something that already
   passes, not the difference between feasible and infeasible. It stays the
   default because it is free and large, but the "required, not an optimisation"
   framing was an artefact of MPS, where eager genuinely was ~500× short.
2. **Peak VRAM stays under 4 GB, in both paths.** The 17.6 GB figure above is
   memory *traffic* per call, not live allocation: `chunk=512` keeps only
   `(num_envs × 21 × 512)` resident, i.e. ~0.9 GB of live intermediates at 1024
   and ~1.8 GB at 2048. Fusion removes traffic, not footprint. Anyone reasoning
   about VRAM pressure from the 17.6 GB number will be wrong by ~20×.

Observed GPU utilisation is near zero in a monitoring UI, which is expected and
not a sign the benchmark measured nothing: five timed iterations at 0.32 ms is
~1.6 ms of GPU work, far below any polling interval. Correctness is established
by the parity check and the explicit `torch.cuda.synchronize()` around the timing
window, not by the utilisation graph.

**The `--chunk` sweep was not run and is not worth running.** It existed to find
headroom; at 3000× over the gate there is none to find.

If more headroom is ever needed, the next lever is a **spatial broad phase**: at
2275 boxes/km², a 500 m segment with a 40 m corridor has ~45 real candidates
rather than 5120 — a ~100× reduction. **Do not build it.** Fusion is more than
enough, and an unnecessary index structure is a correctness risk in the module
that *is* RQ1's independent variable.

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
