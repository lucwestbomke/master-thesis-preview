# Environment: episode structure, curriculum, observations

What the env must do, for whoever builds `src/env/` Block D. Physics lives in
[`PHYSICS.md`](PHYSICS.md), reward in [`REWARD.md`](REWARD.md).

## Episode structure — launch, cue, route

**Launch.** Drones start parked on the MCV and fly out. The chain forms during
transit; that is a phase of the mission, not a preamble. It also creates the
energy tension — flying out costs battery, so the swarm cannot send everyone
everywhere.

**Start close, drive away.** The HVT starts **300–500 m** from the MCV and drives
outward. This resolves a conflict that otherwise has no solution: the MCV must be
*far* for a relay chain to be necessary (a single drone covers everything inside
~1000 m), but *near* for the cue to still be useful on arrival. Starting close and
opening the range gives both, and the chain requirement escalates on its own:

| Time | Range | Solo drone | Chain |
|---|---|---|---|
| t=0 | 400 m | 18.5 Mbps | 1 hop |
| t=60 s | 700 m | 9.5 Mbps | 1 hop |
| t=120 s | 1000 m | 4.7 Mbps | **2 hops** |
| t=240 s | 1400 m | 2.1 Mbps | **3 hops** |

The episode is therefore its own curriculum — easy at the start, hard at the end —
so early training gets dense reward from the opening instead of hitting a wall.

**Cue — its job is to break directional symmetry, not to solve acquisition.**
One-shot at launch, `σ ≈ 150 m`, **never refreshed**.

> An earlier draft refreshed the cue every 10 s from an "external ISR asset."
> That is incoherent: a sensor that can persistently track the HVT through a city
> makes the swarm redundant. The refresh was a mechanism invented to fix cue
> staleness, with a justification bolted on afterwards. The correct fix was to
> shorten transit by starting close.

Precision barely matters — drift during transit swamps `σ` anyway. Without *any*
cue the target sits in a 300–500 m annulus in any direction; five drones would
find it, but a random initial policy never would, so early training gets no
gradient. The cue supplies a vector to fly along from step one. That is all it is
for.

Acquisition difficulty is then set by **street topology, not by a parameter**.
The 830 m recognition range holds only down a clear straight street; Frankfurt's
streets bend, so 100–400 m is more typical. How often long sightlines actually
occur is an empirical question — **measure it in Block B**, do not assume it.

**Route — pre-sampled, not random-at-junctions.** At reset, sample the full route
as a path on the road graph *restricted to the map box*. Random turning behaves
badly: it doubles back, stalls in cul-de-sacs, oscillates around one block, and
leaves the map. Preventing all that amounts to writing a route sampler by
accident. Pre-sampling gives, by construction: the target never leaves the box
(no separate border logic needed), a known episode duration, reproducibility from
a seed, and a route that can be *required* to move away from the MCV. It costs
nothing in difficulty — the drones cannot see the future route either way.

**Speeds — from the OSM road class, not a constant.**

| Road class | Limit | m/s |
|---|---|---|
| residential | 30 km/h | 8.3 |
| secondary | 50 km/h | 13.9 |
| primary | 60 km/h | 16.7 |

> **Exclude primary/trunk from route sampling.** The drone must be meaningfully
> faster than the target or tracking is impossible, and at 20 m/s cruise the
> margin over a 60 km/h target is only 1.2× — not enough to recover after a turn.
> Restricting to residential/secondary gives 1.4–1.8×. Defensible anyway: a
> target moving covertly through a city uses ordinary streets.

Drone: **20 m/s cruise, 25 m/s dash.**

**Randomise per episode:** MCV position in the map, HVT start on a road 300–500 m
from it, and the route. The policy must not be able to memorise one layout.

**Termination — mission failure must NOT terminate the episode.** Two independent
failure modes if it does:

1. *Termination hacking.* If the link requirement starts only at acquisition, the
   optimal policy is to never acquire, never fail, and loiter.
2. *An undesigned curriculum.* Under the old rule (`C_e2e < 5 Mbps` for >5
   consecutive steps), a random initial policy dies around step 6 and the agent
   only ever experiences the first six steps. It cannot learn to track because it
   never reaches the tracking phase.

So:
- **Fixed-length episodes** (truncation), ~600–1200 steps at `dt = 0.25–0.5 s`.
- **Battery exhaustion still terminates** — physical, and unhackable, since
  hovering at the MCV burns power too.
- **Mission failure is a per-step condition**, feeding reward and metrics. The
  chain may drop and re-form, which is what real missions do.
- **Per-step idle penalty** whenever the HVT is unobserved, so loitering accrues
  unbounded negative reward and "never acquire" is strictly worse than trying.

Primary metric becomes **fraction of steps mission-capable** rather than survival
time — richer signal, and it cannot be gamed by refusing to start.

## Curriculum

**Curriculum varies *within* one training run. Fidelity varies *between* runs.**
They are orthogonal and must not be confused:

| | Curriculum | Fidelity (RQ1) |
|---|---|---|
| Set when | changes during training, via callback | fixed at env construction |
| Changes within a run | **yes** | **never** |
| Differs between runs | no — identical schedule everywhere | **yes, that is the point** |
| Purpose | make learning possible | the variable being measured |

Every run uses one fidelity level from first step to last, and every run walks
the same curriculum stages. Four students, four different textbooks, each taught
easy chapters first — then all four sit the same exam.

> ⛔ **Never use channel fidelity as a curriculum axis.** It is RQ1's independent
> variable; training up through fidelity levels would confound the primary result
> beyond repair. Same reasoning forbids ramping building density.

> ⚠️ **The jammer appears in both, and that is fine — but only if the schedule is
> identical everywhere.** Run the same jammer ramp in every condition and let the
> fidelity level decide whether it does anything. In an F0/F1/F2 run the ramp
> still executes and simply has no effect, because there is no SINR to degrade.
> Same parameters, same step counts, nothing to confound.

Budget real time for this. It is where projects of this shape stall, and the cue
is only one axis of four.

| Stage | HVT speed | Jammer | Battery | Episode length | Cue | What it teaches |
|---|---|---|---|---|---|---|
| 1 | **stationary** | off | 3× | 150 steps | exact | fly out, form a chain, hold station |
| 2 | residential (8 m/s) | off | 2× | 300 steps | exact | follow a moving target, keep the chain |
| 3 | full road speed | **on** | 1.5× | 450 steps | σ=150 m | degraded links near the target |
| 4 | full | on | **design value** | 600 steps | σ=150 m | chain escalation, energy, observer handoff |

Reasoning per axis:

- **HVT speed first, and it matters most.** A stationary target decouples "learn
  to relay" from "learn to chase". Those are two hard problems; learning them
  simultaneously from scratch is the likeliest failure mode.
- **Episode length** is nearly free here, because difficulty is *monotone in
  time* — a short episode is literally the easy 1-hop opening. Extending it is a
  curriculum with no extra machinery.
- **Battery** must start generous. An early policy flies inefficiently and would
  drain and die before learning anything. Initial charge is randomised in
  `[0.3, 1.0]` at stage 4 — a swarm mid-sortie has heterogeneous charge — which
  gives `Var(B)` something to act on from step 1.
- **Jammer off first**, since it degrades exactly the first hop, which is the
  hardest link to close.

Two rules that protect the results:

1. **Fixed schedule by step count in the reported runs**, not adaptive
   advancement. Adaptive advancement would let the easier fidelity levels
   progress faster and hand them more experience at the final stage, confounding
   RQ1. Use adaptive advancement during development to *find* the schedule, then
   freeze it and use the same one everywhere.
2. **Mix in earlier stages** (~20 % of episodes) rather than hard-switching, or
   the policy forgets the opening phase it still has to execute every episode.

Optional stretch, only once tracking already works: **stage 5 with no cue at
all** — genuine search. Legitimate as an endpoint; fatal as a starting point,
because that is where it eats the learning signal.

---


## Observations

**Rule: the actor may only see what a real drone could sense or receive.** Global
state belongs to the critic. Violating this quietly turns decentralized execution
into centralized execution and invalidates the whole CTDE framing.

### Actor — ego features (21)
| Feature | Dims | Realizable from |
|---|---|---|
| own velocity | 3 | INS |
| own altitude | 1 | absolute — LoS geometry depends on it |
| battery | 1 | |
| sees HVT (soft flag) | 1 | own sensor |
| relative vector to HVT | 3 | own sensor; zeroed when not seen |
| **relative velocity of HVT** | 3 | own sensor — without this the drone cannot anticipate |
| relative vector to MCV | 3 | MCV position is fixed and briefed |
| measured noise floor | 1 | **how the drone senses the jammer** |
| clearance margin to HVT | 1 | signed metres the ray clears the roofline |
| clearance margin to MCV | 1 | ditto |
| on active relay path | 1 | routing layer |
| current e2e capacity | 1 | reported back down the chain |
| steps since link last OK | 1 | proximity to episode failure |

### Actor — per-neighbour features (9 × N−1)
Relative position (3), relative velocity (3), their battery (1), whether they see
the HVT (1), whether they are on the path (1). All standard MANET position
reporting.

### Edge features (2)
Link capacity `C_ij` and the ray's clearance margin. **This is the only input the
GNN has and DeepSets does not** — it is precisely the rung RQ2 tests.

### How many neighbours — all of them, softly gated
`N−1 ≤ 7`. The graph is **fully connected in the tensor**, with influence scaled
by `E_ij = sigmoid((C_ij − 5.0)·γ)`. A neighbour behind a tower gets weight ≈0 and
its message is suppressed.

Not top-K, not a hard link-quality cutoff: a hard cutoff creates a gradient cliff
when a neighbour flickers across the threshold, and changes tensor shape per
timestep, which wrecks batching. Soft weights give the same effect with a smooth
gradient and a fixed shape.

### Terrain — clearance margins first, raster only if needed
Nothing above tells the drone a tower is *in the way* before a link degrades, so
it can react but never anticipate.

**Clearance margins (already listed) are the cheap half.** Signed metres by which
a ray clears the roofline — negative is blocked, positive is clear with margin.
Free from the slab-intersection code, and smooth where a boolean is a cliff.

**Local height raster is the optional half.** 24×24 cells at 20 m (a 480 m box),
into a 2–3 layer CNN → 64-dim embedding. Two things make this work:

- **Encode height relative to own altitude**, clipped: a cell reading `+100`
  means "something 100 m above me — I cannot see through it". This makes the
  representation **altitude-invariant**, which is a strong inductive bias and
  should help cross-city transfer by preventing the network from memorising
  Frankfurt's absolute heights.
- **Precompute one global grid** (1500 m / 20 m = 75×75) offline in
  `prep_osm.py`; at runtime each drone's patch is a batched **crop/gather**. No
  per-drone rasterization, no shapely, stays on GPU.

**Build order: margins first, raster only if the policy is visibly blind.** This
defers real work and yields a free ablation — *does spatial awareness of buildings
help, or do local sightline measurements suffice?*

### Critic — centralized, training-only
Sees global state: all drone states, HVT position and velocity, the full link
matrix. Two consequences:

- It **does not need to be size-agnostic**. Zero-shot transfer to `N ∈ {3,8}` runs
  the actor alone; the critic is discarded at evaluation. A plain MLP over
  concatenated global state is fine.
- Keep the critic **identical across all three architecture conditions**. If only
  the actor varies, RQ2 isolates the actor. If both vary, it is confounded.

---

