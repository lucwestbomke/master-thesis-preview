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

| Time | Range | Solo drone | Chain @ 5 Mbps | Chain @ **15 Mbps** |
|---|---|---|---|---|
| t=0 | 400 m | 18.5 Mbps | 1 hop | 1 hop |
| t=60 s | 700 m | 9.5 Mbps | 1 hop | **2 hops** |
| t=120 s | 1000 m | 4.7 Mbps | **2 hops** | **2–3 hops** |
| t=240 s | 1400 m | 2.1 Mbps | **3 hops** | **3+ hops** |

> **The rate requirement is 15 Mbps** (raised from 5 in Block E —
> [`DECISIONS.md`](DECISIONS.md)), so the escalation starts *earlier* than the
> middle column: a solo drone drops below the bar before t=60 s rather than
> around t=120 s. Measured under B0 on real geometry, the chain is multi-hop on
> **80 %** of steps overall and **95 %** of last-third steps, with the
> `min(n,3)` divisor saturated on **50 %** of late chain-steps
> ([`BLOCK_E.md`](BLOCK_E.md)). The escalation this table describes is real and
> is exercised.

The episode is therefore its own curriculum — easy at the start, hard at the end —
so early training gets dense reward from the opening instead of hitting a wall.

✅ **Measured, and it now behaves as designed.** B0's mission-capable fraction
peaks at **84 % around t = 40 s and decays to 35 % by t = 240 s**, while
`observed` holds at 98 % throughout: the sensor problem is solved early and the
*chain* is what degrades as the target drives out. At the original 5 Mbps bar
this profile was flat at ~100 % after 60 s and the intended difficulty gradient
did not exist ([`BLOCK_E.md`](BLOCK_E.md)).

**Cue — its job is to break directional symmetry, not to solve acquisition.**
One-shot at launch, `σ ≈ 150 m`, **never refreshed**.

> An earlier draft refreshed the cue every 10 s from an "external ISR asset."
> That is incoherent: a sensor that can persistently track the HVT through a city
> makes the swarm redundant. The refresh was a mechanism invented to fix cue
> staleness, with a justification bolted on afterwards. The correct fix was to
> shorten transit by starting close.

Precision barely matters — drift during transit swamps `σ` anyway. The cue
supplies a vector to fly along from step one. That is all it is for.

> ✅ **Measured (Block D), and the original justification was wrong.** This
> section used to argue that without a cue "a random initial policy never would"
> find the target. Not so — the HVT starts in a 300–500 m annulus the drones
> launch *inside*, and the sensor reaches 830 m. Five drones flying a radial fan
> over 512 real routes ([`../scripts/measure_envelope.py`](../scripts/measure_envelope.py)):
>
> | strategy | ever found | t50 | t90 |
> |---|---|---|---|
> | **no cue**, 5-way fan | **100 %** | **8 s** | 22 s |
> | cue σ=150 m, narrow fan | 99.8 % | 10 s | 22 s |
> | no cue, all five on one bearing | 59.0 % | 13 s | 46 s |
>
> Uncued is *faster*. What matters is **spreading out**, not knowing the
> direction. So the cue is not buying exploration — it buys away the need for
> homogeneous agents to learn symmetry-breaking off the neighbour channel, a
> coordination problem no RQ asks about, sitting in the block most likely to
> stall. That is the honest reason, and it is weaker than the old one.
> Consequently **the no-cue condition is a one-flag ablation, not the "optional
> stage 5" below** — and it is the direct answer to *"isn't the cue a cheat?"*.
> Note it leans on the 360° sensor assumption ([`BLOCK_D.md`](BLOCK_D.md)), which
> is optimistic for search though not for tracking.
>
> **The cue is observable and persistent.** It occupies its own 3 dims (see
> Observations) and is never zeroed, because a stale cue is not misleading:
> `grow_outward` builds near-radial routes, so at t=599 it is still within 17.8°
> of the target's bearing (p90 42.9°, 0.4 % of routes past 90°) and flying to it
> beats sitting at the MCV in 99 % of routes. **It decays in range, not in
> direction.**

Acquisition difficulty is then set by **street topology, not by a parameter**.
The 830 m recognition range holds only down a clear straight street; Frankfurt's
streets bend, so 100–400 m is more typical. How often long sightlines actually
occur is an empirical question — **measure it in Block B**, do not assume it.

> ✅ **Measured** (`scripts/measure_sightlines.py`, 6766 rays over the real box).
> Median sightline **127 m**, p90 387 m. The 100–400 m guess covers 51 % of
> rays — but **40 % fall below 100 m**, so it was optimistic and acquisition is
> harder than assumed. Only **0.2 % exceed 830 m**, so the sensor range never
> binds and street topology really is what sets the difficulty, as claimed.
> Full distributions and the caveat that this is a 2D ground-level measure:
> [`BLOCK_B.md`](BLOCK_B.md) → "Measure this while you are here".

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

| Stage | HVT speed (`speed_scale`) | Jammer | Battery | Episode length | Cue | What it teaches |
|---|---|---|---|---|---|---|
| 1 | **stationary** (0.0) | off | 3× | 150 steps | exact | fly out, form a chain, hold station |
| 2 | half pace (0.5) | off | 2× | 300 steps | exact | follow a moving target, keep the chain |
| 3 | three-quarter (0.75) | **on** | 1.5× | 450 steps | σ=150 m | degraded links near the target |
| 4 | full (1.0) | on | **design value** | 600 steps | σ=150 m | chain escalation, energy, observer handoff |

⚠️ **The speed axis is a scale on the baked route, not a class speed.** An
earlier version of this table quoted free-flow OSM class speeds (8.3 / 13.9 m/s),
but the route bank already has `CONGESTION_FACTOR = 0.70` applied — realised
speeds are median **5.8 m/s**, max 9.7 m/s ([`BLOCK_B.md`](BLOCK_B.md)). Stage
2's old "residential 8 m/s" was therefore *above* the bank's median and the
curriculum silently had no speed axis at all. Implement it as
`idx = (t * speed_scale)` into `route_xy`, and note that a scale below 1.0 also
weakens the escalation, which is correct: earlier stages should be easier.

Battery multipliers are a `capacity_scale` divisor on drain, not a charge above
1.0 — the observation must stay in `[0, 1]`.

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

**No-cue running is no longer a stretch goal.** The measurement in the cue
section shows an uncued fan acquires in 8 s over 100 % of routes, so "genuine
search" is not the hard endpoint this section assumed. Treat it as a one-flag
**ablation** run at full fidelity, not as a fifth curriculum stage — and report
it, because it is the cleanest available answer to whether the cue is doing
unearned work.

---


## Observations

**Rule: the actor may only see what a real drone could sense or receive.** Global
state belongs to the critic. Violating this quietly turns decentralized execution
into centralized execution and invalidates the whole CTDE framing.

### Actor — ego features (24)
| Feature | Dims | Realizable from |
|---|---|---|
| own velocity | 3 | INS |
| own altitude | 1 | absolute — LoS geometry depends on it |
| **relative vector to the cue** | **3** | briefed at launch, never refreshed. Persistent all episode — see the cue section above |
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

> **No time feature, deliberately.** Pardo et al. (2018) separate *time-limited*
> tasks, where the horizon is part of the problem and remaining time belongs in
> the observation, from *time-unlimited* ones, where the limit only diversifies
> training and the correct treatment is partial-episode bootstrapping. This
> mission is the second kind: 240 s covers the hop escalation, but nothing about
> the mission ends there, and `reward.shaping` already says *"truncation is not
> terminal — bootstrap the value there instead."* Observing the clock would let
> the policy condition on an artificial horizon. **The requirement this creates:
> the skrl wrapper must keep `terminated` and `truncated` distinct and bootstrap
> at truncation** — asserted in Block D's smoke test, because wrappers routinely
> collapse the two.
>
> Episode phase is legible anyway without a clock: HVT–MCV separation grows
> monotonically 404 → 1333 m, and relative vector to the MCV is already listed
> above.

### Actor — per-neighbour features (9 × N−1)
Relative position (3), relative velocity (3), their battery (1), whether they see
the HVT (1), whether they are on the path (1). All standard MANET position
reporting.

### Edge features (2)
Link capacity `C_ij` and the ray's clearance margin. **This is the only input the
GNN has and DeepSets does not** — it is precisely the rung RQ2 tests.

### How many neighbours — all of them, softly gated
`N−1 ≤ 7`. The graph is **fully connected in the tensor**, with influence scaled
by `E_ij = sigmoid((C_ij − CAPACITY_THRESHOLD_MBPS)·γ)`. A neighbour behind a
tower gets weight ≈0 and its message is suppressed.

Not top-K, not a hard link-quality cutoff: a hard cutoff creates a gradient cliff
when a neighbour flickers across the threshold, and changes tensor shape per
timestep, which wrecks batching. Soft weights give the same effect with a smooth
gradient and a fixed shape.

#### Why a drone hears from neighbours it cannot carry video to

The obvious objection: if drone *i* has no viable link to *j*, how does *j*'s
position report reach it? Two answers, and the first is the load-bearing one.

**Telemetry and video are three orders of magnitude apart in rate.** The mission
requirement is **15 Mbps** for the sensor feed; a position/velocity/battery/flag
report is tens of bytes at a few Hz — call it 10 kbps. Over 10 MHz that is
0.001 b/s/Hz against 1.5, and inverting `min(0.75·log₂(1+SINR), 7.4)` puts the
telemetry threshold near **−30 dB SINR** against **+4.8 dB** for a single video
hop. **A link can fail for video by 35 dB and still carry telemetry.** Real
tactical MANETs are built exactly this way — a low-rate always-on situational-
awareness channel underneath the high-rate payload.

**And the swarm is a mesh, not a set of point-to-point links.** Even a genuinely
dead direct path does not isolate *j*: its report reaches *i* via any relay. That
is what a MANET is for, and it is the same multi-hop the video uses.

So gating the neighbour channel on the *video* link would be **over-strict, not
conservative** — it would model a radio nobody builds. What the observation does
instead is give every drone the link state and let it draw its own conclusion:
`edge` carries `C_ij` and the clearance margin for *every* pair, so "my link to
*j* is dead" is **present information**, not missing information. That is exactly
what a drone needs to reason about chain geometry, and withholding it would make
the task harder in an unphysical way.

> ⚠️ **Two assumptions this rests on, both optimistic, neither currently
> modelled.** (1) Position reports are **exact and instantaneous** — no latency,
> no error. (2) Telemetry is **jam-immune**. The second is thin in one specific
> place: the observer sits directly over the jammer, where received jam power is
> maximal — roughly −51 dBm at 78 m against a −97 dBm noise floor — so its
> incoming telemetry SINR lands near the −30 dB threshold rather than
> comfortably above it. If the neighbour channel is ever challenged, the honest
> answer is a **sensitivity analysis** gating telemetry at two or three SINR
> floors, not an asserted value — the same move `routing.py` makes with
> `reuse_limit`. Do not build it speculatively.

**Do not confuse the two planes.** The *data path* is a chain, chosen per step by
the routing DP. The *control plane* is a broadcast mesh. Restricting a drone to
"its two chain neighbours" would conflate them — and would be circular, since you
need to see all the links to choose which two become the chain.

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

