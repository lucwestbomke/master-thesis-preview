# Reward design

Implemented in [`src/env/reward.py`](../src/env/reward.py) as a **pure function of
a state summary**, and validated in `src/env/test_reward.py` by scoring scripted
policies and asserting their ranking.

> **This is the highest-leverage decision in the project.** Every other
> hyperparameter affects how fast you learn; the reward defines *what* is
> optimal. Get it wrong and the agent converges cleanly to the wrong behaviour.


## Structure
```
r =  w_mission · [observed AND C_e2e ≥ 15 Mbps]    # team — IS the headline metric
   + γ·Φ(s′) − Φ(s)                                 # potential-based shaping
   − w_idle    · [HVT not observed]                 # team
   − w_energy  · normalised power draw              # individual
   − λ         · Var(B_1..B_N)                      # team
   − w_effort  · ‖a‖²                               # individual, small
```

## The primary term is the metric, deliberately
`fraction of steps mission-capable` is both the dominant reward term and the
headline metric. Keeping them identical means the policy optimises exactly the
number that gets reported — no gap to explain later.

## Potential-based shaping — the only safe way to add guidance
Adding `F = γ·Φ(s′) − Φ(s)` for **any** `Φ` provably leaves the optimal policy
unchanged (Ng, Harada & Russell 1999): summed over a trajectory the terms
telescope to `γ^T Φ(s_T) − Φ(s_0)`, which depends only on the endpoints and so
adds the same constant to every policy's return.

> A naive "bonus for being close to the HVT" is a **salary** — 400 steps of
> loitering pays 15× what 20 steps pays, and it keeps growing whether or not the
> mission is ever accomplished. PBRS is a **one-time payment** for real progress;
> round trips cancel exactly, so there is nothing to farm.

Two rules:
1. **`Φ = 0` at genuine terminal states** (battery death), or `γ^T Φ(s_T)`
   survives the telescoping and reintroduces a policy-dependent bias. Truncation
   at 600 steps is fine provided the value is bootstrapped there.
2. **Everything *inside* `Φ` is free.** The proof holds for **any** `Φ`, so
   nothing within it can move the optimum: `k`, `d_ref_m`, `τ_c`, `τ_l`,
   `w_hold`, `d_hold_m`, **and the component weights `w_a` / `w_o` / `w_l`**,
   which this file calls "suggested" below and which nobody has moved.

   > ⚠️ This point used to read "the scale of `Φ` is free… it is *the one
   > quantity*… every other weight changes the objective", which was taken to
   > lock `w_a` / `w_o` / `w_l`. It does not — they sit inside the potential and
   > are as free as `k`. Corrected 2026-08-25.

   ⛔ The **objective** weights are the constrained ones — `mission`, `idle`,
   `energy`, `battery_variance`, `effort` define what is optimal, and of those
   only `λ` (`battery_variance`) is sweepable. The rest are pinned by the
   behavioural orderings in this file.

## The potential
```
Φ(s) = k · [ w_a·Φ_approach + w_o·Φ_observe + w_l·Φ_link ]      k ≈ 10
```
| Component | Form | Job |
|---|---|---|
| `Φ_approach` | `1 − min(d_min, D_ref)/D_ref`, `d_min` = nearest drone→HVT, `D_ref` ≈ map diagonal | coarse; non-zero anywhere on the map so the agent is never blind |
| `Φ_observe` | `sigmoid(clearance_best / τ_c)`, `τ_c ≈ 15 m` | fine; rewards correct *geometry*, not mere proximity |
| `Φ_link` | `sigmoid((C_e2e − 15.0) / τ_l)`, `τ_l = 6 Mbps` | gradient below threshold, where the binary indicator has none |
| `Φ_observe`'s **hold factor** | `× (1 − w_h + w_h·(1 − min(r_obs, d_hold)/d_hold))`, `w_h = 0` **off by default** | ⚠️ puts a gradient in the regime where every term is flat — see below. 📏 **A null at 5 seeds**; superseded by `Φ_standoff` |
| **`Φ_standoff`** | `observed · sigmoid((d* − r_obs)/τ_r)`, `d* = 127 m`, `τ_r = 40 m`, `w_s = 0` **off by default** | the closing decision, graded *at the threshold* — see **Φ v2** |
| **`Φ_cover`** | `½·[mean_m(1 − Π_i(1 − f_im)) + mean_i(max_m f_im)]`, `f = 1/(1+(d/120 m)²)`, `w_v = 0` **off by default** | the only component that is not blind to four drones out of five — see **Φ v2** |

Each lands in `[0,1]`. Shipped `w_a=0.25, w_o=0.35, w_l=0.40` — tilted toward
the link, which was assumed to be the hardest and last-learned stage. ⚠️ The
`Φ v2` preset re-allocates them; see below.

**The handover is the design.** Far out only `Φ_approach` moves; once a drone is
close it saturates and `Φ_observe` takes over; once observing, only `Φ_link`
still improves. Three mission stages, each with a live gradient, no dead zones.

> ☠️ **That last paragraph is the design intent and it is not what the shipped
> potential does.** Measured, not argued — see the next two sections.

### ☠️ `Φ` was audited on 2026-08-27 and it is switched off where it matters

📏 `scripts/measure_potential.py`, eval split, stage 4, F4, MPS. It banks the
states a policy actually visits and scores any `RewardWeights` over them, so a
candidate potential can be judged **before** a training run. Two defects, and
they are different in kind:

**1. There is no gradient along the axis the whole diagnosis turns on.** Sweep
the observer 250 m → 60 m with the ray clear and a chain carrying 25 Mbps —
exactly the decision "close to B0's 89 m or stand off at 184 m", with everything
else held where the policy already has it:

| | swing over the band | per 8 m step | vs the energy term's 0.0544 |
|---|---|---|---|
| shipped `Φ` | **+0.320** | 0.0133 | **0.25×** |
| `Φ v2` | +1.717 | 0.0774 | **1.42×** |

**2. `Φ` is exactly constant in four drones out of five.** Every shipped
component reduces the swarm with a hard `min` (`nearest_dist_m`), a hard `max`
(`best_clearance_m`) or the router's chosen path (`e2e_capacity_mbps`). So a
drone that is not currently the nearest, the clearest or on the chain can fly
**anywhere at all** without moving `Φ` by one bit — and those are precisely the
drones that have to pre-position for the relay and the handoff. Measured with
four drones working the axis and the fifth stranded to one side:

| stranded drone, one 8 m step home | shipped | `Φ v2` | whole trip home, shipped → v2 |
|---|---|---|---|
| 200 m off-axis | **0.0000** | 0.0078 | 0.000 → **+0.339** |
| 500 m off-axis | **0.0000** | 0.0010 | 0.000 → **+0.446** |
| 800 m off-axis | **0.0000** | 0.0003 | 0.000 → **+0.465** |

📏 And the behaviour that goes with it: learned policies sit pressed against the
map boundary on **15–23 %** of steps where B0 sits there **0.9 %**, and
`off_axis_m` is **252 m** against B0's **105 m**. Out there the shipped
potential is not weak, it is **absent**.

⚠️ **This retro-explains the shape of every null in Block G.** `w_hold`,
`w_relay`, `d_ref 400` and `potential_scale 30` all moved a potential whose
gradient in the operating regime was between 0.013 and 0.03 per step. None of
them could have worked, and the reason is arithmetic rather than mechanism.

### `Φ` v2 — the rebuild, `reward.PHI_V2`, off by default

Ships as one preset behind `--phi v2`; `w_standoff = w_cover = 0.0` reproduces
the shipped potential **bitwise**, so `test_golden.py` is untouched.

```
Φ = k · [ 0.05·Φ_approach + 0.20·Φ_observe + 0.20·Φ_standoff
                          + 0.15·Φ_link    + 0.40·Φ_cover ]        k = 10
```

🔒 **The weights sum to 1.0 and `k` stays 10: `Φ` is redistributed, never
inflated.** 📏 The reason is measured. PBRS pays `γ·Φ(s′) − Φ(s)`, so a policy
that *holds* a state pays `(γ−1)·Φ` every step — at `γ = 0.997` and `Φ ≈ 9` that
is **−0.027/step**, and the instrument measures B0's mean shaping at
**−0.018/step** against the learned policy's −0.016. **The drag is proportional
to `Φ`, and therefore largest for the best policy.** Tripling `k` triples it,
which is the most likely reason `potential_scale = 30` measured as a *null*
rather than as an improvement in the 81-run sweep.

**`Φ_standoff` — the closing decision, and it is not `w_hold` again.** The
variable is the same and both the construction and the scale are different:

| | `w_hold` (null at 5 seeds) | `Φ_standoff` |
|---|---|---|
| enters `Φ` as | a **factor on** `Φ_observe` | an **additive component** |
| budget | `w_observe · w_hold` ≤ 0.21 | `w_standoff` = 0.20, set directly |
| shape | linear ramp to `d_hold` = 400 m | logistic centred on **127 m** |
| over 291 → 79 m | +0.74 total, 0.03/step | +1.72 total, **0.077/step** |
| at its maximum setting | a distant sightline is worth **zero**, discouraging acquisition | `Φ_observe` still pays in full; standoff only ever **adds** |

📏 The centre is Block B's measured along-street sightline median, **127 m**: B0's
observer stands at 88.8 m *inside* it and the learned observer at 184 m *outside*
it. The deficit is a **threshold**, so the budget is spent at the threshold
rather than smeared over 400 m of ramp.

⚠️ Gated on `observed`, the boolean — **not** on `sigmoid(clearance/τ_c)`. A test
forced the distinction: a ray blocked by 20 m of building still reads
`sigmoid(−20/15) = 0.21`, so grading the gate would leak a fifth of the closing
pull to a blind drone, which is this file's own first trap.

**`Φ_cover` — the term-blindness fix.** Sample 16 points along the MCV→HVT axis;
`f_im = 1/(1 + (d_im/120 m)²)` is drone `i`'s cover of sample `m`. Two halves,
and each is the other's degenerate case:

* `covered = mean_m [1 − Π_i(1 − f_im)]` — a soft OR. Rewards **spreading**, and
  cannot be satisfied by huddling at either end of the axis. ⚠️ On its own it
  gives a *redundant* drone no gradient, and that is correct arithmetic rather
  than a bug: `∂covered/∂f_j = Π_{i≠j}(1 − f_i)` is ~0 once the axis is held.
  Measured at **+0.0003/step** for a fifth drone 200 m off-axis — still nothing.
* `mustered = mean_i [max_m f_im]` — each drone's own proximity to the axis.
  Non-vanishing everywhere. ⚠️ On its own it is maximised by every drone sitting
  **on the MCV**, which is distance zero from the segment and covers none of it.

Fixed 50/50, not a knob: they are two halves of one statement — *be on the axis,
and be spread along it*.

🔍 **`∂covered/∂f_j = Π_{i≠j}(1 − f_i)` is the interesting part.** A drone's
marginal value at a point is exactly *how uncovered that point is by everybody
else* — a differentiated role pressure out of a **team** quantity with no agent
index in it, which is what this file's homogeneity rule requires and what Block
G's per-drone `w_relay` could not produce.

⚠️ **The kernel is Cauchy, not Gaussian, deliberately.** At 800 m and
`r_cover = 120 m` it still reads 0.022 and still has slope. An exponential tail —
or the hard `min` every shipped component uses — is numerically zero out there,
and a drone at the map edge is exactly the one that has to be told to come back.

📏 `r_cover_m = 120` was **chosen by sweep and the obvious derivation was the one
the sweep rejected.** `R`/2 = 262 m ("two drones whose discs touch are one hop
apart") saturates the term at the behaviour it is meant to reward:

| `r_cover` | 120 | 180 | 262 | 400 m |
|---|---|---|---|---|
| B0 − learned separation | **1.247** | 1.114 | 0.872 | 0.575 |
| B0's p1 (of a 3.0 maximum) | **1.428** | 2.000 | 2.446 | 2.750 |
| whole trip home from 500 m | **0.335** | 0.286 | 0.242 | 0.184 |

120 m wins on all three, and is *corroborated* — not derived — by Block B's 127 m
sightline median.

### How to judge a candidate `Φ` without training anything

⚠️ **The ideal `Φ` is `V*`.** Ng, Harada & Russell show the shaped problem's value
function is `V − Φ`, so `Φ = V*` makes every advantage immediate. That gives an
offline test with no training run in it: correlate `Φ` against the **discounted
future `mission_capable` return** of banked states, and check it ranks B0's
states above a learned policy's.

| pooled over both banks | shipped | `Φ v2` |
|---|---|---|
| corr(`Φ`, discounted future capable) | +0.270 | **+0.301** |
| mean `Φ`(B0) − mean `Φ`(learned) | +2.101 | **+3.584** |

The correlation gain is modest and is reported as such; the **separation** is the
number that moved, by 71 %. This is the check that could have told the block's
four failed interventions from a good one, and it costs a minute.

### ☠️ The objective pays drones to keep moving, and it outweighs `Φ` by 100×

📏 Measured 2026-08-26. The rotary-wing power curve is U-shaped — `energy.py`
exists because the original spec had it backwards — and normalised to hover:

| speed | 0 | 5 | 10 | **13.3** | 15 | 20 | 25 m/s |
|---|---|---|---|---|---|---|---|
| `P / P_hover` | 1.000 | 0.855 | 0.671 | **0.638** | 0.646 | 0.758 | 1.004 |

So at `w_energy = 0.15`, **cruising at 13.3 m/s instead of holding station pays
`+0.0544` per step** — `+32.6` over a 600-step episode, or **33 steps of
`mission_capable`**.

⚠️ **Against the entire reachable `Φ` swing of 0.32, that is 102×.** Per step it
is worse still: closing 8 m (one step of travel) moves `Φ` by ~**0.013**, while
cruising rather than holding pays **0.054**. **The energy term is four times
stronger than the shaping and points the other way.**

**This is not a bug and the objective is not misspecified.** The power curve is
physically right, `w_energy` is pinned by the behavioural orderings below, and B0
pays the cost and still earns 244.1 episode return against a learned policy's
90.5 — the reward ranks the target behaviour 2.7× higher. What is wrong is the
**gradient balance**: a drone weighing "hold this position" against "keep
cruising" sees a certain −0.054/step now, against a future mission gain that `Φ`
signals at 0.013/step.

📏 It also explains the measured behaviour that four interventions could not
move. Idle drones cruise rather than hold, so `off_axis_m` is **252 m** against
B0's **105 m**; the observer stands at **184 m** rather than closing to B0's
**89 m**, because holding station over a moving target is the expensive option.
And every `Φ`-side intervention tried so far — `w_hold`, `w_relay`, `d_ref 400`,
`potential_scale 30` — is worth ≲1 against an opposing force of ~30.

⛔ **Do not "fix" this by cutting `w_energy`.** It is an objective weight, pinned
by ordering, and lowering it changes what is optimal. The lever is `Φ`, which is
free — and the number this section exists to supply is **the scale `Φ` has to
reach: a per-step gradient competitive with 0.054, not a total swing of 0.32.**

### ⚠️ Why `Φ_observe` needs a hold factor — the flat-success problem

Added 2026-08-25, **off by default** (`w_hold = 0` reproduces the shipped
potential bitwise). Look at every reward term while the swarm is *succeeding*:

| term | value when a drone is observing over a live chain |
|---|---|
| `w_mission · capable` | 1.0 — flat |
| `w_idle · ¬observed` | 0 — flat |
| `Φ_observe = sigmoid(clearance/15)` | `occlusion` returns **1e4** for a clear ray → `sigmoid(667)` = **1.0**, flat |
| `Φ_link = sigmoid((C−15)/6)` | a formed chain carries ~60 Mbps → **0.999**, flat |

**Everything is flat.** Nothing distinguishes an action that will hold the
sightline from one that will drift out of it, and the policy only hears about the
drift ~30 steps later through a GAE window whose effective horizon at λ = 0.95 is
~20 steps. 📏 That is the measured deficit: B0 holds the observer role 264.6
steps, every learned policy 27–51.

📏 **It is a zero gradient, not a weak one**, which is why scaling could not fix
it. The 81-run sweep moved `d_ref_m` 1500 → 400 (3.8× the closing gradient) and
`potential_scale` 10 → 30. Both were nulls. You cannot fix a zero by multiplying
it — an earlier reading of this deficit as "the pull toward closing is too weak"
is therefore **wrong**, and the sweep is what refuted it.

`hold` grades the sightline by the range of the drone *holding* it — the argmax
of clearance, **not** `nearest_dist_m`, because a drone can be nearest and blind
on the wrong side of a building. At a 40–80 m ceiling range is a cheap monotone
stand-in for elevation angle: B0 parks its observer at 79 m (≈37°, a short
near-vertical ray that survives the HVT moving down a street) where learned
policies loiter at 291 m (≈12°, a long canyon ray one corner kills).

Measured effect on `Φ` of closing 291 m → 79 m with the ray clear:
**0.000 shipped**, 0.74 at `w_hold = 0.4`, 1.11 at `w_hold = 0.6`.

🔧 Sane range is `w_hold ∈ [0, 0.6]`: at 1.0 a distant-but-clear sightline is
worth zero potential, which would discourage acquiring at all. Team quantity, so
once someone is parked the pull stops for everyone — no clustering.

⛔ **Do not do this in `r` instead.** A "consecutive observed steps" bonus encodes
the hypothesis into the objective and then lets us discover it, and it is
non-Markovian. Inside `Φ` the PBRS proof bounds the damage to learning speed.

Three traps this avoids:
- **Distance to the HVT is the wrong measure.** The observation envelope is a
  wedge down the street plus an overhead cone, not a disc — a drone 20 m away
  across the street sees nothing while one 300 m down the street sees fine.
  Hence `clearance`, not range.
- **Per-drone potentials cause clustering.** All five drones get pulled onto the
  HVT and nobody relays. `d_min` and `clearance_best` are **team** quantities, so
  once one drone has the target the pull stops for everyone else.
- **A product form deadlocks at t=0.** `Φ_observe × Φ_link` is flat at episode
  start, when both are ≈0 and neither can improve without the other. **Sum.**

> ⚠️ `τ_c` and `τ_l` are starting values reasoned from geometry (22 m buildings,
> 8 m of travel per step) and from the threshold (40 % of it, so `τ_l` moved
> 2 → 6 Mbps when the requirement moved 5 → 15). **Re-tune them
> in Block G, not before** — safely, since they live in the potential and cannot
> move the optimum. Block E deliberately did not: it now supplies the empirical
> `clearance_best` and `C_e2e` distributions the retune needs, but the thresholds
> can only affect *learning speed*, which cannot be measured until a learner
> exists ([`DECISIONS.md`](DECISIONS.md)).

## Setting the remaining weights — by behavioural ordering, not sweeping
Write down pairs of behaviours you know how to rank, and require the reward to
rank them correctly. Each pair gives an inequality; the inequalities pin the
weights. Compute the energy quantities from the rotary-wing model.

Encoded as `weight_constraints_satisfied()` and asserted in `test_reward.py`,
including a guard that each constraint actually rejects a bad setting.

| Required ordering | Constraint | Status |
|---|---|---|
| trying beats loitering | `w_idle > w_energy·(e_dash − e_loiter)` | ✅ |
| full success beats safe partial success | `w_mission > w_idle` | ✅ |
| mission beats perfect battery balance | `w_mission > λ·Var_max` | ✅ |
| energy cannot veto flying | `w_mission > w_energy·e_dash` | ✅ |
| control effort stays a heuristic | `w_effort < 0.1·w_energy` | ✅ |

**Chosen values:** `w_mission=1.0` (the unit), `w_idle=0.3`, `w_energy=0.15`,
`w_effort=0.01`, `λ=0.5` (swept), `k=10`.

> **The energy inequality has the opposite sign to intuition.** Because the power
> curve is U-shaped, flying at 13 m/s costs **0.64** of hover draw and even a
> 25 m/s dash costs **1.00**. Flying is not more expensive than hovering, so the
> lazy optimum is not an energy story — `w_idle` exists to break a tie that
> energy alone would leave open, and it is sized against dash-versus-loiter, not
> motion-versus-stillness.

**Only `λ` is swept**, because the right amount of load-balancing pressure is not
derivable from physics. That is a far better justification than "we did not know."

> The lazy optimum survives fixed-length episodes: never acquiring means never
> flying out, which *saves energy*. Zero mission reward at low cost beats zero
> mission reward at high cost. `w_idle` exists to break exactly that, and the
> first constraint above sizes it.

## Known degenerate optima — check for these explicitly
- **Free-riding.** Mission reward is shared, energy cost is individual, so the
  selfish optimum is to let the others work. `λ·Var(B)` is the counter-mechanism,
  not merely a rotation device.
- **Variance term's own optimum.** All drones hovering ⇒ `Var(B)=0` ⇒ zero
  penalty. Doing nothing scores perfectly on that term and must be dominated.
- **Capacity over-optimisation.** Reward capacity linearly and the swarm clusters
  for 74 Mbps when 5 is required, abandoning coverage. Saturate.
- **Observation clustering.** Reward per-drone sighting and nobody relays. Reward
  the *mission*, not the sighting.

## Discount factor is part of the reward design
Effective horizon is `1/(1−γ)`. The episode is 600 steps and **difficulty is
concentrated at the end** — the 3-hop regime only appears after t≈120 s. At the
PPO default `γ=0.99` the horizon is 100 steps, so the agent is structurally blind
to the hard part and would optimise the easy opening. Use **`γ ≈ 0.997–0.999`**.

## Validate the reward before training anything
`test_reward.py` scores four scripted policies and asserts the ranking.

> ⚠️ **The table below is synthetic.** The four "policies" are hand-written
> `Snapshot` stubs, which is the right scope for a unit test of `reward.py` but
> means the ordering was never checked against states the environment actually
> produces. Block E closed that loop:
> **`tests/test_baseline_reward_ordering.py`** scores real policies through the
> real env and asserts the same ranking — B0 (+230) > B0-geodesic (+164) >
> random (−164) > lazy (< 0), 5 seeds on the eval split. Quote the real numbers
> in the thesis, not these.
>
> The stubs are also why the requirement change was not silent: their capacities
> were magic numbers sized against a 5 Mbps bar, and at 15 they crossed it and
> inverted three tests. They now express intent (`GOOD`, `OK`, `POOR`) relative
> to `CAPACITY_THRESHOLD_MBPS` instead of restating it.

Current stub values (100 steps, mean over 5 agents):

| Policy | Return | Per step |
|---|---|---|
| B0 heuristic | **+62.2** | +0.62 |
| fixed formation | +21.7 | +0.22 |
| all-chase, no relay | −11.0 | −0.11 |
| lazy, never launches | **−44.6** | −0.45 |

Note the ordering is *strict* at every rung, and the two failure modes are
separated: seeing without relaying beats seeing nothing, but never beats a
working chain. If that ordering ever breaks, the reward is wrong — found in
milliseconds rather than after a three-hour run.

Log **every term separately** in W&B. The total is nearly useless for diagnosis;
one term contributing 95 % of the magnitude is the signature of a scaling error
and is invisible in the aggregate.

## Constraints that protect the experiments
- The reward function must be **byte-identical across F0–F4**. Only the physics
  feeding it changes. Consequence to expect rather than discover: under F0
  capacity is binary, so `Φ_link` is degenerate and carries no gradient — that is
  part of what "training under a simplified channel" *means*, not a bug.
- The reward must **not depend on agent index**, or homogeneity breaks and the
  "roles emerge rather than being assigned" claim collapses.

