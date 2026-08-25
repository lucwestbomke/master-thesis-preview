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
| `Φ_observe`'s **hold factor** | `× (1 − w_h + w_h·(1 − min(r_obs, d_hold)/d_hold))`, `w_h = 0` **off by default** | ⚠️ puts a gradient in the regime where every term is flat — see below |

Each lands in `[0,1]`. Suggested `w_a=0.25, w_o=0.35, w_l=0.40` — tilted toward
the link, which is the hardest and last-learned stage.

**The handover is the design.** Far out only `Φ_approach` moves; once a drone is
close it saturates and `Φ_observe` takes over; once observing, only `Φ_link`
still improves. Three mission stages, each with a live gradient, no dead zones.

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

