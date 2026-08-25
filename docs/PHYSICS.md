# Physics and scenario

Everything in this file is **implemented and unit-tested** in
[`src/env/channel.py`](../src/env/channel.py),
[`src/env/routing.py`](../src/env/routing.py) and
[`src/env/energy.py`](../src/env/energy.py), with hand-computed assertions in the
co-located test files.

> **Do not change these formulas without updating the tests and checking against
> the cited standard.** They appear in the methodology chapter and must stay
> traceable. Every one of them replaced an error in the original project spec —
> see [`DECISIONS.md`](DECISIONS.md).


## Link classes — one model does not fit all
| Link | Model | Why |
|---|---|---|
| Drone ↔ drone (A2A) | FSPL + 20 dB blockage penalty when occluded | Both endpoints are above rooftop; a ground street-canyon model does not describe this at all. |
| Drone ↔ HVT / MCV (A2G) | **3GPP TR 36.777 UMi-AV** | TR 38.901 UMi is specified for UE heights 1.5–22.5 m and is **not valid for aerial nodes**. |
| Jammer → drone | Same A2G UMi-AV | Jammer is ground-mounted on the HVT. |

> ⚠️ The TR 36.777 coefficients in `channel.py` are marked `TODO(verify)`. Check
> them against the actual 3GPP document before citing. Same for the rotary-wing
> energy constants.

## 📏 TR 36.777 verification status — 2026-08-26, PARTIAL

The `TODO(verify)` on `channel.py`'s UMi-AV coefficients was worked, against
secondary literature rather than the 3GPP document (which is not reachable from
the dev environment). Result: **three of the four constants corroborated, one
still open.** ⛔ The marker stays until a human reads the UMi-AV table.

| constant | status |
|---|---|
| LoS intercept **30.9** | ✅ corroborated |
| LoS slope **22.25 − 0.5·log₁₀(h)** | ✅ corroborated |
| NLoS slope **43.2 − 7.6·log₁₀(h)** | ✅ corroborated, two independent sources (quoted as `4.32 − 0.76·log₁₀(h)`) |
| **NLoS intercept 32.4** | ⚠️ **NOT confirmed** |

⚠️ The open one has a specific reason for suspicion, which is why it is worth a
human's five minutes: **32.4 is also the intercept of TR 38.901's *terrestrial*
UMi Street-Canyon LoS model** (`32.4 + 21·log₁₀(d₃D) + 20·log₁₀(f_c)`). That is
exactly the neighbouring constant a transcription slip lands on. It is equally
possible the two genuinely coincide — but it cannot be assumed.

**What the pass also fixed and found:**

1. ☠️ **The LoS branch was missing its `max(FSPL, ·)`.** TR 36.777 floors LoS at
   free space; the code did not, so at short range it returned a path loss
   *below vacuum*. Added. 📏 **No measured number moved**, and provably so: the
   floor binds only below 9.5–18.7 m of 3-D separation while `ALT_MIN_M` = 40 m
   makes every drone↔MCV ray at least 40 m long. `test_channel.py` pins that
   vacuity against `ALT_MIN_M`, so lowering the altitude band trips it.
2. ⚠️ **Frequency extrapolation.** TR 36.777 is an **LTE** study item; using it
   at 3.5 GHz rests on the `20·log₁₀(f_c)` term scaling outside the bands it was
   fitted in. Standard and defensible — but say it in Chapter 3 rather than
   leaving it implicit.
3. ✅ **`blockage_db = 20.0` was worked and is closed** — see the section below.
4. Docstring correction: the claimed "NLoS ~17 dB above LoS" measures **15.0 dB**
   at the quoted geometry (16–30 dB across the operating band).

**Internal consistency, all passing:** NLoS ≥ LoS everywhere (min margin
+1.5 dB); LoS sits 1.1–2.7 dB above FSPL across the band, rising with distance as
a canyon model should; the hand-computed tests reproduce both branches to 1e-3.

### ✅ Is TR 36.777 still the right reference? Yes — checked 2026-08-26

📏 **TR 38.901 does not cover aerial UEs, through Release 19 (June 2025).** The
Rel-19 channel-model study adds Suburban Macro, realistic handheld UT antennas,
near-field propagation for extremely large arrays, spatial non-stationarity,
polarization variability and ISAC — and **no aerial UE heights**. Its UMi/UMa
remain specified for UT heights of 1.5–22.5 m, which is exactly the range this
project operates *above*.

Rel-18's UAV work item is RAN2/RAN3 in character — identification, mobility,
broadcast — and does not supersede the Rel-15 channel model. **So TR 36.777
remains the current 3GPP aerial path-loss reference and there is nothing newer to
migrate to.** ⚠️ And note what that implies: 3GPP has **no air-to-air model at
all**, which is why `pathloss_a2a_db` is a separate construction rather than a
rung of the same standard.

### 📏 The A2A blockage penalty — assumed, physically low, measurably harmless

`scripts/verify_blockage.py` regenerates everything here.

**The physics says 20 dB is too low.** Occluded A2A rays in the real Frankfurt
geometry do not graze — the median ray passes **60.5 m inside** the obstruction
(p25 10.9 m, p90 111.9 m), because at a 40–80 m altitude band the only blockers
tall enough to matter are the towers, not the ~20 m median fabric Block B
measured. The first Fresnel radius at the median 235 m link is **2.2 m**, so a
60 m depth is ~27 Fresnel radii — deep shadow, not diffraction fringe. Single
knife-edge (ITU-R P.526) over the measured depth distribution gives a **median
43.3 dB**, and **90.4 %** of occluded A2A links exceed the modelled 20 dB.

**But it governs almost nothing.** Of the occluded edges on B0's *chosen* relay
chain, only **16.8 % are A2A** — the other **83.2 % are drone↔MCV**, which runs
on the TR 36.777 NLoS branch and never touches this constant:

| `blockage_db` | 20 | 30 | 40 |
|---|---|---|---|
| B0 mission-capable | 59.7 % | 60.8 % | 59.5 % |

⚠️ **The honest methodology sentence is therefore not "20 dB is correct".** It is:
*the A2A blockage penalty is an assumed 20 dB; the physically-motivated value is
nearer 40 dB; the reported metric is insensitive to it across that range
(±0.7 pp), because 83 % of occluded chain edges are air-to-ground.* That is a
stronger position than a citation would have produced, because it is a statement
about the result rather than about the input.

⛔ **Do not change the constant on the physics alone.** It would re-derive every
number in Blocks D–F for a sub-IQR effect, and the environment is frozen. Re-open
only if a rung-by-rung sweep shows a rung where it binds — F2/F3 are untested;
F0/F1 never reach this code at all, because `binary_capacity` skips path loss.

### ✅ A2A is a different model, deliberately, and that is correct

TR 36.777 covers **air-to-ground only** — an aerial UE against a ground-mounted
eNodeB. It says nothing about drone↔drone links where both ends are above
rooftop, and applying a street-canyon model to a ray that never enters the canyon
would be a category error. `pathloss_a2a_db` is therefore FSPL + a blockage
penalty, dispatched by `core.is_a2a`, and `test_channel.py` asserts the two paths
cannot converge.

📏 **The free-space choice has direct empirical support:** measurement-based A2A
modelling in built-up areas finds that when both UAVs are above **50 m**, A2A
path loss is well described by free space (arXiv:2301.12229). ⚠️ This project's
band is **40–80 m**, so the bottom of it sits just under that finding — one
sentence in the methodology, not silence.

## SINR — linear domain, with intra-swarm interference
```
SINR_lin(i→j) = P_rx(i→j) / ( Σ_{k∉{i,j}, k active} P_rx(k→j) + P_jam(j) + N0 )
SINR_dB       = 10·log10(SINR_lin)
```
Interference and noise sum in the **linear** domain. The earlier spec had
`SINR_dB = P_sig − (P_jam + N0)`, which adds two dBm quantities — a product, not
a sum — and returned ~+100 dB for realistic urban links, silently deleting the
jammer from every experiment. A regression test pins this.

Node `j`'s own transmission is excluded via a zeroed diagonal — half-duplex, it
does not receive its own emission.

**`tx_mask` carries the MAC assumption — set it deliberately.** The routing
divisor `min(n_hops, 3)` presumes a spatial-reuse TDMA schedule, under which a
≤3-hop chain never has two hops active at once. So when evaluating a link, the
mask must contain only the transmitters active *in that slot* — for short chains,
one node, and SINR reduces to signal over jammer-plus-noise. Passing every node
while also applying the divisor double-counts the half-duplex cost, and made a
feasible 3-hop chain look infeasible during scenario design. The
uncoordinated-access mode (all nodes concurrent, no divisor) stays available for
worst-case analysis. Pinned by tests.

## Noise floor — derived, never hardcoded
```
N0_dBm = -174 + 10·log10(B_Hz) + NF_dB        # B=10 MHz, NF=7 dB → -97.0 dBm
```

## Rate — Shannon with implementation loss and a modulation cap
```
SE     = min( 0.75 · log2(1 + SINR_lin), 7.4 )   b/s/Hz
C_Mbps = B_Hz · SE / 1e6
```
Unbounded Shannon reports throughput no real radio delivers.

## Multi-hop end-to-end capacity and routing
```
C_e2e = min_i(C_i) / min(n_hops, 3)            # half-duplex with spatial reuse
```
Half-duplex relays on one channel must be scheduled, but hops far enough apart
transmit concurrently, so a linear chain saturates near **1/3** of single-link
capacity rather than degrading as `1/n` (Li et al., MobiCom 2001; cf. Gupta &
Kumar 1999).

> A `/n_hops` divisor **plus** full concurrent interference double-counts: `/n`
> is the pure-TDMA schedule, in which only one hop is active and there is no
> intra-chain interference to charge. The two cannot both be true. `min(n, 3)`
> is the form consistent with the interference model in `channel.py`.

Short chains are still preferred — that pressure now comes from physics rather
than an arbitrary factor: every extra hop is another concurrent transmitter
raising everyone's noise floor, and must itself clear the SINR bar.

`reuse_limit` is a **parameter, not a constant** (`=max_hops` recovers strict
TDMA, `=1` removes the penalty). Report the headline result under at least two
duplexing settings — it converts a soft modelling assumption into a robustness
check.

Path selection maximises `min_i(C_i)/min(n,3)` via a hop-limited widest-path DP:
```
W[h][j] = max_i min( W[h-1][i], C[i][j] )      # answer: max_h W[h][dst]/min(h,3)
```
Sources are all drones currently holding a valid HVT observation; if none, mission
capacity is 0. Fully batched, exact, no per-env Python loop.

## Bandwidth and threshold — chosen so the constraint actually binds
`B = 10 MHz`, threshold **`15 Mbps`** end-to-end. At the originally-specified
20 MHz a single hop needed only −7.2 dB SINR, which a swarm satisfies by accident
and which makes the jammer decorative. Narrowing to 10 MHz fixed that for the
*single-hop* case.

> ⚠️ **The threshold was 5 Mbps and is now 15**, changed in Block E on measured
> evidence rather than on the link-budget argument above — which turned out to be
> necessary but not sufficient. At 10 MHz / 5 Mbps a 3-hop chain needs +4.8 dB
> SINR per hop, which sounds binding and is not: measured over real geometry the
> chain's *bottleneck* carries a median **37.6 Mbps, 8× the bar**, so
> `mission_capable` reduced to `observed` for every policy tested, the multi-hop
> divisor `min(n,3)` changed nothing, and a scripted baseline reached 93 %.
>
> At 15 Mbps the bottleneck sits at ~3× the bar, the divisor flips 27 % of
> chain-steps, and the binding constraint moves from the sensor to the relay
> chain. Both values are defensible for the payload — 5 Mbps is one compressed HD
> stream, 15 is a dual EO/IR feed at low latency — so this chose among defensible
> values on measurement rather than inventing one.
>
> **The lesson for anyone re-deriving this**: a per-hop SINR margin computed at
> the *threshold* says what the chain must clear, not what it actually carries.
> Only the second number tells you whether the constraint binds. Full ledger in
> [`DECISIONS.md`](DECISIONS.md), measurements in [`BLOCK_E.md`](BLOCK_E.md).

The constant lives in exactly one place, `src/env/reward.py`
(`CAPACITY_THRESHOLD_MBPS`), and everything else imports it — three scripts kept
private copies before Block E, which is how a stale bar silently re-derives the
altitude ceiling.

## Scenario — derived, not chosen
Every parameter is fixed from an external source, and the operating area is then
*solved for* so that a single drone fails while the swarm succeeds. Regenerate
with [`scripts/scenario_design.py`](../scripts/scenario_design.py) and
[`scripts/link_budget_check.py`](../scripts/link_budget_check.py);
`tests/test_scenario_sizing.py` pins the trade-off table.

| Parameter | Value | Basis |
|---|---|---|
| City | **Frankfurt**, 1500 m box over Bankenviertel + fabric | heterogeneous: low-rise gives a workable observation envelope, towers block A2A |
| Operating area | **1500 m** | solo drone manages ~1.7 Mbps (fails); swarm ~24 Mbps (feasible) |
| Ptx | **30 dBm, fixed** | UAV tactical MANET radios are 0.5–2 W |
| Jammer, in-band | 30 dBm | vehicle C-UAS barrage emitter |
| Flight altitude | band **40–80 m** | above fabric, below towers; inside TR 36.777's 22.5–300 m band. Both ends derived below |

> ⚠️ **Never raise Ptx to make the energy term measurable.** At 40 dBm a
> *blocked* A2A link still carries 15 Mbps over 2.8 km, so one drone spans any
> simulable map and the relay chain becomes unnecessary. Range grows with power
> far faster than the mission area can absorb.

## Observation envelope — an angle constraint, not a distance one
The ray must clear the roofline, which fixes an elevation angle (~66° for
Frankfurt), not a range:
- **across-street:** within `(W/2)·h/H_b` — 36 m at 80 m altitude, 91 m at 200 m.
  Flying higher buys lateral freedom.
- **along-street:** the roofline never blocks; the sensor limits instead
  (~830 m to recognise a vehicle, ~2.8 km to detect one).

> **Measured (Block B).** Both bullets were assumptions; both are now checked
> against the real box (`scripts/measure_sightlines.py`).
> Across-street envelope: median **43 m**, p10–p90 **24–88 m** — the 36 m figure
> is a fair central value but the spread is a factor of ~3.7, so report the
> distribution. Canyon ratio `H_b/W`: median **0.93** against the assumed 1.10;
> street width median **21 m** against the assumed 20 m.
> Along-street sightline: median **127 m**, p90 387 m, and **99.8 % fall below
> 830 m**. The sensor range is therefore a **non-binding ceiling** — occlusion
> cuts first everywhere — which is what keeps RQ1 measuring channel physics
> rather than sensor specification. The 830 m / 2.8 km pair itself has no
> derivation in this repo; treat it as unverified and do not defend the exact
> value. See [`BLOCK_B.md`](BLOCK_B.md) → "The 830 m recognition range".

So the envelope is a wedge down the street plus an overhead cone — **not a
36 m disc**. Compute it from real footprints, never from a radius.

## Sensor model — 360°, occlusion-limited

`sees = (clearance ≥ 0) & (range ≤ 830 m)`. No pointing state, no attitude, no
slew: a **gimballed EO/IR turret** has continuous 360° azimuth and full downward
elevation and holds lock once tracking, so at `dt = 0.4 s` with one target
"pointed at the target" is a fair approximation. Modelling it properly needs a
pointing state plus a pointing action or auto-tracker — scope creep into gimbal
control, excluded for the same reason as rigid-body flight dynamics.

**Where this is optimistic is search, not tracking**, so it props up exactly one
result: the no-cue ablation ([`ENVIRONMENT.md`](ENVIRONMENT.md)). Report that one
as *"unaided acquisition is feasible given a gimballed sensor with negligible
slew cost"*, never as *"the cue is unnecessary"*. A minimum depression angle was
considered and rejected — it would trade one unsourced constant for two, and a
binding sensor parameter confounds RQ1 ([`DECISIONS.md`](DECISIONS.md)).

## Altitude band — 40 to 80 m, and both ends are derived

Nothing in the model charges for altitude (propulsion power is speed-only, a full
climb is 0.55 % of the pack), and both physical effects improve with height, so
**the band is the entire altitude policy** — expect `a_z` to saturate at the
ceiling. Measured with the production kernel on the real boxes
([`../scripts/measure_envelope.py`](../scripts/measure_envelope.py)):

| altitude | A2A links blocked | HVT visible at 100–200 m offset | drone inside a building box |
|---|---|---|---|
| 40 m | 44.6 % | 22.7 % | 3.3 % |
| 80 m | 31.2 % | 38.2 % | 1.9 % |
| **120 m** | **24.6 %** | **48.1 %** | 1.4 % |
| 180 m | 10.2 % | 55.9 % | — |
| 230 m | **0.0 %** | — | — |

**Ceiling — the scenario's own definition, not a comfort margin.** The mission is
*defined* as one a single platform cannot accomplish; that is what makes it a
swarm problem and it is condition **W1** in `scenario_design.py`. Measured on real
geometry, a best-placed solo drone hovering over the HVT is mission-capable at
maximum separation **3.3 % of the time at 80 m, 23.2 % at 100 m, 57.4 % at 120 m**.
So 80 m is the altitude at which the scenario remains the scenario. Raising it
also weakens RQ1 in the same motion, since A2A blockage falls from 31 % to 25 %.

Civil UAS rules (the EU open category is believed to cap at 120 m AGL) are
consistent with this but are **corroboration, not justification** — the number is
derived from W1, so no regulatory citation is load-bearing and the earlier
`TODO(verify)` is discharged.

The consequence to state plainly in the methodology: with the ceiling equal to
the nominal altitude, and every gradient pointing up, the vertical action
dimension is effectively degenerate — the swarm flies at 80 m and the interesting
decisions are horizontal.

**Floor:** a *model-validity* limit, not a flight rule. At 10 m altitude 37 % of
positions sit inside a building box, and `occlusion.py` ignores boxes containing
an endpoint — a convention chosen for a 1 % case — so the drone would see through
the building it stands in. Separately `pathloss_a2g_umi_av_db` clamps `h` to
22.5 m, silently substituting a different altitude below TR 36.777's floor.

## Why altitude does not remove the need for a relay chain

Both endpoints that matter are on the ground. The ground-LoS radius is
`(W/2)·h/H_b` — 50 m at 80 m, 75 m at 120 m — so a clear ray to a ground node
1400 m away would need **2240 m** of altitude. Climbing makes the *middle* of the
chain cheap and leaves both *ends* exactly as hard. At 120 m, 1400 m from the MCV:

| | capacity |
|---|---|
| single hop, **clear** ray | **46.1 Mbps** |
| single hop, **blocked** ray | 4.8 Mbps |
| 2-hop: A2A 1200 m + A2G 200 m | **27.6 Mbps** |
| threshold | 5.0 Mbps |

**The chain is required by blockage, not by range** — which is what makes RQ1 a
real question rather than a formality: a connectivity-radius model would capture
a range requirement perfectly, so there would be no gap to measure.

## Energy — implemented in [`src/env/energy.py`](../src/env/energy.py)
```
P(V) = P_0·(1 + 3V²/U_tip²)                              # blade profile  ↑ with V
     + P_i·(√(1 + V⁴/4v_0⁴) − V²/2v_0²)^½                # induced        ↓ with V
     + ½·d_0·ρ·s·A·V³                                    # parasite       ↑ with V
P_total = P(‖v‖)/η_drivetrain + κ·‖a‖² + P_tx_DC
```
Standard rotorcraft aerodynamics, as presented in **Zeng, Xu & Zhang (2019)**.
Induced power *falls* with forward speed faster than profile power rises, so the
curve is **U-shaped** — for the default airframe the minimum sits at **13.3 m/s
and costs 58 % less than hovering**.

> The earlier `P_hover + α‖v‖²` form asserted the opposite, that hovering is
> cheapest. Energy sets the cost of the observer role, so that error would have
> inverted the behaviour the reward is meant to produce. A regression test pins
> it.

**Constants are derived, not quoted.** Momentum theory gives the dominant hover
term from mass and rotor geometry alone (`v_0 = √(W/2ρA)`), which is checkable in
a way a copied table is not — and it yields a validation a paper's example
constants cannot: **predicted endurance against published flight time.** The
default ~5.9 kg / 21-inch airframe predicts **56.8 min** of hover on 548 Wh
against ~55 min published.

Two details that matter numerically:
- Battery drain is **electrical**, not shaft — divide by drivetrain efficiency
  (~0.80). Omitting it overstates endurance by ~25 %.
- The induced bracket is evaluated as `1/(√(1+x²)+x)`, algebraically identical to
  `√(1+x²)−x` but without the catastrophic cancellation.

**Climb power** `W·v_z/η` is added on top for vertical motion — real physics,
traceable, and cheap. It does **not** bind: a full 40 → 120 m climb at 5 m/s
costs ~0.55 % of a 548 Wh pack, which is why the altitude band rather than the
energy term is what controls altitude (see above).

`κ‖a‖²` is an explicit **control-effort heuristic**, not physics; it defaults to
zero and must be opted into. `P_tx_DC` is a **constant** (Ptx is fixed), ~7 W or
1.6 % of draw — kept for completeness, but flight energy is what the policy
controls.

> ⚠️ **Battery does not bind in one episode.** 240 s of hovering burns ~7 % of a
> 548 Wh pack. This measurement is what reframed RQ3 from energy-driven rotation
> to geometric handoff, and why initial charge is randomised in `[0.3, 1.0]`.
> A test asserts it stays under 25 %; if that ever fails, revisit RQ3.

> ⚠️ `tip_speed_ms`, `solidity`, `profile_drag_coeff`, `fuselage_drag_ratio` are
> `TODO(verify)` — not usually published per airframe, so they use documented
> typical ranges. Same standing as the TR 36.777 coefficients.

## Graph
- GNN edge weight (continuous, no hard cutoff — avoids gradient cliffs):
  `E_ij = sigmoid((C_ij − 5.0) · gamma)`
- Jammer is mounted on the HVT and moves with it.

---

