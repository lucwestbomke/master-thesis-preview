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
`B = 10 MHz`, threshold `5 Mbps` end-to-end. A 3-hop chain then needs **+4.8 dB
SINR per hop**. At the originally-specified 20 MHz a single hop needed only
−7.2 dB, which a swarm satisfies by accident and which makes the jammer
decorative.

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
| Flight altitude | 80 m nominal | above fabric, below towers; inside TR 36.777's 22.5–300 m band |

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

So the envelope is a wedge down the street plus an overhead cone — **not a
36 m disc**. Compute it from real footprints, never from a radius.

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

