# Decisions and reversals

Every entry here was **proposed, then rejected on evidence**. The point of this
file is that neither a human nor an AI session re-proposes them six months from
now, having forgotten why they died.

If you are about to suggest something on this list, read the row first. If you
still think it is right, the evidence is reproducible — go and disprove it.

Detailed numbers for the transmit-power nulls are in
[`NEGATIVE_RESULTS.md`](NEGATIVE_RESULTS.md). Full reasoning for anything else is
in the git log; commit messages are long on purpose.

---

## Errors in the original spec (corrected, tests pin them)

| Was | Now | Why it mattered |
|---|---|---|
| `SINR_dB = P_sig − (P_jam + N0)` | linear-domain sum, then convert | Adding two dBm values is a *product*. Returned ~**+100 dB** SINR for realistic urban links, silently deleting the jammer from every experiment. |
| TR 38.901 UMi for all links | TR 36.777 UMi-AV for A2G; FSPL + blockage for A2A | 38.901 UMi is specified for UE heights **1.5–22.5 m** and is not valid for aerial nodes. Doubly invalid for drone-to-drone above rooftop. |
| Noise floor hardcoded at −100 dBm | `−174 + 10log10(B) + NF` | Must track bandwidth. At 10 MHz / 7 dB NF it is −97 dBm. |
| Unbounded Shannon | `min(0.75·log2(1+SINR), 7.4)` b/s/Hz | Shannon is an upper bound; real NR caps at 256QAM. Unbounded reports fantasy throughput. |
| `B = 20 MHz` | `B = 10 MHz` | At 20 MHz, 5 Mbps needs only **−7.2 dB** SINR — satisfied by accident, jammer decorative. |
| `P_hover + α‖v‖²` | rotary-wing model (U-shaped) | Asserted hovering is cheapest. It is not: cruise at 13.3 m/s costs **0.64** of hover draw. Energy sets the observer's cost, so the sign error would have inverted the intended behaviour. |
| Multi-hop capacity undefined | `min(Cᵢ)/min(n,3)` + widest-path DP | The relay chain was the premise but was never specified. |
| `/n_hops` **and** full concurrent interference | `min(n,3)`, with `tx_mask` carrying the MAC | Double-counted the half-duplex cost. `/n` *is* the pure-TDMA schedule, which has no intra-chain interference to charge. Made a feasible 3-hop chain look infeasible. |
| Terminate on link loss > 5 steps | fixed-length episodes; failure is per-step | Two failures: the policy learns never to acquire so it can never fail, **and** a random initial policy dies by step 6 and never reaches the tracking phase at all. |

---

## Design directions abandoned

### Adaptive transmit power — three framings, three nulls
**Do not reintroduce Ptx as an action.** Evidence in `NEGATIVE_RESULTS.md`.

| Framing | Result | Reason |
|---|---|---|
| Energy saving | ~1.6 % of power draw | Raising the ceiling to fix it destroys the mission instead — at 40 dBm one drone spans any simulable map. |
| Interference management | **0.0 %** vs a fair baseline | One flow + routing-aware MAC + ≤3 hops ⇒ the reuse schedule never runs two transmitters at once. Nothing to manage. |
| Detectability / EMCON | 0.1–1.1 % | Exposure saturates: the observer must sit in the threat's LoS and is always detected; everyone else is already below a −100 dBm ESM floor. |

Condition **E4** reproduces the null empirically alongside the analysis.
The one untested route is **multiple concurrent flows**, which would create real
contention — listed as future work because it turns the project into a
distributed link-scheduling study.

### Ptx ceiling of 40 dBm
Briefly specified to enlarge the energy term. A *blocked* A2A link at 10 W still
carries 15 Mbps over 2.8 km, so a single drone spans any map up to 2 km and the
relay chain — the entire premise — becomes unnecessary. **Range grows with power
far faster than the mission area can absorb.** Fixed at 30 dBm.

### Refreshed cue ("external ISR asset updates every 10 s")
Incoherent: a sensor that can persistently track the HVT through a city makes the
swarm redundant. It was a mechanism invented to fix cue staleness with a
justification attached afterwards. **The correct fix was geometric** — start the
HVT 300–500 m away so transit is short enough that one cue survives it.

### Blind search for the HVT
Rejected as a *starting* condition. Exploration is RL's weakest point and a
sparse "found it" reward over 1500 m² would swamp the learning signal. The cue
exists to break directional symmetry, not to solve acquisition — difficulty comes
from street topology and from the target moving during transit. Legitimate as an
optional final curriculum stage, once tracking already works.

### Manhattan (and any uniform-tall city) as the map
Canyon ratio `H_b/W ≈ 8.3` gives a **4.8 m** across-street observation envelope —
the drone would have to hover within five metres of a moving vehicle, so the
observation task collapses into "be exactly overhead" with no spatial decision
left. Uniform-*low* cities (Paris, Barcelona) have the opposite problem: the drone
is above every roof, air-to-air never blocks, chains collapse to two hops.
**Frankfurt wins because it is heterogeneous** — low fabric gives a workable 36 m
envelope while the tower cluster still blocks A2A.

### Energy-driven role rotation as RQ3
A realistic airframe burns only **7 %** of a 548 Wh pack over a 240 s episode.
Batteries never bind, `Var(B)` stays tiny, λ has nothing to act on. That risks the
worst kind of null — *"there was nothing to explain"* rather than *"λ does not
explain it"* — which is uninterpretable. Replaced by **geometric handoff**, which
is forced by the environment rather than contingent on a reward parameter.
λ survives as a secondary ablation, made informative by randomising initial charge
in `[0.3, 1.0]`.

### skrl's `PettingZooWrapper` for training
Verified against the installed version: it round-trips every action and
observation through NumPy on each step (`untensorize_space` / `tensorize_space`)
and exposes `num_envs == 1`; its vectorized paths are Isaac Lab-only. That
contradicts the stay-in-VRAM rule and caps throughput at single-env Python speed.
The adapter is kept for API-compliance tests and visual debugging only.

### `SAGEConv` for the GNN rung
☠️ **Never.** It cannot ingest edge features at all, so it would silently collapse
the GNN rung into the DeepSets rung and leave RQ2 measuring nothing — and it is
the layer people reach for by default.

---

## Still open

| Question | Blocked on |
|---|---|
| `τ_c`, `τ_l` retuning | a running env (Block D) — safe to change, they live in the potential |
| Local height raster | whether the policy is visibly blind without it (Block D/G) |
| LoD2 vs OSM `building:levels` coverage for the Frankfurt box | Block B — **must be checked, not assumed** |
| Second city for cross-morphology transfer | Block B; candidates London City (similar structure, different topology) or Barcelona (maximum contrast) |
| Verifying TR 36.777 and rotorcraft constants against primary sources | you, with the actual documents — **do not cite numbers an AI produced** |
