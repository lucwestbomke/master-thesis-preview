# Negative results: why transmit power control is not the research question

This project began as a study of **joint motion and transmit-power control**. Three
independent justifications for adaptive transmit power were tested numerically
against fair baselines, before any training code was written. All three came out
null, for three different structural reasons.

This is recorded rather than deleted for two reasons. It belongs in the thesis —
a reader considering the same design deserves the answer — and it is the evidence
that the final research question was chosen on grounds rather than taste.

Reproduce with [`scripts/scenario_design.py`](../scripts/scenario_design.py) and
[`scripts/link_budget_check.py`](../scripts/link_budget_check.py).

---

## 1. Energy: the telecom term is too small to measure

**Claim tested:** adaptive Ptx saves meaningful energy against fixed Ptx.

At a realistic ceiling, transmit power is a rounding error against flight power.
With `P_hover ≈ 250 W` for a 2 kg quadrotor and a 25 % PA efficiency:

| Ptx | RF | DC draw | Share of total |
|---|---|---|---|
| 20 dBm | 0.1 W | 0.4 W | 0.16 % |
| 30 dBm | 1 W | 4 W | **1.6 %** |
| 40 dBm | 10 W | 40 W | 13.8 % |

1.6 % is unmeasurable against seed variance in RL returns.

**Why raising the ceiling does not fix it.** Link range grows with transmit power
far faster than a simulable urban operating area can absorb. At 40 dBm a
*blocked* drone-to-drone link still carries 15 Mbps over 2.8 km and 5 Mbps over
6.3 km, so a single drone spans any map up to 2 km and the relay chain — the
premise of the whole scenario — becomes unnecessary:

| Map | Ptx 10 | 20 | 30 | 40 |
|---|---|---|---|---|
| 300 m | infeasible | **trivial** | **trivial** | **trivial** |
| 600 m | infeasible | contested | **trivial** | **trivial** |
| 1200 m | infeasible | infeasible | contested | **trivial** |
| 2000 m | infeasible | infeasible | contested | **trivial** |

The energy share and the mission's existence are in direct opposition. There is
no ceiling that satisfies both.

---

## 2. Interference: with one flow and a MAC, there is nothing to manage

**Claim tested:** adaptive Ptx raises end-to-end throughput by managing
intra-swarm interference.

Measured against a **fair** baseline — fixed transmit power plus ordinary
routing-aware medium access, where only nodes on the selected path transmit —
exhaustive per-drone power allocation *with full state knowledge* gives:

**0.0 % improvement**, at every map size (1200/1500/2000 m), every jammer power
(20/30/40 dBm), and every spare-drone placement (50 m to 3200 m offset).

**Why.** With a single data flow and a chain of at most three hops, the
spatial-reuse schedule never runs two transmitters concurrently. There is no
interference to manage, so maximum power is optimal for every node on the path.

An earlier version of this analysis appeared to show a ~4000 % gain. That
baseline had all five drones transmitting continuously — no medium access
control at all. It measured the absence of a MAC, not the value of power
control, and would not survive a defence.

**What would restore it.** Contention has to come from somewhere structural:
concurrent flows (≥2 simultaneous sensor feeds), or chains of ≥4 hops where the
reuse period forces overlap. Multiple flows is the one route not numerically
ruled out here; it is listed as future work because it turns the project into a
distributed link-scheduling study rather than a multi-agent control one.

---

## 3. Detectability: the exposure landscape saturates

**Claim tested:** adaptive Ptx reduces the probability of being located by a
threat ESM/DF receiver, at equal throughput.

Modelled as `p_det(i) = sigmoid((P_rx,threat(i) − S_esm)/T)` with
`S_esm = −100 dBm` (wideband ESM sensitivity over 10 MHz) and `T = 3 dB`. The
path loss drone→HVT is the jammer path in reverse, so it costs nothing to add.

Minimum total exposure at equal throughput (≥5 Mbps):

| Relay position | Best fixed Ptx | Best per-drone Ptx | Reduction |
|---|---|---|---|
| 200 m from HVT | 1.827 | 1.826 | 0.1 % |
| 600 m | 1.062 | 1.061 | 0.1 % |
| 900 m | 1.013 | 1.002 | 1.1 % |
| 1200 m | 1.005 | 1.000 | 0.5 % |
| joint over relay position too | 1.003 | 1.000 | **0.3 %** |

**Why.** The landscape is binary. The observer must sit inside the observation
envelope to see the HVT, which puts it in the threat's line of sight; at any
power that closes a link it is detected, `p_det = 1.00`. Every other drone is
already below the ESM floor at full power — a relay 1500 m away and shadowed
receives −104.6 dBm at 30 dBm transmit. Total exposure is ≈1.0 in every
configuration, so there is no gradient: detection saturates on both sides.

---

## Conclusion

Three framings, three unrelated structural reasons, one consistent answer: **in
this scenario there is no interesting transmit decision to make.** Links are
either easily closed or impossible, with little middle ground where fine power
tuning matters; a single flow under ordinary medium access creates no contention;
and the threat is either adjacent or irrelevant.

The hard part of the problem is **motion under occlusion** — holding a sensor on a
vehicle moving through a street network while buildings cut the line of sight, and
reconfiguring a relay chain around them. That is where the thesis went.

Condition E4 in the experiment matrix reproduces this empirically with a learned
4-dim (motion + power) policy, so the finding rests on both analysis and
experiment.

---

## A modelling inconsistency this exposed

Worth recording separately, because it is a live constraint on the code rather
than a discarded idea.

`routing.py` divides end-to-end rate by `min(n_hops, 3)`, which presumes a
spatial-reuse **TDMA schedule**. `channel.py` computes SINR with every active
node treated as a concurrent interferer. Both cannot be true at once: under a
reuse-3 schedule a ≤3-hop chain never has two hops active simultaneously, so
intra-chain interference is zero.

Resolution: the MAC assumption is now explicit, and `sinr_db`'s `tx_mask` is the
parameter that carries it — callers pass only the transmitters active in the slot
being evaluated. Under the scheduled MAC with a ≤3-hop chain that is a single
transmitter, and per-link SINR reduces to signal over jammer-plus-noise. The
uncoordinated-access case (all nodes concurrent) remains available for worst-case
analysis. See the tests in `src/env/test_channel.py`.
