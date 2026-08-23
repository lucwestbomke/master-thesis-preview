# Model architectures

Actor/critic design and the rules that keep RQ2's comparison valid. Research
framing is in [`THESIS_PLAN.md`](THESIS_PLAN.md).


> Layer choice is now settled (custom MPNN — see below). Widths and depths remain
> hyperparameters for the equal-budget search, not findings.

## The ladder isolates one factor per rung
| | Neighbours read as | Permutation-invariant | Size-agnostic | Uses link quality |
|---|---|---|---|---|
| Flat MLP | concatenated vector, max-N padded + masked | ✗ | ✗ | ✗ |
| DeepSets | `ρ(Σᵢ φ(xᵢ))` — shared embed, then pool | ✓ | ✓ | ✗ |
| GNN | same, messages weighted by `edge_weight` | ✓ | ✓ | ✓ |

MLP → DeepSets isolates permutation invariance. DeepSets → GNN isolates the
*relational* part, which is RQ2's actual claim. Comparing a GNN only against a
flat MLP conflates the two and is the weaker experiment.

The MLP needs **max-N padding plus masking** or it cannot be evaluated off-N at
all, which would rig the transfer comparison toward the GNN.

**The env delivers that padding.** [`BLOCK_D.md`](BLOCK_D.md) fixes the
observation contract: structured keys `ego (B,N,24)`, `neighbour (B,N,N-1,9)`,
`edge (B,N,N-1,2)` for PyG batching and debugging, plus a **`flat (B,N,108)`**
packing at `N_max = 8` — 24 ego + 7×9 neighbour + 7×2 edge + 7 validity bits —
because skrl's rollout storage wants one fixed-shape tensor per agent.

**And the unpacking is shared, so the padding is identical by construction rather
than by discipline.** Block E added `core.unpack_flat()` — the inverse of the
env's own `_pack`, living next to it — which B0 and all three architectures
consume. A second, hand-rolled unpacking somewhere else is exactly how that
guarantee gets lost.

Ego is 24, not 21: a persistent 3-dim vector to the cue was added, and no time
feature was ([`ENVIRONMENT.md`](ENVIRONMENT.md) → Observations).

## Layer choice — the edge features are the whole point
RQ2's GNN rung exists **only** to test whether link quality should modulate who a
drone listens to. If the layer cannot ingest edge features, the GNN rung silently
becomes the DeepSets rung and RQ2 measures nothing.

> ⚠️ **The same failure can happen without touching the layer, and nearly did.**
> The edge capacity feature is normalised to threshold units and clamped at
> `CAPACITY_CLAMP`. That was 4.0 against a 5 Mbps threshold, i.e. a 20 Mbps
> ceiling — while real drone-drone links run to 74 Mbps. Measured under B0,
> **57.5 % of capacity values sat pinned at the clamp** and only ~36 % varied at
> all. A GNN cannot weight messages by a constant, so the rung would have been
> handicapped by a *normalisation constant* rather than by the architecture, and
> the resulting null would have looked like a finding about relational structure.
> `CAPACITY_CLAMP` is now 5.0 against a 15 Mbps threshold — a 75 Mbps ceiling,
> just above the 7.4 b/s/Hz × 10 MHz physical cap — so the observation saturates
> only where the physics does. Informative share: **93.7 %**.
>
> **Check this before trusting any RQ2 null.** If the feature does not vary,
> nothing downstream of it can.

| PyG layer | Edge features | Verdict |
|---|---|---|
| `SAGEConv` (GraphSAGE) | **none** | ☠️ **Never use here.** Collapses GNN into DeepSets. This is the default people reach for. |
| `GCNConv` | scalar weight, degree-normalised | Poor fit — the normalisation assumes a different graph structure |
| `GATv2Conv` | ✓ via `edge_dim` — enters the attention weights | Good fit |
| `NNConv` | ✓ — edge features generate the message weight matrix | Expressive but the hypernetwork emits 256×256 values. Expensive. |
| `GINEConv` | ✓ additive only (`x_j + e_ij`) | Cheap, blunt |
| `TransformerConv` | ✓ | Heavier than this graph needs |

**Decision: a custom layer on PyG's `MessagePassing` base**, with
`message(x_i, x_j, e_ij) = MLP([x_i, x_j, e_ij])`.

This is *not* inventing an architecture — it is the standard MPNN formulation of
Gilmer et al. (2017), ~20 lines on top of PyG, and fully citable. It is preferred
here because **it makes the ablation exact**: the DeepSets rung is the identical
layer with `e_ij` zeroed. Same code path, same parameter count, same optimiser,
one input masked. No confound is possible. Two differently-named layers would
always invite "maybe GATv2 is just a better layer."

Fallback if an off-the-shelf named layer is preferred: **`GATv2Conv` with
`edge_dim=2`**. Attention fits conceptually — "how much should I listen to this
neighbour" is exactly what link capacity says — and GATv2 (Brody et al., 2022)
fixed the static-attention flaw in the original GAT, so it is the right citation.

## Rules that keep the comparison honest
1. **Do not invent an architecture.** Either the MPNN formulation above or a
   citable PyG layer. Designing a novel GNN is a different thesis.
2. **Equal hyperparameter budget** across all three, and say so in the
   methodology. Tuning the GNN harder than the baselines is the single most
   likely way this result gets dismissed.
3. **Match parameter counts** to within ~20 %, so the comparison is not
   capacity-vs-capacity.
4. **Sanity floor, now with numbers.** Any architecture must beat a random policy
   and at least match B0. Failing that is a bug, not a finding. Measured in
   [`BLOCK_E.md`](BLOCK_E.md), eval split, 5 seeds × 64 episodes, median [IQR],
   at the **15 Mbps** requirement:

   | | mission-capable |
   |---|---|
   | random | 10.9 % [1.1] |
   | `B0-geodesic` | 47.1 % [3.1] |
   | **B0 — the floor to match** | **57.2 % [3.5]** |
   | `B0-oracle` (upper bound) | 56.8 % [3.2] |
   | *sensor-only ceiling* (`observed`) | *93.0 %* |

   **~36 points of headroom, and all of it is relay geometry** — the swarm can
   see the target 93 % of the time and deliver the feed 57 % of the time. That is
   the gap a learned policy has to close, and it is exactly the coordination
   problem RQ2 asks whether relational structure helps with.

   Two things to plan around:

   - **The N-transfer columns now measure the chain**, and **N = 8 is where RQ2's
     claim is testable.** B0 scores 36.4 / 57.2 / 74.3 % at N = 3/5/8 while
     `observed` stays flat at ~93 %, so off-N transfer is a *relay-scaling* test —
     which is what RQ2 wanted it to be. But the informative column is N = 8, not
     N = 3: the `B0-geodesic` → `B0` gap, i.e. what better control is *worth*, runs
     **+3.2 / +10.1 / +25.9 pp**. Three drones on a three-hop chain have one viable
     arrangement and cleverness buys almost nothing; eight have many. It is also
     the `N_MAX = 8` padding boundary, where a flat MLP is at its limit and the
     size-agnostic rungs should not be — exactly the contrast RQ2 tests.

     **Do not train at more than one N to exploit this.** It would turn the
     zero-shot transfer columns into in-distribution tests and delete the
     experiment ([`DECISIONS.md`](DECISIONS.md)).
   - **Difficulty is concentrated late in the episode** (capable decays 84 % →
     35 % as the HVT drives out). An architecture that only fixes the opening
     will not show up. Report the second half separately.

   Regenerate with `uv run python scripts/eval_baseline.py --only ladder transfer`.

## Depth follows graph diameter — and "layer" means two different things
Do not confuse these:

- **Message-passing layers** = how far information travels across the graph. One
  layer reaches direct neighbours; two reaches neighbours-of-neighbours. Nothing
  to do with capacity.
- **MLP hidden layers** = ordinary network depth, inside each message-passing
  layer and in the heads. This is where capacity lives.

The graph is softly fully connected at `N ≤ 8`, so its diameter is **1**: after
one message-passing layer every drone has already heard every other. A second
layer buys two-hop relational structure. A third propagates nothing new and
causes **over-smoothing**, where all node representations converge — a documented
GNN failure mode, not a rule of thumb.

So **2 message-passing layers** is the ceiling the graph justifies, while width
stays normal. A reasonable build:

| Component | Shape | Params |
|---|---|---|
| Ego encoder | 24 → 256 → 256 | ~70k |
| Message function φ (×2 layers) | (256+256+2) → 256 → 256 | ~400k |
| Policy head | 256 → 256 → 6 | ~67k |
| **Total actor** | | **~550k** |

Width is a hyperparameter and belongs in the equal-budget search; 256 is the
starting point, not a finding.

## Expect a null on the in-distribution rung
At `N=5` the graph is tiny and GNN ≈ DeepSets is a plausible outcome. The
interesting result lives in the **off-N transfer** columns — `N = 8` above all,
where better control is worth +25.9 pp against +3.2 pp at `N = 3`. A clean null,
reported as such, is still a contribution.

> The cross-city column was **cut** on 2026-08-23: Hessen's LoD2 service covers
> no other city, so a second map is a full Block-B rebuild rather than "one extra
> OSM extract" ([`DECISIONS.md`](DECISIONS.md)). RQ2's generalisation claim is now
> swarm size only.

Block E leaves this expectation where it was, rather than strengthening it. At
the 15 Mbps requirement a scripted controller reaches 57.2 % against a 93.0 %
sensor ceiling, so the headline metric has real room and the three rungs can
separate on it. (At the original 5 Mbps bar B0 scored 93.2 % and the metric was
saturated — that is what the requirement change fixed.)

---

