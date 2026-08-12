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
because skrl's rollout storage wants one fixed-shape tensor per agent. All three
architectures consume `flat` and unpack it, so the padding is identical across
rungs by construction rather than by discipline.

Ego is 24, not 21: a persistent 3-dim vector to the cue was added, and no time
feature was ([`ENVIRONMENT.md`](ENVIRONMENT.md) → Observations).

## Layer choice — the edge features are the whole point
RQ2's GNN rung exists **only** to test whether link quality should modulate who a
drone listens to. If the layer cannot ingest edge features, the GNN rung silently
becomes the DeepSets rung and RQ2 measures nothing.

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
4. **Sanity floor:** any architecture must beat a random policy and at least
   match the B0 scripted heuristic. Failing that is a bug, not a finding.

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
interesting result lives in the **off-N and cross-city transfer** columns. A
clean null, reported as such, is still a contribution.

---

