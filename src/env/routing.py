"""
Multi-hop relay routing: which chain carries the sensor feed to the MCV, and
what end-to-end rate it delivers.

The original project spec asserted a "multi-hop relay chain" but never said how
the path is selected or how end-to-end throughput is computed. Both are defined
here, and both materially shape the optimal swarm geometry.

End-to-end rate model
---------------------
A chain of half-duplex decode-and-forward relays cannot transmit and receive at
the same time, so an n-hop chain delivers at most

    C_e2e = min_i(C_i) / n

That divisor is what gives the swarm a reason to prefer *short* chains. Without
it, adding relays is free and the learned behaviour degenerates into long
daisy-chains that no real radio could sustain.

Path selection
--------------
Maximising `min_i(C_i) / n` jointly over path and hop count is a hop-limited
widest-path (maximum-bottleneck) problem. Solved exactly by a small dynamic
program over hop count:

    W[h][j] = max_i  min( W[h-1][i], C[i][j] )

`W[h][j]` is the best bottleneck capacity reachable at j within h hops. The
answer is `max_h W[h][dst] / h`. With M <= 9 nodes the DP is a handful of
(B,M,M) reductions -- fully batched, no Python loop over environments, and
exact.

Cycles need no explicit exclusion: revisiting a node can only lower the
bottleneck while increasing h, so a cyclic path is never optimal.
"""

from __future__ import annotations

import torch

# Sources start with unbounded bottleneck capacity. Finite (not inf) so that
# multiplying by a zero mask cannot produce NaN.
_SOURCE_SENTINEL = 1e9


def best_relay_capacity(
    cap_mbps: torch.Tensor,
    source_mask: torch.Tensor,
    dst_index: int,
    max_hops: int,
) -> torch.Tensor:
    """Best achievable end-to-end mission capacity, over all sources and chains.

    Parameters
    ----------
    cap_mbps    : (B, M, M) per-link capacity, `[..., i, j]` = i -> j
    source_mask : (B, M) bool, nodes currently holding a valid HVT observation
    dst_index   : index of the MCV
    max_hops    : chain length limit (M - 1 is the useful maximum)

    Returns
    -------
    (B,) end-to-end capacity in Mbps. Zero where no drone is observing the HVT
    -- no observation means there is no feed to relay, which is a mission
    failure regardless of how good the radio links are.
    """
    b, m, _ = cap_mbps.shape
    eye = torch.eye(m, device=cap_mbps.device, dtype=cap_mbps.dtype)
    cap = cap_mbps * (1.0 - eye)  # no self-loops

    # The MCV can never be a source; excluding it here is what stops the
    # sentinel from leaking through to the result.
    valid_src = source_mask.clone()
    valid_src[:, dst_index] = False

    src = valid_src.to(cap.dtype) * _SOURCE_SENTINEL  # (B, M)
    frontier = src.clone()
    best = torch.zeros(b, device=cap.device, dtype=cap.dtype)

    for hops in range(1, max_hops + 1):
        # W_new[j] = max_i min(W[i], cap[i, j])
        widened = torch.minimum(frontier.unsqueeze(-1), cap).amax(dim=1)  # (B, M)
        # Sources always remain available as fresh starting points, so a longer
        # chain never destroys a shorter one's option.
        frontier = torch.maximum(widened, src)
        best = torch.maximum(best, frontier[:, dst_index] / hops)

    # No observing drone means there is no feed to relay, regardless of link
    # quality. Kept as a tensor op -- calling .item() here would force a GPU
    # sync inside the env step, which the project's device rules forbid.
    no_source = ~valid_src.any(dim=-1)
    return torch.where(no_source, torch.zeros_like(best), best)


def link_alive(capacity: torch.Tensor, threshold_mbps: float) -> torch.Tensor:
    """Discrete mission-success test on the end-to-end rate."""
    return capacity >= threshold_mbps
