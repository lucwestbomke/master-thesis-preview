"""
Multi-hop relay routing: which chain carries the sensor feed to the MCV, and
what end-to-end rate it delivers.

The original project spec asserted a "multi-hop relay chain" but never said how
the path is selected or how end-to-end throughput is computed. Both are defined
here, and both materially shape the optimal swarm geometry.

Why half-duplex at all
----------------------
A node with one radio cannot transmit and receive on the same frequency: its own
emission arrives ~100 dB above the wanted signal. In-band full duplex needs
elaborate self-interference cancellation and is not fielded on small UAVs.
Frequency-division relaying (one clean channel per hop) would dodge the problem,
but assumes spectrum abundance -- which is exactly what a contested EW scenario
denies. Carrying one radio per hop is ruled out by SWaP. So half-duplex,
single-channel is the honest default. It is nonetheless an assumption, so
`reuse_limit` exposes it: see the robustness note below.

End-to-end rate model
---------------------
Half-duplex relays sharing one channel must be scheduled, but hops far enough
apart can transmit at the same time. Adjacent hops cannot (they share a node);
next-nearest usually cannot (interference range exceeds transmission range).
The classic result for a linear chain is therefore a *constant* penalty rather
than a per-hop one -- throughput saturates near 1/3 of single-link capacity
regardless of chain length (Li et al., MobiCom 2001; cf. Gupta & Kumar 1999):

    C_e2e = min_i(C_i) / min(n, reuse_limit)          # reuse_limit = 3

An earlier revision of this module used `/ n` for all n while *also* charging
full concurrent intra-swarm interference in channel.py. That double-counts: `/n`
is the pure-TDMA schedule, in which only one hop is ever active and there is no
intra-chain interference to charge. Concurrent transmission and a `/n` divisor
cannot both be true. The `min(n, 3)` form is the one consistent with the
interference model actually implemented.

Pressure toward short chains does not disappear with the constant divisor -- it
just comes from physics instead of an arbitrary factor. Every extra hop is
another concurrent transmitter raising everyone's noise floor, and must itself
clear the SINR bar.

`reuse_limit` is deliberately a parameter, not a constant: setting it to
`max_hops` recovers the strict-TDMA `/n` model and setting it to 1 removes the
half-duplex penalty entirely. Reporting the main result under more than one
duplexing assumption turns a soft spot in the modelling into a robustness check.

Path selection
--------------
Maximising `min_i(C_i) / min(n, reuse_limit)` jointly over path and hop count is a hop-limited
widest-path (maximum-bottleneck) problem. Solved exactly by a small dynamic
program over hop count:

    W[h][j] = max_i  min( W[h-1][i], C[i][j] )

`W[h][j]` is the best bottleneck capacity reachable at j within h hops. The
answer is `max_h W[h][dst] / min(h, reuse_limit)`. With M <= 9 nodes the DP is a
handful of (B,M,M) reductions -- fully batched, no Python loop over
environments, and exact.

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
    reuse_limit: int = 3,
) -> torch.Tensor:
    """Best achievable end-to-end mission capacity, over all sources and chains.

    Parameters
    ----------
    cap_mbps    : (B, M, M) per-link capacity, `[..., i, j]` = i -> j
    source_mask : (B, M) bool, nodes currently holding a valid HVT observation
    dst_index   : index of the MCV
    max_hops    : chain length limit (M - 1 is the useful maximum)
    reuse_limit : spatial-reuse period of the half-duplex schedule. 3 is the
                  standard linear-chain result; pass `max_hops` for strict TDMA
                  (`/n`), or 1 to disable the half-duplex penalty. Report the
                  headline result under at least two settings.

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
        divisor = float(min(hops, reuse_limit))
        best = torch.maximum(best, frontier[:, dst_index] / divisor)

    # No observing drone means there is no feed to relay, regardless of link
    # quality. Kept as a tensor op -- calling .item() here would force a GPU
    # sync inside the env step, which the project's device rules forbid.
    no_source = ~valid_src.any(dim=-1)
    return torch.where(no_source, torch.zeros_like(best), best)


def link_alive(capacity: torch.Tensor, threshold_mbps: float) -> torch.Tensor:
    """Discrete mission-success test on the end-to-end rate."""
    return capacity >= threshold_mbps
