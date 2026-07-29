"""
Link-budget model: path loss, interference, SINR, and achievable rate.

Every function here is batched, pure-torch, and device-agnostic. Nothing calls
`.cpu()` or `.numpy()`, so this module can run inside a vectorized env step on
GPU. Leading dimension is `num_envs` throughout.

Conventions
-----------
- Power in dBm, distance in metres, carrier frequency in GHz, bandwidth in Hz.
- Node index layout: 0..N-1 are drones, N is the MCV base. The HVT is *not* a
  comms node -- it is the optical tracking target and the jammer source.
- Link matrices are (B, M, M) with `[..., i, j]` meaning "transmitter i to
  receiver j". The diagonal is always forced to zero power (a node does not
  receive its own transmission).

Model choices are justified in docs/THESIS_PLAN.md section 5. The two that most
need defending in the methodology chapter:

1. SINR is computed by summing interference and noise in the *linear* domain.
   The original project spec had `SINR_dB = P_sig - (P_jam + N0)`, which adds
   two dBm quantities -- in linear terms a product, not a sum. That formula
   yields ~+100 dB SINR for a realistic urban link, i.e. it silently removes the
   jammer from the experiment.

2. Air-to-ground links use 3GPP TR 36.777 UMi-AV, not TR 38.901 UMi. 38.901 UMi
   is specified for UE heights of 1.5-22.5 m and is not valid for aerial nodes.
   Air-to-air links above rooftop height use free-space loss plus a blockage
   penalty, since a ground street-canyon model does not describe them at all.
"""

from __future__ import annotations

import math

import torch

# Physical / system constants
THERMAL_NOISE_DBM_PER_HZ = -174.0
_TINY_MW = 1e-30  # floor for log of a linear power, avoids -inf

# Rate model. Shannon is an upper bound; real 5G NR is limited by its modulation
# and coding set. 0.75 is a conventional implementation-loss factor and 7.4
# b/s/Hz is roughly the 256QAM ceiling in NR.
DEFAULT_IMPL_LOSS = 0.75
DEFAULT_SE_CAP_BPS_HZ = 7.4


# --------------------------------------------------------------------------- #
# Unit conversion
# --------------------------------------------------------------------------- #


def dbm_to_mw(dbm: torch.Tensor) -> torch.Tensor:
    return torch.pow(10.0, dbm / 10.0)


def mw_to_dbm(mw: torch.Tensor) -> torch.Tensor:
    return 10.0 * torch.log10(mw.clamp_min(_TINY_MW))


def noise_floor_dbm(bandwidth_hz: float, noise_figure_db: float = 7.0) -> float:
    """Thermal noise floor over the given bandwidth, including receiver NF.

    Must track bandwidth -- the original spec hardcoded -100 dBm, which only
    coincidentally resembles kTB at 20 MHz and ignores the noise figure
    entirely. At B=10 MHz, NF=7 dB this returns -97.0 dBm.
    """
    return THERMAL_NOISE_DBM_PER_HZ + 10.0 * math.log10(bandwidth_hz) + noise_figure_db


# --------------------------------------------------------------------------- #
# Path loss
# --------------------------------------------------------------------------- #


def fspl_db(d_m: torch.Tensor, fc_ghz: float) -> torch.Tensor:
    """Free-space path loss. FSPL = 20log10(d_m) + 20log10(f_GHz) + 32.44."""
    d = d_m.clamp_min(1.0)  # below 1 m the far-field assumption fails anyway
    return 20.0 * torch.log10(d) + 20.0 * math.log10(fc_ghz) + 32.44


def pathloss_a2a_db(
    d_m: torch.Tensor,
    occluded: torch.Tensor,
    fc_ghz: float = 3.5,
    blockage_db: float = 20.0,
) -> torch.Tensor:
    """Air-to-air (drone <-> drone) path loss.

    Both endpoints sit above rooftop height, so the ray is close to free-space
    unless a tall building intersects it. Modelled as a two-state channel:
    FSPL, plus a structural attenuation penalty when the ray is occluded.

    `occluded` is a bool tensor broadcastable to `d_m`.
    """
    return fspl_db(d_m, fc_ghz) + blockage_db * occluded.to(d_m.dtype)


def pathloss_a2g_umi_av_db(
    d_3d_m: torch.Tensor,
    h_uav_m: torch.Tensor,
    los: torch.Tensor,
    fc_ghz: float = 3.5,
) -> torch.Tensor:
    """Air-to-ground path loss, 3GPP TR 36.777 UMi-AV.

    Valid for UAV heights of roughly 22.5-300 m -- the regime this thesis
    operates in, and precisely the regime TR 38.901 UMi excludes.

        LoS  : 30.9 + (22.25 - 0.5*log10(h)) * log10(d3d) + 20*log10(fc)
        NLoS : max(LoS, 32.4 + (43.2 - 7.6*log10(h)) * log10(d3d) + 20*log10(fc))

    TODO(verify): these coefficients must be checked against the actual 3GPP
    TR 36.777 document (UMi-AV table) before anything derived from them appears
    in the methodology chapter. They are self-consistent and give physically
    sensible values (LoS lands ~1 dB above FSPL at 100 m altitude, NLoS ~17 dB
    above LoS) but sensible is not the same as correct. Shadow fading is not
    modelled here; add it as a separate zero-mean term if the thesis needs it.
    """
    d = d_3d_m.clamp_min(1.0)
    h = h_uav_m.clamp_min(22.5)
    log_d = torch.log10(d)
    log_h = torch.log10(h)
    fc_term = 20.0 * math.log10(fc_ghz)

    pl_los = 30.9 + (22.25 - 0.5 * log_h) * log_d + fc_term
    pl_nlos = torch.maximum(pl_los, 32.4 + (43.2 - 7.6 * log_h) * log_d + fc_term)
    return torch.where(los.to(torch.bool), pl_los, pl_nlos)


# --------------------------------------------------------------------------- #
# SINR and rate
# --------------------------------------------------------------------------- #


def received_power_dbm(ptx_dbm: torch.Tensor, pathloss_db: torch.Tensor) -> torch.Tensor:
    """(B, M) transmit powers against a (B, M, M) loss matrix -> (B, M, M) Prx.

    Antenna gains are folded into `ptx_dbm` by the caller if used.
    """
    return ptx_dbm.unsqueeze(-1) - pathloss_db


def sinr_db(
    prx_dbm: torch.Tensor,
    jam_dbm: torch.Tensor,
    n0_dbm: float,
    tx_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-link SINR including intra-swarm interference.

    Parameters
    ----------
    prx_dbm : (B, M, M)   received power at j from i
    jam_dbm : (B, M)      jammer power received at each node
    n0_dbm  : float       thermal noise floor for the channel bandwidth
    tx_mask : (B, M)      bool, which nodes are actively transmitting

    Returns
    -------
    (B, M, M) SINR in dB for each candidate link i -> j.

    Interference model
    ------------------
    All active transmitters share one band (full spatial reuse, worst case). For
    a link i -> j, every other active transmitter k contributes interference at
    j. Node j's own transmission is excluded via the zeroed diagonal: a
    half-duplex node does not self-interfere in its own receive slot. The
    half-duplex cost is instead charged once, end-to-end, as the `/ n_hops`
    divisor in routing.py.

    This is a deliberately conservative assumption -- a real tactical MANET MAC
    would schedule to avoid some of this. State it as such in the methodology.
    """
    m = prx_dbm.shape[1]
    eye = torch.eye(m, device=prx_dbm.device, dtype=prx_dbm.dtype)

    prx_mw = dbm_to_mw(prx_dbm) * (1.0 - eye)  # (B,M,M), diagonal killed
    contrib = prx_mw * tx_mask.to(prx_mw.dtype).unsqueeze(-1)  # silence non-Tx rows

    total_at_rx = contrib.sum(dim=1, keepdim=True)  # (B,1,M) all energy landing on j
    interference_mw = total_at_rx - contrib  # (B,M,M) minus the wanted signal

    noise_mw = dbm_to_mw(torch.as_tensor(n0_dbm, device=prx_dbm.device, dtype=prx_dbm.dtype))
    jam_mw = dbm_to_mw(jam_dbm).unsqueeze(1)  # (B,1,M) broadcast over Tx

    sinr_lin = contrib / (interference_mw + jam_mw + noise_mw).clamp_min(_TINY_MW)
    return 10.0 * torch.log10(sinr_lin.clamp_min(_TINY_MW))


def capacity_mbps(
    sinr_db_: torch.Tensor,
    bandwidth_hz: float,
    impl_loss: float = DEFAULT_IMPL_LOSS,
    se_cap: float = DEFAULT_SE_CAP_BPS_HZ,
) -> torch.Tensor:
    """Achievable rate with an implementation-loss factor and a modulation cap.

    Unbounded Shannon reports throughput no real radio delivers; capping at the
    256QAM ceiling keeps high-SINR links honest.
    """
    sinr_lin = torch.pow(10.0, sinr_db_ / 10.0)
    se = (impl_loss * torch.log2(1.0 + sinr_lin)).clamp(max=se_cap)
    return bandwidth_hz * se / 1e6


def pairwise_distance_m(pos: torch.Tensor) -> torch.Tensor:
    """(B, M, 3) node positions -> (B, M, M) Euclidean 3D distances."""
    return torch.cdist(pos, pos)
