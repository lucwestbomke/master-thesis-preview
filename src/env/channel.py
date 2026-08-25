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

    ## Why this is NOT the TR 36.777 model, and that is correct

    TR 36.777 covers **air-to-ground only** -- an aerial UE against a
    ground-mounted eNodeB. It says nothing about drone<->drone links, where both
    endpoints are above rooftop, and applying a street-canyon model to a ray that
    never enters the canyon would be a category error. So A2A and A2G genuinely
    use different models here, deliberately.

    ✅ **The free-space choice has direct empirical support.** Measurement-based
    A2A modelling in built-up areas finds that "when both UAVs are higher than
    50 m, the path loss of A2A channels can be described with a free-space
    propagation model" (Path Loss Analysis for Low-Altitude Air-to-Air
    Millimeter-Wave, arXiv:2301.12229). ⚠️ This project's altitude band is
    **40-80 m**, so the lower end sits just under that 50 m finding -- worth a
    sentence in the methodology rather than silence.

    ## 📏 `blockage_db = 20.0` -- assumed, physically low, and measurably harmless

    Verified 2026-08-26 by `scripts/verify_blockage.py`, which regenerates all of
    the following. It is an **assumed** constant with no citation, so it was
    defended the other way: by showing the result does not depend on it.

    **The physics says 20 dB is too low.** Occluded A2A rays in the real
    Frankfurt geometry do not graze -- the median ray passes **60.5 m inside**
    the obstruction (p25 10.9 m, p90 111.9 m), because at 40-80 m altitude the
    only blockers tall enough to matter are the towers. The first Fresnel radius
    at the median 235 m link is **2.2 m**, so a 60 m depth is ~27 Fresnel radii:
    deep shadow, not diffraction fringe. Single knife-edge (ITU-R P.526) over
    those depths gives a **median 43.3 dB**, and **90.4 %** of occluded A2A links
    exceed the modelled 20 dB.

    **But it governs almost nothing.** Of the occluded edges on B0's *chosen*
    relay chain, only **16.8 % are A2A**; the other **83.2 % are drone<->MCV**,
    which runs on the TR 36.777 NLoS branch and never touches this constant.
    Sweeping it through B0 at stage 4 / F4 moves the headline metric by less than
    the seed IQR:

        blockage_db      20      30      40
        mission-capable  59.7 %  60.8 %  59.5 %

    ⚠️ So the honest statement for the methodology is **not** "20 dB is correct" --
    it is "the A2A blockage penalty is an assumed 20 dB; the physically-motivated
    value is nearer 40 dB; the reported metric is insensitive to it across that
    range (±0.7 pp), because 83 % of occluded chain edges are air-to-ground."
    That is a stronger position than a citation would have given.

    ⛔ Do **not** change the constant on the strength of the physics alone. It
    would re-derive every number in Blocks D-F for a sub-IQR effect, and the env
    is frozen. Re-open only if a rung-by-rung sweep (F2/F3 are untested; F0/F1
    never reach this code, since `binary_capacity` skips path loss entirely)
    shows a rung where it *does* bind.

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

    Valid for UAV heights of 22.5-300 m -- the regime this thesis operates in,
    and precisely the regime TR 38.901 UMi excludes (it stops at 22.5 m).

        LoS  : max(FSPL, 30.9 + (22.25 - 0.5*log10(h)) * log10(d3d) + 20*log10(fc))
        NLoS : max(LoS,  32.4 + (43.2  - 7.6*log10(h)) * log10(d3d) + 20*log10(fc))

    `h` is the AERIAL UE height in metres, `d3d` the 3-D separation in metres,
    `fc` in **GHz**. The caller passes `h = max(z_i, z_j)`, which for a
    drone<->MCV link is the drone's altitude -- the aerial UE, as the model
    intends.

    ## Verification status (2026-08-26) -- ⚠️ PARTIAL, do not treat as closed

    Checked against secondary sources rather than against the 3GPP document
    itself, because the document is paywalled/unreachable from here. What that
    established:

    * ✅ **LoS, all constants.** The formula
      `max(FSPL, 30.9 + (22.25 - 0.5 log10(h)) log10(d3D) + 20 log10(fc))` is
      quoted verbatim in the literature, matching intercept, slope, height
      correction and frequency term.
    * ✅ **NLoS slope and height correction.** Two independent sources give the
      NLoS path-loss exponent as `4.32 - 0.76 log10(h)`, i.e. `43.2 - 7.6
      log10(h)` in dB-per-decade form. Matches.
    * ⚠️ **NLoS intercept `32.4` is NOT independently confirmed**, and there is a
      specific reason for suspicion: `32.4` is also the intercept of TR 38.901's
      *terrestrial* UMi Street-Canyon **LoS** model
      (`32.4 + 21 log10(d3D) + 20 log10(fc)`), which is exactly the sort of
      neighbouring constant a transcription slip lands on. **This is the one
      number a human still has to read off TR 36.777's UMi-AV table.**

    Internal consistency checks that do pass: NLoS >= LoS everywhere (min margin
    +1.5 dB over d in [1, 1585] m); LoS sits 1.1-2.7 dB above FSPL across the
    operating band, rising with distance as a canyon model should; NLoS sits
    16-30 dB above LoS.

    ⚠️ **Frequency extrapolation.** TR 36.777 is an LTE study item; using it at
    3.5 GHz relies on the `20 log10(fc)` term scaling correctly outside the bands
    it was fitted in. Defensible and standard, but state it in the methodology
    rather than leaving it implicit.

    Shadow fading is **not** modelled here (TR 36.777 specifies it per scenario);
    add it as a separate zero-mean term if the thesis needs it. Neither is the
    TR 36.777 LOS *probability* model -- this project ray-traces LoS against real
    LoD2 geometry instead, which is deliberate: RQ1 cannot ablate occlusion if
    occlusion is a random variable.

    ⚠️ The `max(FSPL, .)` on the LoS branch is **provably vacuous in this
    scenario** and is present for exactness only. FSPL exceeds the UMi-AV LoS
    expression only below 9.5-18.7 m (depending on `h`), while the 40 m altitude
    floor puts every drone<->MCV separation above 40 m by construction. Measured
    minimum over a rollout: 38 m at `h_min`, still clear of the crossover.
    """
    d = d_3d_m.clamp_min(1.0)
    h = h_uav_m.clamp_min(22.5)
    log_d = torch.log10(d)
    log_h = torch.log10(h)
    fc_term = 20.0 * math.log10(fc_ghz)

    pl_los = torch.maximum(fspl_db(d, fc_ghz), 30.9 + (22.25 - 0.5 * log_h) * log_d + fc_term)
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

    Interference model -- `tx_mask` carries the MAC assumption
    ---------------------------------------------------------
    Every node flagged in `tx_mask` interferes with every link it is not the
    transmitter of. Node j's own emission is excluded via the zeroed diagonal:
    a half-duplex node does not self-interfere in its own receive slot.

    Which nodes belong in `tx_mask` is a medium-access decision, and the caller
    must make it deliberately -- getting it wrong silently changes the physics:

    - **Scheduled MAC (default for this project).** routing.py divides
      end-to-end rate by `min(n_hops, reuse_limit)`, which presumes a
      spatial-reuse TDMA schedule. Under a reuse-3 schedule a chain of <=3 hops
      never has two hops active at once, so when evaluating a link the mask
      should contain *only the transmitters active in that slot*. For short
      chains that is one node, and SINR reduces to signal over jammer-plus-noise.
    - **Uncoordinated access.** All active nodes concurrent, for worst-case
      analysis. Then the routing divisor must not also be applied, or the
      half-duplex cost is charged twice.

    Passing every node while also dividing by `min(n, 3)` double-counts. That
    combination made a feasible 3-hop chain look infeasible during scenario
    design; see docs/NEGATIVE_RESULTS.md.
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
