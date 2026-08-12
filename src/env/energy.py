"""
Rotary-wing flight power, radio DC draw, and battery depletion.

Batched, pure-torch, device-agnostic; no `.cpu()`, `.numpy()` or `.item()`, so it
runs inside a vectorized env step on GPU.

Why this model
--------------
Propulsion power for a rotorcraft is **U-shaped in airspeed**: it falls from
hover to a minimum near 10-20 m/s, then rises again. Hovering is *expensive*.

The project's original spec used `P_hover + alpha*||v||^2`, which asserts the
opposite -- that hovering is the cheapest state. That is false for rotary-wing
aircraft, and energy is load-bearing here (it sets the cost of the observer role
and drives the load-balancing term in the reward), so the model cannot rest on a
form that rewards hovering when reality does not.

The closed form is standard rotorcraft aerodynamics, three additive terms:

    P(V) = P_0 * (1 + 3V^2/U_tip^2)                        # blade profile
         + P_i * (sqrt(1 + V^4/(4 v_0^4)) - V^2/(2 v_0^2))^(1/2)   # induced
         + 0.5 * d_0 * rho * s * A * V^3                   # parasite

Profile and parasite rise with speed; induced *falls*. Their sum is the U.
Presented in Zeng, Xu & Zhang (2019), "Energy Minimization for Wireless
Communication with Rotary-Wing UAV", IEEE TWC -- the citation the UAV-comms
literature converged on, though the aerodynamics predates it.

Parameters: derived, not quoted
-------------------------------
The constants are *derived from mass and rotor geometry via momentum theory*
rather than copied from a paper's example aircraft. Two reasons:

1. The dominant hover term follows exactly from published numbers --
   `v_0 = sqrt(W / 2*rho*A)` needs only mass and rotor size. A derivation is
   checkable in a way a quoted table is not.
2. It gives a validation a paper's constants cannot: predicted hover power
   against *published endurance*. For the ~5.9 kg / 21-inch airframe below the
   model predicts ~57 min of hover on 548 Wh against ~55 min published -- see
   `test_endurance_matches_published_flight_time`.

TODO(verify): `tip_speed_ms`, `solidity`, `profile_drag_coeff` and
`fuselage_drag_ratio` are not usually published per airframe and use documented
typical ranges. Check them (and the momentum-theory chain) against a rotorcraft
reference before the methodology chapter. Same standing as the TR 36.777
coefficients in channel.py -- do not cite numbers an AI produced.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

GRAVITY = 9.81
AIR_DENSITY_SEA_LEVEL = 1.225

# Radio. Ptx is fixed at 30 dBm, so P_tx_DC is a constant that shifts the budget
# by ~1.6% and does not vary with the action -- kept for completeness only.
PA_EFFICIENCY = 0.25
RADIO_CIRCUIT_W = 3.0


@dataclass(frozen=True)
class Rotorcraft:
    """Airframe parameters. Derived quantities come from momentum theory."""

    mass_kg: float
    n_rotors: int
    rotor_radius_m: float
    tip_speed_ms: float = 112.0  # TODO(verify): typical multirotor 100-150 m/s
    solidity: float = 0.05  # TODO(verify): blade area / disc area, 0.05-0.1
    profile_drag_coeff: float = 0.012  # TODO(verify): delta, ~0.012
    fuselage_drag_ratio: float = 0.6  # TODO(verify): d_0
    air_density: float = AIR_DENSITY_SEA_LEVEL
    induced_power_factor: float = 1.15  # non-ideal induced losses
    drivetrain_efficiency: float = 0.80  # motor + ESC; shaft W -> electrical W

    @property
    def weight_n(self) -> float:
        return self.mass_kg * GRAVITY

    @property
    def disc_area_m2(self) -> float:
        return self.n_rotors * math.pi * self.rotor_radius_m**2

    @property
    def induced_velocity_hover_ms(self) -> float:
        """v_0 = sqrt(W / 2*rho*A). Momentum theory, exact."""
        return math.sqrt(self.weight_n / (2.0 * self.air_density * self.disc_area_m2))

    @property
    def induced_power_hover_w(self) -> float:
        """P_i = k * W * v_0. The dominant hover term."""
        return self.induced_power_factor * self.weight_n * self.induced_velocity_hover_ms

    @property
    def blade_profile_power_w(self) -> float:
        """P_0 = (delta/8) * rho * s * A * U_tip^3."""
        return (
            (self.profile_drag_coeff / 8.0)
            * self.air_density
            * self.solidity
            * self.disc_area_m2
            * self.tip_speed_ms**3
        )


# A ~6 kg quadrotor with 21-inch rotors: the ISR class this thesis models.
DEFAULT_AIRFRAME = Rotorcraft(mass_kg=5.9, n_rotors=4, rotor_radius_m=0.2667)


def propulsion_power_w(speed_ms: torch.Tensor, craft: Rotorcraft) -> torch.Tensor:
    """Shaft power at forward airspeed. Batched; any shape in, same shape out."""
    v = speed_ms.clamp_min(0.0)
    v0 = craft.induced_velocity_hover_ms

    profile = craft.blade_profile_power_w * (1.0 + 3.0 * v**2 / craft.tip_speed_ms**2)

    # Induced term. The literal form sqrt(1+x^2) - x loses all precision at large
    # x through catastrophic cancellation; the algebraically identical
    # 1/(sqrt(1+x^2) + x) is stable everywhere.
    x = v**2 / (2.0 * v0**2)
    induced = craft.induced_power_hover_w * torch.sqrt(1.0 / (torch.sqrt(1.0 + x**2) + x))

    parasite = (
        0.5
        * craft.fuselage_drag_ratio
        * craft.air_density
        * craft.solidity
        * craft.disc_area_m2
        * v**3
    )
    return profile + induced + parasite


def hover_power_w(craft: Rotorcraft) -> float:
    """P(0) = P_0 + P_i exactly: profile and induced terms, parasite vanishes."""
    return craft.blade_profile_power_w + craft.induced_power_hover_w


def electrical_power_w(shaft_power_w: torch.Tensor, craft: Rotorcraft) -> torch.Tensor:
    """Battery drain is electrical, not shaft. Motor and ESC losses matter --
    omitting them overestimates endurance by ~25%."""
    return shaft_power_w / craft.drivetrain_efficiency


def climb_power_w(
    vertical_speed_ms: torch.Tensor, craft: Rotorcraft = DEFAULT_AIRFRAME
) -> torch.Tensor:
    """Extra electrical draw while climbing: `W * v_z / eta`.

    Rate of change of potential energy, divided by drivetrain efficiency. Only
    the ascending half is charged -- a descent recovers nothing on a multirotor,
    which windmills rather than regenerating.

    Added because nothing else in the model charges for altitude
    (`propulsion_power_w` is a function of horizontal speed only), which made
    altitude a free good. It is physically right and traceable, but it does
    **not** bind: a full 40 -> 120 m climb at 5 m/s costs ~0.55 % of a 548 Wh
    pack, so the altitude band -- not this term -- is what governs how high the
    swarm flies. See docs/BLOCK_D.md.
    """
    return craft.weight_n * vertical_speed_ms.clamp_min(0.0) / craft.drivetrain_efficiency


def radio_dc_power_w(ptx_dbm: float = 30.0) -> float:
    """Constant: Ptx is not an action. ~4 W of PA plus the always-on front end."""
    return (10.0 ** (ptx_dbm / 10.0) / 1000.0) / PA_EFFICIENCY + RADIO_CIRCUIT_W


def total_power_w(
    speed_ms: torch.Tensor,
    accel_ms2: torch.Tensor,
    craft: Rotorcraft = DEFAULT_AIRFRAME,
    control_effort_coeff: float = 0.0,
    ptx_dbm: float = 30.0,
) -> torch.Tensor:
    """Electrical draw per drone: flight + control effort + radio.

    `control_effort_coeff * ||a||^2` is an explicit **heuristic**, not physics --
    present it as such in the methodology. Default 0 so it must be opted into.
    """
    flight = electrical_power_w(propulsion_power_w(speed_ms, craft), craft)
    return flight + control_effort_coeff * accel_ms2**2 + radio_dc_power_w(ptx_dbm)


def min_power_speed_ms(craft: Rotorcraft, v_max: float = 40.0, n: int = 4001) -> float:
    """Airspeed minimising propulsion power -- the bottom of the U."""
    grid = torch.linspace(0.0, v_max, n)
    return grid[torch.argmin(propulsion_power_w(grid, craft))].item()


def endurance_s(
    battery_wh: float, craft: Rotorcraft = DEFAULT_AIRFRAME, speed_ms: float = 0.0
) -> float:
    """Flight time at constant airspeed, including the radio. Diagnostic only --
    never call this inside `step()`."""
    p = total_power_w(torch.tensor(speed_ms), torch.tensor(0.0), craft).item()
    return battery_wh * 3600.0 / p


def battery_fraction_used(
    battery_wh: float, duration_s: float, craft: Rotorcraft = DEFAULT_AIRFRAME
) -> float:
    """Share of capacity a hovering drone burns over `duration_s`.

    Sizing check for RQ3: a realistic airframe uses only ~8-17% over a 240 s
    episode, so batteries never bind and `Var(B)` has nothing to act on. This is
    why initial charge is randomised in [0.3, 1.0] rather than starting full.
    """
    return duration_s / endurance_s(battery_wh, craft)
