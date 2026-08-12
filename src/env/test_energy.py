"""
Rotary-wing energy tests. Expected values are hand-computed from momentum theory
and the closed form, not captured from a run -- these numbers set the cost of the
observer role and therefore shape the reward, so a regression must fail loudly.
"""

import math

import pytest
import torch

from .energy import (
    DEFAULT_AIRFRAME,
    GRAVITY,
    PA_EFFICIENCY,
    RADIO_CIRCUIT_W,
    Rotorcraft,
    battery_fraction_used,
    climb_power_w,
    electrical_power_w,
    endurance_s,
    hover_power_w,
    min_power_speed_ms,
    propulsion_power_w,
    radio_dc_power_w,
    total_power_w,
)

C = DEFAULT_AIRFRAME


# --------------------------------------------------------------------------- #
# Momentum theory -- the derived constants
# --------------------------------------------------------------------------- #


def test_disc_area_hand_computed():
    # A = n * pi * r^2 = 4 * pi * 0.2667^2 = 0.8938320 m^2
    assert C.disc_area_m2 == pytest.approx(0.8938320, abs=1e-6)


def test_weight_hand_computed():
    assert C.weight_n == pytest.approx(5.9 * 9.81, abs=1e-6)


def test_induced_velocity_is_momentum_theory():
    # v0 = sqrt(W / (2*rho*A)) = sqrt(57.879 / (2*1.225*0.8938320))
    #    = sqrt(57.879 / 2.189889) = sqrt(26.42958) = 5.141023 m/s
    assert C.induced_velocity_hover_ms == pytest.approx(5.141023, abs=1e-6)


def test_induced_power_hand_computed():
    # P_i = k * W * v0 = 1.15 * 57.879 * 5.141023 = 342.1908 W
    assert C.induced_power_hover_w == pytest.approx(342.1908, abs=1e-3)


def test_blade_profile_power_hand_computed():
    # P_0 = (delta/8) * rho * s * A * U_tip^3
    #     = 0.0015 * 1.225 * 0.05 * 0.8938320 * 112^3 = 115.3738 W
    assert C.blade_profile_power_w == pytest.approx(115.3738, abs=1e-3)


def test_induced_power_dominates_in_hover():
    """The reason mass and rotor size are enough to size the model."""
    assert C.induced_power_hover_w > 2.5 * C.blade_profile_power_w


# --------------------------------------------------------------------------- #
# The U-shape -- the entire reason this model replaced alpha*v^2
# --------------------------------------------------------------------------- #


def test_hover_power_is_profile_plus_induced_exactly():
    # At V=0 the parasite term vanishes and both brackets evaluate to 1.
    # P_0 + P_i = 115.3738 + 342.1908 = 457.5647 W
    got = propulsion_power_w(torch.tensor(0.0), C).item()
    assert got == pytest.approx(457.5647, rel=1e-6)
    assert got == pytest.approx(C.blade_profile_power_w + C.induced_power_hover_w, rel=1e-6)
    assert got == pytest.approx(hover_power_w(C), rel=1e-6)


def test_regression_hovering_is_not_the_cheapest_state():
    """Pins the error the original spec contained.

    `P_hover + alpha*||v||^2` asserts hovering is cheapest. For a rotorcraft it
    is not: induced power falls with forward speed faster than profile power
    rises, so cruising costs materially less than hovering. Energy sets the cost
    of the observer role, so getting this backwards would invert the behaviour
    the reward is meant to produce.
    """
    p_hover = propulsion_power_w(torch.tensor(0.0), C).item()
    p_cruise = propulsion_power_w(torch.tensor(min_power_speed_ms(C)), C).item()
    assert p_cruise < p_hover
    assert p_hover / p_cruise > 1.4  # hovering costs ~58% more here


def test_minimum_power_speed_is_in_the_expected_band():
    """Rotorcraft minimum-power speed lands around 10-20 m/s."""
    assert 8.0 < min_power_speed_ms(C) < 20.0


def test_power_curve_falls_then_rises():
    v = torch.linspace(0.0, 40.0, 401)
    p = propulsion_power_w(v, C)
    v_min = min_power_speed_ms(C)
    before, after = v < v_min - 1.0, v > v_min + 1.0
    assert torch.all(torch.diff(p[before]) < 0), "must fall before the minimum"
    assert torch.all(torch.diff(p[after]) > 0), "must rise after the minimum"


def test_parasite_term_dominates_at_high_speed():
    # Parasite scales V^3, so doubling speed well past the minimum should push
    # power up by clearly more than 4x (which V^2 would give).
    p20 = propulsion_power_w(torch.tensor(20.0), C).item()
    p40 = propulsion_power_w(torch.tensor(40.0), C).item()
    assert p40 / p20 > 3.0


def test_induced_term_uses_the_numerically_stable_identity():
    """sqrt(1+x^2) - x  ==  1 / (sqrt(1+x^2) + x), the latter without cancellation."""
    # float64 cancellation already costs ~8 digits by x=1e4, which is itself
    # the point -- the identity is exact, the naive evaluation is not.
    x = torch.logspace(-3, 4, 200, dtype=torch.float64)
    naive = torch.sqrt(1.0 + x**2) - x
    stable = 1.0 / (torch.sqrt(1.0 + x**2) + x)
    assert torch.allclose(naive, stable, rtol=1e-6, atol=0.0)
    # And the naive form does collapse in float32 where the stable one does not.
    xf = torch.tensor([1e5], dtype=torch.float32)
    assert (torch.sqrt(1.0 + xf**2) - xf).item() == 0.0
    assert (1.0 / (torch.sqrt(1.0 + xf**2) + xf)).item() > 0.0


# --------------------------------------------------------------------------- #
# Radio and totals
# --------------------------------------------------------------------------- #


def test_radio_dc_power_hand_computed():
    # 30 dBm = 1 W RF; / 0.25 PA efficiency = 4 W; + 3 W front end = 7 W
    assert radio_dc_power_w(30.0) == pytest.approx(7.0, abs=1e-9)
    assert radio_dc_power_w(30.0) == pytest.approx(1.0 / PA_EFFICIENCY + RADIO_CIRCUIT_W, abs=1e-9)


def test_radio_is_a_small_constant_share_of_the_budget():
    """~1.6% of draw -- which is why adaptive Ptx was a null. See NEGATIVE_RESULTS."""
    flight = electrical_power_w(propulsion_power_w(torch.tensor(0.0), C), C).item()
    assert radio_dc_power_w(30.0) / (flight + radio_dc_power_w(30.0)) < 0.03


def test_electrical_exceeds_shaft_power():
    shaft = propulsion_power_w(torch.tensor(15.0), C)
    assert electrical_power_w(shaft, C).item() > shaft.item()
    assert electrical_power_w(shaft, C).item() == pytest.approx(
        shaft.item() / C.drivetrain_efficiency, rel=1e-6
    )


def test_control_effort_is_opt_in():
    v, a = torch.tensor(10.0), torch.tensor(5.0)
    assert total_power_w(v, a, C).item() == pytest.approx(
        total_power_w(v, a, C, control_effort_coeff=0.0).item(), abs=1e-9
    )
    assert total_power_w(v, a, C, control_effort_coeff=1.0).item() > total_power_w(v, a, C).item()


# --------------------------------------------------------------------------- #
# Validation against something external
# --------------------------------------------------------------------------- #


def test_endurance_matches_published_flight_time():
    """The check a paper's example constants cannot give you.

    A ~5.9 kg quadrotor on 21-inch rotors with ~548 Wh publishes roughly 55 min
    of hover. The derived model predicts ~57 min. Agreement to a few percent is
    what justifies deriving the constants from mass and rotor geometry rather
    than copying a table.
    """
    minutes = endurance_s(548.0, C) / 60.0
    assert 48.0 < minutes < 65.0


def test_cruising_extends_endurance_over_hovering():
    assert endurance_s(548.0, C, speed_ms=min_power_speed_ms(C)) > endurance_s(548.0, C, 0.0)


def test_battery_does_not_bind_in_one_episode():
    """Pins the measurement that reframed RQ3.

    240 s of hovering burns well under a quarter of a realistic battery, so
    `Var(B)` has nothing to act on and energy-driven role rotation has no
    mechanism. Hence geometric handoff, and randomised initial charge.
    """
    used = battery_fraction_used(548.0, 240.0, C)
    assert used < 0.25, "if this ever exceeds 25%, revisit RQ3's framing"
    assert used == pytest.approx(0.070, abs=0.01)


def test_heavier_airframe_costs_more_to_hover():
    heavy = Rotorcraft(mass_kg=10.0, n_rotors=4, rotor_radius_m=0.2667)
    assert hover_power_w(heavy) > hover_power_w(C)


def test_larger_rotors_cost_less_to_hover():
    """Disc loading: more area, lower induced velocity, cheaper hover."""
    small = Rotorcraft(mass_kg=5.9, n_rotors=4, rotor_radius_m=0.15)
    assert C.induced_power_hover_w < small.induced_power_hover_w


# --------------------------------------------------------------------------- #
# Shapes / device hygiene
# --------------------------------------------------------------------------- #


def test_batched_shapes_and_finiteness():
    b, n = 16, 5
    speed = torch.rand(b, n) * 30.0
    accel = torch.rand(b, n) * 5.0
    p = total_power_w(speed, accel, C, control_effort_coeff=0.1)
    assert p.shape == (b, n)
    assert torch.isfinite(p).all() and torch.all(p > 0)


def test_negative_speed_is_clamped_not_nan():
    p = propulsion_power_w(torch.tensor([-5.0, 0.0]), C)
    assert torch.isfinite(p).all()
    assert p[0].item() == pytest.approx(p[1].item(), abs=1e-6)


def test_matches_the_closed_form_termwise():
    """Independent re-implementation of the published expression."""
    v = 17.0
    v0, u = C.induced_velocity_hover_ms, C.tip_speed_ms
    profile = C.blade_profile_power_w * (1 + 3 * v**2 / u**2)
    induced = C.induced_power_hover_w * math.sqrt(
        math.sqrt(1 + v**4 / (4 * v0**4)) - v**2 / (2 * v0**2)
    )
    parasite = 0.5 * C.fuselage_drag_ratio * C.air_density * C.solidity * C.disc_area_m2 * v**3
    expected = profile + induced + parasite
    assert propulsion_power_w(torch.tensor(v), C).item() == pytest.approx(expected, rel=1e-5)


def test_climb_power_is_potential_energy_over_efficiency():
    craft = DEFAULT_AIRFRAME
    v_z = torch.tensor([0.0, 1.0, 5.0])
    got = climb_power_w(v_z, craft)
    expect = craft.mass_kg * GRAVITY * v_z / craft.drivetrain_efficiency
    assert torch.allclose(got, expect, rtol=1e-6)


def test_descending_is_not_charged_and_is_not_regenerative():
    """A multirotor windmills on descent; it neither pays nor recovers here."""
    assert torch.all(climb_power_w(torch.tensor([-1.0, -8.0])) == 0.0)


def test_a_full_climb_does_not_meaningfully_dent_the_battery():
    """The measurement behind BLOCK_D.md's altitude decision.

    If this ever exceeds a few percent, the altitude band stops being the only
    thing governing how high the swarm flies and the reasoning there must be
    revisited.
    """
    v_z, band_m = 5.0, 120.0 - 40.0
    seconds = band_m / v_z
    joules = climb_power_w(torch.tensor(v_z)).item() * seconds
    fraction = joules / (548.0 * 3600.0)
    assert fraction < 0.01, f"climb costs {fraction:.1%} of the pack, not negligible"
