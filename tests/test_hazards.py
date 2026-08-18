"""People and vehicles: separation rules, soft cost, and the squeeze scenario."""

from __future__ import annotations

import pytest

from autonomy import (
    City,
    GridPlanner,
    HazardField,
    MissionRunner,
    RegulatoryProfile,
    SafetyMonitor,
    SafetyVerdict,
    Takeoff,
    Vec3,
    car_on_road,
    pedestrian_on_sidewalk,
)
from autonomy.backends import KinematicBackend
from autonomy.patrol import NavigateTo


class Permissive(SafetyMonitor):
    def evaluate(self, telemetry):
        return SafetyVerdict.ok()


def standing(north=60.0, east=0.0, name="person"):
    """A pedestrian who is effectively stationary, for geometric tests."""
    return pedestrian_on_sidewalk(
        north=north, east_start=east, east_end=east + 0.001,
        speed_mps=0.001, name=name,
    )


class TestRegulatoryProfiles:
    def test_basic_is_stricter_than_advanced(self):
        basic, advanced = RegulatoryProfile.basic(), RegulatoryProfile.advanced()
        assert basic.person_radius_m > advanced.person_radius_m
        assert not basic.overflight_allowed
        assert advanced.overflight_allowed


class TestSeparation:
    def test_inside_the_keep_out_is_blocked(self):
        f = HazardField(pedestrians=[standing()], rules=RegulatoryProfile.advanced())
        assert f.blocked_at(Vec3(60, 3, -20), 0.0) is not None
        assert f.blocked_at(Vec3(60, 40, -20), 0.0) is None

    def test_separation_is_horizontal_not_slant(self):
        """The rule is written horizontally, and the risk behaves that way:
        being high above someone does not make hitting them less likely."""
        f = HazardField(pedestrians=[standing()], rules=RegulatoryProfile.advanced())
        assert f.blocked_at(Vec3(60, 2, -80), 0.0) is not None

    def test_margin_applies_to_people(self):
        """Regression: the person keep-out originally ignored `margin`, making
        the most safety-critical constraint the only one with no safety factor.
        The margin carries the caller's tracking error."""
        f = HazardField(pedestrians=[standing()], rules=RegulatoryProfile.advanced())
        probe = Vec3(60, 8, -20)  # outside the bare 5 m rule
        assert f.blocked_at(probe, 0.0, margin=0.0) is None
        assert f.blocked_at(probe, 0.0, margin=6.0) is not None

    def test_overflight_blocked_under_basic_rules(self):
        f = HazardField(pedestrians=[standing()], rules=RegulatoryProfile.basic())
        reason = f.blocked_at(Vec3(60, 0.5, -40), 0.0)
        assert reason is not None

    def test_overflight_permitted_under_advanced_rules(self):
        f = HazardField(pedestrians=[standing()], rules=RegulatoryProfile.advanced())
        # Far enough for separation, directly overhead -- legal when advanced.
        assert f.blocked_at(Vec3(60, 0, -40), 0.0) is not None  # still inside 5 m
        f2 = HazardField(
            pedestrians=[standing()],
            rules=RegulatoryProfile("test", 1.0, True, 5.0),
        )
        assert f2.blocked_at(Vec3(60, 2.0, -40), 0.0) is None

    def test_empty_field_blocks_nothing(self):
        assert HazardField().blocked_at(Vec3(0, 0, -10), 0.0) is None
        assert HazardField().is_empty


class TestVehicles:
    def test_car_body_is_a_collision_hazard(self):
        car = car_on_road(north=0, east_start=-50, east_end=50, speed_mps=10.0)
        f = HazardField(vehicles=[car], rules=RegulatoryProfile.advanced())
        at_car = car.position_at(0.0)
        assert f.blocked_at(Vec3(at_car.north, at_car.east, -1.0), 0.0) is not None

    def test_above_the_roof_is_not_a_collision(self):
        car = car_on_road(north=0, east_start=-50, east_end=50, speed_mps=10.0)
        f = HazardField(vehicles=[car], rules=RegulatoryProfile.advanced())
        at_car = car.position_at(0.0)
        assert f.blocked_at(Vec3(at_car.north, at_car.east, -25.0), 0.0) is None

    def test_car_moves_so_the_hazard_moves(self):
        car = car_on_road(north=0, east_start=-50, east_end=50, speed_mps=10.0)
        f = HazardField(vehicles=[car], rules=RegulatoryProfile.advanced())
        start = car.position_at(0.0)
        probe = Vec3(start.north, start.east, -1.0)
        assert f.blocked_at(probe, 0.0) is not None
        assert f.blocked_at(probe, 4.0) is None, "car should have driven on"

    def test_overflight_of_occupants_blocked_under_basic(self):
        car = car_on_road(north=0, east_start=-50, east_end=50, speed_mps=10.0)
        f = HazardField(vehicles=[car], rules=RegulatoryProfile.basic())
        at_car = car.position_at(0.0)
        assert f.blocked_at(Vec3(at_car.north, at_car.east, -30.0), 0.0) is not None


class TestSoftCost:
    def test_cost_rises_as_you_approach(self):
        f = HazardField(pedestrians=[standing()], rules=RegulatoryProfile.advanced())
        near = f.risk_cost(Vec3(60, 8, -20), 0.0)
        far = f.risk_cost(Vec3(60, 22, -20), 0.0)
        assert near > far >= 0.0

    def test_cost_is_zero_beyond_the_comfort_radius(self):
        f = HazardField(pedestrians=[standing()], rules=RegulatoryProfile.advanced())
        assert f.risk_cost(Vec3(60, 200, -20), 0.0) == 0.0

    def test_low_flight_near_a_road_is_penalised(self):
        """The squeeze under a conductor: low and close to traffic."""
        car = car_on_road(north=0, east_start=-50, east_end=50, speed_mps=10.0)
        f = HazardField(vehicles=[car], rules=RegulatoryProfile.advanced())
        at_car = car.position_at(0.0)
        low = f.risk_cost(Vec3(at_car.north, at_car.east + 6, -5.0), 0.0)
        high = f.risk_cost(Vec3(at_car.north, at_car.east + 6, -30.0), 0.0)
        assert low > high


async def _fly(rules, *, hazard_weight=3.0, tracking_error=3.5):
    hz = HazardField(
        vehicles=[car_on_road(north=60, east_start=-90, east_end=90, speed_mps=11.0)],
        pedestrians=[pedestrian_on_sidewalk(north=68, east_start=-30, east_end=30)],
        rules=rules,
    )
    planner = GridPlanner(
        City(extent_m=250), cell_m=5, margin_m=2.5,
        min_altitude_m=4, max_altitude_m=60,
        hazard_weight=hazard_weight, tracking_error_m=tracking_error,
    )
    leg = NavigateTo(Vec3(130, 0, -14), planner, speed_mps=5.0, hazards=hz)
    runner = MissionRunner(
        KinematicBackend(max_speed_mps=7.0, battery_endurance_s=3000),
        [Takeoff(14.0), leg],
        safety=Permissive(), verbose=False, timeout_s=600, hazards=hz,
    )
    await runner.run()

    closest = min(
        hz.pedestrians[0].separation(Vec3(s.north, s.east, s.down), s.t)
        for s in runner.track
    )
    overflights = sum(
        1 for s in runner.track
        if hz.pedestrians[0].is_overhead(Vec3(s.north, s.east, s.down), s.t)
    )
    reached = leg.index >= len(leg.route) and not leg.failed
    return reached, closest, overflights


@pytest.mark.asyncio
async def test_basic_rules_are_respected_while_still_reaching_the_target():
    reached, closest, overflights = await _fly(RegulatoryProfile.basic())
    assert reached, "should still find a route, just a wider one"
    assert closest >= 30.0, f"basic ops require 30 m, flew {closest:.1f} m"
    assert overflights == 0


@pytest.mark.asyncio
async def test_advanced_rules_permit_a_tighter_route():
    reached, closest, _ = await _fly(RegulatoryProfile.advanced())
    assert reached
    assert closest >= 5.0, f"advanced ops require 5 m, flew {closest:.1f} m"


@pytest.mark.asyncio
async def test_advanced_flies_closer_than_basic():
    """The profiles must actually change behaviour, not just the numbers."""
    _, basic_closest, _ = await _fly(RegulatoryProfile.basic())
    _, adv_closest, _ = await _fly(RegulatoryProfile.advanced())
    assert adv_closest < basic_closest


@pytest.mark.asyncio
async def test_tracking_error_keeps_the_flown_path_legal():
    """A plan legal by exactly the minimum is flown in violation of it, because
    the vehicle cuts corners the plan did not."""
    _, closest, _ = await _fly(
        RegulatoryProfile.advanced(), hazard_weight=0.0, tracking_error=3.5
    )
    assert closest >= 5.0, f"flown path violated the minimum at {closest:.1f} m"
