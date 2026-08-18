"""End-to-end: infer conductors from poles, then actually avoid them.

The whole chain under one assertion -- camera detections, triangulation, span
inference, obstacle export, replanning, and a return leg that respects the map
it just built.
"""

from __future__ import annotations

from collections import Counter

import pytest

from autonomy import (
    Cable,
    City,
    GridPlanner,
    MissionRunner,
    ReturnToLaunch,
    SafetyMonitor,
    SafetyVerdict,
    Takeoff,
    ThinObstacleField,
    Vec3,
)
from autonomy.backends import KinematicBackend
from autonomy.mapping import PerceptionPipeline
from autonomy.patrol import NavigateTo, PlannedReturn
from autonomy.perception import DetectorModel, ObjectClass

# A drone must clear a conductor by more than its own position error. Grazing
# it by centimetres is not "clear", it is luck.
CLEARANCE_M = 2.5


class Permissive(SafetyMonitor):
    def evaluate(self, telemetry):
        return SafetyVerdict.ok()


def scenario():
    """A transmission crossing at flight altitude, on the direct route north."""
    poles = [Vec3(60.0, -60.0 + i * 45.0, -26.0) for i in range(4)]
    truth = ThinObstacleField(cables=tuple(
        Cable(
            start=Vec3(60, -60 + i * 45, -25.0),
            end=Vec3(60, -60 + (i + 1) * 45, -25.0),
            sag_m=1.4, name=f"tx-{i}",
        )
        for i in range(3)
    ))
    targets = [(ObjectClass.POLE, p, f"P{i}") for i, p in enumerate(poles)]
    return poles, truth, targets


# Each run is a full simulated flight, so identical configurations are cached
# rather than reflown -- several tests interrogate different aspects of the
# same flight and reflying it tripled the suite runtime for no extra coverage.
_FLIGHT_CACHE: dict[tuple[bool, bool], tuple] = {}


async def fly(*, perceive: bool, planned_rtl: bool):
    key = (perceive, planned_rtl)
    if key not in _FLIGHT_CACHE:
        _FLIGHT_CACHE[key] = await _fly_uncached(perceive, planned_rtl)
    return _FLIGHT_CACHE[key]


async def _fly_uncached(perceive: bool, planned_rtl: bool):
    _, truth, targets = scenario()
    planner = GridPlanner(
        City(extent_m=300), cell_m=6, margin_m=3,
        min_altitude_m=8, max_altitude_m=70, thin_margin_m=9.0,
    )
    pipe = (
        PerceptionPipeline(
            detector=DetectorModel(seed=17, mounts_deg=(0.0, 90.0, 180.0, 270.0)),
            truth=targets, rebuild_interval_s=1.5,
        )
        if perceive else None
    )
    leg = NavigateTo(Vec3(140, 0, -24), planner, speed_mps=6.0, perception=pipe)
    rtl = (
        PlannedReturn(planner, speed_mps=6.0, perception=pipe)
        if planned_rtl else ReturnToLaunch()
    )
    runner = MissionRunner(
        KinematicBackend(max_speed_mps=8.0, battery_endurance_s=3000),
        [Takeoff(24.0), leg, rtl],
        safety=Permissive(), perception=pipe, verbose=False, timeout_s=500,
    )
    await runner.run()

    breaches = Counter()
    closest = float("inf")
    for a, b in zip(runner.track, runner.track[1:]):
        pa = Vec3(a.north, a.east, a.down)
        pb = Vec3(b.north, b.east, b.down)
        for cable in truth.cables:
            for i in range(6):
                closest = min(closest, cable.distance_to(pa + (pb - pa) * (i / 5)))
        if not truth.segment_clear(pa, pb, margin=CLEARANCE_M):
            breaches[a.phase] += 1
    return pipe, leg, breaches, closest


@pytest.mark.asyncio
async def test_without_perception_the_route_grazes_the_conductor():
    """The baseline this whole feature exists to fix."""
    _, _, breaches, closest = await fly(perceive=False, planned_rtl=False)
    assert sum(breaches.values()) > 0
    assert closest < 1.0, f"expected a near miss, closest was {closest:.2f} m"


@pytest.mark.asyncio
async def test_perception_clears_the_planned_leg():
    """Detect poles, infer the span, replan -- the outbound leg goes clean."""
    pipe, leg, breaches, _ = await fly(perceive=True, planned_rtl=False)
    assert breaches["navigate"] == 0, (
        f"planned leg should avoid the inferred conductor, got {breaches['navigate']}"
    )
    assert leg.wire_replans > 0, "should have replanned around a newly inferred wire"
    pipe.rebuild()
    assert len(pipe.spans) >= 2, "should have inferred most of the line"


@pytest.mark.asyncio
async def test_direct_rtl_ignores_the_map_it_just_built():
    """Regression cover for the hole this found: RTL handed the job to the
    flight controller, which flies straight home through a mapped obstacle."""
    _, _, breaches, _ = await fly(perceive=True, planned_rtl=False)
    assert breaches["rtl"] > 0, (
        "if this stops failing, plain ReturnToLaunch has become map-aware and "
        "PlannedReturn may be redundant"
    )


@pytest.mark.asyncio
async def test_planned_return_closes_the_gap():
    """The full chain: nothing breaches, on either leg."""
    _, _, breaches, closest = await fly(perceive=True, planned_rtl=True)
    assert sum(breaches.values()) == 0, f"breaches remain: {dict(breaches)}"
    assert closest > CLEARANCE_M, f"closest approach {closest:.2f} m"


@pytest.mark.asyncio
async def test_standoff_matches_the_configured_margin():
    """Clearance should reflect thin_margin, not luck. An inferred wire needs
    a wider berth than a mapped building because its position carries the
    triangulation error of the poles it came from."""
    _, _, _, closest = await fly(perceive=True, planned_rtl=True)
    assert closest > 5.0, (
        f"expected standoff near the 9 m thin_margin, got {closest:.2f} m"
    )


@pytest.mark.asyncio
async def test_planned_return_falls_back_when_planning_fails():
    """A degraded return beats hovering until the battery runs out."""
    sealed = City(extent_m=20)  # goal outside bounds: no route exists
    planner = GridPlanner(sealed, cell_m=6, margin_m=3)
    rtl = PlannedReturn(planner, speed_mps=6.0, home=Vec3(500, 500, 0))
    runner = MissionRunner(
        KinematicBackend(battery_endurance_s=1200),
        [Takeoff(12.0), rtl],
        safety=Permissive(), verbose=False, timeout_s=200,
    )
    await runner.run()
    assert rtl.direct, "should have fallen back to a direct return"
