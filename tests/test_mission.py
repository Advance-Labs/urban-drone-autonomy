"""End-to-end mission runs against the kinematic backend.

These use a permissive stub safety policy so they exercise the runner and the
behaviour state machine independently of the real policy, which is still
being written. Once `SafetyMonitor.evaluate` is implemented, the stub can go
and these become genuine integration tests.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from autonomy import (
    FollowPath,
    Hold,
    MissionRunner,
    ReturnToLaunch,
    SafetyAction,
    SafetyLimits,
    SafetyMonitor,
    SafetyVerdict,
    Takeoff,
    Vec3,
    survey_grid,
)
from autonomy.backends import KinematicBackend


class PermissiveSafety(SafetyMonitor):
    """Never intervenes. Isolates runner behaviour from policy."""

    def evaluate(self, telemetry):
        return SafetyVerdict.ok()


class BatteryFloorSafety(SafetyMonitor):
    """Minimal real policy: land below a floor. Used to test preemption."""

    def __init__(self, floor: float) -> None:
        super().__init__()
        self.floor = floor

    def evaluate(self, telemetry):
        if telemetry.battery_remaining < self.floor:
            return SafetyVerdict(
                SafetyAction.LAND_NOW,
                f"battery {telemetry.battery_remaining:.0%} below floor",
            )
        return SafetyVerdict.ok()


@pytest.mark.asyncio
async def test_hover_mission_completes_and_lands():
    runner = MissionRunner(
        vehicle=KinematicBackend(),
        behaviors=[Takeoff(10.0), Hold(5.0), ReturnToLaunch()],
        safety=PermissiveSafety(),
        verbose=False,
    )
    result = await runner.run()
    assert result.completed
    assert result.aborted_reason is None


@pytest.mark.asyncio
async def test_survey_mission_visits_every_waypoint():
    waypoints = survey_grid(width_m=40, height_m=30, spacing_m=15, altitude_m=15)
    path = FollowPath(waypoints, speed_mps=7.0)
    runner = MissionRunner(
        vehicle=KinematicBackend(max_speed_mps=9.0),
        behaviors=[Takeoff(15.0), path, ReturnToLaunch()],
        safety=PermissiveSafety(),
        verbose=False,
    )
    result = await runner.run()
    assert result.completed
    assert path.index == len(waypoints), "did not reach every waypoint"


@pytest.mark.asyncio
async def test_safety_preempts_the_active_behaviour():
    """A dying battery mid-survey must cut the mission short, not let the
    path behaviour run to completion first."""
    waypoints = survey_grid(width_m=300, height_m=300, spacing_m=40, altitude_m=25)
    path = FollowPath(waypoints, speed_mps=7.0)
    runner = MissionRunner(
        vehicle=KinematicBackend(battery_endurance_s=60.0),
        behaviors=[Takeoff(25.0), path, ReturnToLaunch()],
        safety=BatteryFloorSafety(0.30),
        verbose=False,
    )
    result = await runner.run()
    assert not result.completed
    assert result.aborted_reason is not None and "battery" in result.aborted_reason
    assert path.index < len(waypoints), "safety fired too late to matter"


@pytest.mark.asyncio
async def test_aborted_mission_still_lands():
    runner = MissionRunner(
        vehicle=KinematicBackend(battery_endurance_s=60.0),
        behaviors=[Takeoff(25.0), Hold(600.0), ReturnToLaunch()],
        safety=BatteryFloorSafety(0.40),
        verbose=False,
    )
    result = await runner.run()
    assert result.aborted_reason is not None
    assert result.log[-1].altitude < 1.0, "aborted mission left the vehicle airborne"


@pytest.mark.asyncio
async def test_mission_timeout_is_enforced():
    runner = MissionRunner(
        vehicle=KinematicBackend(battery_endurance_s=1e6),
        behaviors=[Takeoff(10.0), Hold(1e6)],
        safety=PermissiveSafety(),
        timeout_s=45.0,
        verbose=False,
    )
    result = await runner.run()
    assert not result.completed
    assert "timeout" in (result.aborted_reason or "")


@pytest.mark.asyncio
async def test_real_policy_flies_a_nominal_mission():
    """The shipped policy must not be so twitchy it aborts a healthy flight."""
    runner = MissionRunner(
        vehicle=KinematicBackend(battery_endurance_s=1200.0),
        behaviors=[Takeoff(15.0), Hold(10.0), ReturnToLaunch()],
        safety=SafetyMonitor(SafetyLimits(max_radius_m=200)),
        verbose=False,
    )
    result = await runner.run()
    assert result.completed, f"nominal flight aborted: {result.aborted_reason}"


@pytest.mark.asyncio
async def test_real_policy_diverts_when_home_is_out_of_reach():
    """Stranded far out on a small pack: divert to a site rather than
    attempt an RTL that cannot finish."""
    from autonomy import LandingSite, LandingSiteSelector

    sites = LandingSiteSelector(
        sites=[LandingSite(position=Vec3(240, 0, 0), name="ballfield")]
    )
    waypoints = [Vec3(250, 0, -25), Vec3(300, 50, -25)]
    runner = MissionRunner(
        vehicle=KinematicBackend(battery_endurance_s=90.0, max_speed_mps=9.0),
        behaviors=[Takeoff(25.0), FollowPath(waypoints, speed_mps=8.0), ReturnToLaunch()],
        safety=SafetyMonitor(SafetyLimits(max_radius_m=1000), landing_sites=sites),
        verbose=False,
    )
    result = await runner.run()
    assert not result.completed
    assert result.landed_safely, "diverted mission must still reach the ground"
    assert any("DIVERT" in e.phase or "LAND" in e.phase for e in result.log)


@pytest.mark.asyncio
async def test_link_loss_does_not_abort_on_a_brief_glitch():
    vehicle = KinematicBackend(battery_endurance_s=1200.0)
    runner = MissionRunner(
        vehicle=vehicle,
        behaviors=[Takeoff(12.0), Hold(20.0), ReturnToLaunch()],
        safety=SafetyMonitor(SafetyLimits(link_loss_grace_s=6.0)),
        verbose=False,
    )

    async def glitch() -> None:
        """Drop the link briefly, well inside the grace period."""
        while (await vehicle.telemetry()).elapsed_s < 8.0:
            await asyncio.sleep(0)
        vehicle.set_link(False)
        while (await vehicle.telemetry()).elapsed_s < 10.0:
            await asyncio.sleep(0)
        vehicle.set_link(True)

    await asyncio.gather(runner.run(), glitch())
    assert runner._abort_reason is None, "a 2 s dropout should not abort"


@pytest.mark.asyncio
async def test_flight_record_serialises_for_the_viewer():
    runner = MissionRunner(
        vehicle=KinematicBackend(battery_endurance_s=1200.0, wind=Vec3(3, 1, 0)),
        behaviors=[Takeoff(12.0), Hold(4.0), ReturnToLaunch()],
        safety=SafetyMonitor(),
        verbose=False,
    )
    result = await runner.run()
    payload = result.to_dict()
    assert payload["track"], "viewer needs a continuous track, not just events"
    assert {"t", "n", "e", "d", "batt", "phase"} <= set(payload["track"][0])
    assert payload["context"]["geofence_radius_m"] > 0
    json.dumps(payload)  # must be JSON-serialisable end to end


def test_runner_rejects_empty_mission():
    with pytest.raises(ValueError):
        MissionRunner(vehicle=KinematicBackend(), behaviors=[])


@pytest.mark.asyncio
async def test_follow_path_rejects_empty_waypoints():
    with pytest.raises(ValueError):
        FollowPath([], speed_mps=5.0)


@pytest.mark.asyncio
async def test_behaviors_run_in_order():
    order: list[str] = []

    class Spy(Hold):
        def __init__(self, tag: str) -> None:
            super().__init__(1.0)
            self.tag = tag

        async def on_start(self, vehicle, telemetry):
            order.append(self.tag)
            await super().on_start(vehicle, telemetry)

    runner = MissionRunner(
        vehicle=KinematicBackend(),
        behaviors=[Takeoff(8.0), Spy("a"), Spy("b"), ReturnToLaunch()],
        safety=PermissiveSafety(),
        verbose=False,
    )
    await runner.run()
    assert order == ["a", "b"]
