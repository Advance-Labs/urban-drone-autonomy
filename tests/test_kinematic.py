"""Vehicle model behaviour. These pin the properties autonomy code relies on."""

from __future__ import annotations

import pytest

from autonomy import FlightMode, Vec3
from autonomy.backends import KinematicBackend


async def fly_for(backend: KinematicBackend, seconds: float, dt: float = 0.1) -> None:
    for _ in range(int(seconds / dt)):
        await backend.step(dt)


@pytest.mark.asyncio
async def test_starts_grounded_and_disarmed():
    b = KinematicBackend()
    await b.connect()
    t = await b.telemetry()
    assert not t.armed
    assert t.mode is FlightMode.GROUNDED
    assert t.altitude == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_takeoff_requires_arming():
    b = KinematicBackend()
    await b.connect()
    with pytest.raises(RuntimeError, match="not armed"):
        await b.takeoff(10.0)


@pytest.mark.asyncio
async def test_takeoff_reaches_commanded_altitude():
    b = KinematicBackend()
    await b.connect()
    await b.arm()
    await b.takeoff(15.0)
    await fly_for(b, 30)
    t = await b.telemetry()
    assert t.altitude == pytest.approx(15.0, abs=0.5)


@pytest.mark.asyncio
async def test_vehicle_does_not_teleport():
    """Guidance must respect the speed ceiling -- autonomy code that assumes
    instant arrival is the single most common sim-to-hardware failure."""
    b = KinematicBackend(max_speed_mps=5.0)
    await b.connect()
    await b.arm()
    await b.takeoff(10.0)
    await fly_for(b, 20)
    await b.goto(Vec3(100, 0, -10), 5.0)
    await fly_for(b, 2)
    t = await b.telemetry()
    assert t.position.north < 12.0, "moved further than max speed allows"


@pytest.mark.asyncio
async def test_decelerates_on_arrival_instead_of_overshooting():
    b = KinematicBackend(max_speed_mps=8.0)
    await b.connect()
    await b.arm()
    await b.takeoff(10.0)
    await fly_for(b, 20)
    await b.goto(Vec3(40, 0, -10), 8.0)
    await fly_for(b, 40)
    t = await b.telemetry()
    assert t.position.north == pytest.approx(40.0, abs=1.5)
    assert t.velocity.norm < 0.5, "should have settled, not sailed past"


@pytest.mark.asyncio
async def test_battery_drains_faster_when_moving():
    idle = KinematicBackend(battery_endurance_s=100.0)
    await idle.connect()
    await idle.arm()
    await idle.takeoff(5.0)
    await fly_for(idle, 12)

    busy = KinematicBackend(battery_endurance_s=100.0)
    await busy.connect()
    await busy.arm()
    await busy.takeoff(5.0)
    await fly_for(busy, 2)
    await busy.goto(Vec3(200, 0, -5), 8.0)
    await fly_for(busy, 10)

    assert (await busy.telemetry()).battery_remaining < (await idle.telemetry()).battery_remaining


@pytest.mark.asyncio
async def test_landing_disarms_the_vehicle():
    b = KinematicBackend()
    await b.connect()
    await b.arm()
    await b.takeoff(10.0)
    await fly_for(b, 20)
    await b.land()
    await fly_for(b, 40)
    t = await b.telemetry()
    assert not t.armed
    assert t.altitude == pytest.approx(0.0, abs=0.2)


@pytest.mark.asyncio
async def test_never_sinks_below_ground():
    b = KinematicBackend()
    await b.connect()
    await b.arm()
    await b.takeoff(3.0)
    await fly_for(b, 10)
    await b.land()
    await fly_for(b, 60)
    assert (await b.telemetry()).position.down <= 0.0 + 1e-9


@pytest.mark.asyncio
async def test_integration_is_stable_at_large_timesteps():
    """The exponential-lag form must not diverge when dt exceeds the time
    constant -- a naive Euler blend would explode here."""
    b = KinematicBackend(accel_tau_s=0.3)
    await b.connect()
    await b.arm()
    await b.takeoff(10.0)
    for _ in range(50):
        await b.step(2.0)  # dt far larger than tau
    t = await b.telemetry()
    assert t.altitude == pytest.approx(10.0, abs=1.0)
    assert t.velocity.norm < 5.0


@pytest.mark.parametrize("bad", [{"max_speed_mps": 0}, {"accel_tau_s": -1}, {"battery_endurance_s": 0}])
def test_rejects_invalid_configuration(bad):
    with pytest.raises(ValueError):
        KinematicBackend(**bad)


class TestWind:
    """Regression cover for the steady-state error a P-controller shows under
    a constant disturbance. Without wind feedforward the vehicle parks
    downwind of every setpoint and never arrives."""

    @pytest.mark.asyncio
    async def test_holds_station_in_wind(self):
        b = KinematicBackend(max_speed_mps=8.0, wind=Vec3(5.0, 2.0, 0.0))
        await b.connect()
        await b.arm()
        await b.takeoff(15.0)
        await fly_for(b, 40)
        await b.goto(Vec3(0, 0, -15), 6.0)
        await fly_for(b, 60)
        t = await b.telemetry()
        drift = t.position.horizontal_distance_to(Vec3())
        assert drift < 1.0, f"parked {drift:.1f} m downwind instead of holding"

    @pytest.mark.asyncio
    async def test_reaches_a_waypoint_upwind(self):
        b = KinematicBackend(max_speed_mps=8.0, wind=Vec3(-5.0, 0.0, 0.0))
        await b.connect()
        await b.arm()
        await b.takeoff(15.0)
        await fly_for(b, 40)
        target = Vec3(60, 0, -15)
        await b.goto(target, 7.0)
        await fly_for(b, 90)
        assert (await b.telemetry()).position.horizontal_distance_to(target) < 1.5

    @pytest.mark.asyncio
    async def test_takeoff_is_not_slowed_by_crosswind(self):
        """The 75 s takeoff that exposed this bug -- should be ~8 s."""
        b = KinematicBackend(wind=Vec3(5.0, 2.0, 0.0))
        await b.connect()
        await b.arm()
        await b.takeoff(20.0)
        await fly_for(b, 20)
        t = await b.telemetry()
        assert t.altitude == pytest.approx(20.0, abs=0.6)

    @pytest.mark.asyncio
    async def test_telemetry_reports_ground_velocity_and_wind(self):
        wind = Vec3(4.0, 0.0, 0.0)
        b = KinematicBackend(wind=wind)
        await b.connect()
        await b.arm()
        await b.takeoff(10.0)
        await fly_for(b, 25)
        t = await b.telemetry()
        assert t.wind.north == pytest.approx(4.0)
        # Station-keeping: ground speed near zero even though the vehicle is
        # crabbing into the wind at ~4 m/s airspeed.
        assert t.velocity.horizontal_distance_to(Vec3()) < 1.0

    @pytest.mark.asyncio
    async def test_headwind_costs_more_battery_than_calm(self):
        async def run(wind: Vec3) -> float:
            b = KinematicBackend(battery_endurance_s=600.0, wind=wind)
            await b.connect()
            await b.arm()
            await b.takeoff(10.0)
            await fly_for(b, 20)
            await b.goto(Vec3(80, 0, -10), 6.0)
            await fly_for(b, 60)
            return (await b.telemetry()).battery_remaining

        assert await run(Vec3(-5, 0, 0)) < await run(Vec3())
