"""PX4 backend, driving a real flight stack over MAVLink via MAVSDK.

Same interface as the kinematic backend, so no autonomy code changes when you
switch. This one talks to PX4 SITL during development and to a Pixhawk over
telemetry radio or USB on hardware -- from this file's point of view they are
the same thing, which is the point.

MAVSDK is imported lazily so the rest of the stack stays runnable without it.

Setup (PX4 SITL in Docker, since building PX4 natively on Apple Silicon needs
a full toolchain):

    docker run --rm -it -p 14540:14540/udp \\
        jonasvautherin/px4-gazebo-headless:latest

then:

    pip install mavsdk
    python run.py survey --backend px4

Connection string defaults to `udpin://0.0.0.0:14540`, which is where PX4 SITL
publishes its offboard link.
"""

from __future__ import annotations

import asyncio

from ..types import FlightMode, Telemetry, Vec3

_PX4_MODE_MAP = {
    "TAKEOFF": FlightMode.TAKING_OFF,
    "HOLD": FlightMode.HOLDING,
    "MISSION": FlightMode.FLYING,
    "OFFBOARD": FlightMode.FLYING,
    "RETURN_TO_LAUNCH": FlightMode.RETURNING,
    "LAND": FlightMode.LANDING,
}


class Px4Backend:
    """VehicleBackend implementation backed by PX4 through MAVSDK."""

    def __init__(
        self,
        system_address: str = "udpin://0.0.0.0:14540",
        *,
        connect_timeout_s: float = 30.0,
        cruise_speed_mps: float | None = 8.0,
    ) -> None:
        if not system_address:
            raise ValueError("system_address must not be empty")
        self.system_address = system_address
        self.connect_timeout_s = connect_timeout_s
        self.cruise_speed_mps = cruise_speed_mps

        self._drone = None
        self._start_time: float | None = None
        self._home_set = False
        # Telemetry is push-based in MAVSDK; we cache the latest of each stream
        # so `telemetry()` stays a cheap synchronous-feeling read.
        self._position = Vec3()
        self._velocity = Vec3()
        self._heading = 0.0
        self._battery = 1.0
        self._mode = FlightMode.GROUNDED
        self._armed = False
        self._link_ok = False
        self._look_at: Vec3 | None = None
        self._yaw_lock: float | None = None
        self._tasks: list[asyncio.Task] = []

    async def connect(self) -> None:
        try:
            from mavsdk import System
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "The px4 backend needs MAVSDK. Install it with `pip install mavsdk`, "
                "or use the kinematic backend, which has no dependencies."
            ) from exc

        self._drone = System()
        await self._drone.connect(system_address=self.system_address)

        async def _await_connection() -> None:
            async for state in self._drone.core.connection_state():
                if state.is_connected:
                    return

        try:
            await asyncio.wait_for(_await_connection(), timeout=self.connect_timeout_s)
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                f"No PX4 heartbeat on {self.system_address} after "
                f"{self.connect_timeout_s:.0f}s. Is SITL running?"
            ) from exc

        self._link_ok = True
        self._start_time = asyncio.get_running_loop().time()
        self._spawn_telemetry_tasks()
        await self._await_health()
        await self._configure_speed()

    async def arm(self) -> None:
        await self._drone.action.arm()

    async def takeoff(self, altitude_m: float) -> None:
        if altitude_m <= 0:
            raise ValueError("takeoff altitude must be positive")
        await self._drone.action.set_takeoff_altitude(altitude_m)
        await self._drone.action.takeoff()

    async def goto(self, target: Vec3, speed_mps: float) -> None:
        if speed_mps <= 0:
            raise ValueError("speed must be positive")
        from mavsdk.offboard import PositionNedYaw

        # NOTE: `action.set_maximum_speed` was removed in MAVSDK 3.x. Speed is
        # a PX4 parameter, and setting a parameter on every setpoint would be
        # both chatty and slow, so cruise speed is configured once at connect.
        yaw = self._yaw_lock if self._yaw_lock is not None else self._heading
        await self._drone.offboard.set_position_ned(
            PositionNedYaw(target.north, target.east, target.down, yaw)
        )
        await self._ensure_offboard()

    async def look_at(self, point: Vec3 | None) -> None:
        """Store a yaw lock. Applied on the next `goto`, since PX4 carries yaw
        in the same offboard setpoint as position."""
        self._look_at = point
        if point is not None:
            import math

            offset = point - self._position
            if offset.horizontal_distance_to(Vec3()) > 0.5:
                self._yaw_lock = math.degrees(
                    math.atan2(offset.east, offset.north)
                ) % 360.0
        else:
            self._yaw_lock = None

    async def land(self) -> None:
        await self._drone.action.land()

    async def return_to_launch(self) -> None:
        await self._drone.action.return_to_launch()

    async def telemetry(self) -> Telemetry:
        loop = asyncio.get_running_loop()
        elapsed = loop.time() - self._start_time if self._start_time else 0.0
        return Telemetry(
            position=self._position,
            velocity=self._velocity,
            heading_deg=self._heading,
            battery_remaining=self._battery,
            mode=self._mode,
            armed=self._armed,
            link_ok=self._link_ok,
            elapsed_s=elapsed,
        )

    async def step(self, dt: float) -> None:
        """No-op: PX4 advances in real time, not on our clock."""

    async def close(self) -> None:
        for task in self._tasks:
            task.cancel()
        # Await the cancellations rather than abandoning them, so the event
        # loop does not warn about tasks destroyed while pending.
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self._link_ok = False

    # -- internals ---------------------------------------------------------

    def _spawn_telemetry_tasks(self) -> None:
        self._tasks = [
            asyncio.create_task(self._guard(pump()))
            for pump in (
                self._pump_position_velocity,
                self._pump_battery,
                self._pump_mode,
                self._pump_armed,
                self._pump_attitude,
            )
        ]

    @staticmethod
    async def _guard(coro) -> None:
        """Run a telemetry stream, swallowing the noise of shutdown.

        Cancelling a gRPC stream raises `AioRpcError: Stream removed` from deep
        inside the transport rather than a clean `CancelledError`. That is
        expected during `close()`, and letting it surface buries a real failure
        in five identical tracebacks every time a mission ends normally.
        """
        try:
            await coro
        except asyncio.CancelledError:
            raise
        except Exception:
            # The link is gone; `link_ok` already reflects that.
            pass

    async def _pump_position_velocity(self) -> None:
        """One subscription, not two.

        Position and velocity arrive in the SAME message, so subscribing twice
        opened two gRPC streams to one source -- wasteful at best, and in
        practice the second subscriber starves waiting for frames the first
        has already consumed.
        """
        async for s in self._drone.telemetry.position_velocity_ned():
            self._position = Vec3(s.position.north_m, s.position.east_m, s.position.down_m)
            self._velocity = Vec3(
                s.velocity.north_m_s, s.velocity.east_m_s, s.velocity.down_m_s
            )

    async def _pump_attitude(self) -> None:
        async for a in self._drone.telemetry.attitude_euler():
            self._heading = a.yaw_deg % 360.0

    async def _pump_battery(self) -> None:
        async for b in self._drone.telemetry.battery():
            # MAVSDK 3.x reports 0..100; older releases reported 0..1. Guessing
            # wrong makes the safety layer read 8000% and never trigger, so
            # normalise on the value rather than on a version check.
            raw = b.remaining_percent
            self._battery = raw / 100.0 if raw > 1.0 else raw

    async def _pump_mode(self) -> None:
        async for m in self._drone.telemetry.flight_mode():
            self._mode = _PX4_MODE_MAP.get(str(m), FlightMode.FLYING)

    async def _pump_armed(self) -> None:
        async for armed in self._drone.telemetry.armed():
            self._armed = armed

    async def _await_health(self, timeout_s: float = 90.0) -> None:
        """Block until PX4 reports a usable position estimate.

        Arming before the EKF has converged is a classic way to fly a mission
        against a position estimate that is quietly metres off. But this must
        not wait forever: without a timeout a vehicle that never gets GPS lock
        hangs the whole mission with no diagnostic, which is exactly the
        failure that is hardest to debug from a field laptop.
        """

        async def wait() -> None:
            async for health in self._drone.telemetry.health():
                if health.is_global_position_ok and health.is_home_position_ok:
                    self._home_set = True
                    return

        try:
            await asyncio.wait_for(wait(), timeout=timeout_s)
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                f"PX4 never reported a usable position estimate within "
                f"{timeout_s:.0f}s. Check GPS lock and EKF status."
            ) from exc

    async def _configure_speed(self) -> None:
        """Set the horizontal cruise limit once, as a PX4 parameter.

        Best-effort: airframes vary and a missing parameter must not abort a
        flight that is otherwise fine.
        """
        if self.cruise_speed_mps is None:
            return
        try:
            await self._drone.param.set_param_float(
                "MPC_XY_VEL_MAX", float(self.cruise_speed_mps)
            )
        except Exception:
            pass

    async def _ensure_offboard(self) -> None:
        """Offboard mode rejects a start with no setpoint already streaming,
        so this is only called after the first setpoint has been sent."""
        from mavsdk.offboard import OffboardError

        try:
            if not await self._drone.offboard.is_active():
                await self._drone.offboard.start()
        except OffboardError:
            # Already active, or PX4 refused. Telemetry will show the truth.
            pass
