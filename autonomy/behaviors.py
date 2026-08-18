"""Behaviours: the mission state machine.

Each behaviour is a small object with one job and a clear completion test.
The mission runner ticks the active one and moves on when it reports done.

Why a state machine rather than a script of `await` calls: a script blocks.
If takeoff is a blocking await, nothing can preempt it when the battery drops
or the geofence trips. Behaviours are non-blocking -- each `tick` returns
promptly, so the safety monitor gets a turn on every cycle and can abort
mid-behaviour. That property is the whole reason for the shape.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .backends.base import VehicleBackend
from .types import Telemetry, Vec3


class Behavior(ABC):
    """One step of a mission."""

    name: str = "behavior"

    def __init__(self) -> None:
        self._started = False

    async def tick(self, vehicle: VehicleBackend, telemetry: Telemetry) -> None:
        """Called once per control cycle while this behaviour is active.

        Issues `on_start` the first time, then `on_tick` thereafter. Must not
        block for longer than a control cycle.
        """
        if not self._started:
            self._started = True
            await self.on_start(vehicle, telemetry)
        else:
            await self.on_tick(vehicle, telemetry)

    async def on_start(self, vehicle: VehicleBackend, telemetry: Telemetry) -> None:
        """Issue the command that sets this behaviour in motion."""

    async def on_tick(self, vehicle: VehicleBackend, telemetry: Telemetry) -> None:
        """Optional per-cycle work. Most behaviours command once and wait."""

    @abstractmethod
    def is_complete(self, telemetry: Telemetry) -> bool:
        """True when the runner should advance to the next behaviour."""

    def reset(self) -> None:
        self._started = False

    def describe(self) -> str:
        return self.name


class Takeoff(Behavior):
    """Arm and climb to a target altitude."""

    name = "takeoff"

    def __init__(self, altitude_m: float) -> None:
        super().__init__()
        if altitude_m <= 0:
            raise ValueError("takeoff altitude must be positive")
        self.altitude = altitude_m

    async def on_start(self, vehicle: VehicleBackend, telemetry: Telemetry) -> None:
        await vehicle.arm()
        await vehicle.takeoff(self.altitude)

    def is_complete(self, telemetry: Telemetry) -> bool:
        # 0.5 m tolerance, and require the climb to have settled -- arriving at
        # altitude with 3 m/s still on the clock means an overshoot is coming.
        return (
            abs(telemetry.altitude - self.altitude) < 0.5
            and abs(telemetry.velocity.down) < 0.3
        )

    def describe(self) -> str:
        return f"takeoff to {self.altitude:.0f} m"


class FollowPath(Behavior):
    """Fly an ordered list of waypoints, advancing on arrival."""

    name = "follow_path"

    def __init__(
        self,
        waypoints: list[Vec3],
        *,
        speed_mps: float = 6.0,
        arrival_radius_m: float = 1.5,
    ) -> None:
        super().__init__()
        if not waypoints:
            raise ValueError("path needs at least one waypoint")
        if speed_mps <= 0:
            raise ValueError("speed must be positive")
        if arrival_radius_m <= 0:
            raise ValueError("arrival radius must be positive")
        self.waypoints = list(waypoints)
        self.speed = speed_mps
        self.arrival_radius = arrival_radius_m
        self.index = 0

    async def on_start(self, vehicle: VehicleBackend, telemetry: Telemetry) -> None:
        await vehicle.goto(self.waypoints[0], self.speed)

    async def on_tick(self, vehicle: VehicleBackend, telemetry: Telemetry) -> None:
        if self.index >= len(self.waypoints):
            return
        if self._reached(telemetry, self.waypoints[self.index]):
            self.index += 1
            if self.index < len(self.waypoints):
                await vehicle.goto(self.waypoints[self.index], self.speed)

    def _reached(self, telemetry: Telemetry, waypoint: Vec3) -> bool:
        return (waypoint - telemetry.position).norm <= self.arrival_radius

    def is_complete(self, telemetry: Telemetry) -> bool:
        return self.index >= len(self.waypoints)

    @property
    def progress(self) -> tuple[int, int]:
        return min(self.index + 1, len(self.waypoints)), len(self.waypoints)

    def reset(self) -> None:
        super().reset()
        self.index = 0

    def describe(self) -> str:
        return f"follow path ({len(self.waypoints)} waypoints)"


class Hold(Behavior):
    """Hover in place for a fixed duration.

    Useful as a settle step between behaviours, and as the thing you put in a
    mission when you want to prove the vehicle can just sit still.
    """

    name = "hold"

    def __init__(self, duration_s: float) -> None:
        super().__init__()
        if duration_s < 0:
            raise ValueError("hold duration cannot be negative")
        self.duration = duration_s
        self._until: float | None = None

    async def on_start(self, vehicle: VehicleBackend, telemetry: Telemetry) -> None:
        self._until = telemetry.elapsed_s + self.duration
        await vehicle.goto(telemetry.position, 1.0)

    def is_complete(self, telemetry: Telemetry) -> bool:
        return self._until is not None and telemetry.elapsed_s >= self._until

    def reset(self) -> None:
        super().reset()
        self._until = None

    def describe(self) -> str:
        return f"hold {self.duration:.0f} s"


class ReturnToLaunch(Behavior):
    """Fly home and land."""

    name = "rtl"

    async def on_start(self, vehicle: VehicleBackend, telemetry: Telemetry) -> None:
        await vehicle.return_to_launch()

    async def on_tick(self, vehicle: VehicleBackend, telemetry: Telemetry) -> None:
        # Once home is reached laterally, commit to the descent.
        if telemetry.position.horizontal_distance_to(Vec3()) < 2.0 and telemetry.altitude > 0.3:
            await vehicle.land()

    def is_complete(self, telemetry: Telemetry) -> bool:
        return not telemetry.armed and telemetry.altitude < 0.3

    def describe(self) -> str:
        return "return to launch"


class Land(Behavior):
    """Descend and disarm at the current position."""

    name = "land"

    async def on_start(self, vehicle: VehicleBackend, telemetry: Telemetry) -> None:
        await vehicle.land()

    def is_complete(self, telemetry: Telemetry) -> bool:
        return not telemetry.armed and telemetry.altitude < 0.3

    def describe(self) -> str:
        return "land"


class DivertAndLand(Behavior):
    """Fly to an emergency landing site, then put down.

    Used when the battery can no longer reach home. Transits at the site's
    altitude, then commits to the descent once overhead -- descending early
    would waste the little charge left fighting ground effect on the way in.
    """

    name = "divert"

    def __init__(
        self,
        target: Vec3,
        *,
        speed_mps: float = 6.0,
        site_name: str = "site",
        arrival_radius_m: float = 3.0,
    ) -> None:
        super().__init__()
        if speed_mps <= 0:
            raise ValueError("speed must be positive")
        if arrival_radius_m <= 0:
            raise ValueError("arrival radius must be positive")
        self.target = target
        self.speed = speed_mps
        self.site_name = site_name
        self.arrival_radius = arrival_radius_m
        self._descending = False

    async def on_start(self, vehicle: VehicleBackend, telemetry: Telemetry) -> None:
        # Hold current altitude on the way over; only the horizontal position
        # is the transit goal.
        await vehicle.goto(
            Vec3(self.target.north, self.target.east, telemetry.position.down),
            self.speed,
        )

    async def on_tick(self, vehicle: VehicleBackend, telemetry: Telemetry) -> None:
        if self._descending:
            return
        overhead = telemetry.position.horizontal_distance_to(self.target)
        if overhead <= self.arrival_radius:
            self._descending = True
            await vehicle.land()

    def is_complete(self, telemetry: Telemetry) -> bool:
        return not telemetry.armed and telemetry.altitude < 0.3

    def reset(self) -> None:
        super().reset()
        self._descending = False

    def describe(self) -> str:
        return f"divert to {self.site_name}"
