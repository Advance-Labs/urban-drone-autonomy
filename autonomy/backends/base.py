"""The seam between autonomy logic and whatever is actually flying.

This is the most important boundary in the project. Everything above it
(behaviours, planner, safety) is written once. Everything below it is
swappable: a kinematic sim today, PX4 SITL tomorrow, a Pixhawk on a real
airframe after that.

Real drones draw this line at MAVLink -- autonomy code sends high-level
intent (`goto`, `takeoff`, `land`) and the flight controller runs the
400 Hz attitude loops. We draw it in the same place, so nothing above this
file has to change when the thing below it becomes real hardware.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..types import Telemetry, Vec3


@runtime_checkable
class VehicleBackend(Protocol):
    """Minimal command set an autonomy stack needs from a multirotor.

    Every method is intent, not actuation. No backend implementation should
    ever expose motor outputs to the layers above it.
    """

    async def connect(self) -> None:
        """Establish the link and block until telemetry is flowing."""
        ...

    async def arm(self) -> None:
        """Arm the motors. Must fail loudly rather than silently no-op."""
        ...

    async def takeoff(self, altitude_m: float) -> None:
        """Climb to `altitude_m` above launch and hold."""
        ...

    async def goto(self, target: Vec3, speed_mps: float) -> None:
        """Command a position setpoint. Returns immediately -- arrival is
        observed through telemetry, never assumed from the call returning."""
        ...

    async def look_at(self, point: Vec3 | None) -> None:
        """Point the airframe (and thus a fixed camera) at `point`.

        `None` releases the lock and returns to heading-follows-velocity.
        Separate from `goto` because where you fly and where you look are
        independent: observing a doorway while orbiting it needs both at once.
        """
        ...

    async def land(self) -> None:
        """Descend and disarm at the current horizontal position."""
        ...

    async def return_to_launch(self) -> None:
        """Fly home and land. Backend decides the route and altitude."""
        ...

    async def telemetry(self) -> Telemetry:
        """Current vehicle state. Cheap enough to call every control tick."""
        ...

    async def step(self, dt: float) -> None:
        """Advance simulated time by `dt` seconds.

        A no-op for backends attached to something that moves in real time.
        Its presence lets the mission runner drive a sim faster than
        wall-clock without the autonomy layer knowing the difference.
        """
        ...

    async def close(self) -> None:
        """Release the link. Safe to call twice."""
        ...
