"""Core value types shared by every layer of the autonomy stack.

Coordinates use the NED convention that PX4 and ArduPilot both speak:
    north = +x, east = +y, down = +z
so altitude above launch is always `-down`. Keeping NED here means the
MAVSDK backend needs no frame conversion, and the kinematic sim can't
quietly disagree with the real flight controller about which way is up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True, slots=True)
class Vec3:
    """A point or vector in the local NED frame, metres."""

    north: float = 0.0
    east: float = 0.0
    down: float = 0.0

    @property
    def altitude(self) -> float:
        """Height above the launch point, metres. Positive is up."""
        return -self.down

    def __add__(self, other: Vec3) -> Vec3:
        return Vec3(self.north + other.north, self.east + other.east, self.down + other.down)

    def __sub__(self, other: Vec3) -> Vec3:
        return Vec3(self.north - other.north, self.east - other.east, self.down - other.down)

    def __mul__(self, scalar: float) -> Vec3:
        return Vec3(self.north * scalar, self.east * scalar, self.down * scalar)

    @property
    def norm(self) -> float:
        return math.sqrt(self.north**2 + self.east**2 + self.down**2)

    def normalised(self) -> Vec3:
        """Unit vector in the same direction. Zero vectors stay zero."""
        n = self.norm
        return self if n < 1e-9 else self * (1.0 / n)

    def horizontal_distance_to(self, other: Vec3) -> float:
        """Ground-track distance, ignoring altitude difference."""
        return math.hypot(other.north - self.north, other.east - self.east)


class FlightMode(str, Enum):
    """Coarse vehicle mode. Deliberately smaller than PX4's full mode set --
    the autonomy layer only needs to know what it is allowed to command next."""

    GROUNDED = "grounded"
    TAKING_OFF = "taking_off"
    HOLDING = "holding"
    FLYING = "flying"
    RETURNING = "returning"
    LANDING = "landing"


@dataclass(frozen=True, slots=True)
class Telemetry:
    """One snapshot of vehicle state, as the autonomy layer sees it.

    Frozen so a behaviour can stash a reading without it mutating underneath.
    """

    position: Vec3
    velocity: Vec3
    heading_deg: float
    battery_remaining: float  # 0.0 - 1.0
    mode: FlightMode
    armed: bool
    link_ok: bool
    elapsed_s: float
    # Estimated wind in the local frame, metres/second, pointing the way the
    # air is moving. Real vehicles derive this from the gap between commanded
    # and achieved velocity; zero means "no estimate yet", not "no wind".
    wind: Vec3 = Vec3()

    @property
    def altitude(self) -> float:
        return self.position.altitude


class SafetyAction(str, Enum):
    """What the safety monitor wants the mission to do right now.

    Ordered by severity: a later member always wins over an earlier one when
    two rules fire on the same tick.
    """

    CONTINUE = "continue"
    WARN = "warn"
    RETURN_TO_LAUNCH = "return_to_launch"
    LAND_NOW = "land_now"

    @property
    def severity(self) -> int:
        order = [
            SafetyAction.CONTINUE,
            SafetyAction.WARN,
            SafetyAction.RETURN_TO_LAUNCH,
            SafetyAction.LAND_NOW,
        ]
        return order.index(self)


@dataclass(frozen=True, slots=True)
class SafetyVerdict:
    """A safety decision plus the human-readable reason it was made.

    The reason is not decoration -- it is what gets logged and what you read
    in the post-flight review to understand why a mission cut short.
    """

    action: SafetyAction
    reason: str = ""
    # Where to go, when the action alone is ambiguous. LAND_NOW with a target
    # means "divert to this spot and land"; LAND_NOW without one means the
    # situation is bad enough to put it down right here.
    target: Vec3 | None = None

    @staticmethod
    def ok() -> SafetyVerdict:
        return SafetyVerdict(SafetyAction.CONTINUE)
