"""Surveillance sensing: what the camera can actually see, and what it has seen.

The interesting constraint in an urban environment is that being *near* a
target is not the same as observing it. Three things must hold at once:

    range      the target is close enough to resolve
    field      it falls inside the camera cone, given where the gimbal points
    sight      no building sits between the lens and the target

The third is what makes cities hard. A rooftop orbit that looks like perfect
coverage on a map can observe nothing at all, because every sightline into the
courtyard is blocked. Modelling occlusion is the entire point -- without it,
coverage planning degenerates into "fly within N metres", which teaches nothing.

Deliberately not modelled: resolution as a function of range (a target 200 m
out is "seen" as well as one at 20 m), motion blur, lighting, and rolling
shutter. For a low-light night sensor those matter a great deal, and the
`min_light` field is a placeholder marking where that belongs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .types import Telemetry, Vec3
from .world import City


@dataclass(frozen=True, slots=True)
class Camera:
    """A gimballed camera rigidly yawed with the airframe.

    Defaults approximate the STARVIS low-light module in the hardware plan:
    a moderately wide lens on a gimbal held at a fixed depression angle.
    """

    fov_deg: float = 62.0
    range_m: float = 140.0
    depression_deg: float = 35.0  # gimbal tilt below horizontal
    min_light: float = 0.0  # reserved: lux floor for a night sensor

    def __post_init__(self) -> None:
        if not 0 < self.fov_deg < 180:
            raise ValueError("fov must be within 0..180 degrees")
        if self.range_m <= 0:
            raise ValueError("range must be positive")
        if not -90 <= self.depression_deg <= 90:
            raise ValueError("depression must be within -90..90 degrees")

    def boresight(self, heading_deg: float) -> Vec3:
        """Unit vector the lens points along, in NED."""
        yaw = math.radians(heading_deg)
        pitch = math.radians(self.depression_deg)
        horizontal = math.cos(pitch)
        return Vec3(
            horizontal * math.cos(yaw),
            horizontal * math.sin(yaw),
            math.sin(pitch),  # +down, since depression looks downward
        )

    def sees(
        self,
        eye: Vec3,
        heading_deg: float,
        target: Vec3,
        city: City | None = None,
    ) -> bool:
        """All three conditions: range, field of view, and clear sightline."""
        offset = target - eye
        distance = offset.norm
        if distance > self.range_m or distance < 1e-6:
            return False

        cos_angle = self._dot(offset.normalised(), self.boresight(heading_deg))
        if cos_angle < math.cos(math.radians(self.fov_deg / 2)):
            return False

        return city is None or city.has_line_of_sight(eye, target)

    @staticmethod
    def _dot(a: Vec3, b: Vec3) -> float:
        return a.north * b.north + a.east * b.east + a.down * b.down


@dataclass(frozen=True, slots=True)
class SurveillanceTarget:
    """A point of interest that must be observed to satisfy the mission."""

    position: Vec3
    name: str = "target"
    # Seconds of accumulated observation before the target counts as covered.
    # A single frame proves nothing; dwell is what produces usable footage.
    required_dwell_s: float = 3.0

    def __post_init__(self) -> None:
        if self.required_dwell_s < 0:
            raise ValueError(f"{self.name}: dwell cannot be negative")


@dataclass(slots=True)
class CoverageTracker:
    """Accumulates observation time per target across a flight."""

    targets: list[SurveillanceTarget] = field(default_factory=list)
    camera: Camera = field(default_factory=Camera)
    city: City | None = None
    dwell: dict[str, float] = field(default_factory=dict)
    first_seen: dict[str, float] = field(default_factory=dict)

    def update(self, telemetry: Telemetry, dt: float) -> list[str]:
        """Advance coverage by one tick. Returns targets newly satisfied."""
        if dt <= 0:
            raise ValueError("dt must be positive")
        newly_covered = []
        for t in self.targets:
            if not self.camera.sees(
                telemetry.position, telemetry.heading_deg, t.position, self.city
            ):
                continue
            was_covered = self.is_covered(t)
            self.dwell[t.name] = self.dwell.get(t.name, 0.0) + dt
            self.first_seen.setdefault(t.name, telemetry.elapsed_s)
            if not was_covered and self.is_covered(t):
                newly_covered.append(t.name)
        return newly_covered

    def is_covered(self, target: SurveillanceTarget) -> bool:
        return self.dwell.get(target.name, 0.0) >= target.required_dwell_s

    def visible_now(self, telemetry: Telemetry) -> list[str]:
        """Targets in view this instant, for the live HUD."""
        return [
            t.name
            for t in self.targets
            if self.camera.sees(
                telemetry.position, telemetry.heading_deg, t.position, self.city
            )
        ]

    @property
    def covered(self) -> list[str]:
        return [t.name for t in self.targets if self.is_covered(t)]

    @property
    def missed(self) -> list[str]:
        return [t.name for t in self.targets if not self.is_covered(t)]

    @property
    def fraction(self) -> float:
        return len(self.covered) / len(self.targets) if self.targets else 1.0

    def summary(self) -> dict:
        return {
            "covered": self.covered,
            "missed": self.missed,
            "fraction": round(self.fraction, 3),
            "dwell": {k: round(v, 1) for k, v in self.dwell.items()},
        }

    def to_dict(self) -> dict:
        """Scene data for the viewer."""
        return {
            "camera": {
                "fov_deg": self.camera.fov_deg,
                "range_m": self.camera.range_m,
                "depression_deg": self.camera.depression_deg,
            },
            "targets": [
                {
                    "name": t.name,
                    "n": t.position.north,
                    "e": t.position.east,
                    "d": t.position.down,
                    "dwell_req": t.required_dwell_s,
                    "covered": self.is_covered(t),
                }
                for t in self.targets
            ],
        }


def observation_point(
    target: SurveillanceTarget,
    camera: Camera,
    city: City,
    *,
    standoff_m: float = 45.0,
    altitude_m: float = 40.0,
    margin_m: float = 4.0,
    candidates: int = 24,
) -> Vec3 | None:
    """Find a vantage point with an actual view of `target`.

    Rings the target at `standoff_m` and keeps the first candidate that is
    collision-free *and* has clear line of sight. Returns None when the target
    is unobservable from that ring at that height -- a courtyard surrounded by
    towers genuinely has no vantage, and the honest answer is to say so rather
    than return a point that sees a wall.
    """
    if standoff_m <= 0 or altitude_m <= 0:
        raise ValueError("standoff and altitude must be positive")
    if candidates < 3:
        raise ValueError("need at least 3 candidate bearings")

    for i in range(candidates):
        bearing = 2 * math.pi * i / candidates
        eye = Vec3(
            target.position.north + standoff_m * math.cos(bearing),
            target.position.east + standoff_m * math.sin(bearing),
            -altitude_m,
        )
        if not city.is_free(eye, margin_m) or not city.in_bounds(eye):
            continue
        heading = math.degrees(math.atan2(
            target.position.east - eye.east, target.position.north - eye.north
        )) % 360.0
        if camera.sees(eye, heading, target.position, city):
            return eye
    return None
