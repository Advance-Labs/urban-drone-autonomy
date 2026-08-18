"""People and vehicles: the hazards that are not really obstacles.

## Why these need their own module

A building is a collision problem: stay out of the volume it occupies. A
pedestrian is not. A person is 0.4 m across, so as a collision hazard they are
almost irrelevant — the aircraft could fly within a metre and never touch them.
What matters is that under Transport Canada rules a drone must keep a defined
horizontal distance from bystanders, and flying *over* an uninvolved person is
its own violation regardless of height.

So the keep-out volume around a person is roughly a hundred times their
physical size, and it is asymmetric: passing 30 m to one side is fine, passing
directly overhead at 40 m is not. Modelling people with the same sphere used
for buildings gets both facts wrong.

Vehicles sit in between. A car is a genuine collision hazard at low level, it
moves fast and predictably along a road, and it usually contains people — which
matters because the low-level gap under a powerline is exactly where a drone
squeezed between conductor and road surface ends up.

## Hard limits versus soft cost

Both are modelled. `blocked_at` is the hard floor: physical collision, and the
minimum legal separation that must never be crossed. `risk_cost` is a soft
gradient on top, so that among legal routes the planner prefers the one that
gives people more room. A purely binary constraint produces a path that hugs
the legal minimum exactly, which is technically compliant and obviously wrong.

## Regulatory numbers

Defaults follow Canadian **basic** operations: 30 m horizontal from bystanders,
no overflight. Advanced certification reduces the horizontal minimum
substantially and permits some overflight; `RegulatoryProfile.advanced()`
carries those figures. VERIFY against current CARs Part IX before flying —
these are modelling defaults, not legal advice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .dynamic import MovingObstacle
from .types import Vec3


@dataclass(frozen=True, slots=True)
class RegulatoryProfile:
    """Separation minima the flight must respect."""

    name: str
    # Horizontal distance that must be kept from an uninvolved person.
    person_radius_m: float
    # Whether the aircraft may pass directly above a person at all.
    overflight_allowed: bool
    # Soft-cost radius: beyond the hard minimum, staying further is preferred.
    comfort_radius_m: float

    @staticmethod
    def basic() -> RegulatoryProfile:
        return RegulatoryProfile("basic", 30.0, False, 60.0)

    @staticmethod
    def advanced() -> RegulatoryProfile:
        # Advanced operations permit much closer work, including some
        # overflight, subject to aircraft declaration and pilot certification.
        return RegulatoryProfile("advanced", 5.0, True, 25.0)


@dataclass(slots=True)
class Pedestrian:
    """A person moving along a path, carrying a legal exclusion volume."""

    walk: MovingObstacle
    name: str = "person"
    # Physical size, used only for the collision floor. The separation that
    # actually governs planning comes from the RegulatoryProfile.
    body_radius_m: float = 0.45
    height_m: float = 1.8

    def position_at(self, t: float) -> Vec3:
        return self.walk.position_at(t)

    def separation(self, p: Vec3, t: float) -> float:
        """Horizontal distance from `p` to the person. Horizontal, because
        that is how the rule is written and how the risk behaves -- height
        above someone does not make hitting them less likely."""
        return p.horizontal_distance_to(self.position_at(t))

    def is_overhead(self, p: Vec3, t: float, footprint_m: float = 3.0) -> bool:
        """Directly above, within a small footprint.

        A separate question from separation: a failure over someone lands on
        them whatever the altitude, which is why overflight is regulated on
        its own terms rather than folded into a distance.
        """
        return self.separation(p, t) <= footprint_m and p.altitude > self.height_m


@dataclass(slots=True)
class Vehicle:
    """A car moving along a road. A real collision hazard at low level."""

    drive: MovingObstacle
    name: str = "car"
    length_m: float = 4.6
    width_m: float = 1.9
    height_m: float = 1.5
    # Occupants make a vehicle a person-carrying object, so overflight of a
    # car inherits the same concern as overflight of a pedestrian.
    carries_people: bool = True

    def position_at(self, t: float) -> Vec3:
        return self.drive.position_at(t)

    def contains(self, p: Vec3, t: float, margin: float = 0.0) -> bool:
        """Collision test against the vehicle body.

        Treated as an upright cylinder of the car's half-length: orientation
        along the road is known but not worth tracking, and a circumscribing
        radius is the honest conservative choice.
        """
        centre = self.position_at(t)
        radius = self.length_m / 2 + margin
        horizontal = p.horizontal_distance_to(centre)
        return horizontal <= radius and 0.0 <= p.altitude <= self.height_m + margin


@dataclass(slots=True)
class HazardField:
    """All people and vehicles, queried at a point in space and time."""

    pedestrians: list[Pedestrian] = field(default_factory=list)
    vehicles: list[Vehicle] = field(default_factory=list)
    rules: RegulatoryProfile = field(default_factory=RegulatoryProfile.basic)

    @property
    def is_empty(self) -> bool:
        return not self.pedestrians and not self.vehicles

    def blocked_at(self, p: Vec3, t: float, margin: float = 0.0) -> str | None:
        """Hard constraint: collision, legal minimum, or prohibited overflight."""
        for v in self.vehicles:
            if v.contains(p, t, margin):
                return f"{v.name} (collision)"

        # `margin` MUST apply to people too. It carries the caller's tracking
        # error, and the whole point of that allowance is that a plan legal by
        # exactly the minimum gets flown in violation of it. Ignoring margin
        # here -- as this did originally -- silently made the most
        # safety-critical constraint the only one without a safety factor.
        keep_out = self.rules.person_radius_m + margin
        for person in self.pedestrians:
            if person.separation(p, t) < keep_out:
                return f"{person.name} ({keep_out:.0f} m separation)"
            if not self.rules.overflight_allowed and person.is_overhead(
                p, t, footprint_m=3.0 + margin
            ):
                return f"{person.name} (overflight)"

        if not self.rules.overflight_allowed:
            for v in self.vehicles:
                if v.carries_people and p.horizontal_distance_to(v.position_at(t)) < 3.0:
                    if p.altitude > v.height_m:
                        return f"{v.name} (overflight of occupants)"
        return None

    def risk_cost(self, p: Vec3, t: float) -> float:
        """Soft penalty, 0 upward, for being near people.

        Added to the planner's move cost so that among *legal* routes the one
        giving people more room wins. Without it the planner produces a path
        that hugs the legal minimum exactly, which is compliant and obviously
        wrong -- nobody flies a survey deliberately 30.1 m from a pedestrian.
        """
        cost = 0.0
        for person in self.pedestrians:
            d = person.separation(p, t)
            if d < self.rules.comfort_radius_m:
                # Linear ramp from the comfort radius inward. Quadratic would
                # concentrate all the pressure at the last few metres, which
                # is too late to influence a route.
                cost += 1.0 - (d / self.rules.comfort_radius_m)
        for v in self.vehicles:
            d = p.horizontal_distance_to(v.position_at(t))
            if d < 15.0 and p.altitude < 12.0:
                # Low and close to a road: the squeeze under a conductor.
                cost += 0.5 * (1.0 - d / 15.0)
        return cost

    def to_dict(self, t: float) -> dict:
        """Positions at time `t`, for the viewer."""
        return {
            "rules": self.rules.name,
            "person_radius_m": self.rules.person_radius_m,
            "pedestrians": [
                {
                    "name": p.name,
                    "n": round(p.position_at(t).north, 2),
                    "e": round(p.position_at(t).east, 2),
                    "waypoints": [
                        {"n": w.north, "e": w.east, "d": w.down} for w in p.walk.waypoints
                    ],
                }
                for p in self.pedestrians
            ],
            "vehicles": [
                {
                    "name": v.name,
                    "n": round(v.position_at(t).north, 2),
                    "e": round(v.position_at(t).east, 2),
                    "d": round(v.position_at(t).down, 2),
                    "length": v.length_m,
                    "waypoints": [
                        {"n": w.north, "e": w.east, "d": w.down} for w in v.drive.waypoints
                    ],
                }
                for v in self.vehicles
            ],
        }


def pedestrian_on_sidewalk(
    *,
    north: float,
    east_start: float,
    east_end: float,
    speed_mps: float = 1.4,
    name: str = "pedestrian",
) -> Pedestrian:
    """Someone walking a straight sidewalk at a normal pace (~1.4 m/s)."""
    return Pedestrian(
        walk=MovingObstacle(
            waypoints=[Vec3(north, east_start, 0.0), Vec3(north, east_end, 0.0)],
            speed_mps=speed_mps, radius_m=0.45, name=name, loop=False,
        ),
        name=name,
    )


def car_on_road(
    *,
    north: float,
    east_start: float,
    east_end: float,
    speed_mps: float = 11.0,
    name: str = "car",
) -> Vehicle:
    """A car on a residential street (~40 km/h)."""
    return Vehicle(
        drive=MovingObstacle(
            waypoints=[Vec3(north, east_start, -0.75), Vec3(north, east_end, -0.75)],
            speed_mps=speed_mps, radius_m=2.3, name=name, loop=False,
        ),
        name=name,
    )
