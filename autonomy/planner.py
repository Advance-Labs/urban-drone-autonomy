"""Path planning: turning a mission intent into an ordered list of waypoints.

Kept pure -- these are functions from parameters to lists of `Vec3`, with no
vehicle, no async, and no I/O. That makes the geometry trivially testable and
means a planner bug can never be confused with a control bug.
"""

from __future__ import annotations

import math

from .types import Vec3


def survey_grid(
    *,
    width_m: float,
    height_m: float,
    spacing_m: float,
    altitude_m: float,
    origin: Vec3 | None = None,
) -> list[Vec3]:
    """Boustrophedon ("lawnmower") coverage pattern over a rectangle.

    This is the workhorse pattern for mapping, search, and any camera task
    that needs uniform coverage: fly a leg, shift sideways by the sensor
    swath, fly back the other way.

    `spacing_m` should be the ground footprint of your sensor minus the
    overlap you want. For photogrammetry, 70-80% overlap is typical, so a
    camera seeing 20 m across would use a spacing near 5 m.

    Args:
        width_m: extent along east, metres.
        height_m: extent along north, metres.
        spacing_m: distance between adjacent legs, metres.
        altitude_m: survey height above launch, metres.
        origin: south-west corner. Defaults to the launch point.

    Returns:
        Waypoints in flight order.
    """
    if width_m <= 0 or height_m <= 0:
        raise ValueError("survey area must have positive extent")
    if spacing_m <= 0:
        raise ValueError("spacing must be positive")
    if altitude_m <= 0:
        raise ValueError("survey altitude must be positive")

    base = origin or Vec3()
    down = -altitude_m
    legs = max(1, math.ceil(width_m / spacing_m) + 1)

    waypoints: list[Vec3] = []
    for i in range(legs):
        east = base.east + min(i * spacing_m, width_m)
        # Alternate direction each leg -- that is what makes it a lawnmower
        # rather than a comb with wasted return transits.
        north_pair = (
            (base.north, base.north + height_m)
            if i % 2 == 0
            else (base.north + height_m, base.north)
        )
        waypoints.append(Vec3(north_pair[0], east, down))
        waypoints.append(Vec3(north_pair[1], east, down))

    return waypoints


def orbit(
    *,
    centre: Vec3,
    radius_m: float,
    altitude_m: float,
    points: int = 16,
) -> list[Vec3]:
    """Circular loiter around a point of interest.

    Used for inspection and for keeping a gimballed camera pointed at one
    target while the vehicle moves -- the standard "look at that thing from
    every angle" primitive.
    """
    if radius_m <= 0:
        raise ValueError("orbit radius must be positive")
    if points < 3:
        raise ValueError("orbit needs at least 3 points")
    if altitude_m <= 0:
        raise ValueError("orbit altitude must be positive")

    return [
        Vec3(
            centre.north + radius_m * math.cos(2 * math.pi * i / points),
            centre.east + radius_m * math.sin(2 * math.pi * i / points),
            -altitude_m,
        )
        for i in range(points)
    ]


def path_length(waypoints: list[Vec3]) -> float:
    """Total 3D path distance, metres. Useful for feasibility checks."""
    return sum(
        (waypoints[i + 1] - waypoints[i]).norm for i in range(len(waypoints) - 1)
    )


def estimated_duration_s(waypoints: list[Vec3], cruise_speed_mps: float) -> float:
    """Rough flight time for a path, ignoring accel and turn losses.

    Optimistic by design -- treat it as a lower bound when deciding whether a
    mission fits in one battery.
    """
    if cruise_speed_mps <= 0:
        raise ValueError("cruise speed must be positive")
    return path_length(waypoints) / cruise_speed_mps
