"""Moving obstacles and the replanning they force.

A static planner is a solved problem: plan once, fly the path. Movement breaks
that assumption in a specific way -- the path you are flying was valid when you
planned it and may not be now. Handling this well is mostly about *when* to
replan, not how.

Three strategies, in increasing order of how much they cost:

  every tick     always correct, wildly wasteful. At 10 Hz over a 30 s leg that
                 is 300 A* searches to produce the same path 297 times.
  fixed interval cheap and predictable, but blind between ticks. A fast mover
                 can cross your path inside the interval.
  on invalidation replan only when the current path is actually blocked, plus a
                 slow heartbeat as a backstop. This is what is implemented.

The invalidation check is far cheaper than a replan -- it is a sweep along the
remaining path rather than a graph search -- so it can run every tick while
replans stay rare.

Prediction: obstacles are extrapolated forward at constant velocity over the
planning horizon. That is crude but correct for the dominant case (another
aircraft transiting) and fails safe for the rest, because an obstacle that
turns is one the invalidation check will catch on a later tick.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .types import Vec3


@dataclass(slots=True)
class MovingObstacle:
    """A sphere following a waypoint loop at constant speed.

    A sphere rather than a box because a moving object's orientation is rarely
    known well enough to justify anything tighter, and an inflated sphere is
    the honest conservative hull.
    """

    waypoints: list[Vec3]
    speed_mps: float
    radius_m: float = 4.0
    name: str = "mover"
    loop: bool = True
    _distance: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        if len(self.waypoints) < 2:
            raise ValueError(f"{self.name}: need at least two waypoints")
        if self.speed_mps <= 0:
            raise ValueError(f"{self.name}: speed must be positive")
        if self.radius_m <= 0:
            raise ValueError(f"{self.name}: radius must be positive")

    @property
    def leg_lengths(self) -> list[float]:
        pts = self.waypoints + ([self.waypoints[0]] if self.loop else [])
        return [(pts[i + 1] - pts[i]).norm for i in range(len(pts) - 1)]

    @property
    def circuit_length(self) -> float:
        return sum(self.leg_lengths)

    def position_at(self, t: float) -> Vec3:
        """Where it is at absolute mission time `t`.

        Pure function of time, deliberately: prediction and collision checking
        both need to ask about the future, and a stateful `advance()` could
        only ever answer about now.
        """
        total = self.circuit_length
        if total < 1e-9:
            return self.waypoints[0]

        travelled = self.speed_mps * max(0.0, t)
        if self.loop:
            travelled %= total
        else:
            # Ping-pong: reverse at the end rather than teleporting home.
            cycle = travelled % (2 * total)
            travelled = cycle if cycle <= total else 2 * total - cycle

        pts = self.waypoints + ([self.waypoints[0]] if self.loop else [])
        for i, leg in enumerate(self.leg_lengths):
            if travelled <= leg or i == len(self.leg_lengths) - 1:
                frac = travelled / leg if leg > 1e-9 else 0.0
                return pts[i] + (pts[i + 1] - pts[i]) * min(1.0, frac)
            travelled -= leg
        return pts[-1]

    def velocity_at(self, t: float, dt: float = 0.1) -> Vec3:
        """Finite-difference velocity, for prediction and display."""
        return (self.position_at(t + dt) - self.position_at(t)) * (1.0 / dt)

    def contains(self, p: Vec3, t: float, margin: float = 0.0) -> bool:
        return (p - self.position_at(t)).norm <= self.radius_m + margin


@dataclass(slots=True)
class DynamicField:
    """All moving obstacles, queried at a point in time."""

    obstacles: list[MovingObstacle] = field(default_factory=list)

    def blocker_at(self, p: Vec3, t: float, margin: float = 0.0) -> str | None:
        for o in self.obstacles:
            if o.contains(p, t, margin):
                return o.name
        return None

    def segment_blocker(
        self,
        a: Vec3,
        b: Vec3,
        t_start: float,
        speed_mps: float,
        margin: float = 0.0,
        step_m: float = 2.0,
    ) -> str | None:
        """Space-time sweep. Returns the blocking obstacle's name, or None.

        Checking against current positions is the classic mistake -- it clears
        a path that a mover will occupy by the time you arrive. Each sample is
        tested at the time the vehicle would actually reach it.

        Returns the name rather than a bare bool because the blockage is
        usually mid-segment, so a caller cannot recover which obstacle it was
        by re-testing the endpoints -- and "replanning around something" is
        not a useful log line when three obstacles are in play.
        """
        span = (b - a).norm
        if span < 1e-9:
            return self.blocker_at(a, t_start, margin)
        steps = max(2, math.ceil(span / step_m))
        for i in range(steps + 1):
            frac = i / steps
            p = a + (b - a) * frac
            arrival = t_start + (span * frac) / max(speed_mps, 0.1)
            hit = self.blocker_at(p, arrival, margin)
            if hit is not None:
                return hit
        return None

    def segment_clear(
        self,
        a: Vec3,
        b: Vec3,
        t_start: float,
        speed_mps: float,
        margin: float = 0.0,
        step_m: float = 2.0,
    ) -> bool:
        return self.segment_blocker(a, b, t_start, speed_mps, margin, step_m) is None

    def path_clear(
        self,
        waypoints: list[Vec3],
        start_index: int,
        position: Vec3,
        t_now: float,
        speed_mps: float,
        margin: float = 0.0,
    ) -> str | None:
        """Check the *remaining* path. Returns the blocking obstacle, or None.

        Cheap enough to run every control tick, which is what lets replans stay
        rare without leaving the vehicle blind between them.
        """
        legs = [position] + waypoints[start_index:]
        elapsed = t_now
        for a, b in zip(legs, legs[1:]):
            hit = self.segment_blocker(a, b, elapsed, speed_mps, margin)
            if hit is not None:
                return hit
            elapsed += (b - a).norm / max(speed_mps, 0.1)
        return None

    def to_dict(self, t: float) -> list[dict]:
        """Positions at time `t`, for the viewer."""
        return [
            {
                "name": o.name,
                "n": round(o.position_at(t).north, 2),
                "e": round(o.position_at(t).east, 2),
                "d": round(o.position_at(t).down, 2),
                "r": o.radius_m,
            }
            for o in self.obstacles
        ]


def crossing_traffic(
    *,
    north: float = 0.0,
    east_span: float = 160.0,
    altitude_m: float = 25.0,
    speed_mps: float = 9.0,
    radius_m: float = 5.0,
    name: str = "traffic",
) -> MovingObstacle:
    """Another aircraft transiting across the working area, back and forth."""
    return MovingObstacle(
        waypoints=[
            Vec3(north, -east_span / 2, -altitude_m),
            Vec3(north, east_span / 2, -altitude_m),
        ],
        speed_mps=speed_mps,
        radius_m=radius_m,
        name=name,
        loop=False,  # ping-pong, so it does not teleport back across
    )
