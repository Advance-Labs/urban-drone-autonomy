"""Thin and porous obstacles: the things that actually bring small drones down.

Buildings are the easy case. A tower is metres across, reflects everything, and
any sensor sees it from hundreds of metres. Wires are the opposite: a 10 mm
conductor is smaller than the angular resolution of most lidar beyond about
30 m, has almost no radar cross-section, and presents no texture for stereo
matching. Wire strike is a leading cause of small-UAS loss, and the reason is
not that operators ignore wires -- it is that the sensors genuinely cannot
resolve them in time.

Foliage is a third category and behaves differently again: it produces sensor
returns but is not a hard obstacle. A branch tip stops nothing; a trunk stops
everything. Treating every return as solid makes a drone that refuses to fly
near trees; treating none as solid makes one that flies into a trunk.

This module models all three honestly, including the cases where detection
fails. A simulator that always sees the wire teaches exactly the wrong lesson.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .spatial import SpatialHash
from .types import Vec3


def _closest_point_on_segment(p: Vec3, a: Vec3, b: Vec3) -> Vec3:
    """Nearest point to `p` on the segment ab. Used for cable distance."""
    ab = b - a
    length_sq = ab.north**2 + ab.east**2 + ab.down**2
    if length_sq < 1e-12:
        return a
    ap = p - a
    t = (ap.north * ab.north + ap.east * ab.east + ap.down * ab.down) / length_sq
    t = max(0.0, min(1.0, t))
    return a + ab * t


@dataclass(frozen=True, slots=True)
class Cable:
    """A catenary-sagged conductor strung between two supports.

    Diameter defaults to 12 mm, typical of a residential distribution line.
    Transmission conductors run 25-40 mm; guy wires and comms drops can be
    5 mm or less, and those are effectively invisible to everything.
    """

    start: Vec3
    end: Vec3
    diameter_m: float = 0.012
    sag_m: float = 2.0
    name: str = "cable"
    segments: int = 8

    def __post_init__(self) -> None:
        if self.diameter_m <= 0:
            raise ValueError(f"{self.name}: diameter must be positive")
        if self.sag_m < 0:
            raise ValueError(f"{self.name}: sag cannot be negative")
        if self.segments < 1:
            raise ValueError(f"{self.name}: need at least one segment")

    def points(self) -> list[Vec3]:
        """Polyline approximation of the catenary.

        A parabola rather than a true cosh: over a single span the difference
        is centimetres, and the sag matters here only because it puts the
        lowest point of the wire well below both supports -- which is exactly
        where a drone crossing a street at "safe" pole height meets it.
        """
        out = []
        for i in range(self.segments + 1):
            t = i / self.segments
            p = self.start + (self.end - self.start) * t
            droop = self.sag_m * 4 * t * (1 - t)  # 0 at ends, max at midspan
            out.append(Vec3(p.north, p.east, p.down + droop))
        return out

    def distance_to(self, p: Vec3) -> float:
        """Shortest distance from `p` to the wire surface."""
        pts = self.points()
        best = math.inf
        for a, b in zip(pts, pts[1:]):
            best = min(best, (p - _closest_point_on_segment(p, a, b)).norm)
        return max(0.0, best - self.diameter_m / 2)

    def contains(self, p: Vec3, margin: float = 0.0) -> bool:
        return self.distance_to(p) <= margin


@dataclass(frozen=True, slots=True)
class Foliage:
    """A porous volume: a tree canopy or hedge.

    `density` is the fraction of the volume that actually stops an airframe.
    Outer canopy is perhaps 0.15 -- leaves and twigs that a drone punches
    through with cosmetic damage. A trunk is 1.0. Modelling this as a single
    hard sphere is why naive obstacle maps make trees look like buildings.
    """

    centre: Vec3
    radius_m: float
    density: float = 0.2
    trunk_radius_m: float = 0.25
    name: str = "tree"

    def __post_init__(self) -> None:
        if self.radius_m <= 0:
            raise ValueError(f"{self.name}: radius must be positive")
        if not 0.0 <= self.density <= 1.0:
            raise ValueError(f"{self.name}: density must be within 0..1")

    def contains(self, p: Vec3, margin: float = 0.0) -> bool:
        """Solid only at the trunk. The canopy returns sensor energy but does
        not stop the airframe, so it must not be treated as occupied."""
        horizontal = math.hypot(p.north - self.centre.north, p.east - self.centre.east)
        below_canopy = p.altitude <= self.centre.altitude + self.radius_m
        return below_canopy and horizontal <= self.trunk_radius_m + margin

    def canopy_contains(self, p: Vec3) -> bool:
        """Inside the leafy volume: produces returns, degrades visibility."""
        return (p - self.centre).norm <= self.radius_m

    def occlusion_factor(self) -> float:
        """How much of a sightline through the canopy survives.

        Leaves scatter; a target behind a canopy is partly visible and mostly
        not. Used to degrade line-of-sight rather than block it outright.
        """
        return 1.0 - self.density


@dataclass(frozen=True)
class ThinObstacleField:
    """The collection of wires and foliage layered over a City."""

    cables: tuple[Cable, ...] = ()
    foliage: tuple[Foliage, ...] = ()

    def _index(self) -> SpatialHash:
        """Spatial index over cables and foliage, built once per field.

        Distance to a sagging cable means eight segment-distance computations,
        so scanning 448 of them measured at 8.5 ms per query -- two hours of
        collision tests for a single search. Bucketing makes it constant time.

        Cached on the instance despite `frozen=True`, via object.__setattr__:
        the field is logically immutable and the index is derived state.
        """
        cached = getattr(self, "_hash_cache", None)
        if cached is not None:
            return cached
        index: SpatialHash = SpatialHash(cell_m=25.0)
        for c in self.cables:
            pts = c.points()
            ns = [q.north for q in pts]
            es = [q.east for q in pts]
            index.insert(c, min(ns), max(ns), min(es), max(es))
        for f in self.foliage:
            index.insert(
                f, f.centre.north - f.radius_m, f.centre.north + f.radius_m,
                f.centre.east - f.radius_m, f.centre.east + f.radius_m,
            )
        object.__setattr__(self, "_hash_cache", index)
        return index

    def solid_at(self, p: Vec3, margin: float = 0.0) -> str | None:
        """Name of whatever would physically stop the airframe at `p`."""
        for obj in self._index().query(p.north, p.east, margin):
            if obj.contains(p, margin):
                return obj.name
        return None

    def segment_clear(self, a: Vec3, b: Vec3, margin: float = 0.0, step_m: float = 0.4) -> bool:
        """Sampled sweep. The step must be far finer than for buildings --
        a 12 mm wire sampled every 1.5 m is stepped over almost every time."""
        span = (b - a).norm
        steps = max(2, math.ceil(span / step_m))
        for i in range(steps + 1):
            if self.solid_at(a + (b - a) * (i / steps), margin) is not None:
                return False
        return True

    def sightline_transmission(self, eye: Vec3, target: Vec3, step_m: float = 1.0) -> float:
        """Fraction of a sightline's energy surviving foliage, 0..1."""
        span = (target - eye).norm
        if span < 1e-9:
            return 1.0
        steps = max(2, math.ceil(span / step_m))
        transmission = 1.0
        index = self._index()
        for i in range(1, steps):
            p = eye + (target - eye) * (i / steps)
            for f in index.query(p.north, p.east):
                if isinstance(f, Foliage) and f.canopy_contains(p):
                    # Compounding per sample: a long path through canopy
                    # attenuates far more than a clipped corner.
                    transmission *= 1.0 - f.density * (step_m / max(f.radius_m, 0.1))
        return max(0.0, transmission)


def powerline_corridor(
    *,
    north: float,
    east_start: float,
    east_end: float,
    height_m: float = 9.0,
    spans: int = 3,
    diameter_m: float = 0.012,
    sag_m: float = 1.8,
) -> list[Cable]:
    """A run of distribution wire on poles, at typical residential height.

    9 m is the usual clearance for a street crossing -- squarely inside the
    altitude band a small drone uses for anything low-level, which is why
    these are the wires that get hit.
    """
    if spans < 1:
        raise ValueError("need at least one span")
    cables = []
    for i in range(spans):
        t0 = i / spans
        t1 = (i + 1) / spans
        cables.append(
            Cable(
                start=Vec3(north, east_start + (east_end - east_start) * t0, -height_m),
                end=Vec3(north, east_start + (east_end - east_start) * t1, -height_m),
                diameter_m=diameter_m,
                sag_m=sag_m,
                name=f"line-{i}",
            )
        )
    return cables
