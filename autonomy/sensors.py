"""Sensor simulation, including the cases where the sensor fails.

The central idea: a simulator that always detects the wire teaches the wrong
lesson. What matters operationally is not "can this sensor see a cable" but
"at what range, with what probability, and is that further than my stopping
distance". This module models the miss.

## Why thin obstacles are hard

Every ranging sensor samples the world at some angular resolution. A target of
width `w` at range `r` subtends roughly `w / r` radians. If that is smaller
than the beam spacing, the target falls *between* beams and returns nothing --
not because the sensor is bad, but because it never pointed at it.

For a HORIZONTAL wire the governing spacing is the VERTICAL one: the wire
spans hundreds of beams along a scan row, so what decides detection is whether
any beam row lands on it. Vertical spacing is typically the coarser of the two,
so quoting horizontal resolution flatters the sensor on exactly the targets
that matter most.

Worked example, a 12 mm distribution wire against a lidar with 0.4 deg
(7.0 mrad) vertical spacing:

    range 10 m -> subtense 1.2 mrad -> ~17% of scans intersect it
    range 20 m -> subtense 0.6 mrad -> ~8.6%
    range 40 m -> subtense 0.3 mrad -> ~4.3%

Now compare against stopping distance. A 1.5 kg quad at 8 m/s with a 0.6 s
velocity time constant needs about 17 m to stop including reaction latency,
and at 8 m/s the wire is typically declared around 25 m -- thin but positive
margin. At 12 m/s the margin inverts and the aircraft cannot stop in time.
Detection depends on *accumulating* returns while closing, which is why real
wire-strike systems need high scan rates as much as fine resolution.

## What is modelled

  * angular subtense vs. sensor resolution -> detection probability
  * range falloff and reflectivity (a bare conductor is a poor lidar target)
  * foliage as a partial occluder that also generates clutter returns
  * stereo depth failing on untextured thin objects regardless of range
  * per-scan randomness, seeded, so a run is reproducible

## What is not

No multipath, no sun glare, no rain, no rolling-shutter artefacts, no motion
blur. Those matter for a real integration and would each change the numbers.
Treat the ranges here as optimistic upper bounds, not a specification.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .thin_obstacles import Cable, Foliage, ThinObstacleField
from .types import Vec3
from .world import City

# Reflectivity relative to a diffuse white surface. Bare aluminium conductor is
# specular -- it mostly reflects the beam away from the receiver rather than
# back to it, which is why wires are worse lidar targets than their size alone
# would suggest.
REFLECTIVITY = {
    "cable": 0.25,
    "foliage": 0.45,
    "building": 0.75,
}


@dataclass(frozen=True, slots=True)
class LidarSpec:
    """A spinning or solid-state lidar.

    Defaults approximate a small automotive-grade unit of the sort that fits a
    2 kg airframe. High-end survey lidar reaches 0.05 deg and would roughly
    quadruple every detection range below.
    """

    range_m: float = 80.0
    h_fov_deg: float = 120.0
    v_fov_deg: float = 30.0
    h_resolution_deg: float = 0.2
    v_resolution_deg: float = 0.4
    scan_hz: float = 10.0
    range_noise_m: float = 0.03
    # Minimum returned energy that still counts as a detection.
    sensitivity: float = 0.06

    def __post_init__(self) -> None:
        if self.range_m <= 0 or self.scan_hz <= 0:
            raise ValueError("lidar range and scan rate must be positive")
        if self.h_resolution_deg <= 0 or self.v_resolution_deg <= 0:
            raise ValueError("angular resolution must be positive")

    @property
    def h_resolution_rad(self) -> float:
        return math.radians(self.h_resolution_deg)

    @property
    def v_resolution_rad(self) -> float:
        return math.radians(self.v_resolution_deg)

    def resolution_across(self, axis: Vec3) -> float:
        """Beam spacing measured ACROSS a long thin target's axis.

        For an extended object, what decides detection is whether any beam
        lands on its narrow dimension. A horizontal wire spans hundreds of
        beams along a scan row, so horizontal resolution is irrelevant -- the
        question is whether any beam *row* falls on it, which is governed by
        the vertical spacing. And vertical spacing is typically the coarser of
        the two, so using horizontal resolution flatters the sensor by roughly
        2x on exactly the targets that matter most.
        """
        horizontal = math.hypot(axis.north, axis.east)
        vertical = abs(axis.down)
        if horizontal >= vertical:
            return self.v_resolution_rad  # horizontal wire: rows decide
        return self.h_resolution_rad      # vertical wire: columns decide


@dataclass(frozen=True, slots=True)
class StereoSpec:
    """A stereo depth camera.

    Modelled mainly to show where it *fails*. Stereo matching needs texture;
    a smooth grey cable against a smooth grey sky produces no correspondence,
    so detection collapses regardless of pixel resolution. This is why depth
    cameras that map buildings beautifully still fly into wires.
    """

    range_m: float = 25.0
    h_fov_deg: float = 87.0
    pixels_across: int = 1280
    baseline_m: float = 0.05
    frame_hz: float = 30.0
    # Fraction of the time a low-texture linear target is matched at all.
    thin_target_match_rate: float = 0.12

    @property
    def angular_resolution_rad(self) -> float:
        return math.radians(self.h_fov_deg) / self.pixels_across


@dataclass(frozen=True, slots=True)
class Detection:
    """One sensor return."""

    kind: str  # "cable" | "foliage" | "building"
    name: str
    range_m: float
    bearing_deg: float
    elevation_deg: float
    confidence: float
    sensor: str

    @property
    def is_hard(self) -> bool:
        """Whether this return represents something that would stop the
        airframe. Foliage returns are real but mostly passable."""
        return self.kind != "foliage"


# Lidar range ratings are quoted against a standard diffuse target, almost
# always 10% reflectivity. The rating therefore ALREADY includes the inverse-
# square falloff, so applying 1/r^2 again on top of it double-counts and makes
# everything vanish at half the advertised range.
RANGE_SPEC_REFLECTIVITY = 0.10


def detection_probability(
    width_m: float,
    range_m: float,
    resolution_rad: float,
    reflectivity: float,
    max_range_m: float,
    sensitivity: float = 0.0,
) -> float:
    """Chance that one scan produces a return from a target of `width_m`.

    Two effects multiply:

    `beam coverage` -- the target subtends `width / range` radians. If that is
    below the beam spacing, only that fraction of scans happen to intersect it.
    Pure geometry, and for wires it is overwhelmingly the dominant term.

    `energy margin` -- how far inside its rated range the target is, corrected
    for reflectivity. A brighter-than-standard target is detectable beyond the
    nominal range; a dark one falls short. This term saturates at 1.0, because
    a strong return is not more detected than an adequate one.

    `sensitivity` is accepted for interface stability but no longer used: the
    energy limit is now expressed through the range rating, which is how
    datasheets actually specify it.
    """
    if range_m <= 0.01:
        return 0.0

    effective_range = max_range_m * math.sqrt(reflectivity / RANGE_SPEC_REFLECTIVITY)
    if range_m > effective_range:
        return 0.0

    subtense = width_m / range_m
    beam_coverage = min(1.0, subtense / resolution_rad)

    # Soft rolloff over the last 15% of usable range, where returns get marginal.
    margin = (effective_range - range_m) / (0.15 * effective_range)
    return beam_coverage * min(1.0, max(0.0, margin))


@dataclass
class SensorSuite:
    """Lidar plus stereo, scanning a world of buildings, wires, and foliage."""

    lidar: LidarSpec = field(default_factory=LidarSpec)
    stereo: StereoSpec | None = field(default_factory=StereoSpec)
    city: City | None = None
    thin: ThinObstacleField = field(default_factory=ThinObstacleField)
    seed: int = 17

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def scan(self, eye: Vec3, heading_deg: float) -> list[Detection]:
        """One sensor frame. Stochastic -- call it every tick and accumulate."""
        out: list[Detection] = []
        for cable in self.thin.cables:
            hit = self._scan_cable(eye, heading_deg, cable)
            if hit:
                out.append(hit)
        for tree in self.thin.foliage:
            hit = self._scan_foliage(eye, heading_deg, tree)
            if hit:
                out.append(hit)
        if self.city:
            out.extend(self._scan_buildings(eye, heading_deg))
        return out

    # -- per-target models -------------------------------------------------

    def _scan_cable(self, eye: Vec3, heading_deg: float, cable: Cable) -> Detection | None:
        nearest, distance = self._nearest_point(eye, cable.points())
        if not self._in_fov(eye, heading_deg, nearest):
            return None

        axis = cable.end - cable.start
        p_lidar = detection_probability(
            cable.diameter_m, distance,
            self.lidar.resolution_across(axis), REFLECTIVITY["cable"],
            self.lidar.range_m,
        )
        # Foliage between us and the wire attenuates the return.
        p_lidar *= self.thin.sightline_transmission(eye, nearest)

        sensor = "lidar"
        p = p_lidar
        if self.stereo and distance <= self.stereo.range_m:
            # Stereo rarely matches a thin untextured target, but when it does
            # it is an independent chance, so the suite beats either alone.
            p_stereo = self.stereo.thin_target_match_rate
            combined = 1.0 - (1.0 - p_lidar) * (1.0 - p_stereo)
            if combined > p_lidar:
                sensor = "lidar+stereo"
            p = combined

        if self._rng.random() > p:
            return None

        return Detection(
            kind="cable", name=cable.name,
            range_m=distance + self._rng.gauss(0, self.lidar.range_noise_m),
            bearing_deg=self._bearing(eye, nearest),
            elevation_deg=self._elevation(eye, nearest),
            confidence=round(p, 3), sensor=sensor,
        )

    def _scan_foliage(self, eye: Vec3, heading_deg: float, tree: Foliage) -> Detection | None:
        offset = tree.centre - eye
        distance = offset.norm
        if not self._in_fov(eye, heading_deg, tree.centre):
            return None
        # A canopy is a big target: detection is easy. Interpreting it is not.
        p = detection_probability(
            tree.radius_m * 2, distance,
            self.lidar.h_resolution_rad, REFLECTIVITY["foliage"],
            self.lidar.range_m,
        )
        if self._rng.random() > p:
            return None
        return Detection(
            kind="foliage", name=tree.name,
            range_m=distance - tree.radius_m,
            bearing_deg=self._bearing(eye, tree.centre),
            elevation_deg=self._elevation(eye, tree.centre),
            confidence=round(p, 3), sensor="lidar",
        )

    def _scan_buildings(self, eye: Vec3, heading_deg: float) -> list[Detection]:
        out = []
        for b in self.city.buildings:
            face = Vec3(b.north, b.east, -min(b.height / 2, -eye.down))
            distance = (face - eye).norm
            if distance > self.lidar.range_m or not self._in_fov(eye, heading_deg, face):
                continue
            if not self.city.has_line_of_sight(eye, face):
                continue
            out.append(
                Detection(
                    kind="building", name=b.name, range_m=distance,
                    bearing_deg=self._bearing(eye, face),
                    elevation_deg=self._elevation(eye, face),
                    confidence=1.0, sensor="lidar",
                )
            )
        return out

    # -- geometry helpers --------------------------------------------------

    @staticmethod
    def _nearest_point(eye: Vec3, points: list[Vec3]) -> tuple[Vec3, float]:
        best, best_d = points[0], math.inf
        for p in points:
            d = (p - eye).norm
            if d < best_d:
                best, best_d = p, d
        return best, best_d

    @staticmethod
    def _bearing(eye: Vec3, target: Vec3) -> float:
        o = target - eye
        return math.degrees(math.atan2(o.east, o.north)) % 360.0

    @staticmethod
    def _elevation(eye: Vec3, target: Vec3) -> float:
        o = target - eye
        return math.degrees(math.atan2(-o.down, math.hypot(o.north, o.east)))

    def _in_fov(self, eye: Vec3, heading_deg: float, target: Vec3) -> bool:
        bearing = self._bearing(eye, target)
        off_axis = abs((bearing - heading_deg + 180.0) % 360.0 - 180.0)
        if off_axis > self.lidar.h_fov_deg / 2:
            return False
        return abs(self._elevation(eye, target)) <= self.lidar.v_fov_deg / 2


def stopping_distance_m(speed_mps: float, accel_tau_s: float, reaction_s: float = 0.3) -> float:
    """How far the vehicle travels between deciding to stop and stopping.

    With first-order velocity dynamics the asymptotic stop distance is
    `v * tau`; practically the vehicle is at 5% of speed after 3 tau, so
    `3 * v * tau` is the honest figure, plus whatever it travels during the
    detection-to-command latency.
    """
    if speed_mps < 0 or accel_tau_s <= 0:
        raise ValueError("invalid vehicle dynamics")
    return speed_mps * reaction_s + 3.0 * speed_mps * accel_tau_s


def wire_detection_range(
    cable: Cable,
    suite: SensorSuite,
    *,
    confidence: float = 0.9,
    scan_hz: float | None = None,
    closing_speed_mps: float = 8.0,
    max_range_m: float = 80.0,
    step_m: float = 0.5,
    required_returns: int = 3,
) -> float:
    """Range at which a wire is *declared*, with `confidence`, while closing.

    Integrates per-scan probability over the approach: at each range the
    vehicle gets `scan_hz * step / speed` looks, and returns accumulate.

    `required_returns` is why this is not simply "probability of one hit". A
    single isolated return is indistinguishable from a bird, a dust mote, or
    sensor noise; a system that stops on one would be unusable. Real wire-
    detection logic demands several consistent returns before declaring an
    obstacle, and that requirement costs range -- which is the whole reason
    scan rate matters as much as angular resolution.

    Arrival counts are modelled as Poisson, which is the right limit for many
    independent low-probability looks.

    Compare the result against `stopping_distance_m` to know whether the
    sensor buys you anything at a given speed.
    """
    rate = scan_hz if scan_hz is not None else suite.lidar.scan_hz
    if closing_speed_mps <= 0:
        raise ValueError("closing speed must be positive")
    if required_returns < 1:
        raise ValueError("need at least one return to declare a target")

    resolution = suite.lidar.resolution_across(cable.end - cable.start)
    expected = 0.0  # Poisson lambda: cumulative expected returns so far
    r = max_range_m
    while r > 0.5:
        p_single = detection_probability(
            cable.diameter_m, r, resolution,
            REFLECTIVITY["cable"], suite.lidar.range_m,
        )
        expected += p_single * (rate * step_m / closing_speed_mps)
        if _poisson_at_least(required_returns, expected) >= confidence:
            return r
        r -= step_m
    return 0.0


def _poisson_at_least(k: int, lam: float) -> float:
    """P(N >= k) for N ~ Poisson(lam)."""
    if lam <= 0:
        return 0.0
    term = math.exp(-lam)
    cumulative = term
    for i in range(1, k):
        term *= lam / i
        cumulative += term
    return max(0.0, 1.0 - cumulative)
