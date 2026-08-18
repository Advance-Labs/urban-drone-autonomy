"""Emergency landing site selection.

Invoked when the battery can no longer get the vehicle home. At that point
"return to launch" is off the table and the question becomes: of the places we
could still reach, which is the least bad?

Scoring balances four things that genuinely trade against each other:

  reachability  can we actually get there on the charge remaining? A perfect
                site we cannot reach scores zero. This is a hard gate, not a
                weight -- an unreachable "best" site is how you crash.
  quality       operator-assigned. An open field is not a parking lot is not
                a road. Supplied with the site, because the vehicle has no way
                to know this on its own without terrain classification.
  proximity     every metre flown is charge spent and risk taken.
  wind          approach direction matters. Landing into wind lowers ground
                speed and shortens the touchdown, which is why crewed aircraft
                do it and why it is worth points here.

The honest limitation: sites are supplied by the operator ahead of the flight.
This does not find landing sites from imagery -- that is a perception problem
needing a downward camera and a terrain classifier. What it does is choose
correctly among known options under real constraints, which is the part that
has to be right when the battery is nearly flat.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .types import Telemetry, Vec3


@dataclass(frozen=True, slots=True)
class LandingSite:
    """A pre-surveyed place the vehicle may put down."""

    position: Vec3
    name: str = "site"
    radius_m: float = 5.0
    # 0.0 = last resort, 1.0 = ideal. Operator judgement: an open grass field
    # is ~1.0, a quiet car park ~0.6, a road shoulder ~0.2.
    quality: float = 1.0
    # Preferred approach bearing in degrees, if the site is directional
    # (a clearing with trees on three sides). None means any direction.
    preferred_approach_deg: float | None = None

    def __post_init__(self) -> None:
        if self.radius_m <= 0:
            raise ValueError(f"site {self.name!r}: radius must be positive")
        if not 0.0 <= self.quality <= 1.0:
            raise ValueError(f"site {self.name!r}: quality must be within 0..1")


@dataclass(frozen=True, slots=True)
class SiteAssessment:
    """A scored candidate, kept explicit so the choice can be explained."""

    site: LandingSite
    reachable: bool
    distance_m: float
    battery_needed: float
    battery_margin: float
    wind_score: float
    total_score: float
    note: str = ""


@dataclass(slots=True)
class LandingSiteSelector:
    """Chooses an emergency landing site from a known set."""

    sites: list[LandingSite] = field(default_factory=list)
    cruise_speed_mps: float = 6.0
    descent_speed_mps: float = 1.5
    # Charge that must remain after touchdown. Arriving at exactly 0% means
    # arriving with no margin for a go-around or a misjudged flare.
    reserve: float = 0.05
    # Relative weights. Reachability is a gate rather than a weight.
    weight_quality: float = 0.45
    weight_proximity: float = 0.35
    weight_wind: float = 0.20

    def assess(self, site: LandingSite, telemetry: Telemetry) -> SiteAssessment:
        """Score one site against the current state."""
        distance = telemetry.position.horizontal_distance_to(site.position)
        needed = self._battery_to_reach(distance, telemetry)
        margin = telemetry.battery_remaining - needed - self.reserve
        reachable = margin >= 0.0

        wind_score = self._wind_score(site, telemetry)
        # Proximity falls off over a 200 m scale -- far enough that nearby
        # sites are barely separated, close enough that a distant site is
        # clearly penalised.
        proximity_score = math.exp(-distance / 200.0)

        total = (
            self.weight_quality * site.quality
            + self.weight_proximity * proximity_score
            + self.weight_wind * wind_score
        )
        if not reachable:
            total = 0.0

        return SiteAssessment(
            site=site,
            reachable=reachable,
            distance_m=distance,
            battery_needed=needed,
            battery_margin=margin,
            wind_score=wind_score,
            total_score=total,
            note="" if reachable else f"unreachable, short by {abs(margin):.0%}",
        )

    def select(self, telemetry: Telemetry) -> SiteAssessment | None:
        """Best reachable site, or None if nothing is within reach.

        None is a meaningful answer, not a failure: it means land immediately
        at the current position, because every alternative is further away
        than the remaining charge allows.
        """
        if not self.sites:
            return None
        assessments = [self.assess(s, telemetry) for s in self.sites]
        reachable = [a for a in assessments if a.reachable]
        if not reachable:
            return None
        return max(reachable, key=lambda a: a.total_score)

    def explain(self, telemetry: Telemetry) -> list[SiteAssessment]:
        """Every candidate scored, best first. For logs and post-flight review."""
        return sorted(
            (self.assess(s, telemetry) for s in self.sites),
            key=lambda a: a.total_score,
            reverse=True,
        )

    # -- internals ---------------------------------------------------------

    def _battery_to_reach(self, distance_m: float, telemetry: Telemetry) -> float:
        """Charge needed to fly `distance_m` and descend, as a fraction.

        Uses the drain rate observed so far this flight rather than a fixed
        constant, so it stays honest as the pack ages or the air gets cold.
        """
        spent = 1.0 - telemetry.battery_remaining
        if telemetry.elapsed_s < 1.0 or spent <= 0.0:
            return 0.0
        drain_per_second = spent / telemetry.elapsed_s

        # Ground speed against wind. A headwind on the outbound leg costs real
        # time, and time is charge.
        ground_speed = self._effective_ground_speed(distance_m, telemetry)
        transit_s = distance_m / max(ground_speed, 0.5)
        descent_s = telemetry.altitude / self.descent_speed_mps
        return drain_per_second * (transit_s + descent_s)

    def _effective_ground_speed(self, distance_m: float, telemetry: Telemetry) -> float:
        """Cruise speed adjusted for the wind component along the route.

        Conservative: assumes the worst case that the wind opposes us, scaled
        by how strong it is. Being pessimistic about reachability is the
        correct bias when the alternative is running out mid-transit.
        """
        headwind = min(telemetry.wind.norm, self.cruise_speed_mps * 0.8)
        return self.cruise_speed_mps - headwind

    def _wind_score(self, site: LandingSite, telemetry: Telemetry) -> float:
        """0..1 for how well the approach lines up with landing into wind.

        With no wind estimate, or a site with no preferred direction, this is
        neutral -- it should not tip a decision it has no information about.
        """
        wind_speed = telemetry.wind.norm
        if wind_speed < 0.5:
            return 0.5  # calm: no preference, stay neutral

        # Direction the vehicle will be travelling on approach.
        leg = site.position - telemetry.position
        if leg.horizontal_distance_to(Vec3()) < 1.0:
            approach_deg = telemetry.heading_deg
        else:
            approach_deg = math.degrees(math.atan2(leg.east, leg.north)) % 360.0

        if site.preferred_approach_deg is not None:
            approach_deg = site.preferred_approach_deg

        # Bearing the wind is blowing toward.
        wind_deg = math.degrees(math.atan2(telemetry.wind.east, telemetry.wind.north)) % 360.0
        # Landing into wind means travelling opposite to where the wind blows,
        # so the ideal approach heading is wind_deg + 180.
        ideal = (wind_deg + 180.0) % 360.0
        error = abs((approach_deg - ideal + 180.0) % 360.0 - 180.0)  # 0..180

        # 0 deg error -> 1.0, 180 deg (downwind) -> 0.0
        return 1.0 - error / 180.0
