"""Safety monitoring: the layer with veto power over the mission.

Runs on every control tick, before the active behaviour is allowed to issue a
command. If it returns anything above CONTINUE, the mission runner acts on it
immediately and the behaviour does not get its turn.

The split here is deliberate:
  * measuring the situation (how far from home, can we still get back, are we
    outside the fence) is objective and lives in `SafetyMonitor` helpers;
  * deciding what to DO about it is a judgement call and lives in one place,
    `evaluate`, so the policy is auditable rather than scattered through
    a dozen behaviours.
"""

from __future__ import annotations

from dataclasses import dataclass

from .landing import LandingSiteSelector
from .types import SafetyAction, SafetyVerdict, Telemetry, Vec3


@dataclass(frozen=True, slots=True)
class SafetyLimits:
    """Envelope the vehicle is expected to stay inside.

    Defaults are tuned for a small quad flown within visual line of sight,
    which is also what Transport Canada basic operations assume.
    """

    max_radius_m: float = 120.0
    max_altitude_m: float = 100.0  # TC caps VLOS operations at 122 m AGL
    min_altitude_m: float = 1.0
    battery_warn: float = 0.35
    battery_return: float = 0.25
    battery_critical: float = 0.12
    return_reserve: float = 0.08  # margin required on top of the return cost
    cruise_speed_mps: float = 6.0
    # How long to tolerate a dropped control link before giving up on it.
    # Telemetry links glitch constantly; an instant RTL on one lost packet
    # would abort missions for no reason. Long enough to ride out a dropout,
    # short enough that a real failure still leaves charge to get home.
    link_loss_grace_s: float = 5.0


class SafetyMonitor:
    """Stateless-per-tick evaluation of whether the flight is still safe.

    Holds only the limits and a small amount of latch state, so it can be
    unit-tested by feeding it synthetic `Telemetry` without any vehicle.
    """

    def __init__(
        self,
        limits: SafetyLimits | None = None,
        *,
        landing_sites: LandingSiteSelector | None = None,
    ) -> None:
        self.limits = limits or SafetyLimits()
        self.landing_sites = landing_sites or LandingSiteSelector(
            cruise_speed_mps=(limits or SafetyLimits()).cruise_speed_mps
        )
        self._warned = False
        # Wall-clock (mission-elapsed) instant the link was first seen down.
        # None means the link is currently up.
        self._link_lost_at: float | None = None

    # -- objective measurements -------------------------------------------

    def distance_from_home(self, telemetry: Telemetry) -> float:
        """Straight-line ground distance back to the launch point, metres."""
        return telemetry.position.horizontal_distance_to(Vec3())

    def geofence_breach(self, telemetry: Telemetry) -> str | None:
        """Describe the fence violation, or None if inside the envelope."""
        if self.distance_from_home(telemetry) > self.limits.max_radius_m:
            return (
                f"radius {self.distance_from_home(telemetry):.0f} m "
                f"exceeds {self.limits.max_radius_m:.0f} m"
            )
        if telemetry.altitude > self.limits.max_altitude_m:
            return (
                f"altitude {telemetry.altitude:.0f} m "
                f"exceeds {self.limits.max_altitude_m:.0f} m"
            )
        return None

    def battery_to_return(self, telemetry: Telemetry) -> float:
        """Fraction of a full pack needed to fly home and land from here.

        Derived from the vehicle's own observed drain rate rather than a fixed
        table, so it stays honest as the pack ages or the wind picks up.
        """
        spent = 1.0 - telemetry.battery_remaining
        if telemetry.elapsed_s < 1.0 or spent <= 0.0:
            return 0.0

        drain_per_second = spent / telemetry.elapsed_s
        transit_s = self.distance_from_home(telemetry) / max(self.limits.cruise_speed_mps, 0.1)
        descent_s = telemetry.altitude / 1.5  # land at ~1.5 m/s
        return drain_per_second * (transit_s + descent_s)

    def can_still_return(self, telemetry: Telemetry) -> bool:
        """True while there is enough charge to get home plus the reserve."""
        needed = self.battery_to_return(telemetry) + self.limits.return_reserve
        return telemetry.battery_remaining >= needed

    # -- policy ------------------------------------------------------------

    def evaluate(self, telemetry: Telemetry) -> SafetyVerdict:
        """Decide what the mission should do, given the current state.

        Policy, as specified by the operator:

          * Low battery means **return**, always, as long as there is enough
            charge to actually get home. The gate is the computed return cost,
            not a fixed percentage, because a fixed number is wrong in wind,
            in the cold, and on an aged pack.
          * If there is not enough charge to get home, stop trying. Divert to
            the best reachable landing site instead, choosing on quality,
            distance and wind. An RTL you cannot finish becomes a crash
            somewhere unplanned.
          * A dropped link starts a countdown rather than an immediate abort.
            Links glitch; missions should survive a glitch. If it stays down
            past the grace period, come home.

        Every rule is evaluated and the most severe verdict wins, so
        overlapping conditions resolve deterministically regardless of the
        order they are checked in.
        """
        candidates = [
            self._check_battery(telemetry),
            self._check_link(telemetry),
            self._check_geofence(telemetry),
        ]
        return max(candidates, key=lambda v: v.action.severity)

    def reset(self) -> None:
        """Clear latched state between flights."""
        self._warned = False
        self._link_lost_at = None

    # -- individual rules --------------------------------------------------

    def _check_battery(self, telemetry: Telemetry) -> SafetyVerdict:
        """Return while we still can; divert and land once we cannot."""
        remaining = telemetry.battery_remaining

        # Absolute floor. Below this, nothing is worth attempting except
        # getting on the ground as directly as possible.
        if remaining <= self.limits.battery_critical:
            return self._divert_verdict(
                telemetry, f"battery {remaining:.0%} at or below critical"
            )

        # The real gate: is there still enough charge to reach home?
        if not self.can_still_return(telemetry):
            needed = self.battery_to_return(telemetry) + self.limits.return_reserve
            return self._divert_verdict(
                telemetry,
                f"battery {remaining:.0%} short of the {needed:.0%} needed to reach home",
            )

        if remaining <= self.limits.battery_return:
            return SafetyVerdict(
                SafetyAction.RETURN_TO_LAUNCH,
                f"battery {remaining:.0%} below return threshold, home still reachable",
            )

        if remaining <= self.limits.battery_warn and not self._warned:
            self._warned = True
            return SafetyVerdict(SafetyAction.WARN, f"battery {remaining:.0%}")

        return SafetyVerdict.ok()

    def _check_link(self, telemetry: Telemetry) -> SafetyVerdict:
        """Countdown on link loss, rather than an immediate abort."""
        if telemetry.link_ok:
            if self._link_lost_at is not None:
                self._link_lost_at = None
                return SafetyVerdict(SafetyAction.WARN, "control link restored")
            return SafetyVerdict.ok()

        if self._link_lost_at is None:
            self._link_lost_at = telemetry.elapsed_s
            return SafetyVerdict(
                SafetyAction.WARN,
                f"control link lost, {self.limits.link_loss_grace_s:.0f}s grace started",
            )

        down_for = telemetry.elapsed_s - self._link_lost_at
        if down_for < self.limits.link_loss_grace_s:
            return SafetyVerdict.ok()  # still counting down

        # Grace expired. Same battery logic applies -- if home is out of
        # reach, diverting beats a doomed RTL.
        if not self.can_still_return(telemetry):
            return self._divert_verdict(
                telemetry, f"link down {down_for:.0f}s and home is out of range"
            )
        return SafetyVerdict(
            SafetyAction.RETURN_TO_LAUNCH, f"control link down {down_for:.0f}s"
        )

    def _check_geofence(self, telemetry: Telemetry) -> SafetyVerdict:
        breach = self.geofence_breach(telemetry)
        if breach is None:
            return SafetyVerdict.ok()
        return SafetyVerdict(SafetyAction.RETURN_TO_LAUNCH, f"geofence: {breach}")

    def _divert_verdict(self, telemetry: Telemetry, reason: str) -> SafetyVerdict:
        """LAND_NOW, aimed at the best reachable site if there is one.

        No site attached means land where we are, which is the correct
        outcome when nothing else is within reach.
        """
        choice = self.landing_sites.select(telemetry)
        if choice is None:
            return SafetyVerdict(
                SafetyAction.LAND_NOW, f"{reason}; landing in place"
            )
        return SafetyVerdict(
            SafetyAction.LAND_NOW,
            (
                f"{reason}; diverting to {choice.site.name} "
                f"({choice.distance_m:.0f} m, {choice.battery_margin:.0%} margin)"
            ),
            target=choice.site.position,
        )
