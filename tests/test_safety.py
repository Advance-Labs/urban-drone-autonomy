"""Safety layer.

The `TestMeasurements` block covers the objective helpers and passes today.

The `TestPolicy` block is the *specification* for `SafetyMonitor.evaluate`,
which is deliberately left unimplemented. These tests fail until the policy is
written -- that is their job. They encode the contract, not the implementation,
so several reasonable policies satisfy them.
"""

from __future__ import annotations

import pytest

from autonomy import (
    FlightMode,
    SafetyAction,
    SafetyLimits,
    SafetyMonitor,
    Telemetry,
    Vec3,
)


NOMINAL_ENDURANCE_S = 900.0


def telem(
    *,
    north: float = 0.0,
    east: float = 0.0,
    altitude: float = 20.0,
    battery: float = 1.0,
    link_ok: bool = True,
    elapsed: float | None = None,
    wind: Vec3 | None = None,
) -> Telemetry:
    """Build a synthetic telemetry sample. No vehicle needed.

    `elapsed` defaults to whatever time it *should* have taken to burn down to
    `battery` at a realistic rate. That consistency matters: the policy derives
    its drain rate from `spent / elapsed`, so a sample claiming 80% consumed in
    60 s implies a 75-second pack and would make every site look unreachable.
    """
    if elapsed is None:
        elapsed = (1.0 - battery) * NOMINAL_ENDURANCE_S
    return Telemetry(
        position=Vec3(north, east, -altitude),
        velocity=Vec3(),
        heading_deg=0.0,
        battery_remaining=battery,
        mode=FlightMode.FLYING,
        armed=True,
        link_ok=link_ok,
        elapsed_s=elapsed,
        wind=wind or Vec3(),
    )


class TestMeasurements:
    """Objective helpers -- these pass already."""

    def test_distance_from_home_ignores_altitude(self):
        m = SafetyMonitor()
        assert m.distance_from_home(telem(north=30, east=40, altitude=100)) == pytest.approx(50.0)

    def test_geofence_clean_inside_envelope(self):
        m = SafetyMonitor(SafetyLimits(max_radius_m=100, max_altitude_m=80))
        assert m.geofence_breach(telem(north=50, altitude=50)) is None

    def test_geofence_flags_radius(self):
        m = SafetyMonitor(SafetyLimits(max_radius_m=100))
        reason = m.geofence_breach(telem(north=150))
        assert reason is not None and "radius" in reason

    def test_geofence_flags_altitude(self):
        m = SafetyMonitor(SafetyLimits(max_altitude_m=80))
        reason = m.geofence_breach(telem(altitude=120))
        assert reason is not None and "altitude" in reason

    def test_return_cost_grows_with_distance(self):
        m = SafetyMonitor()
        near = m.battery_to_return(telem(north=20, battery=0.8))
        far = m.battery_to_return(telem(north=200, battery=0.8))
        assert far > near

    def test_return_cost_is_zero_before_any_history(self):
        """With no elapsed time there is no drain rate to extrapolate from."""
        m = SafetyMonitor()
        assert m.battery_to_return(telem(elapsed=0.0, battery=1.0)) == 0.0

    def test_can_still_return_is_false_when_far_and_low(self):
        m = SafetyMonitor()
        assert not m.can_still_return(telem(north=400, battery=0.15, elapsed=300))


class TestPolicy:
    """Contract for `evaluate`. Failing until the policy is implemented."""

    def test_nominal_flight_continues(self):
        m = SafetyMonitor()
        assert m.evaluate(telem(battery=0.9, north=10)).action is SafetyAction.CONTINUE

    def test_warns_once_on_low_battery(self):
        """A warning should latch -- re-warning every tick at 10 Hz is noise,
        not information."""
        m = SafetyMonitor(SafetyLimits(battery_warn=0.35, battery_return=0.25))
        first = m.evaluate(telem(battery=0.30))
        second = m.evaluate(telem(battery=0.30))
        assert first.action is SafetyAction.WARN
        assert second.action is SafetyAction.CONTINUE

    def test_returns_home_on_low_battery(self):
        m = SafetyMonitor(SafetyLimits(battery_return=0.25, battery_critical=0.12))
        assert m.evaluate(telem(battery=0.20)).action is SafetyAction.RETURN_TO_LAUNCH

    def test_lands_immediately_on_critical_battery(self):
        """Below critical, flying home is a worse bet than landing where you
        are -- an RTL you cannot finish becomes a crash somewhere unplanned."""
        m = SafetyMonitor(SafetyLimits(battery_critical=0.12))
        assert m.evaluate(telem(battery=0.08, north=50)).action is SafetyAction.LAND_NOW

    def test_geofence_breach_triggers_return(self):
        m = SafetyMonitor(SafetyLimits(max_radius_m=100))
        verdict = m.evaluate(telem(north=150, battery=0.9))
        assert verdict.action is SafetyAction.RETURN_TO_LAUNCH
        assert verdict.reason, "a breach must explain itself in the log"

    def test_link_loss_starts_a_countdown_not_an_abort(self):
        """Telemetry links glitch constantly. An instant RTL on one dropped
        packet would abort missions for no reason."""
        m = SafetyMonitor(SafetyLimits(link_loss_grace_s=5.0))
        first = m.evaluate(telem(link_ok=False, battery=0.9, elapsed=100.0))
        assert first.action is SafetyAction.WARN
        still_waiting = m.evaluate(telem(link_ok=False, battery=0.9, elapsed=103.0))
        assert still_waiting.action is SafetyAction.CONTINUE

    def test_link_loss_returns_home_once_grace_expires(self):
        m = SafetyMonitor(SafetyLimits(link_loss_grace_s=5.0))
        m.evaluate(telem(link_ok=False, battery=0.9, elapsed=100.0))
        expired = m.evaluate(telem(link_ok=False, battery=0.9, elapsed=106.0))
        assert expired.action is SafetyAction.RETURN_TO_LAUNCH

    def test_link_recovery_cancels_the_countdown(self):
        m = SafetyMonitor(SafetyLimits(link_loss_grace_s=5.0))
        m.evaluate(telem(link_ok=False, battery=0.9, elapsed=100.0))
        m.evaluate(telem(link_ok=True, battery=0.9, elapsed=102.0))
        # A fresh dropout must restart the clock, not inherit the old one.
        again = m.evaluate(telem(link_ok=False, battery=0.9, elapsed=103.0))
        assert again.action is SafetyAction.WARN

    def test_critical_battery_outranks_geofence(self):
        """Two rules fire at once; the more severe action must win regardless
        of the order the policy happens to check them in."""
        m = SafetyMonitor(SafetyLimits(max_radius_m=100, battery_critical=0.12))
        assert m.evaluate(telem(north=200, battery=0.05)).action is SafetyAction.LAND_NOW

    def test_every_non_continue_verdict_has_a_reason(self):
        m = SafetyMonitor(SafetyLimits(max_radius_m=100))
        for sample in [
            telem(battery=0.05),
            telem(north=500, battery=0.9),
            telem(link_ok=False, battery=0.9),
        ]:
            verdict = m.evaluate(sample)
            if verdict.action is not SafetyAction.CONTINUE:
                assert verdict.reason.strip(), f"{verdict.action} logged with no reason"

    def test_reset_clears_the_warning_latch(self):
        m = SafetyMonitor(SafetyLimits(battery_warn=0.35))
        m.evaluate(telem(battery=0.30))
        m.reset()
        assert m.evaluate(telem(battery=0.30)).action is SafetyAction.WARN

    def test_returns_while_juice_remains_regardless_of_threshold(self):
        """The operator's rule: return whenever we still *can*, and the gate is
        the computed cost, not a fixed percentage."""
        m = SafetyMonitor(SafetyLimits(battery_return=0.25))
        near_home = telem(north=15, battery=0.22)
        assert m.can_still_return(near_home)
        assert m.evaluate(near_home).action is SafetyAction.RETURN_TO_LAUNCH

    def test_diverts_instead_of_attempting_an_impossible_rtl(self):
        """Far out and low: an RTL that cannot finish is a crash somewhere
        unplanned, so the policy must land rather than try."""
        m = SafetyMonitor(SafetyLimits(battery_return=0.25))
        stranded = telem(north=900, battery=0.18)
        assert not m.can_still_return(stranded)
        verdict = m.evaluate(stranded)
        assert verdict.action is SafetyAction.LAND_NOW
        assert "short of" in verdict.reason or "critical" in verdict.reason
