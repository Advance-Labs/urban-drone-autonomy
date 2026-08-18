"""Emergency landing site selection under battery and wind constraints."""

from __future__ import annotations

import pytest

from autonomy import FlightMode, SafetyAction, SafetyLimits, SafetyMonitor, Telemetry, Vec3
from autonomy.landing import LandingSite, LandingSiteSelector


def telem(
    *,
    north: float = 0.0,
    east: float = 0.0,
    altitude: float = 20.0,
    battery: float = 0.2,
    elapsed: float = 720.0,
    wind: Vec3 | None = None,
) -> Telemetry:
    return Telemetry(
        position=Vec3(north, east, -altitude),
        velocity=Vec3(),
        heading_deg=0.0,
        battery_remaining=battery,
        mode=FlightMode.FLYING,
        armed=True,
        link_ok=True,
        elapsed_s=elapsed,
        wind=wind or Vec3(),
    )


class TestSiteValidation:
    def test_rejects_impossible_quality(self):
        with pytest.raises(ValueError, match="quality"):
            LandingSite(position=Vec3(), quality=1.5)

    def test_rejects_zero_radius(self):
        with pytest.raises(ValueError, match="radius"):
            LandingSite(position=Vec3(), radius_m=0)


class TestSelection:
    def test_no_sites_returns_none(self):
        assert LandingSiteSelector().select(telem()) is None

    def test_prefers_the_closer_of_two_equal_sites(self):
        sel = LandingSiteSelector(
            sites=[
                LandingSite(position=Vec3(30, 0, 0), name="near"),
                LandingSite(position=Vec3(150, 0, 0), name="far"),
            ]
        )
        assert sel.select(telem(battery=0.5)).site.name == "near"

    def test_prefers_better_quality_when_distances_are_similar(self):
        sel = LandingSiteSelector(
            sites=[
                LandingSite(position=Vec3(40, 0, 0), name="road", quality=0.2),
                LandingSite(position=Vec3(-40, 0, 0), name="field", quality=1.0),
            ]
        )
        assert sel.select(telem(battery=0.5)).site.name == "field"

    def test_unreachable_sites_are_never_chosen(self):
        """A perfect site we cannot reach scores zero -- that is the whole
        point of the reachability gate."""
        sel = LandingSiteSelector(
            sites=[
                LandingSite(position=Vec3(5000, 0, 0), name="perfect", quality=1.0),
                LandingSite(position=Vec3(20, 0, 0), name="scrappy", quality=0.3),
            ]
        )
        choice = sel.select(telem(battery=0.15))
        assert choice is not None and choice.site.name == "scrappy"

    def test_returns_none_when_nothing_is_reachable(self):
        """None means land in place, which is correct when every alternative
        is further than the remaining charge allows."""
        sel = LandingSiteSelector(
            sites=[LandingSite(position=Vec3(9000, 0, 0), name="far")]
        )
        assert sel.select(telem(battery=0.10)) is None

    def test_reserve_is_respected(self):
        near_miss = LandingSiteSelector(
            sites=[LandingSite(position=Vec3(400, 0, 0))], reserve=0.5
        )
        assert near_miss.select(telem(battery=0.2)) is None


class TestWind:
    def test_headwind_reduces_reachability(self):
        """Fighting wind costs time, and time is charge."""
        sites = [LandingSite(position=Vec3(300, 0, 0), name="downrange")]
        calm = LandingSiteSelector(sites=sites).assess(sites[0], telem(battery=0.3))
        windy = LandingSiteSelector(sites=sites).assess(
            sites[0], telem(battery=0.3, wind=Vec3(-5, 0, 0))
        )
        assert windy.battery_needed > calm.battery_needed

    def test_landing_into_wind_scores_higher_than_downwind(self):
        into = LandingSite(position=Vec3(100, 0, 0), name="into", preferred_approach_deg=0.0)
        down = LandingSite(position=Vec3(100, 0, 0), name="down", preferred_approach_deg=180.0)
        # Wind blowing toward the south (negative north) means an ideal
        # approach heading of due north.
        sample = telem(battery=0.5, wind=Vec3(-6, 0, 0))
        sel = LandingSiteSelector(sites=[into, down])
        assert sel.assess(into, sample).wind_score > sel.assess(down, sample).wind_score

    def test_calm_wind_is_neutral_and_does_not_tip_the_choice(self):
        site = LandingSite(position=Vec3(50, 0, 0), preferred_approach_deg=90.0)
        sel = LandingSiteSelector(sites=[site])
        assert sel.assess(site, telem(battery=0.5)).wind_score == pytest.approx(0.5)


class TestExplain:
    def test_explain_ranks_every_candidate(self):
        sel = LandingSiteSelector(
            sites=[
                LandingSite(position=Vec3(20, 0, 0), name="a", quality=0.9),
                LandingSite(position=Vec3(80, 0, 0), name="b", quality=0.5),
                LandingSite(position=Vec3(9000, 0, 0), name="unreachable"),
            ]
        )
        ranked = sel.explain(telem(battery=0.4))
        assert len(ranked) == 3
        assert [a.total_score for a in ranked] == sorted(
            (a.total_score for a in ranked), reverse=True
        )
        assert ranked[-1].site.name == "unreachable"
        assert "unreachable" in ranked[-1].note


class TestPolicyIntegration:
    def test_stranded_vehicle_diverts_to_a_site(self):
        sel = LandingSiteSelector(
            sites=[LandingSite(position=Vec3(820, 0, 0), name="ballfield")]
        )
        m = SafetyMonitor(SafetyLimits(), landing_sites=sel)
        verdict = m.evaluate(telem(north=800, battery=0.18, elapsed=740))
        assert verdict.action is SafetyAction.LAND_NOW
        assert verdict.target is not None, "should have picked a site to divert to"
        assert "ballfield" in verdict.reason

    def test_stranded_with_no_sites_lands_in_place(self):
        m = SafetyMonitor(SafetyLimits())
        verdict = m.evaluate(telem(north=800, battery=0.18, elapsed=740))
        assert verdict.action is SafetyAction.LAND_NOW
        assert verdict.target is None
        assert "in place" in verdict.reason
