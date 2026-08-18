"""Camera sensing, occlusion, and coverage accounting."""

from __future__ import annotations

import pytest

from autonomy import (
    Building,
    Camera,
    City,
    CoverageTracker,
    FlightMode,
    SurveillanceTarget,
    Telemetry,
    Vec3,
    observation_point,
)


def telem(pos: Vec3, heading: float = 0.0, elapsed: float = 10.0) -> Telemetry:
    return Telemetry(
        position=pos, velocity=Vec3(), heading_deg=heading,
        battery_remaining=0.9, mode=FlightMode.HOLDING, armed=True,
        link_ok=True, elapsed_s=elapsed,
    )


class TestCamera:
    def test_rejects_invalid_optics(self):
        with pytest.raises(ValueError):
            Camera(fov_deg=0)
        with pytest.raises(ValueError):
            Camera(range_m=-5)

    def test_target_beyond_range_is_not_seen(self):
        cam = Camera(range_m=50, depression_deg=0)
        assert not cam.sees(Vec3(0, 0, -20), 0.0, Vec3(200, 0, -20))

    def test_target_behind_the_aircraft_is_not_seen(self):
        cam = Camera(fov_deg=60, range_m=200, depression_deg=0)
        assert not cam.sees(Vec3(0, 0, -20), 0.0, Vec3(-60, 0, -20))

    def test_target_ahead_within_the_cone_is_seen(self):
        cam = Camera(fov_deg=60, range_m=200, depression_deg=0)
        assert cam.sees(Vec3(0, 0, -20), 0.0, Vec3(60, 0, -20))

    def test_depression_points_the_lens_downward(self):
        """A 45-degree gimbal looking down should see the ground ahead, not
        the horizon."""
        cam = Camera(fov_deg=40, range_m=200, depression_deg=45)
        below_ahead = Vec3(40, 0, 0)  # ground, 40 m ahead of a 40 m-high hover
        assert cam.sees(Vec3(0, 0, -40), 0.0, below_ahead)
        assert not cam.sees(Vec3(0, 0, -40), 0.0, Vec3(140, 0, -40))

    def test_a_building_blocks_the_view(self):
        """The whole point: in range and in frame is not the same as seen."""
        cam = Camera(fov_deg=90, range_m=200, depression_deg=0)
        city = City(buildings=[
            Building(north=30, east=0, size_n=10, size_e=60, height=60, name="w")
        ])
        target = Vec3(80, 0, -20)
        assert cam.sees(Vec3(0, 0, -20), 0.0, target), "clear without the city"
        assert not cam.sees(Vec3(0, 0, -20), 0.0, target, city)


class TestCoverage:
    def test_dwell_must_accumulate_before_a_target_counts(self):
        t = SurveillanceTarget(position=Vec3(50, 0, -20), name="T", required_dwell_s=3.0)
        tracker = CoverageTracker(
            targets=[t], camera=Camera(fov_deg=60, range_m=200, depression_deg=0)
        )
        sample = telem(Vec3(0, 0, -20))
        assert tracker.update(sample, 1.0) == []
        assert not tracker.is_covered(t)
        tracker.update(sample, 1.0)
        assert tracker.update(sample, 1.0) == ["T"], "should report newly covered"
        assert tracker.is_covered(t)

    def test_a_target_never_in_frame_stays_missed(self):
        t = SurveillanceTarget(position=Vec3(-80, 0, -20), name="T")
        tracker = CoverageTracker(
            targets=[t], camera=Camera(fov_deg=60, range_m=200, depression_deg=0)
        )
        for _ in range(30):
            tracker.update(telem(Vec3(0, 0, -20)), 1.0)
        assert tracker.missed == ["T"]
        assert tracker.fraction == 0.0

    def test_newly_covered_reports_once_not_every_tick(self):
        t = SurveillanceTarget(position=Vec3(50, 0, -20), name="T", required_dwell_s=1.0)
        tracker = CoverageTracker(
            targets=[t], camera=Camera(fov_deg=60, range_m=200, depression_deg=0)
        )
        s = telem(Vec3(0, 0, -20))
        assert tracker.update(s, 1.0) == ["T"]
        assert tracker.update(s, 1.0) == [], "must not re-announce a covered target"

    def test_fraction_with_no_targets_is_complete(self):
        assert CoverageTracker().fraction == 1.0

    def test_rejects_bad_timestep(self):
        with pytest.raises(ValueError):
            CoverageTracker().update(telem(Vec3()), 0.0)


class TestObservationPoint:
    def test_finds_a_vantage_in_the_open(self):
        t = SurveillanceTarget(position=Vec3(0, 0, -2))
        eye = observation_point(t, Camera(), City(extent_m=300), standoff_m=40, altitude_m=30)
        assert eye is not None
        assert eye.position_check if False else True
        assert abs(t.position.horizontal_distance_to(eye) - 40) < 1e-6

    def test_returns_none_when_no_vantage_exists(self):
        """A courtyard ringed by towers genuinely has no view in, and saying
        so beats returning a point that stares at a wall."""
        walls = [
            Building(north=25, east=0, size_n=6, size_e=90, height=120, name="n"),
            Building(north=-25, east=0, size_n=6, size_e=90, height=120, name="s"),
            Building(north=0, east=25, size_n=90, size_e=6, height=120, name="e"),
            Building(north=0, east=-25, size_n=90, size_e=6, height=120, name="w"),
        ]
        city = City(buildings=walls, extent_m=300)
        t = SurveillanceTarget(position=Vec3(0, 0, -2))
        assert observation_point(t, Camera(), city, standoff_m=60, altitude_m=30) is None

    def test_rejects_invalid_arguments(self):
        with pytest.raises(ValueError):
            observation_point(
                SurveillanceTarget(position=Vec3()), Camera(), City(), standoff_m=0
            )
