"""Sensor modelling: detection, and more importantly, non-detection.

The value of this module is that it says NO. A simulator that always sees the
wire produces autonomy that is confidently wrong on real hardware, so most of
these tests pin failure modes rather than successes.
"""

from __future__ import annotations

import pytest

from autonomy import (
    Cable,
    City,
    Foliage,
    LidarSpec,
    SensorSuite,
    StereoSpec,
    ThinObstacleField,
    Vec3,
    detection_probability,
    powerline_corridor,
    stopping_distance_m,
    wire_detection_range,
)


class TestDetectionProbability:
    def test_thin_targets_are_missed_more_at_range(self):
        lid = LidarSpec()
        near = detection_probability(0.012, 10, lid.v_resolution_rad, 0.25, 80)
        far = detection_probability(0.012, 40, lid.v_resolution_rad, 0.25, 80)
        assert near > far > 0.0

    def test_probability_scales_inversely_with_range(self):
        """Subtense is width/range, so halving range doubles the chance."""
        lid = LidarSpec()
        p20 = detection_probability(0.012, 20, lid.v_resolution_rad, 0.25, 80)
        p10 = detection_probability(0.012, 10, lid.v_resolution_rad, 0.25, 80)
        assert p10 == pytest.approx(2 * p20, rel=0.1)

    def test_large_targets_saturate_at_certainty(self):
        """A building is not probabilistically detected -- it fills the beam."""
        lid = LidarSpec()
        assert detection_probability(10.0, 40, lid.h_resolution_rad, 0.75, 80) == 1.0

    def test_finer_resolution_helps_thin_targets(self):
        coarse = LidarSpec(v_resolution_deg=0.8)
        fine = LidarSpec(v_resolution_deg=0.1)
        assert (
            detection_probability(0.012, 20, fine.v_resolution_rad, 0.25, 80)
            > detection_probability(0.012, 20, coarse.v_resolution_rad, 0.25, 80)
        )

    def test_beyond_effective_range_is_zero(self):
        assert detection_probability(1.0, 500, 0.003, 0.25, 80) == 0.0

    def test_rating_is_not_double_counted(self):
        """A lidar's rated range already includes 1/r^2 falloff. Applying it
        again made buildings vanish at half the advertised range."""
        lid = LidarSpec(range_m=80)
        assert detection_probability(10.0, 70, lid.h_resolution_rad, 0.75, 80) > 0.9

    def test_dark_targets_fall_short_of_the_rated_range(self):
        bright = detection_probability(5.0, 78, 0.003, 0.75, 80)
        dark = detection_probability(5.0, 78, 0.003, 0.05, 80)
        assert bright > 0 and dark == 0.0


class TestResolutionAxis:
    def test_horizontal_wire_is_limited_by_vertical_spacing(self):
        """A horizontal wire spans hundreds of beams along a row, so what
        matters is whether any beam ROW lands on it."""
        lid = LidarSpec(h_resolution_deg=0.1, v_resolution_deg=0.5)
        assert lid.resolution_across(Vec3(0, 1, 0)) == pytest.approx(
            lid.v_resolution_rad
        )

    def test_vertical_wire_is_limited_by_horizontal_spacing(self):
        lid = LidarSpec(h_resolution_deg=0.1, v_resolution_deg=0.5)
        assert lid.resolution_across(Vec3(0, 0, 1)) == pytest.approx(
            lid.h_resolution_rad
        )


class TestWireStrikeMargin:
    """The operationally meaningful result: can we stop in time?"""

    def setup_method(self):
        self.cable = Cable(start=Vec3(0, -40, -9), end=Vec3(0, 40, -9))
        self.suite = SensorSuite(lidar=LidarSpec(), stereo=StereoSpec())

    def test_slow_flight_detects_in_time(self):
        det = wire_detection_range(self.cable, self.suite, closing_speed_mps=3.0)
        assert det > stopping_distance_m(3.0, 0.6)

    def test_fast_flight_cannot_stop_in_time(self):
        """This is the real-world result: wire strike is a leading cause of
        small-UAS loss precisely because speed outruns detection."""
        det = wire_detection_range(self.cable, self.suite, closing_speed_mps=12.0)
        assert det < stopping_distance_m(12.0, 0.6)

    def test_requiring_confirmation_costs_range(self):
        """One return is a bird; several are a wire. Demanding confirmation is
        correct and it measurably shortens the usable detection range."""
        one = wire_detection_range(self.cable, self.suite, required_returns=1)
        three = wire_detection_range(self.cable, self.suite, required_returns=3)
        assert one > three

    def test_thinner_wire_is_detected_later(self):
        thin = Cable(start=Vec3(0, -40, -9), end=Vec3(0, 40, -9), diameter_m=0.005)
        assert (
            wire_detection_range(thin, self.suite)
            < wire_detection_range(self.cable, self.suite)
        )

    def test_better_sensor_raises_the_safe_speed(self):
        good = SensorSuite(
            lidar=LidarSpec(v_resolution_deg=0.1, h_resolution_deg=0.05, scan_hz=40)
        )
        assert (
            wire_detection_range(self.cable, good, closing_speed_mps=8.0)
            > wire_detection_range(self.cable, self.suite, closing_speed_mps=8.0)
        )

    def test_rejects_invalid_parameters(self):
        with pytest.raises(ValueError):
            wire_detection_range(self.cable, self.suite, closing_speed_mps=0)
        with pytest.raises(ValueError):
            wire_detection_range(self.cable, self.suite, required_returns=0)
        with pytest.raises(ValueError):
            stopping_distance_m(5.0, 0.0)


class TestScanning:
    def test_scan_is_reproducible_for_a_seed(self):
        thin = ThinObstacleField(cables=tuple(powerline_corridor(
            north=20, east_start=-40, east_end=40, spans=2,
        )))
        a = SensorSuite(thin=thin, seed=5)
        b = SensorSuite(thin=thin, seed=5)
        eye, hdg = Vec3(0, 0, -12), 0.0
        ra = [a.scan(eye, hdg) for _ in range(40)]
        rb = [b.scan(eye, hdg) for _ in range(40)]
        assert [len(x) for x in ra] == [len(x) for x in rb]

    def test_a_wire_behind_the_aircraft_is_never_returned(self):
        thin = ThinObstacleField(cables=(
            Cable(start=Vec3(-60, -20, -9), end=Vec3(-60, 20, -9), name="behind"),
        ))
        suite = SensorSuite(thin=thin, seed=1)
        assert all(
            not suite.scan(Vec3(0, 0, -12), 0.0) for _ in range(50)
        ), "wire directly behind should never be in the forward FOV"

    def test_wire_ahead_is_eventually_returned(self):
        thin = ThinObstacleField(cables=(
            Cable(start=Vec3(25, -30, -11), end=Vec3(25, 30, -11), name="ahead"),
        ))
        suite = SensorSuite(thin=thin, seed=3)
        hits = sum(len(suite.scan(Vec3(0, 0, -12), 0.0)) for _ in range(300))
        assert hits > 0, "should accumulate returns over many scans"

    def test_detections_carry_range_and_confidence(self):
        thin = ThinObstacleField(cables=(
            Cable(start=Vec3(15, -30, -11), end=Vec3(15, 30, -11), name="w"),
        ))
        suite = SensorSuite(thin=thin, seed=2)
        for _ in range(400):
            for d in suite.scan(Vec3(0, 0, -12), 0.0):
                assert d.kind == "cable" and d.is_hard
                assert 0 < d.confidence <= 1.0
                assert d.range_m > 0
                return
        pytest.skip("no return in the sample window; probabilistic by design")

    def test_foliage_returns_are_not_hard_obstacles(self):
        thin = ThinObstacleField(foliage=(
            Foliage(centre=Vec3(30, 0, -8), radius_m=6.0, name="oak"),
        ))
        suite = SensorSuite(thin=thin, seed=4)
        for _ in range(50):
            for d in suite.scan(Vec3(0, 0, -8), 0.0):
                if d.kind == "foliage":
                    assert not d.is_hard, "canopy must not read as solid"
                    return
        pytest.skip("no foliage return in window")


class TestFoliageAndCables:
    def test_canopy_is_passable_but_the_trunk_is_not(self):
        tree = Foliage(centre=Vec3(0, 0, -10), radius_m=8.0, trunk_radius_m=0.3)
        assert tree.canopy_contains(Vec3(5, 0, -10))
        assert not tree.contains(Vec3(5, 0, -10)), "leaves stop nothing"
        assert tree.contains(Vec3(0, 0, -5)), "the trunk stops everything"

    def test_foliage_attenuates_but_does_not_block_sight(self):
        thin = ThinObstacleField(foliage=(
            Foliage(centre=Vec3(20, 0, -10), radius_m=8.0, density=0.4),
        ))
        t = thin.sightline_transmission(Vec3(0, 0, -10), Vec3(40, 0, -10))
        assert 0.0 < t < 1.0

    def test_cable_sag_dips_below_its_supports(self):
        """Sag is why a drone crossing at 'safe' pole height still hits."""
        c = Cable(start=Vec3(0, -30, -10), end=Vec3(0, 30, -10), sag_m=3.0)
        lowest = max(p.down for p in c.points())
        assert lowest > -10.0, "midspan must hang below the poles"

    def test_cable_distance_accounts_for_diameter(self):
        c = Cable(start=Vec3(0, -10, -10), end=Vec3(0, 10, -10), diameter_m=0.02, sag_m=0)
        assert c.distance_to(Vec3(1.0, 0, -10)) == pytest.approx(1.0 - 0.01, abs=1e-6)

    def test_thin_field_sweep_catches_a_wire(self):
        thin = ThinObstacleField(cables=(
            Cable(start=Vec3(0, -20, -10), end=Vec3(0, 20, -10), sag_m=0.0),
        ))
        assert not thin.segment_clear(Vec3(-5, 0, -10), Vec3(5, 0, -10), margin=0.3)

    def test_rejects_invalid_geometry(self):
        with pytest.raises(ValueError):
            Cable(start=Vec3(), end=Vec3(0, 1, 0), diameter_m=0)
        with pytest.raises(ValueError):
            Foliage(centre=Vec3(), radius_m=5, density=2.0)
        with pytest.raises(ValueError):
            powerline_corridor(north=0, east_start=0, east_end=10, spans=0)
