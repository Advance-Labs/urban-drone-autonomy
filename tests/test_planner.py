"""Planner geometry. Pure functions, so these are fast and deterministic."""

from __future__ import annotations

import math

import pytest

from autonomy import Vec3, estimated_duration_s, orbit, path_length, survey_grid


def test_survey_covers_full_width():
    wps = survey_grid(width_m=60, height_m=40, spacing_m=12, altitude_m=20)
    easts = sorted({round(w.east, 3) for w in wps})
    assert easts[0] == 0.0
    assert easts[-1] == 60.0, "final leg must reach the far edge of the area"


def test_survey_alternates_direction():
    """The defining property of a lawnmower pattern: consecutive legs reverse."""
    wps = survey_grid(width_m=40, height_m=30, spacing_m=10, altitude_m=15)
    legs = [(wps[i], wps[i + 1]) for i in range(0, len(wps), 2)]
    directions = [math.copysign(1, end.north - start.north) for start, end in legs]
    for a, b in zip(directions, directions[1:]):
        assert a != b, "adjacent legs must run in opposite directions"


def test_survey_holds_constant_altitude():
    wps = survey_grid(width_m=30, height_m=30, spacing_m=10, altitude_m=25)
    assert all(w.altitude == pytest.approx(25.0) for w in wps)


def test_survey_spacing_never_exceeds_sensor_swath():
    """Gaps wider than the requested spacing would leave uncovered ground."""
    wps = survey_grid(width_m=55, height_m=20, spacing_m=12, altitude_m=20)
    easts = sorted({round(w.east, 6) for w in wps})
    gaps = [b - a for a, b in zip(easts, easts[1:])]
    assert max(gaps) <= 12 + 1e-6


@pytest.mark.parametrize(
    "kwargs",
    [
        {"width_m": 0, "height_m": 10, "spacing_m": 5, "altitude_m": 10},
        {"width_m": 10, "height_m": 10, "spacing_m": 0, "altitude_m": 10},
        {"width_m": 10, "height_m": 10, "spacing_m": 5, "altitude_m": -1},
    ],
)
def test_survey_rejects_invalid_geometry(kwargs):
    with pytest.raises(ValueError):
        survey_grid(**kwargs)


def test_orbit_points_are_equidistant_from_centre():
    centre = Vec3(30, 0, 0)
    wps = orbit(centre=centre, radius_m=20, altitude_m=15, points=12)
    for w in wps:
        assert centre.horizontal_distance_to(w) == pytest.approx(20.0, abs=1e-6)


def test_orbit_rejects_degenerate_shapes():
    with pytest.raises(ValueError):
        orbit(centre=Vec3(), radius_m=10, altitude_m=10, points=2)


def test_path_length_matches_hand_computed():
    wps = [Vec3(0, 0, -10), Vec3(3, 4, -10), Vec3(3, 4, -20)]
    assert path_length(wps) == pytest.approx(15.0)


def test_estimated_duration_scales_inversely_with_speed():
    wps = survey_grid(width_m=40, height_m=40, spacing_m=10, altitude_m=20)
    assert estimated_duration_s(wps, 4.0) == pytest.approx(
        estimated_duration_s(wps, 8.0) * 2
    )
