"""Path planning around obstacles."""

from __future__ import annotations

import pytest

from autonomy import Building, City, GridPlanner, Vec3, city_block_grid


def wall_city() -> City:
    """A single long wall between origin and the far side, with gaps at the ends."""
    return City(
        buildings=[Building(north=0, east=0, size_n=10, size_e=120, height=50, name="wall")],
        extent_m=200,
    )


class TestPlanning:
    def test_finds_a_route_around_a_wall(self):
        planner = GridPlanner(wall_city(), cell_m=6, margin_m=3)
        result = planner.plan(Vec3(-60, 0, -20), Vec3(60, 0, -20))
        assert result.ok, result.reason
        assert len(result.waypoints) >= 2

    def test_the_route_is_actually_collision_free(self):
        """The property that matters: every segment clears the obstacles."""
        city = wall_city()
        planner = GridPlanner(city, cell_m=6, margin_m=3)
        result = planner.plan(Vec3(-60, 0, -20), Vec3(60, 0, -20))
        assert result.ok
        for a, b in zip(result.waypoints, result.waypoints[1:]):
            assert city.segment_clear(a, b, 3.0), f"segment {a} -> {b} clips a building"

    def test_route_through_a_real_city_is_clear(self):
        city = city_block_grid(blocks=4, seed=5)
        planner = GridPlanner(city, cell_m=6, margin_m=4)
        result = planner.plan(Vec3(0, 0, -20), Vec3(140, 140, -25))
        assert result.ok, result.reason
        for a, b in zip(result.waypoints, result.waypoints[1:]):
            assert city.segment_clear(a, b, 4.0)

    def test_straight_line_stays_straight(self):
        """Smoothing must collapse open-ground runs, not leave a staircase."""
        planner = GridPlanner(City(extent_m=300), cell_m=6, margin_m=2)
        result = planner.plan(Vec3(-80, -80, -20), Vec3(80, 80, -20))
        assert result.ok
        assert len(result.waypoints) <= 3, f"staircase not smoothed: {len(result.waypoints)}"

    def test_smoothing_shortens_the_path(self):
        city = wall_city()
        planner = GridPlanner(city, cell_m=6, margin_m=3)
        result = planner.plan(Vec3(-60, -40, -20), Vec3(60, 40, -20))
        assert result.ok
        direct = (Vec3(60, 40, -20) - Vec3(-60, -40, -20)).norm
        # Detouring around a wall must cost something, but not absurdly much.
        assert direct <= result.length_m <= direct * 2.2

    def test_goal_inside_a_building_is_nudged_out(self):
        """Targets sit on walls constantly; failing outright would be useless."""
        planner = GridPlanner(wall_city(), cell_m=6, margin_m=3)
        result = planner.plan(Vec3(-60, 0, -20), Vec3(0, 0, -20))
        assert result.ok, "should relocate to the nearest free cell"

    def test_unreachable_goal_reports_failure(self):
        sealed = City(
            buildings=[
                Building(north=60, east=0, size_n=8, size_e=400, height=200, name="n"),
                Building(north=-60, east=0, size_n=8, size_e=400, height=200, name="s"),
                Building(north=0, east=60, size_n=400, size_e=8, height=200, name="e"),
                Building(north=0, east=-60, size_n=400, size_e=8, height=200, name="w"),
            ],
            extent_m=150,
        )
        planner = GridPlanner(sealed, cell_m=6, margin_m=2, max_altitude_m=90)
        result = planner.plan(Vec3(0, 0, -20), Vec3(120, 0, -20))
        assert not result.ok
        assert result.reason

    def test_prefers_going_around_over_climbing_over(self):
        """The climb penalty exists so the planner stops pogoing."""
        city = City(
            buildings=[Building(north=0, east=0, size_n=10, size_e=40, height=60, name="b")],
            extent_m=200,
        )
        planner = GridPlanner(city, cell_m=6, margin_m=3, max_altitude_m=100)
        result = planner.plan(Vec3(-50, 0, -20), Vec3(50, 0, -20))
        assert result.ok
        peak = max(w.altitude for w in result.waypoints)
        assert peak < 60, f"climbed to {peak:.0f} m instead of routing around"

    def test_respects_the_altitude_floor(self):
        planner = GridPlanner(City(extent_m=200), cell_m=6, min_altitude_m=12)
        result = planner.plan(Vec3(0, 0, -2), Vec3(60, 0, -2))
        assert result.ok
        assert all(w.altitude >= 11.9 for w in result.waypoints)

    def test_rejects_invalid_configuration(self):
        with pytest.raises(ValueError):
            GridPlanner(City(), cell_m=0)
        with pytest.raises(ValueError):
            GridPlanner(City(), min_altitude_m=50, max_altitude_m=20)


class TestCornerCutting:
    def test_does_not_thread_a_diagonal_through_a_corner(self):
        """The classic grid bug: a diagonal step slipping between two blocked
        cells and producing a path straight through solid wall."""
        city = City(
            buildings=[
                Building(north=0, east=0, size_n=12, size_e=12, height=60, name="a"),
                Building(north=12, east=12, size_n=12, size_e=12, height=60, name="b"),
            ],
            extent_m=120,
        )
        planner = GridPlanner(city, cell_m=6, margin_m=0.5)
        result = planner.plan(Vec3(12, -12, -20), Vec3(-12, 12, -20))
        if result.ok:
            for a, b in zip(result.waypoints, result.waypoints[1:]):
                assert city.segment_clear(a, b, 0.5)
