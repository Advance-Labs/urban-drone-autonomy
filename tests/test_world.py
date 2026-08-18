"""Urban world model: occupancy, swept collision, and occlusion."""

from __future__ import annotations

import pytest

from autonomy import Building, City, Vec3, city_block_grid


def tower(n=0.0, e=0.0, size=20.0, height=40.0, name="t") -> Building:
    return Building(north=n, east=e, size_n=size, size_e=size, height=height, name=name)


class TestBuilding:
    def test_rejects_degenerate_geometry(self):
        with pytest.raises(ValueError):
            Building(north=0, east=0, size_n=0, size_e=10, height=10)
        with pytest.raises(ValueError):
            Building(north=0, east=0, size_n=10, size_e=10, height=-1)

    def test_contains_inside_and_outside(self):
        b = tower(size=20, height=30)
        assert b.contains(Vec3(0, 0, -15))
        assert not b.contains(Vec3(0, 0, -35)), "above the roof is clear air"
        assert not b.contains(Vec3(30, 0, -15)), "beyond the footprint"

    def test_margin_inflates_in_all_three_axes(self):
        b = tower(size=20, height=30)
        just_outside = Vec3(12, 0, -15)
        assert not b.contains(just_outside)
        assert b.contains(just_outside, margin=4.0)
        assert b.contains(Vec3(0, 0, -32), margin=4.0), "margin must apply upward too"


class TestCity:
    def test_below_ground_is_never_free(self):
        assert not City().is_free(Vec3(0, 0, 5))  # down=+5 is underground

    def test_occupant_names_the_obstacle(self):
        c = City(buildings=[tower(name="hall")])
        assert c.occupant(Vec3(0, 0, -10)).name == "hall"
        assert c.occupant(Vec3(100, 0, -10)) is None

    def test_segment_through_a_building_is_blocked(self):
        c = City(buildings=[tower(size=20, height=40)])
        assert not c.segment_clear(Vec3(-40, 0, -20), Vec3(40, 0, -20))

    def test_segment_over_the_roof_is_clear(self):
        c = City(buildings=[tower(size=20, height=40)])
        assert c.segment_clear(Vec3(-40, 0, -60), Vec3(40, 0, -60))

    def test_thin_walls_are_not_tunnelled(self):
        """A coarse sampling step can step straight over a thin building."""
        thin = Building(north=0, east=0, size_n=2.0, size_e=60, height=30, name="wall")
        c = City(buildings=[thin])
        assert not c.segment_clear(
            Vec3(-30, 0, -15), Vec3(30, 0, -15), step_m=0.5
        )

    def test_line_of_sight_is_blocked_by_a_building(self):
        c = City(buildings=[tower(size=20, height=40)])
        assert not c.has_line_of_sight(Vec3(-50, 0, -10), Vec3(50, 0, -10))

    def test_line_of_sight_ignores_margin(self):
        """Sight is blocked only where a wall physically is -- unlike planning,
        where standoff matters."""
        c = City(buildings=[tower(size=20, height=40)])
        # Passing just outside the footprint: blocked for planning at margin 4,
        # but visually clear.
        assert c.has_line_of_sight(Vec3(-50, 11, -10), Vec3(50, 11, -10))
        assert not c.segment_clear(Vec3(-50, 11, -10), Vec3(50, 11, -10), margin=4.0)

    def test_target_against_a_wall_is_not_self_occluding(self):
        c = City(buildings=[tower(size=20, height=40)])
        on_the_wall = Vec3(10.0, 0, -5)
        assert c.has_line_of_sight(Vec3(60, 0, -20), on_the_wall)


class TestGenerator:
    def test_is_deterministic_for_a_seed(self):
        a = city_block_grid(seed=3)
        b = city_block_grid(seed=3)
        assert [x.name for x in a.buildings] == [x.name for x in b.buildings]
        assert [round(x.height, 6) for x in a.buildings] == [
            round(x.height, 6) for x in b.buildings
        ]

    def test_different_seeds_differ(self):
        a = city_block_grid(seed=1)
        b = city_block_grid(seed=2)
        assert [round(x.height) for x in a.buildings] != [
            round(x.height) for x in b.buildings
        ]

    def test_launch_point_is_clear(self):
        """Takeoff must not begin inside a wall."""
        c = city_block_grid(blocks=4, seed=11)
        assert c.is_free(Vec3(0, 0, -1), margin=4.0)

    def test_rejects_invalid_parameters(self):
        with pytest.raises(ValueError):
            city_block_grid(blocks=0)
        with pytest.raises(ValueError):
            city_block_grid(street_m=-5)

    def test_serialises_for_the_viewer(self):
        d = city_block_grid(blocks=3).to_dict()
        assert d["buildings"] and {"n", "e", "sn", "se", "h"} <= set(d["buildings"][0])
