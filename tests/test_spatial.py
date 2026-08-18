"""Spatial indexing: correctness first, then the speed it exists for."""

from __future__ import annotations

import time

import pytest

from autonomy.osm import Footprint, OsmCity, PowerRun, furnish_power, power_obstacles
from autonomy.spatial import SpatialHash
from autonomy.thin_obstacles import Cable, ThinObstacleField
from autonomy.types import Vec3


class TestSpatialHash:
    def test_rejects_bad_cell_size(self):
        with pytest.raises(ValueError):
            SpatialHash(cell_m=0)

    def test_query_finds_an_item_in_its_cell(self):
        h: SpatialHash[str] = SpatialHash(cell_m=10.0)
        h.insert("a", 0, 5, 0, 5)
        assert "a" in list(h.query(2, 2))

    def test_query_misses_a_distant_item(self):
        h: SpatialHash[str] = SpatialHash(cell_m=10.0)
        h.insert("a", 0, 5, 0, 5)
        assert list(h.query(500, 500)) == []

    def test_large_items_span_several_cells(self):
        h: SpatialHash[str] = SpatialHash(cell_m=10.0)
        h.insert("big", 0, 95, 0, 95)
        assert "big" in list(h.query(5, 5))
        assert "big" in list(h.query(90, 90))

    def test_radius_widens_the_query(self):
        h: SpatialHash[str] = SpatialHash(cell_m=10.0)
        h.insert("a", 0, 1, 0, 1)
        assert list(h.query(45, 0)) == []
        assert "a" in list(h.query(45, 0, radius=50))

    def test_stats_report_bucket_load(self):
        h: SpatialHash[int] = SpatialHash(cell_m=10.0)
        for i in range(20):
            h.insert(i, i * 15, i * 15 + 2, 0, 2)
        s = h.stats()
        assert s["items"] == 20 and s["buckets"] > 1


def grid_city(n: int = 14) -> OsmCity:
    """A synthetic city dense enough that indexing matters."""
    city = OsmCity(extent_m=400)
    for i in range(n):
        for j in range(n):
            n0, e0 = -300 + i * 45, -300 + j * 45
            city.footprints.append(
                Footprint(
                    points=((n0, e0), (n0 + 20, e0), (n0 + 20, e0 + 20), (n0, e0 + 20)),
                    height=8.0, name=f"b{i}{j}",
                )
            )
    return city


class TestIndexedCollisionMatchesBruteForce:
    """The index must not change any answer, only the time to get it."""

    def test_is_free_agrees_with_exhaustive_search(self):
        city = grid_city()
        probes = [
            Vec3(-300 + i * 7.3, -300 + i * 11.7, -4.0) for i in range(400)
        ]
        for p in probes:
            brute = not any(f.contains(p, 3.0) for f in city.footprints)
            assert city.is_free(p, 3.0) == brute, f"disagreement at {p}"

    def test_occupant_agrees_with_exhaustive_search(self):
        city = grid_city()
        for i in range(200):
            p = Vec3(-300 + i * 5.1, -300 + i * 9.3, -4.0)
            brute = next((f for f in city.footprints if f.contains(p, 2.0)), None)
            found = city.occupant(p, 2.0)
            assert (found is None) == (brute is None)

    def test_cable_field_agrees_with_exhaustive_search(self):
        cables = tuple(
            Cable(start=Vec3(i * 12, -60, -9), end=Vec3(i * 12, 60, -9),
                  sag_m=1.2, name=f"c{i}")
            for i in range(30)
        )
        field = ThinObstacleField(cables=cables)
        for i in range(300):
            p = Vec3(-20 + i * 1.3, -70 + i * 0.5, -8.0)
            brute = any(c.contains(p, 3.0) for c in cables)
            assert (field.solid_at(p, 3.0) is not None) == brute

    def test_index_rebuilds_when_footprints_are_appended(self):
        """`furnish_power` and the loader both append after construction."""
        city = grid_city(4)
        p = Vec3(500, 500, -4.0)
        assert city.is_free(p, 1.0)
        city.footprints.append(
            Footprint(points=((495, 495), (510, 495), (510, 510), (495, 510)),
                      height=9.0, name="late")
        )
        assert not city.is_free(p, 1.0), "index must notice the new building"


class TestPerformance:
    """These exist because the linear scan made a realistic city unusable:
    8.5 ms per wire query is two hours of collision tests for one search."""

    def test_building_queries_are_fast(self):
        city = grid_city()
        probes = [Vec3(i % 300 - 150, i % 250 - 125, -5.0) for i in range(2000)]
        start = time.perf_counter()
        for p in probes:
            city.is_free(p, 3.0)
        per_call_us = (time.perf_counter() - start) / len(probes) * 1e6
        assert per_call_us < 120, f"{per_call_us:.0f} us/call is too slow to plan with"

    def test_cable_queries_are_fast(self):
        cables = tuple(
            Cable(start=Vec3(i * 8, -100, -9), end=Vec3(i * 8, 100, -9),
                  sag_m=1.2, name=f"c{i}")
            for i in range(120)
        )
        field = ThinObstacleField(cables=cables)
        probes = [Vec3(i % 400 - 200, i % 180 - 90, -8.0) for i in range(2000)]
        start = time.perf_counter()
        for p in probes:
            field.solid_at(p, 2.5)
        per_call_us = (time.perf_counter() - start) / len(probes) * 1e6
        assert per_call_us < 200, f"{per_call_us:.0f} us/call is too slow to plan with"


class TestFurnishing:
    def _city_with_road(self) -> OsmCity:
        city = OsmCity(extent_m=300)
        city.roads.append([(0.0, -150.0), (0.0, 150.0)])
        return city

    def test_a_long_road_gains_a_line(self):
        city = self._city_with_road()
        assert furnish_power(city) == 1
        assert len(city.power) == 1

    def test_synthesised_runs_are_flagged(self):
        """A hazard we invented must never read as observed truth."""
        city = self._city_with_road()
        furnish_power(city)
        assert not city.power[0].sourced

    def test_poles_are_added_as_landmarks(self):
        """Poles are what the camera pipeline actually detects."""
        city = self._city_with_road()
        furnish_power(city)
        assert sum(1 for f in city.furniture if f.kind == "pole") >= 3

    def test_short_roads_are_skipped(self):
        city = OsmCity(extent_m=300)
        city.roads.append([(0.0, 0.0), (0.0, 20.0)])
        assert furnish_power(city) == 0

    def test_roads_near_a_real_line_are_left_alone(self):
        """OSM data wins; we only fill gaps."""
        city = self._city_with_road()
        city.power.append(
            PowerRun(points=((0.0, -100.0), (0.0, 100.0)), height=9.0, sourced=True)
        )
        assert furnish_power(city) == 0

    def test_the_line_is_offset_from_the_road_centre(self):
        """Poles stand on the verge, not in the carriageway."""
        city = self._city_with_road()
        furnish_power(city, offset_m=5.0)
        assert all(abs(n) > 3.0 for n, _ in city.power[0].points)

    def test_rejects_invalid_parameters(self):
        with pytest.raises(ValueError):
            furnish_power(self._city_with_road(), span_m=0)

    def test_runs_convert_to_sagging_conductors(self):
        city = self._city_with_road()
        furnish_power(city, span_m=40.0)
        field = power_obstacles(city)
        assert field.cables
        cable = field.cables[0]
        lowest = max(q.down for q in cable.points())
        assert lowest > cable.start.down, "midspan must hang below the poles"
