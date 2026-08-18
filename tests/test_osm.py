"""OSM import: polygon footprints, height estimation, and projection.

Network-free: these exercise the conversion and geometry against a canned
Overpass payload. The live fetch is covered by a separately marked test.
"""

from __future__ import annotations

import json

import pytest

from autonomy.osm import (
    DEFAULT_HEIGHTS,
    Footprint,
    OsmCity,
    _area,
    _height_for,
    _to_local,
    build_city,
    road_segment,
)
from autonomy.types import Vec3


def square(size: float = 20.0, height: float = 10.0, n0: float = 0.0, e0: float = 0.0):
    return Footprint(
        points=((n0, e0), (n0 + size, e0), (n0 + size, e0 + size), (n0, e0 + size)),
        height=height, name="sq",
    )


class TestFootprintGeometry:
    def test_rejects_degenerate_outlines(self):
        with pytest.raises(ValueError):
            Footprint(points=((0, 0), (1, 1)), height=5)
        with pytest.raises(ValueError):
            Footprint(points=((0, 0), (1, 0), (1, 1)), height=0)

    def test_point_inside_is_contained(self):
        assert square().contains(Vec3(10, 10, -5))

    def test_point_outside_is_not(self):
        assert not square().contains(Vec3(50, 50, -5))

    def test_above_the_roof_is_clear(self):
        assert not square(height=10).contains(Vec3(10, 10, -30))

    def test_margin_extends_the_footprint(self):
        f = square()
        just_out = Vec3(-3, 10, -5)
        assert not f.contains(just_out)
        assert f.contains(just_out, margin=5.0)

    def test_concave_shapes_are_handled(self):
        """Real footprints are L-shaped. A bounding box would wrongly fill the
        notch, closing a gap that is actually flyable."""
        L = Footprint(
            points=((0, 0), (30, 0), (30, 10), (10, 10), (10, 30), (0, 30)),
            height=10, name="L",
        )
        assert L.contains(Vec3(5, 5, -5)), "inside the arm"
        assert not L.contains(Vec3(25, 25, -5)), "in the notch, must be free"

    def test_area_is_correct(self):
        assert _area([(0, 0), (10, 0), (10, 10), (0, 10)]) == pytest.approx(100.0)


class TestHeightEstimation:
    def test_explicit_height_is_used_and_flagged_as_data(self):
        h, estimated = _height_for({"height": "24.5", "building": "yes"})
        assert h == pytest.approx(24.5) and not estimated

    def test_height_with_unit_suffix_parses(self):
        h, estimated = _height_for({"height": "18 m"})
        assert h == pytest.approx(18.0) and not estimated

    def test_levels_convert_to_metres(self):
        h, estimated = _height_for({"building:levels": "5"})
        assert h > 12.0 and not estimated

    def test_missing_height_falls_back_by_type_and_is_flagged(self):
        h, estimated = _height_for({"building": "apartments"})
        assert h == DEFAULT_HEIGHTS["apartments"] and estimated

    def test_unknown_type_still_gets_a_height(self):
        h, estimated = _height_for({"building": "something_unheard_of"})
        assert h > 0 and estimated

    def test_garbage_height_falls_back_rather_than_crashing(self):
        h, estimated = _height_for({"height": "tall", "building": "house"})
        assert h == DEFAULT_HEIGHTS["house"] and estimated

    def test_apartments_are_taller_than_houses(self):
        """Getting this backwards plans a route through a building."""
        assert DEFAULT_HEIGHTS["apartments"] > DEFAULT_HEIGHTS["house"]


class TestProjection:
    def test_origin_maps_to_zero(self):
        n, e = _to_local(43.6800, -79.5300, 43.6800, -79.5300)
        assert n == pytest.approx(0.0) and e == pytest.approx(0.0)

    def test_north_offset_is_positive_north(self):
        n, e = _to_local(43.6900, -79.5300, 43.6800, -79.5300)
        assert n > 1000 and abs(e) < 1.0

    def test_east_offset_shrinks_with_latitude(self):
        """Longitude degrees narrow toward the poles; ignoring that stretches
        a city east-west."""
        _, e_equator = _to_local(0.0, 0.01, 0.0, 0.0)
        _, e_toronto = _to_local(43.8, -79.36, 43.8, -79.37)
        assert e_toronto < e_equator


CANNED = {
    "elements": [
        {
            "type": "way", "id": 1,
            "tags": {"building": "apartments"},
            "geometry": [
                {"lat": 43.6800, "lon": -79.5300},
                {"lat": 43.6804, "lon": -79.5300},
                {"lat": 43.6804, "lon": -79.5294},
                {"lat": 43.6800, "lon": -79.5294},
                {"lat": 43.6800, "lon": -79.5300},
            ],
        },
        {   # A shed: below the area threshold, should be dropped.
            "type": "way", "id": 2,
            "tags": {"building": "shed"},
            "geometry": [
                {"lat": 43.6795, "lon": -79.5305},
                {"lat": 43.67951, "lon": -79.5305},
                {"lat": 43.67951, "lon": -79.53049},
                {"lat": 43.6795, "lon": -79.5305},
            ],
        },
        {
            "type": "way", "id": 3,
            "tags": {"highway": "residential", "name": "Test Rd"},
            "geometry": [
                {"lat": 43.6790, "lon": -79.5315},
                {"lat": 43.6790, "lon": -79.5275},
            ],
        },
    ]
}


@pytest.fixture
def canned_city(tmp_path, monkeypatch):
    """A city built from a canned payload, with the network hard-blocked.

    Overpass is a shared free service; a test suite that reaches it is both
    flaky and rude. Blocking urlopen makes a cache-key mismatch fail loudly
    here instead of silently hammering a public endpoint -- which is exactly
    what happened when the key gained an `include_furniture` component.
    """
    import autonomy.osm as osm_module

    def blocked(*args, **kwargs):
        raise AssertionError(
            "test attempted a live Overpass request; the cache key probably "
            "changed and no longer matches the fixture filename"
        )

    monkeypatch.setattr(osm_module.urllib.request, "urlopen", blocked)

    cache = tmp_path / "cache"
    cache.mkdir()
    # Key format must track fetch_osm(): lat_lon_radius_{roads}{furniture}
    (cache / "43.68000_-79.53000_300_11.json").write_text(json.dumps(CANNED))
    return build_city(lat=43.6800, lon=-79.5300, radius_m=300, cache_dir=cache)


class TestBuildCity:
    def test_buildings_and_roads_are_separated(self, canned_city):
        assert len(canned_city.footprints) == 1
        assert len(canned_city.roads) == 1

    def test_tiny_structures_are_dropped(self, canned_city):
        """Sheds and bin stores triple the footprint count while obstructing
        nothing a drone would route around."""
        assert all(f.kind != "shed" for f in canned_city.footprints)

    def test_height_comes_from_the_type_table(self, canned_city):
        f = canned_city.footprints[0]
        assert f.height == DEFAULT_HEIGHTS["apartments"]
        assert f.height_estimated

    def test_duplicate_closing_node_is_dropped(self, canned_city):
        """OSM repeats the first node to close a way; keeping it puts a
        zero-length edge in the polygon."""
        assert len(canned_city.footprints[0].points) == 4

    def test_city_interface_matches_the_procedural_one(self, canned_city):
        """Drop-in for `City`, so the planner needs no changes."""
        for method in ("is_free", "segment_clear", "has_line_of_sight",
                       "tallest", "in_bounds", "occupant", "buildings"):
            assert hasattr(canned_city, method)

    def test_collision_works_through_the_city_interface(self, canned_city):
        f = canned_city.footprints[0]
        n, e = f.centroid
        assert not canned_city.is_free(Vec3(n, e, -3.0))
        assert canned_city.is_free(Vec3(n, e, -f.height - 5.0))

    def test_road_segment_returns_endpoints(self, canned_city):
        seg = road_segment(canned_city)
        assert seg is not None
        assert (seg[1] - seg[0]).norm > 100

    def test_road_segment_is_none_without_roads(self):
        assert road_segment(OsmCity()) is None

    def test_serialises_as_polygons(self, canned_city):
        d = canned_city.to_dict()
        assert d["footprints"] and len(d["footprints"][0]["pts"]) == 4
        assert "est" in d["footprints"][0], "viewer needs to know if height is a guess"

    def test_stats_report_how_much_was_guessed(self, canned_city):
        s = canned_city.stats()
        assert s["heights_estimated"] == 1
        assert s["heights_from_data"] == 0


class TestValidation:
    def test_negative_radius_is_rejected(self):
        from autonomy.osm import fetch_osm

        with pytest.raises(ValueError):
            fetch_osm(lat=0, lon=0, radius_m=0)
