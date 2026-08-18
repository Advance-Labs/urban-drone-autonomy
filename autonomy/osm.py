"""Build a city model from OpenStreetMap.

Real building footprints, real streets, real heights where they are tagged.
Free, global, and good enough to fly against — which matters, because a
procedural block grid is a fine unit test and a poor rehearsal.

## The height problem

Most OSM buildings carry no height. In a probe of North York, zero of the
first twelve had `height` or `building:levels`. So height is estimated from
the `building` tag: a `house` is not a `retail` is not an `apartments`. That
is a guess, and it is stated as one — `Footprint.height_estimated` records
whether the number came from data or from a table, so a mission planner can
treat estimated heights with the caution they deserve.

Getting this wrong is not cosmetic. Plan a route at 15 m over a block of
`apartments` assumed to be 6 m and you have planned a collision.

## Footprints are polygons, not boxes

The procedural city uses axis-aligned boxes, which is right for testing a
planner. Real footprints are L-shaped, angled to the street, and often
concave. Approximating them by bounding boxes inflates the built area
substantially and closes street canyons that are actually flyable, so this
module keeps the polygon and does point-in-polygon collision.

## Caching

Overpass is a shared free service. Results are cached to disk on first fetch
and reused, so repeated runs cost one request. Be a good citizen: cache, do
not hammer, and keep bounding boxes small.
"""

from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .spatial import SpatialHash
from .types import Vec3
from .world import City

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Metres per storey, including floor structure. 3.2 m is a reasonable
# residential/commercial average; industrial single storeys are much taller.
STOREY_M = 3.2

# Fallback heights by `building` tag, in metres. Deliberately conservative --
# over-estimating a building costs a slightly higher route, under-estimating
# one costs the aircraft.
DEFAULT_HEIGHTS: dict[str, float] = {
    "house": 7.0,
    "detached": 7.0,
    "semidetached_house": 7.0,
    "bungalow": 4.5,
    "terrace": 8.0,
    "residential": 10.0,
    "apartments": 18.0,
    "commercial": 12.0,
    "retail": 8.0,
    "office": 20.0,
    "industrial": 10.0,
    "warehouse": 10.0,
    "school": 9.0,
    "university": 15.0,
    "hospital": 20.0,
    "church": 14.0,
    "garage": 3.0,
    "garages": 3.0,
    "shed": 3.0,
    "roof": 3.5,
    "yes": 8.0,  # untyped: the most common tag, and the least informative
}

EARTH_R = 6_378_137.0


@dataclass(frozen=True, slots=True)
class Footprint:
    """A building outline with a height, in local NED metres."""

    points: tuple[tuple[float, float], ...]  # (north, east) pairs
    height: float
    name: str = "building"
    kind: str = "yes"
    height_estimated: bool = True

    def __post_init__(self) -> None:
        if len(self.points) < 3:
            raise ValueError(f"{self.name}: a footprint needs at least 3 points")
        if self.height <= 0:
            raise ValueError(f"{self.name}: height must be positive")

    @property
    def centroid(self) -> tuple[float, float]:
        n = sum(p[0] for p in self.points) / len(self.points)
        e = sum(p[1] for p in self.points) / len(self.points)
        return n, e

    def contains(self, p: Vec3, margin: float = 0.0) -> bool:
        """Point-in-polygon with an outward margin.

        Ray casting for the inside test, then a distance-to-edge check for the
        margin. Doing the margin as a true polygon offset would be correct and
        far more code; edge distance is equivalent everywhere except at sharp
        convex corners, where it is slightly conservative -- the right
        direction to be wrong.
        """
        if p.altitude > self.height + margin or p.altitude < 0.0:
            return False
        if self._inside(p.north, p.east):
            return True
        return margin > 0.0 and self._distance_to_edge(p.north, p.east) <= margin

    def _inside(self, n: float, e: float) -> bool:
        inside = False
        pts = self.points
        j = len(pts) - 1
        for i in range(len(pts)):
            ni, ei = pts[i]
            nj, ej = pts[j]
            if (ei > e) != (ej > e):
                if n < (nj - ni) * (e - ei) / (ej - ei) + ni:
                    inside = not inside
            j = i
        return inside

    def _distance_to_edge(self, n: float, e: float) -> float:
        best = math.inf
        pts = self.points
        for i in range(len(pts)):
            a, b = pts[i], pts[(i + 1) % len(pts)]
            best = min(best, _point_segment_distance(n, e, a[0], a[1], b[0], b[1]))
        return best

    def bounds(self) -> tuple[float, float, float, float]:
        ns = [p[0] for p in self.points]
        es = [p[1] for p in self.points]
        return min(ns), max(ns), min(es), max(es)


def _point_segment_distance(
    pn: float, pe: float, an: float, ae: float, bn: float, be: float
) -> float:
    dn, de = bn - an, be - ae
    length_sq = dn * dn + de * de
    if length_sq < 1e-12:
        return math.hypot(pn - an, pe - ae)
    t = max(0.0, min(1.0, ((pn - an) * dn + (pe - ae) * de) / length_sq))
    return math.hypot(pn - (an + t * dn), pe - (ae + t * de))


# Typical conductor heights. Transmission runs on towers well above rooftops;
# distribution runs on wooden poles at roughly the height of a second storey,
# which is precisely the altitude band a low-flying drone uses.
POWER_HEIGHTS = {"line": 28.0, "minor_line": 9.0}
SUPPORT_HEIGHTS = {"tower": 32.0, "pole": 9.5}


@dataclass(frozen=True, slots=True)
class PowerRun:
    """A conductor route: a polyline of support positions at a given height."""

    points: tuple[tuple[float, float], ...]
    height: float
    kind: str = "minor_line"
    name: str = "line"
    sourced: bool = True  # from OSM, as opposed to synthesised along a road


@dataclass(frozen=True, slots=True)
class PointFeature:
    """A pole, lamp or tree: something small, tall and easy to detect."""

    north: float
    east: float
    height: float
    kind: str
    name: str = ""


@dataclass(slots=True)
class OsmCity:
    """A city built from real footprints. Drop-in for `City` in the planner."""

    footprints: list[Footprint] = field(default_factory=list)
    roads: list[list[tuple[float, float]]] = field(default_factory=list)
    sidewalks: list[list[tuple[float, float]]] = field(default_factory=list)
    power: list[PowerRun] = field(default_factory=list)
    furniture: list[PointFeature] = field(default_factory=list)
    extent_m: float = 300.0
    name: str = "osm"
    origin_lat: float = 0.0
    origin_lon: float = 0.0
    # Built lazily: footprints are appended during construction, so indexing
    # eagerly would index an empty list.
    _index: SpatialHash | None = None
    _indexed_count: int = 0

    # -- City-compatible interface ----------------------------------------

    def _footprint_index(self) -> SpatialHash:
        """Spatial index over footprints, rebuilt if the list has grown.

        Without this every collision query scanned all 273 polygons; a real
        A* spent minutes in point-in-polygon tests that a bucket lookup
        answers in microseconds.
        """
        if self._index is None or self._indexed_count != len(self.footprints):
            index: SpatialHash = SpatialHash(cell_m=30.0)
            for f in self.footprints:
                min_n, max_n, min_e, max_e = f.bounds()
                index.insert(f, min_n, max_n, min_e, max_e)
            self._index = index
            self._indexed_count = len(self.footprints)
        return self._index

    @property
    def buildings(self) -> list[Footprint]:
        """Named to match `City.buildings` so the planner needs no changes."""
        return self.footprints

    def is_free(self, p: Vec3, margin: float = 0.0) -> bool:
        if p.altitude < 0.0:
            return False
        for f in self._footprint_index().query(p.north, p.east, margin):
            if f.contains(p, margin):
                return False
        return True

    def occupant(self, p: Vec3, margin: float = 0.0) -> Footprint | None:
        for f in self._footprint_index().query(p.north, p.east, margin):
            if f.contains(p, margin):
                return f
        return None

    def segment_clear(
        self, a: Vec3, b: Vec3, margin: float = 0.0, step_m: float = 1.5
    ) -> bool:
        span = (b - a).norm
        if span < 1e-9:
            return self.is_free(a, margin)
        steps = max(2, math.ceil(span / step_m))
        for i in range(steps + 1):
            if not self.is_free(a + (b - a) * (i / steps), margin):
                return False
        return True

    def has_line_of_sight(self, eye: Vec3, target: Vec3, step_m: float = 2.0) -> bool:
        span = (target - eye).norm
        if span < 1e-9:
            return True
        steps = max(2, math.ceil(span / step_m))
        index = self._footprint_index()
        for i in range(1, steps):
            p = eye + (target - eye) * (i / steps)
            if any(f.contains(p) for f in index.query(p.north, p.east)):
                return False
        return True

    def tallest(self) -> float:
        return max((f.height for f in self.footprints), default=0.0)

    def in_bounds(self, p: Vec3) -> bool:
        return abs(p.north) <= self.extent_m and abs(p.east) <= self.extent_m

    def to_dict(self) -> dict:
        """Scene data for the viewer, as polygons rather than boxes."""
        return {
            "name": self.name,
            "extent_m": self.extent_m,
            "origin": {"lat": self.origin_lat, "lon": self.origin_lon},
            "footprints": [
                {
                    "h": round(f.height, 1),
                    "est": f.height_estimated,
                    "kind": f.kind,
                    "pts": [[round(n, 1), round(e, 1)] for n, e in f.points],
                }
                for f in self.footprints
            ],
            "roads": [
                [[round(n, 1), round(e, 1)] for n, e in way] for way in self.roads
            ],
            "sidewalks": [
                [[round(n, 1), round(e, 1)] for n, e in way] for way in self.sidewalks
            ],
            "power": [
                {"h": r.height, "kind": r.kind, "sourced": r.sourced,
                 "pts": [[round(n, 1), round(e, 1)] for n, e in r.points]}
                for r in self.power
            ],
            "furniture": [
                {"n": round(f.north, 1), "e": round(f.east, 1),
                 "h": f.height, "kind": f.kind}
                for f in self.furniture
            ],
        }

    def stats(self) -> dict:
        estimated = sum(1 for f in self.footprints if f.height_estimated)
        return {
            "buildings": len(self.footprints),
            "roads": len(self.roads),
            "sidewalks": len(self.sidewalks),
            "power_runs": len(self.power),
            "power_synthesised": sum(1 for r in self.power if not r.sourced),
            "furniture": len(self.furniture),
            "tallest_m": round(self.tallest(), 1),
            "heights_estimated": estimated,
            "heights_from_data": len(self.footprints) - estimated,
        }


def _to_local(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
    """Equirectangular projection to local NED metres.

    Exact enough over a few hundred metres, which is all a flight area is.
    Anything larger would want a proper UTM projection.
    """
    north = math.radians(lat - lat0) * EARTH_R
    east = math.radians(lon - lon0) * EARTH_R * math.cos(math.radians(lat0))
    return north, east


def _height_for(tags: dict) -> tuple[float, bool]:
    """Height in metres, and whether it had to be guessed."""
    raw = tags.get("height")
    if raw:
        try:
            return float(str(raw).replace("m", "").strip()), False
        except ValueError:
            pass
    levels = tags.get("building:levels")
    if levels:
        try:
            return max(2.5, float(levels) * STOREY_M), False
        except ValueError:
            pass
    return DEFAULT_HEIGHTS.get(tags.get("building", "yes"), 8.0), True


def fetch_osm(
    *,
    lat: float,
    lon: float,
    radius_m: float = 300.0,
    cache_dir: Path | None = None,
    timeout_s: float = 60.0,
    include_roads: bool = True,
    include_furniture: bool = True,
) -> dict:
    """Fetch raw OSM elements around a point, caching to disk.

    Overpass is a shared free service. The cache means a repeated run costs
    nothing; keep radii small and do not hammer it.
    """
    if radius_m <= 0:
        raise ValueError("radius must be positive")

    cache_dir = cache_dir or Path(__file__).resolve().parent.parent / ".osm-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = (
        f"{lat:.5f}_{lon:.5f}_{int(radius_m)}"
        f"_{int(include_roads)}{int(include_furniture)}.json"
    )
    cached = cache_dir / key
    if cached.exists():
        return json.loads(cached.read_text())

    # Degrees per metre at this latitude, for the bounding box.
    dlat = math.degrees(radius_m / EARTH_R)
    dlon = math.degrees(radius_m / (EARTH_R * math.cos(math.radians(lat))))
    bbox = f"{lat - dlat},{lon - dlon},{lat + dlat},{lon + dlon}"

    parts = [f'way["building"]({bbox});']
    if include_roads:
        parts.append(
            f'way["highway"~"^(residential|secondary|tertiary|primary|service)$"]({bbox});'
        )
    if include_furniture:
        # OSM tags real power infrastructure. Where a probe finds it, there is
        # no need to invent it -- but coverage is thin, so this is the
        # exception rather than the rule (see `furnish_power`).
        parts += [
            f'way["power"~"^(line|minor_line)$"]({bbox});',
            f'node["power"~"^(pole|tower)$"]({bbox});',
            f'way["highway"="footway"]({bbox});',
            f'node["highway"="street_lamp"]({bbox});',
            f'node["natural"="tree"]({bbox});',
        ]
    query = f"[out:json][timeout:{int(timeout_s)}];\n(\n" + "\n".join(parts) + "\n);\nout body geom;"

    request = urllib.request.Request(
        OVERPASS_URL, data=query.encode(),
        headers={"User-Agent": "drone-lab/0.1 (autonomy research sim)"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(
            f"Overpass request failed: {exc}. The importer needs network access "
            f"on first fetch; afterwards {cached} is reused offline."
        ) from exc

    cached.write_text(json.dumps(payload))
    return payload


def build_city(
    *,
    lat: float,
    lon: float,
    radius_m: float = 300.0,
    cache_dir: Path | None = None,
    name: str = "osm",
    min_area_m2: float = 12.0,
) -> OsmCity:
    """Fetch and convert OSM data into a flyable city model.

    `min_area_m2` drops sheds and bin stores, which triple the footprint count
    while obstructing nothing a drone would route around.
    """
    payload = fetch_osm(lat=lat, lon=lon, radius_m=radius_m, cache_dir=cache_dir)

    footprints: list[Footprint] = []
    roads: list[list[tuple[float, float]]] = []
    sidewalks: list[list[tuple[float, float]]] = []
    power: list[PowerRun] = []
    furniture: list[PointFeature] = []

    for element in payload.get("elements", []):
        tags = element.get("tags", {})

        # Nodes: poles, towers, lamps, trees. These have a position, not a
        # geometry, and they are the landmarks the camera pipeline detects.
        if element.get("type") == "node":
            kind = tags.get("power") or tags.get("highway") or tags.get("natural")
            if not kind:
                continue
            n, e = _to_local(element["lat"], element["lon"], lat, lon)
            height = SUPPORT_HEIGHTS.get(kind, 4.5 if kind == "street_lamp" else 8.0)
            furniture.append(
                PointFeature(north=n, east=e, height=height, kind=kind,
                             name=f"n{element.get('id')}")
            )
            continue

        geometry = element.get("geometry") or []
        if len(geometry) < 2:
            continue
        local = [_to_local(g["lat"], g["lon"], lat, lon) for g in geometry]

        if "power" in tags:
            kind = tags["power"]
            power.append(
                PowerRun(points=tuple(local),
                         height=POWER_HEIGHTS.get(kind, 9.0),
                         kind=kind, name=f"w{element.get('id')}", sourced=True)
            )
            continue

        if tags.get("highway") == "footway":
            sidewalks.append(local)
            continue

        if "building" in tags:
            # OSM closes ways by repeating the first node; drop the duplicate.
            if len(local) > 3 and local[0] == local[-1]:
                local = local[:-1]
            if len(local) < 3 or _area(local) < min_area_m2:
                continue
            height, estimated = _height_for(tags)
            footprints.append(
                Footprint(
                    points=tuple(local), height=height,
                    name=tags.get("name") or f"w{element.get('id')}",
                    kind=tags.get("building", "yes"),
                    height_estimated=estimated,
                )
            )
        elif "highway" in tags:
            roads.append(local)

    return OsmCity(
        footprints=footprints, roads=roads, sidewalks=sidewalks,
        power=power, furniture=furniture,
        extent_m=radius_m, name=name,
        origin_lat=lat, origin_lon=lon,
    )


def _area(points: list[tuple[float, float]]) -> float:
    """Shoelace area in square metres."""
    total = 0.0
    for i in range(len(points)):
        n1, e1 = points[i]
        n2, e2 = points[(i + 1) % len(points)]
        total += n1 * e2 - n2 * e1
    return abs(total) / 2.0


def furnish_power(
    city: OsmCity,
    *,
    height_m: float = 9.0,
    offset_m: float = 5.0,
    span_m: float = 40.0,
    min_road_length_m: float = 60.0,
    max_runs: int | None = None,
) -> int:
    """Synthesise distribution lines along roads that OSM has not mapped.

    OSM power coverage is patchy and biased: transmission is mapped well
    because it is visible from satellite imagery, residential distribution
    badly because it is not. A probe of Princess Anne Manor, Etobicoke returned
    zero lines for a neighbourhood that certainly has one down every street.

    Under-modelling here is the dangerous direction, so streets without a
    tagged line get one: offset to the verge where poles actually stand, at
    distribution height, with poles at a typical span. Every synthesised run
    is flagged `sourced=False`, because a hazard the aircraft "knows about"
    only because we invented it must never be confused with observed truth.

    Returns the number of runs added.
    """
    if height_m <= 0 or span_m <= 0:
        raise ValueError("height and span must be positive")

    def near_existing(way: list[tuple[float, float]]) -> bool:
        """Is this road already served by a mapped line?

        Measured against the line's SEGMENTS, not its vertices. A transmission
        span can be 300 m between towers, so a road running directly beneath
        it sits far from every vertex while being directly under the wire --
        and would get a duplicate synthetic line laid on top of a real one.

        Several points along the road are sampled, because a road can join a
        power corridor partway along its length.
        """
        # Resample rather than using vertices: a straight road is two points,
        # and if it overhangs the line at both ends then neither endpoint is
        # near it while the whole middle runs directly underneath.
        samples = _resample(way, 25.0)
        for probe in samples:
            for run in city.power:
                for i in range(len(run.points) - 1):
                    a, b = run.points[i], run.points[i + 1]
                    if _point_segment_distance(
                        probe[0], probe[1], a[0], a[1], b[0], b[1]
                    ) < 30.0:
                        return True
        return False

    added = 0
    for way in city.roads:
        if max_runs is not None and added >= max_runs:
            break
        if len(way) < 2:
            continue
        length = sum(
            math.hypot(way[i + 1][0] - way[i][0], way[i + 1][1] - way[i][1])
            for i in range(len(way) - 1)
        )
        if length < min_road_length_m or near_existing(way):
            continue

        # Offset perpendicular to the road, to the verge.
        offset = _offset_way(way, offset_m)
        poles = _resample(offset, span_m)
        if len(poles) < 2:
            continue
        city.power.append(
            PowerRun(points=tuple(poles), height=height_m,
                     kind="minor_line", name=f"synth-{added}", sourced=False)
        )
        # Poles are what the camera pipeline actually detects.
        for i, (n, e) in enumerate(poles):
            city.furniture.append(
                PointFeature(north=n, east=e, height=height_m + 0.5,
                             kind="pole", name=f"synth-{added}-p{i}")
            )
        added += 1
    return added


def _offset_way(
    way: list[tuple[float, float]], distance: float
) -> list[tuple[float, float]]:
    """Shift a polyline sideways by `distance`, using per-segment normals."""
    out = []
    for i in range(len(way)):
        a = way[max(0, i - 1)]
        b = way[min(len(way) - 1, i + 1)]
        dn, de = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dn, de)
        if length < 1e-9:
            out.append(way[i])
            continue
        # Left normal of the direction of travel.
        out.append((way[i][0] - de / length * distance,
                    way[i][1] + dn / length * distance))
    return out


def _resample(way: list[tuple[float, float]], spacing: float) -> list[tuple[float, float]]:
    """Even points along a polyline, for pole placement."""
    if len(way) < 2:
        return list(way)
    out = [way[0]]
    carry = 0.0
    for i in range(len(way) - 1):
        a, b = way[i], way[i + 1]
        seg = math.hypot(b[0] - a[0], b[1] - a[1])
        if seg < 1e-9:
            continue
        travelled = spacing - carry
        while travelled <= seg:
            t = travelled / seg
            out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
            travelled += spacing
        carry = (carry + seg) % spacing
    if math.hypot(out[-1][0] - way[-1][0], out[-1][1] - way[-1][1]) > spacing / 3:
        out.append(way[-1])
    return out


def power_obstacles(city: OsmCity, *, sag_ratio: float = 0.03):
    """Convert every power run into avoidable conductors."""
    from .thin_obstacles import Cable, ThinObstacleField

    cables = []
    for run in city.power:
        for i in range(len(run.points) - 1):
            a, b = run.points[i], run.points[i + 1]
            span = math.hypot(b[0] - a[0], b[1] - a[1])
            if span < 1.0:
                continue
            cables.append(
                Cable(
                    start=Vec3(a[0], a[1], -run.height),
                    end=Vec3(b[0], b[1], -run.height),
                    sag_m=max(0.4, span * sag_ratio),
                    name=f"{run.name}-{i}",
                )
            )
    return ThinObstacleField(cables=tuple(cables))


def free_launch(
    city: "OsmCity | City",
    *,
    margin_m: float = 6.0,
    max_radius_m: float = 250.0,
    step_m: float = 8.0,
) -> Vec3:
    """A ground position near the map origin that is clear of buildings.

    An arbitrary lat/lon lands on a rooftop surprisingly often -- of four
    Toronto areas sampled, two put the origin inside a building (a 38 m block
    downtown, and the Scarborough Civic Centre). A vehicle spawned there
    crashes on the first tick, which looks like a planner failure and is not.

    Spirals outward on an Archimedean path so the search covers area evenly
    rather than sweeping one ring at a time.
    """
    if margin_m < 0:
        raise ValueError("margin cannot be negative")

    origin = Vec3(0.0, 0.0, -1.0)
    if city.is_free(origin, margin_m):
        return Vec3(0.0, 0.0, 0.0)

    turns = 0.0
    while True:
        radius = step_m * turns / (2 * math.pi)
        if radius > max_radius_m:
            # Nothing clear nearby: return the origin and let the caller see
            # the collision rather than silently launching somewhere absurd.
            return Vec3(0.0, 0.0, 0.0)
        p = Vec3(radius * math.cos(turns), radius * math.sin(turns), -1.0)
        if city.is_free(p, margin_m):
            return Vec3(p.north, p.east, 0.0)
        turns += 0.6


def road_segment(city: OsmCity, index: int = 0) -> tuple[Vec3, Vec3] | None:
    """Endpoints of a road, for placing traffic and pedestrians realistically.

    Cars belong on roads and pedestrians beside them; scattering either at
    random puts hazards inside buildings and teaches the planner nothing.
    """
    usable = [w for w in city.roads if len(w) >= 2]
    if not usable:
        return None
    way = usable[index % len(usable)]
    return (
        Vec3(way[0][0], way[0][1], 0.0),
        Vec3(way[-1][0], way[-1][1], 0.0),
    )
