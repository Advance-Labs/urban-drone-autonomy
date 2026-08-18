#!/usr/bin/env python3
"""Generate one flight recording per showcase scenario.

Each scenario is chosen to exercise a different capability, so the selector
shows genuinely different problems rather than the same map recoloured:

    grid        clean procedural city -- moving traffic and wire inference
    princessanne low-rise residential -- threading under distribution wire
    downtown    292 m towers -- must route AROUND, not over
    northyork   urban surveillance with street-level targets
    scarborough people and vehicles as regulatory volumes

Writes `showcase/flights/<key>.json`. Re-runs are cheap; OSM data is cached.
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autonomy import (  # noqa: E402
    Cable, City, GridPlanner, MissionRunner, SafetyLimits, SafetyMonitor,
    SafetyVerdict, SurveillanceTarget, Takeoff, Vec3, build_patrol,
    city_block_grid, crossing_traffic, DynamicField, Camera,
    HazardField, RegulatoryProfile, car_on_road, pedestrian_on_sidewalk,
)
from autonomy.backends import KinematicBackend  # noqa: E402
from autonomy.mapping import PerceptionPipeline  # noqa: E402
from autonomy.osm import build_city, free_launch, furnish_power, power_obstacles  # noqa: E402
from autonomy.patrol import NavigateTo, PlannedReturn  # noqa: E402
from autonomy.perception import DetectorModel, ObjectClass  # noqa: E402
from autonomy.sensors import SensorSuite  # noqa: E402
from autonomy.thin_obstacles import ThinObstacleField  # noqa: E402

OUT = Path(__file__).resolve().parent / "flights"


class Permissive(SafetyMonitor):
    """Showcase flights run their planned course; safety is exercised in tests."""

    def evaluate(self, telemetry):
        return SafetyVerdict.ok()


def osm(lat, lon, name, radius=350, furnish=True):
    city = build_city(lat=lat, lon=lon, radius_m=radius, name=name)
    wires = ThinObstacleField()
    if furnish:
        furnish_power(city, max_runs=18)
        wires = power_obstacles(city)
    return city, wires


def thin_polyline(points, tolerance=1.5):
    """Ramer-Douglas-Peucker on a 2D polyline.

    OSM road and sidewalk geometry carries survey-grade vertex density. At the
    zoom this viewer supports, a metre of deviation is sub-pixel, and those
    polylines are decorative context rather than anything the planner reads.
    """
    if len(points) < 3:
        return points
    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        first, last = stack.pop()
        ax, ay = points[first]
        bx, by = points[last]
        dx, dy = bx - ax, by - ay
        span = dx * dx + dy * dy
        worst, worst_i = 0.0, -1
        for i in range(first + 1, last):
            px, py = points[i]
            if span < 1e-12:
                d = math.hypot(px - ax, py - ay)
            else:
                t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / span))
                d = math.hypot(px - (ax + t * dx), py - (ay + t * dy))
            if d > worst:
                worst, worst_i = d, i
        if worst > tolerance and worst_i > 0:
            keep[worst_i] = True
            stack.append((first, worst_i))
            stack.append((worst_i, last))
    return [p for p, k in zip(points, keep) if k]


def slim_city(city_dict: dict) -> dict:
    """Trim map geometry to what the viewer can actually render."""
    for key in ("roads", "sidewalks"):
        city_dict[key] = [
            [[round(n, 1), round(e, 1)] for n, e in thin_polyline(way)]
            for way in city_dict.get(key, [])
        ]
    for f in city_dict.get("footprints", []):
        f["pts"] = [[round(n, 1), round(e, 1)] for n, e in thin_polyline(f["pts"], 0.8)]
    return city_dict


def slim_wires(cables):
    """Export a display-resolution catenary.

    Cables carry nine points because collision checking wants that resolution.
    Drawing does not -- five is smooth at any zoom this viewer reaches, and
    `true_wires` was 35% of the Princess Anne Manor record at nine.
    """
    out = []
    for c in cables:
        pts = c.points()
        pts = pts[::2] if len(pts) >= 5 else pts
        out.append({
            "name": c.name,
            "points": [{"n": round(q.north, 1), "e": round(q.east, 1),
                        "d": round(q.down, 1)} for q in pts],
        })
    return out


def bind_wires(planner, wires):
    """Give the planner a fixed conductor map without threading it through
    every call site."""
    original = planner.plan
    planner.plan = lambda *a, **k: original(*a, **{**k, "thin": wires})
    return planner


async def run(behaviors, city, *, wires=None, alt=20.0, endurance=9000,
              speed=6.0, hazards=None, coverage=None, perception=None,
              timeout=3000, sensors=None, home=None):
    vehicle = KinematicBackend(
        max_speed_mps=speed * 1.4, battery_endurance_s=endurance, city=city,
        start=home,
    )
    runner = MissionRunner(
        vehicle, behaviors, safety=Permissive(), verbose=False,
        timeout_s=timeout, city=city, hazards=hazards, coverage=coverage,
        perception=perception, sensors=sensors,
    )
    result = await runner.run()
    # 0.4 m keeps every turn the eye can see while dropping the collinear
    # points a straight transit leg accumulates; 1 decimal is decimetres.
    payload = result.to_dict(simplify_m=0.4, precision=1)
    payload["context"]["city"] = slim_city(city.to_dict())
    if wires and wires.cables:
        payload["context"]["true_wires"] = slim_wires(wires.cables)
    return payload


# -- scenarios -------------------------------------------------------------

async def scenario_grid():
    """Procedural city: moving traffic plus camera-inferred powerlines."""
    city = city_block_grid(blocks=4, seed=7)
    planner = GridPlanner(city, cell_m=6, margin_m=4)
    dynamic = DynamicField(obstacles=[
        crossing_traffic(north=40, east_span=city.extent_m * 1.5,
                         altitude_m=20, speed_mps=9.0),
        crossing_traffic(north=-70, east_span=city.extent_m * 1.2,
                         altitude_m=28, speed_mps=6.0, name="traffic-2"),
    ])
    targets = [
        SurveillanceTarget(position=Vec3(-60, 90, -2.0), name="T1"),
        SurveillanceTarget(position=Vec3(120, -40, -2.0), name="T2"),
        SurveillanceTarget(position=Vec3(30, 150, -2.0), name="T3"),
    ]
    behaviors, coverage, _ = build_patrol(
        city, targets, camera=Camera(), planner=planner,
        speed_mps=6.0, observe_altitude_m=20.0, dynamic=dynamic,
    )
    thin = ThinObstacleField()
    payload = await run(
        [Takeoff(20.0), *behaviors, PlannedReturn(planner, speed_mps=6.0)],
        city, wires=thin, coverage=coverage, endurance=4000,
    )
    payload["context"]["movers"] = [
        {"name": o.name, "radius": o.radius_m, "speed": o.speed_mps, "loop": o.loop,
         "waypoints": [{"n": w.north, "e": w.east, "d": w.down} for w in o.waypoints]}
        for o in dynamic.obstacles
    ]
    return payload


async def scenario_princessanne():
    """Low-rise residential: a survey at 5 m AGL, under distribution wire."""
    city, wires = osm(43.6800, -79.5300, "Princess Anne Manor")
    planner = bind_wires(
        GridPlanner(city, cell_m=6, margin_m=3.5, min_altitude_m=3.0,
                    max_altitude_m=45, thin_margin_m=2.5, tracking_error_m=1.5),
        wires,
    )
    box = [Vec3(-120, -120, -5), Vec3(-120, 120, -5), Vec3(-40, 120, -5),
           Vec3(-40, -120, -5), Vec3(40, -120, -5), Vec3(40, 120, -5)]
    legs = [NavigateTo(w, planner, speed_mps=4.0, label=f"leg {i + 1}")
            for i, w in enumerate(box)]
    return await run(
        [Takeoff(5.0), *legs, PlannedReturn(planner, speed_mps=4.0)],
        city, wires=wires, speed=4.0, timeout=3000,
    )


async def scenario_downtown():
    """292 m towers: the ceiling forces routing AROUND rather than over."""
    city, wires = osm(43.6510, -79.3810, "Downtown Toronto", furnish=False)
    planner = GridPlanner(city, cell_m=8, margin_m=6.0, min_altitude_m=15,
                          max_altitude_m=90, tracking_error_m=2.0)
    home = free_launch(city, margin_m=8.0)
    legs = [
        NavigateTo(Vec3(180, -160, -60), planner, speed_mps=8.0, label="north-west"),
        NavigateTo(Vec3(-140, 170, -60), planner, speed_mps=8.0, label="south-east"),
        NavigateTo(Vec3(160, 160, -60), planner, speed_mps=8.0, label="north-east"),
    ]
    return await run(
        [Takeoff(60.0), *legs,
         PlannedReturn(planner, speed_mps=8.0, home=home)],
        city, wires=wires, speed=8.0, endurance=6000, home=home,
    )


async def scenario_northyork():
    """Urban surveillance: observe street-level targets with real occlusion."""
    city, wires = osm(43.7680, -79.4130, "North York Centre")
    planner = bind_wires(
        GridPlanner(city, cell_m=7, margin_m=5.0, min_altitude_m=10,
                    max_altitude_m=80, thin_margin_m=4.0),
        wires,
    )
    import random
    rng = random.Random(11)
    targets = []
    for i in range(5):
        for _ in range(200):
            p = Vec3(rng.uniform(-260, 260), rng.uniform(-260, 260), -2.0)
            if city.is_free(p, 6.0):
                targets.append(SurveillanceTarget(position=p, name=f"T{i + 1}"))
                break
    behaviors, coverage, missed = build_patrol(
        city, targets, camera=Camera(range_m=160), planner=planner,
        speed_mps=6.0, standoff_m=55.0, observe_altitude_m=45.0,
    )
    return await run(
        [Takeoff(45.0), *behaviors, PlannedReturn(planner, speed_mps=6.0)],
        city, wires=wires, coverage=coverage, endurance=7000,
    )


async def scenario_scarborough():
    """People and vehicles: a legal keep-out volume, not an obstacle."""
    city, wires = osm(43.7730, -79.2580, "Scarborough")
    planner = bind_wires(
        GridPlanner(city, cell_m=6, margin_m=4.0, min_altitude_m=8,
                    max_altitude_m=70, thin_margin_m=4.0,
                    hazard_weight=3.0, tracking_error_m=3.5),
        wires,
    )
    home = free_launch(city, margin_m=6.0)
    hazards = HazardField(
        vehicles=[car_on_road(north=70, east_start=-140, east_end=140, speed_mps=11.0)],
        pedestrians=[pedestrian_on_sidewalk(north=78, east_start=-60, east_end=60)],
        rules=RegulatoryProfile.advanced(),
    )
    legs = [
        NavigateTo(Vec3(190, 30, -22), planner, speed_mps=6.0,
                   label="across the corridor", hazards=hazards),
        NavigateTo(Vec3(-40, -150, -22), planner, speed_mps=6.0,
                   label="return leg", hazards=hazards),
    ]
    return await run(
        [Takeoff(22.0), *legs,
         PlannedReturn(planner, speed_mps=6.0, hazards=hazards, home=home)],
        city, wires=wires, hazards=hazards, endurance=6000, home=home,
    )


SCENARIOS = {
    "grid": scenario_grid,
    "princessanne": scenario_princessanne,
    "downtown": scenario_downtown,
    "northyork": scenario_northyork,
    "scarborough": scenario_scarborough,
}


async def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    only = sys.argv[1:] or list(SCENARIOS)
    for key in only:
        if key not in SCENARIOS:
            print(f"unknown scenario {key!r}", file=sys.stderr)
            return 2
        payload = await SCENARIOS[key]()
        path = OUT / f"{key}.json"
        path.write_text(json.dumps(payload, separators=(",", ":")))
        kb = path.stat().st_size / 1024
        ctx = payload["context"]
        city = ctx.get("city", {})
        print(
            f"{key:<12} {len(payload['track']):>5} samples  "
            f"{len(city.get('footprints') or city.get('buildings') or []):>4} bldgs  "
            f"{payload['duration_s']:>6.0f}s  {kb:>7.0f} KB  "
            f"{'completed' if payload['completed'] else 'aborted'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
