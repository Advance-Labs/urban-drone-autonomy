#!/usr/bin/env python3
"""CLI for flying missions in sim.

    python run.py survey
    python run.py survey --width 60 --height 40 --altitude 25
    python run.py orbit --radius 20
    python run.py hover

    # wind, emergency landing sites, and a flight record for the 3D viewer
    python run.py survey --wind 5,2 --sites --log flight.json

    python run.py survey --backend px4      # needs PX4 SITL + `pip install mavsdk`

Backends are interchangeable: the mission definitions below never mention
which one is in use.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

from autonomy import (
    Camera,
    DynamicField,
    MovingObstacle,
    SensorSuite,
    ThinObstacleField,
    crossing_traffic,
    powerline_corridor,
    FollowPath,
    GridPlanner,
    Hold,
    SurveillanceTarget,
    build_patrol,
    city_block_grid,
    LandingSite,
    LandingSiteSelector,
    MissionRunner,
    ReturnToLaunch,
    SafetyLimits,
    SafetyMonitor,
    Takeoff,
    Vec3,
    estimated_duration_s,
    orbit,
    path_length,
    survey_grid,
)
from autonomy.backends import KinematicBackend


def parse_wind(raw: str | None) -> Vec3:
    """`--wind 5,2` means 5 m/s toward north, 2 m/s toward east."""
    if not raw:
        return Vec3()
    try:
        north, east = (float(part) for part in raw.split(","))
    except ValueError as exc:
        raise SystemExit(f"--wind expects 'north,east', got {raw!r}") from exc
    return Vec3(north, east, 0.0)


def demo_sites() -> list[LandingSite]:
    """A stand-in set of pre-surveyed emergency sites.

    On a real flight these come from the operator walking the area beforehand.
    Quality is a human judgement the vehicle cannot make on its own.
    """
    return [
        LandingSite(position=Vec3(60, 40, 0), name="school field", quality=1.0, radius_m=25),
        LandingSite(position=Vec3(-40, 80, 0), name="car park", quality=0.55, radius_m=15),
        LandingSite(position=Vec3(120, -60, 0), name="road shoulder", quality=0.2, radius_m=8),
        # Spread further out so a survey that ranges well beyond the launch
        # point still has something in reach. Real sites come from walking the
        # area; these exist so the divert path is exercisable from the CLI.
        LandingSite(position=Vec3(320, 120, 0), name="north pasture", quality=0.9, radius_m=30),
        LandingSite(position=Vec3(180, 300, 0), name="east clearing", quality=0.7, radius_m=20),
    ]


def build_city_patrol(args: argparse.Namespace):
    """Urban surveillance: generate a city, scatter targets, plan a patrol.

    Returns (behaviours, city, coverage) so the runner can score observation
    and the viewer can draw the buildings.
    """
    if args.osm:
        from autonomy.osm import build_city as build_osm_city
        try:
            lat, lon = (float(x) for x in args.osm.split(","))
        except ValueError as exc:
            raise SystemExit(f"--osm expects 'lat,lon', got {args.osm!r}") from exc
        city = build_osm_city(lat=lat, lon=lon, radius_m=args.osm_radius)
        st = city.stats()
        print(
            f"osm: {st['buildings']} buildings, {st['roads']} roads, "
            f"tallest {st['tallest_m']:.0f} m "
            f"({st['heights_estimated']} heights estimated, "
            f"{st['heights_from_data']} from data)"
        )
    else:
        city = city_block_grid(blocks=args.blocks, seed=args.seed)
    planner = GridPlanner(city, cell_m=args.cell, margin_m=args.margin)
    camera = Camera(fov_deg=args.fov, range_m=args.cam_range)

    # Targets at street level between blocks, which is the hard case: a
    # rooftop target is visible from anywhere above, a doorway is not.
    rng = random.Random(args.seed + 1)
    targets = []
    for i in range(args.targets):
        for _ in range(60):  # rejection-sample a spot outside the buildings
            p = Vec3(
                rng.uniform(-city.extent_m * 0.8, city.extent_m * 0.8),
                rng.uniform(-city.extent_m * 0.8, city.extent_m * 0.8),
                -2.0,
            )
            if city.is_free(p, 2.0):
                targets.append(SurveillanceTarget(position=p, name=f"T{i + 1}"))
                break

    dynamic = DynamicField(
        obstacles=[
            crossing_traffic(north=40, east_span=city.extent_m * 1.5,
                             altitude_m=args.altitude, speed_mps=9.0),
            crossing_traffic(north=-70, east_span=city.extent_m * 1.2,
                             altitude_m=args.altitude + 8, speed_mps=6.0,
                             name="traffic-2"),
        ] if args.movers else []
    )
    hazards = None
    if args.hazards:
        from autonomy.hazards import (
            HazardField, RegulatoryProfile, car_on_road, pedestrian_on_sidewalk,
        )
        rules = (RegulatoryProfile.basic() if args.hazards == "basic"
                 else RegulatoryProfile.advanced())
        reach = city.extent_m * 0.5
        hazards = HazardField(
            vehicles=[car_on_road(north=reach * 0.5, east_start=-reach, east_end=reach)],
            pedestrians=[pedestrian_on_sidewalk(
                north=reach * 0.5 + 7, east_start=-reach * 0.4, east_end=reach * 0.4)],
            rules=rules,
        )
        print(f"  hazards: 1 car, 1 pedestrian, {rules.name} rules "
              f"({rules.person_radius_m:.0f} m separation)")

    sensors = None
    if args.wires:
        thin = ThinObstacleField(
            cables=tuple(powerline_corridor(
                north=0.0, east_start=-city.extent_m, east_end=city.extent_m,
                height_m=9.0, spans=4,
            ))
        )
        sensors = SensorSuite(city=city, thin=thin)

    behaviors, coverage, unobservable = build_patrol(
        city, targets,
        camera=camera,
        planner=planner,
        speed_mps=args.speed,
        standoff_m=args.standoff,
        observe_altitude_m=args.altitude,
        dynamic=dynamic,
        hazards=hazards,
    )

    print(
        f"city: {len(city.buildings)} buildings, tallest {city.tallest():.0f} m, "
        f"{len(targets)} targets"
    )
    if unobservable:
        print(f"  no vantage found for: {', '.join(unobservable)}")
    if not behaviors:
        raise SystemExit("no observable targets -- try --standoff or --targets")

    if args.movers:
        print(f"  {len(dynamic.obstacles)} moving obstacles")
    return (
        [Takeoff(args.altitude), *behaviors, ReturnToLaunch()],
        city, coverage, dynamic, sensors, hazards,
    )


def build_mission(args: argparse.Namespace) -> list:
    """Translate CLI arguments into a behaviour sequence."""
    if args.command == "hover":
        return [Takeoff(args.altitude), Hold(args.duration), ReturnToLaunch()]

    if args.command == "survey":
        waypoints = survey_grid(
            width_m=args.width,
            height_m=args.height,
            spacing_m=args.spacing,
            altitude_m=args.altitude,
        )
    elif args.command == "orbit":
        waypoints = orbit(
            centre=Vec3(args.radius * 1.5, 0.0, 0.0),
            radius_m=args.radius,
            altitude_m=args.altitude,
            points=args.points,
        )
    else:  # pragma: no cover - argparse rejects anything else
        raise ValueError(f"unknown command {args.command}")

    print(
        f"planned {len(waypoints)} waypoints, {path_length(waypoints):.0f} m, "
        f"~{estimated_duration_s(waypoints, args.speed) / 60:.1f} min at {args.speed:.0f} m/s"
    )
    return [
        Takeoff(args.altitude),
        FollowPath(waypoints, speed_mps=args.speed),
        ReturnToLaunch(),
    ]


def build_backend(args: argparse.Namespace, city=None):
    if args.backend == "kinematic":
        return KinematicBackend(
            max_speed_mps=args.speed * 1.4,
            battery_endurance_s=args.endurance,
            wind=parse_wind(args.wind),
            city=city,
        )
    from autonomy.backends import Px4Backend

    return Px4Backend(args.connect)


async def main() -> int:
    parser = argparse.ArgumentParser(description="Fly an autonomy mission in sim.")
    parser.add_argument(
        "command",
        choices=["survey", "orbit", "hover", "patrol"],
        help="mission type; patrol flies urban surveillance around obstacles",
    )
    parser.add_argument("--backend", choices=["kinematic", "px4"], default="kinematic")
    parser.add_argument("--connect", default="udpin://0.0.0.0:14540", help="PX4 address")
    parser.add_argument("--altitude", type=float, default=20.0, help="metres AGL")
    parser.add_argument("--speed", type=float, default=6.0, help="cruise m/s")
    parser.add_argument("--width", type=float, default=60.0, help="survey east extent")
    parser.add_argument("--height", type=float, default=40.0, help="survey north extent")
    parser.add_argument("--spacing", type=float, default=12.0, help="leg spacing")
    parser.add_argument("--radius", type=float, default=20.0, help="orbit radius")
    parser.add_argument("--points", type=int, default=16, help="orbit waypoints")
    parser.add_argument("--duration", type=float, default=30.0, help="hover seconds")
    parser.add_argument("--endurance", type=float, default=900.0, help="sim hover seconds")
    parser.add_argument("--geofence", type=float, default=200.0, help="radius limit, m")
    parser.add_argument("--wind", help="steady wind as 'north,east' in m/s")
    parser.add_argument(
        "--sites", action="store_true", help="load demo emergency landing sites"
    )
    parser.add_argument("--log", help="write a flight record JSON for the 3D viewer")
    parser.add_argument("--realtime", action="store_true", help="run at wall-clock speed")
    # patrol-only
    parser.add_argument("--blocks", type=int, default=4, help="city grid size")
    parser.add_argument("--seed", type=int, default=7, help="city generator seed")
    parser.add_argument("--targets", type=int, default=5, help="surveillance targets")
    parser.add_argument("--cell", type=float, default=6.0, help="planner grid, m")
    parser.add_argument("--margin", type=float, default=4.0, help="building standoff, m")
    parser.add_argument("--standoff", type=float, default=45.0, help="observation range, m")
    parser.add_argument("--fov", type=float, default=62.0, help="camera FOV, degrees")
    parser.add_argument("--cam-range", type=float, default=140.0, help="camera range, m")
    parser.add_argument("--movers", action="store_true", help="add moving obstacles")
    parser.add_argument("--wires", action="store_true", help="add powerlines + sensors")
    parser.add_argument("--osm", help="build the city from OpenStreetMap: 'lat,lon'")
    parser.add_argument("--osm-radius", type=float, default=350.0, help="OSM radius, m")
    parser.add_argument(
        "--hazards", choices=["basic", "advanced"],
        help="add a car and a pedestrian under the given regulatory profile",
    )
    args = parser.parse_args()

    city = None
    coverage = None
    dynamic = None
    sensors = None
    hazards = None
    if args.command == "patrol":
        behaviors, city, coverage, dynamic, sensors, hazards = build_city_patrol(args)
        # An urban patrol ranges much further than the default fence allows.
        args.geofence = max(args.geofence, city.extent_m * 1.6)
    else:
        behaviors = build_mission(args)

    sites = LandingSiteSelector(
        sites=demo_sites() if args.sites else [], cruise_speed_mps=args.speed
    )
    runner = MissionRunner(
        vehicle=build_backend(args, city=city),
        behaviors=behaviors,
        safety=SafetyMonitor(
            SafetyLimits(
                cruise_speed_mps=args.speed,
                max_radius_m=args.geofence,
                max_altitude_m=max(100.0, (city.tallest() + 30) if city else 100.0),
            ),
            landing_sites=sites,
        ),
        realtime=args.realtime,
        coverage=coverage,
        city=city,
        dynamic=dynamic,
        sensors=sensors,
        hazards=hazards,
    )

    result = await runner.run()

    if coverage is not None:
        s = coverage.summary()
        print(
            f"\ncoverage {s['fraction']:.0%}  "
            f"covered={','.join(s['covered']) or '-'}  "
            f"missed={','.join(s['missed']) or '-'}"
        )

    if args.log:
        path = Path(args.log)
        path.write_text(json.dumps(result.to_dict(), indent=1))
        print(f"\nflight record -> {path}  ({len(result.track)} samples)")
        print(f"open viewer/index.html and load it, or: python viewer/serve.py {path}")

    print()
    if result.completed:
        print(f"Mission complete in {result.duration_s:.0f}s.")
        return 0
    print(f"Mission aborted after {result.duration_s:.0f}s: {result.aborted_reason}")
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
