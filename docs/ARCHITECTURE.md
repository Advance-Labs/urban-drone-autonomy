# Architecture

The design decisions, and why each one is the way it is. For measured results
see [FINDINGS.md](FINDINGS.md); for how to run things see the
[README](../README.md).

## The layer map

```
        ┌────────────────────────────────────────────┐
        │  run.py                     mission CLI    │
        ├────────────────────────────────────────────┤
        │  mission.py                 control loop   │  safety runs FIRST
        ├──────────────┬──────────────┬──────────────┤
        │  safety.py   │ behaviors.py │  patrol.py   │  veto │ states │ urban
        │  landing.py  ├──────────────┴──────────────┤
        │              │ planner.py   navigation.py  │  geometry │ 3D A*
        ├──────────────┴─────────────────────────────┤
        │  world.py  osm.py  dynamic.py  hazards.py  │  what is out there
        │  thin_obstacles.py  spatial.py             │
        ├────────────────────────────────────────────┤
        │  perception.py ──► mapping.py              │  what we can SEE of it
        ├────────────────────────────────────────────┤
        │  backends/     VehicleBackend              │  ◄── the seam
        ├────────────────────────────────────────────┤
        │  kinematic  │  px4  │  (hardware)          │
        └────────────────────────────────────────────┘
```

## The five decisions that shape everything

### 1. `VehicleBackend` is the seam

Everything above it is written once. The autonomy layer never touches motors —
it sends intent (`goto`, `takeoff`, `land`, `look_at`) exactly the way real
autonomy sends MAVLink to a flight controller.

That is why the same missions run against a kinematic model, PX4 SITL, and
eventually a Pixhawk with no changes above the line. It is verified rather than
asserted: the PX4 SITL run uses the identical behaviour objects.

### 2. Safety evaluates before behaviours, every tick

A behaviour can never sneak one last command in on the cycle where the geofence
trips. Ordering inside `MissionRunner._tick` is the safety-critical part of the
whole system.

### 3. Behaviours are non-blocking

If takeoff were a blocking `await`, nothing could preempt it when the battery
drops. Each `tick()` returns promptly, so the safety monitor always gets its
turn. This is the entire reason for the state-machine shape.

### 4. Hard constraints and soft costs are both modelled

A binary constraint produces a path that hugs the legal minimum exactly, which
is compliant and obviously wrong — nobody deliberately flies 30.1 m from a
pedestrian. So `HazardField` has `blocked_at` (the floor) *and* `risk_cost` (a
gradient), and the planner adds the cost to its move weight.

### 5. Standoff scales with map uncertainty

Three different margins exist, and the difference is the point:

| margin | default | covers |
|---|---|---|
| `margin_m` | 4 m | airframe + position error, around known buildings |
| `thin_margin_m` | 10 m | above, **plus** the triangulation error of an inferred wire |
| `tracking_error_m` | 3.5 m | the gap between the *planned* path and the *flown* one |

A mapped building is where the map says it is. An inferred wire carries the
error of the poles it was derived from. Clearing an obstacle by less than your
uncertainty about it is not clearing it.

## Conventions

**NED throughout.** North `+x`, east `+y`, down `+z`, so altitude is `-down`.
PX4 and ArduPilot both use this internally, so matching it means no frame
conversion at the boundary — and the sign flip lives in exactly one property
(`Vec3.altitude`) rather than being rediscovered as a bug on the first hardware
flight.

**Simulated versus real.** `perception.py` simulates a detector's output
statistics — no network, no weights. `mapping.py` is real code that would run
unchanged against a real model. That split lets the mapping logic, which is
ours to get right, be developed against realistic noise long before a model
exists. Every module states its own scope and limits in its docstring.

**Determinism.** Every generator and stochastic model takes a seed. A mission
that fails must be reproducible; a randomly different city on each run makes a
planner impossible to debug.

**Fail closed.** Unimplemented safety refuses to launch rather than returning a
permissive default. `observation_point` returns `None` when a target genuinely
has no vantage rather than a point that stares at a wall.

## Module reference

| module | role |
|---|---|
| `types.py` | Value types, NED frame, safety verdicts |
| `backends/base.py` | The `VehicleBackend` protocol — the seam |
| `backends/kinematic.py` | Dependency-free vehicle model: wind, battery, collision |
| `backends/px4.py` | MAVSDK / PX4, verified against SITL |
| `planner.py` | Pure geometry: survey grids, orbits |
| `navigation.py` | 3D A* with inflation, corner checks, climb penalty, smoothing |
| `behaviors.py` | Mission state machine |
| `patrol.py` | Navigate, observe, and map-aware `PlannedReturn` |
| `safety.py` | Envelope monitoring and the return-or-divert policy |
| `landing.py` | Emergency landing site scoring |
| `world.py` | Procedural city: boxes, collision, line of sight |
| `osm.py` | OpenStreetMap import, polygon footprints, power furnishing |
| `dynamic.py` | Moving obstacles, space-time collision |
| `hazards.py` | People and vehicles as regulatory volumes |
| `thin_obstacles.py` | Cables with catenary sag, porous foliage |
| `sensors.py` | Lidar/stereo detection **and non-detection** |
| `perception.py` | Simulated camera detector (statistics, not a net) |
| `mapping.py` | Real triangulation and structure inference |
| `surveillance.py` | Camera FOV, occlusion, coverage scoring |
| `spatial.py` | Uniform spatial hash for collision queries |
| `mission.py` | Control loop, flight log, track export |
| `viewer/` | Dependency-free 3D replay, free-camera navigation |

## Testing approach

275 tests, no network, no hardware. Three things they are written to do:

**Pin properties, not implementations.** "The route is collision-free" rather
than "the route has 7 waypoints".

**Make failure modes falsifiable.** The vehicle model can crash into buildings,
using the *airframe's* radius rather than the planner's standoff — if those
were the same number, a planner bug that shaved the margin could never produce
a crash and the test would be vacuous.

**Encode operational bars, not magic numbers.** The head-on mapping test
asserts error exceeds the *midspan sag it must resolve*, not an arbitrary
threshold.

Tests hard-block `urlopen`, so a cache-key change fails loudly instead of
silently hammering a public Overpass endpoint.
