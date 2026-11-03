# Urban Drone Autonomy

**Mission autonomy for a multirotor, developed in simulation and designed to move to hardware unchanged.**

Pure standard library to run. `pytest` to test. MAVSDK only if you want to fly it against a real PX4 flight stack.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue?style=flat-square)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-275%20passing-2ea44f?style=flat-square)](tests/)
[![PX4 SITL](https://img.shields.io/badge/PX4%20SITL-verified-19A0E0?style=flat-square)](docker/)
[![Dependencies](https://img.shields.io/badge/runtime%20deps-none-black?style=flat-square)](requirements.txt)
[![MCP servers live](https://whobuildsagents.com/badge/advancelabs.dev.svg)](https://whobuildsagents.com/advancelabs.dev)

### ✈︎ [Fly it in your browser → drone.advancelabs.dev](https://drone.advancelabs.dev)

Five environments, four of them real cities imported from OpenStreetMap. Every number on that page was produced by the simulator, so there is no marketing copy to drift out of sync with the code.

| Environment | Buildings | Tallest | What it exercises |
|---|---:|---:|---|
| Test grid | 25 | 54 m | Moving traffic, replanning on invalidation |
| Princess Anne Manor | 308 | 48 m | 5 m AGL, threading under distribution wire |
| Scarborough | 115 | 125 m | Car and pedestrian as legal keep-out volumes |
| North York Centre | 182 | 134 m | Surveillance with real occlusion |
| Downtown Toronto | 161 | **292 m** | Towers exceed the ceiling — route around, not over |

---

## Why this exists

Most drone-autonomy demos answer *"can it fly a path?"* That question is easy. The
interesting ones are the failures — the cases where a sensor is confidently wrong,
where the regulation binds harder than the physics, and where the safe-looking
fallback is the thing that kills you.

This repository is the record of chasing those cases down and measuring them. The
headline results below are all negative or constraining, and that is the point:
**they changed the hardware design.**

---

## Headline results

| Question | Answer |
|---|---|
| Can lidar see powerlines? | **No**, not on this budget — 5.5 m/s max near wires with a mid sensor |
| Can a camera? | Not directly. But it can **infer** them from poles: 100% recall, zero wire detections |
| Does 360° camera coverage help? | It removes the need to fly a survey pattern first, not detection quality |
| Can it fly under wires at 5 m? | Yes at 2.5 m standoff. At 7 m the corridor **closes entirely** |
| Is an inferred map good enough? | To *avoid* a corridor, yes. To *thread* one, no — that needs a survey |

<details>
<summary><b>Why lidar fails, in one paragraph</b></summary>

<br>

A wire is missed because of **geometry**, not signal strength. It returns plenty of
energy, but no beam ever points at it. A 12 mm distribution conductor subtends
1.2 mrad at 10 m, against a typical 0.4° (7 mrad) vertical beam spacing — so
single-scan detection probability is 17% at 10 m and 4.3% at 40 m. For a horizontal
wire the limiting spacing is the *vertical* one, because the wire spans hundreds of
beams along a scan row and what matters is whether any beam **row** lands on it.

Requiring three confirming returns — which you must, since one return is a bird or
dust — flips 8 m/s from safe to strike. A lidar permitting normal cruise near wires
costs more than twice the airframe budget. The mitigation is a 30 m AGL floor for
autonomous legs, which avoids the problem for free.

Full method, tables and caveats in **[FINDINGS.md](docs/FINDINGS.md)**.

</details>

<details>
<summary><b>The RTL trap</b></summary>

<br>

Perception cleaned the outbound leg completely. Every remaining wire breach was
**return-to-launch flying straight home through the map it had just built**.

| configuration | closest approach | breaches |
|---|---:|---:|
| no perception, direct RTL | 0.23 m | 16 (9 outbound, 7 return) |
| perception, direct RTL | 0.34 m | 7 (**all on the return**) |
| perception + `PlannedReturn` | **8.83 m** | **0** |

Plain `ReturnToLaunch` delegates to the flight controller, which knows nothing about
inferred obstacles. And RTL is what fires on low battery, on link loss, on any abort
— which is to say, it fires exactly when things are already going wrong.

</details>

---

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q            # 275 tests, no network needed
```

**Missions**

```bash
# open-field survey, with wind and emergency landing sites
python run.py survey --sites --wind 5,2 --log flight.json

# urban surveillance: plan around buildings, observe street-level targets
python run.py patrol --endurance 1400 --log patrol.json

# add moving traffic, powerlines, and sensor modelling
python run.py patrol --movers --wires --endurance 1600

# a real city from OpenStreetMap, with a car and a pedestrian
python run.py patrol --osm 43.6800,-79.5300 --hazards advanced

# fly the recording in 3D
python viewer/serve.py patrol.json
```

**Against a real flight stack**

```bash
docker compose -f docker/docker-compose.yml up --abort-on-container-exit
```

---

## What it does

**Flies missions safely.** A behaviour state machine with a safety monitor holding
veto power. Low battery means return *while there is still charge to get home* — the
gate is the computed return cost, not a fixed percentage. When home is out of reach it
diverts to the best reachable landing site, scored on quality, distance and wind. Link
loss starts a countdown rather than aborting, because telemetry links glitch constantly.

**Navigates real environments.** 3D A\* around polygon building footprints, with a
climb penalty so it routes around rather than pogoing over. Moving obstacles are
avoided in space-time — checked against where they *will* be when the vehicle arrives
— and replanning fires on path invalidation rather than on a timer.

**Sees what a camera can see, and admits what it cannot.** A 12 mm conductor is
sub-pixel at range and physically unresolvable; a utility pole is trivial. So the
system detects poles, infers which are connected, and avoids the catenary between
them — recovering a powerline with **zero wire detections**.

**Respects people as a legal constraint.** A pedestrian's keep-out volume is ~100×
their body size and asymmetric, because overflight is separately prohibited. Modelled
as a hard limit plus a soft cost, so among legal routes the planner prefers the one
giving more room.

**Replays in 3D.** Dependency-free: a hand-rolled camera matrix on a 2D canvas. Orbit,
pan, fly the camera with WASD, follow the vehicle.

---

## Backends

The autonomy layer talks to a `VehicleBackend` seam, so the same mission code runs
against a kinematic sim today, PX4 SITL next, and a Pixhawk after that.

| Backend | Needs | Use for |
|---|---|---|
| `kinematic` | nothing | iterating on logic in under a second |
| `px4` | Docker + `pip install mavsdk` | trusting the numbers |

The kinematic model is **not a flight dynamics model** — no attitude, no rotor thrust,
no aerodynamics. It reproduces what the autonomy layer reacts to: finite acceleration,
arrival overshoot, wind, and a battery that drains faster under load. Logic that
handles it correctly handles PX4 correctly, because the mistakes it punishes are the
ones that bite on hardware.

Verified against PX4 SITL: takeoff, hold, RTL and land, 43 s, using the identical
behaviour objects. PX4 binds MAVLink to the container's loopback, so publishing ports
does not work — the compose file sets up a shared network with fixed IPs, which is
why it exists.

---

## Viewer controls

A **free camera**, the way Unity and Roblox Studio behave: it has a position and a
look direction, and turning it does not swing you around a pivot.

| Input | Action |
|---|---|
| drag | look around (turns in place) |
| `W` `A` `S` `D` | fly along the ground, where you are looking |
| `Q` `E` | descend / climb |
| shift | move faster |
| right-drag, middle-drag, shift-drag | pan |
| scroll | dolly forward and back |
| alt-drag | orbit the point ahead |
| double-click | focus on that point of the track |
| `F` / `R` / space | follow the vehicle / reframe / play |

On touch, one finger looks, two fingers pinch to zoom and slide to pan. Those are
handled explicitly because `touch-action: none` — needed for single-finger look —
also disables the browser's own pinch.

---

## Repository map

| Path | What |
|---|---|
| [`autonomy/`](autonomy/) | The stack: behaviours, planner, perception, safety, sensors, OSM import |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | The design, and why each piece is shaped that way |
| [`docs/FINDINGS.md`](docs/FINDINGS.md) | **Every measured result**, with method and caveats |
| [`showcase/`](showcase/) | Builds the five-environment page at drone.advancelabs.dev |
| [`viewer/`](viewer/) | Dependency-free 3D flight replay |
| [`tests/`](tests/) | 275 tests across 17 modules |
| [`docker/`](docker/) | PX4 SITL compose setup |

---

## Honest limitations

- **All figures come from the simulator, not from flight.** Treat them as
  well-reasoned estimates to be validated against hardware, not as specifications.
- **The kinematic backend is not a flight dynamics model.** See *Backends* above.
- **OSM building heights are mostly absent** and therefore estimated from the
  `building` tag, and flagged as estimated. A route at 15 m over an `apartments`
  polygon assumed to be 6 m is a collision.
- **OSM power coverage is thin.** The Princess Anne Manor import returned *zero*
  mapped conductors for a neighbourhood that certainly has one down every street, so
  the wires the planner avoids there are synthesized along the street graph.
  Under-modelling is the dangerous direction.
- **Regulatory constants are modelling defaults, not legal advice.** Verify against
  the current CARs Part IX before flying anything. In Canadian airspace "autonomous"
  means *pilot-supervised* autonomy, with a human able to take control at all times.

---

## License & attribution

Source code is licensed under the **[Apache License 2.0](LICENSE)**.

City environments are derived from **OpenStreetMap** data — © OpenStreetMap
contributors, available under the [Open Database License (ODbL) v1.0](https://opendatacommons.org/licenses/odbl/1-0/).
The ODbL covers the *data* independently of the Apache-2.0 licence covering the code;
attribution is required on anything produced from it. See [NOTICE](NOTICE) for detail.

---

Built by [Advance Labs Inc.](https://advancelabs.dev), a Canadian software studio.
We take on autonomy and AI contract development — [get in touch](https://advancelabs.dev/services).
