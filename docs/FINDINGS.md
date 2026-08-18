# Findings

Every measured result from the sim, with how it was obtained and what it means
operationally. This is the reusable part: the code will change, these numbers
are what the code was written to discover.

**All figures come from the simulator, not from flight.** The models are
documented in each module and their limits stated. Treat everything here as a
well-reasoned estimate to be validated against hardware, not as a
specification.

---

## 1. Wire strike: lidar cannot solve it on this budget

**Method.** Angular subtense of a conductor against lidar beam spacing, with
detection accumulating over an approach, requiring 3 confirming returns at 90%
confidence. Compared against stopping distance for a 1.5 kg quad (0.6 s
velocity time constant, 0.3 s reaction). See `autonomy/sensors.py`.

Single-scan detection of a 12 mm distribution conductor, 0.4° vertical spacing:

| range | subtense | detection chance |
|---|---|---|
| 10 m | 1.2 mrad | 17% |
| 20 m | 0.6 mrad | 8.6% |
| 40 m | 0.3 mrad | 4.3% |

Maximum safe speed near wires, by sensor:

| sensor | approx cost | max safe speed |
|---|---|---|
| cheap 16-line, 0.8° @ 10 Hz | ~$400 | 3.5 m/s |
| mid 32-line, 0.4° @ 10 Hz | ~$1,200 | 5.5 m/s |
| good 64-line, 0.2° @ 20 Hz | ~$4,000+ | 13.0 m/s |
| survey-grade, 0.1° @ 40 Hz | much more | 19.5 m/s |

**Why.** A wire is missed because of *geometry*, not signal strength — it
returns plenty of energy, but no beam ever points at it. For a horizontal wire
the limiting spacing is the **vertical** one: the wire spans hundreds of beams
along a scan row, so what matters is whether any beam *row* lands on it, and
vertical spacing is typically the coarser of the two.

**Requiring confirmation changes the answer.** One return is a bird or dust.
Demanding three flips 8 m/s from safe to strike.

**Operational conclusion.** A lidar that permits normal cruise near wires costs
more than twice the airframe budget. The mitigation is a **30 m AGL floor for
autonomous legs**, which avoids the problem for free.

---

## 2. Camera perception: infer the wires instead

**Method.** Simulated detector statistics (per-class recall vs apparent pixel
size, bounding-box jitter as bearing noise) feeding real triangulation and
structure inference. See `autonomy/perception.py` and `autonomy/mapping.py`.

**Result: 5/5 poles triangulated, 4/4 spans inferred, 100% recall and
precision — with zero wire detections.**

**Why it works.** A 12 mm conductor at 40 m is a third of a pixel: physically
unresolvable. A 9 m utility pole at the same range is ~250 px and trivial. So
detect the poles, infer which pairs are connected, and treat the catenary
between them as an obstacle you never saw.

### Flight geometry dominates detector quality

Triangulation error scales roughly as `range² × angular_noise / baseline`:

| flight path | camera rig | recall | mean pole error |
|---|---|---|---|
| head-on approach | 1 forward | 75% | 2.64 m |
| head-on approach | 4 cameras | 100% | 0.38 m |
| head-on approach | 6 cameras (360°) | 100% | 0.07 m |
| lateral sweep | 1 forward | 100% | 0.06 m |
| lateral sweep | 6 cameras | 100% | 0.04 m |

Midspan sag on a 40 m span is ~1.2 m, and that is the bar: a map with more
error than the sag cannot tell you whether you are above the wire or through
it. **Head-on with one camera fails that bar while looking confident**, which
is the dangerous case.

**Operational conclusion.** 360° coverage buys freedom from flying a survey
pattern, not better detection. One forward camera suffices if you fly a lateral
sweep across the corridor first. A rear camera alone does not help on a
straight-in approach — it only sees where you have been. Extra cameras are not
free: more frames means more false positives, and one 4-camera run reported six
poles where five existed.

---

## 3. Avoidance: the map has to reach the planner, including RTL

**Method.** A transmission crossing at flight altitude, on the direct route.
Clearance bar set at 2.5 m, because a drone must clear a conductor by more than
its own position error.

| configuration | closest approach | breaches |
|---|---|---|
| no perception, direct RTL | 0.23 m | 16 (9 outbound, 7 return) |
| perception, direct RTL | 0.34 m | 7 (**all on the return**) |
| perception + `PlannedReturn` | **8.83 m** | **0** |

**The middle row is the finding.** Perception cleaned the outbound leg
completely; every remaining breach was return-to-launch flying straight home
through the map it had just built. Plain `ReturnToLaunch` delegates to the
flight controller, which knows nothing about inferred obstacles — and RTL is
what fires on low battery, on link loss, on any abort.

---

## 4. Standoff decides whether low-level flight is possible

**Method.** A survey of a real residential grid (OSM import of Princess
Anne Manor, Etobicoke) at 5 m AGL, under distribution wire at 9 m.

| configuration | legs completed | closest wire |
|---|---|---|
| 5 m AGL, 2.5 m standoff | **6/6** | 1.82 m |
| 5 m AGL, 7 m standoff | **4/6** | 6.87 m |
| 20 m AGL, over the wires | 6/6 | 9.00 m |

At 7 m standoff the corridor closes: wire at 9 m over 7-9 m houses leaves no
gap that wide, and two legs become unreachable.

**This connects §2 and §3.** An inferred map needs 7-10 m of standoff to cover
its triangulation error. So **camera-inferred maps are good enough to avoid a
corridor and not good enough to thread one.** Threading needs a surveyed map,
where 2.5 m is justified. Those are two different operational capabilities.

---

## 5. People are a regulatory volume, not an obstacle

**Method.** Transport Canada separation minima modelled as hard constraint plus
soft cost. See `autonomy/hazards.py`. **Verify against current CARs Part IX
before flying** — these are modelling defaults, not legal advice.

| profile | separation | overflight | closest flown |
|---|---|---|---|
| basic | 30 m | prohibited | 30.1 m |
| advanced | 5 m | permitted | 5.5 m |

A person is 0.4 m across; as a collision hazard they barely register. What
governs is the legal separation, which is ~100× their body size and asymmetric
— passing 30 m to one side is fine, passing overhead at 40 m is not.

**Plan against the limit plus tracking error.** With `tracking_error_m` at zero
the flown path reached **4.3 m against a 5 m rule**: legal on paper, violated in
the air, because guidance lag and corner-cutting shave the margin the plan
assumed.

---

## 6. Wind breaks proportional guidance

**Method.** Constant wind added to the kinematic vehicle model.

A P-controller commanding *airspeed* from position error cannot reject a
constant disturbance. Near the setpoint the command tapers to zero, so ground
velocity tapers to exactly the wind and the vehicle parks downwind forever. The
equilibrium is computable: at 8.4 m/s over a 10 m brake radius in a 5.4 m/s
wind, **6.5 m downwind of every setpoint**. Arrival radius is 0.8 m, so it never
arrives — takeoff took 75 s instead of 8 s and the mission timed out.

Fixed with **wind feedforward**, which is what real position controllers use:
subtract the wind estimate from the command. No integrator, so no windup.

---

## 7. Performance: realistic worlds need a spatial index

Both collision tests were linear scans — fine for 25 procedural boxes,
catastrophic for 273 polygon footprints plus 448 catenaries:

| query | linear scan | indexed | speedup |
|---|---|---|---|
| `is_free` (buildings) | 596 µs | 6.5 µs | 92× |
| `solid_at` (conductors) | 8 468 µs | 14.1 µs | 601× |

A single A* would have spent over two hours in collision tests; the first
low-altitude survey simply timed out.

The 3D viewer had the same shape of problem:

| per frame | naive | fixed |
|---|---|---|
| `getComputedStyle` calls | 4 496 | 0 |
| canvas stroke/fill calls | 4 676 | 339 |

Theme colours were resolved per drawn segment, forcing a style recalculation
thousands of times a frame for values that never change.

**The pattern in both cases:** the cost is invisible until the scene becomes
realistic, which is exactly when it matters.

---

## 8. Real map data: mind the gaps

An OSM import of Princess Anne Manor, Etobicoke (350 m radius):

| feature | count | note |
|---|---|---|
| buildings | 308 | **308 of 308 heights estimated**, not tagged |
| roads | 58 | |
| sidewalks | 84 | |
| power lines | 0 | for a neighbourhood with one down every street |
| towers / poles | 0 / 0 | |

Zero is not an unusual reading — it is the common one. Every conductor the
planner avoids in this environment is synthesized along the street graph,
because OSM knew about none of them.

**Coverage is biased.** Transmission is mapped well because it is visible from
satellite imagery; residential distribution badly because it is not. Building
heights are largely absent, so they are estimated from the `building` tag and
flagged as estimated — a route at 15 m over `apartments` assumed to be 6 m is a
planned collision.

`furnish_power` fills the distribution gap, flagging every synthesised run so it
never reads as observed truth.

---

## 9. PX4 SITL: the backend seam holds

The same behaviours, safety monitor and mission runner used by the kinematic
sim flew against real PX4: takeoff to 12 m, 8 s hold, RTL and land, 43 s,
completed.

Three MAVSDK 3.x differences cost real time and are worth recording:
`telemetry` is a property not a method; `action.set_maximum_speed` was removed
(cruise speed is a PX4 parameter); and `battery.remaining_percent` is 0-100
rather than 0-1 — guessing wrong made the safety layer read **8000%** and never
trigger.

Networking: PX4 binds MAVLink to the container's loopback, so publishing ports
does not work. A shared Docker network with fixed IPs is required.

---

## What none of this establishes

- No hardware has been flown. Every number is from a model.
- The sensor model omits multipath, sun glare, rain, and motion blur.
- The detector is simulated statistics, not a trained network.
- Synthesised power lines are plausible, not surveyed — a mission that succeeds
  against them proves the planner works, not that a given street is flyable.
- Wires remain the dominant real-world risk, and the honest mitigation is still
  altitude discipline rather than sensing.
