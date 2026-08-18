"""The control loop: safety first, then the active behaviour, every tick.

Ordering inside `_tick` is the safety-critical part of this file. Safety is
evaluated before the behaviour is allowed to act, so a behaviour can never
issue one last command on the cycle where the geofence trips.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .backends.base import VehicleBackend

if TYPE_CHECKING:
    from .dynamic import DynamicField
    from .hazards import HazardField
    from .mapping import PerceptionPipeline
    from .sensors import SensorSuite
    from .surveillance import CoverageTracker
    from .world import City
from .behaviors import Behavior, DivertAndLand, Land, ReturnToLaunch
from .safety import SafetyMonitor
from .types import SafetyAction, Telemetry


@dataclass(slots=True)
class LogEntry:
    """One line of the flight log."""

    t: float
    phase: str
    altitude: float
    battery: float
    note: str = ""

    def format(self) -> str:
        note = f"  {self.note}" if self.note else ""
        return (
            f"[{self.t:6.1f}s] {self.phase:<22} "
            f"alt {self.altitude:5.1f} m  batt {self.battery * 100:4.0f}%{note}"
        )


@dataclass(slots=True)
class Sample:
    """One trajectory point. Recorded every tick, unlike `LogEntry` which
    only fires on phase changes -- the viewer needs the continuous path."""

    t: float
    north: float
    east: float
    down: float
    heading: float
    battery: float
    phase: str

    def as_dict(self, precision: int = 2) -> dict:
        return {
            "t": round(self.t, 2),
            "n": round(self.north, precision),
            "e": round(self.east, precision),
            "d": round(self.down, precision),
            "hdg": round(self.heading, 1),
            "batt": round(self.battery, 4),
            "phase": self.phase,
        }


def simplify_track(track: list["Sample"], tolerance_m: float) -> list["Sample"]:
    """Ramer-Douglas-Peucker over the 3D path, preserving phase boundaries.

    Chosen over a uniform stride because stride is blind to shape: it throws
    away the same fraction of a tight turn as of a straight transit, so the
    corner rounds off while the straight line keeps points it never needed.
    RDP keeps whatever it takes to stay within `tolerance_m` of the original.

    Phase changes are split points rather than candidates for removal -- they
    are where the colour boundary sits in the viewer, and losing one moves a
    visible edge.
    """
    if len(track) < 3 or tolerance_m <= 0:
        return track

    runs: list[list[Sample]] = [[track[0]]]
    for sample in track[1:]:
        if sample.phase != runs[-1][-1].phase:
            runs.append([runs[-1][-1], sample])   # share the boundary sample
        else:
            runs[-1].append(sample)

    out: list[Sample] = []
    for run in runs:
        kept = _rdp(run, tolerance_m)
        out.extend(kept if not out else kept[1:])
    return out


def _rdp(points: list["Sample"], tol: float) -> list["Sample"]:
    """Iterative RDP. Recursion would blow the stack on a long flight."""
    if len(points) < 3:
        return list(points)

    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]

    while stack:
        first, last = stack.pop()
        worst, worst_i = 0.0, -1
        ax, ay, az = points[first].north, points[first].east, points[first].down
        bx, by, bz = points[last].north, points[last].east, points[last].down
        dx, dy, dz = bx - ax, by - ay, bz - az
        span_sq = dx * dx + dy * dy + dz * dz

        for i in range(first + 1, last):
            px, py, pz = points[i].north, points[i].east, points[i].down
            if span_sq < 1e-12:
                dist = math.dist((px, py, pz), (ax, ay, az))
            else:
                t = ((px - ax) * dx + (py - ay) * dy + (pz - az) * dz) / span_sq
                t = max(0.0, min(1.0, t))
                dist = math.dist(
                    (px, py, pz), (ax + t * dx, ay + t * dy, az + t * dz)
                )
            if dist > worst:
                worst, worst_i = dist, i

        if worst > tol and worst_i > 0:
            keep[worst_i] = True
            stack.append((first, worst_i))
            stack.append((worst_i, last))

    return [p for p, k in zip(points, keep) if k]


@dataclass(slots=True)
class MissionResult:
    """Outcome of a flight, for assertions in tests and review afterwards."""

    completed: bool
    aborted_reason: str | None
    duration_s: float
    log: list[LogEntry] = field(default_factory=list)
    track: list[Sample] = field(default_factory=list)
    context: dict = field(default_factory=dict)

    @property
    def landed_safely(self) -> bool:
        return bool(self.track) and self.track[-1].down > -0.5

    def to_dict(self, *, simplify_m: float = 0.0, precision: int = 2) -> dict:
        """Serialisable flight record, consumed by `viewer/index.html`.

        Args:
            simplify_m: if positive, thin the track with Ramer-Douglas-Peucker
                to this tolerance in metres. The recorder samples far finer
                than any viewer draws, and a straight transit leg costs
                hundreds of collinear points that carry no information. Zero
                keeps every sample, which is what analysis wants.
            precision: decimal places for coordinates. The default of 2 is
                centimetre resolution on a map hundreds of metres across --
                generous. 1 is decimetres and saves roughly a tenth of the
                file with nothing visible lost.
        """
        track = simplify_track(self.track, simplify_m) if simplify_m > 0 else self.track
        return {
            "completed": self.completed,
            "aborted_reason": self.aborted_reason,
            "duration_s": round(self.duration_s, 1),
            "context": self.context,
            "events": [
                {
                    "t": round(e.t, 2),
                    "phase": e.phase,
                    "alt": round(e.altitude, 2),
                    "batt": round(e.battery, 4),
                    "note": e.note,
                }
                for e in self.log
            ],
            "track": [s.as_dict(precision) for s in track],
        }


class MissionRunner:
    """Executes a behaviour sequence with a safety monitor holding veto power."""

    def __init__(
        self,
        vehicle: VehicleBackend,
        behaviors: list[Behavior],
        *,
        safety: SafetyMonitor | None = None,
        tick_hz: float = 10.0,
        timeout_s: float = 1800.0,
        realtime: bool = False,
        verbose: bool = True,
        coverage: "CoverageTracker | None" = None,
        city: "City | None" = None,
        dynamic: "DynamicField | None" = None,
        sensors: "SensorSuite | None" = None,
        perception: "PerceptionPipeline | None" = None,
        hazards: "HazardField | None" = None,
    ) -> None:
        if not behaviors:
            raise ValueError("mission needs at least one behaviour")
        if tick_hz <= 0:
            raise ValueError("tick_hz must be positive")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")

        self.vehicle = vehicle
        self.behaviors = list(behaviors)
        self.safety = safety or SafetyMonitor()
        self.dt = 1.0 / tick_hz
        self.timeout_s = timeout_s
        self.realtime = realtime
        self.verbose = verbose

        self.coverage = coverage
        self.city = city
        self.dynamic = dynamic
        self.sensors = sensors
        self.perception = perception
        self.hazards = hazards
        self.detections: list[dict] = []
        self._first_cable_at: float | None = None
        self.log: list[LogEntry] = []
        self.track: list[Sample] = []
        self._index = 0
        self._aborting = False
        self._abort_reason: str | None = None

    async def run(self) -> MissionResult:
        """Fly the mission. Always attempts to land, even on abort or error."""
        await self.vehicle.connect()
        self.safety.reset()

        try:
            while self._index < len(self.behaviors):
                telemetry = await self.vehicle.telemetry()
                self._sample(telemetry)
                if telemetry.elapsed_s > self.timeout_s:
                    self._record(telemetry, "TIMEOUT", "mission exceeded time budget")
                    await self._abort("mission timeout")
                    break
                await self._tick(telemetry)
                await self.vehicle.step(self.dt)

                if self.coverage is not None:
                    for name in self.coverage.update(telemetry, self.dt):
                        self._record(telemetry, "OBSERVED", f"{name} covered")

                if self.sensors is not None:
                    self._run_sensors(telemetry)

                if self.perception is not None:
                    self.perception.step(
                        telemetry.position, telemetry.heading_deg, telemetry.elapsed_s
                    )

                crashed = getattr(self.vehicle, "crashed_into", None)
                if crashed:
                    self._record(telemetry, "CRASH", f"struck {crashed}")
                    self._abort_reason = f"collision with {crashed}"
                    break

                if self.realtime:
                    await asyncio.sleep(self.dt)
        finally:
            final = await self.vehicle.telemetry()
            self._sample(final)
            await self.vehicle.close()

        return MissionResult(
            completed=self._index >= len(self.behaviors) and not self._abort_reason,
            aborted_reason=self._abort_reason,
            duration_s=final.elapsed_s,
            log=self.log,
            track=self.track,
            context=self._context(),
        )

    def _context(self) -> dict:
        """Static scene data the viewer draws around the flight path."""
        limits = self.safety.limits
        sites = getattr(self.safety, "landing_sites", None)
        extra: dict = {}
        if self.city is not None:
            extra["city"] = self.city.to_dict()
        if self.dynamic is not None and self.dynamic.obstacles:
            extra["movers"] = [
                {"name": o.name, "radius": o.radius_m, "speed": o.speed_mps,
                 "waypoints": [{"n": w.north, "e": w.east, "d": w.down}
                               for w in o.waypoints], "loop": o.loop}
                for o in self.dynamic.obstacles
            ]
        if self.hazards is not None and not self.hazards.is_empty:
            extra["hazards"] = self.hazards.to_dict(0.0)
        if self.detections:
            extra["detections"] = self.detections[:4000]
        if self.perception is not None:
            extra["perception"] = self.perception.to_dict()
            extra["perception_summary"] = self.perception.summary()
        if self.sensors is not None and self.sensors.thin.cables:
            # Ground truth, so the viewer can show which real conductors the
            # inferred map caught and which it missed.
            extra["true_wires"] = [
                {"name": c.name,
                 "points": [{"n": round(q.north, 2), "e": round(q.east, 2),
                             "d": round(q.down, 2)} for q in c.points()]}
                for c in self.sensors.thin.cables
            ]
        if self.coverage is not None:
            extra["surveillance"] = self.coverage.to_dict()
            extra["coverage_summary"] = self.coverage.summary()
        return extra | {
            "geofence_radius_m": limits.max_radius_m,
            "max_altitude_m": limits.max_altitude_m,
            "wind": {
                "n": round(getattr(self.vehicle, "wind", None).north, 2)
                if getattr(self.vehicle, "wind", None)
                else 0.0,
                "e": round(getattr(self.vehicle, "wind", None).east, 2)
                if getattr(self.vehicle, "wind", None)
                else 0.0,
            },
            "landing_sites": [
                {
                    "name": s.name,
                    "n": s.position.north,
                    "e": s.position.east,
                    "radius": s.radius_m,
                    "quality": s.quality,
                }
                for s in (sites.sites if sites else [])
            ],
        }

    def _run_sensors(self, telemetry: Telemetry) -> None:
        """Scan and log thin-obstacle returns.

        Only cable and foliage returns are recorded. Building returns are
        certain at any usable range and would swamp the log without telling
        anyone anything they did not already know from the map.
        """
        for d in self.sensors.scan(telemetry.position, telemetry.heading_deg):
            if d.kind == "building":
                continue
            self.detections.append(
                {
                    "t": round(telemetry.elapsed_s, 2),
                    "kind": d.kind,
                    "name": d.name,
                    "range": round(d.range_m, 2),
                    "bearing": round(d.bearing_deg, 1),
                    "conf": d.confidence,
                }
            )
            if d.kind == "cable" and self._first_cable_at is None:
                self._first_cable_at = telemetry.elapsed_s
                self._record(
                    telemetry, "CABLE RETURN",
                    f"{d.name} at {d.range_m:.1f} m (p={d.confidence:.2f})",
                )

    def _sample(self, telemetry: Telemetry) -> None:
        """Record a trajectory point, skipping ones that carry no information.

        At 10 Hz a hovering vehicle emits ten identical samples a second, which
        inflates the flight record and the viewer's work for nothing. Keep a
        sample when the vehicle has actually moved, when the phase changes, or
        when enough time has passed that the gap would show.
        """
        phase = (
            self.behaviors[self._index].name
            if self._index < len(self.behaviors)
            else "done"
        )
        if self.track:
            last = self.track[-1]
            moved = math.dist(
                (last.north, last.east, last.down),
                (
                    telemetry.position.north,
                    telemetry.position.east,
                    telemetry.position.down,
                ),
            )
            stale = telemetry.elapsed_s - last.t
            if moved < 0.5 and stale < 1.0 and last.phase == phase:
                return

        self.track.append(
            Sample(
                t=telemetry.elapsed_s,
                north=telemetry.position.north,
                east=telemetry.position.east,
                down=telemetry.position.down,
                heading=telemetry.heading_deg,
                battery=telemetry.battery_remaining,
                phase=phase,
            )
        )

    # -- internals ---------------------------------------------------------

    async def _tick(self, telemetry: Telemetry) -> None:
        # 1. Safety always evaluates first and can preempt the behaviour.
        if not self._aborting:
            verdict = self.safety.evaluate(telemetry)
            if verdict.action is SafetyAction.WARN:
                self._record(telemetry, "WARN", verdict.reason)
            elif verdict.action is SafetyAction.RETURN_TO_LAUNCH:
                self._record(telemetry, "SAFETY RTL", verdict.reason)
                await self._abort(verdict.reason, ReturnToLaunch())
                return
            elif verdict.action is SafetyAction.LAND_NOW:
                # A target means the policy picked a reachable landing site;
                # without one, the situation is bad enough to land in place.
                if verdict.target is not None:
                    recovery = DivertAndLand(verdict.target, site_name="site")
                    self._record(telemetry, "SAFETY DIVERT", verdict.reason)
                else:
                    recovery = Land()
                    self._record(telemetry, "SAFETY LAND", verdict.reason)
                await self._abort(verdict.reason, recovery)
                return

        # 2. Only then does the active behaviour get to act.
        current = self.behaviors[self._index]
        await current.tick(self.vehicle, telemetry)

        if current.is_complete(telemetry):
            self._record(telemetry, current.describe(), "complete")
            self._index += 1

    async def _abort(self, reason: str, replacement: Behavior | None = None) -> None:
        """Replace the remaining mission with a recovery behaviour."""
        if self._aborting:
            return
        self._aborting = True
        self._abort_reason = reason
        self.behaviors = self.behaviors[: self._index] + (
            [replacement] if replacement else []
        )
        if replacement is None:
            self._index = len(self.behaviors)

    def _record(self, telemetry: Telemetry, phase: str, note: str = "") -> None:
        entry = LogEntry(
            t=telemetry.elapsed_s,
            phase=phase,
            altitude=telemetry.altitude,
            battery=telemetry.battery_remaining,
            note=note,
        )
        self.log.append(entry)
        if self.verbose:
            print(entry.format())
