"""Urban surveillance behaviours: navigate around obstacles, then observe.

`NavigateTo` replans through the city rather than flying a straight line, so
it is the obstacle-aware counterpart to `FollowPath`. `ObserveTarget` holds a
vantage point until the camera has accumulated enough dwell.

`build_patrol` composes them into a full mission, and does the part that
actually matters: choosing an observation point that can *see* each target
rather than one that is merely near it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .behaviors import Behavior
from .backends.base import VehicleBackend
from .dynamic import DynamicField
from .navigation import GridPlanner
from .surveillance import (
    Camera,
    CoverageTracker,
    SurveillanceTarget,
    observation_point,
)
from .types import Telemetry, Vec3
from .world import City

if TYPE_CHECKING:
    from .hazards import HazardField
    from .mapping import PerceptionPipeline


class NavigateTo(Behavior):
    """Fly to a point, routing around buildings and replanning around movers.

    Replans on *invalidation* rather than on a timer. Checking whether the
    remaining path is still clear is a sweep, not a graph search, so it can run
    every tick while replans stay rare. A slow heartbeat backstops it, to catch
    an obstacle that has drifted enough to matter without ever formally
    blocking the path.
    """

    name = "navigate"

    def __init__(
        self,
        target: Vec3,
        planner: GridPlanner,
        *,
        speed_mps: float = 6.0,
        arrival_radius_m: float = 3.0,
        label: str = "",
        dynamic: DynamicField | None = None,
        hazards: "HazardField | None" = None,
        perception: "PerceptionPipeline | None" = None,
        replan_heartbeat_s: float = 4.0,
        replan_cooldown_s: float = 0.8,
    ) -> None:
        super().__init__()
        if speed_mps <= 0:
            raise ValueError("speed must be positive")
        self.target = target
        self.planner = planner
        self.speed = speed_mps
        self.arrival_radius = arrival_radius_m
        self.label = label
        self.dynamic = dynamic
        self.hazards = hazards
        # Live camera-derived map. Read fresh on every check rather than
        # snapshotted, because its whole purpose is to grow mid-flight: a
        # conductor inferred thirty seconds in must invalidate a path planned
        # before it existed.
        self.perception = perception
        self.heartbeat = replan_heartbeat_s
        # Without a cooldown a persistently blocked path replans every tick,
        # which is the pathological case this design exists to avoid.
        self.cooldown = replan_cooldown_s

        self.route: list[Vec3] = []
        self.index = 0
        self.failed = False
        self.replans = 0
        self.blocked_by: str | None = None
        self.wire_replans = 0
        self._last_plan_t = -1e9

    async def on_start(self, vehicle: VehicleBackend, telemetry: Telemetry) -> None:
        # Release any look-at lock from a previous observation, so heading
        # follows the direction of travel again while transiting.
        await vehicle.look_at(None)
        await self._replan(vehicle, telemetry)

    async def on_tick(self, vehicle: VehicleBackend, telemetry: Telemetry) -> None:
        if self.failed or self.index >= len(self.route):
            return

        if self._needs_replan(telemetry):
            await self._replan(vehicle, telemetry)
            return

        if (self.route[self.index] - telemetry.position).norm <= self.arrival_radius:
            self.index += 1
            if self.index < len(self.route):
                await vehicle.goto(self.route[self.index], self.speed)

    def _thin(self):
        """Current inferred conductor map, or None."""
        if self.perception is None or not self.perception.obstacles.cables:
            return None
        return self.perception.obstacles

    def _needs_replan(self, telemetry: Telemetry) -> bool:
        has_dynamic = self.dynamic is not None and self.dynamic.obstacles
        has_hazards = self.hazards is not None and not self.hazards.is_empty
        thin = self._thin()
        if not has_dynamic and thin is None and not has_hazards:
            return False
        if telemetry.elapsed_s - self._last_plan_t < self.cooldown:
            return False

        # A newly inferred wire across the remaining path is the reason this
        # check exists: the route was legal when planned and is not any more.
        if thin is not None:
            legs = [telemetry.position] + self.route[self.index:]
            for a, b in zip(legs, legs[1:]):
                if not thin.segment_clear(a, b, self.planner.thin_margin):
                    self.blocked_by = "inferred conductor"
                    self.wire_replans += 1
                    return True

        if self.hazards is not None and not self.hazards.is_empty:
            legs = [telemetry.position] + self.route[self.index:]
            elapsed = telemetry.elapsed_s
            for a, b in zip(legs, legs[1:]):
                if not self.planner._hazard_clear(
                    a, b, self.hazards, elapsed, self.speed
                ):
                    self.blocked_by = "person or vehicle"
                    return True
                elapsed += (b - a).norm / max(self.speed, 0.1)

        if has_dynamic:
            blocker = self.dynamic.path_clear(
                self.route, self.index, telemetry.position,
                telemetry.elapsed_s, self.speed, self.planner.margin,
            )
            if blocker is not None:
                self.blocked_by = blocker
                return True
        return telemetry.elapsed_s - self._last_plan_t >= self.heartbeat

    async def _replan(self, vehicle: VehicleBackend, telemetry: Telemetry) -> None:
        self._last_plan_t = telemetry.elapsed_s
        result = self.planner.plan(
            telemetry.position, self.target,
            dynamic=self.dynamic, thin=self._thin(), hazards=self.hazards,
            t_now=telemetry.elapsed_s, speed_mps=self.speed,
        )
        if not result.ok or not result.waypoints:
            # Hold position rather than flying blind at the target. The mover
            # will usually clear, and the heartbeat will retry.
            await vehicle.goto(telemetry.position, 1.0)
            # Hold and retry only if something transient might clear. A purely
            # static failure will not resolve itself, so give up rather than
            # hover until the battery runs out.
            transient = (self.dynamic is not None and self.dynamic.obstacles) or (
                self.perception is not None
            )
            if not transient:
                self.failed = True
            return
        if self.route:
            self.replans += 1
        self.route = result.waypoints
        self.index = 0
        await vehicle.goto(self.route[0], self.speed)

    def is_complete(self, telemetry: Telemetry) -> bool:
        return self.failed or self.index >= len(self.route)

    def reset(self) -> None:
        super().reset()
        self.route = []
        self.index = 0
        self.failed = False
        self.replans = 0
        self.blocked_by = None
        self._last_plan_t = -1e9

    def describe(self) -> str:
        bits = []
        if self.replans:
            bits.append(f"{self.replans} replans")
        if self.wire_replans:
            bits.append(f"{self.wire_replans} for wires")
        suffix = f" (+{', '.join(bits)})" if bits else ""
        return f"navigate to {self.label or 'waypoint'}{suffix}"


class PlannedReturn(Behavior):
    """Return to launch by routing around what we know, not in a straight line.

    Plain `ReturnToLaunch` hands the job to the flight controller, which flies
    direct. That is fine over open ground and dangerous anywhere a map exists:
    RTL is what fires on low battery, on link loss, on any abort, so it is
    precisely the behaviour that must not drive through a conductor the
    aircraft spent the whole flight mapping.

    Falls back to a direct return if planning fails. A degraded return beats
    hovering until the battery runs out, and the situations that trigger RTL
    are rarely ones with time to spare.
    """

    name = "rtl"

    def __init__(
        self,
        planner: GridPlanner,
        *,
        speed_mps: float = 6.0,
        perception: "PerceptionPipeline | None" = None,
        dynamic: DynamicField | None = None,
        hazards: "HazardField | None" = None,
        home: Vec3 | None = None,
        arrival_radius_m: float = 3.0,
    ) -> None:
        super().__init__()
        self.planner = planner
        self.speed = speed_mps
        self.perception = perception
        self.dynamic = dynamic
        self.hazards = hazards
        self.home = home or Vec3()
        self.arrival_radius = arrival_radius_m
        self.route: list[Vec3] = []
        self.index = 0
        self.direct = False
        self._descending = False

    def _thin(self):
        if self.perception is None or not self.perception.obstacles.cables:
            return None
        return self.perception.obstacles

    async def on_start(self, vehicle: VehicleBackend, telemetry: Telemetry) -> None:
        await vehicle.look_at(None)
        # Transit home at the current altitude, then descend once overhead --
        # descending early would fly the approach through everything below.
        goal = Vec3(self.home.north, self.home.east, telemetry.position.down)
        result = self.planner.plan(
            telemetry.position, goal,
            dynamic=self.dynamic, thin=self._thin(), hazards=self.hazards,
            t_now=telemetry.elapsed_s, speed_mps=self.speed,
        )
        if not result.ok or not result.waypoints:
            self.direct = True
            await vehicle.return_to_launch()
            return
        self.route = result.waypoints
        self.index = 0
        await vehicle.goto(self.route[0], self.speed)

    async def on_tick(self, vehicle: VehicleBackend, telemetry: Telemetry) -> None:
        if self.direct or self._descending:
            return
        if self.index < len(self.route):
            if (self.route[self.index] - telemetry.position).norm <= self.arrival_radius:
                self.index += 1
                if self.index < len(self.route):
                    await vehicle.goto(self.route[self.index], self.speed)
            return
        # Route flown; commit to the descent.
        if telemetry.position.horizontal_distance_to(self.home) < 6.0:
            self._descending = True
            await vehicle.land()
        else:
            await vehicle.return_to_launch()
            self.direct = True

    def is_complete(self, telemetry: Telemetry) -> bool:
        return not telemetry.armed and telemetry.altitude < 0.3

    def reset(self) -> None:
        super().reset()
        self.route = []
        self.index = 0
        self.direct = False
        self._descending = False

    def describe(self) -> str:
        return "return to launch (direct)" if self.direct else "return to launch (planned)"


class ObserveTarget(Behavior):
    """Hold a vantage point until the camera has banked enough dwell."""

    name = "observe"

    def __init__(
        self,
        target: SurveillanceTarget,
        coverage: CoverageTracker,
        *,
        timeout_s: float = 45.0,
    ) -> None:
        super().__init__()
        if timeout_s <= 0:
            raise ValueError("observation timeout must be positive")
        self.target = target
        self.coverage = coverage
        self.timeout = timeout_s
        self._deadline: float | None = None

    async def on_start(self, vehicle: VehicleBackend, telemetry: Telemetry) -> None:
        self._deadline = telemetry.elapsed_s + self.timeout
        await vehicle.goto(telemetry.position, 1.0)  # station-keep
        # Turn to face it. The camera is rigidly yawed with the airframe, so
        # arriving at a vantage point without this leaves the lens pointing
        # wherever the vehicle happened to be travelling -- which observes
        # nothing at all, however good the vantage was.
        await vehicle.look_at(self.target.position)

    def is_complete(self, telemetry: Telemetry) -> bool:
        if self.coverage.is_covered(self.target):
            return True
        # Give up rather than stall the whole mission on one blocked target --
        # a courtyard with no sightline would otherwise consume the battery.
        return self._deadline is not None and telemetry.elapsed_s >= self._deadline

    def reset(self) -> None:
        super().reset()
        self._deadline = None

    def describe(self) -> str:
        return f"observe {self.target.name}"


def build_patrol(
    city: City,
    targets: list[SurveillanceTarget],
    *,
    camera: Camera | None = None,
    planner: GridPlanner | None = None,
    speed_mps: float = 6.0,
    standoff_m: float = 45.0,
    observe_altitude_m: float = 40.0,
    dynamic: DynamicField | None = None,
    hazards: "HazardField | None" = None,
) -> tuple[list[Behavior], CoverageTracker, list[str]]:
    """Compose a full surveillance patrol.

    Returns the behaviour list, a coverage tracker to attach to the mission,
    and the names of any targets that were unobservable from the start -- worth
    surfacing, because "no vantage exists" is a planning answer the operator
    needs before launch, not a surprise at 30% battery.
    """
    cam = camera or Camera()
    nav = planner or GridPlanner(city)
    coverage = CoverageTracker(targets=targets, camera=cam, city=city)

    behaviors: list[Behavior] = []
    unobservable: list[str] = []

    for t in targets:
        vantage = observation_point(
            t, cam, city,
            standoff_m=standoff_m,
            altitude_m=observe_altitude_m,
            margin_m=nav.margin,
        )
        if vantage is None:
            unobservable.append(t.name)
            continue
        behaviors.append(
            NavigateTo(vantage, nav, speed_mps=speed_mps, label=t.name,
                       dynamic=dynamic, hazards=hazards)
        )
        behaviors.append(ObserveTarget(t, coverage))

    return behaviors, coverage, unobservable
