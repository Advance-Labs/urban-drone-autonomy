"""A dependency-free multirotor stand-in, good enough to develop autonomy against.

Scope, stated honestly: this is NOT a flight dynamics model. There is no
attitude, no rotor thrust, no aerodynamics. It is a velocity-controlled point
mass with a first-order lag, which reproduces the behaviour the autonomy layer
actually reacts to -- finite acceleration, finite top speed, overshoot on
aggressive setpoints, and a battery that drains faster when you work harder.

That is deliberately the right fidelity for layer-4 work. A behaviour state
machine that handles this correctly will handle PX4 SITL correctly, because the
things it gets wrong (arriving before the vehicle does, assuming instant stops)
are exactly the things this model punishes. Use PX4 SITL when you need to trust
the numbers; use this when you need to iterate on logic in under a second.
"""

from __future__ import annotations

import math

from typing import TYPE_CHECKING

from ..types import FlightMode, Telemetry, Vec3

if TYPE_CHECKING:
    from ..world import City


class KinematicBackend:
    """Velocity-controlled point-mass quadrotor with battery accounting."""

    def __init__(
        self,
        *,
        max_speed_mps: float = 8.0,
        accel_tau_s: float = 0.6,
        arrival_radius_m: float = 0.8,
        battery_endurance_s: float = 900.0,
        hover_current_fraction: float = 0.55,
        wind: Vec3 | None = None,
        city: "City | None" = None,
        airframe_radius_m: float = 0.4,
        start: Vec3 | None = None,
    ) -> None:
        """
        Args:
            max_speed_mps: horizontal/vertical speed ceiling.
            accel_tau_s: first-order lag on velocity. Bigger is more sluggish.
                0.6 s is representative of a 5" quad in position-hold mode.
            arrival_radius_m: how close counts as "at the setpoint".
            battery_endurance_s: seconds of continuous hover on a full pack.
            hover_current_fraction: share of draw spent just staying airborne;
                the remainder scales with how fast you are actually moving.
            wind: steady wind vector, metres/second, pointing the way the air
                moves. Adds to ground velocity and costs charge to fight,
                which is what makes wind-aware landing selection meaningful
                rather than decorative.
            city: optional obstacle world. When supplied the vehicle can crash
                into buildings, which is what makes obstacle avoidance a
                falsifiable claim rather than a decorative path shape.
            start: launch position. Defaults to the map origin, which is fine
                for a procedural city built around it and wrong for a real one
                -- an arbitrary lat/lon frequently lands on a rooftop, and a
                vehicle that spawns inside a wall crashes on the first tick.
            airframe_radius_m: collision hull. Uses the vehicle's own size,
                not the planner's standoff margin -- if they were the same
                number, a planner bug that shaves the margin could never
                produce a crash and the test would be vacuous.
        """
        if max_speed_mps <= 0:
            raise ValueError("max_speed_mps must be positive")
        if accel_tau_s <= 0:
            raise ValueError("accel_tau_s must be positive")
        if battery_endurance_s <= 0:
            raise ValueError("battery_endurance_s must be positive")
        if not 0.0 <= hover_current_fraction <= 1.0:
            raise ValueError("hover_current_fraction must be within 0..1")

        self.max_speed = max_speed_mps
        self.accel_tau = accel_tau_s
        self.arrival_radius = arrival_radius_m
        self.endurance = battery_endurance_s
        self.hover_fraction = hover_current_fraction
        self.wind = wind or Vec3()
        self.city = city
        self.airframe_radius = airframe_radius_m
        self.crashed_into: str | None = None

        self._pos = start or Vec3()
        self._vel = Vec3()
        self._setpoint: Vec3 | None = None
        self._commanded_speed = max_speed_mps
        self._heading = 0.0
        self._battery = 1.0
        self._mode = FlightMode.GROUNDED
        self._armed = False
        self._elapsed = 0.0
        self._closed = False
        self._link_up = True
        self._look_at: Vec3 | None = None
        self.yaw_rate_dps = 90.0
        # Set by `land`/`return_to_launch` so `step` knows to disarm on touchdown.
        self._disarm_on_ground = False

    # -- VehicleBackend ----------------------------------------------------

    async def connect(self) -> None:
        self._closed = False

    async def arm(self) -> None:
        if self._closed:
            raise RuntimeError("cannot arm: backend is closed")
        if self._battery <= 0.0:
            raise RuntimeError("cannot arm: battery empty")
        self._armed = True

    async def takeoff(self, altitude_m: float) -> None:
        if not self._armed:
            raise RuntimeError("cannot take off: not armed")
        if altitude_m <= 0:
            raise ValueError("takeoff altitude must be positive")
        self._mode = FlightMode.TAKING_OFF
        self._disarm_on_ground = False
        self._setpoint = Vec3(self._pos.north, self._pos.east, -altitude_m)
        self._commanded_speed = min(self.max_speed, 3.0)  # climb gently

    async def goto(self, target: Vec3, speed_mps: float) -> None:
        if not self._armed:
            raise RuntimeError("cannot goto: not armed")
        if speed_mps <= 0:
            raise ValueError("speed must be positive")
        self._mode = FlightMode.FLYING
        self._disarm_on_ground = False
        self._setpoint = target
        self._commanded_speed = min(speed_mps, self.max_speed)

    async def look_at(self, point: Vec3 | None) -> None:
        self._look_at = point

    async def land(self) -> None:
        self._look_at = None
        self._mode = FlightMode.LANDING
        self._disarm_on_ground = True
        self._setpoint = Vec3(self._pos.north, self._pos.east, 0.0)
        self._commanded_speed = min(self.max_speed, 1.5)  # descend gently

    async def return_to_launch(self) -> None:
        self._mode = FlightMode.RETURNING
        self._disarm_on_ground = True
        # Match PX4: transit home at the current altitude, then descend.
        self._setpoint = Vec3(0.0, 0.0, min(self._pos.down, -2.0))
        self._commanded_speed = self.max_speed

    async def telemetry(self) -> Telemetry:
        return Telemetry(
            position=self._pos,
            # Ground velocity, matching what GPS reports on a real vehicle.
            velocity=self._ground_velocity(),
            heading_deg=self._heading,
            battery_remaining=self._battery,
            mode=self._mode,
            armed=self._armed,
            link_ok=not self._closed and self._link_up,
            elapsed_s=self._elapsed,
            wind=self.wind,
        )

    def set_link(self, up: bool) -> None:
        """Force the control link up or down. For exercising link-loss policy."""
        self._link_up = up

    # Largest internal integration step. Velocity uses an exponential lag that
    # is stable at any dt, but *position* is not: at dt=2 s a vehicle doing
    # 3 m/s jumps 6 m, sailing past a target 4 m away and then oscillating.
    # Substepping bounds that error regardless of what the caller passes,
    # which matters because the PX4 backend ticks at whatever rate the
    # network delivers rather than a rate we choose.
    _MAX_SUBSTEP_S = 0.05

    async def step(self, dt: float) -> None:
        """Integrate one tick of the vehicle model.

        Subdivides internally, so a caller ticking at 0.5 Hz gets the same
        trajectory as one ticking at 100 Hz.
        """
        if dt <= 0:
            raise ValueError("dt must be positive")

        substeps = max(1, math.ceil(dt / self._MAX_SUBSTEP_S))
        h = dt / substeps
        for _ in range(substeps):
            if self.crashed_into is not None:
                break
            previous = self._pos
            self._integrate(h)
            self._check_collision(previous)

        self._elapsed += dt
        self._drain_battery(dt)
        self._update_mode()

    def _integrate(self, h: float) -> None:
        """Advance position and velocity by one bounded substep.

        `_vel` is airspeed -- what the vehicle achieves relative to the air.
        Ground velocity adds wind on top, which is why a headwind slows the
        transit and a tailwind speeds it: the controller commands airspeed but
        the world moves at ground speed.
        """
        desired = self._desired_velocity()
        # First-order lag toward the desired velocity. The exponential form is
        # stable at any h, unlike a naive `v += (target - v) * h / tau`
        # which diverges once h exceeds tau.
        blend = 1.0 - math.exp(-h / self.accel_tau)
        self._vel = self._vel + (desired - self._vel) * blend
        self._pos = self._pos + (self._vel + self.wind) * h

        if self._pos.down > 0.0:  # never sink below ground
            self._pos = Vec3(self._pos.north, self._pos.east, 0.0)
            self._vel = Vec3(self._vel.north, self._vel.east, 0.0)

        self._update_heading(h)

    def _update_heading(self, h: float) -> None:
        """Slew the airframe heading, either toward a look-at lock or along
        the direction of travel.

        Slewed rather than snapped: a real airframe yaws at a finite rate, and
        instant yaw would let a camera sweep a whole street in one tick and
        claim coverage it never actually had time to record.
        """
        if self._look_at is not None:
            offset = self._look_at - self._pos
            if offset.horizontal_distance_to(Vec3()) > 0.5:
                desired = math.degrees(math.atan2(offset.east, offset.north)) % 360.0
                # Shortest way round the circle.
                delta = (desired - self._heading + 180.0) % 360.0 - 180.0
                limit = self.yaw_rate_dps * h
                self._heading = (
                    self._heading + max(-limit, min(limit, delta))
                ) % 360.0
            return

        ground = self._ground_velocity()
        if ground.horizontal_distance_to(Vec3()) > 0.3:
            self._heading = math.degrees(math.atan2(ground.east, ground.north)) % 360.0

    def _ground_velocity(self) -> Vec3:
        """What GPS would report: airspeed plus the air's own motion."""
        return self._vel + self.wind

    def _check_collision(self, previous: Vec3) -> None:
        """Detect a strike, testing the swept segment rather than the endpoint.

        A point test at 8 m/s with a 0.05 s substep can jump 0.4 m, which is
        enough to tunnel through a thin wall and land on the far side with no
        collision ever registered.
        """
        if self.city is None:
            return
        if self.city.segment_clear(previous, self._pos, self.airframe_radius, step_m=0.3):
            return
        hit = self.city.occupant(self._pos, self.airframe_radius)
        self.crashed_into = hit.name if hit else "terrain"
        self._armed = False
        self._vel = Vec3()
        self._setpoint = None

    async def close(self) -> None:
        self._closed = True

    # -- internals ---------------------------------------------------------

    def _desired_velocity(self) -> Vec3:
        """Guidance toward the setpoint. Returns a desired *airspeed*.

        Two things are happening here.

        The `min(1.0, distance / brake_radius)` term stops the vehicle sailing
        past every waypoint: closing speed tapers as it arrives.

        The wind subtraction is feedforward, and it is not optional. Guidance
        computes a desired *ground* velocity, but the vehicle only controls
        airspeed -- ground velocity is airspeed plus wind. Without the
        correction, proportional guidance parks the vehicle at the distance
        where its tapered command exactly equals the wind and holds it there
        forever: at 8.4 m/s over a 10 m brake radius in a 5.4 m/s wind, that
        is 6.5 m downwind of every setpoint, well outside any sane arrival
        radius. A P-controller cannot reject a constant disturbance; real
        position controllers feed the wind estimate forward for exactly this
        reason, which avoids the windup an integral term would bring.
        """
        if self._setpoint is None or not self._armed:
            return Vec3()

        error = self._setpoint - self._pos
        distance = error.norm
        if distance < 1e-6:
            # Still need to hold station against the wind.
            return self._clamp_airspeed(self.wind * -1.0)

        brake_radius = max(self.arrival_radius, self._commanded_speed * self.accel_tau * 2.0)
        speed = self._commanded_speed * min(1.0, distance / brake_radius)
        desired_ground = error.normalised() * speed
        return self._clamp_airspeed(desired_ground - self.wind)

    def _clamp_airspeed(self, v: Vec3) -> Vec3:
        """Cap a commanded airspeed at what the airframe can actually do."""
        return v if v.norm <= self.max_speed else v.normalised() * self.max_speed

    def _drain_battery(self, dt: float) -> None:
        if not self._armed:
            return
        effort = self._vel.norm / self.max_speed
        draw = self.hover_fraction + (1.0 - self.hover_fraction) * effort
        self._battery = max(0.0, self._battery - (dt / self.endurance) * draw)

    def _update_mode(self) -> None:
        if not self._armed:
            self._mode = FlightMode.GROUNDED
            return

        on_ground = self._pos.altitude < 0.15 and self._ground_velocity().norm < 0.2
        if on_ground and self._disarm_on_ground:
            self._armed = False
            self._mode = FlightMode.GROUNDED
            self._setpoint = None
            return

        if self._setpoint is not None and self.at_setpoint():
            if self._mode in (FlightMode.TAKING_OFF, FlightMode.FLYING):
                self._mode = FlightMode.HOLDING

    def at_setpoint(self) -> bool:
        """True once the vehicle is inside the arrival radius and settled.

        Settled means low *ground* speed. In steady wind a station-keeping
        vehicle crabs into it, so airspeed stays non-zero even while it holds
        position perfectly -- judging on airspeed would mean never arriving.
        """
        if self._setpoint is None:
            return True
        return (
            (self._setpoint - self._pos).norm <= self.arrival_radius
            and self._ground_velocity().norm < 0.5
        )
