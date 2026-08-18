"""Moving obstacles, space-time collision checking, and replanning."""

from __future__ import annotations

import pytest

from autonomy import (
    City,
    DynamicField,
    GridPlanner,
    MovingObstacle,
    Vec3,
    crossing_traffic,
)


def mover(**kw) -> MovingObstacle:
    base = dict(
        waypoints=[Vec3(0, -50, -20), Vec3(0, 50, -20)],
        speed_mps=10.0, radius_m=5.0, name="m", loop=False,
    )
    return MovingObstacle(**(base | kw))


class TestMotion:
    def test_rejects_degenerate_definitions(self):
        with pytest.raises(ValueError):
            MovingObstacle(waypoints=[Vec3()], speed_mps=5)
        with pytest.raises(ValueError):
            mover(speed_mps=0)
        with pytest.raises(ValueError):
            mover(radius_m=-1)

    def test_starts_at_the_first_waypoint(self):
        assert mover().position_at(0.0).east == pytest.approx(-50)

    def test_moves_at_the_stated_speed(self):
        m = mover(speed_mps=10.0)
        travelled = (m.position_at(3.0) - m.position_at(0.0)).norm
        assert travelled == pytest.approx(30.0, abs=0.1)

    def test_position_is_a_pure_function_of_time(self):
        """Prediction and collision checking both ask about the future, so
        querying must not mutate any internal cursor."""
        m = mover()
        first = m.position_at(4.0)
        m.position_at(9.0)
        assert m.position_at(4.0) == first

    def test_ping_pong_reverses_instead_of_teleporting(self):
        m = mover(speed_mps=10.0, loop=False)  # 100 m span, 10 s each way
        assert m.position_at(10.0).east == pytest.approx(50, abs=0.5)
        assert m.position_at(15.0).east == pytest.approx(0, abs=0.5)
        assert m.position_at(20.0).east == pytest.approx(-50, abs=0.5)

    def test_loop_wraps_around(self):
        m = MovingObstacle(
            waypoints=[Vec3(0, 0, -20), Vec3(0, 30, -20), Vec3(30, 30, -20)],
            speed_mps=10.0, loop=True,
        )
        circuit = m.circuit_length
        assert (m.position_at(circuit / 10.0) - m.position_at(0.0)).norm < 0.5

    def test_velocity_points_along_travel(self):
        v = mover(speed_mps=8.0).velocity_at(2.0)
        assert v.east > 7.0 and abs(v.north) < 0.5


class TestSpaceTime:
    def test_blocker_reported_by_name(self):
        f = DynamicField(obstacles=[mover(name="intruder")])
        assert f.blocker_at(Vec3(0, -50, -20), 0.0) == "intruder"
        assert f.blocker_at(Vec3(0, 50, -20), 0.0) is None

    def test_segment_checks_where_the_obstacle_WILL_be(self):
        """The classic mistake is checking current positions, which clears a
        path the mover occupies by the time you arrive."""
        f = DynamicField(obstacles=[mover(speed_mps=10.0)])
        # The mover starts at east -50 and is at east 0 after 5 s.
        # Our vehicle takes 5 s to cover 30 m at 6 m/s toward that spot.
        a, b = Vec3(-30, 0, -20), Vec3(0, 0, -20)
        assert f.blocker_at(b, 0.0) is None, "clear right now"
        assert not f.segment_clear(a, b, t_start=0.0, speed_mps=6.0), (
            "but occupied by arrival time"
        )

    def test_segment_clear_when_the_mover_has_passed(self):
        f = DynamicField(obstacles=[mover(speed_mps=10.0)])
        # By t=20 the mover is back at east -50, far from east +40.
        assert f.segment_clear(
            Vec3(-10, 40, -20), Vec3(10, 40, -20), t_start=20.0, speed_mps=6.0
        )

    def test_path_clear_reports_the_blocking_obstacle(self):
        f = DynamicField(obstacles=[mover(name="X", speed_mps=10.0)])
        route = [Vec3(0, 0, -20), Vec3(0, 20, -20)]
        blocked = f.path_clear(route, 0, Vec3(-30, 0, -20), 0.0, 6.0)
        assert blocked == "X"

    def test_empty_field_never_blocks(self):
        assert DynamicField().path_clear([Vec3(9, 9, -9)], 0, Vec3(), 0.0, 6.0) is None


class TestPlanningAroundMovers:
    def test_plan_avoids_a_mover_in_space_time(self):
        city = City(extent_m=200)
        f = DynamicField(obstacles=[
            MovingObstacle(
                waypoints=[Vec3(0, 0, -20), Vec3(0, 1, -20)],
                speed_mps=0.1, radius_m=25.0, name="parked", loop=False,
            )
        ])
        planner = GridPlanner(city, cell_m=6, margin_m=2)
        result = planner.plan(
            Vec3(-80, 0, -20), Vec3(80, 0, -20),
            dynamic=f, t_now=0.0, speed_mps=6.0,
        )
        assert result.ok, result.reason
        # Every waypoint must clear the (effectively static) blob.
        for w in result.waypoints:
            assert f.blocker_at(w, 0.0, 2.0) is None

    def test_smoothing_does_not_reconnect_through_a_mover(self):
        """String-pulling must respect movers too, or it cheerfully collapses
        the detour the search just computed."""
        city = City(extent_m=200)
        f = DynamicField(obstacles=[
            MovingObstacle(
                waypoints=[Vec3(0, 0, -20), Vec3(0, 1, -20)],
                speed_mps=0.05, radius_m=22.0, name="blob", loop=False,
            )
        ])
        planner = GridPlanner(city, cell_m=6, margin_m=2)
        result = planner.plan(
            Vec3(-70, 0, -20), Vec3(70, 0, -20),
            dynamic=f, t_now=0.0, speed_mps=6.0,
        )
        assert result.ok
        for a, b in zip(result.waypoints, result.waypoints[1:]):
            assert f.segment_clear(a, b, 0.0, 6.0, 2.0), "smoothed through the blob"

    def test_crossing_traffic_helper_is_sane(self):
        t = crossing_traffic(north=30, east_span=100, altitude_m=25, speed_mps=8)
        assert t.position_at(0.0).altitude == pytest.approx(25)
        assert not t.loop, "should ping-pong, not teleport back across"
