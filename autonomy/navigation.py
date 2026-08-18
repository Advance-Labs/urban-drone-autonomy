"""Path planning through an urban environment.

A* over a coarse 3D voxel grid, followed by string-pulling to remove the
staircase artefacts that grid search always produces.

Why a grid and not RRT: the search space here is small (a few hundred metres,
a few tens of thousands of cells) and grid A* is deterministic, complete, and
optimal on that graph. RRT wins in high dimensions or with vehicle dynamics in
the loop; neither applies to a multirotor that can stop and turn in place.

Why string-pulling matters: raw A* output follows cell boundaries, so a
straight diagonal run across open ground comes out as a staircase of 26 tiny
segments. Flying that literally means 26 decelerations. Collapsing waypoints
that have clear line of sight between them recovers the straight line and cuts
flight time substantially.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

from typing import TYPE_CHECKING

from .types import Vec3
from .world import City

if TYPE_CHECKING:
    from .dynamic import DynamicField
    from .hazards import HazardField
    from .thin_obstacles import ThinObstacleField


@dataclass(frozen=True, slots=True)
class PlanResult:
    """A path plus enough context to explain a failure."""

    waypoints: list[Vec3]
    expanded: int
    ok: bool
    reason: str = ""

    @property
    def length_m(self) -> float:
        return sum(
            (self.waypoints[i + 1] - self.waypoints[i]).norm
            for i in range(len(self.waypoints) - 1)
        )


class GridPlanner:
    """3D A* with obstacle inflation and line-of-sight smoothing."""

    def __init__(
        self,
        city: City,
        *,
        cell_m: float = 6.0,
        margin_m: float = 4.0,
        min_altitude_m: float = 6.0,
        max_altitude_m: float = 90.0,
        climb_penalty: float = 1.6,
        thin_margin_m: float = 10.0,
        hazard_weight: float = 3.0,
        tracking_error_m: float = 3.5,
    ) -> None:
        """
        Args:
            cell_m: grid resolution. Smaller finds tighter routes but costs
                cubically in search time.
            margin_m: standoff inflated around every building. Must exceed the
                vehicle's own radius plus position error, or a "clear" path
                still clips masonry.
            min_altitude_m: floor, to keep planned routes off the pavement
                where people are.
            max_altitude_m: ceiling. Under Canadian VLOS rules this stays well
                below 122 m AGL.
            climb_penalty: vertical motion costs more than horizontal, because
                on a multirotor it genuinely does. Without this the planner
                happily pogos over every obstacle instead of going around.
            tracking_error_m: how far the FLOWN path is expected to deviate
                from the planned one -- guidance lag, arrival radius, and
                corner-cutting between waypoints. Added to every hazard
                separation check, because a plan that is legal by exactly the
                minimum is flown in violation of it. Measured at ~0.7 m of
                overshoot for a 5 m grid at 5 m/s; 3.5 m is a deliberately
                generous allowance around people.
            hazard_weight: how strongly to prefer distance from people beyond
                the legal minimum. Multiplies `HazardField.risk_cost` into the
                move cost, so a detour of up to roughly this many cell-lengths
                is worth taking to gain a comfortable margin.
            thin_margin_m: standoff around INFERRED conductors. Deliberately
                much larger than `margin_m`: a mapped building is where the map
                says it is, but an inferred wire carries the triangulation
                error of the poles it was derived from, plus catenary sag we
                only estimated. Clearing it by less than the map error is not
                clearing it at all.
        """
        if cell_m <= 0:
            raise ValueError("cell size must be positive")
        if margin_m < 0:
            raise ValueError("margin cannot be negative")
        if max_altitude_m <= min_altitude_m:
            raise ValueError("altitude ceiling must exceed the floor")

        self.city = city
        self.cell = cell_m
        self.margin = margin_m
        self.min_alt = min_altitude_m
        self.max_alt = max_altitude_m
        self.climb_penalty = climb_penalty
        self.thin_margin = thin_margin_m
        self.hazard_weight = hazard_weight
        self.tracking_error = tracking_error_m
        self._neighbours = self._build_neighbourhood()

    # -- public -----------------------------------------------------------

    def plan(
        self,
        start: Vec3,
        goal: Vec3,
        *,
        max_expansions: int = 120_000,
        dynamic: "DynamicField | None" = None,
        thin: "ThinObstacleField | None" = None,
        hazards: "HazardField | None" = None,
        t_now: float = 0.0,
        speed_mps: float | None = None,
    ) -> PlanResult:
        """Find a collision-free route from `start` to `goal`.

        With a `DynamicField`, moving obstacles are avoided in space-time: a
        cell is judged against where the obstacle will be at the time the
        vehicle would actually arrive there, estimated from distance-so-far
        along the search. Planning against current positions is the classic
        error -- it produces a route straight through where a mover is heading.

        With a `ThinObstacleField` -- typically the live output of the camera
        perception pipeline -- inferred conductors become no-fly volumes. These
        are avoided by a WIDER margin than buildings, because a building's
        position is known and an inferred wire's is not: the standoff has to
        cover the mapping error as well as the airframe. See `thin_margin`.

        With a `HazardField`, people and vehicles contribute BOTH a hard
        constraint and a soft cost. The hard part is collision and legal
        separation; the soft part biases the search away from people even where
        it would be legal to fly closer. Binary constraints alone produce a
        route that hugs the legal minimum exactly, which nobody would fly on
        purpose.
        """
        cruise = speed_mps or 6.0
        start_c = self._to_cell(start)
        goal_c = self._to_cell(goal)

        if not self._cell_free(goal_c):
            nudged = self._nearest_free(goal_c)
            if nudged is None:
                return PlanResult([], 0, False, "goal is inside an obstacle")
            goal_c = nudged
        if not self._cell_free(start_c):
            nudged = self._nearest_free(start_c)
            if nudged is None:
                return PlanResult([], 0, False, "start is inside an obstacle")
            start_c = nudged

        came: dict[tuple, tuple] = {}
        g = {start_c: 0.0}
        open_heap = [(self._h(start_c, goal_c), start_c)]
        seen: set[tuple] = set()
        expanded = 0

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current in seen:
                continue
            seen.add(current)
            expanded += 1

            if current == goal_c:
                raw = self._reconstruct(came, current)
                # Smoothing must respect movers too, or string-pulling
                # cheerfully reconnects across the gap the search avoided.
                return PlanResult(
                    self.smooth(
                        raw, dynamic=dynamic, thin=thin, hazards=hazards,
                        t_now=t_now, speed_mps=cruise,
                    ),
                    expanded, True,
                )

            if expanded > max_expansions:
                return PlanResult([], expanded, False, "search budget exhausted")

            for step, cost in self._neighbours:
                nxt = (current[0] + step[0], current[1] + step[1], current[2] + step[2])
                if nxt in seen or not self._cell_free(nxt):
                    continue
                if thin is not None and thin.cables:
                    if thin.solid_at(self._to_point(nxt), self.thin_margin):
                        continue
                hazard_penalty = 0.0
                if hazards is not None and not hazards.is_empty:
                    arrival = t_now + g[current] / max(cruise, 0.1)
                    point = self._to_point(nxt)
                    if hazards.blocked_at(
                        point, arrival, self.margin + self.tracking_error
                    ):
                        continue
                    hazard_penalty = hazards.risk_cost(point, arrival) * self.hazard_weight
                if dynamic is not None and dynamic.obstacles:
                    # g[current] is distance travelled, so g/speed is the
                    # arrival time -- which is when the mover matters.
                    arrival = t_now + g[current] / max(cruise, 0.1)
                    if dynamic.blocker_at(self._to_point(nxt), arrival, self.margin):
                        continue
                # Diagonal moves must not cut through a corner that both
                # orthogonal neighbours block -- the classic grid-planner bug
                # that produces paths threading solid walls.
                if not self._corner_clear(current, nxt):
                    continue
                tentative = g[current] + cost * self.cell + hazard_penalty * self.cell
                if tentative < g.get(nxt, math.inf):
                    g[nxt] = tentative
                    came[nxt] = current
                    heapq.heappush(open_heap, (tentative + self._h(nxt, goal_c), nxt))

        return PlanResult([], expanded, False, "no route exists")

    def smooth(
        self,
        path: list[Vec3],
        *,
        dynamic: "DynamicField | None" = None,
        thin: "ThinObstacleField | None" = None,
        hazards: "HazardField | None" = None,
        t_now: float = 0.0,
        speed_mps: float = 6.0,
    ) -> list[Vec3]:
        """String-pulling: keep only waypoints the path actually needs.

        Walks forward from each kept waypoint to the furthest later one still
        reachable in a straight line, and drops everything between.
        """
        if len(path) <= 2:
            return path
        out = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1:
                thin_ok = (
                    thin is None
                    or not thin.cables
                    or thin.segment_clear(path[i], path[j], self.thin_margin)
                )
                hazard_ok = hazards is None or hazards.is_empty or self._hazard_clear(
                    path[i], path[j], hazards, t_now, speed_mps
                )
                thin_ok = thin_ok and hazard_ok
                if thin_ok and self.city.segment_clear(path[i], path[j], self.margin) and (
                    dynamic is None
                    or not dynamic.obstacles
                    or dynamic.segment_clear(
                        path[i], path[j], t_now, speed_mps, self.margin
                    )
                ):
                    break
                j -= 1
            out.append(path[j])
            i = j
        return out

    def _hazard_clear(self, a: Vec3, b: Vec3, hazards, t_start: float,
                      speed_mps: float, step_m: float = 3.0) -> bool:
        """Space-time sweep against people and vehicles.

        Smoothing without this happily reconnects straight across the
        exclusion zone the search just detoured around.
        """
        span = (b - a).norm
        steps = max(2, math.ceil(span / step_m))
        for i in range(steps + 1):
            frac = i / steps
            arrival = t_start + (span * frac) / max(speed_mps, 0.1)
            if hazards.blocked_at(
                a + (b - a) * frac, arrival, self.margin + self.tracking_error
            ):
                return False
        return True

    def reachable(self, a: Vec3, b: Vec3) -> bool:
        """Direct line of flight between two points, standoff included."""
        return self.city.segment_clear(a, b, self.margin)

    # -- internals --------------------------------------------------------

    def _build_neighbourhood(self) -> list[tuple[tuple[int, int, int], float]]:
        """26-connected moves, each with its Euclidean cost.

        Vertical components are scaled by `climb_penalty` so the search prefers
        routing around a building to climbing over it, which is both faster and
        far better for battery.
        """
        moves = []
        for dn in (-1, 0, 1):
            for de in (-1, 0, 1):
                for dd in (-1, 0, 1):
                    if dn == de == dd == 0:
                        continue
                    cost = math.sqrt(dn * dn + de * de + (dd * self.climb_penalty) ** 2)
                    moves.append(((dn, de, dd), cost))
        return moves

    def _to_cell(self, p: Vec3) -> tuple[int, int, int]:
        return (
            round(p.north / self.cell),
            round(p.east / self.cell),
            round(max(self.min_alt, min(self.max_alt, p.altitude)) / self.cell),
        )

    def _to_point(self, c: tuple[int, int, int]) -> Vec3:
        return Vec3(c[0] * self.cell, c[1] * self.cell, -c[2] * self.cell)

    def _cell_free(self, c: tuple[int, int, int]) -> bool:
        p = self._to_point(c)
        if not (self.min_alt - 1e-6 <= p.altitude <= self.max_alt + 1e-6):
            return False
        if not self.city.in_bounds(p):
            return False
        return self.city.is_free(p, self.margin)

    def _corner_clear(self, a: tuple, b: tuple) -> bool:
        """Reject diagonal moves that squeeze between two blocked cells."""
        dn, de, dd = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        if abs(dn) + abs(de) + abs(dd) < 2:
            return True  # orthogonal move, nothing to cut
        for axis in range(3):
            step = [0, 0, 0]
            step[axis] = (dn, de, dd)[axis]
            if step == [0, 0, 0]:
                continue
            side = (a[0] + step[0], a[1] + step[1], a[2] + step[2])
            if self._cell_free(side):
                return True
        return False

    def _h(self, a: tuple, b: tuple) -> float:
        """Euclidean heuristic, scaled to metres. Admissible because the
        climb penalty only ever makes real moves cost more, never less."""
        return (
            math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)
            * self.cell
        )

    def _nearest_free(self, c: tuple, radius: int = 6) -> tuple | None:
        """Spiral outward for a usable cell near a blocked endpoint.

        Endpoints land inside inflated obstacles constantly -- a surveillance
        target is usually a doorway on a wall -- so failing outright would make
        the planner useless in exactly its intended job.
        """
        best, best_d = None, math.inf
        for dn in range(-radius, radius + 1):
            for de in range(-radius, radius + 1):
                for dd in range(-radius, radius + 1):
                    cand = (c[0] + dn, c[1] + de, c[2] + dd)
                    if not self._cell_free(cand):
                        continue
                    d = dn * dn + de * de + dd * dd
                    if d < best_d:
                        best, best_d = cand, d
        return best

    def _reconstruct(self, came: dict, node: tuple) -> list[Vec3]:
        path = [node]
        while node in came:
            node = came[node]
            path.append(node)
        return [self._to_point(c) for c in reversed(path)]
