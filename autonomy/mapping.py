"""Turning noisy 2D detections into a 3D obstacle map you can plan against.

Unlike `perception.py`, none of this is simulated. It consumes `Detection2D`
records and would run unchanged on a real detector's output.

## The idea

Detecting a conductor from a camera is close to hopeless -- at 40 m a 12 mm
wire is a third of a pixel wide. Detecting the POLES that carry it is easy:
tall, rigid, vertical, usually silhouetted, and detectable to well past 100 m.

So do not try to see the wire. See the poles, work out which ones are joined,
and infer the catenary hanging between them. The vehicle then avoids a wire it
never observed. This is what makes a camera-only budget airframe viable where
the lidar analysis said you needed a four-thousand-dollar sensor.

## The three steps

1. `LandmarkTracker` accumulates bearing observations of each object across
   frames and triangulates a 3D position. A single monocular frame gives a ray,
   not a point; two rays from sufficiently separated viewpoints give a fix.
   Vehicle motion is what supplies the baseline -- the same principle as stereo,
   with time standing in for a second lens.

2. `StructureGraph` decides which poles are connected. Distribution poles run
   in lines at regular spacing, so the prior is strong: adjacent poles at a
   plausible span, roughly collinear with the local run, are almost certainly
   joined. Any wire detections that happen to land near the span raise
   confidence, but are never required.

3. Confident spans become catenaries, which become obstacles.

## Where this is weak, stated plainly

Triangulation needs baseline: flying straight at a pole gives almost no
parallax and the range estimate stays poor until you move laterally. Span
inference assumes a regular distribution line and will connect two unrelated
poles that happen to sit at a typical spacing. And a pole detected but never
triangulated contributes nothing. Each is a real failure mode, and each is
why spans carry a confidence rather than a boolean.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .perception import Detection2D, DetectorModel, ObjectClass
from .thin_obstacles import Cable, ThinObstacleField
from .types import Vec3


@dataclass(slots=True)
class Observation:
    """One bearing ray: where we were, and which way the object lay."""

    eye: Vec3
    bearing_deg: float
    elevation_deg: float
    confidence: float
    t: float

    def direction(self) -> Vec3:
        """Unit vector along the ray, in NED."""
        yaw = math.radians(self.bearing_deg)
        pitch = math.radians(self.elevation_deg)
        h = math.cos(pitch)
        return Vec3(h * math.cos(yaw), h * math.sin(yaw), -math.sin(pitch))


@dataclass(slots=True)
class Landmark:
    """An object being triangulated from repeated observations."""

    cls: ObjectClass
    observations: list[Observation] = field(default_factory=list)
    position: Vec3 | None = None
    confidence: float = 0.0
    # Largest distance between any two observing viewpoints. This is the
    # single best predictor of whether the fix is trustworthy.
    baseline_m: float = 0.0

    @property
    def observation_count(self) -> int:
        return len(self.observations)

    @property
    def is_fixed(self) -> bool:
        return self.position is not None


def _triangulate(observations: list[Observation]) -> tuple[Vec3, float] | None:
    """Least-squares intersection of bearing rays.

    Minimises the summed squared distance from the point to every ray, which
    has a closed form: sum of (I - dd^T) matrices, solved against the same
    matrices applied to each ray origin. Returns the point and the baseline.

    Rays that are nearly parallel produce a near-singular system -- that is
    precisely the flying-straight-at-it case -- so the determinant is checked
    rather than trusted.
    """
    if len(observations) < 2:
        return None

    baseline = 0.0
    for i, a in enumerate(observations):
        for b in observations[i + 1:]:
            baseline = max(baseline, (b.eye - a.eye).norm)
    if baseline < 1.0:
        return None  # no parallax, no fix

    # Accumulate A = sum(I - d d^T), b = sum((I - d d^T) o)
    a_mat = [[0.0] * 3 for _ in range(3)]
    b_vec = [0.0, 0.0, 0.0]
    for obs in observations:
        d = obs.direction()
        dv = (d.north, d.east, d.down)
        ov = (obs.eye.north, obs.eye.east, obs.eye.down)
        w = obs.confidence
        for r in range(3):
            for c in range(3):
                m = (1.0 if r == c else 0.0) - dv[r] * dv[c]
                a_mat[r][c] += w * m
                b_vec[r] += w * m * ov[c]

    det = (
        a_mat[0][0] * (a_mat[1][1] * a_mat[2][2] - a_mat[1][2] * a_mat[2][1])
        - a_mat[0][1] * (a_mat[1][0] * a_mat[2][2] - a_mat[1][2] * a_mat[2][0])
        + a_mat[0][2] * (a_mat[1][0] * a_mat[2][1] - a_mat[1][1] * a_mat[2][0])
    )
    if abs(det) < 1e-6:
        return None  # degenerate: rays effectively coincident

    inv = _invert3(a_mat, det)
    point = [sum(inv[r][c] * b_vec[c] for c in range(3)) for r in range(3)]
    return Vec3(point[0], point[1], point[2]), baseline


def _invert3(m: list[list[float]], det: float) -> list[list[float]]:
    """Inverse of a 3x3 matrix via cofactors."""
    c = [
        [
            m[1][1] * m[2][2] - m[1][2] * m[2][1],
            m[0][2] * m[2][1] - m[0][1] * m[2][2],
            m[0][1] * m[1][2] - m[0][2] * m[1][1],
        ],
        [
            m[1][2] * m[2][0] - m[1][0] * m[2][2],
            m[0][0] * m[2][2] - m[0][2] * m[2][0],
            m[0][2] * m[1][0] - m[0][0] * m[1][2],
        ],
        [
            m[1][0] * m[2][1] - m[1][1] * m[2][0],
            m[0][1] * m[2][0] - m[0][0] * m[2][1],
            m[0][0] * m[1][1] - m[0][1] * m[1][0],
        ],
    ]
    return [[c[r][k] / det for k in range(3)] for r in range(3)]


@dataclass
class LandmarkTracker:
    """Associates detections across frames and triangulates them.

    Association is by angular proximity, which is the right criterion for
    bearing-only data: two detections belong to the same object if the rays
    point close to the same way from nearby viewpoints. Position-based
    association would need the position we do not yet have.
    """

    associate_deg: float = 4.0
    min_observations: int = 3
    min_baseline_m: float = 4.0
    min_confidence: float = 0.5
    landmarks: list[Landmark] = field(default_factory=list)

    def ingest(self, eye: Vec3, detections: list[Detection2D]) -> None:
        """Fold one frame of detections into the map."""
        for d in detections:
            if d.confidence < self.min_confidence:
                continue  # weak scores are mostly hallucinations
            obs = Observation(eye, d.bearing_deg, d.elevation_deg, d.confidence, d.frame_t)
            match = self._match(eye, d)
            if match is None:
                self.landmarks.append(Landmark(cls=d.cls, observations=[obs]))
            else:
                match.observations.append(obs)
                self._solve(match)

    def _match(self, eye: Vec3, d: Detection2D) -> Landmark | None:
        """Find the landmark this detection most likely belongs to."""
        best, best_err = None, self.associate_deg
        for lm in self.landmarks:
            if lm.cls is not d.cls:
                continue
            if lm.position is not None:
                # Once fixed, predict where it should appear and compare.
                predicted = self._bearing(eye, lm.position)
                err = abs((predicted - d.bearing_deg + 180.0) % 360.0 - 180.0)
            else:
                last = lm.observations[-1]
                err = abs((last.bearing_deg - d.bearing_deg + 180.0) % 360.0 - 180.0)
                # Reject if the viewpoint has barely moved but the bearing has
                # jumped -- that is a different object, not the same one.
                if (eye - last.eye).norm < 0.5 and err > self.associate_deg / 2:
                    continue
            if err < best_err:
                best, best_err = lm, err
        return best

    def _solve(self, lm: Landmark) -> None:
        if lm.observation_count < self.min_observations:
            return
        result = _triangulate(lm.observations)
        if result is None:
            return
        position, baseline = result
        if baseline < self.min_baseline_m:
            return
        if position.altitude < -2.0:
            return  # underground: a bad fix, not a landmark
        lm.position = position
        lm.baseline_m = baseline
        # Confidence grows with evidence and with geometry, capped so a
        # triangulated landmark never reads as certain.
        mean_conf = sum(o.confidence for o in lm.observations) / lm.observation_count
        support = min(1.0, lm.observation_count / 8.0)
        geometry = min(1.0, baseline / 20.0)
        lm.confidence = round(min(0.97, mean_conf * (0.5 + 0.25 * support + 0.25 * geometry)), 3)

    def fixed(self, cls: ObjectClass | None = None) -> list[Landmark]:
        return [
            lm for lm in self.landmarks
            if lm.is_fixed and (cls is None or lm.cls is cls)
        ]

    @staticmethod
    def _bearing(eye: Vec3, target: Vec3) -> float:
        o = target - eye
        return math.degrees(math.atan2(o.east, o.north)) % 360.0


@dataclass(frozen=True, slots=True)
class Span:
    """An inferred conductor run between two poles."""

    a: Vec3
    b: Vec3
    confidence: float
    corroborated: bool  # a wire detection actually landed near this span

    @property
    def length_m(self) -> float:
        return self.a.horizontal_distance_to(self.b)


@dataclass
class StructureGraph:
    """Infers which poles carry conductors between them.

    The prior does the heavy lifting: distribution poles run in lines at
    regular spacing, so two poles at a plausible span that continue the local
    direction of a run are almost certainly joined.
    """

    min_span_m: float = 25.0
    max_span_m: float = 70.0
    # Height below the pole top where conductors are strung.
    attach_below_top_m: float = 0.8
    sag_ratio: float = 0.03  # sag as a fraction of span; ~3% is typical
    wire_corroboration_m: float = 12.0
    min_span_confidence: float = 0.45

    def infer(
        self,
        poles: list[Landmark],
        wire_landmarks: list[Landmark] | None = None,
    ) -> list[Span]:
        """Build spans from triangulated poles."""
        fixed = [p for p in poles if p.is_fixed]
        wires = [w for w in (wire_landmarks or []) if w.is_fixed]
        spans: list[Span] = []

        for i, a in enumerate(fixed):
            for b in fixed[i + 1:]:
                distance = a.position.horizontal_distance_to(b.position)
                if not self.min_span_m <= distance <= self.max_span_m:
                    continue
                if self._crosses_a_pole(a, b, fixed):
                    continue  # a nearer pole sits between them; not a span

                corroborated = self._wire_support(a, b, wires)
                confidence = self._score(a, b, distance, corroborated)
                if confidence < self.min_span_confidence:
                    continue
                spans.append(
                    Span(a.position, b.position, round(confidence, 3), corroborated)
                )
        return spans

    def _crosses_a_pole(self, a: Landmark, b: Landmark, poles: list[Landmark]) -> bool:
        """Reject a long span that skips over an intervening pole.

        Without this, a run of four poles produces spans between every pair,
        including the full-length one, and the map gains conductors that do
        not exist.
        """
        for other in poles:
            if other is a or other is b:
                continue
            if self._perpendicular_distance(other.position, a.position, b.position) < 8.0:
                between = (
                    min(a.position.north, b.position.north) - 1
                    <= other.position.north
                    <= max(a.position.north, b.position.north) + 1
                    and min(a.position.east, b.position.east) - 1
                    <= other.position.east
                    <= max(a.position.east, b.position.east) + 1
                )
                if between:
                    return True
        return False

    @staticmethod
    def _perpendicular_distance(p: Vec3, a: Vec3, b: Vec3) -> float:
        """Horizontal distance from p to the line ab."""
        ax, ay = a.north, a.east
        bx, by = b.north, b.east
        px, py = p.north, p.east
        dx, dy = bx - ax, by - ay
        denom = math.hypot(dx, dy)
        if denom < 1e-9:
            return math.hypot(px - ax, py - ay)
        return abs(dy * px - dx * py + bx * ay - by * ax) / denom

    def _wire_support(self, a: Landmark, b: Landmark, wires: list[Landmark]) -> bool:
        """Did any wire detection land near the midpoint of this span?"""
        mid = (a.position + b.position) * 0.5
        return any(
            (w.position - mid).norm < self.wire_corroboration_m for w in wires
        )

    def _score(self, a: Landmark, b: Landmark, distance: float, corroborated: bool) -> float:
        """Confidence that these two poles are joined.

        Pole confidence bounds it -- an inferred span can never be more certain
        than the poles it rests on. Typical spacing raises it, corroboration
        raises it further, and a large height mismatch lowers it, since
        conductors run between poles of similar height.
        """
        base = min(a.confidence, b.confidence)
        # Distribution spans cluster near 40 m; score by closeness to that.
        typical = 1.0 - min(1.0, abs(distance - 40.0) / 40.0)
        height_gap = abs(a.position.altitude - b.position.altitude)
        height_ok = 1.0 - min(1.0, height_gap / 6.0)
        score = base * (0.55 + 0.25 * typical + 0.20 * height_ok)
        return min(0.97, score * (1.25 if corroborated else 1.0))

    def to_obstacles(self, spans: list[Span]) -> ThinObstacleField:
        """Convert inferred spans into cables the planner can avoid.

        The conductor hangs below the pole tops and sags at midspan, so the
        obstacle sits meaningfully lower than either endpoint -- which is
        exactly the geometry that catches aircraft crossing at "safe" pole
        height.
        """
        cables = []
        for i, s in enumerate(spans):
            drop = self.attach_below_top_m
            cables.append(
                Cable(
                    start=Vec3(s.a.north, s.a.east, s.a.down + drop),
                    end=Vec3(s.b.north, s.b.east, s.b.down + drop),
                    sag_m=max(0.5, s.length_m * self.sag_ratio),
                    name=f"inferred-{i}",
                )
            )
        return ThinObstacleField(cables=tuple(cables))


@dataclass
class PerceptionPipeline:
    """Detector, tracker, and structure inference, driven from the flight loop.

    Rebuilds the structure graph on a timer rather than every frame. Inference
    is O(poles^2) and the map only changes when a new landmark gets fixed, so
    running it at the control rate would burn CPU to produce the same answer
    dozens of times a second.
    """

    detector: "DetectorModel"
    tracker: LandmarkTracker = field(default_factory=LandmarkTracker)
    graph: StructureGraph = field(default_factory=StructureGraph)
    rebuild_interval_s: float = 2.0
    truth: list[tuple[ObjectClass, Vec3, str]] = field(default_factory=list)

    spans: list[Span] = field(default_factory=list)
    obstacles: ThinObstacleField = field(default_factory=ThinObstacleField)
    _last_build: float = -1e9
    _frames: int = 0
    _detections: int = 0

    def step(self, eye: Vec3, heading_deg: float, t: float) -> list[Detection2D]:
        """One perception frame. Returns the raw detections, for logging."""
        detections = self.detector.observe(eye, heading_deg, self.truth, t)
        self.tracker.ingest(eye, detections)
        self._frames += 1
        self._detections += len(detections)

        if t - self._last_build >= self.rebuild_interval_s:
            self._last_build = t
            self.rebuild()
        return detections

    def rebuild(self) -> None:
        poles = self.tracker.fixed(ObjectClass.POLE)
        wires = self.tracker.fixed(ObjectClass.WIRE)
        self.spans = self.graph.infer(poles, wires)
        self.obstacles = self.graph.to_obstacles(self.spans)

    @property
    def fixed_poles(self) -> list[Landmark]:
        return self.tracker.fixed(ObjectClass.POLE)

    def summary(self) -> dict:
        return {
            "frames": self._frames,
            "detections": self._detections,
            "landmarks": len(self.tracker.landmarks),
            "fixed_poles": len(self.fixed_poles),
            "spans": len(self.spans),
            "corroborated": sum(1 for s in self.spans if s.corroborated),
        }

    def to_dict(self) -> dict:
        """Inferred structure, for the viewer."""
        return {
            "poles": [
                {
                    "n": round(p.position.north, 2),
                    "e": round(p.position.east, 2),
                    "d": round(p.position.down, 2),
                    "conf": p.confidence,
                    "obs": p.observation_count,
                }
                for p in self.fixed_poles
            ],
            # Export the CABLE geometry, not the pole endpoints. The viewer
            # must draw the volume the planner actually avoided; re-deriving
            # sag in JavaScript would be a second implementation of the same
            # maths, free to drift from this one and quietly misreport
            # clearance.
            "spans": [
                {
                    "conf": span.confidence,
                    "corr": span.corroborated,
                    "points": [
                        {"n": round(q.north, 2), "e": round(q.east, 2),
                         "d": round(q.down, 2)}
                        for q in cable.points()
                    ],
                }
                for span, cable in zip(self.spans, self.obstacles.cables)
            ],
        }


def map_quality(
    spans: list[Span],
    truth_cables: list[Cable],
    *,
    tolerance_m: float = 8.0,
) -> dict:
    """Score an inferred map against ground truth. For evaluation only.

    Reports recall (what fraction of real conductors we inferred) and
    precision (what fraction of inferred conductors are real). Both matter and
    they trade against each other: a permissive threshold finds every wire and
    also invents several, which turns into a vehicle that refuses to fly.
    """
    if not truth_cables:
        return {"recall": 1.0, "precision": 1.0 if not spans else 0.0, "matched": 0}

    def midpoint(a: Vec3, b: Vec3) -> Vec3:
        return (a + b) * 0.5

    truth_mids = [midpoint(c.start, c.end) for c in truth_cables]
    span_mids = [midpoint(s.a, s.b) for s in spans]

    matched_truth = sum(
        1 for tm in truth_mids
        if any((tm - sm).norm <= tolerance_m for sm in span_mids)
    )
    matched_spans = sum(
        1 for sm in span_mids
        if any((tm - sm).norm <= tolerance_m for tm in truth_mids)
    )
    return {
        "recall": round(matched_truth / len(truth_mids), 3),
        "precision": round(matched_spans / len(spans), 3) if spans else 0.0,
        "matched": matched_truth,
        "inferred": len(spans),
        "actual": len(truth_cables),
    }
