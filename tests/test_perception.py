"""Camera detection model and the structure-inference pipeline it feeds."""

from __future__ import annotations

import pytest

from autonomy.mapping import (
    LandmarkTracker,
    Landmark,
    Observation,
    StructureGraph,
    map_quality,
)
from autonomy.perception import (
    CameraIntrinsics,
    DetectorModel,
    ObjectClass,
    PROFILES,
    max_detection_range,
)
from autonomy.thin_obstacles import Cable
from autonomy.types import Vec3


class TestIntrinsics:
    def test_rejects_impossible_optics(self):
        with pytest.raises(ValueError):
            CameraIntrinsics(width_px=0)
        with pytest.raises(ValueError):
            CameraIntrinsics(hfov_deg=200)

    def test_apparent_size_falls_with_range(self):
        c = CameraIntrinsics()
        assert c.apparent_px(9.0, 20) > c.apparent_px(9.0, 80)

    def test_apparent_size_is_inverse_in_range(self):
        c = CameraIntrinsics()
        assert c.apparent_px(9.0, 40) == pytest.approx(2 * c.apparent_px(9.0, 80))

    def test_a_conductor_is_sub_pixel_at_range(self):
        """The core fact this whole design works around."""
        c = CameraIntrinsics()
        assert c.apparent_px(0.012, 40) < 1.0


class TestDetectorRecall:
    def test_poles_are_detectable_far_beyond_conductors(self):
        pole = max_detection_range(ObjectClass.POLE)
        wire = max_detection_range(ObjectClass.WIRE)
        assert pole > 60, f"poles should be visible well out, got {pole:.0f} m"
        assert wire < pole / 5, "conductors must be far harder than poles"

    def test_recall_is_zero_below_the_resolution_floor(self):
        """Not 'unlikely' -- unresolvable. No model recovers sub-sample detail."""
        m = DetectorModel()
        assert m.recall(ObjectClass.WIRE, 0.012, 300.0) == 0.0

    def test_recall_rises_with_apparent_size(self):
        m = DetectorModel()
        near = m.recall(ObjectClass.POLE, 9.0, 15.0)
        far = m.recall(ObjectClass.POLE, 9.0, 90.0)
        assert near > far > 0.0

    def test_recall_never_exceeds_the_class_ceiling(self):
        m = DetectorModel()
        assert m.recall(ObjectClass.POLE, 9.0, 2.0) <= PROFILES[ObjectClass.POLE].ceiling_recall


class TestObservation:
    def test_objects_behind_the_camera_are_never_detected(self):
        m = DetectorModel(seed=1)
        behind = [(ObjectClass.POLE, Vec3(-40, 0, -4.5), "p")]
        for _ in range(60):
            real = [d for d in m.observe(Vec3(), 0.0, behind, 0.0) if not d.is_spurious]
            assert not real

    def test_objects_ahead_are_detected_most_frames(self):
        m = DetectorModel(seed=2)
        ahead = [(ObjectClass.POLE, Vec3(30, 0, -4.5), "p")]
        hits = sum(
            1 for _ in range(60)
            for d in m.observe(Vec3(), 0.0, ahead, 0.0) if d.truth_id == "p"
        )
        assert hits > 30, f"expected a strong majority, got {hits}/60"

    def test_false_positives_occur_and_carry_low_confidence(self):
        m = DetectorModel(seed=3)
        spurious = [
            d for _ in range(400)
            for d in m.observe(Vec3(), 0.0, [], 0.0) if d.is_spurious
        ]
        assert spurious, "a detector that never hallucinates is not realistic"
        assert max(d.confidence for d in spurious) < 0.7, (
            "false positives should cluster low, which is what makes a "
            "confidence threshold useful"
        )

    def test_disabling_false_positives_silences_them(self):
        m = DetectorModel(seed=4, false_positives=False)
        assert all(not m.observe(Vec3(), 0.0, [], 0.0) for _ in range(50))

    def test_detections_are_reproducible_for_a_seed(self):
        targets = [(ObjectClass.POLE, Vec3(35, 5, -4.5), "p")]
        a = [len(DetectorModel(seed=9).observe(Vec3(), 0.0, targets, 0.0)) for _ in range(1)]
        b = [len(DetectorModel(seed=9).observe(Vec3(), 0.0, targets, 0.0)) for _ in range(1)]
        assert a == b


class TestTriangulation:
    def _observe_along(self, tracker: LandmarkTracker, truth: Vec3, path: list[Vec3]):
        m = DetectorModel(seed=11, false_positives=False)
        for i, eye in enumerate(path):
            heading = 0.0
            dets = m.observe(eye, heading, [(ObjectClass.POLE, truth, "p")], float(i))
            tracker.ingest(eye, dets)

    def test_lateral_motion_fixes_a_landmark(self):
        truth = Vec3(60, 0, -4.5)
        tracker = LandmarkTracker()
        # Fly across the front: this is what supplies parallax.
        self._observe_along(tracker, truth, [Vec3(0, e, -6) for e in range(-20, 21, 4)])
        fixed = tracker.fixed(ObjectClass.POLE)
        assert fixed, "should have triangulated the pole"
        assert (fixed[0].position - truth).norm < 8.0

    def test_flying_straight_at_it_yields_no_usable_fix(self):
        """Approaching head-on gives almost no parallax. Being honest about
        this matters -- it is the geometry a naive test would accidentally
        avoid, and the one a real flight hits constantly."""
        truth = Vec3(120, 0, -4.5)
        tracker = LandmarkTracker(min_baseline_m=15.0)
        self._observe_along(tracker, truth, [Vec3(n, 0, -6) for n in range(0, 30, 3)])
        # Either no fix, or a fix the tracker itself rates as weak.
        for lm in tracker.fixed(ObjectClass.POLE):
            assert lm.confidence < 0.8

    def test_two_observations_from_one_spot_do_not_fix(self):
        tracker = LandmarkTracker()
        obs = [Observation(Vec3(), 0.0, 0.0, 0.9, float(i)) for i in range(5)]
        lm = Landmark(cls=ObjectClass.POLE, observations=obs)
        tracker.landmarks.append(lm)
        tracker._solve(lm)
        assert not lm.is_fixed, "zero baseline cannot triangulate"

    def test_confidence_grows_with_baseline_and_support(self):
        truth = Vec3(50, 0, -4.5)
        short = LandmarkTracker(min_baseline_m=2.0)
        self._observe_along(short, truth, [Vec3(0, e, -6) for e in range(-3, 4, 1)])
        wide = LandmarkTracker(min_baseline_m=2.0)
        self._observe_along(wide, truth, [Vec3(0, e, -6) for e in range(-25, 26, 3)])
        if short.fixed() and wide.fixed():
            assert wide.fixed()[0].confidence >= short.fixed()[0].confidence


def pole(n: float, e: float, height: float = 9.0, conf: float = 0.85) -> Landmark:
    return Landmark(
        cls=ObjectClass.POLE,
        position=Vec3(n, e, -height),
        confidence=conf,
        baseline_m=20.0,
        observations=[],
    )


class TestStructureInference:
    def test_adjacent_poles_are_connected(self):
        spans = StructureGraph().infer([pole(0, 0), pole(0, 40)])
        assert len(spans) == 1
        assert spans[0].length_m == pytest.approx(40.0)

    def test_distant_poles_are_not_connected(self):
        assert StructureGraph().infer([pole(0, 0), pole(0, 300)]) == []

    def test_very_close_poles_are_not_connected(self):
        """Two poles 5 m apart are a sign gantry or a duplicate detection,
        not a conductor span."""
        assert StructureGraph().infer([pole(0, 0), pole(0, 5)]) == []

    def test_a_run_connects_neighbours_not_endpoints(self):
        """Without this the four-pole run also gains a phantom conductor
        spanning the entire line."""
        poles = [pole(0, e) for e in (0, 40, 80, 120)]
        spans = StructureGraph().infer(poles)
        assert len(spans) == 3, f"expected 3 adjacent spans, got {len(spans)}"
        assert all(s.length_m == pytest.approx(40.0) for s in spans)

    def test_low_confidence_poles_yield_no_spans(self):
        """An inferred span can never be surer than the poles it rests on."""
        assert StructureGraph().infer([pole(0, 0, conf=0.2), pole(0, 40, conf=0.2)]) == []

    def test_wire_corroboration_raises_confidence(self):
        poles = [pole(0, 0), pole(0, 40)]
        plain = StructureGraph().infer(poles)
        wire = Landmark(
            cls=ObjectClass.WIRE, position=Vec3(0, 20, -8), confidence=0.6,
        )
        backed = StructureGraph().infer(poles, [wire])
        assert backed[0].confidence > plain[0].confidence
        assert backed[0].corroborated and not plain[0].corroborated

    def test_mismatched_heights_lower_confidence(self):
        same = StructureGraph().infer([pole(0, 0, 9.0), pole(0, 40, 9.0)])
        differing = StructureGraph().infer([pole(0, 0, 9.0), pole(0, 40, 3.0)])
        if differing:
            assert differing[0].confidence < same[0].confidence


class TestObstacleExport:
    def test_spans_become_sagging_cables_below_the_pole_tops(self):
        graph = StructureGraph()
        spans = graph.infer([pole(0, 0), pole(0, 40)])
        field = graph.to_obstacles(spans)
        assert len(field.cables) == 1
        cable = field.cables[0]
        # Attachment below the top, and midspan below that again.
        assert cable.start.altitude < 9.0
        lowest = max(p.down for p in cable.points())
        assert lowest > cable.start.down, "midspan must hang below the attachment"

    def test_inferred_obstacles_block_a_path_through_them(self):
        graph = StructureGraph()
        field = graph.to_obstacles(graph.infer([pole(0, 0), pole(0, 40)]))
        cable = field.cables[0]
        through = Vec3(0, 20, cable.points()[4].down)
        assert not field.segment_clear(
            Vec3(-10, 20, through.down), Vec3(10, 20, through.down), margin=0.5
        )


class TestFlightGeometry:
    """The dominant factor in map quality is how you fly, not the detector.

    Triangulation error scales roughly as range^2 * angular_noise / baseline,
    so a lateral sweep across a line of poles produces a usable map while a
    head-on approach produces a confident-looking wrong one.
    """

    def _survey(self, path: list[Vec3], seed: int = 31):
        from autonomy.mapping import PerceptionPipeline
        from autonomy.perception import DetectorModel

        poles = [Vec3(0.0, -80.0 + i * 40.0, -9.0) for i in range(5)]
        targets = [(ObjectClass.POLE, p, f"P{i}") for i, p in enumerate(poles)]
        pipe = PerceptionPipeline(detector=DetectorModel(seed=seed), truth=targets)

        for i, eye in enumerate(path):
            pipe.step(eye, 0.0, float(i) * 0.1)
        pipe.rebuild()

        errors = []
        for lm in pipe.fixed_poles:
            nearest = min(poles, key=lambda t: (t - lm.position).norm)
            gap = (nearest - lm.position).norm
            if gap < 30.0:  # ignore spurious landmarks
                errors.append(gap)
        return pipe, (sum(errors) / len(errors) if errors else float("inf"))

    def _interpolate(self, a: Vec3, b: Vec3, n: int) -> list[Vec3]:
        return [a + (b - a) * (i / n) for i in range(n + 1)]

    def test_lateral_sweep_resolves_the_sag(self):
        path = self._interpolate(Vec3(-70, -90, -22), Vec3(-70, 90, -22), 180)
        pipe, err = self._survey(path)
        assert err < self.TYPICAL_SAG_M, (
            f"lateral sweep should beat the {self.TYPICAL_SAG_M} m sag, got {err:.2f} m"
        )
        assert len(pipe.spans) >= 3

    # Midspan sag on a 40 m distribution span, ~3% of span. The map has to
    # resolve better than this to be worth having: an error larger than the
    # sag cannot tell you whether you are above the wire or through it.
    TYPICAL_SAG_M = 1.2

    def test_head_on_approach_cannot_resolve_the_sag(self):
        """All poles may still be 'fixed' -- the danger is a map that looks
        confident while carrying error larger than the clearance it exists to
        protect. Anchored to the sag rather than a round number, so the test
        states an operational bar instead of a magic threshold."""
        path = self._interpolate(Vec3(-140, 0, -22), Vec3(-35, 0, -22), 180)
        _, err = self._survey(path)
        assert err > self.TYPICAL_SAG_M, (
            f"head-on error {err:.2f} m should exceed the {self.TYPICAL_SAG_M} m "
            "sag it must resolve; if not, the noise model has stopped being realistic"
        )

    def test_lateral_beats_head_on(self):
        _, lateral = self._survey(
            self._interpolate(Vec3(-70, -90, -22), Vec3(-70, 90, -22), 180)
        )
        _, head_on = self._survey(
            self._interpolate(Vec3(-140, 0, -22), Vec3(-35, 0, -22), 180)
        )
        assert lateral < head_on


class TestCameraRig:
    """What extra cameras actually buy: geometry, not detection quality."""

    def _survey(self, path: list[Vec3], mounts: tuple[float, ...]):
        from autonomy.mapping import PerceptionPipeline
        from autonomy.perception import DetectorModel

        poles = [Vec3(0.0, -80.0 + i * 40.0, -9.0) for i in range(5)]
        targets = [(ObjectClass.POLE, p, f"P{i}") for i, p in enumerate(poles)]
        pipe = PerceptionPipeline(
            detector=DetectorModel(seed=31, mounts_deg=mounts), truth=targets
        )
        for i, eye in enumerate(path):
            pipe.step(eye, 0.0, i * 0.1)
        pipe.rebuild()

        errors = []
        for lm in pipe.fixed_poles:
            nearest = min(poles, key=lambda t: (t - lm.position).norm)
            gap = (nearest - lm.position).norm
            if gap < 30.0:
                errors.append(gap)
        return pipe, (sum(errors) / len(errors) if errors else float("inf"))

    @staticmethod
    def _lerp(a: Vec3, b: Vec3, n: int) -> list[Vec3]:
        return [a + (b - a) * (i / n) for i in range(n + 1)]

    def test_coverage_adds_up(self):
        from autonomy.perception import DetectorModel

        assert DetectorModel(mounts_deg=(0.0,)).coverage_deg < 90
        assert DetectorModel(
            mounts_deg=tuple(i * 60.0 for i in range(6))
        ).coverage_deg == 360.0

    def test_full_coverage_rescues_a_head_on_approach(self):
        """The payoff: a rig removes the need to fly a survey pattern first,
        because poles stay in view as they pass abeam -- which is exactly when
        parallax is richest."""
        path = self._lerp(Vec3(-140, 0, -22), Vec3(-35, 0, -22), 180)
        _, forward = self._survey(path, (0.0,))
        _, full = self._survey(path, tuple(i * 60.0 for i in range(6)))
        assert full < forward / 2, (
            f"360 coverage should transform head-on geometry: "
            f"{forward:.2f} m -> {full:.2f} m"
        )

    def test_rear_camera_alone_does_not_help_head_on(self):
        """Coverage has to be where the parallax is. A rear camera sees where
        you have been, which on a straight-in approach is empty sky."""
        path = self._lerp(Vec3(-140, 0, -22), Vec3(-35, 0, -22), 180)
        _, forward = self._survey(path, (0.0,))
        _, fwd_rear = self._survey(path, (0.0, 180.0))
        assert fwd_rear == pytest.approx(forward, rel=0.25)

    def test_a_good_flight_path_needs_no_extra_cameras(self):
        """Worth knowing before spending: fly the right pattern and one
        camera is enough."""
        path = self._lerp(Vec3(-70, -90, -22), Vec3(-70, 90, -22), 180)
        _, forward = self._survey(path, (0.0,))
        assert forward < 1.0


class TestMapQuality:
    def test_perfect_inference_scores_one(self):
        truth = [Cable(start=Vec3(0, 0, -8), end=Vec3(0, 40, -8), name="c")]
        graph = StructureGraph()
        spans = graph.infer([pole(0, 0), pole(0, 40)])
        q = map_quality(spans, truth)
        assert q["recall"] == 1.0 and q["precision"] == 1.0

    def test_missed_conductors_lower_recall(self):
        truth = [
            Cable(start=Vec3(0, 0, -8), end=Vec3(0, 40, -8), name="a"),
            Cable(start=Vec3(200, 0, -8), end=Vec3(200, 40, -8), name="b"),
        ]
        spans = StructureGraph().infer([pole(0, 0), pole(0, 40)])
        assert map_quality(spans, truth)["recall"] == 0.5

    def test_phantom_spans_lower_precision(self):
        truth = [Cable(start=Vec3(0, 0, -8), end=Vec3(0, 40, -8), name="a")]
        spans = StructureGraph().infer([pole(0, 0), pole(0, 40), pole(300, 0), pole(300, 40)])
        q = map_quality(spans, truth)
        assert q["precision"] < 1.0

    def test_no_truth_and_no_spans_is_perfect(self):
        assert map_quality([], [])["recall"] == 1.0
