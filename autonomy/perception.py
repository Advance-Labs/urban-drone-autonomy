"""Camera perception: a stand-in for a learned detector, with honest errors.

## What is simulated and what is real

This module SIMULATES a detector. It does not run a neural network -- there is
no camera and no trained weights here. What it reproduces is a detector's
*output statistics*: which objects get found at which apparent size, how
confidence varies, how often it hallucinates, and which classes it confuses.

The downstream module (`mapping.py`) is the opposite: that code is real. Given
detections it triangulates landmarks, associates them across frames, and infers
structure. It would run unchanged against a real detector's output, because it
only consumes the `Detection2D` records defined here.

That split is deliberate. Simulating the perception layer lets the *mapping and
avoidance* logic -- the part that is genuinely ours to get right -- be developed
and tested against realistic noise long before a model exists.

## Why detection difficulty varies so sharply by class

Apparent size in pixels is what governs a detector, and it varies enormously:

    a 9 m utility pole at 40 m   -> ~250 px tall on a 1080p sensor: trivial
    a 0.7 m sign at 40 m         -> ~20 px: reliable but not certain
    a 12 mm conductor at 40 m    -> 0.3 px wide: below one pixel

The wire is not merely hard, it is physically unresolvable -- no architecture
recovers detail finer than the sensor samples. This is why wire detection is
approached as *segmentation of a long thin structure* (many sub-pixel-wide
pixels forming a continuous line, integrated along its length) rather than as
bounding-box detection, and why even then it fails against a bright sky.

Which is the whole argument for `mapping.py`: detect the poles, infer the span.

## Path to a real model

Nothing here needs to stay simulated. For poles, lamps and signs, a fine-tuned
YOLO-class detector or RT-DETR is the standard choice, trained on street-level
imagery -- Mapillary Vistas and Cityscapes both carry pole, street-light and
traffic-sign classes. For conductors, thin-structure segmentation is its own
literature, with TTPLA (aerial transmission towers and power lines) the usual
public starting point. VERIFY dataset licences and class coverage before
relying on any of these; they are pointers, not a validated shortlist.

On a budget airframe, a Jetson Orin Nano runs a small YOLO in real time; a
Raspberry Pi 5 needs an accelerator to keep up.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from enum import Enum

from .types import Vec3


class ObjectClass(str, Enum):
    """Classes a street-furniture detector would be trained on."""

    POLE = "pole"
    LAMP = "lamp"
    SIGN = "sign"
    WIRE = "wire"
    TREE = "tree"


# Physical size used to compute apparent pixel size, and the detector's
# per-class quality once the object IS resolvable. The recall ceilings are
# rough stand-ins for published per-class AP on street-furniture datasets:
# large rigid verticals do well, thin and cluttered classes do not.
@dataclass(frozen=True, slots=True)
class ClassProfile:
    width_m: float
    height_m: float
    ceiling_recall: float  # best achievable, given plenty of pixels
    min_pixels: float      # below this the class is effectively invisible
    false_positive_rate: float  # spurious detections per frame


PROFILES: dict[ObjectClass, ClassProfile] = {
    # A utility pole is the easiest street object there is: tall, rigid,
    # vertical, and usually silhouetted.
    ObjectClass.POLE: ClassProfile(0.35, 9.0, 0.94, 18.0, 0.010),
    ObjectClass.LAMP: ClassProfile(0.6, 1.0, 0.80, 12.0, 0.020),
    ObjectClass.SIGN: ClassProfile(0.75, 0.75, 0.85, 12.0, 0.030),
    ObjectClass.TREE: ClassProfile(6.0, 8.0, 0.90, 30.0, 0.015),
    # Conductors are a different problem entirely -- see the module docstring.
    ObjectClass.WIRE: ClassProfile(0.012, 0.012, 0.55, 1.2, 0.045),
}


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    """A pinhole camera. Defaults approximate the STARVIS module in the plan."""

    width_px: int = 1920
    height_px: int = 1080
    hfov_deg: float = 62.0

    def __post_init__(self) -> None:
        if self.width_px <= 0 or self.height_px <= 0:
            raise ValueError("image dimensions must be positive")
        if not 0 < self.hfov_deg < 180:
            raise ValueError("hfov must be within 0..180 degrees")

    @property
    def focal_px(self) -> float:
        """Focal length in pixels, from horizontal FOV."""
        return (self.width_px / 2) / math.tan(math.radians(self.hfov_deg) / 2)

    @property
    def vfov_deg(self) -> float:
        return math.degrees(2 * math.atan((self.height_px / 2) / self.focal_px))

    def apparent_px(self, size_m: float, range_m: float) -> float:
        """Pixels an object of `size_m` spans at `range_m`.

        This single number is what decides whether a detector has any chance.
        """
        if range_m <= 0.01:
            return float(self.width_px)
        return size_m * self.focal_px / range_m


@dataclass(frozen=True, slots=True)
class Detection2D:
    """One detector output, in the form a real model would emit.

    Deliberately camera-relative: a bearing and elevation from the optical
    axis, plus apparent size. A monocular detector cannot know range, and
    pretending otherwise would let the mapping layer cheat.
    """

    cls: ObjectClass
    bearing_deg: float      # world-frame bearing to the detection
    elevation_deg: float    # world-frame elevation
    pixel_height: float
    confidence: float
    frame_t: float
    # Ground truth identity, for scoring only. The mapping layer MUST NOT read
    # this -- it exists so tests can measure association accuracy.
    truth_id: str | None = None

    @property
    def is_spurious(self) -> bool:
        return self.truth_id is None


@dataclass
class DetectorModel:
    """Simulates a fine-tuned street-furniture detector.

    Recall rises with apparent size and saturates at the class ceiling; below
    `min_pixels` it collapses. Confidence is sampled around the recall value,
    so weak detections carry low scores the way a real model's would.
    """

    intrinsics: CameraIntrinsics = field(default_factory=CameraIntrinsics)
    hfov_margin_deg: float = 0.0
    confidence_floor: float = 0.35
    seed: int = 23
    false_positives: bool = True
    # Yaw offsets of each camera relative to the airframe nose, degrees.
    # A single forward camera is (0.0,). Four at 90 deg spacing give full
    # coverage with a 62 deg lens only if you accept gaps; six is the usual
    # choice for a genuinely seamless sphere.
    mounts_deg: tuple[float, ...] = (0.0,)
    # Bounding-box centre jitter, in pixels. A real detector never localises
    # exactly: the box breathes a few pixels frame to frame. Converted to
    # angular noise this is what makes triangulation error behave properly --
    # with exact bearings any two viewpoints give a perfect fix, and the
    # baseline that dominates real accuracy stops mattering at all.
    localisation_noise_px: float = 2.5

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def recall(self, cls: ObjectClass, size_m: float, range_m: float) -> float:
        """Probability this object is detected in one frame."""
        profile = PROFILES[cls]
        px = self.intrinsics.apparent_px(size_m, range_m)
        if px < profile.min_pixels:
            # Not "unlikely" -- unresolvable. A detector cannot recover detail
            # finer than the sensor samples.
            return 0.0
        # Saturating curve: doubling pixels helps a lot when small, little
        # when already large.
        scale = px / (px + profile.min_pixels * 3.0)
        return profile.ceiling_recall * scale

    @property
    def coverage_deg(self) -> float:
        """Total unique horizontal coverage of the rig, degrees."""
        covered = set()
        for mount in self.mounts_deg:
            for d in range(360):
                off = abs((d - mount + 180.0) % 360.0 - 180.0)
                if off <= self.intrinsics.hfov_deg / 2:
                    covered.add(d)
        return float(len(covered))

    def observe(
        self,
        eye: Vec3,
        heading_deg: float,
        targets: list[tuple[ObjectClass, Vec3, str]],
        frame_t: float,
    ) -> list[Detection2D]:
        """Run one frame over a list of (class, position, id) ground truths."""
        out: list[Detection2D] = []
        for cls, position, truth_id in targets:
            offset = position - eye
            distance = offset.norm
            if distance < 0.5 or not self._in_frame(eye, heading_deg, position):
                continue

            profile = PROFILES[cls]
            p = self.recall(cls, profile.height_m, distance)
            if p <= 0.0 or self._rng.random() > p:
                continue

            confidence = min(0.99, max(self.confidence_floor, self._rng.gauss(p, 0.08)))
            # Weak detections localise worse, so scale jitter by how marginal
            # this one is -- a 20-pixel pole wobbles far more than a 300-pixel one.
            jitter = self.localisation_noise_px * (1.0 + (1.0 - p))
            noise_deg = math.degrees(jitter / self.intrinsics.focal_px)
            out.append(
                Detection2D(
                    cls=cls,
                    bearing_deg=(
                        self._bearing(eye, position) + self._rng.gauss(0, noise_deg)
                    ) % 360.0,
                    elevation_deg=(
                        self._elevation(eye, position) + self._rng.gauss(0, noise_deg)
                    ),
                    pixel_height=self.intrinsics.apparent_px(profile.height_m, distance),
                    confidence=round(confidence, 3),
                    frame_t=frame_t,
                    truth_id=truth_id,
                )
            )

        if self.false_positives:
            out.extend(self._hallucinate(heading_deg, frame_t))
        return out

    def _hallucinate(self, heading_deg: float, frame_t: float) -> list[Detection2D]:
        """Spurious detections.

        Not decoration: a mapping layer that assumes every detection is real
        will happily triangulate a landmark out of two unrelated false
        positives and plant a phantom obstacle in the flight path.
        """
        out = []
        half = self.intrinsics.hfov_deg / 2
        for cls, profile in PROFILES.items():
            if self._rng.random() > profile.false_positive_rate:
                continue
            out.append(
                Detection2D(
                    cls=cls,
                    bearing_deg=(heading_deg + self._rng.uniform(-half, half)) % 360.0,
                    elevation_deg=self._rng.uniform(-15, 15),
                    pixel_height=self._rng.uniform(profile.min_pixels, profile.min_pixels * 4),
                    # False positives cluster at low confidence, which is what
                    # makes a confidence threshold worth having.
                    confidence=round(self._rng.uniform(0.35, 0.62), 3),
                    frame_t=frame_t,
                    truth_id=None,
                )
            )
        return out

    # -- geometry ----------------------------------------------------------

    @staticmethod
    def _bearing(eye: Vec3, target: Vec3) -> float:
        o = target - eye
        return math.degrees(math.atan2(o.east, o.north)) % 360.0

    @staticmethod
    def _elevation(eye: Vec3, target: Vec3) -> float:
        o = target - eye
        return math.degrees(math.atan2(-o.down, math.hypot(o.north, o.east)))

    def _in_frame(self, eye: Vec3, heading_deg: float, target: Vec3) -> bool:
        """Visible to ANY camera on the rig.

        With several mounts this is a union of frusta, which is the whole
        point of a multi-camera rig: an object that leaves the forward view
        as you fly past stays visible to a side-looking lens, and it is
        exactly that continued observation across a long baseline that makes
        triangulation converge.
        """
        if abs(self._elevation(eye, target)) > self.intrinsics.vfov_deg / 2:
            return False
        bearing = self._bearing(eye, target)
        half = self.intrinsics.hfov_deg / 2 + self.hfov_margin_deg
        for mount in self.mounts_deg:
            axis = heading_deg + mount
            off_axis = abs((bearing - axis + 180.0) % 360.0 - 180.0)
            if off_axis <= half:
                return True
        return False


def max_detection_range(
    cls: ObjectClass,
    intrinsics: CameraIntrinsics | None = None,
    *,
    min_recall: float = 0.5,
) -> float:
    """Furthest range at which `cls` is detected with at least `min_recall`.

    The headline comparison this module exists to make: poles are detectable
    from far enough away to plan around, conductors are not detectable at all.
    """
    intr = intrinsics or CameraIntrinsics()
    model = DetectorModel(intrinsics=intr)
    profile = PROFILES[cls]
    r = 1.0
    best = 0.0
    while r < 500.0:
        if model.recall(cls, profile.height_m, r) >= min_recall:
            best = r
        r += 0.5
    return best
