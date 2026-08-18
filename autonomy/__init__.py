"""Autonomy stack for the drone-lab side project.

Layer map (bottom to top):
    backends/   what is flying -- kinematic sim, PX4 SITL, or real hardware
    types       shared value types, NED frame
    planner     pure geometry: missions -> waypoint lists
    behaviors   the mission state machine
    safety      envelope monitoring, holds veto power over behaviours
    mission     the control loop that ties the above together
"""

from .behaviors import (
    Behavior,
    DivertAndLand,
    FollowPath,
    Hold,
    Land,
    ReturnToLaunch,
    Takeoff,
)
from .dynamic import DynamicField, MovingObstacle, crossing_traffic
from .hazards import (
    HazardField, Pedestrian, RegulatoryProfile, Vehicle,
    car_on_road, pedestrian_on_sidewalk,
)
from .landing import LandingSite, LandingSiteSelector, SiteAssessment
from .sensors import (
    Detection, LidarSpec, SensorSuite, StereoSpec,
    detection_probability, stopping_distance_m, wire_detection_range,
)
from .thin_obstacles import Cable, Foliage, ThinObstacleField, powerline_corridor
from .mission import LogEntry, MissionResult, MissionRunner, Sample
from .mapping import (
    Landmark, LandmarkTracker, PerceptionPipeline, Span, StructureGraph, map_quality,
)
from .navigation import GridPlanner, PlanResult
from .osm import free_launch
from .perception import (
    CameraIntrinsics, Detection2D, DetectorModel, ObjectClass, max_detection_range,
)
from .patrol import NavigateTo, ObserveTarget, PlannedReturn, build_patrol
from .surveillance import (
    Camera,
    CoverageTracker,
    SurveillanceTarget,
    observation_point,
)
from .world import Building, City, city_block_grid
from .planner import estimated_duration_s, orbit, path_length, survey_grid
from .safety import SafetyLimits, SafetyMonitor
from .types import (
    FlightMode,
    SafetyAction,
    SafetyVerdict,
    Telemetry,
    Vec3,
)

__all__ = [
    "Behavior",
    "Building",
    "CameraIntrinsics",
    "Detection2D",
    "DetectorModel",
    "Landmark",
    "LandmarkTracker",
    "ObjectClass",
    "PerceptionPipeline",
    "Span",
    "StructureGraph",
    "map_quality",
    "max_detection_range",
    "Cable",
    "Detection",
    "DynamicField",
    "Foliage",
    "HazardField",
    "Pedestrian",
    "RegulatoryProfile",
    "Vehicle",
    "car_on_road",
    "pedestrian_on_sidewalk",
    "LidarSpec",
    "MovingObstacle",
    "SensorSuite",
    "StereoSpec",
    "ThinObstacleField",
    "crossing_traffic",
    "detection_probability",
    "powerline_corridor",
    "stopping_distance_m",
    "wire_detection_range",
    "Camera",
    "City",
    "CoverageTracker",
    "DivertAndLand",
    "GridPlanner",
    "NavigateTo",
    "ObserveTarget",
    "PlannedReturn",
    "PlanResult",
    "SurveillanceTarget",
    "build_patrol",
    "city_block_grid",
    "observation_point",
    "FlightMode",
    "FollowPath",
    "Hold",
    "Land",
    "LandingSite",
    "LandingSiteSelector",
    "LogEntry",
    "MissionResult",
    "MissionRunner",
    "ReturnToLaunch",
    "SafetyAction",
    "SafetyLimits",
    "SafetyMonitor",
    "SafetyVerdict",
    "Sample",
    "SiteAssessment",
    "Takeoff",
    "Telemetry",
    "Vec3",
    "estimated_duration_s",
    "free_launch",
    "orbit",
    "path_length",
    "survey_grid",
]
