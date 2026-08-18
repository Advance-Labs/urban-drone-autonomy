"""Vehicle backends. Import `Px4Backend` lazily -- it needs MAVSDK installed."""

from .base import VehicleBackend
from .kinematic import KinematicBackend

__all__ = ["KinematicBackend", "VehicleBackend", "Px4Backend"]


def __getattr__(name: str):
    """Defer the PX4 import so `pip install mavsdk` stays optional."""
    if name == "Px4Backend":
        from .px4 import Px4Backend

        return Px4Backend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
