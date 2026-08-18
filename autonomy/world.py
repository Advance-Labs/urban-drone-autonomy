"""Urban world model: buildings, collision, and line of sight.

Buildings are axis-aligned boxes. That is a real simplification, and it is the
right one here: the autonomy problems this exists to exercise -- can I plan a
path through a street canyon, will I clip a corner, can the camera actually see
that doorway -- all depend on occupancy and occlusion, not on architectural
detail. A box grid reproduces both, exactly, at a thousandth of the cost of a
mesh.

What it deliberately does not model: overhangs, bridges, trees, wires, or
terrain relief. Wires in particular are the thing that actually kills small
drones in cities, and nothing here will teach you to avoid them.

Coordinates follow the NED convention used everywhere else: north +x, east +y,
down +z, so a building of height h occupies down from 0 to -h.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .types import Vec3


@dataclass(frozen=True, slots=True)
class Building:
    """An axis-aligned box obstacle standing on the ground plane."""

    north: float
    east: float
    size_n: float
    size_e: float
    height: float
    name: str = "building"

    def __post_init__(self) -> None:
        if self.size_n <= 0 or self.size_e <= 0:
            raise ValueError(f"{self.name}: footprint must be positive")
        if self.height <= 0:
            raise ValueError(f"{self.name}: height must be positive")

    @property
    def min_n(self) -> float:
        return self.north - self.size_n / 2

    @property
    def max_n(self) -> float:
        return self.north + self.size_n / 2

    @property
    def min_e(self) -> float:
        return self.east - self.size_e / 2

    @property
    def max_e(self) -> float:
        return self.east + self.size_e / 2

    def contains(self, p: Vec3, margin: float = 0.0) -> bool:
        """True if `p` is inside the building, grown by `margin` on all sides.

        The margin is how standoff distance is enforced: planning against an
        inflated obstacle is what keeps the vehicle off the glass, rather than
        trusting a controller to track a path that grazes it.
        """
        return (
            self.min_n - margin <= p.north <= self.max_n + margin
            and self.min_e - margin <= p.east <= self.max_e + margin
            and p.altitude <= self.height + margin
        )


@dataclass(slots=True)
class City:
    """A collection of buildings plus the airspace bounds around them."""

    buildings: list[Building] = field(default_factory=list)
    extent_m: float = 300.0
    name: str = "city"

    def is_free(self, p: Vec3, margin: float = 0.0) -> bool:
        """True if a vehicle of the given standoff could occupy `p`."""
        if p.altitude < 0.0:
            return False
        return not any(b.contains(p, margin) for b in self.buildings)

    def occupant(self, p: Vec3, margin: float = 0.0) -> Building | None:
        """Which building `p` is inside, if any. For crash reporting."""
        for b in self.buildings:
            if b.contains(p, margin):
                return b
        return None

    def segment_clear(
        self, a: Vec3, b: Vec3, margin: float = 0.0, step_m: float = 1.5
    ) -> bool:
        """Sampled collision check along the segment a->b.

        Sampling rather than analytic box intersection: with a margin applied
        the analytic version buys accuracy the planner cannot use, and the
        sampled form generalises unchanged if buildings ever stop being boxes.
        `step_m` must stay well below the smallest building dimension or thin
        structures can be stepped straight over.
        """
        span = (b - a).norm
        if span < 1e-9:
            return self.is_free(a, margin)
        steps = max(2, math.ceil(span / step_m))
        for i in range(steps + 1):
            p = a + (b - a) * (i / steps)
            if not self.is_free(p, margin):
                return False
        return True

    def has_line_of_sight(self, eye: Vec3, target: Vec3, step_m: float = 2.0) -> bool:
        """Can the camera at `eye` actually see `target`?

        No margin: a wall blocks sight only where it physically is, unlike
        planning where standoff matters. This is what makes urban surveillance
        interesting -- being in range is not the same as having a view.
        """
        span = (target - eye).norm
        if span < 1e-9:
            return True
        steps = max(2, math.ceil(span / step_m))
        # Skip the endpoints: a target sitting against a wall would otherwise
        # occlude itself.
        for i in range(1, steps):
            p = eye + (target - eye) * (i / steps)
            if any(b.contains(p) for b in self.buildings):
                return False
        return True

    def tallest(self) -> float:
        return max((b.height for b in self.buildings), default=0.0)

    def in_bounds(self, p: Vec3) -> bool:
        return abs(p.north) <= self.extent_m and abs(p.east) <= self.extent_m

    def to_dict(self) -> dict:
        """Scene data for the 3D viewer."""
        return {
            "name": self.name,
            "extent_m": self.extent_m,
            "buildings": [
                {
                    "n": b.north, "e": b.east,
                    "sn": b.size_n, "se": b.size_e,
                    "h": b.height, "name": b.name,
                }
                for b in self.buildings
            ],
        }


def city_block_grid(
    *,
    blocks: int = 4,
    block_m: float = 60.0,
    street_m: float = 26.0,
    min_height: float = 12.0,
    max_height: float = 55.0,
    seed: int = 7,
) -> City:
    """Generate a rectangular street grid with buildings of varied height.

    Deterministic for a given seed, so a mission that fails is reproducible --
    a randomly different city on every run makes debugging a planner
    impossible.

    The street width matters more than anything else here: at 26 m with a 4 m
    standoff there is roughly 18 m of usable canyon, which is enough for a
    planner to find a route but tight enough that it has to actually try.
    """
    if blocks < 1:
        raise ValueError("need at least one block")
    if street_m <= 0 or block_m <= 0:
        raise ValueError("block and street dimensions must be positive")
    if min_height <= 0 or max_height < min_height:
        raise ValueError("invalid height range")

    rng = random.Random(seed)
    pitch = block_m + street_m
    span = (blocks - 1) * pitch / 2
    buildings: list[Building] = []

    for i in range(blocks):
        for j in range(blocks):
            n = -span + i * pitch
            e = -span + j * pitch
            # Leave the launch block open so takeoff is not inside a wall.
            if abs(n) < pitch / 2 and abs(e) < pitch / 2:
                continue
            # Split some blocks into two smaller buildings for variety, which
            # also creates the narrow gaps a planner should refuse to use.
            if rng.random() < 0.35:
                half = block_m / 2 - 3
                for offset in (-half / 1.6, half / 1.6):
                    buildings.append(
                        Building(
                            north=n + offset, east=e,
                            size_n=half, size_e=block_m,
                            height=rng.uniform(min_height, max_height),
                            name=f"blk{i}{j}{'a' if offset < 0 else 'b'}",
                        )
                    )
            else:
                buildings.append(
                    Building(
                        north=n, east=e,
                        size_n=block_m, size_e=block_m,
                        height=rng.uniform(min_height, max_height),
                        name=f"blk{i}{j}",
                    )
                )

    return City(buildings=buildings, extent_m=span + pitch, name="downtown grid")
