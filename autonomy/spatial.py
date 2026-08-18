"""A uniform spatial hash, for collision queries against a realistic world.

Both the building and conductor collision tests were linear scans. That is
fine for the 25 boxes of a procedural test city and catastrophic for a real
one: a 350 m import of North York carries 273 polygon footprints and 448
catenary segments, and an exhaustive wire query measured at 8.5 ms. A single
A* search would have spent over two hours in collision tests.

The structure is deliberately the simplest thing that fixes it. Objects are
inserted into every grid cell their bounding box overlaps; a query looks only
in the cells its own query box overlaps. No tree, no rebalancing, no
tuning beyond the cell size.

Cell size is the one real parameter. Too small and large objects are inserted
into hundreds of cells; too large and each bucket degenerates toward the
linear scan this exists to avoid. Roughly the size of a typical object works
out well -- around 30 m for buildings, which is a house plus its garden.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Generic, TypeVar

T = TypeVar("T")


class SpatialHash(Generic[T]):
    """Bucket objects by 2D grid cell. Altitude is not indexed.

    Deliberately 2D: buildings and conductors are separated horizontally far
    more than vertically, so a third axis would multiply the bookkeeping while
    barely shrinking the candidate sets.
    """

    def __init__(self, cell_m: float = 30.0) -> None:
        if cell_m <= 0:
            raise ValueError("cell size must be positive")
        self.cell = cell_m
        self._buckets: dict[tuple[int, int], list[T]] = {}
        self._count = 0

    def __len__(self) -> int:
        return self._count

    def insert(self, item: T, min_n: float, max_n: float, min_e: float, max_e: float) -> None:
        """Add `item` to every cell its bounding box touches."""
        for key in self._cells(min_n, max_n, min_e, max_e):
            self._buckets.setdefault(key, []).append(item)
        self._count += 1

    def query(self, n: float, e: float, radius: float = 0.0) -> Iterator[T]:
        """Every item whose bucket overlaps the query box.

        May yield the same item twice when it spans several cells, which is
        harmless for the "is anything hit?" tests this serves -- deduplicating
        would cost a set allocation on every call, in the hot path.
        """
        for key in self._cells(n - radius, n + radius, e - radius, e + radius):
            bucket = self._buckets.get(key)
            if bucket:
                yield from bucket

    def query_segment(
        self, n0: float, e0: float, n1: float, e1: float, radius: float = 0.0
    ) -> set[int]:
        """Indices of candidates near a segment, deduplicated.

        Returns identities rather than objects because a segment usually spans
        many cells and the duplicates would matter here.
        """
        found: set[int] = set()
        for key in self._cells(
            min(n0, n1) - radius, max(n0, n1) + radius,
            min(e0, e1) - radius, max(e0, e1) + radius,
        ):
            for item in self._buckets.get(key, ()):
                found.add(id(item))
        return found

    def candidates_near_segment(
        self, n0: float, e0: float, n1: float, e1: float, radius: float = 0.0
    ) -> list[T]:
        """Deduplicated candidates whose cells overlap the segment's box."""
        seen: set[int] = set()
        out: list[T] = []
        for key in self._cells(
            min(n0, n1) - radius, max(n0, n1) + radius,
            min(e0, e1) - radius, max(e0, e1) + radius,
        ):
            for item in self._buckets.get(key, ()):
                if id(item) not in seen:
                    seen.add(id(item))
                    out.append(item)
        return out

    def _cells(
        self, min_n: float, max_n: float, min_e: float, max_e: float
    ) -> Iterator[tuple[int, int]]:
        n0 = int(min_n // self.cell)
        n1 = int(max_n // self.cell)
        e0 = int(min_e // self.cell)
        e1 = int(max_e // self.cell)
        for cn in range(n0, n1 + 1):
            for ce in range(e0, e1 + 1):
                yield (cn, ce)

    def stats(self) -> dict:
        sizes = [len(b) for b in self._buckets.values()]
        return {
            "items": self._count,
            "buckets": len(self._buckets),
            "max_bucket": max(sizes, default=0),
            "mean_bucket": round(sum(sizes) / len(sizes), 1) if sizes else 0.0,
        }
