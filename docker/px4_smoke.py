#!/usr/bin/env python3
"""Fly a real mission against PX4 SITL through the same autonomy stack.

This is the proof that the `VehicleBackend` seam actually holds: the behaviours,
safety monitor, and mission runner below are the identical objects the kinematic
sim uses. Only the backend differs.

Run via `docker compose -f docker/docker-compose.yml up --abort-on-container-exit`.
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, "/sim")

from autonomy import (  # noqa: E402
    Hold,
    MissionRunner,
    ReturnToLaunch,
    SafetyLimits,
    SafetyMonitor,
    Takeoff,
    Vec3,
)
from autonomy.backends import Px4Backend  # noqa: E402


async def main() -> int:
    address = os.environ.get("PX4_ADDRESS", "udpin://0.0.0.0:14540")
    print(f"connecting to PX4 at {address}", flush=True)

    backend = Px4Backend(address, connect_timeout_s=90.0)

    # Connect explicitly first so each stage is visible; a bare hang inside
    # the mission runner tells you nothing about which step stalled.
    print("waiting for heartbeat...", flush=True)
    await backend.connect()
    t = await backend.telemetry()
    print(f"connected. armed={t.armed} mode={t.mode} alt={t.altitude:.1f}m", flush=True)

    runner = MissionRunner(
        vehicle=backend,
        behaviors=[Takeoff(12.0), Hold(8.0), ReturnToLaunch()],
        safety=SafetyMonitor(SafetyLimits(max_radius_m=300, max_altitude_m=100)),
        # PX4 runs in real time, so the loop must too. `step` is a no-op on
        # this backend -- the world advances whether we tick or not.
        realtime=True,
        tick_hz=5.0,
        timeout_s=180.0,
    )

    result = await runner.run()

    print(flush=True)
    print(f"completed      : {result.completed}", flush=True)
    print(f"aborted        : {result.aborted_reason}", flush=True)
    print(f"duration       : {result.duration_s:.0f}s", flush=True)
    print(f"track samples  : {len(result.track)}", flush=True)
    if result.track:
        peak = max(-s.down for s in result.track)
        print(f"peak altitude  : {peak:.1f} m", flush=True)
    return 0 if result.completed else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
