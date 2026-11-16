"""Redundancy: an independent, deterministic safety supervisor with a simple fallback controller.

The supervisor does NOT use the AI pipeline's perception or planning. It uses one independent range sensor (radar-like:
nearest in-corridor distance and closing speed) and the vehicle state, and it
  1. monitors the primary pipeline (exceptions, latency budget, non-finite outputs, staleness),
  2. enforces a safety envelope (a command that cannot stop before the nearest object is overridden), and
  3. runs a fallback controller (brake as needed / limp at low speed, hold lane) and latches SAFE_STOP on repeated faults.
In production this must run on separate hardware with its own sensor; here it is a separate class with no shared state.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

NORMAL, DEGRADED, SAFE_STOP = "NORMAL", "DEGRADED", "SAFE_STOP"


@dataclass
class PrimaryOutput:
    accel: float
    steer: float
    latency_s: float = 0.0
    ok: bool = True               # False if the primary raised an exception / produced no output


@dataclass
class RangeReading:
    distance: float               # nearest in-corridor object ahead (m), inf if none
    closing_speed: float          # m/s, positive if closing


class SafetySupervisor:
    def __init__(self, max_brake: float = 6.0, latency_budget_s: float = 0.15, stale_s: float = 0.6, limp_speed: float = 3.0,
                 d_margin: float = 3.0, fault_latch_s: float = 3.0):
        self.max_brake, self.budget, self.stale, self.limp, self.d_margin, self.latch = max_brake, latency_budget_s, stale_s, limp_speed, d_margin, fault_latch_s
        self.mode, self.fault_since, self.last_primary_ok_t = NORMAL, None, 0.0
        self.log: List[Tuple[float, str, str]] = []

    def reset(self):
        self.mode, self.fault_since, self.last_primary_ok_t, self.log = NORMAL, None, 0.0, []

    def _required_decel(self, v: float, rng: RangeReading) -> float:
        if not math.isfinite(rng.distance):
            return 0.0
        d = max(rng.distance - self.d_margin, 0.1)
        v_rel = max(v, v + rng.closing_speed * 0.0)               # ego speed governs the stopping distance for static objects
        return v_rel ** 2 / (2 * d)

    def fallback(self, v: float, rng: RangeReading) -> Tuple[float, str]:
        need = self._required_decel(v, rng)
        if need > 0.35 * self.max_brake:
            return -min(self.max_brake, max(need * 1.2, 2.0)), "brake"
        if self.mode == SAFE_STOP:
            return -min(self.max_brake, max(2.0, 1.2 * v)), "stop"
        return float(np.clip(0.8 * (self.limp - v), -3.0, 1.0)), "limp"

    def arbitrate(self, t: float, v: float, primary: Optional[PrimaryOutput], rng: RangeReading) -> Tuple[float, Optional[float], str]:
        """Return (accel, steering or None = keep primary/hold lane, reason). `steer=None` means the caller holds the lane."""
        fault = None
        if primary is None or not primary.ok:
            fault = "primary failure"
        elif primary.latency_s > self.budget:
            fault = f"latency {primary.latency_s * 1000:.0f} ms"
        elif not (math.isfinite(primary.accel) and math.isfinite(primary.steer)):
            fault = "non-finite command"
        elif t - self.last_primary_ok_t > self.stale:
            fault = "stale output"
        if fault is None:
            self.last_primary_ok_t = t
        need = self._required_decel(v, rng)
        # safety envelope: the primary command must be able to stop before the nearest object
        if fault is None and need > 0.6 * self.max_brake and primary.accel > -need:
            self.log.append((t, "envelope", f"need {need:.1f} m/s^2, primary {primary.accel:.1f}"))
            return -min(self.max_brake, need * 1.2), primary.steer, "envelope override"
        if fault is not None:
            if self.fault_since is None:
                self.fault_since = t
                self.log.append((t, "fault", fault))
            self.mode = SAFE_STOP if (t - self.fault_since) > self.latch else DEGRADED
            a, why = self.fallback(v, rng)
            return a, None, f"{self.mode}: {fault} -> {why}"
        if self.mode != NORMAL and self.mode != SAFE_STOP:       # recovered after a transient fault
            self.mode, self.fault_since = NORMAL, None
        return primary.accel, primary.steer, "primary"
