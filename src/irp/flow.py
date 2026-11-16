"""Traffic-flow analytics from tracked road users on a static camera.

* Virtual counting gates: a track crossing a gate segment is counted with its direction.
* Headways: time gaps between consecutive crossings of a gate (all directions pooled).
* Bursts: periods of elevated road occupancy (platoons). On signalised junctions bursts are
  expected to follow the signal cycle; this is a hypothesis to be checked against the video,
  not an assumption made by the code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]


@dataclass(frozen=True)
class Gate:
    """Counting segment in normalised image coordinates.

    `labels` = (label for positive->negative side crossing, label for negative->positive).
    The side of a point is the cross product of the gate direction and (point - p1).
    """

    name: str
    p1: Point
    p2: Point
    labels: Tuple[str, str]


def default_gates() -> List[Gate]:
    return [
        Gate("mid-vertical", (0.50, 0.40), (0.50, 1.00), ("L->R", "R->L")),
        Gate("near-horizontal", (0.00, 0.80), (1.00, 0.80), ("away", "toward")),
    ]


def side(p: Point, a: Point, b: Point) -> float:
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _segments_intersect(p: Point, q: Point, a: Point, b: Point) -> bool:
    """True if segment p-q crosses gate a-b. A point exactly on the line counts as the negative side,
    so a track that lands on the gate is still counted once."""
    if (side(p, a, b) > 0) == (side(q, a, b) > 0):
        return False
    return (side(a, p, q) > 0) != (side(b, p, q) > 0)


def track_crossings(base_points: Sequence[Tuple[int, Point]], gate: Gate, size: Tuple[int, int]) -> List[Tuple[int, str]]:
    """Crossings (frame_index, direction_label) of a track whose base points are (frame, (x, y)) in pixels."""
    w, h = size
    a, b = (gate.p1[0] * w, gate.p1[1] * h), (gate.p2[0] * w, gate.p2[1] * h)
    out: List[Tuple[int, str]] = []
    for (_, p), (f1, q) in zip(base_points[:-1], base_points[1:]):
        if _segments_intersect(p, q, a, b):
            out.append((f1, gate.labels[0] if side(p, a, b) > 0 else gate.labels[1]))
    return out


def headway_stats(times_s: Sequence[float]) -> Dict[str, object]:
    t = np.sort(np.asarray(times_s, float))
    if len(t) < 2:
        return {"n": int(len(t)), "mean_s": None, "median_s": None, "p90_s": None}
    gaps = np.diff(t)
    return {"n": int(len(t)), "mean_s": round(float(gaps.mean()), 2), "median_s": round(float(np.median(gaps)), 2),
            "p90_s": round(float(np.percentile(gaps, 90)), 2)}


def per_bin_counts(times_s: Sequence[float], duration_s: float, bin_s: float = 10.0) -> List[int]:
    edges = np.arange(0.0, duration_s + bin_s, bin_s)
    counts, _ = np.histogram(np.asarray(times_s, float), bins=edges)
    return [int(c) for c in counts]


def detect_bursts(t_s: Sequence[float], occupancy: Sequence[float], fps: float, rel_threshold: float = 0.35,
                  min_abs: float = 0.01, merge_gap_s: float = 3.0, min_dur_s: float = 2.0) -> Dict[str, object]:
    """Find bursts of elevated occupancy using a smoothed series, a relative threshold and hysteresis merging.

    Returns {"bursts": [{start_s, end_s, peak}], "n": int, "mean_duration_s", "mean_interval_s", "interval_cv"}.
    """
    t = np.asarray(t_s, float)
    occ = np.asarray(occupancy, float)
    if len(occ) < 3:
        return {"bursts": [], "n": 0, "mean_duration_s": None, "mean_interval_s": None, "interval_cv": None}
    k = max(int(round(fps)), 1)
    sm = np.convolve(occ, np.ones(k) / k, mode="same")
    thr = max(rel_threshold * float(np.percentile(sm, 95)), min_abs)
    active = sm > thr
    segs: List[List[int]] = []
    start = None
    for i, a in enumerate(active):
        if a and start is None:
            start = i
        if (not a or i == len(active) - 1) and start is not None:
            segs.append([start, i if not a else i + 1])
            start = None
    merged: List[List[int]] = []
    for s, e in segs:
        if merged and t[min(s, len(t) - 1)] - t[merged[-1][1] - 1] < merge_gap_s:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    bursts = []
    for s, e in merged:
        if t[e - 1] - t[s] >= min_dur_s:
            bursts.append({"start_s": round(float(t[s]), 2), "end_s": round(float(t[e - 1]), 2),
                           "peak": round(float(sm[s:e].max()), 4)})
    starts = np.array([b["start_s"] for b in bursts])
    intervals = np.diff(starts) if len(starts) > 1 else np.array([])
    return {
        "bursts": bursts, "n": len(bursts), "threshold": round(thr, 4),
        "mean_duration_s": round(float(np.mean([b["end_s"] - b["start_s"] for b in bursts])), 2) if bursts else None,
        "mean_interval_s": round(float(intervals.mean()), 2) if len(intervals) else None,
        "interval_cv": round(float(intervals.std() / intervals.mean()), 3) if len(intervals) > 1 and intervals.mean() > 0 else None,
    }
