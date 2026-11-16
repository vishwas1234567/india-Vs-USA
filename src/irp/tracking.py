"""Kalman-filter + Hungarian-assignment tracker for foreground blobs.

Drop-in alternative to `irp.static.CentroidTracker` (same interface):
  * constant-velocity Kalman filter per track (predicts through short occlusions),
  * globally optimal assignment (Hungarian algorithm; greedy fallback without SciPy),
  * association cost mixes normalised centre distance and (1 - IoU) with a distance gate.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:  # SciPy is optional
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover
    linear_sum_assignment = None

Box = Tuple[int, int, int, int]

_F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], float)
_H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
_Q = np.diag([1.0, 1.0, 4.0, 4.0])
_R = np.diag([9.0, 9.0])


class _Kalman:
    """2-D constant-velocity filter, state [x, y, vx, vy] in pixels and pixels/frame."""

    def __init__(self, x: float, y: float):
        self.x = np.array([x, y, 0.0, 0.0])
        self.P = np.diag([10.0, 10.0, 100.0, 100.0])

    def predict(self) -> np.ndarray:
        self.x = _F @ self.x
        self.P = _F @ self.P @ _F.T + _Q
        return self.x[:2].copy()

    def update(self, z: Sequence[float]) -> None:
        y = np.asarray(z, float) - _H @ self.x
        S = _H @ self.P @ _H.T + _R
        K = self.P @ _H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ _H) @ self.P


def iou(a: Box, b: Box) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def assign(cost: np.ndarray, invalid: float = 1e6) -> List[Tuple[int, int]]:
    """Minimum-cost assignment; pairs with cost >= invalid are discarded."""
    if cost.size == 0:
        return []
    if linear_sum_assignment is not None:
        rows, cols = linear_sum_assignment(cost)
        return [(int(r), int(c)) for r, c in zip(rows, cols) if cost[r, c] < invalid]
    pairs, used_r, used_c = [], set(), set()  # greedy fallback
    for flat in np.argsort(cost, axis=None):
        r, c = divmod(int(flat), cost.shape[1])
        if cost[r, c] >= invalid:
            break
        if r not in used_r and c not in used_c:
            pairs.append((r, c))
            used_r.add(r)
            used_c.add(c)
    return pairs


class KalmanTracker:
    def __init__(self, max_dist: float = 80.0, max_miss: int = 10):
        self.max_dist, self.max_miss = max_dist, max_miss
        self.tracks: Dict[int, dict] = {}
        self.finished: List[dict] = []
        self._next_id = 1

    def update(self, boxes: Sequence[Box], frame_idx: int) -> None:
        ids = list(self.tracks)
        preds = {tid: self.tracks[tid]["kf"].predict() for tid in ids}
        cost = np.full((len(ids), len(boxes)), 1e6)
        for i, tid in enumerate(ids):
            t = self.tracks[tid]
            px, py = preds[tid]
            ox, oy = t["c"]
            shifted = (t["box"][0] + px - ox, t["box"][1] + py - oy, t["box"][2] + px - ox, t["box"][3] + py - oy)
            for j, b in enumerate(boxes):
                d = math.hypot((b[0] + b[2]) / 2 - px, (b[1] + b[3]) / 2 - py)
                if d <= self.max_dist:
                    cost[i, j] = 0.5 * d / self.max_dist + 0.5 * (1.0 - iou(shifted, b))
        matched_t, matched_d = set(), set()
        for i, j in assign(cost):
            t = self.tracks[ids[i]]
            b = boxes[j]
            t["kf"].update(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
            t["c"] = (float(t["kf"].x[0]), float(t["kf"].x[1]))
            t["box"], t["miss"], t["matched"] = b, 0, True
            t["hist"].append((frame_idx, t["c"], b))
            matched_t.add(ids[i])
            matched_d.add(j)
        for tid in ids:
            if tid not in matched_t:
                t = self.tracks[tid]
                t["matched"] = False
                t["miss"] += 1
                t["c"] = (float(t["kf"].x[0]), float(t["kf"].x[1]))  # coast on the prediction
                if t["miss"] > self.max_miss:
                    self.finished.append(self.tracks.pop(tid))
        for j, b in enumerate(boxes):
            if j not in matched_d:
                c = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
                self.tracks[self._next_id] = {"id": self._next_id, "box": b, "c": c, "miss": 0, "matched": True,
                                              "hist": [(frame_idx, c, b)], "kf": _Kalman(*c)}
                self._next_id += 1

    def all_tracks(self) -> List[dict]:
        return self.finished + list(self.tracks.values())


def track_detections(tracker: KalmanTracker, dets: Sequence, frame_idx: int) -> None:
    """Run the Kalman tracker on the boxes of `dets` and write the resulting ids into `det.track_id`."""
    tracker.update([tuple(d.box) for d in dets], frame_idx)
    live = {tuple(t["box"]): t["id"] for t in tracker.tracks.values() if t["miss"] == 0}
    for d in dets:
        d.track_id = live.get(tuple(d.box), -1)
