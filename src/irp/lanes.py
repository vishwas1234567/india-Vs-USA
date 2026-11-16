"""Lane detection and ego-path estimation for Indian roads (markings often faded or absent).

Pipeline: bird's-eye-view (BEV) warp from the camera model -> narrow-bright-structure map (top-hat + yellow) ->
histogram peaks -> sliding windows -> quadratic fits X(Z) = a Z^2 + b Z + c (metres) with outlier rejection ->
temporal tracking (EMA) -> ego-lane offset/heading. When markings are weak the ego path falls back to the centre line of the
drivable area (free space), so a path estimate exists even on unmarked roads.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
from scipy.signal import find_peaks

from .camera import CameraModel


@dataclass
class Lane:
    coeffs: np.ndarray            # (a, b, c): X = a Z^2 + b Z + c, metres
    confidence: float
    z_span: Tuple[float, float]
    age: int = 1
    misses: int = 0

    def x_at(self, z) -> np.ndarray:
        a, b, c = self.coeffs
        return a * np.asarray(z, float) ** 2 + b * np.asarray(z, float) + c


@dataclass
class LaneResult:
    lanes: List[Lane]
    state: str                    # clear | faded | absent
    source: str                   # markings | freespace | none
    ego_offset_m: Optional[float] # + = vehicle right of the lane/path centre
    heading_err_rad: Optional[float]
    confidence: float
    centre: Optional[np.ndarray] = None   # ego path polynomial (a, b, c) in metres

    def polylines_image(self, cam: CameraModel, z_range=(6.0, 40.0), n: int = 40) -> List[np.ndarray]:
        zs = np.linspace(z_range[0], z_range[1], n)
        out = []
        for l in self.lanes:
            pts = cam.ground_to_image(l.x_at(zs), zs)
            vis = pts[(pts[:, 0] >= 0) & (pts[:, 0] < cam.width)]
            if len(vis) >= 8:
                out.append(vis.astype(np.float32))
        return out


class LaneDetector:
    def __init__(self, cam: CameraModel, x_range=(-6.0, 6.0), z_range=(6.0, 40.0), px_per_m: float = 30.0,
                 lane_width_m: float = 3.5, track: bool = True, min_conf: float = 0.35):
        self.cam, self.x_range, self.z_range, self.ppm = cam, x_range, z_range, px_per_m
        self.lane_width, self.track, self.min_conf = lane_width_m, track, min_conf
        self.H, self.size, _ = cam.bev_homography(x_range, z_range, px_per_m)
        self.tracks: List[Lane] = []

    # ---- coordinate helpers
    def _to_m(self, xb, yb):
        return self.x_range[0] + np.asarray(xb, float) / self.ppm, self.z_range[1] - np.asarray(yb, float) / self.ppm

    def marking_map(self, bev: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (int(0.5 * self.ppm) | 1, int(0.5 * self.ppm) | 1))
        th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, k)
        thr = max(22.0, float(np.percentile(th, 96)) * 0.6)
        m = (th > thr).astype(np.uint8)
        lab = cv2.cvtColor(bev, cv2.COLOR_BGR2LAB)
        yellow = (lab[..., 2] > 150) & (g > 110)
        m |= yellow.astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        return m

    def _fit_from_peak(self, m: np.ndarray, x0: int):
        h, w = m.shape
        win_h, win_w = int(2.0 * self.ppm), int(0.9 * self.ppm)
        n_win = h // win_h
        xs, ys, hits = [], [], 0
        cx = x0
        for i in range(n_win):
            y1, y0 = h - i * win_h, h - (i + 1) * win_h
            xa, xb = max(cx - win_w // 2, 0), min(cx + win_w // 2, w)
            yy, xx = np.nonzero(m[y0:y1, xa:xb])
            if len(xx) >= 12:
                hits += 1
                xs.extend((xx + xa).tolist()); ys.extend((yy + y0).tolist())
                cx = int(np.mean(xx) + xa)
        if hits < 3 or len(xs) < 40:
            return None
        xm, zm = self._to_m(np.array(xs), np.array(ys))
        coef = np.polyfit(zm, xm, 2)
        for _ in range(2):  # reject outliers
            res = np.abs(np.polyval(coef, zm) - xm)
            keep = res < max(0.3, 2.0 * np.median(res))
            if keep.sum() < 30:
                break
            coef = np.polyfit(zm[keep], xm[keep], 2)
        res = np.abs(np.polyval(coef, zm) - xm)
        inlier = float((res < 0.3).mean())
        coverage = hits / n_win
        conf = float(np.clip(0.6 * coverage + 0.4 * inlier, 0, 1))
        return Lane(coef, conf, (float(zm.min()), float(zm.max())))

    def _detect_markings(self, bev: np.ndarray, road_bev: Optional[np.ndarray]) -> List[Lane]:
        m = self.marking_map(bev)
        if road_bev is not None:
            m &= cv2.dilate(road_bev, np.ones((9, 9), np.uint8))
        h = m.shape[0]
        hist = np.convolve(m[int(h * 0.4):].sum(0).astype(float), np.ones(9) / 9, mode="same")
        if hist.max() <= 0:
            return []
        peaks, _ = find_peaks(hist, height=0.2 * hist.max(), distance=int(1.2 * self.ppm), prominence=0.1 * hist.max())
        lanes = []
        for p in peaks[:8]:
            lane = self._fit_from_peak(m, int(p))
            if lane is not None and lane.confidence >= self.min_conf:
                lanes.append(lane)
        return lanes

    def _track(self, found: List[Lane]) -> List[Lane]:
        used = set()
        for t in self.tracks:
            best, bd = None, 0.8
            for j, f in enumerate(found):
                d = abs(t.x_at(self.z_range[0] + 4) - f.x_at(self.z_range[0] + 4))
                if j not in used and d < bd:
                    best, bd = j, d
            if best is None:
                t.misses += 1
                continue
            used.add(best)
            f = found[best]
            t.coeffs = 0.6 * t.coeffs + 0.4 * f.coeffs
            t.confidence, t.age, t.misses = 0.6 * t.confidence + 0.4 * f.confidence, t.age + 1, 0
            t.z_span = f.z_span
        for j, f in enumerate(found):
            if j not in used:
                self.tracks.append(f)
        self.tracks = [t for t in self.tracks if t.misses <= 6]
        return [t for t in self.tracks if t.misses == 0]

    def _freespace_centre(self, road_bev: np.ndarray):
        h, w = road_bev.shape
        xs, zs = [], []
        for y in range(0, h, 6):
            cols = np.nonzero(road_bev[y])[0]
            if len(cols) < 0.5 * self.ppm:
                continue
            # contiguous run containing (or nearest to) the vehicle axis
            axis = int(-self.x_range[0] * self.ppm)
            runs = np.split(cols, np.nonzero(np.diff(cols) > 3)[0] + 1)
            run = min(runs, key=lambda r: 0 if r[0] <= axis <= r[-1] else min(abs(r[0] - axis), abs(r[-1] - axis)))
            xm, zm = self._to_m((run[0] + run[-1]) / 2, y)
            xs.append(float(xm)); zs.append(float(zm))
        if len(xs) < 8:
            return None
        return np.polyfit(zs, xs, 2), len(xs) / (h / 6)

    def detect(self, bgr: np.ndarray, road_mask: Optional[np.ndarray] = None) -> LaneResult:
        bev = cv2.warpPerspective(bgr, self.H, self.size, flags=cv2.INTER_LINEAR)
        road_bev = None
        if road_mask is not None:
            road_bev = cv2.warpPerspective(road_mask.astype(np.uint8), self.H, self.size, flags=cv2.INTER_NEAREST)
        found = self._detect_markings(bev, road_bev)
        lanes = self._track(found) if self.track else found
        z_ref = self.z_range[0] + 4.0
        left = [l for l in lanes if l.x_at(z_ref) < 0]
        right = [l for l in lanes if l.x_at(z_ref) >= 0]
        ll = max(left, key=lambda l: l.x_at(z_ref)) if left else None     # nearest left line
        rr = min(right, key=lambda l: l.x_at(z_ref)) if right else None   # nearest right line
        centre, conf, source = None, 0.0, "none"
        if ll is not None and rr is not None:
            centre, conf, source = (ll.coeffs + rr.coeffs) / 2, min(ll.confidence, rr.confidence), "markings"
        elif ll is not None:
            centre, conf, source = ll.coeffs + np.array([0, 0, self.lane_width / 2]), 0.7 * ll.confidence, "markings"
        elif rr is not None:
            centre, conf, source = rr.coeffs - np.array([0, 0, self.lane_width / 2]), 0.7 * rr.confidence, "markings"
        state = "clear" if (ll is not None and rr is not None and conf >= 0.5) else "faded" if centre is not None else "absent"
        if centre is None and road_bev is not None:
            fs = self._freespace_centre(road_bev)
            if fs is not None:
                centre, conf, source = fs[0], 0.4 * fs[1], "freespace"
        ego, head = None, None
        if centre is not None:
            ego = float(-np.polyval(centre, z_ref))                    # + = vehicle to the right of the centre line
            head = float(np.arctan(np.polyval(np.polyder(centre), z_ref)))
        return LaneResult(lanes, state, source, ego, head, float(conf), centre)
