"""Camera model, ego-motion estimation, moving/static camera classification and horizon estimation."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class CameraModel:
    """Pinhole camera over flat ground (zero roll). horizon is the row of the vanishing line."""

    width: int
    height: int
    hfov_deg: float = 70.0
    height_m: float = 1.4
    horizon_frac: float = 0.45

    @property
    def f(self) -> float:
        return (self.width / 2) / math.tan(math.radians(self.hfov_deg) / 2)

    @property
    def cx(self) -> float:
        return self.width / 2

    @property
    def horizon_row(self) -> float:
        return self.height * self.horizon_frac

    def ground_to_image(self, X: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """Ground points (lateral X, forward Z in metres) -> pixels (u, v)."""
        Z = np.maximum(np.asarray(Z, float), 1e-3)
        return np.stack([self.cx + self.f * np.asarray(X, float) / Z, self.horizon_row + self.f * self.height_m / Z], -1)

    def image_to_ground(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        dv = np.maximum(np.asarray(v, float) - self.horizon_row, 1e-3)
        Z = self.f * self.height_m / dv
        return np.stack([(np.asarray(u, float) - self.cx) * Z / self.f, Z], -1)

    def bev_homography(self, x_range=(-6.0, 6.0), z_range=(6.0, 40.0), px_per_m: float = 30.0):
        """Homography image->bird's-eye view and the BEV canvas size (width, height)."""
        w, h = int((x_range[1] - x_range[0]) * px_per_m), int((z_range[1] - z_range[0]) * px_per_m)
        g = np.array([[x_range[0], z_range[1]], [x_range[1], z_range[1]], [x_range[1], z_range[0]], [x_range[0], z_range[0]]])
        src = self.ground_to_image(g[:, 0], g[:, 1]).astype(np.float32)
        dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
        return cv2.getPerspectiveTransform(src, dst), (w, h), (x_range, z_range, px_per_m)


@dataclass
class Motion:
    dx: float
    dy: float
    rot: float
    scale: float
    n_inliers: int
    H: Optional[np.ndarray] = None      # ground-plane homography previous->current, in coordinates scaled by `s`
    s: float = 1.0                      # scale between original pixels and the coordinates used for H

    @property
    def magnitude(self) -> float:
        return math.hypot(self.dx, self.dy)


class EgoMotion:
    """Frame-to-frame global motion from sparse optical flow on the static part of the image, with a RANSAC
    similarity model. `moving_camera()` is true when the median background flow exceeds a threshold."""

    def __init__(self, scale_width: int = 320, exclude_bottom_frac: float = 0.0):
        self.sw, self.excl = scale_width, exclude_bottom_frac
        self.prev: Optional[np.ndarray] = None
        self.history: List[float] = []
        self.frac_hist: List[float] = []

    def update(self, bgr: np.ndarray, mask: Optional[np.ndarray] = None) -> Motion:
        h, w = bgr.shape[:2]
        s = self.sw / w
        g = cv2.cvtColor(cv2.resize(bgr, (self.sw, int(h * s))), cv2.COLOR_BGR2GRAY)
        m = Motion(0.0, 0.0, 0.0, 1.0, 0, None, s)
        if self.prev is not None and self.prev.shape == g.shape:
            fm = np.full(g.shape, 255, np.uint8)
            if self.excl:
                fm[int(g.shape[0] * (1 - self.excl)):] = 0
            if mask is not None:
                fm[cv2.resize(mask, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST) > 0] = 0
            p0 = cv2.goodFeaturesToTrack(self.prev, 250, 0.02, 6, blockSize=7, mask=fm)
            if p0 is not None and len(p0) >= 8:
                p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev, g, p0, None, winSize=(21, 21), maxLevel=3)
                ok = (st.ravel() == 1) & (err.ravel() < 12)
                if ok.sum() >= 8:
                    frac = float(np.mean(np.linalg.norm((p1 - p0).reshape(-1, 2)[ok], axis=1) > 0.5))   # features moving > 0.5 px at 320 px width
                    a, inl = cv2.estimateAffinePartial2D(p0[ok], p1[ok], method=cv2.RANSAC, ransacReprojThreshold=2.0)
                    if a is not None:
                        sc = math.hypot(a[0, 0], a[1, 0])
                        m = Motion(float(a[0, 2] / s), float(a[1, 2] / s), math.atan2(a[1, 0], a[0, 0]), sc, int(inl.sum()), None, s)
                    lower = ok & (p0.reshape(-1, 2)[:, 1] > 0.45 * g.shape[0])
                    if lower.sum() >= 14:       # ground-plane homography handles forward motion (parallax of the road)
                        Hm, hin = cv2.findHomography(p0[lower], p1[lower], cv2.RANSAC, 2.0)
                        if Hm is not None and hin is not None and hin.sum() >= 10:
                            m.H = Hm
        self.prev = g
        self.history.append(m.magnitude)
        self.frac_hist.append(locals().get("frac", 0.0))
        return m

    def moving_camera(self, min_frames: int = 10, threshold: float = 0.15) -> bool:
        """True if the camera moves: the median (over frames) fraction of tracked background features moving more than
        0.5 px (at 320 px width) exceeds `threshold`. Flow fractions are used rather than the fitted translation because
        forward driving mostly produces expansion. Heuristic: checked on two real fixed-camera clips (fractions 0.02-0.06)
        and a synthetic drive (0.30); NOT yet validated on real moving-vehicle footage."""
        if len(self.frac_hist) < min_frames:
            return False
        return float(np.median(self.frac_hist[1:])) > threshold


class HorizonEstimator:
    """Horizon row from the vanishing point of long lane/edge lines in the lower half; temporally smoothed."""

    def __init__(self, default_frac: float = 0.45, alpha: float = 0.1):
        self.value, self.alpha, self.default = None, alpha, default_frac
        self.samples: List[float] = []

    def update(self, bgr: np.ndarray) -> float:
        h, w = bgr.shape[:2]
        g = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        edges = cv2.Canny(g, 60, 150)
        edges[: int(h * 0.35)] = 0
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 40, minLineLength=int(h * 0.15), maxLineGap=int(h * 0.04))
        est = None
        if lines is not None:
            segs = []
            for x1, y1, x2, y2 in lines.reshape(-1, 4):
                ang = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
                if 20 < ang < 80 or 100 < ang < 160:   # converging lines only
                    segs.append((x1, y1, x2, y2))
            if len(segs) >= 4:
                ys = []
                for i in range(len(segs)):
                    for j in range(i + 1, len(segs)):
                        a, b = segs[i], segs[j]
                        d = (a[0] - a[2]) * (b[1] - b[3]) - (a[1] - a[3]) * (b[0] - b[2])
                        if abs(d) < 1e-6:
                            continue
                        t = ((a[0] - b[0]) * (b[1] - b[3]) - (a[1] - b[1]) * (b[0] - b[2])) / d
                        y = a[1] + t * (a[3] - a[1])
                        if 0.1 * h < y < 0.8 * h:
                            ys.append(y)
                if len(ys) >= 3:
                    est = float(np.median(ys)) / h
        if est is not None:
            self.samples.append(est)
            self.value = est if self.value is None else (1 - self.alpha) * self.value + self.alpha * est
        return self.value if self.value is not None else self.default

    def robust(self) -> float:
        """Median of all per-frame estimates (more stable than the running value), or the default."""
        return float(np.median(self.samples)) if self.samples else self.default
