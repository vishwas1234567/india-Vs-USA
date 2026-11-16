"""Road understanding: temporally smoothed drivable-area segmentation, free space, scene descriptors, panoptic fusion.

Segmentation here is classical (colour model from the area in front of the vehicle + temporal smoothing); a learned model
(SegFormer / Mask2Former via irp.common.SegformerRoad or `PanopticFuser.from_semantic`) plugs into the same interfaces.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .camera import CameraModel
from .common import ClassicalRoad, Detection

STUFF_ROAD, STUFF_OTHER = 1, 2        # panoptic stuff categories (segment id = category * 1000 + instance)
THING_BASE = {"car": 10, "heavy": 11, "two_wheeler": 12, "pedestrian": 13, "auto": 14, "animal": 15, "unknown": 16, "sign": 17}


class RoadSegmenter:
    """Per-frame road mask from any base segmenter, smoothed over time (exponential moving average of the mask)."""

    def __init__(self, base=None, alpha: float = 0.5):
        self.base, self.alpha, self.prob = base or ClassicalRoad(), alpha, None

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        m = self.base(bgr).astype(np.float32)
        self.prob = m if self.prob is None or self.prob.shape != m.shape else self.alpha * m + (1 - self.alpha) * self.prob
        out = (self.prob > 0.5).astype(np.uint8)
        return cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))


def free_space_depth(road: np.ndarray, cam: CameraModel, n_cols: int = 64, z_max: float = 60.0) -> np.ndarray:
    """Distance (m) along each of n_cols image columns to the first non-road pixel above the bottom edge (flat-ground model)."""
    h, w = road.shape
    cols = np.linspace(0, w - 1, n_cols).astype(int)
    depth = np.full(n_cols, z_max, np.float32)
    for j, c in enumerate(cols):
        col = road[:, c]
        if col[h - 1] == 0:
            depth[j] = 0.0
            continue
        edge = np.nonzero(col[: h - 1] == 0)[0]
        edge = edge[edge > cam.horizon_row] if len(edge) else edge
        v = float(edge.max()) if len(edge) else cam.horizon_row + 1.0
        depth[j] = float(min(z_max, cam.image_to_ground(np.array([c]), np.array([v + 1.0]))[0, 1]))
    return depth


def scene_descriptor(bgr: np.ndarray, road: Optional[np.ndarray] = None, n_objects: int = 0, class_counts: Optional[Dict[str, int]] = None) -> Dict[str, object]:
    """Lighting, glare, haze (dark-channel prior), road coverage and a traffic-complexity index."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(bgr, (160, max(int(bgr.shape[0] * 160 / bgr.shape[1]), 8)))
    dark = cv2.erode(small.min(2), np.ones((7, 7), np.uint8))
    brightness = float(g.mean())
    lighting = "night" if brightness < 50 else "dusk/dim" if brightness < 90 else "day"
    entropy = 0.0
    if class_counts and sum(class_counts.values()):
        p = np.array([v for v in class_counts.values() if v], float)
        p /= p.sum()
        entropy = float(-(p * np.log2(p)).sum())
    return {"brightness": brightness, "lighting": lighting, "glare_frac": float((g >= 250).mean()),
            "haze_dark_channel": float(dark.mean() / 255.0), "road_coverage": float(road.mean()) if road is not None else None,
            "complexity_index": float(n_objects * (1.0 + entropy))}


class PanopticFuser:
    """Fuse a semantic map (road / other) with instance detections into one panoptic id map.

    id = category * 1000 + instance. Stuff: road (1), other (2). Things: THING_BASE[class] + instance (1-based).
    Instance masks are approximate: the detection box minus road pixels (objects occlude the road), unless real masks are given.
    """

    def fuse(self, road: np.ndarray, dets: Sequence[Detection], masks: Optional[Sequence[np.ndarray]] = None) -> np.ndarray:
        pan = np.where(road > 0, STUFF_ROAD * 1000, STUFF_OTHER * 1000).astype(np.int32)
        counters: Dict[str, int] = {}
        order = sorted(range(len(dets)), key=lambda i: -dets[i].box[3])      # far objects first so near ones overwrite
        for i in order:
            d = dets[i]
            cat = d.cat or "unknown"
            counters[cat] = counters.get(cat, 0) + 1
            x1, y1, x2, y2 = (max(int(v), 0) for v in d.box)
            if masks is not None:
                m = masks[i].astype(bool)
            else:
                m = np.zeros(road.shape, bool)
                m[y1:y2, x1:x2] = True
                m &= ~((road > 0) & (np.arange(road.shape[0])[:, None] > y1 + 0.75 * (y2 - y1)))   # keep the lower part of the box as the object
            pan[m] = (THING_BASE.get(cat, 16)) * 1000 + counters[cat]
        return pan

    @staticmethod
    def colorize(pan: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(1)
        lut: Dict[int, Tuple[int, int, int]] = {}
        out = np.zeros(pan.shape + (3,), np.uint8)
        for i in np.unique(pan):
            if i // 1000 == STUFF_ROAD:
                lut[i] = (90, 140, 90)
            elif i // 1000 == STUFF_OTHER:
                lut[i] = (70, 70, 70)
            else:
                lut[i] = tuple(int(v) for v in rng.integers(60, 255, 3))
            out[pan == i] = lut[i]
        return out
