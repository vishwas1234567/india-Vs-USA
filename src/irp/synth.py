"""Synthetic perspective road scenes with exact ground truth (lane polylines, objects) for testing and demos.

These scenes validate geometry and logic. They are NOT evidence of performance on real Indian roads.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .camera import CameraModel


@dataclass
class SceneObject:
    X: float            # lateral position (m)
    Z: float            # forward distance (m)
    width: float = 1.8
    height: float = 1.5
    color: Tuple[int, int, int] = (40, 40, 180)
    kind: str = "car"


def render_road(cam: CameraModel, lane_x: Sequence[float] = (-5.25, -1.75, 1.75, 5.25), curvature: float = 0.0,
                phase: float = 0.0, dashed: Sequence[bool] = (False, True, True, False), paint: float = 1.0,
                ego_offset: float = 0.0, objects: Sequence[SceneObject] = (), noise: float = 3.0, glare: float = 0.0,
                seed: int = 0, z_gt: Tuple[float, float] = (6.0, 40.0), road_half: float = 7.0):
    """Return (image BGR, gt_lanes) where gt_lanes are image polylines of each painted lane line.

    ego_offset: lateral position of the vehicle relative to the lane centre (m, +right): lanes shift by -ego_offset.
    paint in [0,1]: marking brightness contrast (low = faded paint).
    """
    rng = np.random.default_rng(seed)
    h, w = cam.height, cam.width
    img = np.zeros((h, w, 3), np.float32)
    img[:] = (225, 200, 170)                                   # sky
    yy, xx = np.mgrid[0:h, 0:w]
    below = yy > cam.horizon_row + 2
    gnd = cam.image_to_ground(xx[below], yy[below])
    X, Z = gnd[:, 0], gnd[:, 1]
    road = np.abs(X + ego_offset) < road_half
    colour = np.where(road[:, None], np.array([105, 105, 110], np.float32), np.array([70, 110, 90], np.float32))
    for lx, d in zip(lane_x, dashed):
        centre = lx - ego_offset + curvature * Z ** 2
        on = np.abs(X - centre) < 0.08 + 0.004 * Z
        if d:
            on &= ((Z + phase) % 8.0) < 4.0
        colour[on] = colour[on] * (1 - paint) + np.array([235, 235, 235], np.float32) * paint
    img[below] = colour
    for o in sorted(objects, key=lambda o: -o.Z):
        base = cam.ground_to_image(np.array([o.X]), np.array([o.Z]))[0]
        pw = cam.f * o.width / o.Z
        ph = cam.f * o.height / o.Z
        cv2.rectangle(img, (int(base[0] - pw / 2), int(base[1] - ph)), (int(base[0] + pw / 2), int(base[1])), o.color, -1)
    if glare > 0:
        blob = np.exp(-(((xx - 0.7 * w) / (0.35 * w)) ** 2 + ((yy - 0.15 * h) / (0.3 * h)) ** 2))
        img += 255 * glare * blob[..., None]
    img += rng.normal(0, noise, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    zs = np.linspace(z_gt[0], z_gt[1], 40)
    gt = []
    for lx in lane_x:
        pts = cam.ground_to_image(lx - ego_offset + curvature * zs ** 2, zs)
        vis = pts[(pts[:, 0] >= 0) & (pts[:, 0] < w)]
        if len(vis) >= 8 and len(vis) >= 0.6 * len(pts):   # lane is mostly visible: keep its visible part
            gt.append(vis.astype(np.float32))
    return img, gt


def moving_camera_sequence(cam: CameraModel, n: int = 40, speed_mps: float = 8.0, fps: float = 15.0,
                           mover: Optional[SceneObject] = None, seed: int = 0):
    """Frames of a drive: lane dashes stream toward the camera; an optional independently moving object crosses laterally."""
    frames = []
    for i in range(n):
        objs = []
        if mover is not None:
            objs.append(SceneObject(mover.X + 1.5 * i / fps, mover.Z, mover.width, mover.height, mover.color, mover.kind))
        img, _ = render_road(cam, phase=speed_mps * i / fps, objects=objs, seed=seed + i, noise=2.0)
        frames.append(img)
    return frames
