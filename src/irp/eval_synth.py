"""Evaluation on synthetic scenes with exact ground truth: lane detection, road segmentation, motion detection.

These numbers validate geometry and logic only. They are NOT accuracy on real roads.
    python -m irp.eval_synth --out results/_synthetic
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from iscvd.metrics import lane_f1

from .camera import CameraModel, EgoMotion
from .detect import MotionDetector
from .lanes import LaneDetector
from .road import RoadSegmenter
from .synth import SceneObject, moving_camera_sequence, render_road


def _road_mask(cam: CameraModel, half: float = 4.5, ego: float = 0.0) -> np.ndarray:
    yy, xx = np.mgrid[0:cam.height, 0:cam.width]
    m = np.zeros((cam.height, cam.width), np.uint8)
    below = yy > cam.horizon_row + 2
    g = cam.image_to_ground(xx[below], yy[below])
    m[below] = np.abs(g[:, 0] + ego) < half
    return m


def evaluate(seeds: int = 5) -> dict:
    cam = CameraModel(640, 360)
    cases = {"straight": {}, "curved (c=0.002)": {"curvature": 0.002}, "ego offset 0.5 m": {"ego_offset": 0.5},
             "faded paint (25%)": {"paint": 0.25}, "glare": {"glare": 0.5}, "noise sigma 8": {"noise": 8.0},
             "no paint + free-space fallback": {"paint": 0.0, "ego_offset": 0.5, "road_half": 4.5}}
    lanes = {}
    for name, kw in cases.items():
        f1s, errs, ms, srcs = [], [], [], []
        for s in range(seeds):
            img, gt = render_road(cam, seed=s, **kw)
            det = LaneDetector(cam, track=False)
            mask = _road_mask(cam, 4.5, kw.get("ego_offset", 0.0)) if "road_half" in kw else None
            t = time.perf_counter()
            r = det.detect(img, mask)
            ms.append((time.perf_counter() - t) * 1000)
            f1s.append(lane_f1(r.polylines_image(cam), gt, 20.0)["F1"])
            truth = kw.get("ego_offset", 0.0) - kw.get("curvature", 0.0) * 10.0 ** 2     # ego offset at z = 10 m
            if r.ego_offset_m is not None:
                errs.append(abs(r.ego_offset_m - truth))
            srcs.append(r.source)
        lanes[name] = {"F1": float(np.mean(f1s)), "ego_offset_abs_err_m": float(np.mean(errs)) if errs else None,
                       "source": max(set(srcs), key=srcs.count), "ms": float(np.mean(ms)), "runs": seeds}
    cam_s = CameraModel(320, 180)
    objs = [SceneObject(-1.0, 12.0)]
    img, _ = render_road(cam_s, objects=objs, road_half=4.5)
    road = RoadSegmenter()(img)
    gt = _road_mask(cam_s)
    road_iou = float((road & gt).sum() / (road | gt).sum())
    motion = {}
    for name, mover in (("with moving object", SceneObject(-3.0, 14.0, 1.6, 1.4, (30, 60, 200))), ("no moving object", None)):
        frames = moving_camera_sequence(cam_s, n=40, speed_mps=8, mover=mover)
        ego, det, hits, n, fp = EgoMotion(), MotionDetector(), 0, 0, 0
        for i, f in enumerate(frames):
            d = det(f, ego.update(f))
            if i > 8:
                n += 1
                if mover is not None:
                    exp = cam_s.ground_to_image(np.array([-3.0 + 1.5 * i / 15.0]), np.array([14.0]))[0]
                    near = [b for b in d if abs((b.box[0] + b.box[2]) / 2 - exp[0]) < 35 and abs(b.box[3] - exp[1]) < 35]
                    hits += bool(near)
                    fp += len(d) - len(near)
                else:
                    fp += len(d)
        motion[name] = {"object_hit_rate": hits / n if mover is not None else None, "false_boxes_per_frame": fp / n}
    return {"lanes": lanes, "road_iou_synthetic": road_iou, "motion_detector": motion}


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/_synthetic")
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args(argv)
    res = evaluate(a.seeds)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "synthetic.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
