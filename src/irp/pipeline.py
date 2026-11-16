"""Unified perception pipeline for any footage: video file/URL, image, image folder, glob or webcam.

Per frame (vehicle mode): ego-motion -> detection (YOLO / ONNX / motion fallback) -> tracking -> road segmentation + free space ->
lane detection (with free-space fallback) -> scene descriptor -> intent + risk -> overlay. `mode="auto"` pre-scans the first frames:
a fixed camera is routed to the static-camera analyser (irp.static), a moving camera to this pipeline. Stages are timed.

    python -m irp.pipeline --source drive.mp4 --out results/drive1
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .camera import CameraModel, EgoMotion, HorizonEstimator, Motion
from .common import CATEGORY_COLORS, Detection, corridor, est_dist, in_poly, put_text, write_csv
from .detect import build_detector
from .intent import IntentHMM, IntentTracker, simulate_agents
from .lanes import LaneDetector, LaneResult
from .quality import assess_quality
from .road import PanopticFuser, RoadSegmenter, free_space_depth, scene_descriptor
from .runtime import Profiler, VideoWriter, prefetch, select_device
from .sources import FrameSource, open_source
from .tracking import KalmanTracker, track_detections
from .vehicle import assess_risk

CONSERVATIVE = frozenset({"two_wheeler", "pedestrian", "animal", "unknown"})
VULN_FOR_INTENT = ("pedestrian", "two_wheeler", "animal", "unknown")


@dataclass
class PipelineConfig:
    source: str
    out: str = "results/pipeline"
    mode: str = "auto"                 # auto | vehicle | static
    detector: str = "auto"             # auto | motion | onnx:<path> | yolo:<weights>
    stride: int = 1
    start_s: float = 0.0
    duration_s: float = 0.0
    max_frames: int = 0
    max_side: int = 960
    hfov: float = 70.0
    cam_height: float = 1.4
    horizon: float = 0.45
    device: str = "auto"
    write_video: bool = True
    assume_sequence: bool = True
    panoptic: bool = False             # draw the panoptic map instead of the road tint
    label: str = ""


def _hmm() -> IntentHMM:
    o, s = simulate_agents(300, seed=0)
    return IntentHMM().fit_supervised(o, s)


class Pipeline:
    """Stateful per-source processor. `process(frame)` returns (annotated image, per-frame stats)."""

    def __init__(self, cfg: PipelineConfig, detector=None):
        self.cfg = cfg
        self.device = select_device(cfg.device)
        self.detector = detector or build_detector(cfg.detector, device=self.device)
        self.prof = Profiler()
        self.hmm = _hmm()
        self.n = 0
        self.rows: List[dict] = []
        self.class_counts: Counter = Counter()
        self.horizon_est = HorizonEstimator(cfg.horizon)
        self.cam: Optional[CameraModel] = None

    # ---- lazily built per-resolution state
    def _init_for(self, h: int, w: int) -> None:
        self.cam = CameraModel(w, h, self.cfg.hfov, self.cfg.cam_height, self.cfg.horizon)
        self.lanes = LaneDetector(self.cam, track=self.cfg.assume_sequence)
        self.road = RoadSegmenter()
        self.ego = EgoMotion()
        self.tracker = KalmanTracker(max_dist=0.08 * w, max_miss=8)
        self.intent = IntentTracker(self.hmm)
        self.hist: Dict[int, Tuple[float, float]] = {}
        self.fuser = PanopticFuser()
        self.poly = corridor(w, h, int(h * self.cam.horizon_frac))
        self.poly_wide = corridor(w, h, int(h * self.cam.horizon_frac), 1.5)

    def _halfwidth(self, y: float) -> float:
        h, w = self.cam.height, self.cam.width
        hy = self.cam.horizon_row
        return max(0.04 * w + (0.30 * w - 0.04 * w) * (y - hy) / max(h - hy, 1.0), 0.02 * w)

    def process(self, img: np.ndarray, idx: int, t: float, fps: float = 10.0, sequential: bool = True):
        h, w = img.shape[:2]
        if self.cam is None or (self.cam.width, self.cam.height) != (w, h):
            self._init_for(h, w)
        t0 = time.perf_counter()
        with self.prof.stage("ego"):
            motion = self.ego.update(img) if sequential else None
            if self.n < 20:
                self.horizon_est.update(img)
            if self.n == 20 and self.horizon_est.samples:
                self.cam.horizon_frac = float(np.clip(self.horizon_est.robust(), 0.3, 0.6))
                self.lanes = LaneDetector(self.cam, track=self.cfg.assume_sequence)
                self.poly = corridor(w, h, int(self.cam.horizon_row))
                self.poly_wide = corridor(w, h, int(self.cam.horizon_row), 1.5)
        with self.prof.stage("detect"):
            dets: List[Detection] = self.detector(img, motion)
            self.last_dets = dets
        with self.prof.stage("track"):
            if sequential:
                track_detections(self.tracker, dets, idx)
            for d in dets:
                d.dist = est_dist(d.bottom, self.cam.horizon_row, self.cam.f, self.cam.height_m)
        with self.prof.stage("road"):
            road = self.road(img)
        with self.prof.stage("lane"):
            lane = self.lanes.detect(img, road)
        with self.prof.stage("scene"):
            counts = Counter(d.cat for d in dets)
            self.class_counts.update(counts)
            scene = scene_descriptor(img, road, len(dets), dict(counts))
            intents: Dict[int, str] = {}
            for d in dets:
                if d.track_id >= 0 and d.cat in VULN_FOR_INTENT:
                    e = (d.cx - w / 2) / self._halfwidth(d.bottom)
                    prev = self.hist.get(d.track_id)
                    de = 0.0 if prev is None else (e - prev[0]) / max((t - prev[1]), 1e-3)
                    self.hist[d.track_id] = (e, t)
                    state, _ = self.intent.update(d.track_id, e, de, abs(de))
                    intents[d.track_id] = state
            risk, nearest = assess_risk(dets, self.poly, self.poly_wide, CONSERVATIVE)
        with self.prof.stage("render"):
            vis = self._render(img, road, dets, lane, intents, risk, scene, motion, (time.perf_counter() - t0) * 1000)
        latency = (time.perf_counter() - t0) * 1000
        self.prof.t["total"].append(latency / 1000)
        stats = {"frame": idx, "t_s": round(t, 3), "n_det": len(dets), "n_tracks": sum(1 for d in dets if d.track_id >= 0),
                 "road_cov": round(float(road.mean()), 3), "lane_state": lane.state, "lane_source": lane.source,
                 "lane_conf": round(lane.confidence, 3), "ego_offset_m": None if lane.ego_offset_m is None else round(lane.ego_offset_m, 2),
                 "risk": risk, "nearest_vulnerable_m": None if nearest is None else round(nearest, 1),
                 "lighting": scene["lighting"], "ego_motion_px": round(motion.magnitude, 2) if motion else None,
                 "latency_ms": round(latency, 1)}
        self.rows.append(stats)
        self.n += 1
        return vis, stats

    def _render(self, img, road, dets, lane: LaneResult, intents, risk, scene, motion, latency_ms) -> np.ndarray:
        h, w = img.shape[:2]
        vis = img.copy()
        if self.cfg.panoptic:
            vis = cv2.addWeighted(vis, 0.55, self.fuser.colorize(self.fuser.fuse(road, dets)), 0.45, 0)
        else:
            tint = np.zeros_like(vis)
            tint[road > 0] = (0, 150, 0)
            vis = cv2.addWeighted(vis, 1, tint, 0.28, 0)
        for pl in lane.polylines_image(self.cam):
            cv2.polylines(vis, [pl.astype(np.int32)], False, (0, 255, 255), 2, cv2.LINE_AA)
        if lane.centre is not None:
            zs = np.linspace(6, 40, 30)
            pts = self.cam.ground_to_image(np.polyval(lane.centre, zs), zs)
            ok = (pts[:, 0] >= 0) & (pts[:, 0] < w)
            if ok.sum() > 2:
                cv2.polylines(vis, [pts[ok].astype(np.int32)], False, (255, 255, 255) if lane.source == "markings" else (255, 200, 0), 2, cv2.LINE_AA)
        for d in dets:
            x1, y1, x2, y2 = d.box
            col = CATEGORY_COLORS.get(d.cat or "", (180, 180, 180))
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
            tag = f"{d.cat}" + (f"#{d.track_id}" if d.track_id >= 0 else "") + ("" if math.isinf(d.dist) else f" {d.dist:.0f}m")
            if d.track_id in intents and intents[d.track_id] != "passive":
                tag += f" {intents[d.track_id]}"
            cv2.putText(vis, tag, (x1, max(y1 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        hud = [f"lane:{lane.state}/{lane.source} conf {lane.confidence:.2f}", f"risk:{risk}  det:{len(dets)}  {scene['lighting']}",
               f"{latency_ms:.0f} ms ({1000 / max(latency_ms, 1):.0f} fps)  {self.device}"]
        for i, line in enumerate(hud):
            put_text(vis, line, (8, 20 + 18 * i), 0.5)
        return self._bev_inset(vis, dets, lane)

    def _bev_inset(self, vis, dets, lane: LaneResult) -> np.ndarray:
        h, w = vis.shape[:2]
        bw, bh = int(np.clip(0.19 * w, 24, 160)), int(np.clip(0.5 * h, 36, 240))
        if bh + 16 > h or bw + 16 > w:          # tiny frames: skip the inset instead of failing
            return vis
        inset = np.full((bh, bw, 3), 35, np.uint8)
        ppm = bh / 40.0
        for l in lane.lanes:
            zs = np.linspace(0, 40, 30)
            pts = np.c_[bw / 2 + l.x_at(zs) * ppm, bh - zs * ppm]
            cv2.polylines(inset, [pts.astype(np.int32)], False, (0, 255, 255), 1)
        for d in dets:
            if d.dist < 40:
                gx = (d.cx - self.cam.cx) * d.dist / self.cam.f
                cv2.circle(inset, (int(bw / 2 + gx * ppm), int(bh - d.dist * ppm)), 3, CATEGORY_COLORS.get(d.cat or "", (200, 200, 200)), -1)
        cv2.rectangle(inset, (bw // 2 - 3, bh - 8), (bw // 2 + 3, bh - 1), (255, 255, 255), -1)
        vis[8:8 + bh, w - bw - 8:w - 8] = inset
        return vis

    def summary(self, extra: Optional[dict] = None) -> dict:
        rows = self.rows
        lane = Counter(r["lane_state"] for r in rows)
        out = {"frames": len(rows), "device": self.device, "detector": type(self.detector).__name__,
               "class_counts_total": dict(self.class_counts),
               "lane_state_frac": {k: v / max(len(rows), 1) for k, v in lane.items()},
               "lane_source_frac": {k: v / max(len(rows), 1) for k, v in Counter(r["lane_source"] for r in rows).items()},
               "risk_frac": {k: v / max(len(rows), 1) for k, v in Counter(r["risk"] for r in rows).items()},
               "mean_detections": float(np.mean([r["n_det"] for r in rows])) if rows else 0.0,
               "road_cov_mean": float(np.mean([r["road_cov"] for r in rows])) if rows else 0.0,
               "latency_ms_mean": float(np.mean([r["latency_ms"] for r in rows])) if rows else 0.0,
               "latency_ms_p95": float(np.percentile([r["latency_ms"] for r in rows], 95)) if rows else 0.0,
               "fps": 1000.0 / max(float(np.mean([r["latency_ms"] for r in rows])), 1e-9) if rows else 0.0,
               "stages": {k: v for k, v in self.prof.summary().items() if not k.startswith("_")}}
        out.update(extra or {})
        return out


def detect_static_camera(src: FrameSource, n: int = 60) -> bool:
    """Pre-scan: True if the camera is fixed (negligible global motion), so the static-camera analyser applies."""
    if src.kind != "video":
        return False
    ego = EgoMotion()
    for k, f in enumerate(open_source(src.spec, stride=max(1, int(src.info.fps // 10)), max_frames=n, max_side=480)):
        ego.update(f.image)
    return len(ego.frac_hist) >= 10 and not ego.moving_camera()


def run(cfg: PipelineConfig, detector=None) -> dict:
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    src = open_source(cfg.source, stride=cfg.stride, start_s=cfg.start_s, duration_s=cfg.duration_s, max_frames=cfg.max_frames,
                      max_side=cfg.max_side, assume_sequence=cfg.assume_sequence, work_dir=str(out))
    mode = cfg.mode
    if mode == "auto":
        mode = "static" if detect_static_camera(src) else "vehicle"
    (out / "config.json").write_text(json.dumps({**asdict(cfg), "resolved_mode": mode}, indent=2))
    if mode == "static":
        from .static import StaticConfig, analyze
        s = analyze(StaticConfig(source=src.video or cfg.source, out=str(out), label=cfg.label or "auto: fixed camera",
                                 tracker="kalman", start_s=cfg.start_s, duration_s=cfg.duration_s, write_video=cfg.write_video))
        return {"mode": "static", **s}
    pipe = Pipeline(cfg, detector)
    writer = None
    fps_out = max(src.info.fps / max(cfg.stride, 1), 1.0)
    try:
        for frame in prefetch(src):
            vis, _ = pipe.process(frame.image, frame.index, frame.t_s, src.info.fps, sequential=src.info.sequential)
            if cfg.write_video:
                if writer is None:
                    writer = VideoWriter(str(out / "annotated.mp4"), fps_out, (vis.shape[1], vis.shape[0]))
                writer.write(vis)
    finally:
        if writer is not None:
            writer.close()
    if not pipe.rows:
        raise RuntimeError("No frames processed.")
    extra = {"mode": "vehicle", "source": cfg.source, "input_kind": src.kind, "resolution": [src.info.width, src.info.height]}
    if src.kind == "video":
        try:
            extra["quality"] = assess_quality(src.video or cfg.source, 20)
        except Exception as exc:  # quality profile is optional
            extra["quality_error"] = str(exc)
    summary = pipe.summary(extra)
    write_csv(out / "frames.csv", pipe.rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary


def parse_args(argv=None) -> PipelineConfig:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="video file/URL, image, folder, glob or webcam index")
    p.add_argument("--out", default="results/pipeline")
    p.add_argument("--mode", choices=["auto", "vehicle", "static"], default="auto")
    p.add_argument("--detector", default="auto", help="auto | motion | onnx:<path> | yolo:<weights>")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--start", type=float, default=0.0, dest="start_s")
    p.add_argument("--duration", type=float, default=0.0, dest="duration_s")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--max-side", type=int, default=960)
    p.add_argument("--hfov", type=float, default=70.0)
    p.add_argument("--cam-height", type=float, default=1.4)
    p.add_argument("--horizon", type=float, default=0.45)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda:0 | mps")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--no-sequence", action="store_true", help="treat images as independent (no tracking / ego-motion)")
    p.add_argument("--panoptic", action="store_true")
    a = p.parse_args(argv)
    return PipelineConfig(source=a.source, out=a.out, mode=a.mode, detector=a.detector, stride=a.stride, start_s=a.start_s,
                          duration_s=a.duration_s, max_frames=a.max_frames, max_side=a.max_side, hfov=a.hfov, cam_height=a.cam_height,
                          horizon=a.horizon, device=a.device, write_video=not a.no_video, assume_sequence=not a.no_sequence,
                          panoptic=a.panoptic)


def main(argv=None) -> None:
    s = run(parse_args(argv))
    print(json.dumps({k: v for k, v in s.items() if k not in ("quality", "stages", "class_counts_total")}, indent=2, default=float))


if __name__ == "__main__":
    main()
