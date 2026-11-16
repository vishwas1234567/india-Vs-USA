"""Vehicle-mounted perception pipeline for Indian roads.

Detection + tracking (YOLOv8/YOLOv10 + ByteTrack), drivable-area segmentation
(SegFormer, classical fallback), lane-marking quality, pothole candidates,
erratic-motion score, monocular distance and collision-risk level.

The detector, segmenter and pothole detector are injectable callables, so the
whole pipeline can be exercised without deep-learning dependencies (see tests).

Usage:
    python -m irp.vehicle --source drive.mp4 --out results/drive1
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

from .common import (CATEGORY_COLORS, VULNERABLE, ClassicalRoad, Detection, MotionAnalyzer, SegformerRoad,
                     corridor, est_dist, frame_range, in_poly, iter_frames, lane_marking_state, open_video,
                     pothole_candidates, resolve_source, write_csv)

log = logging.getLogger("irp.vehicle")

Detector = Callable[[np.ndarray], List[Detection]]
Segmenter = Callable[[np.ndarray], np.ndarray]
PotholeDetector = Callable[[np.ndarray], List[Tuple[int, int, int, int]]]


@dataclass
class VehicleConfig:
    source: str
    out: str = "results/vehicle"
    label: str = ""
    stride: int = 2
    max_frames: int = 0
    start_s: float = 0.0
    duration_s: float = 0.0
    det_model: str = "yolov8n.pt"
    pothole_model: str = ""
    seg_backend: str = "segformer"          # segformer | classical
    seg_model: str = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
    road_ids: Tuple[int, ...] = (0,)
    device: Optional[str] = None
    conf: float = 0.3
    imgsz: int = 640
    cam_height: float = 1.4                 # camera height above road (m)
    hfov: float = 70.0                      # horizontal field of view (deg)
    horizon: float = 0.45                   # horizon row as fraction of image height
    write_video: bool = True


# --------------------------------------------------------------------------- #
# Deep-learning adapters (lazy imports)
# --------------------------------------------------------------------------- #
class YoloTracker:
    """Ultralytics YOLO + ByteTrack as a `Detector`."""

    def __init__(self, weights: str, conf: float, imgsz: int, device: Optional[str]):
        from ultralytics import YOLO  # type: ignore
        self.model, self.conf, self.imgsz, self.device = YOLO(weights), conf, imgsz, device

    def __call__(self, frame: np.ndarray) -> List[Detection]:
        res = self.model.track(frame, persist=True, conf=self.conf, imgsz=self.imgsz, verbose=False,
                               device=self.device, tracker="bytetrack.yaml")[0]
        dets: List[Detection] = []
        if res.boxes is None or len(res.boxes) == 0:
            return dets
        xyxy = res.boxes.xyxy.cpu().numpy().astype(int)
        cls = res.boxes.cls.cpu().numpy().astype(int)
        conf = res.boxes.conf.cpu().numpy()
        ids = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else [-1] * len(cls)
        for (x1, y1, x2, y2), c, p, tid in zip(xyxy, cls, conf, ids):
            d = Detection((int(x1), int(y1), int(x2), int(y2)), res.names[int(c)], float(p), int(tid))
            if d.cat is not None:
                dets.append(d)
        return dets


class YoloPotholes:
    """Ultralytics YOLO weights trained on potholes as a `PotholeDetector`."""

    def __init__(self, weights: str, conf: float, device: Optional[str]):
        from ultralytics import YOLO  # type: ignore
        self.model, self.conf, self.device = YOLO(weights), conf, device

    def __call__(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        res = self.model.predict(frame, conf=self.conf, verbose=False, device=self.device)[0]
        return [(int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1]))
                for b in res.boxes.xyxy.cpu().numpy()]


# --------------------------------------------------------------------------- #
# Risk model
# --------------------------------------------------------------------------- #
def assess_risk(dets: Sequence[Detection], poly: np.ndarray, poly_wide: np.ndarray,
                vulnerable=VULNERABLE) -> Tuple[str, Optional[float]]:
    """Return (risk level, nearest vulnerable-user distance in the ego corridor).

    HIGH: vulnerable user in corridor closer than 8 m, or erratic user in the wide corridor closer than 12 m.
    MED : vulnerable user in corridor closer than 20 m.
    """
    risk, min_d = "NONE", None
    for d in dets:
        if d.cat is None or d.cat == "sign":
            continue
        is_vuln = d.cat in vulnerable
        if not (is_vuln or d.erratic):
            continue
        inside, wide = in_poly(poly, d.cx, d.bottom), in_poly(poly_wide, d.cx, d.bottom)
        if is_vuln and inside:
            min_d = d.dist if min_d is None else min(min_d, d.dist)
        if (is_vuln and inside and d.dist < 8) or (d.erratic and wide and d.dist < 12):
            risk = "HIGH"
        elif risk != "HIGH" and is_vuln and inside and d.dist < 20:
            risk = "MED"
    if min_d is not None and math.isinf(min_d):
        min_d = None
    return risk, min_d


# --------------------------------------------------------------------------- #
# Drawing and reporting
# --------------------------------------------------------------------------- #
def draw(frame: np.ndarray, road: np.ndarray, dets: Sequence[Detection], potholes: Sequence[Tuple[int, int, int, int]],
         poly: np.ndarray, hud: Sequence[str]) -> np.ndarray:
    out = frame.copy()
    tint = np.zeros_like(out)
    tint[road > 0] = (0, 160, 0)
    out = cv2.addWeighted(out, 1.0, tint, 0.35, 0)
    cv2.polylines(out, [poly], True, (255, 255, 255), 1)
    for d in dets:
        x1, y1, x2, y2 = d.box
        col = CATEGORY_COLORS.get(d.cat or "", (200, 200, 200))
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 3 if d.erratic else 2)
        dist = "" if math.isinf(d.dist) else f" {d.dist:.0f}m"
        tag = f"{d.name}#{d.track_id}{dist}" + (" ERRATIC" if d.erratic else "")
        cv2.putText(out, tag, (x1, max(y1 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    for (x, y, bw, bh) in potholes:
        cv2.ellipse(out, (x + bw // 2, y + bh // 2), (max(bw // 2, 1), max(bh // 2, 1)), 0, 0, 360, (0, 0, 255), 2)
        cv2.putText(out, "pothole?", (x, max(y - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
    for i, line in enumerate(hud):
        cv2.putText(out, line, (10, 22 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, line, (10, 22 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def build_report(s: dict, source: str) -> str:
    lf = s["lane_state_frac"]
    lines = [f"# Indian Road Perception Report\n\nSource: `{source}`\n", "## Summary\n",
             f"- Frames analysed: {s['frames']} ({s['duration_s']:.1f} s, stride {s['stride']})",
             f"- Mean drivable-area coverage: {s['road_cov_mean'] * 100:.1f}% of the image",
             "- Lane marking state: " + ", ".join(f"{k} {v * 100:.0f}%" for k, v in lf.items()),
             f"- Frames with pothole candidates: {s['pothole_frame_frac'] * 100:.0f}% "
             f"(mean {s['pothole_mean']:.2f} per frame)",
             f"- Traffic heterogeneity (Shannon entropy of class mix, bits): {s['heterogeneity']:.2f}",
             f"- Risk frames: HIGH {s['risk_frac']['HIGH'] * 100:.0f}%, MED {s['risk_frac']['MED'] * 100:.0f}%"]
    if s["min_vuln_dist_m"] is not None:
        lines.append(f"- Closest vulnerable road user in ego corridor: ~{s['min_vuln_dist_m']:.1f} m (approximate)")
    lines += ["\n## Road users (unique tracks)\n", "| Class | Tracks | Erratic |", "|---|---|---|"]
    for k, v in sorted(s["unique_tracks"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {k} | {v} | {s['erratic_tracks'].get(k, 0)} |")
    findings = []
    if lf.get("absent", 0) + lf.get("faded", 0) > 0.5:
        findings.append("Lane markings are faded or missing in most frames: rely on drivable-area segmentation.")
    if s["unique_tracks"].get("two_wheeler", 0) > s["unique_tracks"].get("car", 0):
        findings.append("Two-wheelers outnumber cars: expect lane-splitting; use lateral-motion prediction.")
    if sum(s["erratic_tracks"].values()) > 0:
        findings.append("Erratic lateral motion detected: widen safety margins around two-wheelers and pedestrians.")
    if s["pothole_frame_frac"] > 0.15:
        findings.append("Frequent pothole candidates: confirm with a trained model before use in planning.")
    if s["risk_frac"]["HIGH"] > 0.05:
        findings.append("Sustained high-risk intervals: vulnerable road users close inside the ego corridor.")
    lines += ["\n## Findings\n"] + [f"- {f}" for f in (findings or ["No dominant failure pattern."])]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Main run
# --------------------------------------------------------------------------- #
def run(cfg: VehicleConfig, detector: Detector, segmenter: Segmenter,
        pothole_detector: Optional[PotholeDetector] = None) -> dict:
    """Process a video with the supplied models; write outputs and return the summary dict."""
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    path = resolve_source(cfg.source, out)
    cap, info = open_video(path)
    start, end = frame_range(info, cfg.start_s, cfg.duration_s)
    W, H, fps = info.width, info.height, info.fps
    horizon_y = int(H * cfg.horizon)
    fy = (W / 2) / math.tan(math.radians(cfg.hfov) / 2)
    poly, poly_wide = corridor(W, H, horizon_y), corridor(W, H, horizon_y, 1.5)
    writer = None
    if cfg.write_video:
        writer = cv2.VideoWriter(str(out / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                 max(fps / cfg.stride, 1.0), (W, H))
    motion = MotionAnalyzer(ego_compensation=True)
    rows: List[dict] = []
    lane_c: Counter = Counter()
    risk_c: Counter = Counter()
    cls_total: Counter = Counter()
    tracks: Dict[str, Set[int]] = defaultdict(set)
    erratic: Dict[str, Set[int]] = defaultdict(set)
    road_cov: List[float] = []
    pot_counts: List[int] = []
    min_vuln: Optional[float] = None
    n_done = 0
    t0 = time.perf_counter()

    for idx, frame in iter_frames(cap, start, end):
        if (idx - start) % cfg.stride:
            continue
        n_done += 1
        if cfg.max_frames and n_done > cfg.max_frames:
            break
        road = segmenter(frame)
        lane, lscore, _ = lane_marking_state(frame, road)
        pots = list(pothole_candidates(frame, road))
        if pothole_detector is not None:
            pots += list(pothole_detector(frame))

        dets = detector(frame)
        for d in dets:
            d.dist = est_dist(d.bottom, horizon_y, fy, cfg.cam_height)
        motion.update(dets)
        risk, frame_min = assess_risk(dets, poly, poly_wide)
        if frame_min is not None:
            min_vuln = frame_min if min_vuln is None else min(min_vuln, frame_min)

        counts: Counter = Counter()
        for d in dets:
            counts[d.cat] += 1
            cls_total[d.cat] += 1
            if d.track_id >= 0 and d.cat != "sign":
                tracks[d.cat].add(d.track_id)
                if d.erratic:
                    erratic[d.cat].add(d.track_id)
        cov = float(road.mean())
        road_cov.append(cov)
        lane_c[lane] += 1
        risk_c[risk] += 1
        pot_counts.append(len(pots))
        rows.append({
            "frame": idx, "t_s": round(idx / fps, 2), "road_cov": round(cov, 3), "lane_state": lane,
            "lane_score": lscore, "pothole_candidates": len(pots),
            "two_wheeler": counts["two_wheeler"], "pedestrian": counts["pedestrian"], "animal": counts["animal"],
            "auto": counts["auto"], "heavy": counts["heavy"], "car": counts["car"],
            "erratic": sum(d.erratic for d in dets), "risk": risk,
            "min_vuln_dist_m": "" if frame_min is None else round(frame_min, 1)})
        if writer is not None:
            hud = [f"Road {cov * 100:.0f}% | Lane: {lane} ({lscore})", f"Potholes?: {len(pots)} | Risk: {risk}"]
            writer.write(draw(frame, road, dets, pots, poly, hud))
    cap.release()
    if writer is not None:
        writer.release()
    if not rows:
        raise RuntimeError("No frames processed.")

    nf = len(rows)
    tot = sum(cls_total.values()) or 1
    entropy = -sum((v / tot) * math.log2(v / tot) for v in cls_total.values() if v)
    summary = {
        "frames": nf, "stride": cfg.stride, "duration_s": nf * cfg.stride / fps,
        "road_cov_mean": float(np.mean(road_cov)),
        "lane_state_frac": {k: v / nf for k, v in lane_c.items()},
        "pothole_frame_frac": sum(1 for c in pot_counts if c) / nf,
        "pothole_mean": float(np.mean(pot_counts)),
        "heterogeneity": entropy,
        "unique_tracks": {k: len(v) for k, v in tracks.items()},
        "erratic_tracks": {k: len(v) for k, v in erratic.items()},
        "risk_frac": {k: risk_c.get(k, 0) / nf for k in ("HIGH", "MED", "NONE")},
        "min_vuln_dist_m": None if min_vuln is None else float(min_vuln),
        "wall_time_s": round(time.perf_counter() - t0, 1),
    }
    write_csv(out / "frames.csv", rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "report.md").write_text(build_report(summary, cfg.source))
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> VehicleConfig:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="video path or URL (URLs need yt-dlp)")
    p.add_argument("--out", default="results/vehicle")
    p.add_argument("--label", default="")
    p.add_argument("--stride", type=int, default=2, help="process every Nth frame")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--start", type=float, default=0.0, dest="start_s")
    p.add_argument("--duration", type=float, default=0.0, dest="duration_s")
    p.add_argument("--det-model", default="yolov8n.pt")
    p.add_argument("--pothole-model", default="", help="optional YOLO weights trained on potholes")
    p.add_argument("--seg-backend", choices=["segformer", "classical"], default="segformer")
    p.add_argument("--seg-model", default="nvidia/segformer-b0-finetuned-cityscapes-1024-1024")
    p.add_argument("--road-ids", type=int, nargs="+", default=[0], help="class ids treated as drivable")
    p.add_argument("--device", default=None)
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--cam-height", type=float, default=1.4)
    p.add_argument("--hfov", type=float, default=70.0)
    p.add_argument("--horizon", type=float, default=0.45)
    p.add_argument("--no-video", action="store_true")
    a = p.parse_args(argv)
    return VehicleConfig(
        source=a.source, out=a.out, label=a.label, stride=a.stride, max_frames=a.max_frames, start_s=a.start_s,
        duration_s=a.duration_s, det_model=a.det_model, pothole_model=a.pothole_model, seg_backend=a.seg_backend,
        seg_model=a.seg_model, road_ids=tuple(a.road_ids), device=a.device, conf=a.conf, imgsz=a.imgsz,
        cam_height=a.cam_height, hfov=a.hfov, horizon=a.horizon, write_video=not a.no_video)


def build_models(cfg: VehicleConfig) -> Tuple[Detector, Segmenter, Optional[PotholeDetector]]:
    detector = YoloTracker(cfg.det_model, cfg.conf, cfg.imgsz, cfg.device)
    segmenter: Segmenter
    if cfg.seg_backend == "segformer":
        try:
            import torch  # type: ignore
            device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
            segmenter = SegformerRoad(cfg.seg_model, device, cfg.road_ids)
        except Exception as exc:  # offline weights, missing dependency
            log.warning("SegFormer unavailable (%s); using classical road segmentation.", exc)
            segmenter = ClassicalRoad()
    else:
        segmenter = ClassicalRoad()
    pothole = YoloPotholes(cfg.pothole_model, cfg.conf, cfg.device) if cfg.pothole_model else None
    return detector, segmenter, pothole


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = parse_args(argv)
    detector, segmenter, pothole = build_models(cfg)
    summary = run(cfg, detector, segmenter, pothole)
    print(json.dumps(summary, indent=2))
    print(f"Outputs in {cfg.out}/")


if __name__ == "__main__":
    main()
