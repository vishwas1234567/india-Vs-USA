"""Shared building blocks for the Indian-road perception toolkit.

Only OpenCV and NumPy are required here. Heavy deep-learning dependencies
(PyTorch, Transformers, Ultralytics) are imported lazily by the classes that
need them, so the classical pipeline and the test-suite run without them.
"""
from __future__ import annotations

import csv
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]  # x1, y1, x2, y2 (pixels)

# --------------------------------------------------------------------------- #
# Road-user categories
# --------------------------------------------------------------------------- #
VULNERABLE = frozenset({"two_wheeler", "pedestrian", "animal"})

CATEGORY_COLORS: Dict[str, Tuple[int, int, int]] = {  # BGR
    "two_wheeler": (0, 165, 255),
    "pedestrian": (0, 0, 255),
    "animal": (255, 0, 255),
    "auto": (0, 255, 255),
    "heavy": (255, 128, 0),
    "car": (255, 200, 0),
    "sign": (200, 200, 200),
}

_NAMES: Dict[str, Tuple[str, ...]] = {
    "two_wheeler": ("motorcycle", "motorbike", "bicycle", "rider", "motorcyclist", "scooter"),
    "pedestrian": ("person", "pedestrian"),
    "animal": ("cow", "dog", "horse", "sheep", "animal"),
    "auto": ("autorickshaw", "auto rickshaw", "auto", "rickshaw", "tuk tuk"),
    "heavy": ("bus", "truck", "tractor"),
    "car": ("car", "van", "jeep"),
    "sign": ("traffic light", "stop sign", "traffic sign"),
}
_LOOKUP: Dict[str, str] = {n: c for c, names in _NAMES.items() for n in names}


def category(name: str) -> Optional[str]:
    """Map a detector class name (COCO or IDD style) to a road-user category."""
    n = name.lower().replace("_", " ").replace("-", " ").strip()
    if n in _LOOKUP:
        return _LOOKUP[n]
    if any(k in n for k in ("traffic", "sign", "light")):
        return "sign"
    return None


@dataclass
class Detection:
    """One detected (and optionally tracked) object in a frame."""

    box: Box
    name: str
    conf: float = 1.0
    track_id: int = -1
    cat: Optional[str] = None
    dist: float = math.inf
    erratic: bool = False

    def __post_init__(self) -> None:
        if self.cat is None:
            self.cat = category(self.name)

    @property
    def cx(self) -> float:
        return (self.box[0] + self.box[2]) / 2.0

    @property
    def width(self) -> int:
        return max(self.box[2] - self.box[0], 1)

    @property
    def bottom(self) -> int:
        return self.box[3]


# --------------------------------------------------------------------------- #
# Video helpers
# --------------------------------------------------------------------------- #
@dataclass
class VideoInfo:
    fps: float
    width: int
    height: int
    frames: int


def open_video(path: str) -> Tuple[cv2.VideoCapture, VideoInfo]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    info = VideoInfo(
        fps=float(cap.get(cv2.CAP_PROP_FPS) or 30.0),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    if info.frames <= 0:
        raise ValueError("Video frame count unknown; re-encode the file to a seekable container (mp4).")
    return cap, info


def frame_range(info: VideoInfo, start_s: float = 0.0, duration_s: float = 0.0) -> Tuple[int, int]:
    """Return [start, end) frame indices for an optional time window (duration 0 = to the end)."""
    start = max(0, int(round(start_s * info.fps)))
    end = info.frames if duration_s <= 0 else min(info.frames, start + int(round(duration_s * info.fps)))
    if start >= end:
        raise ValueError("Requested time window is empty or beyond the end of the video.")
    return start, end


def iter_frames(cap: cv2.VideoCapture, start: int, end: int) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield (absolute_frame_index, frame) for frames in [start, end)."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)  # always seek: callers may have moved the read position
    for idx in range(start, end):
        ok, frame = cap.read()
        if not ok:
            break
        yield idx, frame


def resolve_source(src: str, out_dir: Path) -> str:
    """Return a local video path; download with yt-dlp if `src` is a URL."""
    if not src.lower().startswith("http"):
        return src
    try:
        import yt_dlp  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Install yt-dlp (pip install yt-dlp) or pass a local video file.") from exc
    opts = {
        "format": "bv*[height<=720]/b[height<=720]/b",
        "outtmpl": str(Path(out_dir) / "source.%(ext)s"),
        "quiet": True,
        "noprogress": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(src, download=True)
        return ydl.prepare_filename(info)


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        Path(path).write_text("")
        return
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------- #
# Drivable-area segmentation
# --------------------------------------------------------------------------- #
class SegformerRoad:
    """SegFormer drivable-area mask. Cityscapes weights are only a baseline for
    Indian roads; pass an IDD-fine-tuned checkpoint and its drivable class ids."""

    def __init__(self, model_id: str, device: str, road_ids: Sequence[int]):
        import torch  # type: ignore
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor  # type: ignore

        self.torch = torch
        self.device = device
        self.road_ids = list(road_ids)
        self.proc = SegformerImageProcessor.from_pretrained(model_id)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_id).to(device).eval()

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        torch = self.torch
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inputs = self.proc(images=rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            logits = torch.nn.functional.interpolate(
                logits, size=bgr.shape[:2], mode="bilinear", align_corners=False)
            pred = logits.argmax(1)[0].cpu().numpy()
        return np.isin(pred, self.road_ids).astype(np.uint8)


class ClassicalRoad:
    """Model-free fallback: learn the road colour (CIELAB) from a patch just in
    front of the vehicle and keep the connected region that matches it."""

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        small = cv2.resize(bgr, (max(w // 2, 8), max(h // 2, 8)))
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
        sh, sw = lab.shape[:2]
        patch = lab[int(sh * .82):int(sh * .97), int(sw * .35):int(sw * .65)].reshape(-1, 3)
        mu = patch.mean(0)
        inv = np.linalg.inv(np.cov(patch.T) + np.eye(3) * 4)
        d = lab.reshape(-1, 3) - mu
        m = np.einsum("ij,jk,ik->i", d, inv, d).reshape(sh, sw)
        mask = (m < 12).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        _, comp = cv2.connectedComponents(mask)
        seed = comp[int(sh * .9), sw // 2]
        out = (comp == seed).astype(np.uint8) if seed > 0 else mask
        return cv2.resize(out, (w, h), interpolation=cv2.INTER_NEAREST)


# --------------------------------------------------------------------------- #
# Lane-marking quality and potholes
# --------------------------------------------------------------------------- #
def lane_marking_state(bgr: np.ndarray, road_mask: np.ndarray) -> Tuple[str, float, int]:
    """Return (state, score in [0,1], n_line_segments) for the lower half of the road.

    score = 0.5*min(1, 40*paint_fraction) + 0.5*min(1, n_segments/6);
    state: clear (>0.6), faded (>0.25), absent otherwise, unknown if no road visible.
    """
    h, w = road_mask.shape
    roi = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    roi[: int(h * 0.5)] = 0
    area = int(roi.sum())
    if area < 0.02 * h * w:
        return "unknown", 0.0, 0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 190), (180, 50, 255))
    yellow = cv2.inRange(hsv, (15, 90, 120), (35, 255, 255))
    paint = cv2.bitwise_and(cv2.bitwise_or(white, yellow), roi * 255)
    paint = cv2.morphologyEx(paint, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    ratio = float(paint.sum() / 255) / area
    lines = cv2.HoughLinesP(cv2.Canny(paint, 50, 150), 1, np.pi / 180, 30,
                            minLineLength=int(h * 0.06), maxLineGap=int(h * 0.05))
    n = 0
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            ang = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
            if 20 < ang < 160:
                n += 1
    score = 0.5 * min(1.0, ratio * 40) + 0.5 * min(1.0, n / 6)
    state = "clear" if score > 0.6 else "faded" if score > 0.25 else "absent"
    return state, round(score, 3), n


def pothole_candidates(bgr: np.ndarray, road_mask: np.ndarray, max_n: int = 6) -> List[Tuple[int, int, int, int]]:
    """Heuristic pothole candidates (x, y, w, h): compact, locally dark blobs in the near field.
    Expect false positives from shadows and oil stains; prefer a trained model."""
    h, w = road_mask.shape
    roi = np.zeros_like(road_mask)
    roi[int(h * 0.55):] = road_mask[int(h * 0.55):]
    roi = cv2.erode(roi, np.ones((15, 15), np.uint8))
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (7, 7), 0)
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(w // 20, 5))
    diff = bg.astype(np.int16) - gray.astype(np.int16)
    cand = (((diff > 25) & (roi > 0)).astype(np.uint8)) * 255
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(cand, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: List[Tuple[int, int, int, int]] = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < 0.0004 * h * w or a > 0.03 * h * w:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        solidity = a / max(cv2.contourArea(cv2.convexHull(c)), 1)
        if solidity < 0.7 or not (0.4 < bw / max(bh, 1) < 4.0):
            continue
        out.append((x, y, bw, bh))
    return sorted(out, key=lambda b: -b[2] * b[3])[:max_n]


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def corridor(w: int, h: int, horizon_y: int, scale: float = 1.0) -> np.ndarray:
    """Trapezoidal ego corridor in front of the vehicle (image coordinates)."""
    cx, b, t = w / 2, 0.30 * w * scale, 0.04 * w * scale
    return np.array([[cx - b, h], [cx + b, h], [cx + t, horizon_y], [cx - t, horizon_y]], np.int32)


def in_poly(poly: np.ndarray, x: float, y: float) -> bool:
    return cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0


def est_dist(y_bottom: float, horizon_y: float, fy: float, cam_h: float) -> float:
    """Flat-ground monocular range: d = f*H / (y_bottom - y_horizon)."""
    dy = y_bottom - horizon_y
    return fy * cam_h / dy if dy > 5 else math.inf


# --------------------------------------------------------------------------- #
# Erratic-motion analysis
# --------------------------------------------------------------------------- #
class MotionAnalyzer:
    """Flags weaving road users from the lateral motion of tracked boxes.

    Lateral step per frame is normalised by box width. For a moving camera the
    median step over all tracks is subtracted (cheap ego-motion compensation).
    """

    def __init__(self, ego_compensation: bool = True, window: int = 12):
        self.ego = ego_compensation
        self.last: Dict[int, float] = {}
        self.hist: Dict[int, deque] = defaultdict(lambda: deque(maxlen=window))

    def update(self, dets: Sequence[Detection]) -> None:
        raw: Dict[int, float] = {}
        for d in dets:
            if d.track_id >= 0 and d.track_id in self.last:
                raw[d.track_id] = (d.cx - self.last[d.track_id]) / d.width
        med = float(np.median(list(raw.values()))) if (self.ego and len(raw) >= 3) else 0.0
        for d in dets:
            d.erratic = False
            tid = d.track_id
            if tid < 0:
                continue
            if tid in raw:
                self.hist[tid].append(raw[tid] - med)
            self.last[tid] = d.cx
            h = np.array(self.hist[tid])
            if len(h) >= 8:
                s = np.sign(h[np.abs(h) > 0.03])
                changes = int((s[1:] != s[:-1]).sum()) if len(s) > 1 else 0
                d.erratic = bool((changes >= 3 and np.abs(h).mean() > 0.05) or np.abs(h).max() > 0.6)


def put_text(img: np.ndarray, text: str, org: Tuple[int, int], scale: float = 0.5, color=(255, 255, 255)) -> None:
    """Readable overlay text: a 1 px black outline made from offset copies (aligned with the text, unlike a thick stroke)."""
    for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1), (0, 2), (2, 0)):
        cv2.putText(img, text, (org[0] + dx, org[1] + dy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
