"""Object detectors behind one interface: detector(frame, motion=None) -> List[Detection].

  MotionDetector   no model needed: ego-motion-compensated frame differencing for a moving camera (CPU, fast)
  OnnxYoloDetector YOLOv8-style ONNX model through OpenCV DNN (CPU, no PyTorch)
  YoloDetector     Ultralytics YOLOv8/v10 weights (CPU or GPU, optional FP16)
  build_detector   "motion" | "onnx:<path>" | "yolo:<weights>" | "auto"

Detections use the toolkit categories (car, heavy, two_wheeler, pedestrian, auto, animal, sign); moving blobs without a
class get category "unknown" and are treated conservatively by the risk logic.
"""
from __future__ import annotations

import math
from collections import deque
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .camera import Motion
from .common import Detection, category

COCO80 = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
          "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
          "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
          "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
          "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
          "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
          "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
          "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ix, iy = max(0.0, min(a[2], b[2]) - max(a[0], b[0])), max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class MotionDetector:
    """Independent-motion blobs after compensating camera motion.

    The current frame is compared with the frame `lag` steps earlier, warped by the composed camera motion, so slow
    objects (a few pixels per frame) still leave a detectable difference. Blobs must persist across two frames."""

    def __init__(self, scale_width: int = 320, diff_thr: float = 22.0, min_area_frac: float = 0.0006,
                 horizon_frac: float = 0.42, persist_iou: float = 0.2, lag: int = 4):
        self.sw, self.thr, self.min_area, self.horizon, self.persist, self.lag = scale_width, diff_thr, min_area_frac, horizon_frac, persist_iou, lag
        self.hist: deque = deque(maxlen=lag + 1)     # (gray, 3x3 transform previous->this frame)
        self.prev_boxes: List[Tuple[float, float, float, float]] = []

    @staticmethod
    def _matrix(motion: Optional[Motion], s: float) -> np.ndarray:
        if motion is None:
            return np.eye(3)
        if motion.H is not None:
            T = np.diag([s / motion.s, s / motion.s, 1.0])
            return T @ motion.H @ np.linalg.inv(T)
        c, sn = motion.scale * math.cos(motion.rot), motion.scale * math.sin(motion.rot)
        return np.array([[c, -sn, motion.dx * s], [sn, c, motion.dy * s], [0, 0, 1.0]])

    def __call__(self, frame: np.ndarray, motion: Optional[Motion] = None) -> List[Detection]:
        h, w = frame.shape[:2]
        s = self.sw / w
        g = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(frame, (self.sw, int(h * s))), cv2.COLOR_BGR2GRAY), (5, 5), 0)
        self.hist.append((g, self._matrix(motion, s)))
        dets: List[Detection] = []
        if len(self.hist) > self.lag:
            ref = self.hist[0][0]
            total = np.eye(3)
            for _, m in list(self.hist)[1:]:
                total = m @ total
            gh, gw = g.shape
            warped = cv2.warpPerspective(ref, total, (gw, gh), borderMode=cv2.BORDER_REPLICATE)
            mask = (cv2.absdiff(g, warped) > self.thr).astype(np.uint8)
            mask[: int(gh * self.horizon)] = 0
            mask[:3], mask[-3:], mask[:, :3], mask[:, -3:] = 0, 0, 0, 0
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boxes = []
            for c in cnts:
                x, y, bw, bh = cv2.boundingRect(c)
                if bw * bh >= self.min_area * gh * gw and bw >= 4 and bh >= 6:
                    boxes.append((x, y, x + bw, y + bh))
            for b in boxes:
                if any(iou_xyxy(b, p) > self.persist for p in self.prev_boxes):
                    conf = float(min(1.0, 0.5 + (b[2] - b[0]) * (b[3] - b[1]) / (0.02 * gh * gw)))
                    dets.append(Detection((int(b[0] / s), int(b[1] / s), int(b[2] / s), int(b[3] / s)), "moving_object", conf,
                                          cat="unknown"))
            self.prev_boxes = boxes
        return dets


def letterbox_params(shape: Tuple[int, int], size: int) -> Tuple[float, int, int]:
    h, w = shape
    r = min(size / w, size / h)
    return r, int((size - round(w * r)) // 2), int((size - round(h * r)) // 2)


def decode_yolov8(out: np.ndarray, names: Sequence[str], conf: float, iou: float, ratio: float, dx: int, dy: int,
                  orig_shape: Tuple[int, int]) -> List[Detection]:
    """Decode a YOLOv8 ONNX output (1, 4+nc, N) or (1, N, 4+nc) into Detections in original image coordinates."""
    o = np.squeeze(out, 0)
    if o.shape[0] < o.shape[1]:
        o = o.T                                          # (N, 4+nc)
    boxes, scores = o[:, :4], o[:, 4:]
    cls = scores.argmax(1)
    conf_all = scores[np.arange(len(cls)), cls]
    keep = conf_all >= conf
    boxes, cls, conf_all = boxes[keep], cls[keep], conf_all[keep]
    if len(boxes) == 0:
        return []
    h, w = orig_shape
    x1 = (boxes[:, 0] - boxes[:, 2] / 2 - dx) / ratio
    y1 = (boxes[:, 1] - boxes[:, 3] / 2 - dy) / ratio
    x2 = (boxes[:, 0] + boxes[:, 2] / 2 - dx) / ratio
    y2 = (boxes[:, 1] + boxes[:, 3] / 2 - dy) / ratio
    xyxy = np.stack([x1, y1, x2, y2], 1).clip([0, 0, 0, 0], [w, h, w, h])
    off = cls[:, None] * 10000.0                          # class-aware NMS
    idx = cv2.dnn.NMSBoxes((xyxy + off).tolist(), conf_all.tolist(), conf, iou)
    dets = []
    for i in np.array(idx).reshape(-1):
        name = names[int(cls[i])] if int(cls[i]) < len(names) else str(int(cls[i]))
        d = Detection(tuple(int(v) for v in xyxy[i]), name, float(conf_all[i]))
        if d.cat is not None:
            dets.append(d)
    return dets


class OnnxYoloDetector:
    def __init__(self, path: str, names: Sequence[str] = COCO80, imgsz: int = 640, conf: float = 0.25, iou: float = 0.45):
        self.net = cv2.dnn.readNetFromONNX(path)
        self.names, self.size, self.conf, self.iou = list(names), imgsz, conf, iou

    def __call__(self, frame: np.ndarray, motion: Optional[Motion] = None) -> List[Detection]:
        r, dx, dy = letterbox_params(frame.shape[:2], self.size)
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        nw, nh = int(round(frame.shape[1] * r)), int(round(frame.shape[0] * r))
        canvas[dy:dy + nh, dx:dx + nw] = cv2.resize(frame, (nw, nh))
        self.net.setInput(cv2.dnn.blobFromImage(canvas, 1 / 255.0, (self.size, self.size), swapRB=True))
        return decode_yolov8(self.net.forward(), self.names, self.conf, self.iou, r, dx, dy, frame.shape[:2])


class YoloDetector:
    """Ultralytics detector (YOLOv8/v10, fine-tuned IDD weights, ...). device: 'cpu' | 'cuda:0'; half=True on GPU."""

    def __init__(self, weights: str, conf: float = 0.25, imgsz: int = 640, device: Optional[str] = None, half: bool = False):
        from ultralytics import YOLO  # type: ignore
        self.model, self.conf, self.imgsz, self.device = YOLO(weights), conf, imgsz, device
        self.half = half and (device or "").startswith("cuda")

    def __call__(self, frame: np.ndarray, motion: Optional[Motion] = None) -> List[Detection]:
        res = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, verbose=False, device=self.device, half=self.half)[0]
        dets = []
        if res.boxes is None or len(res.boxes) == 0:
            return dets
        for (x1, y1, x2, y2), c, p in zip(res.boxes.xyxy.cpu().numpy().astype(int), res.boxes.cls.cpu().numpy().astype(int),
                                          res.boxes.conf.cpu().numpy()):
            d = Detection((int(x1), int(y1), int(x2), int(y2)), res.names[int(c)], float(p))
            if d.cat is not None:
                dets.append(d)
        return dets


def build_detector(spec: str = "auto", device: Optional[str] = None, conf: float = 0.25):
    """'motion' | 'onnx:<path>' | 'yolo:<weights>' | 'auto' (yolo if ultralytics + weights exist, else motion)."""
    kind, _, arg = spec.partition(":")
    if kind == "motion":
        return MotionDetector()
    if kind == "onnx":
        return OnnxYoloDetector(arg, conf=conf)
    if kind == "yolo":
        return YoloDetector(arg or "yolov8n.pt", conf=conf, device=device, half=True)
    if kind == "auto":
        try:
            import ultralytics  # noqa: F401  type: ignore
            return YoloDetector("yolov8n.pt", conf=conf, device=device, half=True)
        except Exception:
            return MotionDetector()
    raise ValueError(f"Unknown detector spec: {spec}")
