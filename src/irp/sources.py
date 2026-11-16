"""Open any input: video file, URL, image, folder of images, glob pattern, or webcam index.

    for frame in open_source("drive.mp4", stride=2, max_side=960): ...
    for frame in open_source("frames/*.jpg"): ...

Frames carry the index, timestamp and name. Large inputs are down-scaled (max_side), unreadable frames are skipped,
image folders are natural-sorted and resized to the first image's size, and EXIF/rotation metadata is honoured by OpenCV.
"""
from __future__ import annotations

import glob
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

import cv2
import numpy as np

from .common import resolve_source

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg")


@dataclass
class Frame:
    index: int
    t_s: float
    image: np.ndarray
    name: str = ""


@dataclass
class SourceInfo:
    kind: str            # video | images | camera
    fps: float
    width: int
    height: int
    n_frames: int        # 0 if unknown (camera)
    sequential: bool     # consecutive frames of one drive (tracking/ego-motion make sense)


def _natural(p) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(p))]


def _scale(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = max_side / max(h, w) if max_side and max(h, w) > max_side else 1.0
    return cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else img


class FrameSource:
    def __init__(self, spec: str, stride: int = 1, start_s: float = 0.0, duration_s: float = 0.0, max_frames: int = 0,
                 max_side: int = 0, assume_sequence: bool = True, fps: float = 10.0, work_dir: str = "."):
        self.spec, self.stride, self.start_s, self.duration_s = str(spec), max(1, stride), start_s, duration_s
        self.max_frames, self.max_side, self.assume_sequence, self.default_fps = max_frames, max_side, assume_sequence, fps
        self.files: List[str] = []
        self.cam: Optional[int] = None
        self.video: Optional[str] = None
        if self.spec.isdigit():
            self.cam, self.kind = int(self.spec), "camera"
        elif self.spec.lower().startswith("http"):
            self.video, self.kind = resolve_source(self.spec, Path(work_dir)), "video"
        elif any(ch in self.spec for ch in "*?[") :
            self.files, self.kind = sorted(glob.glob(self.spec), key=_natural), "images"
        else:
            p = Path(self.spec)
            if p.is_dir():
                self.files = sorted((str(f) for f in p.iterdir() if f.suffix.lower() in IMAGE_EXT), key=_natural)
                self.kind = "images"
            elif p.suffix.lower() in IMAGE_EXT:
                self.files, self.kind = [str(p)], "images"
            elif p.exists():
                self.video, self.kind = str(p), "video"
            else:
                raise FileNotFoundError(f"Input not found: {spec}")
        if self.kind == "images" and not self.files:
            raise FileNotFoundError(f"No images found for: {spec}")
        self.info = self._probe()

    def _probe(self) -> SourceInfo:
        if self.kind == "images":
            first = None
            for f in self.files:                    # the first readable image defines the working size
                first = cv2.imread(f)
                if first is not None:
                    break
            if first is None:
                raise IOError(f"None of the {len(self.files)} image files could be read: {self.spec}")
            first = _scale(first, self.max_side)
            return SourceInfo("images", self.default_fps, first.shape[1], first.shape[0], len(self.files),
                              self.assume_sequence and len(self.files) > 1)
        cap = cv2.VideoCapture(self.cam if self.cam is not None else self.video)
        if not cap.isOpened():
            raise IOError(f"Cannot open video source: {self.spec}")
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.cam is None else 0
        cap.release()
        s = self.max_side / max(w, h) if self.max_side and max(w, h) > self.max_side else 1.0
        return SourceInfo(self.kind, float(fps), int(round(w * s)), int(round(h * s)), max(n, 0), True)

    def __iter__(self) -> Iterator[Frame]:
        yielded = 0
        if self.kind == "images":
            size = (self.info.width, self.info.height)
            for i, f in enumerate(self.files):
                if i % self.stride:
                    continue
                img = cv2.imread(f)
                if img is None:
                    continue  # skip unreadable files
                img = _scale(img, self.max_side)
                if (img.shape[1], img.shape[0]) != size:
                    img = cv2.resize(img, size)
                yield Frame(i, i / self.info.fps, img, Path(f).name)
                yielded += 1
                if self.max_frames and yielded >= self.max_frames:
                    return
            return
        cap = cv2.VideoCapture(self.cam if self.cam is not None else self.video)
        start = int(round(self.start_s * self.info.fps))
        end = start + int(round(self.duration_s * self.info.fps)) if self.duration_s > 0 else 10 ** 12
        if start and self.cam is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        i = start
        try:
            while i < end:
                ok, img = cap.read()
                if not ok:
                    break
                if (i - start) % self.stride == 0:
                    yield Frame(i, i / self.info.fps, _scale(img, self.max_side), "")
                    yielded += 1
                    if self.max_frames and yielded >= self.max_frames:
                        break
                i += 1
        finally:
            cap.release()


def open_source(spec: str, **kw) -> FrameSource:
    return FrameSource(spec, **kw)
