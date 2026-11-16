"""Runtime utilities: device selection (CPU/GPU), stage profiling, threaded prefetch, fast video writing."""
from __future__ import annotations

import queue
import shutil
import subprocess
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import cv2
import numpy as np


def select_device(prefer: str = "auto") -> str:
    """'cuda:0' | 'mps' | 'cpu'. Never imports torch unless it is installed; CPU is always a valid fallback."""
    if prefer not in ("auto", "gpu", "cuda"):
        return prefer
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class Profiler:
    """Accumulates per-stage wall time: `with prof.stage("detect"): ...`; summary() gives mean/p95 ms and FPS."""

    def __init__(self):
        self.t: Dict[str, List[float]] = defaultdict(list)

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.t[name].append(time.perf_counter() - t0)

    def summary(self) -> Dict[str, Dict[str, float]]:
        out = {k: {"mean_ms": 1000 * float(np.mean(v)), "p95_ms": 1000 * float(np.percentile(v, 95)), "n": len(v)}
               for k, v in self.t.items()}
        total = sum(1000 * float(np.mean(v)) for k, v in self.t.items() if k != "total")
        out["_sum_of_stages_ms"] = {"mean_ms": total, "p95_ms": total, "n": 0}
        if "total" in self.t:
            out["_fps"] = {"mean_ms": 1000.0 / max(out["total"]["mean_ms"], 1e-9), "p95_ms": 0.0, "n": 0}
        return out


def prefetch(frames: Iterable, depth: int = 4) -> Iterator:
    """Decode frames in a background thread so I/O overlaps with processing (useful on CPU and GPU alike)."""
    q: "queue.Queue" = queue.Queue(maxsize=depth)
    stop = object()

    def worker():
        try:
            for f in frames:
                q.put(f)
        finally:
            q.put(stop)

    threading.Thread(target=worker, daemon=True).start()
    while True:
        item = q.get()
        if item is stop:
            return
        yield item


class VideoWriter:
    """H.264 writer through an ffmpeg pipe when ffmpeg exists (small files, fast), else OpenCV mp4v."""

    def __init__(self, path: str, fps: float, size: Tuple[int, int], crf: int = 24):
        self.path, self.size, self.proc, self.cv = path, size, None, None
        if shutil.which("ffmpeg"):
            self.proc = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{size[0]}x{size[1]}",
                 "-r", f"{max(fps, 1):.3f}", "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
                 "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
        else:
            self.cv = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), max(fps, 1), size)

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[1] != self.size[0] or frame.shape[0] != self.size[1]:
            frame = cv2.resize(frame, self.size)
        if self.proc is not None:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        else:
            self.cv.write(frame)

    def close(self) -> None:
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait()
        elif self.cv is not None:
            self.cv.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class AdaptiveScale:
    """Keeps per-frame latency near a budget by lowering/raising the processing width (CPU real-time mode)."""

    def __init__(self, budget_ms: float = 100.0, widths: Tuple[int, ...] = (320, 480, 640, 960, 1280), start: int = 640):
        self.budget, self.widths = budget_ms, tuple(sorted(widths))
        self.i = min(range(len(self.widths)), key=lambda k: abs(self.widths[k] - start))
        self.ema: Optional[float] = None

    @property
    def width(self) -> int:
        return self.widths[self.i]

    def update(self, latency_ms: float) -> int:
        self.ema = latency_ms if self.ema is None else 0.8 * self.ema + 0.2 * latency_ms
        if self.ema > 1.15 * self.budget and self.i > 0:
            self.i, self.ema = self.i - 1, None
        elif self.ema < 0.6 * self.budget and self.i < len(self.widths) - 1:
            self.i, self.ema = self.i + 1, None
        return self.width
