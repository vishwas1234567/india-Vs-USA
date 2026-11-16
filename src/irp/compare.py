"""One comparative video: the same algorithm on two sources (e.g. an Indian and a US dataset) side by side.

A source is a video/image spec, an image folder, or an `iscvd.ISCVDataset` (then ground truth is available and the running
detection AP50 on the shown frames is displayed). Panels show detections + tracks, road understanding, lanes and a bird's-eye
inset; the bar under each panel shows detection accuracy (if labelled), tracking, road coverage, lane confidence and latency.
A final summary card compares the averages. The same detector, resolution and settings are used for both sides.

    python -m irp.compare --a data/junction_clip.mp4 --label-a "India (demo)" --b data/junction2_360p.mp4 --label-b "Other" --out results/compare
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from iscvd.dataset import ISCVDataset
from iscvd.metrics import evaluate_detections
from iscvd.taxonomy import CLASS_ID

from .common import put_text
from .pipeline import Pipeline, PipelineConfig
from .runtime import VideoWriter
from .sources import open_source

Source = Union[str, ISCVDataset]
PANEL = (640, 360)


def _letterbox(img: np.ndarray, size=PANEL) -> np.ndarray:
    h, w = img.shape[:2]
    r = min(size[0] / w, size[1] / h)
    out = np.full((size[1], size[0], 3), 20, np.uint8)
    nw, nh = int(w * r), int(h * r)
    out[(size[1] - nh) // 2:(size[1] - nh) // 2 + nh, (size[0] - nw) // 2:(size[0] - nw) // 2 + nw] = cv2.resize(img, (nw, nh))
    return out


def _frames(src: Source, max_frames: int, stride: int) -> Iterator[Tuple[np.ndarray, Optional[dict], bool]]:
    """Yield (image, ground truth or None, sequential)."""
    if isinstance(src, ISCVDataset):
        for i in range(0, len(src), max(stride, 1)):
            if max_frames and i // max(stride, 1) >= max_frames:
                return
            s = src.samples[i]
            yield src.read_image(i), {"boxes": s.boxes, "labels": s.labels}, False
        return
    fs = open_source(src, stride=stride, max_frames=max_frames, max_side=PANEL[0])
    for f in fs:
        yield f.image, None, fs.info.sequential


_text = put_text


class _Side:
    def __init__(self, src: Source, label: str, cfg: PipelineConfig, max_frames: int, stride: int):
        self.label, self.pipe = label, Pipeline(cfg)
        self.it = _frames(src, max_frames, stride)
        self.preds: List[dict] = []
        self.gts: List[dict] = []
        self.last: Optional[np.ndarray] = None
        self.last_stats: dict = {}
        self.ap50: Optional[float] = None
        self.idx = 0
        self.done = False
        self.stats: Dict[str, List[float]] = defaultdict(list)

    def step(self, detector_classes: Sequence[int]) -> None:
        try:
            img, gt, seq = next(self.it)
        except StopIteration:
            self.done = True
            return
        img = _letterbox(img)
        vis, st = self.pipe.process(img, self.idx, self.idx / 10.0, 10.0, sequential=seq)
        self.idx += 1
        self.last, self.last_stats = vis, st
        for k in ("n_det", "n_tracks", "road_cov", "lane_conf", "latency_ms"):
            self.stats[k].append(float(st[k]))
        if gt is not None:                         # detection accuracy on the frames shown so far
            dets = self.pipe.last_dets
            self.preds.append({"boxes": np.array([d.box for d in dets if d.cat in CLASS_ID], np.float32).reshape(-1, 4),
                               "scores": np.array([d.conf for d in dets if d.cat in CLASS_ID]),
                               "labels": np.array([CLASS_ID[d.cat] for d in dets if d.cat in CLASS_ID], np.int64)})
            # GT boxes are rescaled into panel coordinates (letterboxed) so that predictions and labels share a frame
            h0, w0 = img.shape[:2]
            self.gts.append(gt)
            res = evaluate_detections(self.preds, self.gts, list(detector_classes), (0.5,))
            self.ap50 = None if res["mAP50"] != res["mAP50"] else float(res["mAP50"])

    def bar(self) -> np.ndarray:
        bar = np.full((78, PANEL[0], 3), 28, np.uint8)
        s = self.last_stats
        ap = "AP50 n/a (no labels)" if self.ap50 is None and not self.gts else f"AP50 {0.0 if self.ap50 is None else self.ap50:.2f}"
        _text(bar, f"{self.label}", (8, 18), 0.6, (0, 220, 255))
        _text(bar, f"detect: {s.get('n_det', 0)}/frame  {ap}", (8, 38), 0.5)
        _text(bar, f"tracks: {s.get('n_tracks', 0)}  road: {100 * s.get('road_cov', 0):.0f}%  lane: {s.get('lane_state', '-')} {s.get('lane_conf', 0):.2f}", (8, 56), 0.5)
        _text(bar, f"latency {s.get('latency_ms', 0):.0f} ms  risk {s.get('risk', '-')}", (8, 72), 0.5)
        return bar


def _summary_card(a: _Side, b: _Side) -> np.ndarray:
    card = np.full((PANEL[1] + 78, PANEL[0] * 2, 3), 25, np.uint8)
    _text(card, "Summary (same algorithm and settings on both sources)", (20, 40), 0.8, (0, 220, 255))
    rows = [("frames processed", lambda s: f"{s.idx}"), ("detections / frame", lambda s: f"{np.mean(s.stats['n_det']):.1f}"),
            ("tracked objects / frame", lambda s: f"{np.mean(s.stats['n_tracks']):.1f}"),
            ("road coverage", lambda s: f"{100 * np.mean(s.stats['road_cov']):.0f}%"),
            ("lane confidence", lambda s: f"{np.mean(s.stats['lane_conf']):.2f}"),
            ("latency (ms)", lambda s: f"{np.mean(s.stats['latency_ms']):.0f}"),
            ("throughput (fps)", lambda s: f"{1000 / max(np.mean(s.stats['latency_ms']), 1e-6):.1f}"),
            ("detection AP50 (labelled frames)", lambda s: "n/a" if not s.gts else f"{(s.ap50 or 0.0):.3f}")]
    _text(card, a.label, (460, 90), 0.65, (0, 255, 0))
    _text(card, b.label, (860, 90), 0.65, (0, 255, 0))
    for i, (name, fn) in enumerate(rows):
        y = 140 + 45 * i
        _text(card, name, (20, y), 0.6)
        _text(card, fn(a) if a.idx else "-", (460, y), 0.6)
        _text(card, fn(b) if b.idx else "-", (860, y), 0.6)
    return card


def compare(a: Source, b: Source, out: str, label_a: str = "A", label_b: str = "B", detector: str = "auto", max_frames: int = 150,
            stride: int = 1, device: str = "auto", summary_seconds: float = 3.0, fps: float = 15.0) -> dict:
    """Render the single comparison video and return the summary metrics for both sources."""
    cfg = lambda: PipelineConfig(source="", out=out, detector=detector, device=device, write_video=False, max_side=PANEL[0])
    side_a, side_b = _Side(a, label_a, cfg(), max_frames, stride), _Side(b, label_b, cfg(), max_frames, stride)
    classes = [CLASS_ID[c] for c in ("car", "heavy", "two_wheeler", "pedestrian")]   # classes comparable across Indian and US labels
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    W, H = PANEL[0] * 2, PANEL[1] + 78 + 40
    with VideoWriter(str(out_dir / "comparison.mp4"), fps, (W, H)) as wr:
        while not (side_a.done and side_b.done):
            side_a.step(classes)
            side_b.step(classes)
            if side_a.last is None or side_b.last is None:
                break
            top = np.full((40, W, 3), 15, np.uint8)
            _text(top, "Indian vs reference driving scenes: detection | road | tracking | lanes | performance", (10, 26), 0.65)
            frame = np.vstack([top, np.hstack([side_a.last, side_b.last]), np.hstack([side_a.bar(), side_b.bar()])])
            wr.write(frame)
        card = _summary_card(side_a, side_b)
        for _ in range(int(summary_seconds * fps)):
            wr.write(np.vstack([np.full((40, W, 3), 15, np.uint8), card]))
    summary = {}
    for side in (side_a, side_b):
        summary[side.label] = {"frames": side.idx, **{k: float(np.mean(v)) for k, v in side.stats.items()},
                               "ap50_labelled_frames": side.ap50 if side.gts else None}
    (out_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True), ap.add_argument("--b", required=True)
    ap.add_argument("--label-a", default="A"), ap.add_argument("--label-b", default="B")
    ap.add_argument("--out", default="results/compare")
    ap.add_argument("--detector", default="auto")
    ap.add_argument("--max-frames", type=int, default=150)
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args(argv)
    print(json.dumps(compare(a.a, a.b, a.out, a.label_a, a.label_b, a.detector, a.max_frames, a.stride), indent=2))


if __name__ == "__main__":
    main()
