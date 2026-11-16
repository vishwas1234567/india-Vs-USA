"""Benchmark harness: the same detector, input handling, taxonomy and metrics on any dataset.

A detector is any callable: image (BGR uint8) -> {"boxes": (N,4) xyxy, "scores": (N,), "labels": (N,) unified ids}.
`from_irp_detector()` adapts the toolkit's detectors (irp.detect) to that contract.

Protocol (fixed for every dataset, which is what makes numbers comparable):
  * identical detector weights, input size, confidence floor and NMS settings
  * unified taxonomy (iscvd.taxonomy); metrics over (a) every class the dataset can contain and
    (b) the classes shared by all compared datasets
  * classes with no ground truth are excluded from means (reported as NaN), never counted as zero
  * latency measured per image after warm-up (median, p95) with a stated device
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .dataset import ISCVDataset
from .metrics import evaluate_detections
from .taxonomy import CLASSES, CLASS_ID, classes_present, shared_classes

Detector = Callable[[np.ndarray], Dict[str, np.ndarray]]


def from_irp_detector(detector) -> Detector:
    """Adapt an irp detector (returns a list of irp.common.Detection) to the benchmark contract."""
    def run(image: np.ndarray) -> Dict[str, np.ndarray]:
        dets = detector(image)
        keep = [d for d in dets if d.cat in CLASS_ID]
        return {"boxes": np.array([d.box for d in keep], np.float32).reshape(-1, 4),
                "scores": np.array([d.conf for d in keep], np.float32),
                "labels": np.array([CLASS_ID[d.cat] for d in keep], np.int64)}
    return run


def evaluate_detector(detector: Detector, ds: ISCVDataset, class_names: Optional[Sequence[str]] = None,
                      max_images: int = 0, warmup: int = 2) -> Dict[str, object]:
    n = len(ds) if not max_images else min(max_images, len(ds))
    preds, gts, times = [], [], []
    for i in range(n):
        img = ds.read_image(i)
        t0 = time.perf_counter()
        out = detector(img)
        times.append(time.perf_counter() - t0)
        preds.append({k: np.asarray(v) for k, v in out.items()})
        gts.append({"boxes": ds.samples[i].boxes, "labels": ds.samples[i].labels})
    names = list(class_names or ds.classes_present() or CLASSES)
    res = evaluate_detections(preds, gts, [CLASS_ID[c] for c in names])
    t = np.array(times[warmup:] or times)
    res.update({"dataset": ds.name, "n_images": n, "classes": names,
                "latency_ms_median": float(np.median(t) * 1000), "latency_ms_p95": float(np.percentile(t, 95) * 1000),
                "fps": float(1.0 / max(np.mean(t), 1e-9))})
    res["per_class"] = {CLASSES[c]: v for c, v in res["per_class"].items()}
    res["classes_evaluated"] = [CLASSES[c] for c in res["classes_evaluated"]]
    return res


def compare_datasets(detector: Detector, datasets: Dict[str, ISCVDataset], max_images: int = 0) -> Dict[str, object]:
    """Run the identical protocol on several datasets; report all-class and shared-class metrics side by side."""
    shared = shared_classes([d.source_dataset() for d in datasets.values()])
    out: Dict[str, object] = {"shared_classes": shared, "results": {}}
    for label, ds in datasets.items():
        full = evaluate_detector(detector, ds, max_images=max_images)
        sub = evaluate_detector(detector, ds, class_names=shared, max_images=max_images) if shared else None
        out["results"][label] = {"all_classes": full, "shared_classes": sub}
    return out


def to_markdown(comparison: Dict[str, object]) -> str:
    lines = ["| Dataset | Images | mAP50 (all) | mAP50:95 (all) | mAP50 (shared) | FPS | Latency p95 (ms) |", "|---|---|---|---|---|---|---|"]
    for label, r in comparison["results"].items():
        a, s = r["all_classes"], r["shared_classes"]
        lines.append(f"| {label} | {a['n_images']} | {a['mAP50']:.3f} | {a['mAP']:.3f} | "
                     f"{(s['mAP50'] if s else float('nan')):.3f} | {a['fps']:.1f} | {a['latency_ms_p95']:.1f} |")
    return "\n".join(lines)


def save(comparison: Dict[str, object], out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(json.dumps(comparison, indent=2, default=float))
    (out / "comparison.md").write_text(to_markdown(comparison) + "\n")
