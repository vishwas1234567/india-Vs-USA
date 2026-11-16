"""Dataset statistics and Indian-vs-US comparison (class mix, object density, box sizes, image conditions)."""
from __future__ import annotations

import math
from typing import Dict, Optional

import cv2
import numpy as np
from scipy.stats import ks_2samp

from .dataset import ISCVDataset
from .taxonomy import CLASSES


def dataset_stats(ds: ISCVDataset, max_images: int = 300, photometric_images: int = 100, seed: int = 0) -> Dict[str, object]:
    """Annotation statistics over all samples and photometric statistics over a random subset of images."""
    n_obj = np.array([len(s.labels) for s in ds.samples], float)
    counts = ds.class_counts()
    area_frac, aspect = [], []
    sizes: Dict[str, int] = {}
    for s in ds.samples:
        if s.width and s.height:
            sizes[f"{s.width}x{s.height}"] = sizes.get(f"{s.width}x{s.height}", 0) + 1
            if len(s.boxes):
                w, h = (s.boxes[:, 2] - s.boxes[:, 0]).clip(1), (s.boxes[:, 3] - s.boxes[:, 1]).clip(1)
                area_frac.extend((w * h / (s.width * s.height)).tolist())
                aspect.extend((w / h).tolist())
    out: Dict[str, object] = {
        "name": ds.name, "source": ds.source_dataset(), "n_images": len(ds), "n_objects": int(n_obj.sum()),
        "objects_per_image_mean": float(n_obj.mean()) if len(n_obj) else 0.0,
        "objects_per_image_p95": float(np.percentile(n_obj, 95)) if len(n_obj) else 0.0,
        "class_counts": counts,
        "class_share": {k: (v / max(sum(counts.values()), 1)) for k, v in counts.items()},
        "box_area_frac_p10_p50_p90": [float(np.percentile(area_frac, q)) for q in (10, 50, 90)] if area_frac else None,
        "box_aspect_median": float(np.median(aspect)) if aspect else None,
        "box_log10_area_frac": [math.log10(a) for a in area_frac[:5000]],
        "image_sizes": dict(sorted(sizes.items(), key=lambda kv: -kv[1])[:5]),
    }
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(photometric_images, len(ds)), replace=False) if len(ds) else []
    br, ct, sh, ov = [], [], [], []
    for i in idx:
        try:
            g = cv2.cvtColor(ds.read_image(int(i)), cv2.COLOR_BGR2GRAY)
        except IOError:
            continue
        g = cv2.resize(g, (640, int(g.shape[0] * 640 / g.shape[1])))
        br.append(g.mean()); ct.append(g.std()); sh.append(cv2.Laplacian(g, cv2.CV_64F).var()); ov.append((g >= 250).mean())
    if br:
        out["photometric"] = {"brightness_mean": float(np.mean(br)), "contrast_std": float(np.mean(ct)),
                              "sharpness_lap_var_640": float(np.median(sh)), "overexposed_frac": float(np.mean(ov)),
                              "n_images": len(br)}
    return out


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p, q = np.asarray(p, float), np.asarray(q, float)
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(np.where(a > 0, a * np.log2(a / np.maximum(b, 1e-12)), 0.0)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def compare_stats(a: Dict[str, object], b: Dict[str, object]) -> Dict[str, object]:
    """Quantify how two datasets differ: class-mix divergence (bits, 0-1), box-size KS statistic, density ratio."""
    shared = [c for c in CLASSES if a["class_counts"][c] > 0 and b["class_counts"][c] > 0]
    out: Dict[str, object] = {"a": a["name"], "b": b["name"], "shared_classes": shared,
                              "classes_only_in_a": [c for c in CLASSES if a["class_counts"][c] > 0 and b["class_counts"][c] == 0],
                              "classes_only_in_b": [c for c in CLASSES if b["class_counts"][c] > 0 and a["class_counts"][c] == 0]}
    if len(shared) >= 2:
        out["class_mix_js_bits"] = js_divergence([a["class_counts"][c] for c in shared], [b["class_counts"][c] for c in shared])
    if a["box_log10_area_frac"] and b["box_log10_area_frac"]:
        ks = ks_2samp(a["box_log10_area_frac"], b["box_log10_area_frac"])
        out["box_size_ks"] = {"statistic": float(ks.statistic), "pvalue": float(ks.pvalue)}
    out["objects_per_image_ratio_a_over_b"] = (a["objects_per_image_mean"] / b["objects_per_image_mean"]
                                               if b["objects_per_image_mean"] else None)
    return out
