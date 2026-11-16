"""Evaluation metrics shared by every dataset and experiment (so cross-dataset numbers are comparable).

    detection     evaluate_detections()  COCO-style AP at IoU 0.5:0.95 (101-point interpolation), AP50, AP75, per class
    segmentation  ConfusionMatrix        mIoU, per-class IoU, pixel accuracy (+ binary drivable IoU)
    panoptic      PanopticQuality        PQ / SQ / RQ per category (segment id convention: category = id // 1000)
    tracking      clear_mot()            MOTA, MOTP, ID switches, IDF1
    lanes         lane_f1()              precision / recall / F1 with a lateral-distance threshold
    trajectories  ade_fde()              average / final displacement error
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

IOU_THRESHOLDS = tuple(np.round(np.arange(0.5, 0.96, 0.05), 2))


# --------------------------------------------------------------------------- #
# Boxes and detection
# --------------------------------------------------------------------------- #
def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix (N, M) for xyxy boxes."""
    a = np.asarray(a, np.float64).reshape(-1, 4)
    b = np.asarray(b, np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """COCO-style 101-point interpolated AP (precision envelope, no artificial (0, 1) point)."""
    recall, precision = np.asarray(recall, float), np.asarray(precision, float)
    if len(recall) == 0:
        return 0.0
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    idx = np.searchsorted(recall, np.linspace(0, 1, 101), side="left")
    return float(np.mean([envelope[i] if i < len(envelope) else 0.0 for i in idx]))


def _class_ap(preds, gts, cls: int, iou_thr: float) -> Optional[float]:
    """AP of one class at one IoU threshold; None if there is no ground truth for the class."""
    records: List[Tuple[float, int]] = []  # (score, is_tp)
    n_gt = 0
    for p, g in zip(preds, gts):
        gb = np.asarray(g["boxes"]).reshape(-1, 4)[np.asarray(g["labels"]) == cls] if len(g["labels"]) else np.zeros((0, 4))
        n_gt += len(gb)
        pm = np.asarray(p["labels"]) == cls if len(p["labels"]) else np.zeros(0, bool)
        pb, ps = np.asarray(p["boxes"]).reshape(-1, 4)[pm], np.asarray(p["scores"])[pm]
        order = np.argsort(-ps)
        pb, ps = pb[order], ps[order]
        used = np.zeros(len(gb), bool)
        ious = box_iou(pb, gb)
        for i in range(len(pb)):
            tp = 0
            if len(gb):
                j = int(np.argmax(np.where(used, -1.0, ious[i])))
                if not used[j] and ious[i, j] >= iou_thr:
                    used[j], tp = True, 1
            records.append((float(ps[i]), tp))
    if n_gt == 0:
        return None
    if not records:
        return 0.0
    records.sort(key=lambda r: -r[0])
    tp = np.cumsum([r[1] for r in records])
    fp = np.cumsum([1 - r[1] for r in records])
    return average_precision(tp / n_gt, tp / np.maximum(tp + fp, 1e-12))


def evaluate_detections(preds: Sequence[dict], gts: Sequence[dict], classes: Sequence[int],
                        iou_thresholds: Sequence[float] = IOU_THRESHOLDS) -> Dict[str, object]:
    """Detection metrics over a dataset.

    preds[i] = {"boxes": (N,4), "scores": (N,), "labels": (N,)}, gts[i] = {"boxes": (M,4), "labels": (M,)}.
    Classes without ground truth are excluded from the means and reported as NaN (not silently counted as zero).
    """
    per_class: Dict[int, Dict[str, float]] = {}
    for c in classes:
        aps = {t: _class_ap(preds, gts, c, t) for t in iou_thresholds}
        if all(v is None for v in aps.values()):
            per_class[c] = {"AP": float("nan"), "AP50": float("nan"), "AP75": float("nan")}
            continue
        per_class[c] = {"AP": float(np.mean([v for v in aps.values()])),
                        "AP50": aps.get(0.5, float("nan")) if 0.5 in aps else float("nan"),
                        "AP75": aps.get(0.75, float("nan")) if 0.75 in aps else float("nan")}
    valid = [v for v in per_class.values() if not math.isnan(v["AP"])]
    mean = lambda k: float(np.mean([v[k] for v in valid])) if valid else float("nan")
    return {"mAP": mean("AP"), "mAP50": mean("AP50"), "mAP75": mean("AP75"), "per_class": per_class,
            "classes_evaluated": [c for c, v in per_class.items() if not math.isnan(v["AP"])]}


# --------------------------------------------------------------------------- #
# Semantic segmentation
# --------------------------------------------------------------------------- #
class ConfusionMatrix:
    def __init__(self, n_classes: int, ignore_index: int = 255):
        self.n, self.ignore = n_classes, ignore_index
        self.m = np.zeros((n_classes, n_classes), np.int64)

    def update(self, pred: np.ndarray, gt: np.ndarray) -> None:
        pred, gt = np.asarray(pred).ravel(), np.asarray(gt).ravel()
        keep = (gt != self.ignore) & (gt < self.n) & (pred < self.n)
        self.m += np.bincount(gt[keep] * self.n + pred[keep], minlength=self.n ** 2).reshape(self.n, self.n)

    def iou(self) -> np.ndarray:
        tp = np.diag(self.m).astype(float)
        union = self.m.sum(0) + self.m.sum(1) - tp
        return np.where(union > 0, tp / np.maximum(union, 1), np.nan)

    def miou(self) -> float:
        v = self.iou()
        return float(np.nanmean(v)) if np.any(~np.isnan(v)) else float("nan")

    def pixel_accuracy(self) -> float:
        return float(np.diag(self.m).sum() / max(self.m.sum(), 1))


def binary_iou(pred: np.ndarray, gt: np.ndarray, valid: Optional[np.ndarray] = None) -> float:
    """IoU of a binary mask (e.g. drivable area) against ground truth."""
    p, g = np.asarray(pred).astype(bool), np.asarray(gt).astype(bool)
    if valid is not None:
        p, g = p & valid, g & valid
    union = (p | g).sum()
    return float((p & g).sum() / union) if union else float("nan")


# --------------------------------------------------------------------------- #
# Panoptic quality
# --------------------------------------------------------------------------- #
class PanopticQuality:
    """PQ with segments identified by integer ids; category = id // `divisor` (id 0 or negative = void)."""

    def __init__(self, divisor: int = 1000, iou_threshold: float = 0.5):
        self.div, self.thr = divisor, iou_threshold
        self.stats: Dict[int, List[float]] = {}  # category -> [iou_sum, tp, fp, fn]

    def update(self, pred: np.ndarray, gt: np.ndarray) -> None:
        pred, gt = np.asarray(pred, np.int64), np.asarray(gt, np.int64)
        p_ids = [i for i in np.unique(pred) if i > 0]
        g_ids = [i for i in np.unique(gt) if i > 0]
        p_area = {i: int((pred == i).sum()) for i in p_ids}
        g_area = {i: int((gt == i).sum()) for i in g_ids}
        matched_p, matched_g = set(), set()
        pairs = np.stack([gt.ravel(), pred.ravel()], 1)
        pairs = pairs[(pairs[:, 0] > 0) & (pairs[:, 1] > 0)]
        if len(pairs):
            keys, counts = np.unique(pairs, axis=0, return_counts=True)
            for (g, p), inter in zip(keys, counts):
                if g // self.div != p // self.div:
                    continue
                iou = inter / (g_area[g] + p_area[p] - inter)
                if iou > self.thr:  # > 0.5 guarantees a unique match
                    st = self.stats.setdefault(int(g // self.div), [0.0, 0, 0, 0])
                    st[0] += iou
                    st[1] += 1
                    matched_p.add(p)
                    matched_g.add(g)
        for g in g_ids:
            if g not in matched_g:
                self.stats.setdefault(int(g // self.div), [0.0, 0, 0, 0])[3] += 1
        for p in p_ids:
            if p not in matched_p:
                self.stats.setdefault(int(p // self.div), [0.0, 0, 0, 0])[2] += 1

    def result(self) -> Dict[str, object]:
        per_cat = {}
        for c, (iou_sum, tp, fp, fn) in self.stats.items():
            sq = iou_sum / tp if tp else 0.0
            rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + fp + fn) else 0.0
            per_cat[c] = {"PQ": sq * rq, "SQ": sq, "RQ": rq, "tp": tp, "fp": fp, "fn": fn}
        pq = float(np.mean([v["PQ"] for v in per_cat.values()])) if per_cat else float("nan")
        return {"PQ": pq, "per_category": per_cat}


# --------------------------------------------------------------------------- #
# Tracking: CLEAR-MOT and IDF1
# --------------------------------------------------------------------------- #
def clear_mot(gt: Dict[int, List[Tuple[int, Sequence[float]]]], pred: Dict[int, List[Tuple[int, Sequence[float]]]],
              iou_thr: float = 0.5) -> Dict[str, float]:
    """MOTA, MOTP, ID switches and IDF1. Inputs map frame -> [(track_id, xyxy box), ...]."""
    frames = sorted(set(gt) | set(pred))
    prev: Dict[int, int] = {}
    fp = fn = idsw = n_gt = 0
    iou_sum, n_match = 0.0, 0
    pair_counts: Dict[Tuple[int, int], int] = {}
    n_pred_total = 0
    for f in frames:
        g, p = gt.get(f, []), pred.get(f, [])
        n_gt += len(g)
        n_pred_total += len(p)
        gb = np.array([b for _, b in g], float).reshape(-1, 4)
        pb = np.array([b for _, b in p], float).reshape(-1, 4)
        ious = box_iou(gb, pb)
        matches: Dict[int, int] = {}
        gids = [i for i, _ in g]
        pids = [i for i, _ in p]
        for gi, gid in enumerate(gids):  # keep previous correspondences when still valid
            pid = prev.get(gid)
            if pid in pids and ious[gi, pids.index(pid)] >= iou_thr:
                matches[gi] = pids.index(pid)
        free_g = [i for i in range(len(g)) if i not in matches]
        free_p = [j for j in range(len(p)) if j not in matches.values()]
        if free_g and free_p:
            sub = ious[np.ix_(free_g, free_p)]
            r, c = linear_sum_assignment(-sub)
            for ri, ci in zip(r, c):
                if sub[ri, ci] >= iou_thr:
                    matches[free_g[ri]] = free_p[ci]
        for gi, pj in matches.items():
            gid, pid = gids[gi], pids[pj]
            if gid in prev and prev[gid] != pid:
                idsw += 1
            prev[gid] = pid
            iou_sum += ious[gi, pj]
            n_match += 1
            pair_counts[(gid, pid)] = pair_counts.get((gid, pid), 0) + 1
        fn += len(g) - len(matches)
        fp += len(p) - len(matches)
    mota = 1.0 - (fn + fp + idsw) / max(n_gt, 1)
    # IDF1: global assignment of gt ids to predicted ids
    gt_ids = sorted({i for g in gt.values() for i, _ in g})
    pr_ids = sorted({i for p in pred.values() for i, _ in p})
    idtp = 0
    if gt_ids and pr_ids:
        w = np.zeros((len(gt_ids), len(pr_ids)))
        for (gid, pid), cnt in pair_counts.items():
            w[gt_ids.index(gid), pr_ids.index(pid)] = cnt
        r, c = linear_sum_assignment(-w)
        idtp = int(w[r, c].sum())
    idf1 = 2 * idtp / max(n_gt + n_pred_total, 1)
    return {"MOTA": float(mota), "MOTP": float(iou_sum / max(n_match, 1)), "IDSW": int(idsw), "FP": int(fp),
            "FN": int(fn), "IDF1": float(idf1), "gt_detections": int(n_gt)}


# --------------------------------------------------------------------------- #
# Lanes and trajectories
# --------------------------------------------------------------------------- #
def _x_at(poly: np.ndarray, ys: np.ndarray) -> np.ndarray:
    o = np.argsort(poly[:, 1])
    return np.interp(ys, poly[o, 1], poly[o, 0], left=np.nan, right=np.nan)


def lane_distance(pred: np.ndarray, gt: np.ndarray, n: int = 30) -> float:
    """Mean horizontal distance (pixels) between two lane polylines over their shared y-range (inf if < 50% shared)."""
    y0, y1 = max(pred[:, 1].min(), gt[:, 1].min()), min(pred[:, 1].max(), gt[:, 1].max())
    span_gt = gt[:, 1].max() - gt[:, 1].min()
    if y1 <= y0 or (y1 - y0) < 0.5 * max(span_gt, 1e-6):
        return float("inf")
    ys = np.linspace(y0, y1, n)
    return float(np.nanmean(np.abs(_x_at(pred, ys) - _x_at(gt, ys))))


def lane_f1(pred_lanes: Sequence[np.ndarray], gt_lanes: Sequence[np.ndarray], thr_px: float = 20.0) -> Dict[str, float]:
    """Lane precision/recall/F1: one-to-one matching by mean lateral distance below thr_px."""
    if not pred_lanes or not gt_lanes:
        tp = 0
    else:
        d = np.array([[lane_distance(p, g) for g in gt_lanes] for p in pred_lanes])
        d = np.where(np.isfinite(d), d, 1e6)
        r, c = linear_sum_assignment(d)
        tp = int(sum(1 for i, j in zip(r, c) if d[i, j] <= thr_px))
    fp, fn = len(pred_lanes) - tp, len(gt_lanes) - tp
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {"precision": prec, "recall": rec, "F1": 2 * prec * rec / max(prec + rec, 1e-12), "tp": tp, "fp": fp, "fn": fn}


def ade_fde(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """Average and final displacement error for trajectories shaped (N, K, 2) (same units as the inputs)."""
    d = np.linalg.norm(np.asarray(pred, float) - np.asarray(gt, float), axis=-1)
    return {"ADE": float(d.mean()), "FDE": float(d[:, -1].mean())}
