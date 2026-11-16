"""Controlled tracker benchmark with exact (synthetic) ground truth.

Real clips have no labels, so tracker design choices are checked on simulated junction traffic:
objects cross the image at constant velocity, detections are jittered, randomly missed, interrupted by
occlusion gaps, and polluted by false positives. Because each detection knows its source object,
identity errors can be counted exactly.

Metrics (per scenario, mean +- std over seeds):
  frag_per_object  mean number of distinct track ids per true object (1.0 is ideal)
  id_switches      total identity changes along true objects
  purity           fraction of a track's detections that come from its majority object
  count_ratio      confirmed true-object tracks / true objects (1.0 is ideal)
  fp_tracks        confirmed tracks that are mostly false positives

Usage:  python -m irp.bench --out results/_benchmark --seeds 20
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .static import CentroidTracker
from .tracking import KalmanTracker

Box = Tuple[int, int, int, int]

SCENARIOS: Dict[str, dict] = {
    "clean": dict(p_miss=0.0, jitter=1.0, gap_prob=0.0, fp_rate=0.0, n_objects=8),
    "missed": dict(p_miss=0.15, jitter=2.0, gap_prob=0.3, fp_rate=0.1, n_objects=8),
    "occluded": dict(p_miss=0.15, jitter=3.0, gap_prob=0.7, fp_rate=0.2, n_objects=8),
    "dense": dict(p_miss=0.15, jitter=3.0, gap_prob=0.7, fp_rate=0.2, n_objects=20),
}


def simulate(seed: int, n_objects: int = 8, frames: int = 240, size: Tuple[int, int] = (640, 360),
             p_miss: float = 0.1, jitter: float = 2.0, gap_prob: float = 0.3, fp_rate: float = 0.1):
    """Return (detections, sources): per frame a list of boxes and the source object id (-1 = false positive)."""
    rng = np.random.default_rng(seed)
    w, h = size
    objs = []
    for k in range(n_objects):
        left = rng.random() < 0.5
        speed = rng.uniform(3, 12) * (1 if left else -1)
        bw, bh = int(rng.integers(26, 70)), int(rng.integers(20, 50))
        t0 = int(rng.integers(0, max(frames // 3, 1)))
        y = rng.uniform(0.45 * h, 0.9 * h - bh)
        x0 = -bw if left else w
        gap = None
        if rng.random() < gap_prob:
            g0 = t0 + int(rng.integers(8, 40))
            gap = (g0, g0 + int(rng.integers(3, 7)))
        objs.append(dict(id=k, t0=t0, x0=x0, y=y, vx=speed, vy=rng.uniform(-0.4, 0.4), bw=bw, bh=bh, gap=gap))
    dets: List[List[Box]] = []
    srcs: List[List[int]] = []
    for t in range(frames):
        boxes, ids = [], []
        for o in objs:
            if t < o["t0"]:
                continue
            x = o["x0"] + o["vx"] * (t - o["t0"])
            y = o["y"] + o["vy"] * (t - o["t0"])
            if x > w or x + o["bw"] < 0:
                continue
            if o["gap"] and o["gap"][0] <= t < o["gap"][1]:
                continue
            if rng.random() < p_miss:
                continue
            jx, jy = rng.normal(0, jitter, 2)
            boxes.append((int(x + jx), int(y + jy), int(x + jx + o["bw"]), int(y + jy + o["bh"])))
            ids.append(o["id"])
        for _ in range(rng.poisson(fp_rate)):
            fx, fy = rng.uniform(0, w - 30), rng.uniform(0.45 * h, h - 30)
            boxes.append((int(fx), int(fy), int(fx + 24), int(fy + 24)))
            ids.append(-1)
        dets.append(boxes)
        srcs.append(ids)
    return dets, srcs, n_objects


def evaluate(tracker, dets, srcs, n_objects: int, min_len: int = 12) -> Dict[str, float]:
    lookup: Dict[Tuple[int, Box], int] = {}
    for t, (boxes, ids) in enumerate(zip(dets, srcs)):
        for b, i in zip(boxes, ids):
            lookup[(t, tuple(b))] = i
        tracker.update(boxes, t)
    tracks = [t for t in tracker.all_tracks() if len(t["hist"]) >= 1]
    per_obj: Dict[int, List[Tuple[int, int]]] = defaultdict(list)  # object -> [(frame, track id)]
    purities, confirmed_true, fp_tracks = [], 0, 0
    for tr in tracks:
        srcs_of = [lookup.get((f, tuple(b)), -1) for f, _, b in tr["hist"]]
        for (f, _, _), s in zip(tr["hist"], srcs_of):
            if s >= 0:
                per_obj[s].append((f, tr["id"]))
        if len(tr["hist"]) >= min_len:
            major, cnt = Counter(srcs_of).most_common(1)[0]
            purities.append(cnt / len(srcs_of))
            if major >= 0:
                confirmed_true += 1
            else:
                fp_tracks += 1
    frags, switches = [], 0
    for o, seq in per_obj.items():
        seq.sort()
        ids = [i for _, i in seq]
        frags.append(len(set(ids)))
        switches += sum(1 for a, b in zip(ids[:-1], ids[1:]) if a != b)
    return {"frag_per_object": float(np.mean(frags)) if frags else float("nan"), "id_switches": float(switches),
            "purity": float(np.mean(purities)) if purities else float("nan"),
            "count_ratio": confirmed_true / max(n_objects, 1), "fp_tracks": float(fp_tracks)}


def run_benchmark(seeds: int = 20, size: Tuple[int, int] = (640, 360)) -> List[dict]:
    s = size[0] / 960.0
    rows: List[dict] = []
    for name, sc in SCENARIOS.items():
        acc: Dict[str, List[Dict[str, float]]] = {"greedy": [], "kalman": []}
        for seed in range(seeds):
            dets, srcs, n = simulate(seed, size=size, **sc)
            acc["greedy"].append(evaluate(CentroidTracker(80 * s, 10), dets, srcs, n))
            acc["kalman"].append(evaluate(KalmanTracker(80 * s, 10), dets, srcs, n))
        for tracker, results in acc.items():
            row = {"scenario": name, "tracker": tracker, "seeds": seeds}
            for key in ("frag_per_object", "id_switches", "purity", "count_ratio", "fp_tracks"):
                vals = np.array([r[key] for r in results], float)
                row[key] = round(float(np.nanmean(vals)), 3)
                row[key + "_std"] = round(float(np.nanstd(vals)), 3)
            rows.append(row)
    return rows


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="results/_benchmark")
    ap.add_argument("--seeds", type=int, default=20)
    a = ap.parse_args(argv)
    rows = run_benchmark(a.seeds)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "tracker_benchmark.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "scenarios.json").write_text(json.dumps(SCENARIOS, indent=2))
    for r in rows:
        print(r["scenario"], r["tracker"], "frag", r["frag_per_object"], "idsw", r["id_switches"], "purity", r["purity"],
              "count_ratio", r["count_ratio"], "fp", r["fp_tracks"])


if __name__ == "__main__":
    main()
