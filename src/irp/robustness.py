"""Robustness to video degradation: how far do the scene metrics move when quality drops?

A high-quality reference clip is degraded synthetically (resolution, blur, noise, H.264
compression, and a combined "low-quality camera" condition). Each degraded version is analysed
with and without enhancement and compared with the analysis of the original clip. No manual
labels are needed: the reference analysis is the target (reference-based evaluation).

Agreement measures (degraded vs reference):
  occ_pearson      correlation of the per-frame road-occupancy series
  occ_mae_pp       mean absolute occupancy difference (percentage points)
  occ_mean_ratio   ratio of mean occupancies
  tracks_ratio     ratio of confirmed track counts
  direction_l1     L1 distance between normalised direction histograms (0 = identical, 2 = disjoint)

Requires ffmpeg with libx264. Usage:
    python -m irp.robustness --source data/junction_clip.mp4 --out results/_robustness
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .static import StaticConfig, analyze

log = logging.getLogger("irp.robustness")

CONDITIONS: Dict[str, dict] = {
    "res360": {"scale": (640, 360)},
    "res270": {"scale": (480, 270)},
    "res180": {"scale": (320, 180)},
    "blur2": {"blur": 2.0},
    "noise10": {"noise": 10},
    "crf35": {"crf": 35},
    "crf45": {"crf": 45},
    "combo": {"scale": (640, 360), "noise": 6, "crf": 40},
}
DIRECTIONS = ("right->left", "left->right", "toward camera", "away from camera")


def ffmpeg_filters(spec: dict) -> str:
    filters: List[str] = []
    if "blur" in spec:
        filters.append(f"gblur=sigma={spec['blur']}")
    if "scale" in spec:
        filters.append(f"scale={spec['scale'][0]}:{spec['scale'][1]}:flags=area")
    if "noise" in spec:
        filters.append(f"noise=alls={spec['noise']}:allf=t:all_seed=1")
    return ",".join(filters) if filters else "null"


def degrade(src: str, dst: str, spec: dict) -> None:
    """Re-encode `src` to `dst` with the degradations in `spec` (H.264, no audio)."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for the degradation study.")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-vf", ffmpeg_filters(spec), "-an",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", str(spec.get("crf", 18)), "-pix_fmt", "yuv420p", dst]
    subprocess.run(cmd, check=True)


def _occupancy(run_dir: Path) -> Dict[int, float]:
    with open(run_dir / "frames.csv") as fh:
        return {int(r["frame"]): float(r["road_occupancy"]) for r in csv.DictReader(fh) if r["warmup"] == "0"}


def _direction_vector(summary: dict) -> np.ndarray:
    v = np.array([summary["direction_counts"].get(d, 0) for d in DIRECTIONS], float)
    return v / v.sum() if v.sum() > 0 else v


def agreement(ref_dir: Path, run_dir: Path) -> Dict[str, Optional[float]]:
    """Agreement metrics of one analysed run against the reference analysis."""
    ref_s = json.loads((ref_dir / "summary.json").read_text())
    run_s = json.loads((run_dir / "summary.json").read_text())
    ro, qo = _occupancy(ref_dir), _occupancy(run_dir)
    common = sorted(set(ro) & set(qo))
    a, b = np.array([ro[i] for i in common]), np.array([qo[i] for i in common])
    pearson = float(np.corrcoef(a, b)[0, 1]) if len(common) > 2 and a.std() > 0 and b.std() > 0 else None
    return {
        "occ_pearson": None if pearson is None else round(pearson, 4),
        "occ_mae_pp": round(float(np.abs(a - b).mean() * 100), 3),
        "occ_mean_ratio": round(run_s["occupancy_mean"] / ref_s["occupancy_mean"], 3),
        "tracks_ratio": round(run_s["moving_tracks_confirmed"] / max(ref_s["moving_tracks_confirmed"], 1), 3),
        "direction_l1": round(float(np.abs(_direction_vector(ref_s) - _direction_vector(run_s)).sum()), 3),
        "tracks": run_s["moving_tracks_confirmed"], "occ_mean_pct": round(run_s["occupancy_mean"] * 100, 2),
        "frames_compared": len(common),
    }


def run_study(source: str, out_dir: str, tracker: str = "kalman", conditions: Optional[Sequence[str]] = None,
              modes: Sequence[str] = ("none", "full"), resume: bool = True) -> List[dict]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = list(conditions or CONDITIONS)
    ref_dir = out / "runs" / "reference"
    if not (resume and (ref_dir / "summary.json").exists()):
        analyze(StaticConfig(source=source, out=str(ref_dir), tracker=tracker, write_video=False,
                             label="Reference (original clip)"))
    rows: List[dict] = []
    for name in names:
        spec = CONDITIONS[name]
        todo = [m for m in modes if not (resume and (out / "runs" / f"{name}_{m}" / "summary.json").exists())]
        if todo:
            with tempfile.TemporaryDirectory() as td:
                clip = str(Path(td) / f"{name}.mp4")
                degrade(source, clip, spec)
                for m in todo:
                    log.info("analysing %s with enhance=%s", name, m)
                    analyze(StaticConfig(source=clip, out=str(out / "runs" / f"{name}_{m}"), tracker=tracker,
                                         enhance=m, write_video=False, label=f"{name} / enhance={m}"))
    # table covers every condition analysed so far, so the study can be run in several chunks
    for name in CONDITIONS:
        for m in modes:
            run_dir = out / "runs" / f"{name}_{m}"
            if (run_dir / "summary.json").exists():
                rows.append({"condition": name, "enhance": m, **agreement(ref_dir, run_dir)})
    ref_s = json.loads((ref_dir / "summary.json").read_text())
    (out / "reference.json").write_text(json.dumps({"tracks": ref_s["moving_tracks_confirmed"],
                                                    "occ_mean_pct": round(ref_s["occupancy_mean"] * 100, 2),
                                                    "tracker": tracker, "source": source}, indent=2))
    with open(out / "robustness.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="high-quality reference clip")
    ap.add_argument("--out", default="results/_robustness")
    ap.add_argument("--tracker", choices=["greedy", "kalman"], default="kalman")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--modes", default="none,full")
    ap.add_argument("--no-resume", action="store_true")
    a = ap.parse_args(argv)
    rows = run_study(a.source, a.out, a.tracker, a.conditions.split(","), a.modes.split(","), not a.no_resume)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
