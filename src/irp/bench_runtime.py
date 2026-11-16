"""CPU/GPU runtime benchmark of the perception pipeline: per-stage latency at several processing widths.

    python -m irp.bench_runtime --source data/junction2_360p.mp4 --out results/_runtime --frames 90
The device is recorded; on a machine with a GPU pass --detector yolo:<weights> --device cuda:0 to benchmark the learned path.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from .pipeline import Pipeline, PipelineConfig
from .sources import open_source


def run(source: str, widths: Sequence[int], n_frames: int, detector: str, device: str) -> dict:
    out = {"source": source, "detector": detector, "device_request": device, "python": platform.python_version(),
           "opencv": cv2.__version__, "machine": platform.machine(), "results": {}}
    for w in widths:
        frames = [f.image for f in open_source(source, max_frames=n_frames, max_side=w)]
        pipe = Pipeline(PipelineConfig(source, detector=detector, device=device, write_video=False, max_side=w))
        for i, img in enumerate(frames):
            pipe.process(img, i, i / 30.0, 30.0, sequential=True)
        s = pipe.summary()
        out["results"][str(w)] = {"frame_size": [frames[0].shape[1], frames[0].shape[0]], "device": s["device"],
                                  "latency_ms_mean": s["latency_ms_mean"], "latency_ms_p95": s["latency_ms_p95"], "fps": s["fps"],
                                  "stages_ms": {k: v["mean_ms"] for k, v in s["stages"].items()}}
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", default="results/_runtime")
    ap.add_argument("--widths", type=int, nargs="+", default=[320, 480, 640, 960])
    ap.add_argument("--frames", type=int, default=90)
    ap.add_argument("--detector", default="motion")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args(argv)
    res = run(a.source, a.widths, a.frames, a.detector, a.device)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "stage_latency.json").write_text(json.dumps(res, indent=2))
    for w, r in res["results"].items():
        print(w, r["frame_size"], f"{r['latency_ms_mean']:.1f} ms  {r['fps']:.1f} fps", {k: round(v, 1) for k, v in r["stages_ms"].items()})


if __name__ == "__main__":
    main()
