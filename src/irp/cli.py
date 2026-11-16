"""Unified command line:  python -m irp <command> [options]   (or the `irp` console script)

    run        any video / image / folder / URL  -> annotated video + metrics   (irp.pipeline)
    static     fixed-camera junction analysis                                  (irp.static)
    compare    one comparative video of two sources                            (irp.compare)
    dataset    stats | export | eval on IDD / KITTI / BDD100K / Cityscapes     (iscvd)
    forecast   train and evaluate the LSTM trajectory forecaster               (irp.predict)
    sim        closed-loop benchmarks: rl | stacks | faults                    (irp.sim)
    runtime    per-stage latency benchmark                                     (irp.bench_runtime)
    quality    video-quality profile                                           (irp.quality)
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence


def _dataset(argv: Sequence[str]) -> None:
    from iscvd import ISCVDataset, benchmark, stats
    from iscvd.export import export_coco, export_yolo

    ap = argparse.ArgumentParser(prog="irp dataset")
    ap.add_argument("action", choices=["stats", "export", "eval"])
    ap.add_argument("--format", required=True, choices=["idd", "idd_seg", "kitti", "bdd100k", "cityscapes", "yolo", "folder", "video"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="results/_dataset")
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--detector", default="auto", help="for eval: irp detector spec")
    a = ap.parse_args(argv)
    loaders = {"idd": ISCVDataset.from_idd, "idd_seg": ISCVDataset.from_idd_segmentation, "kitti": ISCVDataset.from_kitti,
               "bdd100k": ISCVDataset.from_bdd100k, "cityscapes": ISCVDataset.from_cityscapes, "yolo": ISCVDataset.from_yolo}
    if a.format == "folder":
        ds = ISCVDataset.from_folder(a.root)
    elif a.format == "video":
        ds = ISCVDataset.from_video(a.root, stride=5)
    else:
        ds = loaders[a.format](a.root, a.split)
    if a.max_images:
        ds = ds.subset(range(min(a.max_images, len(ds))))
    if a.action == "stats":
        print(json.dumps({k: v for k, v in stats.dataset_stats(ds).items() if k != "box_log10_area_frac"}, indent=2, default=float))
    elif a.action == "export":
        train, val, test = ds.split((0.8, 0.1, 0.1))
        print(export_yolo({"train": train, "val": val, "test": test}, a.out))
    else:
        from irp.detect import build_detector
        res = benchmark.evaluate_detector(benchmark.from_irp_detector(build_detector(a.detector)), ds, max_images=a.max_images)
        benchmark.save({"shared_classes": [], "results": {ds.name: {"all_classes": res, "shared_classes": None}}}, a.out)
        print(json.dumps({k: v for k, v in res.items() if k != "per_class"}, indent=2, default=float))


def main(argv: Optional[Sequence[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return
    cmd, rest = argv[0], argv[1:]
    if cmd == "run":
        from irp.pipeline import main as f
    elif cmd == "static":
        from irp.static import main as f
    elif cmd == "compare":
        from irp.compare import main as f
    elif cmd == "forecast":
        from irp.predict import main as f
    elif cmd == "sim":
        from irp.sim import main as f
    elif cmd == "runtime":
        from irp.bench_runtime import main as f
    elif cmd == "quality":
        from irp.quality import main as f
    elif cmd == "dataset":
        f = _dataset
    else:
        raise SystemExit(f"Unknown command {cmd!r}. Try: irp --help")
    f(rest)


if __name__ == "__main__":
    main()
