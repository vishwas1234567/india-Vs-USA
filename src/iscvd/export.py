"""Export a dataset to the formats used by common trainers (YOLO txt + data.yaml, COCO json)."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Sequence

import cv2
import numpy as np

from .dataset import ISCVDataset, read_frame
from .taxonomy import CLASSES


def export_yolo(splits: Dict[str, ISCVDataset], out_dir: str, link: str = "copy", classes: Sequence[str] = CLASSES) -> Path:
    """Write images/<split>, labels/<split> and data.yaml. link: copy | symlink | decode (re-encode frames, needed for video)."""
    out = Path(out_dir)
    for split, ds in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i, s in enumerate(ds.samples):
            stem = f"{Path(s.image.split('#')[0]).stem}_{i:06d}"
            dst = out / "images" / split / f"{stem}.jpg"
            if "#" in s.image or link == "decode":
                cv2.imwrite(str(dst), read_frame(s.image))
            elif link == "symlink":
                if not dst.exists():
                    os.symlink(os.path.abspath(s.image), dst)
            else:
                shutil.copy(s.image, dst)
            w, h = s.width, s.height
            if not (w and h):
                h, w = read_frame(s.image).shape[:2]
            lines = []
            for (x1, y1, x2, y2), l in zip(s.boxes, s.labels):
                cx, cy, bw, bh = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h
                lines.append(f"{int(l)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    names = ", ".join(f"'{c}'" for c in classes)
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\n" + "".join(f"{s}: images/{s}\n" for s in splits) + f"nc: {len(classes)}\nnames: [{names}]\n")
    return out / "data.yaml"


def export_coco(ds: ISCVDataset, json_path: str, classes: Sequence[str] = CLASSES) -> Path:
    images, anns = [], []
    for i, s in enumerate(ds.samples):
        w, h = s.width, s.height
        if not (w and h):
            h, w = read_frame(s.image).shape[:2]
        images.append({"id": i, "file_name": Path(s.image.split("#")[0]).name, "width": int(w), "height": int(h)})
        for (x1, y1, x2, y2), l in zip(s.boxes, s.labels):
            anns.append({"id": len(anns), "image_id": i, "category_id": int(l), "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                         "area": float((x2 - x1) * (y2 - y1)), "iscrowd": 0})
    doc = {"images": images, "annotations": anns, "categories": [{"id": i, "name": c} for i, c in enumerate(classes)]}
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))
    return path
