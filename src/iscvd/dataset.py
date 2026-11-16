"""ISCVDataset: one reusable dataset object for Indian and US driving data.

    ds = ISCVDataset.from_idd("IDD_Detection", "train")        # or from_kitti / from_bdd100k / from_cityscapes ...
    ds = ISCVDataset.from_video("drive.mp4", stride=5)          # any footage from a vehicle camera
    train, val, test = ds.split((0.8, 0.1, 0.1), seed=0)       # grouped by sequence to avoid leakage
    item = ds[0]                                                # {"image", "boxes", "labels", "mask", "meta"}
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from . import formats
from .formats import Sample
from .taxonomy import (CITYSCAPES_LABELID_TO_SEMANTIC, CLASSES, IDD_LEVEL3_TO_SEMANTIC, classes_present,
                       semantic_lut)

_VIDEO_CACHE: Dict[str, cv2.VideoCapture] = {}


def read_frame(ref: str) -> np.ndarray:
    """Read an image path or a 'video.mp4#<frame>' reference as a BGR array."""
    if "#" in ref and Path(ref.split("#")[0]).suffix.lower() in formats.VIDEO_EXT:
        path, idx = ref.rsplit("#", 1)
        cap = _VIDEO_CACHE.get(path)
        if cap is None:
            cap = _VIDEO_CACHE[path] = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            raise IOError(f"Cannot read frame {idx} of {path}")
        return frame
    img = cv2.imread(ref, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Cannot read image: {ref}")
    return img


class ISCVDataset:
    def __init__(self, samples: Sequence[Sample], name: str = "dataset", transform: Optional[Callable] = None):
        self.samples: List[Sample] = list(samples)
        self.name = name
        self.transform = transform

    # ---- constructors
    @classmethod
    def from_idd(cls, root, split="train"):
        return cls(formats.load_idd_detection(root, split), f"idd-{split}")

    @classmethod
    def from_idd_segmentation(cls, root, split="train"):
        return cls(formats.load_idd_segmentation(root, split), f"idd_seg-{split}")

    @classmethod
    def from_kitti(cls, root, split="training"):
        return cls(formats.load_kitti(root, split), f"kitti-{split}")

    @classmethod
    def from_bdd100k(cls, root, split="train", labels_json=None):
        return cls(formats.load_bdd100k(root, split, labels_json), f"bdd100k-{split}")

    @classmethod
    def from_cityscapes(cls, root, split="train"):
        return cls(formats.load_cityscapes(root, split), f"cityscapes-{split}")

    @classmethod
    def from_yolo(cls, root, split="train", names=None):
        return cls(formats.load_yolo(root, split, names), f"yolo-{split}")

    @classmethod
    def from_coco(cls, json_path, images_dir, dataset="coco", split="val"):
        return cls(formats.load_coco(json_path, images_dir, dataset, split), f"{dataset}-{split}")

    @classmethod
    def from_folder(cls, path, recursive=False):
        return cls(formats.load_folder(path, recursive), Path(path).name)

    @classmethod
    def from_video(cls, path, stride=1, max_frames=0):
        return cls(formats.load_video(path, stride, max_frames=max_frames), Path(path).stem)

    # ---- access
    def __len__(self) -> int:
        return len(self.samples)

    def read_image(self, i: int) -> np.ndarray:
        return read_frame(self.samples[i].image)

    def read_semantic(self, i: int) -> Optional[np.ndarray]:
        """Unified semantic mask (0 drivable, 1 non-drivable ground, 2 other, 255 ignore) or None."""
        s = self.samples[i]
        if not s.mask:
            return None
        raw = cv2.imread(s.mask, cv2.IMREAD_UNCHANGED)
        if raw is None:
            return None
        if raw.ndim == 3:
            raw = raw[..., 0]
        if s.mask_kind == "idd_level3":
            lut = semantic_lut(IDD_LEVEL3_TO_SEMANTIC)
        elif s.mask_kind == "cityscapes_labelid":
            lut = semantic_lut(CITYSCAPES_LABELID_TO_SEMANTIC)
        else:
            return raw.astype(np.uint8)
        return np.asarray(lut, np.uint8)[raw.astype(np.uint8)]

    def __getitem__(self, i: int) -> dict:
        s = self.samples[i]
        item = {"image": self.read_image(i), "boxes": s.boxes.copy(), "labels": s.labels.copy(),
                "mask": self.read_semantic(i), "lanes": [l.copy() for l in s.lanes],
                "meta": {"dataset": s.dataset, "split": s.split, "path": s.image, **s.meta}}
        return self.transform(item) if self.transform else item

    def __iter__(self) -> Iterator[dict]:
        for i in range(len(self)):
            yield self[i]

    def batches(self, batch_size: int) -> Iterator[List[dict]]:
        for i in range(0, len(self), batch_size):
            yield [self[j] for j in range(i, min(i + batch_size, len(self)))]

    # ---- subsets and splits
    def subset(self, indices: Sequence[int], name: Optional[str] = None) -> "ISCVDataset":
        return ISCVDataset([self.samples[i] for i in indices], name or self.name, self.transform)

    def filter(self, fn: Callable[[Sample], bool]) -> "ISCVDataset":
        return ISCVDataset([s for s in self.samples if fn(s)], self.name, self.transform)

    def sample(self, n: int, seed: int = 0) -> "ISCVDataset":
        rng = np.random.default_rng(seed)
        idx = sorted(rng.choice(len(self), size=min(n, len(self)), replace=False).tolist())
        return self.subset(idx)

    def split(self, fractions: Sequence[float] = (0.8, 0.1, 0.1), seed: int = 0) -> Tuple["ISCVDataset", ...]:
        """Split grouped by sequence (meta['sequence'], else the file's parent folder) so near-duplicate
        consecutive frames never straddle train and test."""
        groups: Dict[str, List[int]] = {}
        for i, s in enumerate(self.samples):
            key = str(s.meta.get("sequence") or Path(s.image.split("#")[0]).parent)
            groups.setdefault(key, []).append(i)
        keys = sorted(groups)
        np.random.default_rng(seed).shuffle(keys)
        total = sum(fractions)
        bounds = np.cumsum([f / total for f in fractions]) * len(keys)
        parts: List[List[int]] = [[] for _ in fractions]
        for rank, k in enumerate(keys):
            p = int(np.searchsorted(bounds, rank + 0.5))
            parts[min(p, len(parts) - 1)].extend(groups[k])
        return tuple(self.subset(sorted(p), f"{self.name}[{n}]") for p, n in zip(parts, range(len(parts))))

    # ---- summaries
    def class_counts(self) -> Dict[str, int]:
        c = Counter()
        for s in self.samples:
            c.update(CLASSES[l] for l in s.labels)
        return {k: c.get(k, 0) for k in CLASSES}

    def source_dataset(self) -> str:
        return self.samples[0].dataset if self.samples else self.name

    def classes_present(self) -> List[str]:
        return classes_present(self.source_dataset())

    def ground_truth(self) -> List[dict]:
        return [{"boxes": s.boxes, "labels": s.labels} for s in self.samples]
