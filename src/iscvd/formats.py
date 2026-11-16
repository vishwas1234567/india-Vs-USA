"""Loaders that turn on-disk datasets into lists of `Sample` with unified labels.

Supported layouts (the standard published ones):
    IDD detection     <root>/{Annotations,JPEGImages,ImageSets/Main}   Pascal-VOC xml
    IDD segmentation  <root>/leftImg8bit/<split>/<seq>/*_leftImg8bit.png + gtFine/.../*_gtFine_labellevel3Ids.png
    KITTI object      <root>/<split>/{image_2,label_2}                 txt (type, trunc, occl, alpha, bbox, ...)
    BDD100K           <root>/images/100k/<split>/*.jpg + labels JSON   (box2d, poly2d lanes)
    Cityscapes        <root>/leftImg8bit/<split>/<city>/*.png + gtFine/<split>/<city>/*_gtFine_{labelIds.png,polygons.json}
    YOLO              <root>/images/<split>, <root>/labels/<split>, names from data.yaml / classes.txt
    COCO json         user-supplied json + image directory
    folder / video    unlabelled images or video frames from any vehicle camera

Loaders are tested against small synthetic fixtures that mimic these layouts (tests/test_iscvd.py); run them on the
real downloads and fix path details if your copy differs.
"""
from __future__ import annotations

import json
import re
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from .taxonomy import CLASSES, CLASS_ID, unified_id

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")


@dataclass
class Sample:
    """One image (or video frame) with unified-label annotations."""

    dataset: str
    split: str
    image: str                                   # file path, or "video.mp4#<frame>" for video frames
    width: int = 0
    height: int = 0
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))  # xyxy pixels
    labels: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.int64))     # unified ids
    raw_labels: List[str] = field(default_factory=list)
    mask: Optional[str] = None                   # semantic mask path
    mask_kind: str = ""                          # idd_level3 | cityscapes_labelid | unified
    lanes: List[np.ndarray] = field(default_factory=list)    # polylines (N,2), pixels
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def has_boxes(self) -> bool:
        return len(self.labels) > 0


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def natural_key(p) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(p))]


def image_size(path: str) -> tuple:
    """(width, height) from the file header (PNG/JPEG) without decoding; falls back to PIL."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(26)
            if head[:8] == b"\x89PNG\r\n\x1a\n":
                w, h = struct.unpack(">II", head[16:24])
                return int(w), int(h)
            if head[:2] == b"\xff\xd8":
                fh.seek(2)
                while True:
                    marker = fh.read(2)
                    if len(marker) < 2 or marker[0] != 0xFF:
                        break
                    if marker[1] in (0xC0, 0xC1, 0xC2):
                        fh.read(3)
                        h, w = struct.unpack(">HH", fh.read(4))
                        return int(w), int(h)
                    (length,) = struct.unpack(">H", fh.read(2))
                    fh.seek(length - 2, 1)
    except (OSError, struct.error):
        pass
    from PIL import Image  # fallback

    with Image.open(path) as im:
        return im.size


def _sample(dataset, split, image, boxes, labels, raw, **kw) -> Sample:
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    labels = np.asarray(labels, np.int64).reshape(-1)
    keep = labels >= 0
    s = Sample(dataset, split, str(image), boxes=boxes[keep], labels=labels[keep],
               raw_labels=[r for r, k in zip(raw, keep) if k], **kw)
    if s.width == 0 and Path(str(image).split("#")[0]).exists():
        s.width, s.height = image_size(str(image))
    return s


def _images(folder: Path, recursive: bool = False) -> List[Path]:
    it = folder.rglob("*") if recursive else folder.glob("*")
    return sorted((p for p in it if p.suffix.lower() in IMAGE_EXT), key=natural_key)


# --------------------------------------------------------------------------- #
# Indian datasets
# --------------------------------------------------------------------------- #
def load_idd_detection(root: str, split: str = "train") -> List[Sample]:
    """IDD detection (Pascal-VOC xml). `split` names a file in ImageSets/Main, else every xml is used."""
    root_p = Path(root)
    listing = root_p / "ImageSets" / "Main" / f"{split}.txt"
    if listing.exists():
        ids = [l.strip() for l in listing.read_text().splitlines() if l.strip()]
    else:
        ids = [str(p.relative_to(root_p / "Annotations").with_suffix("")).replace("\\", "/")
               for p in sorted((root_p / "Annotations").rglob("*.xml"), key=natural_key)]
    out: List[Sample] = []
    for i in ids:
        xml = root_p / "Annotations" / f"{i}.xml"
        img = next((root_p / "JPEGImages" / f"{i}{e}" for e in (".jpg", ".png", ".jpeg")
                    if (root_p / "JPEGImages" / f"{i}{e}").exists()), root_p / "JPEGImages" / f"{i}.jpg")
        if not xml.exists():
            continue
        tree = ET.parse(xml).getroot()
        size = tree.find("size")
        w = int(float(size.findtext("width", "0"))) if size is not None else 0
        h = int(float(size.findtext("height", "0"))) if size is not None else 0
        boxes, labels, raw = [], [], []
        for obj in tree.findall("object"):
            name = (obj.findtext("name") or "").strip()
            bb = obj.find("bndbox")
            if bb is None:
                continue
            boxes.append([float(bb.findtext(k, "0")) for k in ("xmin", "ymin", "xmax", "ymax")])
            labels.append(unified_id("idd", name))
            raw.append(name)
        out.append(_sample("idd", split, img, boxes, labels, raw, width=w, height=h, meta={"id": i}))
    return out


def _polygons_to_boxes(json_path: Path, dataset: str):
    """Instance boxes from a Cityscapes/IDD `*_polygons.json` (objects with label + polygon)."""
    data = json.loads(json_path.read_text())
    boxes, labels, raw = [], [], []
    for obj in data.get("objects", []):
        poly = np.asarray(obj.get("polygon", []), np.float32)
        if poly.size == 0:
            continue
        name = str(obj.get("label", ""))
        uid = unified_id(dataset, name.replace("group", ""))
        boxes.append([poly[:, 0].min(), poly[:, 1].min(), poly[:, 0].max(), poly[:, 1].max()])
        labels.append(uid)
        raw.append(name)
    return boxes, labels, raw, int(data.get("imgWidth", 0)), int(data.get("imgHeight", 0))


def load_cityscapes_like(root: str, split: str = "train", dataset: str = "cityscapes",
                         mask_suffix: str = "_gtFine_labelIds.png", mask_kind: str = "cityscapes_labelid") -> List[Sample]:
    """Cityscapes / IDD-segmentation layout: leftImg8bit + gtFine. Boxes come from the polygons json if present."""
    root_p = Path(root)
    out: List[Sample] = []
    for img in sorted((root_p / "leftImg8bit" / split).rglob("*_leftImg8bit.*"), key=natural_key):
        rel = img.relative_to(root_p / "leftImg8bit" / split)
        stem = rel.name.rsplit("_leftImg8bit", 1)[0]
        gt_dir = root_p / "gtFine" / split / rel.parent
        mask = gt_dir / f"{stem}{mask_suffix}"
        poly = gt_dir / f"{stem}_gtFine_polygons.json"
        boxes, labels, raw, w, h = ([], [], [], 0, 0)
        if poly.exists():
            boxes, labels, raw, w, h = _polygons_to_boxes(poly, dataset)
        out.append(_sample(dataset, split, img, boxes, labels, raw, width=w, height=h,
                           mask=str(mask) if mask.exists() else None, mask_kind=mask_kind,
                           meta={"sequence": str(rel.parent), "frame": stem}))
    return out


def load_idd_segmentation(root: str, split: str = "train") -> List[Sample]:
    return load_cityscapes_like(root, split, dataset="idd_seg", mask_suffix="_gtFine_labellevel3Ids.png",
                                mask_kind="idd_level3")


def load_cityscapes(root: str, split: str = "train") -> List[Sample]:
    return load_cityscapes_like(root, split, dataset="cityscapes")


# --------------------------------------------------------------------------- #
# US datasets
# --------------------------------------------------------------------------- #
def load_kitti(root: str, split: str = "training") -> List[Sample]:
    """KITTI 2D object detection: <root>/<split>/image_2 and label_2 (labels absent for the test split)."""
    root_p = Path(root)
    out: List[Sample] = []
    for img in _images(root_p / split / "image_2"):
        lab = root_p / split / "label_2" / f"{img.stem}.txt"
        boxes, labels, raw = [], [], []
        if lab.exists():
            for line in lab.read_text().splitlines():
                f = line.split()
                if len(f) < 8:
                    continue
                boxes.append([float(v) for v in f[4:8]])
                labels.append(unified_id("kitti", f[0]))
                raw.append(f[0])
        out.append(_sample("kitti", split, img, boxes, labels, raw,
                           meta={"truncated": None, "frame": img.stem}))
    return out


def load_bdd100k(root: str, split: str = "train", labels_json: Optional[str] = None) -> List[Sample]:
    """BDD100K detection (+ lane polylines in `Sample.lanes`). Images: <root>/images/100k/<split>."""
    root_p = Path(root)
    candidates = [labels_json] if labels_json else [
        root_p / "labels" / f"bdd100k_labels_images_{split}.json", root_p / "labels" / f"det_{split}.json",
        root_p / "labels" / "det_20" / f"det_{split}.json", root_p / "labels" / f"{split}.json"]
    path = next((Path(c) for c in candidates if c and Path(c).exists()), None)
    if path is None:
        raise FileNotFoundError(f"No BDD100K label json found under {root_p / 'labels'}")
    items = json.loads(path.read_text())
    out: List[Sample] = []
    for it in items:
        img = root_p / "images" / "100k" / split / it["name"]
        boxes, labels, raw, lanes = [], [], [], []
        for lab in it.get("labels", []) or []:
            cat = str(lab.get("category", ""))
            if "box2d" in lab:
                b = lab["box2d"]
                boxes.append([b["x1"], b["y1"], b["x2"], b["y2"]])
                labels.append(unified_id("bdd100k", cat))
                raw.append(cat)
            elif cat.startswith("lane") and "poly2d" in lab:
                for poly in lab["poly2d"]:
                    verts = np.asarray(poly.get("vertices", []), np.float32)
                    if len(verts) >= 2:
                        lanes.append(verts[:, :2])
        out.append(_sample("bdd100k", split, img, boxes, labels, raw, lanes=lanes,
                           width=1280, height=720, meta={"attributes": it.get("attributes", {})}))
    return out


# --------------------------------------------------------------------------- #
# Generic formats and unlabelled data
# --------------------------------------------------------------------------- #
def load_yolo(root: str, split: str = "train", names: Optional[Sequence[str]] = None) -> List[Sample]:
    """YOLO txt labels (class cx cy w h, normalised). Class names are mapped by name to unified classes."""
    root_p = Path(root)
    if names is None:
        names = CLASSES
        yaml = root_p / "data.yaml"
        if yaml.exists():
            m = re.search(r"names:\s*\[(.*?)\]", yaml.read_text(), re.S)
            if m:
                names = [n.strip().strip("'\"") for n in m.group(1).split(",") if n.strip()]
            else:
                names = re.findall(r"^\s*\d+:\s*(\S+)\s*$", yaml.read_text(), re.M) or CLASSES
    out: List[Sample] = []
    for img in _images(root_p / "images" / split):
        w, h = image_size(str(img))
        lab = root_p / "labels" / split / f"{img.stem}.txt"
        boxes, labels, raw = [], [], []
        if lab.exists():
            for line in lab.read_text().splitlines():
                f = line.split()
                if len(f) < 5:
                    continue
                c, cx, cy, bw, bh = int(f[0]), *map(float, f[1:5])
                nm = names[c] if c < len(names) else str(c)
                boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
                labels.append(CLASS_ID.get(nm, -1))
                raw.append(nm)
        out.append(_sample("yolo", split, img, boxes, labels, raw, width=w, height=h))
    return out


def load_coco(json_path: str, images_dir: str, dataset: str = "coco", split: str = "val") -> List[Sample]:
    """COCO-format json (bbox = x, y, w, h) with category names mapped through the `dataset` table."""
    data = json.loads(Path(json_path).read_text())
    cats = {c["id"]: c["name"] for c in data.get("categories", [])}
    by_img: Dict[int, list] = {}
    for a in data.get("annotations", []):
        by_img.setdefault(a["image_id"], []).append(a)
    out: List[Sample] = []
    for im in data.get("images", []):
        boxes, labels, raw = [], [], []
        for a in by_img.get(im["id"], []):
            x, y, w, h = a["bbox"]
            boxes.append([x, y, x + w, y + h])
            name = cats.get(a["category_id"], "")
            labels.append(unified_id(dataset, name))
            raw.append(name)
        out.append(_sample(dataset, split, Path(images_dir) / im["file_name"], boxes, labels, raw,
                           width=int(im.get("width", 0)), height=int(im.get("height", 0))))
    return out


def load_folder(path: str, recursive: bool = False, dataset: str = "custom", split: str = "all") -> List[Sample]:
    """Unlabelled images from a folder (any vehicle camera)."""
    return [Sample(dataset, split, str(p)) for p in _images(Path(path), recursive)]


def load_video(path: str, stride: int = 1, dataset: str = "custom", split: str = "all", max_frames: int = 0) -> List[Sample]:
    """Unlabelled video frames as samples referencing `video.mp4#<frame index>`."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    idx = range(0, n, max(stride, 1))
    if max_frames:
        idx = list(idx)[:max_frames]
    return [Sample(dataset, split, f"{path}#{i}", width=w, height=h, meta={"frame": i}) for i in idx]
