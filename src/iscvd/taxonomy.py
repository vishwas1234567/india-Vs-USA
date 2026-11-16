"""Unified taxonomy so Indian and US datasets are evaluated with identical classes.

Unified detection classes (ids are the list positions):
    car, heavy, two_wheeler, pedestrian, auto, animal, sign

Design decisions (documented because they affect every cross-dataset number):
  * `rider` is ignored: bike and motorcycle boxes are kept as two_wheeler, so the rider is not counted twice.
    (KITTI `Cyclist` boxes cover rider + bicycle and map to two_wheeler.)
  * `auto` (auto-rickshaw) exists only in Indian data. Cross-dataset comparisons therefore also report the
    classes shared by all compared datasets (see `shared_classes`).
  * Mappings follow the published label sets of each dataset. They are plain dictionaries: verify them against
    the label definitions of the version you downloaded and edit if needed.

Unified semantic (road-understanding) labels: 0 drivable, 1 non-drivable ground (sidewalk etc.), 2 other, 255 ignore.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

CLASSES: List[str] = ["car", "heavy", "two_wheeler", "pedestrian", "auto", "animal", "sign"]
CLASS_ID: Dict[str, int] = {c: i for i, c in enumerate(CLASSES)}

SEMANTIC_NAMES = {0: "drivable", 1: "non_drivable_ground", 2: "other", 255: "ignore"}

# Raw label (lower case) -> unified class (None = ignored)
MAPPINGS: Dict[str, Dict[str, Optional[str]]] = {
    "idd": {
        "car": "car", "vehicle fallback": "car", "truck": "heavy", "bus": "heavy", "caravan": "heavy",
        "trailer": "heavy", "train": "heavy", "motorcycle": "two_wheeler", "bicycle": "two_wheeler",
        "person": "pedestrian", "autorickshaw": "auto", "animal": "animal", "traffic sign": "sign",
        "traffic light": "sign", "rider": None,
    },
    "kitti": {
        "car": "car", "van": "car", "truck": "heavy", "tram": "heavy", "pedestrian": "pedestrian",
        "person_sitting": "pedestrian", "cyclist": "two_wheeler", "misc": None, "dontcare": None,
    },
    "bdd100k": {
        "car": "car", "truck": "heavy", "bus": "heavy", "train": "heavy", "person": "pedestrian",
        "bike": "two_wheeler", "motor": "two_wheeler", "traffic light": "sign", "traffic sign": "sign",
        "rider": None,
    },
    "cityscapes": {
        "car": "car", "truck": "heavy", "bus": "heavy", "caravan": "heavy", "trailer": "heavy", "train": "heavy",
        "person": "pedestrian", "motorcycle": "two_wheeler", "bicycle": "two_wheeler", "traffic sign": "sign",
        "traffic light": "sign", "rider": None,
    },
    "coco": {
        "car": "car", "truck": "heavy", "bus": "heavy", "train": "heavy", "person": "pedestrian",
        "bicycle": "two_wheeler", "motorcycle": "two_wheeler", "cow": "animal", "dog": "animal",
        "horse": "animal", "sheep": "animal", "traffic light": "sign", "stop sign": "sign",
    },
    "unified": {c: c for c in CLASSES},
}
MAPPINGS["idd_seg"] = MAPPINGS["idd"]


def _norm(name: str) -> str:
    return name.lower().replace("_", " ").strip()


def unify(dataset: str, raw: str) -> Optional[str]:
    """Unified class for a raw label of `dataset` (None if ignored or unknown)."""
    table = MAPPINGS.get(dataset.lower(), MAPPINGS["unified"])
    return {_norm(k): v for k, v in table.items()}.get(_norm(raw))


def unified_id(dataset: str, raw: str) -> int:
    """Unified class id, or -1 if the label is ignored/unknown."""
    name = unify(dataset, raw)
    return CLASS_ID[name] if name is not None else -1


def classes_present(dataset: str) -> List[str]:
    """Unified classes that a dataset can ever produce."""
    table = MAPPINGS.get(dataset.lower(), MAPPINGS["unified"])
    return [c for c in CLASSES if c in set(v for v in table.values() if v)]


def shared_classes(datasets: Iterable[str]) -> List[str]:
    """Unified classes available in every listed dataset (the fair set for cross-dataset comparison)."""
    sets = [set(classes_present(d)) for d in datasets]
    common = set.intersection(*sets) if sets else set()
    return [c for c in CLASSES if c in common]


# Semantic label-id tables -> unified semantic labels
IDD_LEVEL3_TO_SEMANTIC: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 19: 1}  # road, parking, drivable fallback | sidewalk...
CITYSCAPES_LABELID_TO_SEMANTIC: Dict[int, int] = {7: 0, 9: 0, 8: 1, 10: 1}               # road, parking | sidewalk, rail track


def semantic_lut(table: Dict[int, int], default: int = 2) -> "list[int]":
    """256-entry lookup table mapping raw label ids to unified semantic labels (255 stays ignore)."""
    lut = [default] * 256
    for k, v in table.items():
        lut[k] = v
    lut[255] = 255
    return lut
