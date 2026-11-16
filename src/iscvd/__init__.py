"""iscvd - Indian Self-driving Car Vision Dataset toolkit.

Reusable loading, preprocessing, augmentation, export, evaluation and benchmarking for Indian (IDD) and US
(KITTI, BDD100K, Cityscapes) driving datasets, plus any footage from a vehicle camera, under one taxonomy.
"""
from .dataset import ISCVDataset, read_frame
from .formats import Sample
from .taxonomy import CLASSES, CLASS_ID, shared_classes, unify, unified_id

__version__ = "1.0.0"
__all__ = ["ISCVDataset", "Sample", "CLASSES", "CLASS_ID", "unify", "unified_id", "shared_classes", "read_frame"]
