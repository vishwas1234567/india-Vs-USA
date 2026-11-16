"""Transforms (boxes and masks are kept consistent) including augmentations for Indian road conditions:
glare/over-exposure, dust haze, motion blur, sensor noise and low-bitrate compression."""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


class Compose:
    def __init__(self, transforms: Sequence[Callable], seed: int = 0):
        self.transforms, self.rng = list(transforms), np.random.default_rng(seed)

    def __call__(self, item: dict) -> dict:
        for t in self.transforms:
            item = t(item, self.rng)
        return item


class Letterbox:
    """Resize keeping aspect ratio and pad to a square/rect canvas; boxes, masks and lanes follow."""

    def __init__(self, size: Tuple[int, int] = (640, 640), pad_value: int = 114):
        self.w, self.h, self.pad = size[0], size[1], pad_value

    def __call__(self, item: dict, rng=None) -> dict:
        img = item["image"]
        h, w = img.shape[:2]
        r = min(self.w / w, self.h / h)
        nw, nh = int(round(w * r)), int(round(h * r))
        dx, dy = (self.w - nw) // 2, (self.h - nh) // 2
        out = np.full((self.h, self.w, 3), self.pad, np.uint8)
        out[dy:dy + nh, dx:dx + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        item = dict(item, image=out)
        if len(item.get("boxes", [])):
            b = item["boxes"].astype(np.float32) * r
            b[:, [0, 2]] += dx
            b[:, [1, 3]] += dy
            item["boxes"] = b
        if item.get("mask") is not None:
            m = np.full((self.h, self.w), 255, np.uint8)
            m[dy:dy + nh, dx:dx + nw] = cv2.resize(item["mask"], (nw, nh), interpolation=cv2.INTER_NEAREST)
            item["mask"] = m
        item["lanes"] = [np.c_[l[:, 0] * r + dx, l[:, 1] * r + dy] for l in item.get("lanes", [])]
        item["letterbox"] = (r, dx, dy)
        return item


class HFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, item: dict, rng) -> dict:
        if rng.random() >= self.p:
            return item
        img = item["image"][:, ::-1].copy()
        w = img.shape[1]
        item = dict(item, image=img)
        if len(item.get("boxes", [])):
            b = item["boxes"].copy()
            b[:, [0, 2]] = w - item["boxes"][:, [2, 0]]
            item["boxes"] = b
        if item.get("mask") is not None:
            item["mask"] = item["mask"][:, ::-1].copy()
        item["lanes"] = [np.c_[w - l[:, 0], l[:, 1]] for l in item.get("lanes", [])]
        return item


def _photometric(fn: Callable[[np.ndarray, np.random.Generator], np.ndarray], p: float):
    def apply(item: dict, rng) -> dict:
        if rng.random() >= p:
            return item
        return dict(item, image=np.clip(fn(item["image"].astype(np.float32), rng), 0, 255).astype(np.uint8))
    return apply


def BrightnessContrast(p: float = 0.5, brightness: float = 30, contrast: float = 0.25):
    return _photometric(lambda im, r: (im - 127.5) * (1 + r.uniform(-contrast, contrast)) + 127.5 + r.uniform(-brightness, brightness), p)


def GaussianNoise(p: float = 0.3, sigma: Tuple[float, float] = (2.0, 10.0)):
    return _photometric(lambda im, r: im + r.normal(0, r.uniform(*sigma), im.shape), p)


def Glare(p: float = 0.3, strength: Tuple[float, float] = (0.3, 0.8)):
    """Bright radial glare (sun / headlight): additive Gaussian blob in the upper half."""
    def fn(im, r):
        h, w = im.shape[:2]
        cx, cy = r.uniform(0.1, 0.9) * w, r.uniform(0.0, 0.5) * h
        yy, xx = np.mgrid[0:h, 0:w]
        blob = np.exp(-(((xx - cx) / (0.35 * w)) ** 2 + ((yy - cy) / (0.35 * h)) ** 2))
        return im + 255 * r.uniform(*strength) * blob[..., None]
    return _photometric(fn, p)


def DustHaze(p: float = 0.3, density: Tuple[float, float] = (0.15, 0.5)):
    """Atmospheric-scattering haze: I = J t + A (1 - t), stronger toward the horizon."""
    def fn(im, r):
        h = im.shape[0]
        t = 1 - r.uniform(*density) * np.linspace(1.0, 0.4, h)[:, None, None]
        return im * t + 220 * (1 - t)
    return _photometric(fn, p)


def MotionBlur(p: float = 0.3, length: Tuple[int, int] = (5, 15)):
    def apply(item: dict, rng) -> dict:
        if rng.random() >= p:
            return item
        k = int(rng.integers(*length))
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2, :] = 1.0 / k
        return dict(item, image=cv2.filter2D(item["image"], -1, kernel))
    return apply


def JpegCompression(p: float = 0.3, quality: Tuple[int, int] = (15, 50)):
    def apply(item: dict, rng) -> dict:
        if rng.random() >= p:
            return item
        ok, enc = cv2.imencode(".jpg", item["image"], [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(*quality))])
        return dict(item, image=cv2.imdecode(enc, cv2.IMREAD_COLOR)) if ok else item
    return apply


def indian_road_augmentation(size: Tuple[int, int] = (640, 640), seed: int = 0) -> Compose:
    """Training pipeline reflecting the conditions that hurt perception in the study: glare, haze, noise, blur, compression."""
    return Compose([HFlip(0.5), BrightnessContrast(), Glare(), DustHaze(), GaussianNoise(), MotionBlur(), JpegCompression(),
                    Letterbox(size)], seed=seed)


def to_chw_float(item: dict) -> np.ndarray:
    """BGR uint8 HxWx3 -> RGB float32 CxHxW in [0, 1] (framework-agnostic network input)."""
    return np.ascontiguousarray(item["image"][..., ::-1].transpose(2, 0, 1), np.float32) / 255.0
