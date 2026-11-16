"""Video-quality assessment, enhancement and stabilisation for low-quality footage.

* assess_quality(): resolution, sharpness, noise, compression blockiness, exposure,
  temporal noise, inter-frame jitter and bits-per-pixel, plus heuristic flags.
* enhance_frame(): none | clahe | full (light bilateral denoise + CLAHE on luminance).
* Stabilizer: translation compensation from the static upper image region
  (buildings and sky are unaffected by traffic), gated by the phase-correlation response.

Usage:
    python -m irp.quality --source data/junction2_360p.mp4 --out results/_quality/junction2.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from .common import open_video

# --------------------------------------------------------------------------- #
# Per-frame quality measures
# --------------------------------------------------------------------------- #
_NOISE_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)


def sharpness(gray: np.ndarray, width: int = 640) -> float:
    """Variance of the Laplacian after resizing to a common width (resolution-normalised)."""
    h, w = gray.shape
    g = cv2.resize(gray, (width, max(int(h * width / w), 1)), interpolation=cv2.INTER_AREA) if w != width else gray
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def noise_sigma(gray: np.ndarray) -> float:
    """Immerkaer fast noise-standard-deviation estimate."""
    h, w = gray.shape
    conv = cv2.filter2D(gray.astype(np.float32), -1, _NOISE_KERNEL)[1:-1, 1:-1]
    return float(math.sqrt(math.pi / 2) / (6.0 * (w - 2) * (h - 2)) * np.abs(conv).sum())


def blockiness(gray: np.ndarray, block: int = 8) -> float:
    """Mean horizontal gradient on block boundaries divided by elsewhere (1.0 = no blocking)."""
    d = np.abs(np.diff(gray.astype(np.float32), axis=1))
    on_boundary = (np.arange(d.shape[1]) % block) == block - 1
    return float(d[:, on_boundary].mean() / max(d[:, ~on_boundary].mean(), 1e-6))


def temporal_noise(prev_gray: np.ndarray, gray: np.ndarray) -> float:
    """Median absolute inter-frame difference (robust to moving objects: most pixels are static)."""
    return float(np.median(np.abs(gray.astype(np.int16) - prev_gray.astype(np.int16))))


def phase_shift(ref_gray: np.ndarray, gray: np.ndarray, region_frac: float = 0.30):
    """Translation (dx, dy, response) of `gray` relative to `ref_gray` from the upper static region."""
    h, w = gray.shape
    rh = max(int(h * region_frac), 16)
    window = cv2.createHanningWindow((w, rh), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(ref_gray[:rh].astype(np.float32), gray[:rh].astype(np.float32), window)
    return float(dx), float(dy), float(resp)


# --------------------------------------------------------------------------- #
# Clip-level assessment
# --------------------------------------------------------------------------- #
def assess_quality(source: str, n_samples: int = 40) -> Dict[str, object]:
    """Quality report for a clip, from n_samples evenly spaced frame pairs."""
    cap, info = open_video(source)
    idxs = np.unique(np.linspace(0, max(info.frames - 2, 0), n_samples).astype(int))
    sharp: List[float] = []
    noise: List[float] = []
    block: List[float] = []
    bright: List[float] = []
    contrast: List[float] = []
    hi: List[float] = []
    lo: List[float] = []
    tnoise: List[float] = []
    jitter: List[float] = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok1, f1 = cap.read()
        ok2, f2 = cap.read()
        if not (ok1 and ok2):
            continue
        g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
        sharp.append(sharpness(g1))
        noise.append(noise_sigma(g1))
        block.append(blockiness(g1))
        bright.append(float(g1.mean()))
        contrast.append(float(g1.std()))
        hi.append(float((g1 >= 250).mean()))
        lo.append(float((g1 <= 5).mean()))
        tnoise.append(temporal_noise(g1, g2))
        dx, dy, resp = phase_shift(g1, g2)
        if resp > 0.05:
            jitter.append(math.hypot(dx, dy))
    cap.release()
    if not sharp:
        raise RuntimeError("Could not read frames for quality assessment.")
    path = Path(source)
    size = path.stat().st_size if path.exists() else 0
    med = lambda v: round(float(np.median(v)), 4)
    report: Dict[str, object] = {
        "source": str(source), "width": info.width, "height": info.height, "fps": info.fps,
        "frames": info.frames, "duration_s": round(info.frames / info.fps, 1),
        "file_mb": round(size / 1e6, 2),
        "bits_per_pixel": round(size * 8 / max(info.frames * info.width * info.height, 1), 4),
        "sharpness_lap_var_640": med(sharp), "noise_sigma": med(noise), "blockiness": med(block),
        "brightness_mean": med(bright), "contrast_std": med(contrast),
        "overexposed_frac": med(hi), "underexposed_frac": med(lo),
        "temporal_noise_median_absdiff": med(tnoise),
        "jitter_px_rms": round(float(np.sqrt(np.mean(np.square(jitter)))), 4) if jitter else None,
        "samples": len(sharp),
    }
    report["flags"] = quality_flags(report)
    return report


def quality_flags(r: Dict[str, object]) -> List[str]:
    """Heuristic flags (thresholds are rules of thumb, not validated standards)."""
    flags = []
    if int(r["height"]) <= 360:
        flags.append("low_resolution")
    if float(r["blockiness"]) > 1.08:
        flags.append("block_artifacts")
    if float(r["noise_sigma"]) > 3.0:
        flags.append("noisy")
    if float(r["sharpness_lap_var_640"]) < 40.0:
        flags.append("soft_or_blurred")
    if float(r["overexposed_frac"]) > 0.02:
        flags.append("glare_or_clipping")
    if r.get("jitter_px_rms") is not None and float(r["jitter_px_rms"]) > 1.0:
        flags.append("camera_jitter")
    return flags


# --------------------------------------------------------------------------- #
# Enhancement and stabilisation
# --------------------------------------------------------------------------- #
ENHANCE_MODES = ("none", "clahe", "full")
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def enhance_frame(bgr: np.ndarray, mode: str = "none") -> np.ndarray:
    """none: identity; clahe: local contrast on luminance; full: light bilateral denoise + clahe."""
    if mode not in ENHANCE_MODES:
        raise ValueError(f"Unknown enhance mode {mode!r}; choose from {ENHANCE_MODES}")
    if mode == "none":
        return bgr
    out = cv2.bilateralFilter(bgr, 5, 25, 5) if mode == "full" else bgr
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    lab[..., 0] = _CLAHE.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


class Stabilizer:
    """Chained translation stabiliser. Shift is estimated frame-to-frame on the static upper
    region, accumulated (clamped to +-max_shift px) and removed from every frame."""

    def __init__(self, max_shift: float = 15.0, min_response: float = 0.05):
        self.max_shift, self.min_response = max_shift, min_response
        self._prev: Optional[np.ndarray] = None
        self._cum = np.zeros(2)
        self.history: List[float] = []

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if self._prev is not None:
            dx, dy, resp = phase_shift(self._prev, gray)
            if resp >= self.min_response:
                self._cum = np.clip(self._cum + np.array([dx, dy]), -self.max_shift, self.max_shift)
        self._prev = gray
        self.history.append(float(np.hypot(*self._cum)))
        if not np.any(self._cum):
            return bgr
        m = np.float32([[1, 0, -self._cum[0]], [0, 1, -self._cum[1]]])
        return cv2.warpAffine(bgr, m, (bgr.shape[1], bgr.shape[0]), borderMode=cv2.BORDER_REPLICATE)

    def summary(self) -> Dict[str, float]:
        h = np.array(self.history) if self.history else np.zeros(1)
        return {"mean_correction_px": round(float(h.mean()), 3), "max_correction_px": round(float(h.max()), 3)}


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--samples", type=int, default=40)
    a = ap.parse_args(argv)
    report = assess_quality(a.source, a.samples)
    text = json.dumps(report, indent=2)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
