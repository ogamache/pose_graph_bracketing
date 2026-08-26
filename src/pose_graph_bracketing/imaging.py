"""Raw image loading, Bayer demosaicing, and cropping."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# Vendor Bayer-pattern name -> OpenCV demosaic code. Note the R/B swap versus
# naive expectation: OpenCV's BayerXX2BGR codes describe the pattern of the
# *second* row/col, so RGGB (row0=RG, row1=GB) maps to COLOR_BayerBG2BGR.
_BAYER_CODES = {
    "RGGB": cv2.COLOR_BayerBG2BGR,
    "BGGR": cv2.COLOR_BayerRG2BGR,
    "GRBG": cv2.COLOR_BayerGB2BGR,
    "GBRG": cv2.COLOR_BayerGR2BGR,
}


def load_raw(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def demosaic(raw: np.ndarray, bayer_pattern: str = "RGGB") -> np.ndarray:
    """Demosaic a single-channel Bayer raw image into a BGR image.

    `bayer_pattern="none"` returns the raw image converted to 3-channel BGR
    as-is (for already-demosaiced/grayscale inputs).
    """
    pattern = bayer_pattern.upper()
    if pattern == "NONE":
        if raw.ndim == 2:
            return cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        return raw
    if pattern not in _BAYER_CODES:
        raise ValueError(f"Unknown bayer_pattern: {bayer_pattern}")
    return cv2.cvtColor(raw, _BAYER_CODES[pattern])


def crop_bottom(image: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return image
    return image[:-px]


def load_preprocessed(
    path: str | Path,
    bayer_pattern: str = "RGGB",
    crop_bottom_px: int = 0,
    clahe_enabled: bool = False,
    clahe_clip_limit: float = 10.0,
    clahe_tile_grid_size: int = 8,
) -> np.ndarray:
    """Load + demosaic + crop in one step. Returns a BGR uint8 image.

    `clahe_enabled` applies local contrast normalization (see `apply_clahe`)
    after cropping -- confirmed to substantially reduce bracketed-exposure
    oscillation (frame-to-frame instability from exposure non-uniformity
    between SAE/MAE/LAE), see docs/cycle_bias_findings.md.
    """
    raw = load_raw(path)
    bgr = demosaic(raw, bayer_pattern)
    bgr = crop_bottom(bgr, crop_bottom_px)
    if clahe_enabled:
        bgr = apply_clahe(bgr, clahe_clip_limit, clahe_tile_grid_size)
    return bgr


def apply_clahe(image_bgr: np.ndarray, clip_limit: float = 10.0, tile_grid_size: int = 8) -> np.ndarray:
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def apply_gaussian_blur(image_bgr: np.ndarray, ksize: int = 5, sigma: float = 0.0) -> np.ndarray:
    k = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.GaussianBlur(image_bgr, (k, k), sigma)
