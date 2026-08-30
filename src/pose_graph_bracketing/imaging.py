"""Raw image loading, Bayer demosaicing, and cropping.

The pipeline is grayscale end-to-end: `demosaic` converts straight from the
raw Bayer mosaic to a single-channel image (OpenCV's BayerXX2GRAY, not a
color demosaic followed by a BGR2GRAY reduction), and every preprocessing
step downstream (CLAHE/equalization, CRF radiance correction, feature
extraction) operates on that 2D uint8 array. Color is only ever
reconstructed transiently for on-screen visualization overlays (see
visualization.py) -- never stored or fed to a feature extractor.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# Vendor Bayer-pattern name -> OpenCV demosaic-to-grayscale code. Note the
# R/B swap versus naive expectation: OpenCV's BayerXX2GRAY codes describe the
# pattern of the *second* row/col, so RGGB (row0=RG, row1=GB) maps to
# COLOR_BayerBG2GRAY.
_BAYER_CODES = {
    "RGGB": cv2.COLOR_BayerBG2GRAY,
    "BGGR": cv2.COLOR_BayerRG2GRAY,
    "GRBG": cv2.COLOR_BayerGB2GRAY,
    "GBRG": cv2.COLOR_BayerGR2GRAY,
}


def load_raw(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def demosaic(raw: np.ndarray, bayer_pattern: str = "RGGB") -> np.ndarray:
    """Demosaic a single-channel Bayer raw image directly into grayscale.

    `bayer_pattern="none"` returns the raw image as-is if already
    single-channel, or reduced to grayscale via BGR2GRAY if it's already
    color (for pre-demosaiced/color inputs).
    """
    pattern = bayer_pattern.upper()
    if pattern == "NONE":
        if raw.ndim == 2:
            return raw
        return cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
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
    clahe_method: str = "clahe",
) -> np.ndarray:
    """Load + demosaic-to-gray + crop in one step. Returns a single-channel
    uint8 grayscale image.

    `clahe_enabled` applies histogram equalization (see `apply_clahe`) after
    cropping -- confirmed to substantially reduce bracketed-exposure
    oscillation (frame-to-frame instability from exposure non-uniformity
    between SAE/MAE/LAE), see docs/cycle_bias_findings.md.
    """
    raw = load_raw(path)
    gray = demosaic(raw, bayer_pattern)
    gray = crop_bottom(gray, crop_bottom_px)
    if clahe_enabled:
        gray = apply_clahe(gray, clahe_clip_limit, clahe_tile_grid_size, clahe_method)
    return gray


def apply_clahe(
    image_gray: np.ndarray,
    clip_limit: float = 10.0,
    tile_grid_size: int = 8,
    method: str = "clahe",
) -> np.ndarray:
    """Apply local (CLAHE) or global histogram equalization directly to a
    single-channel grayscale image.

    `method="clahe"` (default): tiled, contrast-limited equalization
    (`cv2.createCLAHE`). `method="global"`: plain whole-image
    `cv2.equalizeHist` -- no tiling/clipping, so a single dominant histogram
    bin (e.g. a large saturated region) can dominate the redistribution more
    than CLAHE's clip limit allows; `clip_limit`/`tile_grid_size` are then
    ignored.

    (A masked variant that excluded saturated/underexposed pixels from the
    histogram was tried and reverted: it's a per-frame-adaptive stretch --
    each exposure bracket's own valid-pixel population is maximally
    stretched to fill 0-255 independently, so the same real-world radiance
    maps to a different output intensity depending on which bracket
    produced it, actively hurting cross-exposure consistency. See
    radiance.py's normalize_for_matching_fixed docstring for the same
    failure mode diagnosed and fixed, there, with a shared/fixed window --
    if this equalizer needs revisiting, that's the fix to reach for, not
    saturation masking.)
    """
    if method == "global":
        return cv2.equalizeHist(image_gray)
    elif method == "clahe":
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
        return clahe.apply(image_gray)
    else:
        raise ValueError(f"Unknown equalization method: {method}")


def apply_gaussian_blur(image_bgr: np.ndarray, ksize: int = 5, sigma: float = 0.0) -> np.ndarray:
    k = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.GaussianBlur(image_bgr, (k, k), sigma)
