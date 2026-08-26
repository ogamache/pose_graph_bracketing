"""Convert a demosaiced BGR frame + exposure/gain metadata into a
gain/exposure-invariant radiance domain, so frames from different exposure
brackets (SAE/MAE/LAE) become directly comparable for feature matching.

Ported from radiance_bracketing_tracking/radiance/{radiance_map.py,crf.py}.
Used by refine.py to build the radiance-normalized images its LK-based
sub-pixel alignment tracks in.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

SATURATION_HIGH = 254  # >= this raw value is considered saturated
UNDEREXPOSED_MARGIN = 2  # raw value <= black_level + margin is considered underexposed
LOG_EPS = 1e-4


@dataclass
class CameraCRF:
    g: np.ndarray  # shape (256,), g[Z] = ln(irradiance*dt) response, per-pixel-independent
    black_level: float
    weights: np.ndarray  # shape (256,), Debevec weighting function used


def load_crf(path: str) -> CameraCRF:
    """Load a CRF fitted by radiance_bracketing_tracking/scripts/estimate_crf.py.

    `path` may be either the crf.npz file itself or its containing directory.
    """
    npz_path = path if os.path.isfile(path) else os.path.join(path, "crf.npz")
    data = np.load(npz_path)
    return CameraCRF(g=data["g"], black_level=float(data["black_level"]), weights=data["weights"])


def valid_pixel_mask(gray: np.ndarray, black_level: float = 0.0) -> np.ndarray:
    low = black_level + UNDEREXPOSED_MARGIN
    return (gray > low) & (gray < SATURATION_HIGH)


def gain_db_to_linear(gain_db: float) -> float:
    """Basler-style dB gain (voltage/amplitude convention): factor = 10^(dB/20)."""
    return 10.0 ** (gain_db / 20.0)


def to_radiance(
    img: np.ndarray,
    exposure_us: float,
    mode: str = "linear",
    crf: CameraCRF | None = None,
    black_level: float = 0.0,
    gain_db: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert an 8-bit image (any number of channels) to float32 radiance.

    mode="crf": ln(radiance) = g(Z) - ln(dt) - ln(gain_linear), using a
      CameraCRF fitted at a fixed reference gain.
    mode="linear": radiance = (Z - black_level) / (dt * gain_linear).

    Returns (radiance, valid_mask); radiance is scaled up to the overall
    constant ambiguity inherent to both methods -- only relative/log
    differences across the bracket matter for matching.
    """
    mask = valid_pixel_mask(img, black_level=black_level)
    gain_linear = gain_db_to_linear(gain_db)

    if mode == "crf":
        if crf is None:
            raise ValueError("mode='crf' requires a CameraCRF")
        log_dt = np.log(exposure_us)
        log_gain = np.log(gain_linear)
        log_radiance = crf.g[img.astype(np.int64)] - log_dt - log_gain
        radiance = np.exp(log_radiance).astype(np.float32)
    elif mode == "linear":
        radiance = np.clip(img.astype(np.float32) - black_level, 0.0, None) / float(exposure_us * gain_linear)
    else:
        raise ValueError(f"Unknown radiance mode: {mode}")

    return radiance, mask


def to_log(radiance: np.ndarray) -> np.ndarray:
    return np.log(radiance.astype(np.float64) + LOG_EPS)


def normalize_for_matching(log_radiance: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Percentile-normalize log-radiance to float32 [0, 1], suitable for feeding
    directly into a learned feature extractor (no 8-bit quantization).

    Masked-invalid (saturated/underexposed) pixels are zeroed before
    percentile estimation so they don't skew the normalization range.
    """
    valid = log_radiance if mask is None else log_radiance[mask]
    if valid.size == 0:
        valid = log_radiance
    lo, hi = np.percentile(valid, [1, 99])
    if hi <= lo:
        hi = lo + 1e-6
    norm = np.clip((log_radiance - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    if mask is not None:
        norm = norm.copy()
        norm[~mask] = 0.0
    return norm


def radiance_normalize_bgr(
    image_bgr: np.ndarray,
    exposure_us: float,
    gain_db: float = 0.0,
    mode: str = "crf",
    crf: CameraCRF | None = None,
    black_level: float = 0.0,
) -> np.ndarray:
    """BGR uint8 image -> gain/exposure-invariant radiance-normalized BGR
    uint8 image (grayscale content replicated across channels), for
    feeding directly into a keypoint extractor as a drop-in replacement
    for local-contrast normalization (CLAHE). Combines to_radiance ->
    to_log -> normalize_for_matching -> uint8 quantization in one call.
    """
    import cv2

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    radiance, mask = to_radiance(gray, exposure_us, mode=mode, crf=crf, black_level=black_level, gain_db=gain_db)
    norm = normalize_for_matching(to_log(radiance), mask)
    gray_out = np.clip(norm * 255.0, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(gray_out, cv2.COLOR_GRAY2BGR)
