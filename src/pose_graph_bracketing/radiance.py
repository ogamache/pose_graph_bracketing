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
import yaml

SATURATION_HIGH = 250  # >= this raw value is considered saturated
UNDEREXPOSED_MARGIN = 2  # raw value <= black_level + margin is considered underexposed
LOG_EPS = 1e-4

LUMA_WEIGHTS = {"B": 0.114, "G": 0.587, "R": 0.299}  # matches cv2.COLOR_BGR2GRAY

FIXED_NORM_EPS = 1e-12  # tiny relative to real radiance magnitudes (~1e-7 to 1e-2); just avoids log(0)


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


RGB_CHANNELS = ("R", "G", "B")


@dataclass
class CameraCRFv2:
    """Exposure- and gain-aware CRF fitted by
    camera_calibration/scripts/crf_calibration.py on demosaiced R/G/B
    images (same domain this is applied in -- see `radiance_normalize_bgr`'s
    mode="crf_v2" path), unlike the legacy `CameraCRF` which was fit at one
    fixed reference gain on grayscale.

    One response curve and one full-resolution per-pixel black-level model
    per channel (R, G, B) -- fit on real demosaiced color planes, so
    genuinely-different channel sensitivities are captured without the
    cross-color contamination a shared grayscale curve would hide, while
    still applying correction *after* demosaicing (not on the raw mosaic):
    an earlier revision corrected each raw Bayer photosite independently
    before demosaicing, which skipped the noise-averaging demosaic
    incidentally provides and produced visible speckle on real frames at
    higher gain -- reverted.

    `response[channel]` is OpenCV CalibrateDebevec's response function,
    already in *linear* domain (radiance = response[Z] / effective_exposure)
    -- see crf_calibration.py's own docstring and OpenCV's calibrate.cpp,
    which exponentiates the fitted log-response before returning it.
    """

    response: dict[str, np.ndarray]  # channel -> shape (256,), linear-domain
    black_b0: dict[str, np.ndarray]  # channel -> shape (H, W), full-resolution (pre-crop)
    black_b1: dict[str, np.ndarray]
    black_b2: dict[str, np.ndarray]


def load_crf_v2(yaml_path: str) -> CameraCRFv2:
    """Load a per-channel (R/G/B) CRF fitted by
    camera_calibration/scripts/crf_calibration.py on demosaiced images.

    `yaml_path` is the calibration YAML; its `black_level_model.coefficients_file`
    entry (a sibling .npz with one (3, H, W) = b0/b1/b2 array per channel) is
    resolved relative to the YAML's own directory if given as a relative path.
    """
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    response = {ch: np.asarray(data["response_function"][ch]["data"], dtype=np.float64) for ch in RGB_CHANNELS}

    coeff_path = data["black_level_model"]["coefficients_file"]
    if not os.path.isabs(coeff_path):
        coeff_path = os.path.join(os.path.dirname(yaml_path), coeff_path)
    coeffs = np.load(coeff_path)
    black_b0 = {ch: coeffs[ch][0] for ch in RGB_CHANNELS}
    black_b1 = {ch: coeffs[ch][1] for ch in RGB_CHANNELS}
    black_b2 = {ch: coeffs[ch][2] for ch in RGB_CHANNELS}

    return CameraCRFv2(response=response, black_b0=black_b0, black_b1=black_b1, black_b2=black_b2)


def predict_black_level_v2(crf: CameraCRFv2, channel: str, exposure_us: float, gain_db: float) -> np.ndarray:
    """Per-pixel black level BL(t, g_lin) = b0 + b1*t + b2*g_lin, shape (H, W)."""
    gain_linear = gain_db_to_linear(gain_db)
    return crf.black_b0[channel] + crf.black_b1[channel] * exposure_us + crf.black_b2[channel] * gain_linear


def _crop_to_match(array: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Crop `array`'s bottom rows to match `target_shape`, mirroring
    imaging.crop_bottom (the black-level model is calibrated on the
    full-resolution, pre-crop image)."""
    h, w = target_shape
    if array.shape == (h, w):
        return array
    if array.shape[0] < h or array.shape[1] != w:
        raise ValueError(f"Cannot align black-level map {array.shape} to image shape {(h, w)}")
    return array[:h]


def to_radiance_bgr(
    image_bgr: np.ndarray,
    exposure_us: float,
    gain_db: float,
    crf: CameraCRFv2,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert an already-demosaiced BGR uint8 image to a single-plane
    luma radiance, correcting each of the B/G/R planes with its OWN
    channel's CRF/black level (not one shared curve), then combining via
    standard luma weights.

    Returns (luma_radiance, valid_mask), both (H, W) -- float32/bool.
    """
    gain_linear = gain_db_to_linear(gain_db)
    effective_exposure = exposure_us * gain_linear
    h, w = image_bgr.shape[:2]

    bgr_to_channel = {0: "B", 1: "G", 2: "R"}

    luma_radiance = np.zeros((h, w), dtype=np.float64)
    valid = np.ones((h, w), dtype=bool)
    for c, channel in bgr_to_channel.items():
        black = _crop_to_match(predict_black_level_v2(crf, channel, exposure_us, gain_db), (h, w))
        plane = image_bgr[..., c].astype(np.float64)
        corrected = np.clip(plane - black, 0.0, 255.0)
        idx = np.rint(corrected).astype(np.int64)
        plane_radiance = crf.response[channel][idx] / effective_exposure
        luma_radiance += LUMA_WEIGHTS[channel] * plane_radiance
        valid &= (plane > black + UNDEREXPOSED_MARGIN) & (plane < SATURATION_HIGH)

    return luma_radiance.astype(np.float32), valid


def valid_pixel_mask(gray: np.ndarray, black_level: float | np.ndarray = 0.0) -> np.ndarray:
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
    """Convert a single-channel 8-bit image to float32 radiance.

    mode="crf": ln(radiance) = g(Z) - ln(dt) - ln(gain_linear), using a
      CameraCRF fitted at a fixed reference gain.
    mode="linear": radiance = (Z - black_level) / (dt * gain_linear).

    `mode="crf_v2"` (CameraCRFv2, per-R/G/B-channel) is NOT handled here --
    it corrects each BGR plane of an already-demosaiced color image with its
    own channel's curve before combining into luma; see `to_radiance_bgr`.

    Returns (radiance, valid_mask); radiance is scaled up to the overall
    constant ambiguity inherent to both methods -- only relative/log
    differences across the bracket matter for matching.
    """
    gain_linear = gain_db_to_linear(gain_db)
    mask = valid_pixel_mask(img, black_level=black_level)

    if mode == "crf":
        if crf is None or not isinstance(crf, CameraCRF):
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


def dilate_invalid_mask(mask: np.ndarray, dilate_px: int) -> np.ndarray:
    """Shrink a valid-pixel mask by `dilate_px`, so a margin around every
    saturated/underexposed region is also treated as invalid.

    Keypoints cluster at the hard zeroed edge normalize_for_matching puts at
    the mask boundary (and on the CRF's noisy, steep near-saturation
    response just inside it) -- see docs diagnostic
    scripts/diagnose_radiance_saturation_keypoints.py. Growing the invalid
    region pushes that edge away from real scene texture so a learned
    detector is less likely to lock onto it.
    """
    if dilate_px <= 0:
        return mask
    import cv2

    invalid = (~mask).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
    invalid_dilated = cv2.dilate(invalid, kernel)
    return ~invalid_dilated.astype(bool)


BAYER_CHANNELS = ("R", "Gr", "Gb", "B")

# Native photosite (row_offset, col_offset) for each channel, by mosaic
# pattern -- matches camera_calibration/scripts/utils.py's
# split_bayer_channels/BAYER_CHANNEL_OFFSETS convention.
BAYER_CHANNEL_OFFSETS = {
    "rggb": {"R": (0, 0), "Gr": (0, 1), "Gb": (1, 0), "B": (1, 1)},
    "bggr": {"B": (0, 0), "Gb": (0, 1), "Gr": (1, 0), "R": (1, 1)},
    "grbg": {"Gr": (0, 0), "R": (0, 1), "B": (1, 0), "Gb": (1, 1)},
    "gbrg": {"Gb": (0, 0), "B": (0, 1), "R": (1, 0), "Gr": (1, 1)},
}


@dataclass
class CameraCRFBayer:
    """Per-native-Bayer-photosite (R/Gr/Gb/B) CRF fitted directly on the raw,
    non-demosaiced mosaic by camera_calibration/scripts/crf_calibration.py
    -- unlike CameraCRFv2, which is fit post-demosaic on synthesized R/G/B
    planes.

    Corrects each raw photosite independently *before* demosaicing, which
    skips the noise-averaging demosaic incidentally provides and produced
    visible speckle on real frames at higher gain (see CameraCRFv2's
    docstring for the fix this was reverted in favor of) -- reinstated here
    for direct ablation against CameraCRFv2 now that the CRF itself has
    been refit.

    `black_b0/b1/b2[channel]` are at native per-channel (half-resolution)
    photosite resolution, matching `response[channel]`'s own domain.
    """

    response: dict[str, np.ndarray]  # channel -> shape (256,), linear-domain
    black_b0: dict[str, np.ndarray]  # channel -> shape (H/2, W/2)
    black_b1: dict[str, np.ndarray]
    black_b2: dict[str, np.ndarray]


def load_crf_bayer(yaml_path: str) -> CameraCRFBayer:
    """Load a per-native-Bayer-channel (R/Gr/Gb/B) CRF fitted by
    camera_calibration/scripts/crf_calibration.py on the raw mosaic.

    `yaml_path` is the calibration YAML; its `black_level_model.coefficients_file`
    entry is resolved relative to the YAML's own directory if given as a
    relative path, and falls back to a same-named sibling of the YAML if the
    recorded path doesn't exist on disk (the YAML may have been generated on
    a different machine/directory layout).
    """
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    response = {ch: np.asarray(data["response_function"][ch]["data"], dtype=np.float64) for ch in BAYER_CHANNELS}

    coeff_path = data["black_level_model"]["coefficients_file"]
    if not os.path.isabs(coeff_path):
        coeff_path = os.path.join(os.path.dirname(yaml_path), coeff_path)
    if not os.path.isfile(coeff_path):
        # The YAML's recorded path may be stale (generated on a different
        # machine/directory layout) -- fall back to whichever sibling .npz
        # next to the YAML matches its own left/right side.
        yaml_dir = os.path.dirname(yaml_path)
        side = "left" if "left" in os.path.basename(yaml_path).lower() else "right"
        candidates = [
            f for f in os.listdir(yaml_dir) if f.endswith(".npz") and side in f.lower()
        ]
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"black-level coefficients file not found: {coeff_path} "
                f"(and could not find a unique '{side}' .npz fallback in {yaml_dir}: {candidates})"
            )
        coeff_path = os.path.join(yaml_dir, candidates[0])
    coeffs = np.load(coeff_path)
    black_b0 = {ch: coeffs[ch][0] for ch in BAYER_CHANNELS}
    black_b1 = {ch: coeffs[ch][1] for ch in BAYER_CHANNELS}
    black_b2 = {ch: coeffs[ch][2] for ch in BAYER_CHANNELS}

    return CameraCRFBayer(response=response, black_b0=black_b0, black_b1=black_b1, black_b2=black_b2)


def to_radiance_bayer_mosaic(
    raw: np.ndarray,
    exposure_us: float,
    gain_db: float,
    crf: CameraCRFBayer,
    bayer_pattern: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Correct each raw Bayer photosite independently (its own native
    channel's CRF + per-pixel black level), returning un-normalized radiance
    still in native mosaic layout (each pixel holds only its own channel's
    value -- no cross-channel interpolation/demosaic at any point).

    Returns (radiance_mosaic, valid_mask), both full raw resolution,
    float64/bool. See `radiance_normalize_bayer_bgr` for the
    log+normalize+quantize wrapper used by the main pipeline, and
    `scripts/check_cross_bracket_radiance_consistency.py` for a consumer
    that wants this un-normalized form directly (cross-bracket comparison
    would be meaningless on per-frame percentile-stretched values).
    """
    pattern = bayer_pattern.lower()
    if pattern not in BAYER_CHANNEL_OFFSETS:
        raise ValueError(f"Unknown bayer_pattern: {bayer_pattern}")
    offsets = BAYER_CHANNEL_OFFSETS[pattern]

    gain_linear = gain_db_to_linear(gain_db)
    effective_exposure = exposure_us * gain_linear

    radiance_mosaic = np.zeros(raw.shape, dtype=np.float64)
    valid_mosaic = np.zeros(raw.shape, dtype=bool)
    for channel, (r0, c0) in offsets.items():
        plane = raw[r0::2, c0::2].astype(np.float64)
        black = crf.black_b0[channel] + crf.black_b1[channel] * exposure_us + crf.black_b2[channel] * gain_linear
        black = _crop_to_match(black, plane.shape)
        corrected = np.clip(plane - black, 0.0, 255.0)
        idx = np.rint(corrected).astype(np.int64)
        radiance_mosaic[r0::2, c0::2] = crf.response[channel][idx] / effective_exposure
        valid_mosaic[r0::2, c0::2] = (plane > black + UNDEREXPOSED_MARGIN) & (plane < SATURATION_HIGH)

    return radiance_mosaic, valid_mosaic


def radiance_normalize_bayer_bgr(
    raw: np.ndarray,
    exposure_us: float,
    gain_db: float,
    crf: CameraCRFBayer,
    bayer_pattern: str,
    mask_dilate_px: int = 0,
    fixed_normalization: bool = True,
    fixed_window_cache: dict | None = None,
    fixed_window_key: object = None,
    fixed_window_update: bool = True,
) -> np.ndarray:
    """Log-compress and normalize each of the 4 native Bayer channels
    (R/Gr/Gb/B) on its OWN scale, then reassemble the normalized native
    photosites straight back into one full-resolution grayscale image (no
    color demosaic/interpolation across channels at any point).

    Per-channel (rather than one shared) log+normalize matters here: R/Gr/
    Gb/B can have substantially different radiance scales after CRF
    correction, so a single shared window would let one channel's range
    dominate and crush the others.

    `raw` is the un-demosaiced single-channel mosaic (as loaded by
    `imaging.load_raw`, full sensor resolution). Returns a uint8 BGR image
    (this grayscale content replicated across channels), a drop-in
    replacement for `imaging.load_preprocessed` + `radiance_normalize_bgr`.

    `fixed_normalization` (default True): see `radiance_normalize_bgr`'s
    docstring -- same fix (a window per channel, shared across brackets and
    refreshed from the reference bracket -- see `get_or_update_fixed_window`).
    False restores the old per-frame, per-channel percentile behavior, kept
    for ablation.

    `fixed_window_cache`/`fixed_window_key`/`fixed_window_update`: required
    when `fixed_normalization=True` -- see `radiance_normalize_bgr`'s
    docstring; `fixed_window_key` is combined internally with each Bayer
    channel so R/Gr/Gb/B each get their own independently-tracked window.
    """
    import cv2

    pattern = bayer_pattern.lower()
    offsets = BAYER_CHANNEL_OFFSETS[pattern]
    radiance_mosaic, valid_mosaic = to_radiance_bayer_mosaic(raw, exposure_us, gain_db, crf, bayer_pattern)
    if fixed_normalization and fixed_window_cache is None:
        raise ValueError("fixed_normalization=True requires fixed_window_cache")

    gray_u8 = np.zeros(raw.shape, dtype=np.uint8)
    valid_out = np.zeros(raw.shape, dtype=bool)
    for channel, (r0, c0) in offsets.items():
        radiance_plane = radiance_mosaic[r0::2, c0::2]
        valid_plane = valid_mosaic[r0::2, c0::2]

        if fixed_normalization:
            log_plane = np.log(np.clip(radiance_plane, 0.0, None) + FIXED_NORM_EPS)
            lo, hi = get_or_update_fixed_window(
                fixed_window_cache, (fixed_window_key, channel), log_plane, valid_plane, fixed_window_update
            )
            norm_plane = normalize_for_matching_fixed(log_plane, valid_plane, lo, hi)
        else:
            valid_vals = radiance_plane[valid_plane]
            floor = float(np.percentile(valid_vals, 1)) if valid_vals.size else float(radiance_plane.min())
            eps = max(floor * 0.01, 1e-12)
            log_plane = np.log(radiance_plane + eps)
            norm_plane = normalize_for_matching(log_plane, valid_plane)

        gray_u8[r0::2, c0::2] = np.clip(norm_plane * 255.0, 0.0, 255.0).astype(np.uint8)
        valid_out[r0::2, c0::2] = valid_plane

    valid_out = dilate_invalid_mask(valid_out, mask_dilate_px)
    gray_u8[~valid_out] = 0

    return cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)


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


def normalize_for_matching_fixed(log_value: np.ndarray, mask: np.ndarray | None, lo: float, hi: float) -> np.ndarray:
    """Like `normalize_for_matching`, but stretches to a FIXED [lo, hi]
    window shared across every frame/bracket, instead of each frame's own
    percentiles -- see `get_or_update_fixed_window` for how that window is
    obtained.

    Cross-bracket radiance disagreement was traced (see docs/debugging
    session on SAE/MAE match quality) not to the CRF correction itself --
    `check_cross_bracket_radiance_consistency.py` showed the *recovered*
    log-radiance agrees well across brackets -- but to this final display/
    matching-image step: `normalize_for_matching`'s per-frame [1,99]
    percentile stretch uses a different absolute window on every frame
    (SAE's valid pixels are a narrow, bright-skewed subset vs. MAE's
    near-full-scene population), so the SAME recovered radiance value maps
    to a DIFFERENT output intensity depending on which bracket produced it
    -- actively degrading cross-bracket keypoint matching. Using one fixed
    window instead of per-frame content statistics fixes that: the same
    real-world radiance now always maps to the same output intensity,
    regardless of exposure/gain.
    """
    if hi <= lo:
        hi = lo + 1e-6
    norm = np.clip((log_value - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    if mask is not None:
        norm = norm.copy()
        norm[~mask] = 0.0
    return norm


def get_or_update_fixed_window(
    cache: dict, key, log_value: np.ndarray, mask: np.ndarray | None, update: bool
) -> tuple[float, float]:
    """Return the shared [lo, hi] window for `key`.

    `update=True` (re)computes it from `log_value`/`mask` (this frame's own
    [1,99] percentiles, exactly like `normalize_for_matching`) and
    overwrites the cached value -- call this only for frames from the
    designated *reference* bracket (MAE by default: the least noisy of
    SAE/MAE/LAE, since its moderate exposure+gain gives the best photon-
    count-vs-amplified-noise tradeoff for typical scene brightness -- see
    config.PreprocessingConfig.radiance_fixed_normalization_reference_slot).
    `update=False` just returns whatever is currently cached (falling back
    to computing-and-caching from THIS frame if nothing's cached yet, e.g.
    the trajectory starts on a non-reference bracket before any reference
    frame has been seen).

    This is what makes the normalization "shared/global" rather than
    per-frame, while still tracking real scene-brightness drift over a long
    trajectory: a raw index-domain window derived purely from the CRF curve
    doesn't work, because the actual recovered radiance's absolute
    magnitude depends on the (microsecond-scale) exposure it's divided by,
    which varies by orders of magnitude across the bracket cycle and isn't
    knowable from the calibration curve alone -- so the window has to be
    anchored to real frame data. Anchoring it to the reference bracket and
    refreshing on every one of its frames (instead of freezing once at the
    very first frame, which could go stale if scene brightness drifts far
    from it) keeps the window both shared across SAE/MAE/LAE AND
    continuously representative of current conditions.

    `key` should distinguish independent windows that shouldn't share scale
    (e.g. `(mode, side)` for radiance_normalize_bgr, `(side, channel)` per
    Bayer channel for radiance_normalize_bayer_bgr -- see call sites). The
    caller owns `cache`'s lifetime (e.g. one dict per PoseGraphBuilder run).
    """
    if not update and key in cache:
        return cache[key]
    valid = log_value if mask is None else log_value[mask]
    if valid.size == 0:
        valid = log_value
    lo, hi = np.percentile(valid, [1, 99])
    if hi <= lo:
        hi = lo + 1e-6
    cache[key] = (float(lo), float(hi))
    return cache[key]


def radiance_normalize_bgr(
    image_bgr: np.ndarray,
    exposure_us: float,
    gain_db: float = 0.0,
    mode: str = "crf",
    crf: CameraCRF | None = None,
    black_level: float = 0.0,
    mask_dilate_px: int = 0,
    fixed_normalization: bool = True,
    fixed_window_cache: dict | None = None,
    fixed_window_key: object = None,
    fixed_window_update: bool = True,
) -> np.ndarray:
    """BGR uint8 image -> gain/exposure-invariant radiance-normalized BGR
    uint8 image (grayscale content replicated across channels), for
    feeding directly into a keypoint extractor as a drop-in replacement
    for local-contrast normalization (CLAHE). Combines to_radiance ->
    to_log -> normalize -> uint8 quantization in one call.

    `mask_dilate_px`: grows each saturated/underexposed region by this many
    pixels before zeroing it out, so keypoints don't cluster on the hard
    mask-boundary edge -- see `dilate_invalid_mask`.

    `fixed_normalization` (default True): stretch to a window shared across
    every frame but periodically refreshed from a designated reference
    bracket -- see `get_or_update_fixed_window` -- instead of
    `normalize_for_matching`'s per-frame [1,99] percentiles. Confirmed the
    per-frame version was the actual source of cross-bracket (SAE/MAE/LAE)
    matching-image inconsistency, not the CRF correction itself. False
    restores the old per-frame-percentile behavior, kept for ablation.

    `fixed_window_cache`/`fixed_window_key`/`fixed_window_update`: required
    when `fixed_normalization=True`. `cache` is a dict the caller owns for
    the lifetime of one run (e.g. one per PoseGraphBuilder). `key`
    distinguishes independent windows that shouldn't share scale (e.g. left
    vs. right camera). `update` should be True only for frames from the
    designated reference bracket (recompute-and-overwrite the cached
    window) and False for every other frame (just read the current cached
    window). See `get_or_update_fixed_window`.
    """
    import cv2

    if mode == "crf_v2":
        if crf is None or not isinstance(crf, CameraCRFv2):
            raise ValueError("mode='crf_v2' requires a CameraCRFv2")
        if image_bgr.ndim != 3:
            raise ValueError("mode='crf_v2' needs a color BGR image (per-channel R/G/B curves) -- the pipeline is grayscale end-to-end now, so crf_v2 is no longer usable here")
        radiance, mask = to_radiance_bgr(image_bgr, exposure_us, gain_db, crf)
    else:
        gray = image_bgr if image_bgr.ndim == 2 else cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        radiance, mask = to_radiance(gray, exposure_us, mode=mode, crf=crf, black_level=black_level, gain_db=gain_db)
    mask = dilate_invalid_mask(mask, mask_dilate_px)

    if fixed_normalization:
        if fixed_window_cache is None:
            raise ValueError("fixed_normalization=True requires fixed_window_cache")
        log_radiance = np.log(np.clip(radiance.astype(np.float64), 0.0, None) + FIXED_NORM_EPS)
        lo, hi = get_or_update_fixed_window(
            fixed_window_cache, (mode, fixed_window_key), log_radiance, mask, fixed_window_update
        )
        norm = normalize_for_matching_fixed(log_radiance, mask, lo, hi)
    elif mode == "crf_v2":
        # LOG_EPS (1e-4) is calibrated for the legacy `crf` mode's radiance
        # scale. CameraCRFv2's response-curve-based radiance can sit orders
        # of magnitude smaller depending on the calibration's own units --
        # if LOG_EPS then dominates, it crushes the frame's whole dynamic
        # range into a razor-thin log-radiance band that the percentile
        # stretch below blows back up to fill 0-255, amplifying ordinary
        # sensor noise into visible speckle (confirmed on a real frame this
        # session). Use a data-driven epsilon scaled to this frame's own
        # radiance floor instead, so the log transform actually compresses
        # the real dynamic range rather than being swamped by a constant
        # sized for a different CRF's units.
        valid_vals = radiance[mask]
        floor = float(np.percentile(valid_vals, 1)) if valid_vals.size else float(radiance.min())
        eps = max(floor * 0.01, 1e-12)
        log_radiance = np.log(radiance.astype(np.float64) + eps)
        norm = normalize_for_matching(log_radiance, mask)
    else:
        log_radiance = to_log(radiance)
        norm = normalize_for_matching(log_radiance, mask)

    gray_out = np.clip(norm * 255.0, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(gray_out, cv2.COLOR_GRAY2BGR)
