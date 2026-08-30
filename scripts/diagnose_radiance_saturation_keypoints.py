#!/usr/bin/env python3
"""Check whether DISK keypoints (extracted from radiance-normalized frames)
cluster near saturation-mask boundaries.

Hypothesis being tested: when preprocessing.radiance_enabled is on, the
saturated/underexposed-pixel mask is forced to 0 in the normalized image
(radiance.normalize_for_matching), creating a hard synthetic edge at the
mask boundary. Combined with the CRF's steep response near saturation, DISK
(a gradient-driven learned detector) may be drawn to these boundaries rather
than to real scene texture -- and since each exposure bracket (SAE/MAE/LAE)
saturates a different region, these keypoints wouldn't have true
cross-bracket correspondences.

For one frame, this script:
  1. Loads the raw (CLAHE, no radiance) and radiance-normalized images.
  2. Extracts DISK keypoints from the radiance-normalized image (matching
     what the main pipeline does when radiance_enabled=True).
  3. Computes each keypoint's distance to the nearest saturated/underexposed
     mask pixel.
  4. Reports what fraction of keypoints fall within a few pixels of the mask
     boundary vs. a random-pixel baseline, and saves an overlay image.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import scipy.ndimage as ndi

from pose_graph_bracketing.config import Config
from pose_graph_bracketing.dataset import load_sequence
from pose_graph_bracketing.features import DiskExtractor
from pose_graph_bracketing.imaging import load_preprocessed
from pose_graph_bracketing.radiance import dilate_invalid_mask, load_crf, radiance_normalize_bgr, to_radiance


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True, help="Trajectory data dir (contains images_left/)")
    p.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml"))
    p.add_argument("--frame-index", type=int, default=0, help="Sequence index (post drop-filter) to inspect")
    p.add_argument("--boundary-px", type=int, default=5, help="\"Near mask boundary\" distance threshold, in pixels")
    p.add_argument(
        "--mask-dilate-px",
        type=int,
        default=None,
        help="Override preprocessing.radiance_mask_dilate_px -- grows the invalid mask by this many pixels "
        "before it's zeroed out, to test whether it keeps keypoints off the saturation boundary",
    )
    p.add_argument("--out", default=None, help="Optional path to save the overlay PNG (default: not saved)")
    p.add_argument("--no-show", action="store_true", help="Don't display the overlay in a window (needs a display)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config.load(args.config)
    frames = load_sequence(Path(args.data_dir))
    fr = frames[args.frame_index]

    gray = load_preprocessed(
        fr.image_path,
        cfg.dataset.bayer_pattern,
        cfg.preprocessing.crop_bottom_px,
        cfg.preprocessing.clahe_enabled,
        cfg.preprocessing.clahe_clip_limit,
        cfg.preprocessing.clahe_tile_grid_size,
        cfg.preprocessing.clahe_method,
    )

    mask_dilate_px = (
        args.mask_dilate_px if args.mask_dilate_px is not None else cfg.preprocessing.radiance_mask_dilate_px
    )

    crf = load_crf(cfg.preprocessing.radiance_crf_path) if cfg.preprocessing.radiance_mode == "crf" else None
    _, mask = to_radiance(gray, fr.exposure_us, mode=cfg.preprocessing.radiance_mode, crf=crf, gain_db=fr.gain_db)
    mask = dilate_invalid_mask(mask, mask_dilate_px)
    invalid = ~mask  # True where saturated/underexposed (post-dilation)

    radiance_img = radiance_normalize_bgr(
        gray, fr.exposure_us, fr.gain_db, mode=cfg.preprocessing.radiance_mode, crf=crf, mask_dilate_px=mask_dilate_px
    )

    extractor = DiskExtractor(cfg.disk, cfg.tracking)
    feats = extractor.extract(radiance_img)
    kps = feats.keypoints  # (N, 2) float32 (x, y)

    # Distance transform: for every pixel, distance to the nearest invalid (masked) pixel.
    dist_to_invalid = ndi.distance_transform_edt(~invalid)

    h, w = gray.shape
    xs = np.clip(kps[:, 0].round().astype(int), 0, w - 1)
    ys = np.clip(kps[:, 1].round().astype(int), 0, h - 1)
    kp_dist = dist_to_invalid[ys, xs]

    frac_near = float(np.mean(kp_dist <= args.boundary_px))

    rng = np.random.default_rng(0)
    rand_xs = rng.integers(0, w, size=10000)
    rand_ys = rng.integers(0, h, size=10000)
    rand_dist = dist_to_invalid[rand_ys, rand_xs]
    frac_near_random = float(np.mean(rand_dist <= args.boundary_px))

    frac_invalid = float(np.mean(invalid))

    print(f"frame_index={args.frame_index} slot={fr.slot_label} exposure_us={fr.exposure_us:.1f} gain_db={fr.gain_db:.2f}")
    print(f"n_keypoints={len(kps)}")
    print(f"fraction of image that is saturated/underexposed (invalid mask): {frac_invalid:.4f}")
    print(f"keypoints within {args.boundary_px}px of mask boundary: {frac_near:.4f}")
    print(f"random pixels within {args.boundary_px}px of mask boundary: {frac_near_random:.4f}")
    if frac_near_random > 0:
        print(f"enrichment ratio (keypoints vs. random baseline): {frac_near / frac_near_random:.2f}x")
    print(f"median keypoint distance to mask boundary: {np.median(kp_dist):.2f}px "
          f"(random baseline: {np.median(rand_dist):.2f}px)")

    # Overlay: radiance-normalized image, invalid mask in red, keypoints in green/yellow
    # (yellow = within boundary_px of the mask, green = far from it).
    overlay = radiance_img.copy()
    overlay[invalid] = (0, 0, 180)  # red-tinted saturated/underexposed regions (BGR)
    near = kp_dist <= args.boundary_px
    for (x, y) in kps[near]:
        cv2.circle(overlay, (int(round(x)), int(round(y))), 3, (0, 255, 255), -1)  # yellow
    for (x, y) in kps[~near]:
        cv2.circle(overlay, (int(round(x)), int(round(y))), 2, (0, 255, 0), -1)  # green

    if args.out is not None:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), overlay)
        print(f"wrote overlay (red=saturated/underexposed mask, yellow=keypoint near boundary, green=keypoint elsewhere) to {out_path}")

    if not args.no_show:
        cv2.imshow("radiance saturation diagnostic (red=mask, yellow=near boundary, green=elsewhere)", overlay)
        print("press any key in the image window to close it...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
