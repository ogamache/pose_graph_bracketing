#!/usr/bin/env python3
"""Verify (or refute) cross-bracket radiance consistency: for a short run of
consecutive frames covering one SAE/MAE/LAE cycle, checks whether pixels
valid (non-saturated) in two different brackets agree on recovered
log-radiance.

Assumes the scene is ~static across one bracket cycle (a few frames, close
in time) so pixel (x, y) sees roughly the same real-world point in each
bracket -- no motion compensation is attempted, so pick a cycle where the
robot isn't moving fast, or expect some blur from real motion contaminating
the comparison near edges/motion.

For each ordered pair of brackets (e.g. LAE vs MAE), restricts to pixels
valid in both, bins by the reference bracket's raw pixel value Z, and
reports the mean/std log-radiance residual per bin. A residual that's flat
and near zero means good cross-bracket agreement; a residual that grows
(especially at high Z, where LAE's valid pixels concentrate) confirms the
CRF is unreliable in that regime.

Also reports the same comparison on the actual NORMALIZED matching-image
output (what radiance_normalize_bgr/radiance_normalize_bayer_bgr produce --
i.e. what the keypoint extractor actually sees), using
cfg.preprocessing.radiance_fixed_normalization to decide which
normalization path to run. This is the more direct check of the failure
mode investigated in-session: the un-normalized log-radiance can agree well
across brackets while the (old, per-frame-percentile) normalized output
still disagreed badly, because each frame's percentile stretch used a
different absolute window -- see radiance.normalize_for_matching_fixed's
docstring.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from pose_graph_bracketing.config import Config
from pose_graph_bracketing.dataset import load_sequence
from pose_graph_bracketing.imaging import load_preprocessed, load_raw
from pose_graph_bracketing.radiance import (
    load_crf,
    load_crf_bayer,
    load_crf_v2,
    radiance_normalize_bayer_bgr,
    radiance_normalize_bgr,
    to_log,
    to_radiance,
    to_radiance_bayer_mosaic,
    to_radiance_bgr,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True, help="Trajectory data dir (contains images_left/)")
    p.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml"))
    p.add_argument(
        "--start-index", type=int, default=0, help="Sequence index to start scanning from for one bracket cycle"
    )
    p.add_argument("--n-bins", type=int, default=20, help="Number of Z-value bins for the residual histogram")
    return p.parse_args()


def _load_gray(fr, cfg) -> np.ndarray:
    return load_preprocessed(
        fr.image_path,
        cfg.dataset.bayer_pattern,
        cfg.preprocessing.crop_bottom_px,
        cfg.preprocessing.clahe_enabled,
        cfg.preprocessing.clahe_clip_limit,
        cfg.preprocessing.clahe_tile_grid_size,
        cfg.preprocessing.clahe_method,
    )


def main() -> None:
    args = parse_args()
    cfg = Config.load(args.config)
    frames = load_sequence(Path(args.data_dir))

    # Grab one frame per slot label, starting from --start-index, in original
    # sequence order (so they're temporally close).
    by_slot: dict[str, int] = {}
    for i in range(args.start_index, len(frames)):
        slot = frames[i].slot_label
        if slot not in by_slot:
            by_slot[slot] = i
        if len(by_slot) >= 3:
            break

    if len(by_slot) < 2:
        raise SystemExit(f"Only found slots {list(by_slot)} starting at index {args.start_index} -- need at least 2")

    print(f"Using frames: { {slot: frames[i].sequence_index for slot, i in by_slot.items()} }")

    mode = cfg.preprocessing.radiance_mode
    crf = None
    if mode == "crf":
        crf = load_crf(cfg.preprocessing.radiance_crf_path)
    elif mode == "crf_v2":
        crf_path = (
            cfg.preprocessing.radiance_crf_path_right
            if cfg.dataset.side == "right"
            else cfg.preprocessing.radiance_crf_path_left
        )
        crf = load_crf_v2(crf_path)
    elif mode == "crf_bayer":
        crf_path = (
            cfg.preprocessing.radiance_crf_bayer_path_right
            if cfg.dataset.side == "right"
            else cfg.preprocessing.radiance_crf_bayer_path_left
        )
        crf = load_crf_bayer(crf_path)

    log_radiance = {}
    valid_mask = {}
    gray_img = {}
    norm_img = {}
    fixed_window_cache: dict = {}
    ref_slot = cfg.preprocessing.radiance_fixed_normalization_reference_slot
    # Process the reference bracket first regardless of raw sequence order,
    # so the shared window is already primed before any other bracket is
    # normalized -- matching the real trajectory, where the reference
    # bracket (MAE by default) cycles through frequently enough that its
    # window is always already set by an earlier frame by the time a nearby
    # SAE/LAE frame is processed. Testing in raw sequence order instead can
    # spuriously fail if this cycle happens to start on a non-reference slot.
    ordered_slots = sorted(by_slot.items(), key=lambda item: item[0] != ref_slot)
    for slot, idx in ordered_slots:
        fr = frames[idx]
        if mode == "crf_bayer":
            raw = load_raw(fr.image_path)
            radiance, mask = to_radiance_bayer_mosaic(raw, fr.exposure_us, fr.gain_db, crf, cfg.dataset.bayer_pattern)
            z_ref_img = raw
            norm_bgr = radiance_normalize_bayer_bgr(
                raw,
                fr.exposure_us,
                fr.gain_db,
                crf,
                cfg.dataset.bayer_pattern,
                mask_dilate_px=cfg.preprocessing.radiance_mask_dilate_px,
                fixed_normalization=cfg.preprocessing.radiance_fixed_normalization,
                fixed_window_cache=fixed_window_cache,
                fixed_window_key=id(crf),
                fixed_window_update=slot == cfg.preprocessing.radiance_fixed_normalization_reference_slot,
            )
        elif mode == "crf_v2":
            image_bgr = load_preprocessed(
                fr.image_path,
                cfg.dataset.bayer_pattern,
                cfg.preprocessing.crop_bottom_px,
                cfg.preprocessing.clahe_enabled,
                cfg.preprocessing.clahe_clip_limit,
                cfg.preprocessing.clahe_tile_grid_size,
                cfg.preprocessing.clahe_method,
            )
            radiance, mask = to_radiance_bgr(image_bgr, fr.exposure_us, fr.gain_db, crf)
            z_ref_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            norm_bgr = radiance_normalize_bgr(
                image_bgr,
                fr.exposure_us,
                fr.gain_db,
                mode=mode,
                crf=crf,
                mask_dilate_px=cfg.preprocessing.radiance_mask_dilate_px,
                fixed_normalization=cfg.preprocessing.radiance_fixed_normalization,
                fixed_window_cache=fixed_window_cache,
                fixed_window_key=id(crf),
                fixed_window_update=slot == cfg.preprocessing.radiance_fixed_normalization_reference_slot,
            )
        else:
            gray = load_preprocessed(
                fr.image_path,
                cfg.dataset.bayer_pattern,
                cfg.preprocessing.crop_bottom_px,
                cfg.preprocessing.clahe_enabled,
                cfg.preprocessing.clahe_clip_limit,
                cfg.preprocessing.clahe_tile_grid_size,
                cfg.preprocessing.clahe_method,
            )
            radiance, mask = to_radiance(gray, fr.exposure_us, mode=mode, crf=crf, gain_db=fr.gain_db)
            z_ref_img = gray
            norm_bgr = radiance_normalize_bgr(
                gray,
                fr.exposure_us,
                fr.gain_db,
                mode=mode,
                crf=crf,
                mask_dilate_px=cfg.preprocessing.radiance_mask_dilate_px,
                fixed_normalization=cfg.preprocessing.radiance_fixed_normalization,
                fixed_window_cache=fixed_window_cache,
                fixed_window_key=id(crf),
                fixed_window_update=slot == cfg.preprocessing.radiance_fixed_normalization_reference_slot,
            )

        log_radiance[slot] = to_log(radiance)
        valid_mask[slot] = mask
        gray_img[slot] = z_ref_img
        norm_img[slot] = cv2.cvtColor(norm_bgr, cv2.COLOR_BGR2GRAY)
        print(f"  {slot}: exposure_us={fr.exposure_us:.1f} gain_db={fr.gain_db:.2f} "
              f"valid_frac={mask.mean():.3f} mean_Z_valid={z_ref_img[mask].mean():.1f}")

    slots = list(by_slot.keys())
    for i in range(len(slots)):
        for j in range(i + 1, len(slots)):
            a, b = slots[i], slots[j]
            both_valid = valid_mask[a] & valid_mask[b]
            n_common = int(both_valid.sum())
            print(f"\n=== {a} vs {b}: {n_common} commonly-valid pixels ===")
            if n_common < 100:
                print("  too few common valid pixels to compare (brackets barely overlap in exposure) -- skipping")
                continue

            resid = (log_radiance[a] - log_radiance[b])[both_valid]
            z_ref = gray_img[a][both_valid]  # bin by bracket a's raw pixel value

            print(f"  overall log-radiance residual (un-normalized): mean={resid.mean():.4f}  std={resid.std():.4f}")

            bins = np.linspace(0, 254, args.n_bins + 1)
            bin_idx = np.digitize(z_ref, bins) - 1
            print(f"  {'Z range':>14} {'n_px':>8} {'mean_resid':>11} {'std_resid':>10}")
            for k in range(args.n_bins):
                sel = bin_idx == k
                if sel.sum() < 20:
                    continue
                print(f"  [{bins[k]:6.0f},{bins[k+1]:6.0f}) {int(sel.sum()):8d} {resid[sel].mean():11.4f} {resid[sel].std():10.4f}")

            # Same comparison on the actual normalized matching-image output
            # (what the keypoint extractor sees) -- this is what a per-frame
            # percentile normalization (fixed_normalization=False) distorts
            # even when the un-normalized log-radiance above agrees well.
            norm_diff = norm_img[a][both_valid].astype(np.float64) - norm_img[b][both_valid].astype(np.float64)
            print(f"\n  overall normalized-output residual (uint8, 0-255 scale): "
                  f"mean={norm_diff.mean():.2f}  std={norm_diff.std():.2f}")
            print(f"  {'Z range':>14} {'n_px':>8} {'mean_resid':>11} {'std_resid':>10}")
            for k in range(args.n_bins):
                sel = bin_idx == k
                if sel.sum() < 20:
                    continue
                print(f"  [{bins[k]:6.0f},{bins[k+1]:6.0f}) {int(sel.sum()):8d} "
                      f"{norm_diff[sel].mean():11.2f} {norm_diff[sel].std():10.2f}")


if __name__ == "__main__":
    main()
