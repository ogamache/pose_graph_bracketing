#!/usr/bin/env python3
"""Scene-level dynamic-range pixel coverage: how much more of the scene is
actually visible (non-saturated, non-underexposed) when using all three
exposure brackets (SAE/MAE/LAE) vs. the best single exposure (MAE, this
rig's auto-exposure equivalent) alone.

Pure image analysis -- no SLAM pipeline involved, independent of config.
Grounds the "bracketing captures more of the scene" claim in the images
themselves, before asking whether SLAM makes good use of the extra
information (see scripts/compute_robustness.py's landmark-provenance
analysis for that question).

`valid_pixel_mask` below is ported from `vision-refine-oscillation`'s
`radiance.py` -- this branch (`slam-landmark-ba`) has no radiance module
at all (no CRF/exposure-normalization machinery), so rather than pull in
a whole module for one small, self-contained saturation/underexposure
gate, it's inlined directly here.

Cycles are grouped by FrameInfo.sequence_index % 4 (the fixed physical
bracket order MAE,LAE,MAE,SAE -- see dataset.py's _SEQUENCE_SLOT_LABELS);
only complete 4-frame cycles are used (a region's leading/trailing partial
cycle is dropped).
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import cv2
import numpy as np

from pose_graph_bracketing.config import Config
from pose_graph_bracketing.dataset import load_sequence
from pose_graph_bracketing.imaging import load_preprocessed

SATURATION_HIGH = 254  # >= this raw value is considered saturated
UNDEREXPOSED_MARGIN = 2  # raw value <= black_level + margin is considered underexposed


def valid_pixel_mask(gray: np.ndarray, black_level: float = 0.0) -> np.ndarray:
    low = black_level + UNDEREXPOSED_MARGIN
    return (gray > low) & (gray < SATURATION_HIGH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml"))
    parser.add_argument("--side", default="left")
    parser.add_argument("--start-frame", type=int, default=None, help="inclusive index into images_{side}/ sorted by filename")
    parser.add_argument("--end-frame", type=int, default=None, help="inclusive index into images_{side}/ sorted by filename")
    parser.add_argument("--label", default="region", help="just for the printed report header")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config.load(args.config)
    data_dir = Path(args.data_dir)

    frames = load_sequence(data_dir, side=args.side)
    if args.start_frame is not None or args.end_frame is not None:
        start = args.start_frame or 0
        end = args.end_frame if args.end_frame is not None else len(frames) - 1
        frames = frames[start : end + 1]

    cycles: list[list] = []
    current: list = []
    for fr in frames:
        if fr.sequence_index % 4 == 0 and current:
            cycles.append(current)
            current = []
        current.append(fr)
    if len(current) == 4:
        cycles.append(current)
    cycles = [c for c in cycles if len(c) == 4]

    if not cycles:
        print(f"No complete 4-frame bracket cycles found in {args.label} ({len(frames)} frames) -- nothing to report.")
        return

    union_fracs, best_single_fracs, gain_fracs = [], [], []
    frame_area = None

    for cycle in cycles:
        masks = {}
        for fr in cycle:
            gray = load_preprocessed(fr.image_path, cfg.dataset.bayer_pattern, cfg.preprocessing.crop_bottom_px)
            if frame_area is None:
                frame_area = gray.shape[0] * gray.shape[1]
            mask = valid_pixel_mask(gray)
            masks.setdefault(fr.slot_label, []).append(mask)

        mae_masks = masks.get("MAE", [])
        sae_masks = masks.get("SAE", [])
        lae_masks = masks.get("LAE", [])
        if not mae_masks:
            continue

        best_mae_frac = max(m.sum() / frame_area for m in mae_masks)

        union_mask = mae_masks[0].copy()
        for m in mae_masks[1:] + sae_masks + lae_masks:
            union_mask |= m
        union_frac = union_mask.sum() / frame_area

        union_fracs.append(union_frac)
        best_single_fracs.append(best_mae_frac)
        gain_fracs.append(union_frac - best_mae_frac)

    n = len(union_fracs)
    print(f"=== Dynamic-range pixel coverage: {args.label} ({n} complete bracket cycles) ===")
    print(f"Best single exposure (MAE) valid-pixel area: {statistics.mean(best_single_fracs)*100:.1f}% of frame (mean)")
    print(f"Union of all three brackets valid-pixel area: {statistics.mean(union_fracs)*100:.1f}% of frame (mean)")
    print(
        f"Absolute gain from bracketing: {statistics.mean(gain_fracs)*100:.1f} percentage points "
        f"(min {min(gain_fracs)*100:.1f}, max {max(gain_fracs)*100:.1f})"
    )
    print(
        f"Relative gain: {statistics.mean(gain_fracs) / max(statistics.mean(best_single_fracs), 1e-9) * 100:.1f}% "
        f"more valid scene area than MAE alone"
    )


if __name__ == "__main__":
    main()
