#!/usr/bin/env python3
"""Median stereo-triangulated keypoint depth, windowed by a fixed number of
consecutive frames (default 4, matching the bracket cycle length
MAE/LAE/MAE/SAE), over a whole trajectory.

Pure per-frame DISK+LightGlue stereo triangulation (reuses
PoseGraphBuilder's own frontend directly) -- no temporal tracking, no
landmark BA, no smoother. Independent of the scale-bias investigation in
docs/cycle_bias_findings.md; answers a narrower question directly: does
bracketing's keypoint population sit systematically farther (or nearer)
than a same-window-size single-exposure population, over the whole route,
not just a saturation-heavy region?
"""

from __future__ import annotations

import argparse
import statistics as st
from pathlib import Path

import numpy as np

from pose_graph_bracketing.config import Config
from pose_graph_bracketing.dataset import load_stereo_sequence
from pose_graph_bracketing.features import DiskExtractor
from pose_graph_bracketing.matching import LightGlueMatcher
from pose_graph_bracketing.stereo import compute_stereo_observations, load_stereo_rig
from pose_graph_bracketing.imaging import load_preprocessed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml"))
    parser.add_argument("--window", type=int, default=4, help="frames per window (default 4, one bracket cycle)")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--label", default="traj")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config.load(args.config)
    data_dir = Path(args.data_dir)

    frames = load_stereo_sequence(data_dir)
    if args.max_frames is not None:
        frames = frames[: args.max_frames]
    rig = load_stereo_rig(data_dir / "calibration")
    extractor = DiskExtractor(cfg.disk, cfg.tracking)
    matcher = LightGlueMatcher(cfg.lightglue)

    per_frame_median: list[float] = []
    per_frame_n: list[int] = []
    for i, frame in enumerate(frames):
        left = load_preprocessed(frame.image_path, cfg.dataset.bayer_pattern, cfg.preprocessing.crop_bottom_px)
        right = load_preprocessed(frame.right_image_path, cfg.dataset.bayer_pattern, cfg.preprocessing.crop_bottom_px)
        feats_left = extractor.extract(left)
        feats_right = extractor.extract(right)
        match = matcher.match(feats_left, left.shape[:2], feats_right, right.shape[:2])
        obs = compute_stereo_observations(
            feats_left.keypoints, feats_right.keypoints, match.indices_a, match.indices_b,
            rig, cfg.stereo.min_disparity_px, cfg.stereo.max_depth_m,
        )
        depths = obs.points3d[:, 2] if len(obs.points3d) else np.array([])
        per_frame_median.append(float(np.median(depths)) if len(depths) else float("nan"))
        per_frame_n.append(len(depths))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(frames)} frames processed", flush=True)

    window = args.window
    window_medians = []
    for start in range(0, len(frames) - window + 1, window):
        chunk = [m for m in per_frame_median[start:start + window] if not np.isnan(m)]
        if chunk:
            window_medians.append(st.median(chunk))

    n_kp_total = sum(per_frame_n)
    print(f"=== {args.label}: {len(frames)} frames, {len(window_medians)} windows of {window} frames ===")
    print(f"total triangulated keypoints: {n_kp_total} (mean {n_kp_total/len(frames):.1f}/frame)")
    print(f"per-frame median depth: mean={st.mean(per_frame_median):.2f}m  median={st.median(per_frame_median):.2f}m")
    print(f"per-window median depth: mean={st.mean(window_medians):.2f}m  median={st.median(window_medians):.2f}m  "
          f"stdev={st.stdev(window_medians):.2f}m  min={min(window_medians):.2f}m  max={max(window_medians):.2f}m")


if __name__ == "__main__":
    main()
