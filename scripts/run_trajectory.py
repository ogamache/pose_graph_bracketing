#!/usr/bin/env python3
"""Run the bracketed-exposure stereo pose graph over a trajectory and write a TUM trajectory file."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path

import numpy as np

from pose_graph_bracketing.calibration import load_stereo_calibration
from pose_graph_bracketing.config import Config
from pose_graph_bracketing.dataset import (
    drop_low_information_frames,
    drop_low_match_frames,
    load_sequence,
    load_stereo_sequence,
)
from pose_graph_bracketing.graph_builder import PoseGraphBuilder
from pose_graph_bracketing.graph_builder_mono import MonoPoseGraphBuilder
from pose_graph_bracketing.stereo import load_stereo_rig
from pose_graph_bracketing.trajectory_io import write_tum


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", required=True, help="Trajectory data dir (contains images_left/, images_right/, calibration/)"
    )
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml"))
    parser.add_argument("--out", required=True, help="Output TUM trajectory file path")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on number of frames processed")
    parser.add_argument(
        "--max-corners", type=int, default=None, help="Override tracking.max_corners (DISK keypoints per frame)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show a live diagnostic view (kept/discarded keypoints and matches, plus a live trajectory plot) "
        "in cv2 windows as the run progresses -- needs a display.",
    )
    parser.add_argument(
        "--step",
        action="store_true",
        help="With --visualize, pause after each frame and wait for a keypress before advancing "
        "(any key = next frame, 'q'/ESC = quit early). No effect without --visualize.",
    )
    parser.add_argument(
        "--global-ba",
        action="store_true",
        help="Force global bundle adjustment on even if graph.global_bundle_adjust is false in --config -- it's "
        "on by default already. After the incremental run, runs a full batch (non-fixed-lag) bundle adjustment "
        "over every factor added, and writes it to <out>_global_ba.tum alongside the normal (incremental) "
        "output. Stereo mode only.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("run_trajectory")

    cfg = Config.load(args.config)
    data_dir = Path(args.data_dir)

    if args.max_corners is not None:
        cfg.tracking.max_corners = args.max_corners
        log.info("Overriding tracking.max_corners -> %d", args.max_corners)

    if args.visualize:
        cfg.visualization.enabled = True
        log.info("Live diagnostic view enabled (cv2 windows)")

    if args.step:
        cfg.visualization.step = True
        log.info("Step mode enabled -- press any key to advance, 'q'/ESC to quit")

    if cfg.mode == "mono":
        frames = load_sequence(data_dir, side=cfg.dataset.side)
        if args.max_frames is not None:
            frames = frames[: args.max_frames]
        log.info("Loaded %d mono (%s) frames from %s", len(frames), cfg.dataset.side, data_dir)

        if cfg.preprocessing.drop_low_info_frames:
            n_before = len(frames)
            frames = drop_low_information_frames(
                frames,
                cfg.dataset.bayer_pattern,
                cfg.preprocessing.crop_bottom_px,
                cfg.preprocessing.drop_low_info_min_brightness,
                cfg.preprocessing.drop_low_info_max_brightness,
            )
            log.info("Dropped %d/%d low-information frames (brightness outside [%.0f, %.0f])",
                      n_before - len(frames), n_before, cfg.preprocessing.drop_low_info_min_brightness,
                      cfg.preprocessing.drop_low_info_max_brightness)

        if cfg.preprocessing.drop_low_match_frames:
            n_before = len(frames)
            frames = drop_low_match_frames(frames, cfg, cfg.preprocessing.drop_low_match_min_matches)
            log.info("Dropped %d/%d low-match frames (< %d matches against last kept frame)",
                      n_before - len(frames), n_before, cfg.preprocessing.drop_low_match_min_matches)

        calib = load_stereo_calibration(data_dir / "calibration" / "stereo_calibration_left.yaml")
        log.info("Mono calibration loaded: K_left principal point %.1f,%.1f", calib.K[0, 2], calib.K[1, 2])

        builder = MonoPoseGraphBuilder(cfg, calib)
    else:
        frames = load_stereo_sequence(data_dir)
        if args.max_frames is not None:
            frames = frames[: args.max_frames]
        log.info("Loaded %d stereo frames from %s", len(frames), data_dir)

        if cfg.preprocessing.drop_low_info_frames:
            n_before = len(frames)
            frames = drop_low_information_frames(
                frames,
                cfg.dataset.bayer_pattern,
                cfg.preprocessing.crop_bottom_px,
                cfg.preprocessing.drop_low_info_min_brightness,
                cfg.preprocessing.drop_low_info_max_brightness,
            )
            log.info("Dropped %d/%d low-information frames (brightness outside [%.0f, %.0f])",
                      n_before - len(frames), n_before, cfg.preprocessing.drop_low_info_min_brightness,
                      cfg.preprocessing.drop_low_info_max_brightness)

        if cfg.preprocessing.drop_low_match_frames:
            n_before = len(frames)
            frames = drop_low_match_frames(frames, cfg, cfg.preprocessing.drop_low_match_min_matches)
            log.info("Dropped %d/%d low-match frames (< %d matches against last kept frame)",
                      n_before - len(frames), n_before, cfg.preprocessing.drop_low_match_min_matches)

        rig = load_stereo_rig(data_dir / "calibration")
        log.info(
            "Stereo rig loaded: baseline=%.4f m, K_left principal point %.1f,%.1f", rig.baseline_m, rig.K_left[0, 2], rig.K_left[1, 2]
        )

        builder = PoseGraphBuilder(cfg, rig)

    t0 = time.time()
    results = builder.run(frames)
    elapsed = time.time() - t0
    log.info("Processed %d frames in %.1fs (%.2f fps)", len(results), elapsed, len(results) / max(elapsed, 1e-6))

    n_obs = sum(r.n_landmark_observations for r in results)
    log.info("Total landmark observations added: %d (avg %.2f per frame)", n_obs, n_obs / max(len(results), 1))

    zero_obs_frames = getattr(builder, "zero_obs_frames", [])
    if zero_obs_frames:
        log.info(
            "Frames with zero landmark observations: %d/%d -> %s",
            len(zero_obs_frames),
            len(results),
            zero_obs_frames,
        )
        zero_obs_path = str(Path(args.out).with_suffix("")) + "_zero_obs.txt"
        with open(zero_obs_path, "w") as f:
            for idx in zero_obs_frames:
                f.write(f"{idx} {results[idx].frame.timestamp_ns} {results[idx].frame.timestamp_s}\n")
        log.info("Wrote zero-observation frame indices to %s", zero_obs_path)

    timestamps_s = [r.frame.timestamp_s for r in results]
    poses = [r.pose for r in results]
    write_tum(args.out, timestamps_s, poses)
    log.info("Wrote trajectory to %s", args.out)

    if args.global_ba or cfg.graph.global_bundle_adjust:
        if cfg.mode != "stereo":
            log.warning("global bundle adjustment is stereo-mode only, skipping (mode=%s)", cfg.mode)
        else:
            import gtsam

            t0 = time.time()
            ba_values = builder.global_bundle_adjust()
            log.info("Global bundle adjustment: %.1fs", time.time() - t0)
            ba_poses = [ba_values.atPose3(gtsam.symbol("x", idx)) for idx in range(len(results))]

            # A frame with very few raw stereo observations (e.g. frame 1,
            # right after the trivially-unconstrained frame 0) is a near-
            # degenerate constraint on its own pose in the batch problem --
            # tried swapping such a frame's pose back to the incremental
            # value as a safeguard, but that created a worse, sharper
            # visual discontinuity than the original small deviation (a
            # frame artificially disconnected from its now-differently-
            # optimized neighbors) -- reverted. Early frames are simply
            # less reliable in both modes; no per-frame patching here.
            ba_out = str(Path(args.out).with_suffix("")) + "_global_ba.tum"
            write_tum(ba_out, timestamps_s, ba_poses)
            log.info("Wrote global-BA trajectory to %s", ba_out)

    # Robustness-metric logging (see docs/branch_comparison.md /
    # vision-refine-oscillation's docs/cycle_bias_findings.md sections
    # 5a/5b): pure logging, computed from data the run already produced --
    # no effect on estimation.
    frames_csv_path = str(Path(args.out).with_suffix("")) + "_frames.csv"
    with open(frames_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "timestamp_ns", "timestamp_s", "slot_label", "n_landmark_observations"])
        for idx, r in enumerate(results):
            writer.writerow([idx, r.frame.timestamp_ns, r.frame.timestamp_s, r.frame.slot_label, r.n_landmark_observations])
    log.info("Wrote per-frame robustness log to %s", frames_csv_path)

    n_backend_resets = getattr(builder, "n_backend_resets", 0)
    provenance = getattr(builder, "landmark_slot_provenance", {})
    frame_range = getattr(builder, "landmark_frame_range", {})
    creation_depth = getattr(builder, "landmark_creation_depth", {})
    if provenance:
        provenance_csv_path = str(Path(args.out).with_suffix("")) + "_landmark_provenance.csv"
        with open(provenance_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["landmark_id", "first_frame_idx", "last_frame_idx", "slots_seen", "creation_depth_m"])
            for landmark_id, slots in sorted(provenance.items()):
                first_idx, last_idx = frame_range.get(landmark_id, [-1, -1])
                writer.writerow([landmark_id, first_idx, last_idx, "|".join(sorted(slots)), creation_depth.get(landmark_id, "")])
        log.info("Wrote landmark provenance log to %s (%d landmarks)", provenance_csv_path, len(provenance))
    log.info("Backend resets this run: %d", n_backend_resets)

    summary_path = str(Path(args.out).with_suffix("")) + "_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "n_frames": len(results),
                "elapsed_s": elapsed,
                "fps": len(results) / max(elapsed, 1e-6),
                "n_landmark_observations_total": n_obs,
                "n_backend_resets": n_backend_resets,
            },
            f,
            indent=2,
        )
    log.info("Wrote run summary to %s", summary_path)


if __name__ == "__main__":
    main()
