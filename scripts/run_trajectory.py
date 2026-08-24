#!/usr/bin/env python3
"""Run the bracketed-exposure stereo pose graph over a trajectory and write a TUM trajectory file."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from pose_graph_bracketing.calibration import load_stereo_calibration
from pose_graph_bracketing.config import Config
from pose_graph_bracketing.dataset import load_sequence, load_stereo_sequence
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

        calib = load_stereo_calibration(data_dir / "calibration" / "stereo_calibration_left.yaml")
        log.info("Mono calibration loaded: K_left principal point %.1f,%.1f", calib.K[0, 2], calib.K[1, 2])

        builder = MonoPoseGraphBuilder(cfg, calib)
    else:
        frames = load_stereo_sequence(data_dir)
        if args.max_frames is not None:
            frames = frames[: args.max_frames]
        log.info("Loaded %d stereo frames from %s", len(frames), data_dir)

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


if __name__ == "__main__":
    main()
