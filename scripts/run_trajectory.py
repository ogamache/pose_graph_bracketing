#!/usr/bin/env python3
"""Run the bracketed-exposure stereo pose graph over a trajectory and write a TUM trajectory file."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from pose_graph_bracketing.config import Config
from pose_graph_bracketing.dataset import load_stereo_sequence
from pose_graph_bracketing.graph_builder import PoseGraphBuilder
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
        help="Write a recorded (not live) diagnostic MP4 showing kept/discarded keypoints and matches per frame",
    )
    parser.add_argument(
        "--visualize-out", default=None, help="Diagnostic video path (default: <out> with _viz.mp4 suffix)"
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
        cfg.visualization.output_path = args.visualize_out or str(Path(args.out).with_suffix("")) + "_viz.mp4"
        log.info("Diagnostic visualization enabled -> %s", cfg.visualization.output_path)

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

    n_reliable_pairs = sum(r.n_vo_factors for r in results)
    log.info("Total VO factors added: %d (avg %.2f per frame)", n_reliable_pairs, n_reliable_pairs / max(len(results), 1))

    timestamps_s = [r.frame.timestamp_s for r in results]
    poses = [r.pose for r in results]
    write_tum(args.out, timestamps_s, poses)
    log.info("Wrote trajectory to %s", args.out)
    if cfg.visualization.enabled:
        log.info("Wrote diagnostic video to %s", cfg.visualization.output_path)


if __name__ == "__main__":
    main()
