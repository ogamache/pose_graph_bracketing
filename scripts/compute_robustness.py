#!/usr/bin/env python3
"""Trajectory-continuity robustness metrics (see docs/cycle_bias_findings.md
/ ~/.claude/plans/jazzy-seeking-iverson.md section 5a): how badly and how
often a run went visually blind, as a more direct proxy for "does
bracketing keep the SLAM fed" than ATE/RPE alone (both of which can be
flattered by a degenerate/uninformed trajectory -- see this project's
established preference for RPE over ATE for the same reason).

Reads the per-frame CSV run_trajectory.py writes (<out>_frames.csv: idx,
timestamp_ns, timestamp_s, slot_label, n_landmark_observations) and,
optionally, the landmark-provenance CSV (<out>_landmark_provenance.csv:
landmark_id, first_frame_idx, last_frame_idx, slots_seen) for section 5b's
"landmarks only visible thanks to SAE/LAE" analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-csv", required=True)
    parser.add_argument("--provenance-csv", default=None, help="optional, enables section 5b reporting")
    parser.add_argument("--start-frame", type=int, default=None, help="inclusive idx, region crop")
    parser.add_argument("--end-frame", type=int, default=None, help="inclusive idx, region crop")
    parser.add_argument("--low-info-threshold", type=int, default=20)
    parser.add_argument("--n-backend-resets", type=int, default=None, help="from run_trajectory.py's log line, if known")
    parser.add_argument("--label", default="run")
    return parser.parse_args()


def longest_run(flags: list[bool]) -> int:
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def main() -> None:
    args = parse_args()

    n_backend_resets = args.n_backend_resets
    if n_backend_resets is None:
        summary_path = Path(args.frames_csv.replace("_frames.csv", "_summary.json"))
        if summary_path.exists():
            with open(summary_path) as f:
                n_backend_resets = json.load(f).get("n_backend_resets")

    rows = []
    with open(args.frames_csv, newline="") as f:
        for row in csv.DictReader(f):
            idx = int(row["idx"])
            if idx == 0:
                # Frame 0 has no lookback frame to match against, so it's
                # trivially always n_landmark_observations==0 -- not a real
                # "blind" frame. run_trajectory.py's own zero_obs_frames
                # list already excludes it (`if idx > 0 and n_obs == 0`);
                # match that convention here so full-scope counts agree.
                continue
            if args.start_frame is not None and idx < args.start_frame:
                continue
            if args.end_frame is not None and idx > args.end_frame:
                continue
            rows.append(row)

    n = len(rows)
    if n == 0:
        print(f"No frames in range for {args.label} -- nothing to report.")
        return

    n_obs = [int(r["n_landmark_observations"]) for r in rows]
    zero_flags = [x == 0 for x in n_obs]
    low_info_flags = [x < args.low_info_threshold for x in n_obs]

    max_consec_zero = longest_run(zero_flags)
    total_zero = sum(zero_flags)
    max_consec_low = longest_run(low_info_flags)
    total_low = sum(low_info_flags)

    print(f"=== Trajectory-continuity robustness: {args.label} ({n} frames) ===")
    print(f"max_consecutive_zero_obs: {max_consec_zero}")
    print(f"total_zero_obs_frames: {total_zero}/{n} ({total_zero/n*100:.1f}%)")
    print(f"max_consecutive_low_info (<{args.low_info_threshold}): {max_consec_low}")
    print(f"total_low_info_frames: {total_low}/{n} ({total_low/n*100:.1f}%)")
    print(f"mean_landmark_observations: {sum(n_obs)/n:.1f}")
    if n_backend_resets is not None:
        print(f"n_backend_resets (whole run, not region-scoped): {n_backend_resets}")

    if args.provenance_csv:
        prov_rows = []
        with open(args.provenance_csv, newline="") as f:
            for row in csv.DictReader(f):
                first_idx = int(row["first_frame_idx"])
                last_idx = int(row["last_frame_idx"])
                if args.start_frame is not None and last_idx < args.start_frame:
                    continue
                if args.end_frame is not None and first_idx > args.end_frame:
                    continue
                prov_rows.append(row)

        n_landmarks = len(prov_rows)
        if n_landmarks == 0:
            print("No landmarks active in this region.")
            return

        n_pure_mae = sum(1 for r in prov_rows if r["slots_seen"] == "MAE")
        n_mae_exclusive_absent = sum(1 for r in prov_rows if "MAE" not in r["slots_seen"].split("|"))
        n_mixed_with_mae = n_landmarks - n_pure_mae - n_mae_exclusive_absent

        print(f"--- Landmark provenance by exposure slot ({n_landmarks} landmarks active in region) ---")
        print(f"pure-MAE landmarks: {n_pure_mae} ({n_pure_mae/n_landmarks*100:.1f}%)")
        print(f"mixed (MAE + SAE/LAE) landmarks: {n_mixed_with_mae} ({n_mixed_with_mae/n_landmarks*100:.1f}%)")
        print(
            f"MAE-exclusive-absent landmarks (only exist thanks to SAE/LAE): "
            f"{n_mae_exclusive_absent} ({n_mae_exclusive_absent/n_landmarks*100:.1f}%)"
        )


if __name__ == "__main__":
    main()
