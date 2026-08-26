#!/usr/bin/env bash
# Overnight full-trajectory batch on the new default config (zero_motion,
# global_bundle_adjust, block_lae_sae_matches, drop_low_info_frames, CLAHE
# clip=20 -- see configs/default.yaml). 11 runs: the 4 aug_9 datasets used
# throughout the region3 investigation (now on their FULL trajectories) plus
# all 7 aug_25 runs (new data, shared calibration/, ae_zone-based exposure
# labeling). Sequential (single GPU); tolerates a single run's failure
# without aborting the rest; logs progress per run.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="$REPO/docs/results/2026_08_26_full_batch"
mkdir -p "$OUT_ROOT"

AUG25_CALIB=/home/alien/data/yoda/aug_25/calibration

# name | data_dir | ground_truth | calibration_dir (empty = default data_dir/calibration)
RUNS=(
  "b_0fps|/home/alien/data/yoda/aug_9/yoda_aug_9_b_0fps_02ema_1969_12_31-19_13_05/data|/home/alien/data/yoda/aug_9/yoda_aug_9_b_0fps_02ema_1969_12_31-19_13_05/offline/2026_08_11-18_27_17/traj.tum|"
  "ae_0fps|/home/alien/data/yoda/aug_9/yoda_aug_9_ae_0fps_02ema_1969_12_31-19_28_58/data|/home/alien/data/yoda/aug_9/yoda_aug_9_ae_0fps_02ema_1969_12_31-19_28_58/offline/2026_08_23-15_53_53/traj.tum|"
  "b_10fps|/home/alien/data/yoda/aug_9/yoda_aug_9_b_10fps_02ema_1969_12_31-19_27_24/data|/home/alien/data/yoda/aug_9/yoda_aug_9_b_10fps_02ema_1969_12_31-19_27_24/offline/2026_08_11-18_51_00/traj.tum|"
  "ae_10fps|/home/alien/data/yoda/aug_9/yoda_aug_9_ae_10fps_02ema_1969_12_31-19_25_01/data|/home/alien/data/yoda/aug_9/yoda_aug_9_ae_10fps_02ema_1969_12_31-19_25_01/offline/2026_08_11-19_01_45/traj.tum|"
  "vachon_n1_0fps|/home/alien/data/yoda/aug_25/yoda_vachon_n1_0fps_1969_12_31-19_18_57/data|/home/alien/data/yoda/aug_25/yoda_vachon_n1_0fps_1969_12_31-19_18_57/offline/2026_08_26-00_09_24/traj.tum|$AUG25_CALIB"
  "vachon_n1_5fps|/home/alien/data/yoda/aug_25/yoda_vachon_n1_5fps_1969_12_31-19_25_46/data|/home/alien/data/yoda/aug_25/yoda_vachon_n1_5fps_1969_12_31-19_25_46/offline/2026_08_26-00_10_25/traj.tum|$AUG25_CALIB"
  "vachon_n3_0fps|/home/alien/data/yoda/aug_25/yoda_vachon_n3_0fps_1969_12_31-19_20_55/data|/home/alien/data/yoda/aug_25/yoda_vachon_n3_0fps_1969_12_31-19_20_55/offline/2026_08_26-00_11_35/traj.tum|$AUG25_CALIB"
  "vachon_n3_0fps_equal|/home/alien/data/yoda/aug_25/yoda_vachon_n3_0fps_equal_1969_12_31-19_22_33/data|/home/alien/data/yoda/aug_25/yoda_vachon_n3_0fps_equal_1969_12_31-19_22_33/offline/2026_08_26-00_12_32/traj.tum|$AUG25_CALIB"
  "vachon_n3_5fps|/home/alien/data/yoda/aug_25/yoda_vachon_n3_5fps_1969_12_31-19_24_23/data|/home/alien/data/yoda/aug_25/yoda_vachon_n3_5fps_1969_12_31-19_24_23/offline/2026_08_26-00_13_47/traj.tum|$AUG25_CALIB"
  "vachon_tunnel_n1_5fps|/home/alien/data/yoda/aug_25/yoda_vachon_tunnel_n1_5fps_1969_12_31-19_39_06/data|/home/alien/data/yoda/aug_25/yoda_vachon_tunnel_n1_5fps_1969_12_31-19_39_06/offline/2026_08_26-00_16_03/traj.tum|$AUG25_CALIB"
  "vachon_tunnel_n3_5fps|/home/alien/data/yoda/aug_25/yoda_vachon_tunnel_n3_5fps_1969_12_31-19_40_28/data|/home/alien/data/yoda/aug_25/yoda_vachon_tunnel_n3_5fps_1969_12_31-19_40_28/offline/2026_08_26-00_18_37/traj.tum|$AUG25_CALIB"
)

cd "$REPO"

echo "$(date -Iseconds) Starting full batch (${#RUNS[@]} runs) -> $OUT_ROOT"

for entry in "${RUNS[@]}"; do
  IFS='|' read -r name data_dir gt calib_dir <<< "$entry"
  run_out="$OUT_ROOT/$name"
  mkdir -p "$run_out"
  echo "$(date -Iseconds) === $name ===" | tee -a "$OUT_ROOT/batch.log"
  echo "  data_dir=$data_dir" >> "$OUT_ROOT/batch.log"
  echo "  ground_truth=$gt" >> "$OUT_ROOT/batch.log"

  extra_args=()
  if [[ -n "$calib_dir" ]]; then
    extra_args+=(--calibration-dir "$calib_dir")
  fi

  if uv run python scripts/run_trajectory.py \
      --data-dir "$data_dir" \
      --out "$run_out/traj.tum" \
      "${extra_args[@]}" \
      > "$run_out/run.log" 2>&1; then
    echo "$(date -Iseconds)   OK" | tee -a "$OUT_ROOT/batch.log"
  else
    echo "$(date -Iseconds)   FAILED (exit $?) -- see $run_out/run.log" | tee -a "$OUT_ROOT/batch.log"
  fi

  # ground_truth.tum copied alongside for the analysis script's convenience
  cp "$gt" "$run_out/ground_truth.tum" 2>/dev/null || echo "  WARNING: could not copy ground truth from $gt" >> "$OUT_ROOT/batch.log"
done

echo "$(date -Iseconds) Batch complete." | tee -a "$OUT_ROOT/batch.log"
