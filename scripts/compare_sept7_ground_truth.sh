#!/usr/bin/env bash
# Run compare_multi_method_ground_truth.py for every trajectory found under
# docs/results/sept_7_all/pipeline_default/, against lidar ground truth at
# /home/alien/data/yoda/sept_7/<yoda_seq>/<name>/offline/<timestamp>/traj.tum.
#
# For each trajectory, produces two overlay plots + metrics files: one using
# the global-BA pipeline estimates (traj_global_ba.tum) and one using the
# incremental estimates (traj.tum) -- both alongside the three SOTA methods
# (AirSLAM, cuVSLAM, ORBSLAM3).
#
# Usage: compare_sept7_ground_truth.sh [results-root]
#   results-root: default docs/results/sept_7_all
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT=/home/alien/data/yoda/sept_7

RESULTS_ROOT="${1:-$REPO/docs/results/sept_7_all}"

cd "$REPO"
mkdir -p "$RESULTS_ROOT/multi_method"

log() { echo "$(date -Iseconds) $*" | tee -a "$RESULTS_ROOT/compare_multi_method.log"; }

mapfile -t NAMES < <(find "$RESULTS_ROOT/pipeline_default" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | sort)
log "Comparing ${#NAMES[@]} trajectories against ground truth (2 variants each: global_ba, incremental)"

for name in "${NAMES[@]}"; do
  for variant in global_ba incremental; do
    if uv run python scripts/compare_multi_method_ground_truth.py \
        --trajectory-name "$name" \
        --results-root "$RESULTS_ROOT" \
        --data-root "$DATA_ROOT" \
        --variant "$variant" \
        > "$RESULTS_ROOT/multi_method/${name}_${variant}.log" 2>&1; then
      log "  $name [$variant] OK"
    else
      log "  $name [$variant] FAILED -- see $RESULTS_ROOT/multi_method/${name}_${variant}.log"
    fi
  done
done

log "Multi-method ground-truth comparison complete."
