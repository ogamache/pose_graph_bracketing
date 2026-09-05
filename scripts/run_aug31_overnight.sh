#!/usr/bin/env bash
# Overnight aug_31 batch: every *_region0_* trajectory under
# /home/alien/data/yoda/aug_31/data_1st_runs, run through:
#   1. this repo's pipeline with configs/default.yaml
#   2. this repo's pipeline with clahe_enabled: true (all other CLAHE params unchanged)
#   3. the 3 SOTA baselines (AirSLAM, cuVSLAM, ORB_SLAM3) via analyze_bracketing/vslam_bench
#
# No ground truth exists yet for aug_31 -- this script only saves trajectories
# and logs, no ATE/RPE comparison.
#
# Sequential, tolerates any single run's failure without aborting the rest.
# Usage: run_aug31_overnight.sh [output-subfolder-name] [sota-methods]
#   output-subfolder-name: under docs/results/ (default: aug_31)
#   sota-methods: space-separated subset of "airslam cuvslam orbslam3" to run
#     (default: all three). Env var SKIP_PIPELINE=1 skips Part 1 (pipeline_default
#     + pipeline_clahe) entirely, e.g. for a SOTA-only rerun after fixing one method:
#       SKIP_PIPELINE=1 ./scripts/run_aug31_overnight.sh aug_31 airslam
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ANALYZE_BRACKETING="$(cd "$REPO/../analyze_bracketing" && pwd)"
DATA_ROOT=/home/alien/data/yoda/aug_31/data_1st_runs
AUG31_CALIB=/home/alien/data/yoda/aug_31/calibs/yoda_stereo_1969_12_31-19_10_46

OUT_SUBDIR="${1:-aug_31}"
SOTA_METHODS="${2:-airslam cuvslam orbslam3}"
SKIP_PIPELINE="${SKIP_PIPELINE:-0}"
OUT_ROOT="$REPO/docs/results/$OUT_SUBDIR"
mkdir -p "$OUT_ROOT"

cd "$REPO"

log() { echo "$(date -Iseconds) $*" | tee -a "$OUT_ROOT/batch.log"; }

# Discover all _region0_ trajectory dirs
mapfile -t TRAJ_DIRS < <(find "$DATA_ROOT" -maxdepth 1 -mindepth 1 -type d -name "*_region0_*" | sort)
if [[ "${#TRAJ_DIRS[@]}" -eq 0 ]]; then
  log "ERROR: no *_region0_* trajectories found under $DATA_ROOT"
  exit 1
fi
log "Discovered ${#TRAJ_DIRS[@]} _region0_ trajectories under $DATA_ROOT"

# ---------------------------------------------------------------------------
# Part 1: this repo's pipeline (default config, then CLAHE config)
# ---------------------------------------------------------------------------
CLAHE_CONFIG="$OUT_ROOT/clahe_config.yaml"
python3 - "$REPO/configs/default.yaml" "$CLAHE_CONFIG" <<'PY'
import sys, yaml
src, dst = sys.argv[1], sys.argv[2]
with open(src) as f:
    cfg = yaml.safe_load(f)
cfg["preprocessing"]["clahe_enabled"] = True
with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY
log "Wrote CLAHE config -> $CLAHE_CONFIG"

run_pipeline_phase() {
  local phase_name="$1" config_path="$2"
  local phase_root="$OUT_ROOT/$phase_name"
  mkdir -p "$phase_root"
  log "=== Pipeline phase: $phase_name (config=$config_path) ==="
  for traj_dir in "${TRAJ_DIRS[@]}"; do
    local name; name="$(basename "$traj_dir")"
    local run_out="$phase_root/$name"
    mkdir -p "$run_out"
    cp "$config_path" "$run_out/config_used.yaml"
    if uv run python scripts/run_trajectory.py \
        --data-dir "$traj_dir/data" \
        --calibration-dir "$AUG31_CALIB" \
        --config "$config_path" \
        --out "$run_out/traj.tum" \
        > "$run_out/run.log" 2>&1; then
      log "  [$phase_name] $name OK"
    else
      log "  [$phase_name] $name FAILED (exit $?) -- see $run_out/run.log"
    fi
  done
}

if [[ "$SKIP_PIPELINE" == "1" ]]; then
  log "SKIP_PIPELINE=1 -- skipping Part 1 (pipeline_default, pipeline_clahe)"
else
  run_pipeline_phase "pipeline_default" "$REPO/configs/default.yaml"
  run_pipeline_phase "pipeline_clahe" "$CLAHE_CONFIG"
fi

# ---------------------------------------------------------------------------
# Part 2: SOTA methods via analyze_bracketing
# ---------------------------------------------------------------------------
SOTA_OUT="$OUT_ROOT/sota"
mkdir -p "$SOTA_OUT"

declare -A METHOD_RESULTS_NAME=(
  [airslam]="AirSLAM"
  [cuvslam]="cuVSLAM"
  [orbslam3]="ORBSLAM3"
)

cd "$ANALYZE_BRACKETING"
for method in $SOTA_METHODS; do
  results_name="${METHOD_RESULTS_NAME[$method]}"
  log "=== SOTA method: $method ==="
  if uv run python scripts/run_vslam_benchmark.py \
      --method "$method" \
      --trajectory-paths "${TRAJ_DIRS[@]}" \
      > "$OUT_ROOT/sota_${method}.log" 2>&1; then
    log "  [$method] benchmark script OK"
  else
    log "  [$method] benchmark script FAILED (exit $?) -- see $OUT_ROOT/sota_${method}.log"
  fi

  mkdir -p "$SOTA_OUT/$results_name"
  for traj_dir in "${TRAJ_DIRS[@]}"; do
    name="$(basename "$traj_dir")"
    src="$ANALYZE_BRACKETING/results/$results_name/$name.txt"
    if [[ -f "$src" ]]; then
      cp "$src" "$SOTA_OUT/$results_name/$name.txt"
      log "  [$method] copied $name.txt"
    else
      log "  [$method] WARNING: missing output for $name (expected $src)"
    fi
  done
done

cd "$REPO"
log "Overnight aug_31 batch complete. Results under $OUT_ROOT"
