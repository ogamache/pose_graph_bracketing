#!/usr/bin/env bash
# aug_31 batch (fixed layout): every *_region0_occ* trajectory under
# /home/alien/data/yoda/aug_31/<yoda_*>/, run through:
#   1. this repo's pipeline with configs/default.yaml
#   2. this repo's pipeline with clahe_enabled: true (all other CLAHE params unchanged)
#   3. the 3 SOTA baselines (AirSLAM, cuVSLAM, ORB_SLAM3) via analyze_bracketing/vslam_bench
#
# Supersedes scripts/run_aug31_overnight.sh's DATA_ROOT/AUG31_CALIB, which are
# stale against the current disk layout: trajectory data now lives two levels
# deep (<yoda_*>/<..._region0_occN>/data), and calibration files live directly
# under aug_31/calibration/ rather than a calibs/<subdir>/ path.
#
# No ground truth exists yet for aug_31 -- this script only saves trajectories
# and logs, no ATE/RPE comparison.
#
# Sequential: all custom-SLAM runs (both configs) complete before any SOTA
# method starts. Tolerates any single run's failure without aborting the rest.
#
# Usage: run_aug31_all.sh [output-subfolder-name] [sota-methods] [occ-filter]
#   output-subfolder-name: under docs/results/ (default: aug_31_all)
#   sota-methods: space-separated subset of "airslam cuvslam orbslam3" to run
#     (default: all three)
#   occ-filter: "occ0", "occ1", or "both" (default: both)
#   Env var SKIP_PIPELINE=1 skips Part 1 (pipeline_default + pipeline_clahe)
#   entirely, e.g. for a SOTA-only rerun after fixing one method:
#     SKIP_PIPELINE=1 ./scripts/run_aug31_all.sh aug_31_all airslam
#   Env var TRAJ_FILTER restricts discovery to top-level yoda_* dirs whose
#   basename matches this glob, e.g. for a single-trajectory smoke test:
#     TRAJ_FILTER="yoda_bridge_0fps_ae_1_1969_12_31-19_04_09" ./scripts/run_aug31_all.sh aug_31_test airslam occ0
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ANALYZE_BRACKETING="$(cd "$REPO/../analyze_bracketing" && pwd)"
DATA_ROOT="${DATA_ROOT:-/home/alien/data/yoda/aug_31}"
AUG31_CALIB="${AUG31_CALIB:-$DATA_ROOT/calibration}"

OUT_SUBDIR="${1:-aug_31_all}"
SOTA_METHODS="${2:-airslam cuvslam orbslam3}"
OCC_FILTER="${3:-both}"
SKIP_PIPELINE="${SKIP_PIPELINE:-0}"
SKIP_SOTA="${SKIP_SOTA:-0}"
TRAJ_FILTER="${TRAJ_FILTER:-*}"
OUT_ROOT="$REPO/docs/results/$OUT_SUBDIR"
mkdir -p "$OUT_ROOT"

cd "$REPO"

log() { echo "$(date -Iseconds) $*" | tee -a "$OUT_ROOT/batch.log"; }

case "$OCC_FILTER" in
  occ0) OCC_GLOB="*_region0_occ0" ;;
  occ1) OCC_GLOB="*_region0_occ1" ;;
  both) OCC_GLOB="*_region0_occ*" ;;
  *) log "ERROR: invalid occ-filter '$OCC_FILTER' (expected occ0, occ1, or both)"; exit 1 ;;
esac

# Discover all _region0_occN trajectory dirs two levels deep:
# DATA_ROOT/<yoda_*>/<..._region0_occN>
mapfile -t TRAJ_DIRS < <(find "$DATA_ROOT" -mindepth 1 -maxdepth 1 -type d -name "$TRAJ_FILTER" \
  -exec find {} -mindepth 1 -maxdepth 1 -type d -name "$OCC_GLOB" \; | sort)
if [[ "${#TRAJ_DIRS[@]}" -eq 0 ]]; then
  log "ERROR: no $OCC_GLOB trajectories found under $DATA_ROOT (traj-filter=$TRAJ_FILTER)"
  exit 1
fi
log "Discovered ${#TRAJ_DIRS[@]} trajectories under $DATA_ROOT (occ-filter=$OCC_FILTER, traj-filter=$TRAJ_FILTER)"

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

cp "$REPO/configs/default.yaml" "$OUT_ROOT/default_config.yaml"
log "Saved default config -> $OUT_ROOT/default_config.yaml"

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
# Part 2: SOTA methods via analyze_bracketing (runs after all custom-SLAM runs)
# ---------------------------------------------------------------------------
if [[ "$SKIP_SOTA" == "1" ]]; then
  log "SKIP_SOTA=1 -- skipping Part 2 (SOTA methods)"
else
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
fi
log "aug_31 batch complete. Results under $OUT_ROOT"
