#!/usr/bin/env bash
# sept_7 hdrflow batch: every *_region0_occN trajectory under
# /home/alien/data/yoda/sept_7/<yoda_*>/ that has a hdrflow/images_left
# folder, run through:
#   1. this repo's pipeline (default config, then CLAHE config) reading
#      images from hdrflow/ instead of data/ (--image-dir), calibration and
#      images_meta_* still from data/
#   2. the 3 SOTA baselines (AirSLAM, cuVSLAM, ORB_SLAM3) via
#      analyze_bracketing/vslam_bench, with --image-subdir hdrflow
#   3. per-trajectory ATE/RPE comparison against the trajectory's existing
#      ground truth (same GT as the standard-image run -- hdrflow is just an
#      alternate image set for the same physical run)
#
# All results are filed under docs/results/sept_7_all/ alongside the
# existing standard-image results, with every trajectory name suffixed
# "_hdrflow" (pipeline_default/<name>_hdrflow/, sota/<Method>/<name>_hdrflow.txt,
# multi_method/<name>_hdrflow/) so nothing standard-image gets overwritten.
#
# Usage: run_sept7_hdrflow.sh [output-subfolder-name] [sota-methods] [occ-filter]
#   output-subfolder-name: under docs/results/ (default: sept_7_all)
#   sota-methods: space-separated subset of "airslam cuvslam orbslam3" to run
#     (default: all three)
#   occ-filter: "occ0", "occ1", or "both" (default: both)
#   Env var SKIP_PIPELINE=1 skips Part 1 (pipeline_default + pipeline_clahe)
#   Env var SKIP_SOTA=1 skips Part 2 (SOTA methods)
#   Env var SKIP_COMPARE=1 skips Part 3 (multi_method comparison)
#   Env var TRAJ_FILTER restricts discovery to top-level yoda_* dirs whose
#   basename matches this glob, e.g. for a single-trajectory smoke test:
#     TRAJ_FILTER="yoda_ae_4fps_1_1969_12_31-19_50_31" ./scripts/run_sept7_hdrflow.sh sept_7_all orbslam3 occ0
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ANALYZE_BRACKETING="$(cd "$REPO/../analyze_bracketing" && pwd)"
DATA_ROOT="${DATA_ROOT:-/home/alien/data/yoda/sept_7}"

OUT_SUBDIR="${1:-sept_7_all}"
SOTA_METHODS="${2:-airslam cuvslam orbslam3}"
OCC_FILTER="${3:-both}"
SKIP_PIPELINE="${SKIP_PIPELINE:-0}"
SKIP_SOTA="${SKIP_SOTA:-0}"
SKIP_COMPARE="${SKIP_COMPARE:-0}"
TRAJ_FILTER="${TRAJ_FILTER:-*}"
OUT_ROOT="$REPO/docs/results/$OUT_SUBDIR"
mkdir -p "$OUT_ROOT"

cd "$REPO"

log() { echo "$(date -Iseconds) $*" | tee -a "$OUT_ROOT/batch_hdrflow.log"; }

case "$OCC_FILTER" in
  occ0) OCC_GLOB="*_region0_occ0" ;;
  occ1) OCC_GLOB="*_region0_occ1" ;;
  both) OCC_GLOB="*_region0_occ*" ;;
  *) log "ERROR: invalid occ-filter '$OCC_FILTER' (expected occ0, occ1, or both)"; exit 1 ;;
esac

# Discover all _region0_occN trajectory dirs two levels deep that have a
# hdrflow/images_left folder: DATA_ROOT/<yoda_*>/<..._region0_occN>/hdrflow/images_left
mapfile -t TRAJ_DIRS < <(find "$DATA_ROOT" -mindepth 1 -maxdepth 1 -type d -name "$TRAJ_FILTER" \
  -exec find {} -mindepth 1 -maxdepth 1 -type d -name "$OCC_GLOB" \; | sort \
  | while read -r d; do [[ -d "$d/hdrflow/images_left" ]] && echo "$d"; done)
if [[ "${#TRAJ_DIRS[@]}" -eq 0 ]]; then
  log "ERROR: no $OCC_GLOB trajectories with hdrflow/images_left found under $DATA_ROOT (traj-filter=$TRAJ_FILTER)"
  exit 1
fi
log "Discovered ${#TRAJ_DIRS[@]} hdrflow trajectories under $DATA_ROOT (occ-filter=$OCC_FILTER, traj-filter=$TRAJ_FILTER)"

# ---------------------------------------------------------------------------
# Part 1: this repo's pipeline (default config, then CLAHE config), hdrflow images
# ---------------------------------------------------------------------------
CLAHE_CONFIG="$OUT_ROOT/clahe_config.yaml"
if [[ ! -f "$CLAHE_CONFIG" ]]; then
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
fi

run_pipeline_phase() {
  local phase_name="$1" config_path="$2"
  local phase_root="$OUT_ROOT/$phase_name"
  mkdir -p "$phase_root"
  log "=== Pipeline phase: $phase_name (hdrflow, config=$config_path) ==="
  for traj_dir in "${TRAJ_DIRS[@]}"; do
    local name; name="$(basename "$traj_dir")_hdrflow"
    local run_out="$phase_root/$name"
    mkdir -p "$run_out"
    cp "$config_path" "$run_out/config_used.yaml"
    if uv run python scripts/run_trajectory.py \
        --data-dir "$traj_dir/data" \
        --image-dir "$traj_dir/hdrflow" \
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
# Part 2: SOTA methods via analyze_bracketing, hdrflow images
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
    log "=== SOTA method: $method (hdrflow) ==="
    if uv run python scripts/run_vslam_benchmark.py \
        --method "$method" \
        --trajectory-paths "${TRAJ_DIRS[@]}" \
        --image-subdir hdrflow \
        > "$OUT_ROOT/sota_${method}_hdrflow.log" 2>&1; then
      log "  [$method] benchmark script OK"
    else
      log "  [$method] benchmark script FAILED (exit $?) -- see $OUT_ROOT/sota_${method}_hdrflow.log"
    fi

    mkdir -p "$SOTA_OUT/$results_name"
    for traj_dir in "${TRAJ_DIRS[@]}"; do
      name="$(basename "$traj_dir")"
      src="$ANALYZE_BRACKETING/results/$results_name/$name.txt"
      dst="$SOTA_OUT/$results_name/${name}_hdrflow.txt"
      if [[ -f "$src" ]]; then
        cp "$src" "$dst"
        log "  [$method] copied ${name}_hdrflow.txt"
      else
        log "  [$method] WARNING: missing output for $name (expected $src)"
      fi
    done
  done

  cd "$REPO"
fi

# ---------------------------------------------------------------------------
# Part 3: multi_method ATE/RPE comparison (reuses the trajectory's existing
# ground truth; only the result-file naming is suffixed "_hdrflow")
# ---------------------------------------------------------------------------
if [[ "$SKIP_COMPARE" == "1" ]]; then
  log "SKIP_COMPARE=1 -- skipping Part 3 (multi_method comparison)"
else
  log "=== Comparison phase (hdrflow) ==="
  for traj_dir in "${TRAJ_DIRS[@]}"; do
    name="$(basename "$traj_dir")"
    for variant in incremental global_ba; do
      if uv run python scripts/compare_multi_method_ground_truth.py \
          --trajectory-name "$name" \
          --result-name "${name}_hdrflow" \
          --results-root "$OUT_ROOT" \
          --data-root "$DATA_ROOT" \
          --variant "$variant" \
          > "$OUT_ROOT/compare_hdrflow_${name}_${variant}.log" 2>&1; then
        log "  [compare] ${name}_hdrflow ($variant) OK"
      else
        log "  [compare] ${name}_hdrflow ($variant) FAILED -- see $OUT_ROOT/compare_hdrflow_${name}_${variant}.log"
      fi
    done
  done
fi

log "sept_7 hdrflow batch complete. Results under $OUT_ROOT"
