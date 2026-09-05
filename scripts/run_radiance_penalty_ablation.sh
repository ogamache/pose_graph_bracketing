#!/usr/bin/env bash
# Ablation: preprocessing.radiance_penalty_enabled off (baseline, current
# default) vs. on (per-keypoint CRF-bayer radiance measured on the
# untouched image, used as a hard match-rejection threshold in
# LightGlueMatcher.match -- see matching.py/graph_builder.py/radiance.py).
#
# Usage: run_radiance_penalty_ablation.sh <trajectory-dir> [output-subfolder-name]
#   trajectory-dir: e.g.
#     /home/alien/data/yoda/aug_31/yoda_bridge_4fps_1_1969_12_31-19_08_37/yoda_bridge_4fps_1_1969_12_31-19_08_37_region0_occ0
#   output-subfolder-name: under docs/results/ (default: radiance_penalty_ablation)
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TRAJ_DIR="${1:?Usage: run_radiance_penalty_ablation.sh <trajectory-dir> [output-subfolder-name]}"
OUT_SUBDIR="${2:-radiance_penalty_ablation}"
OUT_ROOT="$REPO/docs/results/$OUT_SUBDIR"
mkdir -p "$OUT_ROOT"

TRAJ_NAME="$(basename "$TRAJ_DIR")"
DATA_DIR="$TRAJ_DIR/data"

cd "$REPO"

log() { echo "$(date -Iseconds) $*" | tee -a "$OUT_ROOT/batch.log"; }

PENALTY_CONFIG="$OUT_ROOT/radiance_penalty_config.yaml"
python3 - "$REPO/configs/default.yaml" "$PENALTY_CONFIG" <<'PY'
import sys, yaml
src, dst = sys.argv[1], sys.argv[2]
with open(src) as f:
    cfg = yaml.safe_load(f)
cfg["preprocessing"]["radiance_penalty_enabled"] = True
with open(dst, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY
log "Wrote radiance-penalty config -> $PENALTY_CONFIG"
cp "$REPO/configs/default.yaml" "$OUT_ROOT/baseline_config.yaml"
log "Saved baseline config -> $OUT_ROOT/baseline_config.yaml"

run_phase() {
  local phase_name="$1" config_path="$2"
  local run_out="$OUT_ROOT/$phase_name"
  mkdir -p "$run_out"
  cp "$config_path" "$run_out/config_used.yaml"
  log "=== Phase: $phase_name (config=$config_path) ==="
  if uv run python scripts/run_trajectory.py \
      --data-dir "$DATA_DIR" \
      --config "$config_path" \
      --out "$run_out/traj.tum" \
      --global-ba \
      > "$run_out/run.log" 2>&1; then
    log "  [$phase_name] $TRAJ_NAME OK"
  else
    log "  [$phase_name] $TRAJ_NAME FAILED -- see $run_out/run.log"
  fi
}

run_phase "baseline" "$REPO/configs/default.yaml"
run_phase "radiance_penalty" "$PENALTY_CONFIG"

GT_TUM="$(find "$TRAJ_DIR/offline" -mindepth 2 -maxdepth 2 -name "traj.tum" | sort | tail -1)"
if [[ -z "$GT_TUM" ]]; then
  log "WARNING: no ground truth traj.tum found under $TRAJ_DIR/offline -- skipping ATE/RPE comparison"
  exit 0
fi
log "Ground truth: $GT_TUM"

for phase_name in baseline radiance_penalty; do
  run_out="$OUT_ROOT/$phase_name"
  for variant_suffix in "" "_global_ba"; do
    est="$run_out/traj${variant_suffix}.tum"
    [[ -f "$est" ]] || continue
    if uv run python scripts/compare_ground_truth.py \
        --estimate "$est" \
        --ground-truth "$GT_TUM" \
        --out-plot "$run_out/gt_comparison${variant_suffix}.png" \
        > "$run_out/compare_ground_truth${variant_suffix}.log" 2>&1; then
      log "  [$phase_name] gt comparison${variant_suffix} OK -- see $run_out/compare_ground_truth${variant_suffix}.log"
    else
      log "  [$phase_name] gt comparison${variant_suffix} FAILED -- see $run_out/compare_ground_truth${variant_suffix}.log"
    fi
  done
done

log "Radiance-penalty ablation complete. Results under $OUT_ROOT"
