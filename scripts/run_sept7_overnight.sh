#!/usr/bin/env bash
# One-shot overnight driver for the sept_7_all update:
#   1. Rerun ORB-SLAM3 on standard images (new ORB_SLAM3 repo config),
#      overwriting docs/results/sept_7_all/sota/ORBSLAM3/*.txt in place.
#   2. Refresh multi_method ATE/RPE comparisons for every existing
#      (standard-image) trajectory so they reflect the new ORB-SLAM3 numbers.
#   3. Run the full pipeline (custom x2 configs + AirSLAM + cuVSLAM +
#      ORB-SLAM3) on every trajectory's hdrflow/ images, filing results as
#      <name>_hdrflow entries alongside the standard-image ones.
#
# Usage: ./scripts/run_sept7_overnight.sh
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="/home/alien/data/yoda/sept_7"
OUT_SUBDIR="sept_7_all"
OUT_ROOT="$REPO/docs/results/$OUT_SUBDIR"

cd "$REPO"

log() { echo "$(date -Iseconds) $*" | tee -a "$OUT_ROOT/overnight.log"; }

log "=== Step 1/3: rerun ORB-SLAM3 on standard images ==="
SKIP_PIPELINE=1 DATA_ROOT="$DATA_ROOT" ./scripts/run_aug31_all.sh "$OUT_SUBDIR" orbslam3 both
log "Step 1/3 done"

log "=== Step 2/3: refresh multi_method comparisons (standard images) ==="
for d in "$OUT_ROOT"/multi_method/*/; do
  name="$(basename "$d")"
  for variant in incremental global_ba; do
    if uv run python scripts/compare_multi_method_ground_truth.py \
        --trajectory-name "$name" \
        --results-root "$OUT_ROOT" \
        --data-root "$DATA_ROOT" \
        --variant "$variant" \
        > "$OUT_ROOT/compare_refresh_${name}_${variant}.log" 2>&1; then
      log "  [compare] $name ($variant) OK"
    else
      log "  [compare] $name ($variant) FAILED -- see $OUT_ROOT/compare_refresh_${name}_${variant}.log"
    fi
  done
done
log "Step 2/3 done"

log "=== Step 3/3: full pipeline on hdrflow images ==="
./scripts/run_sept7_hdrflow.sh "$OUT_SUBDIR"
log "Step 3/3 done"

log "Overnight run complete. Results under $OUT_ROOT"
