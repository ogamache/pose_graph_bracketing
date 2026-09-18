#!/usr/bin/env bash
# Two-step batch after the SLAM pipeline was updated post-aug_31:
#   1. Re-run this repo's custom SLAM (default + CLAHE configs) on the aug_31
#      data, overwriting pipeline_default/pipeline_clahe in
#      docs/results/aug_31_all/ in place. SOTA results there (already current)
#      are left untouched.
#   2. Run the full comparison (custom SLAM x2 + all 3 SOTA methods) on the
#      sept_7 data, into docs/results/sept_7_all/.
#
# Usage: run_aug31_refresh_and_sept7.sh [aug31-occ-filter] [sept7-occ-filter]
#   occ-filter: "occ0", "occ1", or "both" (default: both, both)
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

AUG31_OCC="${1:-both}"
SEPT7_OCC="${2:-both}"

echo "=== Step 1/2: refreshing custom-SLAM results on aug_31 (SOTA untouched) ==="
DATA_ROOT=/home/alien/data/yoda/aug_31 \
SKIP_SOTA=1 \
"$REPO/scripts/run_aug31_all.sh" aug_31_all "" "$AUG31_OCC"
step1_status=$?

echo "=== Step 2/2: full run (custom SLAM x2 + 3 SOTA) on sept_7 ==="
DATA_ROOT=/home/alien/data/yoda/sept_7 \
"$REPO/scripts/run_aug31_all.sh" sept_7_all "airslam cuvslam orbslam3" "$SEPT7_OCC"
step2_status=$?

echo "Step 1 (aug_31 refresh) exit=$step1_status; Step 2 (sept_7 full) exit=$step2_status"
