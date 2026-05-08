#!/bin/bash
# Fetch the four target output.csvs from S3 (via fetch_outputs.sh) and run
# scripts/eval_parity.sh on each. Tees per-target logs to /tmp.
#
# Run on the EC2 box:
#
#     cd ~/ProDifEvo-Refinement
#     git pull
#     bash scripts/eval_parity_all.sh
#
# Override targets via env var:
#
#     TARGETS="il20 pdl1" bash scripts/eval_parity_all.sh
set -uo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

TARGETS="${TARGETS:-pdl1 bhrf1 il3 il20}"

echo "Step 0: fetching output.csvs from S3..."
bash scripts/fetch_outputs.sh

for T in $TARGETS; do
    INPUT_CSV="${HOME}/Downloads/rerd_${T}.csv"
    if [ ! -f "$INPUT_CSV" ]; then
        echo "[$T] skipping: no input CSV at $INPUT_CSV"
        continue
    fi
    LOG="/tmp/eval_parity_${T}.log"
    echo
    echo "===================== eval_parity: $T ====================="
    echo "  input : $INPUT_CSV"
    echo "  log   : $LOG"
    bash scripts/eval_parity.sh \
        --input-csv "$INPUT_CSV" \
        --antigen "$T" 2>&1 | tee "$LOG"
done

echo
echo "All done. Per-target logs:"
for T in $TARGETS; do
    echo "  /tmp/eval_parity_${T}.log"
done
echo "Comparison CSVs:"
for T in $TARGETS; do
    echo "  /tmp/rerd_parity/${T}/comparison.csv"
done
