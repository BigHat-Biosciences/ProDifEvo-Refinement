#!/bin/bash
# Score a set of sequences with eval_iptm.py TWICE (separate cache dirs, same
# env, same template/hotspot/recycles) and report the per-sequence ipTM delta
# between the two runs. This characterizes the AF2 stochasticity floor for our
# multi-GPU dispatch — useful as a baseline to compare against RERD↔bonobo
# parity deltas.
#
# Interpretation:
#   * mean |delta| ~0.003 → tight floor; any cross-evaluator gap above that is real.
#   * mean |delta| ~0.008 → floor is loose; cross-evaluator parity at ~0.01 is fine.
#
# Run on the EC2 box:
#
#     cd ~/ProDifEvo-Refinement
#     bash scripts/eval_noise_floor.sh \
#         --input-csv /path/to/sequences.csv \
#         --antigen il20
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

INPUT_CSV="${INPUT_CSV:-}"
ANTIGEN="${ANTIGEN:-}"
while [ $# -gt 0 ]; do
    case "$1" in
        --input-csv) INPUT_CSV="$2"; shift 2 ;;
        --antigen)   ANTIGEN="$2"; shift 2 ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$INPUT_CSV" ] || [ -z "$ANTIGEN" ]; then
    echo "usage: bash scripts/eval_noise_floor.sh --input-csv <csv> --antigen <name>"
    exit 1
fi
if [ ! -f "$INPUT_CSV" ]; then
    echo "ERROR: --input-csv not found: $INPUT_CSV"
    exit 1
fi

AF_GPU_IDS="${AF_GPU_IDS:-1,2,3}"
RERD_CONDA_ENV="${RERD_CONDA_ENV:-RERD}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/rerd_noise_floor/${ANTIGEN}}"

declare -A HOTSPOTS=(
    [pdl1]="A113"
    [bhrf1]="A60,A61,A63,A71"
    [il3]="A23,A25,A26,A31,A40,A104"
    [il20]="A58,A62,A101"
)
HOTSPOT="${HOTSPOTS[$ANTIGEN]:-}"
TEMPLATE_PDB="${REPO_DIR}/datasets/template_${ANTIGEN}.pdb"
ANTIGEN_PDB="${REPO_DIR}/datasets/${ANTIGEN}.pdb"

if [ -z "$HOTSPOT" ]; then
    echo "ERROR: no baked hotspot for antigen=$ANTIGEN"
    exit 1
fi
if [ ! -f "$TEMPLATE_PDB" ]; then
    echo "ERROR: template PDB not found: $TEMPLATE_PDB"
    exit 1
fi
if [ ! -f "$ANTIGEN_PDB" ]; then
    echo "ERROR: antigen PDB not found: $ANTIGEN_PDB"
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mkdir -p "$OUTPUT_ROOT"
RUN1_DIR="${OUTPUT_ROOT}/run1"
RUN2_DIR="${OUTPUT_ROOT}/run2"
mkdir -p "$RUN1_DIR" "$RUN2_DIR"

INPUT_BASENAME="$(basename "$INPUT_CSV")"
RUN1_INPUT="${RUN1_DIR}/${INPUT_BASENAME}"
RUN2_INPUT="${RUN2_DIR}/${INPUT_BASENAME}"
cp "$INPUT_CSV" "$RUN1_INPUT"
cp "$INPUT_CSV" "$RUN2_INPUT"

for CONDA_SH in /opt/conda/etc/profile.d/conda.sh "${HOME}/miniconda3/etc/profile.d/conda.sh" "${HOME}/anaconda3/etc/profile.d/conda.sh"; do
    if [ -f "$CONDA_SH" ]; then
        # shellcheck disable=SC1090
        source "$CONDA_SH"
        break
    fi
done
conda activate "$RERD_CONDA_ENV"

run_eval() {
    local label="$1"
    local in_csv="$2"
    local cache_dir="$3"
    echo "=========================================================="
    echo "$label: eval_iptm.py"
    echo "  input : $in_csv"
    echo "  cache : $cache_dir"
    echo "=========================================================="
    python scripts/eval_iptm.py \
        --input_csv "$in_csv" \
        --antigen_pdb "$ANTIGEN_PDB" \
        --antigen_chain A \
        --template_pdb "$TEMPLATE_PDB" \
        --hotspot "$HOTSPOT" \
        --af_gpu_ids "$AF_GPU_IDS" \
        --cache_dir "$cache_dir" \
        --write_inplace 0
}

run_eval "Step 1/3: run #1" "$RUN1_INPUT" "${RUN1_DIR}/cache"
RUN1_CSV="${RUN1_INPUT%.csv}_w_final_iptm.csv"
[ -f "$RUN1_CSV" ] || { echo "ERROR: run1 output missing at $RUN1_CSV"; exit 1; }

echo
run_eval "Step 2/3: run #2" "$RUN2_INPUT" "${RUN2_DIR}/cache"
RUN2_CSV="${RUN2_INPUT%.csv}_w_final_iptm.csv"
[ -f "$RUN2_CSV" ] || { echo "ERROR: run2 output missing at $RUN2_CSV"; exit 1; }

echo
echo "=========================================================="
echo "Step 3/3: noise-floor comparison (run1 vs run2)"
echo "=========================================================="
python - <<PY
import pandas as pd

r1 = pd.read_csv("$RUN1_CSV")[["sequence", "final_iptm"]].rename(columns={"final_iptm": "iptm_run1"})
r2 = pd.read_csv("$RUN2_CSV")[["sequence", "final_iptm"]].rename(columns={"final_iptm": "iptm_run2"})
m = r1.merge(r2, on="sequence", how="inner")
m["delta"] = m["iptm_run2"] - m["iptm_run1"]
m["abs_delta"] = m["delta"].abs()

print()
print(f"Sequences scored by both runs: {len(m)} / {len(r1)}")
print()
print("Per-row:")
print(m[["iptm_run1", "iptm_run2", "delta"]].to_string(
    index=False, float_format=lambda x: f"{x:+.4f}"))
print()

s = m["delta"]
a = m["abs_delta"]
print("=== Noise floor stats ===")
print(f"  delta (signed): mean={s.mean():+.4f}  std={s.std():.4f}  max={s.max():+.4f}  min={s.min():+.4f}")
print(f"  |delta|       : mean={a.mean():.4f}  std={a.std():.4f}  max={a.max():.4f}")
print()
print("Read against your RERD↔bonobo parity test:")
print("  if mean |delta| here >= bonobo parity max |delta|, parity is at floor.")
print("  if mean |delta| here <  bonobo parity max |delta|, residual conditioning gap exists.")

out = "${OUTPUT_ROOT}/comparison.csv"
m.to_csv(out, index=False)
print()
print(f"Merged comparison written to: {out}")
PY
