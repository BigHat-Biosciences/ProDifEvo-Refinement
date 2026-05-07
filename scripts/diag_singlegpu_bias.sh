#!/bin/bash
# Diagnostic: run a small RERD job on a SINGLE GPU (no multi-GPU dispatch, so
# no race condition possible), then immediately re-eval the resulting output.csv
# with eval_iptm.py (also single-GPU). Compare per-row.
#
# If the systematic +bias (output.csv iptm > eval iptm) you saw on the multi-GPU
# runs ALSO appears here, the race condition isn't the cause; we have a deeper
# bug. If the deltas are near zero, race + selection-bias-from-race is the cause.
#
# Run on the EC2 box:
#
#     cd ~/ProDifEvo-Refinement
#     git pull
#     bash scripts/diag_singlegpu_bias.sh
#
# Override defaults via env vars:
#
#     ANTIGEN=il3 REPEAT_NUM=20 ITERATION=5 bash scripts/diag_singlegpu_bias.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

# ---- Knobs ----
ANTIGEN="${ANTIGEN:-pdl1}"
REPEAT_NUM="${REPEAT_NUM:-8}"      # small, so the run finishes in ~30 min
ITERATION="${ITERATION:-3}"        # 3 iters is enough to see selection effects
DUPLICATE="${DUPLICATE:-5}"
CONDA_ENV="${CONDA_ENV:-RERD}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/rerd_singlegpu_diag}"
SEED_SEQUENCE="${SEED_SEQUENCE:-EVQLVESGGGLVQPGGSLRLSCAASGGFTFSSYAMWFRQAPGKEREFAISGSGGSTYYNADSVKGRFTISRDNAKNTLYLQMNSLRAEDTAVYYCARLSITIRPYYGWGQGTLVTVSS}"

# Per-target hotspots (mirror bonobo's TARGETS dict).
declare -A HOTSPOTS=(
    [pdl1]="A113"
    [bhrf1]="A60,A61,A63,A71"
    [il3]="A23,A25,A26,A31,A40,A104"
    [il20]="A58,A62,A101"
)
HOTSPOT="${HOTSPOTS[$ANTIGEN]:-}"
TEMPLATE_PDB="${REPO_DIR}/datasets/template_${ANTIGEN}.pdb"
if [ ! -f "$TEMPLATE_PDB" ]; then
    echo "ERROR: template not found at $TEMPLATE_PDB. Generate with scripts/generate_template.py."
    exit 1
fi

# Pin to a single GPU so neither the design nor the eval can see other devices.
# (Even with af_gpu_ids="", we want CUDA_VISIBLE_DEVICES restricted so JAX/NBB2
# can't accidentally use a different device than expected.)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Critical: with JAX and torch both on the same GPU, JAX's default 90%
# preallocation starves torch and cuBLAS fails to init. Disable preallocation
# so JAX grows its memory pool dynamically alongside torch.
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mkdir -p "$OUTPUT_ROOT"
ANTIGEN_PDB="${REPO_DIR}/datasets/${ANTIGEN}.pdb"
if [ ! -f "$ANTIGEN_PDB" ]; then
    echo "ERROR: $ANTIGEN_PDB not found. ANTIGEN must be one of {pdl1, bhrf1, il3, il20}."
    exit 1
fi

# ---- Activate conda env ----
for CONDA_SH in /opt/conda/etc/profile.d/conda.sh "${HOME}/miniconda3/etc/profile.d/conda.sh" "${HOME}/anaconda3/etc/profile.d/conda.sh"; do
    if [ -f "$CONDA_SH" ]; then
        # shellcheck disable=SC1090
        source "$CONDA_SH"
        break
    fi
done
conda activate "$CONDA_ENV"
echo "conda env: $CONDA_ENV ($(python -c 'import sys; print(sys.executable)'))"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo

# ---- Step 1: small single-GPU RERD generation ----
echo "=== Step 1: RERD generation (single GPU, --af_gpu_ids '') ==="
echo "  antigen=$ANTIGEN  repeat_num=$REPEAT_NUM  iteration=$ITERATION"
RUN_NAME="diag_singlegpu_${ANTIGEN}"
python ab_refinement.py \
    --antibody_sequence "$SEED_SEQUENCE" \
    --antigen_pdb "$ANTIGEN_PDB" \
    --antigen_chain A \
    --chain_type heavy \
    --cdrs_to_design H1,H2,H3 \
    --numbering_scheme imgt \
    --metrics_name iptm,cdr_plddt,plddt \
    --metrics_list 3,1,1 \
    --repeatnum "$REPEAT_NUM" \
    --duplicate "$DUPLICATE" \
    --iteration "$ITERATION" \
    --decoding SVDD_edit \
    --num_recycles 3 \
    --af_models 0 \
    --template_pdb "$TEMPLATE_PDB" \
    --hotspot "$HOTSPOT" \
    --output_root "$OUTPUT_ROOT" \
    --run_name "$RUN_NAME"

# ---- Step 2: locate the run dir ----
RUN_DIR=$(find "$OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -type d -name '*_ab_*' | sort | tail -1)
OUTPUT_CSV="$RUN_DIR/output.csv"
if [ ! -f "$OUTPUT_CSV" ]; then
    echo "ERROR: expected output.csv at $OUTPUT_CSV"
    exit 1
fi
echo
echo "=== Step 2: run dir: $RUN_DIR ==="
echo

# ---- Step 3: re-eval with eval_iptm.py (also single GPU) ----
echo "=== Step 3: re-eval (single GPU, fresh AF prediction per sequence) ==="
python scripts/eval_iptm.py \
    --input_csv "$OUTPUT_CSV" \
    --antigen_pdb "$ANTIGEN_PDB" \
    --antigen_chain A \
    --template_pdb "$TEMPLATE_PDB" \
    --hotspot "$HOTSPOT" \
    --af_gpu_ids "" \
    --cache_dir "${OUTPUT_ROOT}/iptm_cache" \
    --write_inplace 0

# ---- Step 4: per-row comparison ----
EVAL_CSV="${OUTPUT_CSV%.csv}_w_final_iptm.csv"
if [ ! -f "$EVAL_CSV" ]; then
    echo "ERROR: expected $EVAL_CSV"
    exit 1
fi
echo
echo "=== Step 4: per-row comparison ==="
python - <<PY
import pandas as pd
df = pd.read_csv("$EVAL_CSV")
delta = df["final_iptm"] - df["iptm"]
n = len(df)
n_pos = int((delta > 0.001).sum())
n_neg = int((delta < -0.001).sum())
n_zero = n - n_pos - n_neg
print(f"orig (output.csv):  mean={df['iptm'].mean():.4f}  range=[{df['iptm'].min():.4f}, {df['iptm'].max():.4f}]")
print(f"fresh (re-eval):    mean={df['final_iptm'].mean():.4f}  range=[{df['final_iptm'].min():.4f}, {df['final_iptm'].max():.4f}]")
print(f"delta (fresh-orig): mean={delta.mean():+.4f}  std={delta.std():.4f}")
print(f"                    max={delta.max():+.4f}  min={delta.min():+.4f}")
print(f"                    {n_pos} fresh>orig, {n_neg} fresh<orig, {n_zero} ~equal (out of {n})")
print()
print("Per-row:")
for i, row in df.iterrows():
    print(f"  [{i}] orig={row['iptm']:+.4f}  fresh={row['final_iptm']:+.4f}  delta={row['final_iptm']-row['iptm']:+.4f}")
PY
