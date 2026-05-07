#!/bin/bash
# End-to-end parity test: run a small RERD design, re-eval with this repo's
# eval_iptm.py, then re-eval with bonobo's eval_compiled_final_iptm.py, and
# print a three-way per-row comparison.
#
# What you should see if RERD is at parity with bonobo:
#   * iptm_design   ≈ iptm_rerd_eval  (RERD self-consistency; confirms no race)
#   * iptm_rerd_eval ≈ iptm_bonobo_eval (parity with bonobo's reward conditioning)
#
# Run on the EC2 box:
#
#     cd ~/ProDifEvo-Refinement
#     git pull
#     bash scripts/eval_e2e_parity.sh
#
# Override defaults via env vars:
#
#     ANTIGEN=il3 REPEAT_NUM=16 ITERATION=5 bash scripts/eval_e2e_parity.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

# ---- Knobs ----
ANTIGEN="${ANTIGEN:-pdl1}"
REPEAT_NUM="${REPEAT_NUM:-8}"
ITERATION="${ITERATION:-3}"
DUPLICATE="${DUPLICATE:-5}"
AF_GPU_IDS="${AF_GPU_IDS:-1,2,3}"
RERD_CONDA_ENV="${RERD_CONDA_ENV:-RERD}"
BONOBO_CONDA_ENV="${BONOBO_CONDA_ENV:-bonobo}"
BONOBO_REPO="${BONOBO_REPO:-${HOME}/repos/bonobo}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/rerd_e2e_parity}"
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
ANTIGEN_PDB="${REPO_DIR}/datasets/${ANTIGEN}.pdb"

if [ ! -f "$TEMPLATE_PDB" ]; then
    echo "ERROR: template not found at $TEMPLATE_PDB"
    exit 1
fi
if [ ! -f "$ANTIGEN_PDB" ]; then
    echo "ERROR: antigen PDB not found at $ANTIGEN_PDB"
    exit 1
fi
if [ ! -d "$BONOBO_REPO" ]; then
    echo "ERROR: bonobo repo not found at $BONOBO_REPO (set BONOBO_REPO=...)"
    exit 1
fi

# Multi-GPU: torch on GPU 0, JAX on GPUs 1/2/3.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mkdir -p "$OUTPUT_ROOT"
RERD_RUN_ROOT="${OUTPUT_ROOT}/rerd_run"
RERD_EVAL_CACHE_DIR="${OUTPUT_ROOT}/rerd_eval_cache"
BONOBO_STAGING_DIR="${OUTPUT_ROOT}/bonobo_eval"
BONOBO_CACHE_DIR="${OUTPUT_ROOT}/bonobo_cache"
mkdir -p "$RERD_RUN_ROOT" "$RERD_EVAL_CACHE_DIR" "$BONOBO_STAGING_DIR" "$BONOBO_CACHE_DIR"

# Helper: source conda once.
for CONDA_SH in /opt/conda/etc/profile.d/conda.sh "${HOME}/miniconda3/etc/profile.d/conda.sh" "${HOME}/anaconda3/etc/profile.d/conda.sh"; do
    if [ -f "$CONDA_SH" ]; then
        # shellcheck disable=SC1090
        source "$CONDA_SH"
        break
    fi
done

# ============================================================
# Step 1: RERD design (multi-GPU, bonobo-style template + hotspot)
# ============================================================
echo "=========================================================="
echo "Step 1/4: RERD design"
echo "  antigen=$ANTIGEN  repeat_num=$REPEAT_NUM  iteration=$ITERATION"
echo "  template=$TEMPLATE_PDB  hotspot=$HOTSPOT"
echo "=========================================================="
conda activate "$RERD_CONDA_ENV"
RUN_NAME="e2e_${ANTIGEN}"
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
    --af_gpu_ids "$AF_GPU_IDS" \
    --output_root "$RERD_RUN_ROOT" \
    --run_name "$RUN_NAME"

# Locate the timestamped run dir.
RUN_DIR=$(find "$RERD_RUN_ROOT" -mindepth 1 -maxdepth 1 -type d -name '*_ab_*' | sort | tail -1)
DESIGN_CSV="$RUN_DIR/output.csv"
if [ ! -f "$DESIGN_CSV" ]; then
    echo "ERROR: design output not found at $DESIGN_CSV"
    exit 1
fi
echo "  design output -> $DESIGN_CSV"

# ============================================================
# Step 2: RERD re-eval (this repo's eval_iptm.py, multi-GPU)
# ============================================================
echo
echo "=========================================================="
echo "Step 2/4: RERD re-eval (eval_iptm.py)"
echo "=========================================================="
python scripts/eval_iptm.py \
    --input_csv "$DESIGN_CSV" \
    --antigen_pdb "$ANTIGEN_PDB" \
    --antigen_chain A \
    --template_pdb "$TEMPLATE_PDB" \
    --hotspot "$HOTSPOT" \
    --af_gpu_ids "$AF_GPU_IDS" \
    --cache_dir "$RERD_EVAL_CACHE_DIR" \
    --write_inplace 0
RERD_EVAL_CSV="${DESIGN_CSV%.csv}_w_final_iptm.csv"
if [ ! -f "$RERD_EVAL_CSV" ]; then
    echo "ERROR: RERD eval output not found at $RERD_EVAL_CSV"
    exit 1
fi
echo "  RERD eval output -> $RERD_EVAL_CSV"

# ============================================================
# Step 3: Stage design output for bonobo's eval, then run it
# ============================================================
# Bonobo's eval_compiled_final_iptm.py reads
# {compiled_dir_template}/{method}_{target}.csv. Stage the design's output.csv
# under the name it expects.
echo
echo "=========================================================="
echo "Step 3/4: Bonobo re-eval (eval_compiled_final_iptm.py)"
echo "=========================================================="
STAGED_INPUT="${BONOBO_STAGING_DIR}/rerd_${ANTIGEN}.csv"
cp "$DESIGN_CSV" "$STAGED_INPUT"
echo "  staged for bonobo -> $STAGED_INPUT"

conda activate "$BONOBO_CONDA_ENV"
pushd "$BONOBO_REPO" >/dev/null
python eval_compiled_final_iptm.py \
    --targets "$ANTIGEN" \
    --methods rerd \
    --compiled_dir_template "$BONOBO_STAGING_DIR" \
    --cache_dir "$BONOBO_CACHE_DIR" \
    --write_inplace 0
popd >/dev/null
BONOBO_EVAL_CSV="${BONOBO_STAGING_DIR}/rerd_${ANTIGEN}_w_final_iptm.csv"
if [ ! -f "$BONOBO_EVAL_CSV" ]; then
    echo "ERROR: bonobo eval output not found at $BONOBO_EVAL_CSV"
    exit 1
fi
echo "  bonobo eval output -> $BONOBO_EVAL_CSV"

# ============================================================
# Step 4: Three-way comparison
# ============================================================
echo
echo "=========================================================="
echo "Step 4/4: Three-way ipTM comparison"
echo "=========================================================="
python - <<PY
import pandas as pd

design = pd.read_csv("$DESIGN_CSV")[["sequence", "iptm"]].rename(columns={"iptm": "iptm_design"})
rerd = pd.read_csv("$RERD_EVAL_CSV")[["sequence", "final_iptm"]].rename(columns={"final_iptm": "iptm_rerd_eval"})
bonobo = pd.read_csv("$BONOBO_EVAL_CSV")[["sequence", "final_iptm"]].rename(columns={"final_iptm": "iptm_bonobo_eval"})

m = design.merge(rerd, on="sequence", how="inner").merge(bonobo, on="sequence", how="inner")
m["dlt_design_vs_rerd"]  = m["iptm_rerd_eval"]   - m["iptm_design"]
m["dlt_rerd_vs_bonobo"]  = m["iptm_bonobo_eval"] - m["iptm_rerd_eval"]
m["dlt_design_vs_bonobo"]= m["iptm_bonobo_eval"] - m["iptm_design"]

def stats(col):
    s = m[col]
    return f"mean={s.mean():+.4f}  std={s.std():.4f}  max={s.max():+.4f}  min={s.min():+.4f}"

print()
print(f"Sequences compared: {len(m)} / {len(design)}")
print()
print("Per-row:")
print(m[["iptm_design","iptm_rerd_eval","iptm_bonobo_eval",
         "dlt_design_vs_rerd","dlt_rerd_vs_bonobo","dlt_design_vs_bonobo"]].to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
print()
print("=== Distribution stats ===")
print(f"  design vs RERD eval:    {stats('dlt_design_vs_rerd')}    (RERD self-consistency; should be ~0)")
print(f"  RERD eval vs bonobo:    {stats('dlt_rerd_vs_bonobo')}    (RERD↔bonobo parity; should be ~0 after fixes)")
print(f"  design vs bonobo:       {stats('dlt_design_vs_bonobo')}  (cumulative)")
print()
m.to_csv("${OUTPUT_ROOT}/comparison.csv", index=False)
print("Wrote merged table to ${OUTPUT_ROOT}/comparison.csv")
PY
