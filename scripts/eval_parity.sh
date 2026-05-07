#!/bin/bash
# Score a set of sequences with BOTH RERD's eval_iptm.py AND bonobo's
# eval_compiled_final_iptm.py, then print a side-by-side ipTM comparison.
# No design loop — just feeds a CSV of sequences through both evaluators.
#
# Useful for parity testing without spending hours on a design run.
#
# Required:
#   * --input-csv must have a `sequence` column.
#   * --antigen specifies the target (one of pdl1, bhrf1, il3, il20). The
#     script auto-fills antigen PDB, template PDB, and hotspot from the
#     baked-in datasets/ directory.
#
# Run on the EC2 box:
#
#     cd ~/ProDifEvo-Refinement
#     git pull
#     bash scripts/eval_parity.sh \
#         --input-csv /path/to/sequences.csv \
#         --antigen pdl1
#
# Or via env vars:
#
#     INPUT_CSV=~/Downloads/rerd_pdl1.csv ANTIGEN=pdl1 \
#         bash scripts/eval_parity.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

# ---- Parse CLI args (also accept env vars as fallbacks) ----
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
    echo "usage: bash scripts/eval_parity.sh --input-csv <csv> --antigen <name>"
    echo "       (or set INPUT_CSV and ANTIGEN env vars)"
    exit 1
fi
if [ ! -f "$INPUT_CSV" ]; then
    echo "ERROR: --input-csv not found: $INPUT_CSV"
    exit 1
fi

# ---- Knobs (env-overridable) ----
AF_GPU_IDS="${AF_GPU_IDS:-1,2,3}"
RERD_CONDA_ENV="${RERD_CONDA_ENV:-RERD}"
BONOBO_CONDA_ENV="${BONOBO_CONDA_ENV:-bonobo}"
BONOBO_REPO="${BONOBO_REPO:-${HOME}/bonobo}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/rerd_parity/${ANTIGEN}}"

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
    echo "ERROR: no baked hotspot for antigen=$ANTIGEN. Add it to HOTSPOTS in this script."
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
if [ ! -d "$BONOBO_REPO" ]; then
    echo "ERROR: bonobo repo not found at $BONOBO_REPO (set BONOBO_REPO=...)"
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mkdir -p "$OUTPUT_ROOT"
RERD_EVAL_CACHE_DIR="${OUTPUT_ROOT}/rerd_eval_cache"
BONOBO_STAGING_DIR="${OUTPUT_ROOT}/bonobo_eval"
BONOBO_CACHE_DIR="${OUTPUT_ROOT}/bonobo_cache"
mkdir -p "$RERD_EVAL_CACHE_DIR" "$BONOBO_STAGING_DIR" "$BONOBO_CACHE_DIR"

# Snapshot the source CSV inside our output dir for reproducibility.
INPUT_BASENAME="$(basename "$INPUT_CSV")"
INPUT_SNAPSHOT="${OUTPUT_ROOT}/${INPUT_BASENAME}"
cp "$INPUT_CSV" "$INPUT_SNAPSHOT"

# Source conda once.
for CONDA_SH in /opt/conda/etc/profile.d/conda.sh "${HOME}/miniconda3/etc/profile.d/conda.sh" "${HOME}/anaconda3/etc/profile.d/conda.sh"; do
    if [ -f "$CONDA_SH" ]; then
        # shellcheck disable=SC1090
        source "$CONDA_SH"
        break
    fi
done

# ============================================================
# Step 1: RERD eval (this repo's eval_iptm.py, RERD env)
# ============================================================
echo "=========================================================="
echo "Step 1/3: RERD eval (eval_iptm.py)"
echo "  input  : $INPUT_SNAPSHOT"
echo "  antigen: $ANTIGEN  hotspot: $HOTSPOT"
echo "  template: $TEMPLATE_PDB"
echo "=========================================================="
conda activate "$RERD_CONDA_ENV"
python scripts/eval_iptm.py \
    --input_csv "$INPUT_SNAPSHOT" \
    --antigen_pdb "$ANTIGEN_PDB" \
    --antigen_chain A \
    --template_pdb "$TEMPLATE_PDB" \
    --hotspot "$HOTSPOT" \
    --af_gpu_ids "$AF_GPU_IDS" \
    --cache_dir "$RERD_EVAL_CACHE_DIR" \
    --write_inplace 0
RERD_EVAL_CSV="${INPUT_SNAPSHOT%.csv}_w_final_iptm.csv"
if [ ! -f "$RERD_EVAL_CSV" ]; then
    echo "ERROR: RERD eval output not found at $RERD_EVAL_CSV"
    exit 1
fi
echo "  -> $RERD_EVAL_CSV"

# ============================================================
# Step 2: Bonobo eval (eval_compiled_final_iptm.py, bonobo env)
# ============================================================
# Bonobo's eval reads {compiled_dir}/{method}_{target}.csv. Stage the
# input CSV under that name.
echo
echo "=========================================================="
echo "Step 2/3: Bonobo eval (eval_compiled_final_iptm.py)"
echo "=========================================================="
STAGED_INPUT="${BONOBO_STAGING_DIR}/rerd_${ANTIGEN}.csv"
cp "$INPUT_SNAPSHOT" "$STAGED_INPUT"
echo "  staged for bonobo: $STAGED_INPUT"

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
echo "  -> $BONOBO_EVAL_CSV"

# ============================================================
# Step 3: Side-by-side comparison
# ============================================================
echo
echo "=========================================================="
echo "Step 3/3: Side-by-side ipTM comparison"
echo "=========================================================="
python - <<PY
import pandas as pd

orig = pd.read_csv("$INPUT_SNAPSHOT")
rerd = pd.read_csv("$RERD_EVAL_CSV")[["sequence", "final_iptm"]].rename(
    columns={"final_iptm": "iptm_rerd"}
)
bonobo = pd.read_csv("$BONOBO_EVAL_CSV")[["sequence", "final_iptm"]].rename(
    columns={"final_iptm": "iptm_bonobo"}
)

keep = ["sequence"]
has_orig_iptm = "iptm" in orig.columns
if has_orig_iptm:
    keep.append("iptm")
m = orig[keep].merge(rerd, on="sequence", how="inner").merge(bonobo, on="sequence", how="inner")

# Test 1: RERD self-consistency. Compares the iptm reported by the original
# CSV (typically a design run's output.csv) to a fresh re-eval through
# eval_iptm.py. Mean ~0 confirms multi-GPU dispatch is race-free.
if has_orig_iptm:
    m["delta_input_vs_rerd"] = m["iptm_rerd"] - m["iptm"]

# Test 2: RERD <-> bonobo parity. Compares fresh RERD eval to fresh bonobo
# eval on the same sequences. Mean ~0 confirms the iptm calculation conditioning
# (template, hotspot, rm_binder, prep) is at parity with bonobo.
m["delta_rerd_vs_bonobo"] = m["iptm_rerd"] - m["iptm_bonobo"]

print()
print(f"Sequences scored by both: {len(m)} / {len(orig)}")
print()

cols = []
if has_orig_iptm:
    cols += ["iptm", "iptm_rerd", "delta_input_vs_rerd"]
cols += ["iptm_rerd", "iptm_bonobo", "delta_rerd_vs_bonobo"]
# de-dup adjacent iptm_rerd if both tests are present
if cols.count("iptm_rerd") > 1:
    cols = cols[:cols.index("iptm_rerd") + 1] + [c for c in cols[cols.index("iptm_rerd") + 1:] if c != "iptm_rerd"]
print("Per-row:")
print(m[cols].to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

def stats(s):
    return f"mean={s.mean():+.4f}  std={s.std():.4f}  max={s.max():+.4f}  min={s.min():+.4f}"

print()
if has_orig_iptm:
    print("=== Test 1: RERD self-consistency (input output.csv vs re-eval) ===")
    print("    Tests for multi-GPU race-condition issues. Should be ~0.")
    print(f"    delta_input_vs_rerd:   {stats(m['delta_input_vs_rerd'])}")
    print()
print("=== Test 2: RERD vs bonobo iptm parity ===")
print("    Tests for iptm calculation parity (template/hotspot/rm_binder). Should be ~0.")
print(f"    delta_rerd_vs_bonobo:  {stats(m['delta_rerd_vs_bonobo'])}")

out = "${OUTPUT_ROOT}/comparison.csv"
m.to_csv(out, index=False)
print()
print(f"Merged comparison written to: {out}")
PY
