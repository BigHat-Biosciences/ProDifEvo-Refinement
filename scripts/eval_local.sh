#!/bin/bash
# Pull all completed SageMaker outputs into ~/Downloads, then re-score each
# target's sequences locally through this repo's AF2 reward backend (writing
# final_iptm / final_cdr_plddt / final_plddt columns into a *_w_final_iptm.csv
# next to each downloaded file).
#
# Run from the EC2 utility machine after the SageMaker jobs have completed:
#
#     cd ~/ProDifEvo-Refinement
#     git pull
#     bash scripts/eval_local.sh
#
# Override the conda env or target list via env vars:
#
#     CONDA_ENV=RERD TARGETS="pdl1 il3" bash scripts/eval_local.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

CONDA_ENV="${CONDA_ENV:-RERD}"
TARGETS="${TARGETS:-pdl1 bhrf1 il3 il20}"
AF_GPU_IDS="${AF_GPU_IDS:-1,2,3}"
DOWNLOADS_DIR="${HOME}/Downloads"

# Map each target to its baked antigen PDB (must exist in datasets/).
declare -A ANTIGEN_PDB=(
    [pdl1]="${REPO_DIR}/datasets/pdl1.pdb"
    [bhrf1]="${REPO_DIR}/datasets/bhrf1.pdb"
    [il3]="${REPO_DIR}/datasets/il3.pdb"
    [il20]="${REPO_DIR}/datasets/il20.pdb"
)

# ----- Step 1: pull SageMaker output.csv files into ~/Downloads -----
echo "=== fetching SageMaker outputs ==="
bash "${REPO_DIR}/scripts/fetch_outputs.sh"
echo

# ----- Step 2: activate conda env -----
# `conda activate` needs the conda shell hook; source whichever conda.sh exists.
for CONDA_SH in /opt/conda/etc/profile.d/conda.sh "${HOME}/miniconda3/etc/profile.d/conda.sh" "${HOME}/anaconda3/etc/profile.d/conda.sh"; do
    if [ -f "$CONDA_SH" ]; then
        # shellcheck disable=SC1090
        source "$CONDA_SH"
        break
    fi
done
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found on PATH and no conda.sh discovered. Set up the RERD env first."
    exit 1
fi
conda activate "$CONDA_ENV"
echo "Activated conda env: $CONDA_ENV ($(python -c 'import sys; print(sys.executable)'))"
echo

# ----- Step 3: re-score each target's CSV -----
for TARGET in $TARGETS; do
    INPUT_CSV="${DOWNLOADS_DIR}/rerd_${TARGET}.csv"
    PDB="${ANTIGEN_PDB[$TARGET]:-}"

    if [ -z "$PDB" ]; then
        echo "[$TARGET] no antigen PDB mapping; skipping"
        continue
    fi
    if [ ! -f "$INPUT_CSV" ]; then
        echo "[$TARGET] $INPUT_CSV not found (run not complete?); skipping"
        continue
    fi
    if [ ! -f "$PDB" ]; then
        echo "[$TARGET] $PDB not found; skipping"
        continue
    fi

    echo "=== re-scoring $TARGET ==="
    echo "  input:   $INPUT_CSV"
    echo "  antigen: $PDB"
    python "${REPO_DIR}/scripts/eval_iptm.py" \
        --input_csv "$INPUT_CSV" \
        --antigen_pdb "$PDB" \
        --antigen_chain A \
        --af_gpu_ids "$AF_GPU_IDS" \
        --write_inplace 0
    echo
done

echo "=== all done ==="
echo "Augmented CSVs (with final_iptm, final_cdr_plddt, final_plddt):"
ls -la "${DOWNLOADS_DIR}"/*_w_final_iptm.csv 2>/dev/null || echo "  (none — check the per-target log above)"
