#!/bin/bash
# Parity check for RERD: the ipTM REWARD the model optimized during design vs
# the FINAL ipTM computed by bonobo's evaluator (the only number we report).
#
# Final ipTM is computed by bonobo ONLY, so there is no longer any need to
# cross-check RERD's / VIDD's own evaluators against each other or against
# bonobo — that comparison is extraneous and has been removed. Two tests remain:
#
#   [CRITICAL] reward-vs-bonobo : design-time reward `iptm` (logged in the RERD
#                                 run's output.csv) vs bonobo
#                                 eval_compiled_final_iptm.py `final_iptm` on the
#                                 same sequences. mean ~0 = the reward the model
#                                 optimized matches the metric we report.
#   [CYA]      race check        : design-time reward `iptm` vs a fresh RERD
#                                 re-eval (scripts/eval_iptm.py). mean ~0 = no
#                                 multi-GPU race corrupted the logged reward.
#                                 Skip with RUN_RACE_CHECK=0.
#
# Required:
#   * --input-csv : a RERD design output.csv with a `sequence` column (and an
#                   `iptm` column = the design-time reward; required for the
#                   critical test).
#   * --antigen   : one of pdl1, bhrf1, il3, il20. Auto-fills antigen PDB,
#                   template PDB, and hotspot from datasets/.
#
# Run on the EC2 box:
#
#     cd ~/ProDifEvo-Refinement
#     git pull
#     bash scripts/eval_parity.sh \
#         --input-csv ~/Downloads/rerd_pdl1.csv \
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
# Race check (design reward vs a fresh RERD re-eval) is CYA — on by default,
# set RUN_RACE_CHECK=0 to skip and only run the critical reward-vs-bonobo test.
RUN_RACE_CHECK="${RUN_RACE_CHECK:-1}"

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
# Test A [CRITICAL]: bonobo final ipTM (eval_compiled_final_iptm.py, bonobo env)
# ============================================================
# Bonobo's eval reads {compiled_dir}/{method}_{target}.csv. Stage the input CSV
# under that name so bonobo scores the exact design sequences.
echo "=========================================================="
echo "Test A [CRITICAL]: bonobo final ipTM"
echo "  input  : $INPUT_SNAPSHOT"
echo "  antigen: $ANTIGEN"
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
# Test B [CYA]: RERD re-eval (scripts/eval_iptm.py, RERD env) — race check
# ============================================================
RERD_EVAL_CSV=""
if [ "$RUN_RACE_CHECK" = "1" ]; then
    echo
    echo "=========================================================="
    echo "Test B [CYA]: RERD re-eval (eval_iptm.py) for race check"
    echo "  template: $TEMPLATE_PDB  hotspot: $HOTSPOT"
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
        echo "ERROR: RERD re-eval output not found at $RERD_EVAL_CSV"
        exit 1
    fi
    echo "  -> $RERD_EVAL_CSV"
else
    echo
    echo "(skipping Test B race check; RUN_RACE_CHECK=0)"
fi

# ============================================================
# Report
# ============================================================
echo
echo "=========================================================="
echo "Parity report: RERD reward vs bonobo final ipTM"
echo "=========================================================="
BONOBO_EVAL_CSV="$BONOBO_EVAL_CSV" RERD_EVAL_CSV="$RERD_EVAL_CSV" \
INPUT_SNAPSHOT="$INPUT_SNAPSHOT" OUTPUT_ROOT="$OUTPUT_ROOT" \
python - <<'PY'
import os
import pandas as pd

orig = pd.read_csv(os.environ["INPUT_SNAPSHOT"])
bonobo = pd.read_csv(os.environ["BONOBO_EVAL_CSV"])[["sequence", "final_iptm"]].rename(
    columns={"final_iptm": "iptm_bonobo_final"}
)

if "iptm" not in orig.columns:
    raise SystemExit(
        "ERROR: input CSV has no `iptm` column — cannot run the reward-vs-bonobo "
        "test. Pass a RERD design output.csv (its `iptm` column is the reward)."
    )

keep = ["sequence", "iptm"]
m = orig[keep].rename(columns={"iptm": "iptm_reward"}).merge(bonobo, on="sequence", how="inner")

# CRITICAL: reward the model optimized vs the metric we report.
m["delta_reward_vs_bonobo"] = m["iptm_reward"] - m["iptm_bonobo_final"]

rerd_eval_csv = os.environ.get("RERD_EVAL_CSV", "")
has_race = bool(rerd_eval_csv) and os.path.exists(rerd_eval_csv)
if has_race:
    rerd = pd.read_csv(rerd_eval_csv)[["sequence", "final_iptm"]].rename(
        columns={"final_iptm": "iptm_rerd_reeval"}
    )
    m = m.merge(rerd, on="sequence", how="inner")
    # CYA: design-logged reward vs a fresh re-eval — non-zero => multi-GPU race.
    m["delta_reward_vs_reeval"] = m["iptm_reward"] - m["iptm_rerd_reeval"]

print()
print(f"Sequences compared: {len(m)} / {len(orig)}")
print()

cols = ["iptm_reward", "iptm_bonobo_final", "delta_reward_vs_bonobo"]
if has_race:
    cols = ["iptm_reward", "iptm_rerd_reeval", "iptm_bonobo_final",
            "delta_reward_vs_reeval", "delta_reward_vs_bonobo"]
print("Per-row:")
print(m[cols].to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

def stats(s):
    return f"mean={s.mean():+.4f}  std={s.std():.4f}  max={s.max():+.4f}  min={s.min():+.4f}"

print()
print("=== Test A [CRITICAL]: RERD reward vs bonobo final ipTM ===")
print("    Does the reward the model optimized match the metric we report? ~0 = parity.")
print(f"    delta_reward_vs_bonobo:  {stats(m['delta_reward_vs_bonobo'])}")
if has_race:
    print()
    print("=== Test B [CYA]: RERD reward vs fresh re-eval (race check) ===")
    print("    Non-zero => multi-GPU race corrupted the logged reward. ~0 = clean.")
    print(f"    delta_reward_vs_reeval:  {stats(m['delta_reward_vs_reeval'])}")

out = os.path.join(os.environ["OUTPUT_ROOT"], "comparison.csv")
m.to_csv(out, index=False)
print()
print(f"Merged comparison written to: {out}")
PY
