#!/bin/bash
# Score all baseline design outputs under bonobo's FINAL ipTM evaluator (the
# reported metric) and print a per-(target, method) summary — the paper table.
#
# Bonobo's eval_compiled_final_iptm.py reads {COMPILED_DIR}/{method}_{target}.csv.
# Stage each design run's output.csv there under that exact name first, e.g.:
#
#   mkdir -p /tmp/final_table
#   cp .../rerd_pdl1/output.csv  /tmp/final_table/rerd_pdl1.csv
#   cp .../vidd_pdl1/output.csv  /tmp/final_table/vidd_pdl1.csv
#   ... (rerd_/vidd_ × pdl1/bhrf1/il3/il20)
#
# Each CSV needs a `sequence` column; its own `iptm` (reward) column is carried
# through so the summary can show reward-vs-final side by side. Partial runs are
# fine — whatever rows are present get scored.
#
# Then:
#   COMPILED_DIR=/tmp/final_table bash scripts/eval_all_final_iptm.sh
#
# Overridable env vars: BONOBO_REPO, BONOBO_CONDA_ENV, COMPILED_DIR, CACHE_DIR,
# TARGETS (csv), METHODS (csv).
set -euo pipefail

BONOBO_REPO="${BONOBO_REPO:-${HOME}/bonobo}"
BONOBO_CONDA_ENV="${BONOBO_CONDA_ENV:-bonobo}"
COMPILED_DIR="${COMPILED_DIR:-/tmp/final_table}"
CACHE_DIR="${CACHE_DIR:-${COMPILED_DIR}/final_iptm_cache}"
TARGETS="${TARGETS:-pdl1,bhrf1,il20,il3}"
METHODS="${METHODS:-rerd,vidd}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

if [ ! -d "$BONOBO_REPO" ]; then
    echo "ERROR: bonobo repo not found at $BONOBO_REPO (set BONOBO_REPO=...)"
    exit 1
fi
mkdir -p "$CACHE_DIR"

echo "Staged CSVs in $COMPILED_DIR:"
if ! ls "$COMPILED_DIR"/*.csv >/dev/null 2>&1; then
    echo "  none found — stage {method}_{target}.csv into $COMPILED_DIR first."
    exit 1
fi
ls -la "$COMPILED_DIR"/*.csv

# Source conda once.
for CONDA_SH in /opt/conda/etc/profile.d/conda.sh "${HOME}/miniconda3/etc/profile.d/conda.sh" "${HOME}/anaconda3/etc/profile.d/conda.sh"; do
    if [ -f "$CONDA_SH" ]; then
        # shellcheck disable=SC1090
        source "$CONDA_SH"
        break
    fi
done
conda activate "$BONOBO_CONDA_ENV"

# One bonobo pass scores all (target, method) CSVs. AF binder prep happens once
# per target and is reused across methods, so rerd+vidd for a target share it.
echo
echo "Running bonobo eval_compiled_final_iptm.py (targets=$TARGETS methods=$METHODS) ..."
pushd "$BONOBO_REPO" >/dev/null
python eval_compiled_final_iptm.py \
    --targets "$TARGETS" \
    --methods "$METHODS" \
    --compiled_dir_template "$COMPILED_DIR" \
    --cache_dir "$CACHE_DIR" \
    --write_inplace 0
popd >/dev/null

# Summarize into the paper table.
COMPILED_DIR="$COMPILED_DIR" TARGETS="$TARGETS" METHODS="$METHODS" python - <<'PY'
import os
import pandas as pd

cd = os.environ["COMPILED_DIR"]
targets = [t for t in os.environ["TARGETS"].split(",") if t]
methods = [m for m in os.environ["METHODS"].split(",") if m]

rows = []
for t in targets:
    for m in methods:
        p = os.path.join(cd, f"{m}_{t}_w_final_iptm.csv")
        if not os.path.exists(p):
            print(f"[skip] {p} not found")
            continue
        df = pd.read_csv(p)
        fi = df["final_iptm"].dropna()
        row = dict(
            target=t, method=m, n=len(fi),
            final_mean=fi.mean(), final_median=fi.median(), final_max=fi.max(),
            n_ge_0p5=int((fi >= 0.5).sum()), n_ge_0p7=int((fi >= 0.7).sum()),
        )
        if "iptm" in df.columns:
            rw = df["iptm"].dropna()
            row["reward_mean"] = rw.mean()
            # reward-vs-final gap (the parity axis) on the same rows.
            j = df.dropna(subset=["iptm", "final_iptm"])
            row["reward_minus_final"] = (j["iptm"] - j["final_iptm"]).mean() if len(j) else float("nan")
        rows.append(row)

if not rows:
    raise SystemExit("No *_w_final_iptm.csv produced — check the bonobo eval step above.")

res = pd.DataFrame(rows)
print("\n===== FINAL ipTM (bonobo eval) — paper table =====")
print(res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
out = os.path.join(cd, "final_table.csv")
res.to_csv(out, index=False)
print(f"\nwritten: {out}")
PY
