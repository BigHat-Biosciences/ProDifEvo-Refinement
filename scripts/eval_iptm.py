"""Re-score sequences in a CSV through this repo's AF2 reward backend.

Mirrors bonobo's eval_compiled_final_iptm.py in spirit but uses *this* repo's
reward stack (NBB2 binder pre-fold + IMGT CDR mask + no hotspot + multi-GPU
dispatch), so the scores it produces match what `ab_refinement.py` itself
writes to ``output.csv``. This is the correct comparison for verifying
RERD-internal ipTM values.

If you want bonobo-comparable scores, use bonobo's script instead — it uses
a different binder template, a different rm_binder mask scope, and an
explicit hotspot per target. Same AF2 model, different conditioning.

Reads:  any CSV with a ``sequence`` column (configurable via --sequence_col).
Writes: same CSV with new columns: final_iptm, final_cdr_plddt, final_plddt
        (or a {stem}_w_final_iptm.csv next to it if --write_inplace 0).

Per-sequence cache at {cache_dir}/{stem}_iptm.csv lets you resume after a
crash; sequences already in the cache are skipped on rerun.

Example
-------

    python scripts/eval_iptm.py \\
        --input_csv ~/Downloads/rerd_pdl1.csv \\
        --antigen_pdb datasets/pdl1.pdb \\
        --antigen_chain A \\
        --af_gpu_ids 1,2,3
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# Ensure repo root on sys.path so `from ab_af2_reward import ...` works when
# this script is invoked directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from ab_af2_reward import AbAF2RewardCal  # noqa: E402
from ab_utils import get_cdr_and_framework_indices  # noqa: E402

# Must match ab_af2_reward.ALPHABET — we tokenize directly here to avoid
# pulling in the EvoDiff diffusion model just for its tokenizer.
ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


def tokenize(seq: str) -> List[int]:
    return [ALPHABET.index(c) for c in seq]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__.split("\n\n", 1)[0],
    )
    p.add_argument("--input_csv", required=True, help="CSV with a sequence column.")
    p.add_argument("--sequence_col", default="sequence")
    p.add_argument("--antigen_pdb", required=True)
    p.add_argument("--antigen_chain", default="A")

    p.add_argument(
        "--seed_sequence", default=None,
        help="Antibody seed used for the one-shot NBB2 fold. If None, uses the first "
             "sequence in --input_csv (which assumes all rows share the same framework).",
    )
    p.add_argument("--cdrs_to_design", default="H1,H2,H3",
                   help="Which CDRs to mask from the binder template (rm_binder).")
    p.add_argument("--chain_type", default="heavy", choices=["heavy", "light"])
    p.add_argument("--numbering_scheme", default="imgt", choices=["imgt", "chothia", "kabat"])
    p.add_argument(
        "--cdr_indices", default=None,
        help="Manual CDR indices, e.g. '26-38,55-65,104-117'. Overrides ANARCI detection.",
    )

    # AF2 backend
    p.add_argument("--af_params_dir", default=None,
                   help="If unset, falls back to $AF_PARAMS_DIR or ~/.mber/af_params.")
    p.add_argument("--num_recycles", default=3, type=int)
    p.add_argument("--af_models", default="0", help="Comma-separated AF2 model indices.")
    p.add_argument("--af_gpu_ids", default="",
                   help="Comma-separated JAX device IDs for parallel AF, e.g. '1,2,3'.")
    p.add_argument("--use_template", action="store_true", default=True,
                   help="Use NBB2 binder pre-fold + one-shot template init (default).")
    p.add_argument("--no_template", dest="use_template", action="store_false",
                   help="Disable templating; full hallucination of binder backbone.")
    p.add_argument("--nbb2_weights_dir", default=None)

    # Output / caching
    p.add_argument("--output_csv", default=None,
                   help="Where to write augmented CSV. If unset, see --write_inplace.")
    p.add_argument("--write_inplace", default=1, type=int,
                   help="1: overwrite --input_csv. 0: write {stem}_w_final_iptm.csv.")
    p.add_argument("--cache_dir", default="iptm_cache")
    p.add_argument("--cache_name", default=None,
                   help="Cache file name. Default: {input_csv stem}_iptm.csv.")
    p.add_argument("--chunk_size", default=12, type=int,
                   help="Sequences per reward_metrics call. Smaller = more frequent "
                        "checkpoints but more Python overhead.")

    return p.parse_args()


def load_cache(path: str) -> Dict[str, Tuple[float, float, float]]:
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    out: Dict[str, Tuple[float, float, float]] = {}
    for _, row in df.iterrows():
        out[row["sequence"]] = (
            float(row["final_iptm"]),
            float(row["final_cdr_plddt"]),
            float(row["final_plddt"]),
        )
    return out


def save_cache(path: str, cache: Dict[str, Tuple[float, float, float]]) -> None:
    rows = [
        {
            "sequence": s,
            "final_iptm": v[0],
            "final_cdr_plddt": v[1],
            "final_plddt": v[2],
        }
        for s, v in cache.items()
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    args = parse_args()

    # ---- Load CSV and validate ----
    df = pd.read_csv(args.input_csv)
    if args.sequence_col not in df.columns:
        sys.exit(
            f"--input_csv missing column {args.sequence_col!r}; got: {list(df.columns)}"
        )
    sequences: List[str] = df[args.sequence_col].astype(str).tolist()
    if not sequences:
        sys.exit("No sequences in input CSV.")

    seed_seq: str = args.seed_sequence or sequences[0]
    seq_len = len(seed_seq)
    bad_lens = [(i, len(s)) for i, s in enumerate(sequences) if len(s) != seq_len]
    if bad_lens:
        sys.exit(
            f"All sequences must have the same length as the seed ({seq_len}). "
            f"Found mismatches at rows {bad_lens[:5]}{'...' if len(bad_lens) > 5 else ''}"
        )

    # ---- CDR detection (matches the design loop's logic) ----
    if args.cdr_indices is not None:
        manual_indices: Optional[str] = args.cdr_indices
    else:
        manual_indices = None
    cdr_indices, framework_indices = get_cdr_and_framework_indices(
        sequence=seed_seq,
        scheme=args.numbering_scheme,
        chain_type=args.chain_type,
        cdrs_to_design=args.cdrs_to_design,
        manual_cdr_indices=manual_indices,
    )
    print(f"Seed length: {seq_len}")
    print(f"CDR positions ({len(cdr_indices)}): {cdr_indices}")
    print(f"Framework positions ({len(framework_indices)})")

    # ---- Build reward model (one-shot template init happens lazily on first call) ----
    af_gpu_ids = [int(x) for x in args.af_gpu_ids.split(",") if x.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reward_model = AbAF2RewardCal(
        metrics_name="iptm,cdr_plddt,plddt",
        metrics_list="1,1,1",  # weights don't matter for individual-metric extraction
        run_name="eval_iptm",
        pdb_save_path="/tmp/eval_iptm_pdbs",  # never written since save_pdb=False
        device=device,
        cdr_indices=cdr_indices,
        antigen_pdb=args.antigen_pdb,
        antigen_chain=args.antigen_chain,
        af_params_dir=args.af_params_dir,
        num_recycles=args.num_recycles,
        af_models=tuple(int(x) for x in args.af_models.split(",")),
        use_multimer=True,
        use_template=args.use_template,
        nbb2_weights_dir=args.nbb2_weights_dir,
        af_gpu_ids=af_gpu_ids,
        seed_sequence=seed_seq,
    )

    # ---- Cache setup ----
    os.makedirs(args.cache_dir, exist_ok=True)
    cache_name = args.cache_name or f"{Path(args.input_csv).stem}_iptm.csv"
    cache_path = os.path.join(args.cache_dir, cache_name)
    cache = load_cache(cache_path)
    print(f"Loaded {len(cache)} cached scores from {cache_path}")

    # Score in original CSV order; skip already-cached.
    todo: List[Tuple[int, str]] = [(i, s) for i, s in enumerate(sequences) if s not in cache]
    print(f"Sequences to score: {len(todo)} (out of {len(sequences)})")

    # ---- Score in chunks ----
    for chunk_start in range(0, len(todo), args.chunk_size):
        chunk = todo[chunk_start : chunk_start + args.chunk_size]
        chunk_seqs = [s for _, s in chunk]
        tokens = torch.tensor(
            [tokenize(s) for s in chunk_seqs], dtype=torch.long, device=device
        )
        mask = torch.ones((len(chunk_seqs), seq_len), device=device)

        per_metric, _, _ = reward_model.reward_metrics(
            protein_name="eval",
            mask_for_loss=mask,
            S_sp=tokens,
            save_pdb=False,
        )
        # per_metric: list[list[float]] of shape (chunk_size, 3) -- order matches metrics_name
        for (_, seq), m in zip(chunk, per_metric):
            cache[seq] = (float(m[0]), float(m[1]), float(m[2]))
        save_cache(cache_path, cache)
        done = chunk_start + len(chunk)
        print(f"  chunk {done}/{len(todo)} done, cache -> {cache_path}")

    # ---- Augment input CSV ----
    df["final_iptm"] = [cache.get(s, (np.nan, np.nan, np.nan))[0] for s in sequences]
    df["final_cdr_plddt"] = [cache.get(s, (np.nan, np.nan, np.nan))[1] for s in sequences]
    df["final_plddt"] = [cache.get(s, (np.nan, np.nan, np.nan))[2] for s in sequences]

    if args.output_csv:
        out_path = args.output_csv
    elif args.write_inplace:
        out_path = args.input_csv
    else:
        stem = Path(args.input_csv).with_suffix("")
        out_path = f"{stem}_w_final_iptm.csv"

    df.to_csv(out_path, index=False)
    n_set = df["final_iptm"].notna().sum()
    print(f"Wrote {out_path} ({n_set}/{len(df)} have final_iptm)")

    if "iptm" in df.columns and n_set > 0:
        delta = (df["final_iptm"] - df["iptm"]).abs()
        print(
            f"Original vs final ipTM: mean |Δ|={delta.mean():.4f}, "
            f"max |Δ|={delta.max():.4f} (sanity check — should be ~0 if AF is "
            f"deterministic and template/seed match the design run)"
        )


if __name__ == "__main__":
    main()
