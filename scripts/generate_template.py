"""Build a combined target+binder PDB by NBB2-folding an antibody seed and
concatenating with a target PDB. The output is a single multi-chain PDB that
the runtime AF2 reward backend (ab_af2_reward.AbAF2RewardCal) consumes via
``--template_pdb``.

This is the ONLY place NBB2 is used. The runtime pipeline (ab_refinement.py /
SageMaker / eval_iptm.py) reads pre-made PDBs from disk and never imports
NBB2.

Example
-------

    python scripts/generate_template.py \\
        --antibody_sequence "EVQLVESGGGLVQPGGSLRLSCAASGGFTFSSYAMW..." \\
        --antigen_pdb datasets/pdl1.pdb \\
        --antigen_chain A \\
        --output_pdb datasets/template_pdl1.pdb

Run once per (seed, target). Check the resulting PDB into the repo or copy it
into the docker image so runs can reference it from any host.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path


def _combine_target_and_binder_pdb(
    target_pdb_path: str,
    binder_pdb_str: str,
    target_chain: str,
    binder_chain: str = "H",
) -> str:
    """Concatenate target ATOMs and binder ATOMs into a single multi-chain PDB string.

    Vendored from the previous in-repo helper (formerly in ab_af2_reward.py).
    The binder chain is rewritten to ``binder_chain`` (default 'H'); the target
    keeps its existing chain ID.
    """
    with open(target_pdb_path, "r") as f:
        target_pdb = f.read()

    lines = ["HEADER    PROTEIN", "TITLE     COMBINED TARGET+BINDER (NBB2 TEMPLATE)"]
    atom_count = 1

    for line in target_pdb.splitlines():
        if line.startswith("ATOM"):
            lines.append(f"ATOM  {atom_count:5d}{line[11:]}")
            atom_count += 1
    lines.append(f"TER   {atom_count:5d}      {target_chain}")

    binder_atoms = 0
    for line in binder_pdb_str.splitlines():
        if line.startswith("ATOM"):
            new_line = f"ATOM  {atom_count:5d}{line[11:21]}{binder_chain}{line[22:]}"
            lines.append(new_line)
            atom_count += 1
            binder_atoms += 1
    if binder_atoms > 0:
        lines.append(f"TER   {atom_count:5d}      {binder_chain}")
    lines.append("END")
    return "\n".join(lines)


def fold_with_nbb2(antibody_sequence: str, weights_dir: str) -> str:
    """NBB2-fold a heavy-chain antibody sequence and return the PDB as a string."""
    try:
        from ImmuneBuilder import NanoBodyBuilder2
    except ImportError as e:
        raise ImportError(
            "ImmuneBuilder is required for generate_template.py. Install with:\n"
            "  pip install ImmuneBuilder\n"
            "ImmuneBuilder.refine also needs pdbfixer + openmm (conda-forge)."
        ) from e

    import torch  # noqa: F401  (pulled in transitively; keep explicit)

    os.makedirs(weights_dir, exist_ok=True)
    print(f"[NBB2] Loading model (weights_dir={weights_dir}) ...")
    model = NanoBodyBuilder2(numbering_scheme="raw", weights_dir=weights_dir)

    print(f"[NBB2] Folding sequence (len={len(antibody_sequence)}) ...")
    with torch.no_grad():
        nb = model.predict({"H": antibody_sequence})

    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        nb.save(tmp_path)
        with open(tmp_path, "r") as f:
            return f.read()
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="NBB2-fold antibody seed and combine with target into a PDB template.",
    )
    p.add_argument("--antibody_sequence", required=True,
                   help="Full heavy-chain antibody amino acid sequence.")
    p.add_argument("--antigen_pdb", required=True,
                   help="Path to target antigen PDB file (single chain).")
    p.add_argument("--antigen_chain", default="A",
                   help="Chain ID of the target in the input antigen PDB.")
    p.add_argument("--binder_chain", default="H",
                   help="Chain ID to assign to the binder in the output combined PDB.")
    p.add_argument("--output_pdb", required=True,
                   help="Output path for the combined target+binder PDB.")
    p.add_argument("--nbb2_weights_dir", default=None,
                   help="NBB2 weights dir. If None, falls back to $NBB2_WEIGHTS_DIR "
                        "or ~/.mber/nbb2_weights.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.antigen_pdb):
        sys.exit(f"--antigen_pdb not found: {args.antigen_pdb}")

    weights_dir = os.path.expanduser(
        args.nbb2_weights_dir
        or os.environ.get("NBB2_WEIGHTS_DIR", "~/.mber/nbb2_weights")
    )

    binder_pdb_str = fold_with_nbb2(args.antibody_sequence, weights_dir)

    print(f"[combine] target={args.antigen_pdb} (chain {args.antigen_chain}) "
          f"+ NBB2 binder (chain {args.binder_chain})")
    combined = _combine_target_and_binder_pdb(
        target_pdb_path=args.antigen_pdb,
        binder_pdb_str=binder_pdb_str,
        target_chain=args.antigen_chain,
        binder_chain=args.binder_chain,
    )

    out_path = Path(args.output_pdb)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(combined)
    print(f"[done] wrote {out_path} ({len(combined.splitlines())} lines)")


if __name__ == "__main__":
    main()
