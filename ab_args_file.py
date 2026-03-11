import argparse


def get_ab_args():
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Reward-guided diffusion for antibody CDR design conditional on framework regions.",
    )

    # ---- Antibody-specific arguments ----
    argparser.add_argument(
        "--antibody_sequence", type=str, required=True,
        help="Full antibody heavy or light chain amino acid sequence.",
    )
    argparser.add_argument(
        "--chain_type", type=str, default="heavy", choices=["heavy", "light"],
        help="Antibody chain type.",
    )
    argparser.add_argument(
        "--cdrs_to_design", type=str, default="all",
        help="Which CDRs to design. Comma-separated, e.g. 'H1,H2,H3' or 'H3' or 'all'.",
    )
    argparser.add_argument(
        "--numbering_scheme", type=str, default="imgt", choices=["imgt", "chothia", "kabat"],
        help="Antibody numbering scheme for CDR boundary detection.",
    )
    argparser.add_argument(
        "--cdr_indices", type=str, default=None,
        help="Manual CDR indices (0-based), e.g. '26-38,55-65,104-117'. "
             "If provided, --auto_number is ignored.",
    )
    argparser.add_argument(
        "--auto_number", type=str, default="True",
        help="Use ANARCI for automatic CDR detection. Set to 'False' and provide --cdr_indices for manual mode.",
    )
    argparser.add_argument(
        "--edit_fraction", type=float, default=0.3,
        help="Fraction of CDR positions to re-mask each refinement iteration.",
    )

    # ---- Shared arguments (same as refinement.py) ----
    argparser.add_argument("--run_name", type=str, default="ab_debug", help="Run name for saving results.")
    argparser.add_argument("--seed", type=int, default=0)
    argparser.add_argument("--repeatnum", type=int, default=10, help="Batch size (number of parallel designs).")
    argparser.add_argument("--duplicate", type=int, default=20, help="Number of candidates per step.")
    argparser.add_argument("--decoding", type=str, default="SVDD_edit",
                           choices=["SVDD", "SVDD_edit", "random"],
                           help="Decoding strategy.")

    # Pre-trained diffusion model
    argparser.add_argument("--diffusion_model", type=str, default="38M", choices=["38M", "640M"])
    # ESM model (for structure-based reward)
    argparser.add_argument("--esm_model", type=str, default="650m")

    # Reward settings
    argparser.add_argument(
        "--metrics_name", type=str, required=True,
        help="Comma-separated reward metrics. Supported: "
             "ptm,plddt,cdr_plddt,charge_balance,cdr_hydrophobicity "
             "and all metrics from refinement.py (tm,crmsd,etc.)",
    )
    argparser.add_argument(
        "--metrics_list", type=str, required=True,
        help="Comma-separated weights for each metric.",
    )
    argparser.add_argument("--iteration", type=int, default=50, help="Number of refinement iterations.")

    # Optional: reference PDB for structure comparison metrics
    argparser.add_argument(
        "--ref_pdb", type=str, default=None,
        help="Reference PDB file for structural comparison metrics (tm, crmsd, etc.).",
    )

    args = argparser.parse_args()
    return args
