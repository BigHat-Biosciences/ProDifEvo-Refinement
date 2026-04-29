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

    # Binder design: antigen structure for complex prediction + ipTM reward
    argparser.add_argument(
        "--antigen_pdb", type=str, default=None,
        help="Path to antigen PDB/CIF structure file. Required for 'iptm' metric. "
             "Used directly as the AF2 binder-protocol target template.",
    )
    argparser.add_argument(
        "--antigen_chain", type=str, default=None,
        help="Antigen chain ID(s) to use from the antigen PDB. "
             "If None, all chains are used. For multiple chains, comma-separate: 'A,B'.",
    )

    # ---- AlphaFold2 backend settings ----
    argparser.add_argument(
        "--af_params_dir", type=str, default=None,
        help="Path to the directory containing AlphaFold2 params (e.g. params_model_*_multimer_v3.npz). "
             "If None, falls back to $AF_PARAMS_DIR or ~/.mber/af_params. "
             "Use mber-open/download_weights.sh to fetch.",
    )
    argparser.add_argument(
        "--num_recycles", type=int, default=3,
        help="Number of AF2 recycling iterations per prediction.",
    )
    argparser.add_argument(
        "--af_models", type=str, default="0",
        help="Comma-separated AF2 model indices to ensemble (e.g. '0' or '0,1,2,3,4').",
    )
    argparser.add_argument(
        "--af_use_multimer", action="store_true", default=True,
        help="Use AF2 multimer params (v3 = 'AF2.3M'). Required for ipTM. Default: True.",
    )
    argparser.add_argument(
        "--af_no_multimer", dest="af_use_multimer", action="store_false",
        help="Disable AF2 multimer mode (use monomer params).",
    )

    args = argparser.parse_args()
    return args
