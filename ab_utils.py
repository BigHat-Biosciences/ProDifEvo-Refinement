"""Antibody CDR/framework parsing utilities.

Supports IMGT, Chothia, and Kabat numbering schemes.
Can use ANARCI for automatic numbering or accept manual CDR indices.
"""

import numpy as np

# CDR boundaries by numbering scheme (IMGT numbering on the IMGT-numbered sequence).
# These are the IMGT-defined positions; after ANARCI numbering, CDR residues
# fall at these IMGT position numbers.
IMGT_CDR_BOUNDARIES = {
    "heavy": {
        "H1": (27, 38),
        "H2": (56, 65),
        "H3": (105, 117),
    },
    "light": {
        "L1": (27, 38),
        "L2": (56, 65),
        "L3": (105, 117),
    },
}

CHOTHIA_CDR_BOUNDARIES = {
    "heavy": {
        "H1": (26, 32),
        "H2": (52, 56),
        "H3": (95, 102),
    },
    "light": {
        "L1": (24, 34),
        "L2": (50, 56),
        "L3": (89, 97),
    },
}

KABAT_CDR_BOUNDARIES = {
    "heavy": {
        "H1": (31, 35),
        "H2": (50, 65),
        "H3": (95, 102),
    },
    "light": {
        "L1": (24, 34),
        "L2": (50, 56),
        "L3": (89, 97),
    },
}

SCHEME_MAP = {
    "imgt": IMGT_CDR_BOUNDARIES,
    "chothia": CHOTHIA_CDR_BOUNDARIES,
    "kabat": KABAT_CDR_BOUNDARIES,
}


def parse_manual_cdr_indices(cdr_indices_str):
    """Parse a string of CDR indices like '26-38,55-65,104-117' into a sorted list of ints.

    Args:
        cdr_indices_str: Comma-separated ranges, e.g. '26-38,55-65,104-117'

    Returns:
        Sorted list of integer indices (0-based).
    """
    indices = []
    for span in cdr_indices_str.split(","):
        span = span.strip()
        if "-" in span:
            start, end = span.split("-")
            indices.extend(range(int(start), int(end) + 1))
        else:
            indices.append(int(span))
    return sorted(set(indices))


def parse_cdr_indices_anarci(sequence, scheme="imgt", chain_type="heavy", cdrs_to_design="all"):
    """Use ANARCI to number an antibody sequence and extract CDR indices.

    Args:
        sequence: Raw amino acid sequence string.
        scheme: Numbering scheme ('imgt', 'chothia', 'kabat').
        chain_type: 'heavy' or 'light'.
        cdrs_to_design: Comma-separated CDR names (e.g. 'H1,H2,H3') or 'all'.

    Returns:
        cdr_indices: Sorted list of 0-based indices into the raw sequence that are CDR positions.
        framework_indices: Sorted list of 0-based indices that are framework positions.
    """
    try:
        from anarci import anarci as run_anarci
    except ImportError:
        raise ImportError(
            "ANARCI is required for automatic antibody numbering. "
            "Install with: pip install anarci\n"
            "Alternatively, use --cdr_indices to manually specify CDR positions."
        )

    # Run ANARCI
    results = run_anarci(
        [("query", sequence)],
        scheme=scheme,
        output=False,
    )
    numbering, alignment_details, hit_tables = results

    if numbering[0] is None:
        raise ValueError(
            f"ANARCI could not number the input sequence. "
            f"Ensure this is a valid antibody {chain_type} chain sequence."
        )

    # numbering[0] is a list of domain hits; take the first domain
    numbered_seq = numbering[0][0][0]  # list of ((position_number, insertion_code), amino_acid)

    # Determine which CDRs to design
    boundaries = SCHEME_MAP[scheme][chain_type]
    if cdrs_to_design == "all":
        selected_cdrs = list(boundaries.keys())
    else:
        selected_cdrs = [c.strip() for c in cdrs_to_design.split(",")]

    # Build mapping from IMGT/scheme position to raw sequence index
    cdr_indices = []
    raw_idx = 0
    for (pos_num, insertion_code), aa in numbered_seq:
        if aa == "-":
            continue  # gap in numbering, not present in sequence
        # Check if this position falls in any selected CDR
        for cdr_name in selected_cdrs:
            cdr_start, cdr_end = boundaries[cdr_name]
            if cdr_start <= pos_num <= cdr_end:
                cdr_indices.append(raw_idx)
                break
        raw_idx += 1

    cdr_indices = sorted(set(cdr_indices))
    seq_len = len(sequence)
    framework_indices = sorted(set(range(seq_len)) - set(cdr_indices))

    return cdr_indices, framework_indices


def get_cdr_and_framework_indices(
    sequence,
    scheme="imgt",
    chain_type="heavy",
    cdrs_to_design="all",
    manual_cdr_indices=None,
):
    """Get CDR and framework indices, either from ANARCI or manual specification.

    Args:
        sequence: Raw amino acid sequence string.
        scheme: Numbering scheme.
        chain_type: 'heavy' or 'light'.
        cdrs_to_design: Which CDRs to design.
        manual_cdr_indices: If provided, a string like '26-38,55-65,104-117' (0-based).

    Returns:
        cdr_indices: Sorted list of 0-based CDR position indices.
        framework_indices: Sorted list of 0-based framework position indices.
    """
    seq_len = len(sequence)

    if manual_cdr_indices is not None:
        cdr_indices = parse_manual_cdr_indices(manual_cdr_indices)
        # Validate indices are within sequence bounds
        if max(cdr_indices) >= seq_len or min(cdr_indices) < 0:
            raise ValueError(
                f"CDR indices out of range. Sequence length is {seq_len}, "
                f"but got indices up to {max(cdr_indices)}."
            )
        framework_indices = sorted(set(range(seq_len)) - set(cdr_indices))
    else:
        cdr_indices, framework_indices = parse_cdr_indices_anarci(
            sequence, scheme=scheme, chain_type=chain_type, cdrs_to_design=cdrs_to_design
        )

    if len(cdr_indices) == 0:
        raise ValueError("No CDR indices found. Check your sequence and numbering parameters.")

    print(f"Sequence length: {seq_len}")
    print(f"CDR positions ({len(cdr_indices)}): {cdr_indices}")
    print(f"Framework positions ({len(framework_indices)}): {framework_indices[:5]}...{framework_indices[-5:]}")

    return cdr_indices, framework_indices
