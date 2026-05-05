import numpy as np
import copy
from io import StringIO

# pyrosetta is optional — only the structural-metric functions in this module
# need it (and they all reach pyrosetta via reward_utils.pose_read_pdb, which
# centralizes the install check + init). Confidence-only runs work without it.
try:
    from pyrosetta import rosetta
except ImportError:
    rosetta = None

from biotite.structure import annotate_sse, AtomArray, rmsd, sasa, superimpose
import biotite.structure.io as strucio
from tmtools import tm_align

from reward_utils import *

_HYDROPHOBICS = {"VAL", "ILE", "LEU", "PHE", "MET", "TRP"}


def esm_to_ptm(folding_result: dict, idx=0):
    """
    folding result = esmfold.infer(sequence)
    """
    return folding_result['ptm'].cpu().tolist()[idx] 


def esm_to_plddt(folding_result: dict, idx=0):
    """
    folding result = esmfold.infer(sequence)
    """
    return folding_result['mean_plddt'].cpu().tolist()[idx] * 1.0/100


def pdb_to_tm(ori_pdb_file, gen_pdb_file):
    """
    maximize tm score
    :param ori_pdb_file / gen_pdb_file: pdb file path, or, esmfold.infer_pdbs(sequence)[0]
    """
    pose_ori_pdb = pose_read_pdb(ori_pdb_file)
    pose_gen_pdb = pose_read_pdb(gen_pdb_file)

    seq_ori = pose_ori_pdb.sequence()
    seq_gen = pose_gen_pdb.sequence()

    ca_coor_ori = []
    for i in range(1, pose_ori_pdb.total_residue() + 1):
        if pose_ori_pdb.residue(i).has("CA"):
            ca_coord = pose_ori_pdb.residue(i).xyz("CA")
            ca_coor_ori.append((ca_coord.x, ca_coord.y, ca_coord.z))
            # seq_ori.append(pose_ori_pdb.sequence()[i - 1])
    ca_coor_ori = np.array(ca_coor_ori)
    # seq_ori = ''.join(seq_ori)

    ca_coor_gen = []
    for i in range(1, pose_gen_pdb.total_residue() + 1):
        if pose_gen_pdb.residue(i).has("CA"):
            ca_coord = pose_gen_pdb.residue(i).xyz("CA")
            ca_coor_gen.append((ca_coord.x, ca_coord.y, ca_coord.z))
            # seq_gen.append(pose_gen_pdb.sequence()[i - 1])
    ca_coor_gen = np.array(ca_coor_gen)
    # seq_gen = ''.join(seq_gen)

    tm_results = tm_align(ca_coor_ori, ca_coor_gen, seq_ori, seq_gen)
    return tm_results.tm_norm_chain1


def pdb_to_crmsd(ori_pdb_file, gen_pdb_file, backbone=True):
    """
    minimize rmsd, if backbone, only consider N,CA,C
    """
    pose_ori_pdb = pose_read_pdb(ori_pdb_file)
    pose_gen_pdb = pose_read_pdb(gen_pdb_file)

    if backbone:
        return -rosetta.core.scoring.bb_rmsd(pose_ori_pdb, pose_gen_pdb)
    else:
        return -rosetta.core.scoring.all_atom_rmsd(pose_ori_pdb, pose_gen_pdb)


def pdb_to_drmsd(ori_pdb_file, gen_pdb_file, backbone=True):
    atom_gen = pdb_file_to_atomarray(gen_pdb_file)
    atom_ori = pdb_file_to_atomarray(ori_pdb_file)

    if backbone:
        atom_gen = get_backbone_atoms(atom_gen)
        atom_ori = get_backbone_atoms(atom_ori)

    dp = pairwise_distances(atom_gen.coord)
    dq = pairwise_distances(atom_ori.coord)

    return float(np.sqrt(((dp - dq) ** 2).mean()))


def pdb_to_lddt(ori_pdb_file, gen_pdb_file):
    """
    maximize lddt score
    """
    pose_ori_pdb = pose_read_pdb(ori_pdb_file)
    pose_gen_pdb = pose_read_pdb(gen_pdb_file)

    lddt = rosetta.core.scoring.lddt(pose_ori_pdb, pose_gen_pdb)
    return lddt


def pdb_to_hydrophobic_score(gen_pdb_file, start_residue_index=None, end_residue_index=None):
    """
    Computes ratio of hydrophobic atoms in a biotite AtomArray that are also surface exposed
    Typically, minimize hydrophobic score
    """
    # atom_array = strucio.load_structure(gen_pdb_file)
    atom_array = pdb_file_to_atomarray(gen_pdb_file)

    hydrophobic_mask = np.array([aa in _HYDROPHOBICS for aa in atom_array.res_name])

    if start_residue_index is None and end_residue_index is None:
        selection_mask = np.ones_like(hydrophobic_mask)
    else:
        start_residue_index = 0 if start_residue_index is None else start_residue_index
        end_residue_index = (
            len(hydrophobic_mask) if end_residue_index is None else end_residue_index
        )
        selection_mask = np.array(
            [
                i >= start_residue_index and i < end_residue_index
                for i in range(len(hydrophobic_mask))
            ]
        )

    hydrophobic_surf = np.logical_and(
        selection_mask * hydrophobic_mask, sasa(atom_array)
    )

    return -sum(hydrophobic_surf) / sum(selection_mask * hydrophobic_mask)


def pdb_to_match_ss_score(ori_pdb_file, gen_pdb_file, start=None, end=None):
    """
    whether protein's secondary structure matched predefined class, compute the ratio
    Specify `'a'` for alpha helix, `'b'` for beta sheet, and `'c'` for coils.
    """
    # atom_array = strucio.load_structure(gen_pdb_file)

    atom_array = pdb_file_to_atomarray(gen_pdb_file)

    start = 0 if start is None else start
    end = len(atom_array) if end is None else end

    subprotein = atom_array[
        np.logical_and(
            atom_array.res_id >= start,
            atom_array.res_id < end,
        )
    ]
    sse1 = annotate_sse(subprotein)

    atom_array = pdb_file_to_atomarray(ori_pdb_file)

    start = 0 if start is None else start
    end = len(atom_array) if end is None else end

    subprotein = atom_array[
        np.logical_and(
            atom_array.res_id >= start,
            atom_array.res_id < end,
        )
    ]
    sse2 = annotate_sse(subprotein)
    if len(sse1) != len(sse2):
        raise Exception("Error")
    return np.mean(sse1 == sse2), (sse1 != sse2)


def pdb_to_match_ss_score_original(ori_pdb_file, gen_pdb_file, start=None, end=None):
    """
    whether protein's secondary structure matched predefined class, compute the ratio
    Specify `'a'` for alpha helix, `'b'` for beta sheet, and `'c'` for coils.
    """
    # atom_array = strucio.load_structure(gen_pdb_file)

    atom_array = pdb_file_to_atomarray(gen_pdb_file)

    start = 0 if start is None else start
    end = len(atom_array) if end is None else end

    subprotein = atom_array[
        np.logical_and(
            atom_array.res_id >= start,
            atom_array.res_id < end,
        )
    ]
    sse1 = annotate_sse(subprotein)
    sse2 = 'a'
    return np.mean(sse1 == sse2), (sse1 != sse2)


def pdb_to_surface_expose_score(gen_pdb_file, start=None, end=None):
    """
    maximize surface exposure
    """
    # atom_array = strucio.load_structure(gen_pdb_file)
    atom_array = pdb_file_to_atomarray(gen_pdb_file)

    start = 0 if start is None else start
    end = len(atom_array) if end is None else end

    residue_mask = np.array([res_id in list(range(start, end)) for res_id in atom_array.res_id])
    surface = np.logical_and(residue_mask, sasa(atom_array))

    return sum(surface) / sum(residue_mask)


def symmetry_score(gen_pdb_file, starts, ends, all_to_all_protomer_symmetry=False):
    """
    starts: start residue index list
    ends: end residue index list
    """
    atom_array = pdb_file_to_atomarray(gen_pdb_file)

    assert len(starts) == len(ends)
    centers_of_mass = []
    for i in range(len(starts)):
        start, end = starts[i], ends[i]
        backbone_coordinates = get_backbone_atoms(
            atom_array[
                np.logical_and(
                    atom_array.res_id >= start,
                    atom_array.res_id < end,
                )
            ]
        ).coord
        centers_of_mass.append(get_center_of_mass(backbone_coordinates))
    centers_of_mass = np.vstack(centers_of_mass)

    return (
        -float(np.std(pairwise_distances(centers_of_mass)))
        if all_to_all_protomer_symmetry
        else -float(np.std(adjacent_distances(centers_of_mass)))
    )


def pdb_to_globularity_score(gen_pdb_file, start=None, end=None):
    """
    maximize globularity score, make it as a ball
    """
    # atom_array = strucio.load_structure(gen_pdb_file)
    atom_array = pdb_file_to_atomarray(gen_pdb_file)

    start = 0 if start is None else start
    end = len(atom_array) if end is None else end

    backbone = get_backbone_atoms(
        atom_array[
            np.logical_and(
                atom_array.res_id >= start,
                atom_array.res_id < end,
            )
        ]
    ).coord

    center_of_mass = get_center_of_mass(backbone)
    m = backbone - center_of_mass
    return -float(np.std(np.linalg.norm(m, axis=-1)))


# ============================================================
# Antibody-specific reward functions
# ============================================================

import torch as _torch


def compute_iptm(tm_logits, ab_len, ag_len, max_bin=31, no_bins=64, eps=1e-8):
    """Compute interface pTM (ipTM) from TM-score logits.

    ipTM measures predicted structural accuracy at the antibody-antigen interface.
    Uses the same TM-score formula as pTM but restricted to inter-chain residue pairs.

    Args:
        tm_logits: [N_res, N_res, N_bins] TM-score head logits from ESMFold.
        ab_len: Length of antibody chain (first chain).
        ag_len: Length of antigen chain (second chain).
        max_bin: Maximum bin distance for TM-score computation.
        no_bins: Number of bins.
        eps: Epsilon for numerical stability.

    Returns:
        ipTM score (float).
    """
    device = tm_logits.device
    n = tm_logits.shape[-2]  # total residues (ab + ag)

    boundaries = _torch.linspace(0, max_bin, steps=(no_bins - 1), device=device)
    step = boundaries[1] - boundaries[0]
    bin_centers = _torch.cat([boundaries + step / 2,
                              (boundaries[-1] + step * 1.5).unsqueeze(0)])

    clipped_n = max(n, 19)
    d0 = 1.24 * (clipped_n - 15) ** (1.0 / 3) - 1.8

    probs = _torch.nn.functional.softmax(tm_logits, dim=-1)  # [N, N, bins]
    tm_per_bin = 1.0 / (1 + (bin_centers ** 2) / (d0 ** 2))
    predicted_tm_term = _torch.sum(probs * tm_per_bin, dim=-1)  # [N, N]

    # Inter-chain mask: 1 where residue i and j are from different chains
    interface_mask = _torch.zeros(n, n, device=device)
    interface_mask[:ab_len, ab_len:ab_len + ag_len] = 1.0
    interface_mask[ab_len:ab_len + ag_len, :ab_len] = 1.0

    # For each alignment reference position i, compute TM over inter-chain partners j
    masked_tm = predicted_tm_term * interface_mask  # [N, N]
    n_interface_per_row = interface_mask.sum(dim=-1)  # [N]

    # Only consider rows that have inter-chain partners
    valid = n_interface_per_row > 0
    per_alignment = _torch.zeros(n, device=device)
    per_alignment[valid] = masked_tm[valid].sum(dim=-1) / n_interface_per_row[valid]

    # Return max over alignment positions (standard TM-score convention)
    return per_alignment.max().item()


def esm_to_iptm(folding_result, ab_len, ag_len, idx=0):
    """Extract ipTM from ESMFold complex prediction output.

    Tries to compute proper ipTM from TM-score logits. If logits are not
    available, falls back to the global pTM of the complex as a proxy.

    Args:
        folding_result: Output dict from esmfold.infer().
        ab_len: Antibody sequence length.
        ag_len: Antigen sequence length.
        idx: Batch index.

    Returns:
        ipTM score (float, higher = better predicted interface).
    """
    # Try proper ipTM from TM-score logits
    if 'tm_logits' in folding_result:
        logits = folding_result['tm_logits'][idx]  # [N, N, bins]
        return compute_iptm(logits, ab_len, ag_len)

    # Fallback: use predicted_aligned_error to approximate
    if 'predicted_aligned_error' in folding_result:
        pae = folding_result['predicted_aligned_error'][idx]  # [N, N]
        n = pae.shape[0]
        # Extract inter-chain PAE block and convert to a score (lower PAE = better)
        interface_pae_ab_ag = pae[:ab_len, ab_len:ab_len + ag_len]
        interface_pae_ag_ab = pae[ab_len:ab_len + ag_len, :ab_len]
        mean_interface_pae = (interface_pae_ab_ag.mean() + interface_pae_ag_ab.mean()) / 2
        max_pae = pae.max()
        # Convert to [0,1] score: 1 = perfect (PAE=0), 0 = worst
        return max(0.0, 1.0 - (mean_interface_pae / max_pae).cpu().item())

    # Last resort: use complex pTM as a proxy
    if 'ptm' in folding_result:
        return folding_result['ptm'].cpu().tolist()[idx]

    raise ValueError("Cannot compute ipTM: no tm_logits, predicted_aligned_error, or ptm in output.")


# Amino acid charge at physiological pH (~7.4)
_AA_CHARGE = {
    'D': -1.0, 'E': -1.0,  # acidic (negative)
    'K': 1.0, 'R': 1.0,    # basic (positive)
    'H': 0.1,              # histidine partially protonated at pH 7.4
}


def esm_to_cdr_plddt(folding_result, cdr_indices, idx=0):
    """Compute mean pLDDT restricted to CDR residue positions.

    Args:
        folding_result: Output of esmfold.infer(sequences).
        cdr_indices: List of 0-based CDR residue positions.
        idx: Batch index.

    Returns:
        Mean pLDDT over CDR positions, scaled to [0, 1].
    """
    # ESMFold plddt tensor shape is [B, L, 1]; squeeze trailing dim
    plddt_all = folding_result['plddt'][idx].squeeze()  # [L]
    if plddt_all.dim() > 1:
        plddt_all = plddt_all.mean(dim=-1)
    cdr_plddt = plddt_all[cdr_indices].mean().cpu().item() / 100.0
    return cdr_plddt


def cdr_charge_score(sequence_str, cdr_indices):
    """Score CDR charge balance. More balanced (lower absolute charge) is better.

    Args:
        sequence_str: Full amino acid sequence string.
        cdr_indices: List of 0-based CDR residue positions.

    Returns:
        Negative absolute net charge per CDR residue (higher = more balanced = better).
    """
    cdr_residues = [sequence_str[i] for i in cdr_indices if i < len(sequence_str)]
    if len(cdr_residues) == 0:
        return 0.0
    net_charge = sum(_AA_CHARGE.get(aa, 0.0) for aa in cdr_residues)
    return -abs(net_charge) / len(cdr_residues)


def cdr_hydrophobicity_score(gen_pdb_file, cdr_indices):
    """Score CDR hydrophobicity (lower exposed hydrophobics = better developability).

    Wraps the existing pdb_to_hydrophobic_score with CDR-specific residue range.

    Args:
        gen_pdb_file: PDB file path or string.
        cdr_indices: List of 0-based CDR residue positions.

    Returns:
        Negative ratio of hydrophobic surface-exposed CDR atoms (higher = less hydrophobic = better).
    """
    if len(cdr_indices) == 0:
        return 0.0
    # Use the min/max of CDR indices as the residue range
    # Note: pdb_to_hydrophobic_score uses 0-based residue indexing via atom_array
    start = min(cdr_indices)
    end = max(cdr_indices) + 1
    return pdb_to_hydrophobic_score(gen_pdb_file, start_residue_index=start, end_residue_index=end)


def pair_diversity(seq1, seq2, mask1, mask2):
    n = len(seq1)
    assert len(seq2) == n, "Sequences must be the same length"
    assert len(mask1) == n and len(mask2) == n, "Masks must be the same length as sequences"
    combined_mask = np.array(mask1) * np.array(mask2)
    effective_positions = np.sum(combined_mask)
    if effective_positions == 0:
        return 0.0

    differences = sum(1 for l in range(n) if combined_mask[l] == 1 and seq1[l] != seq2[l])
    return differences / effective_positions


def set_diversity(sequences, masks):
    m = len(sequences)
    diversity_matrix = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            if i != j:
                diversity_matrix[i, j] = pair_diversity(sequences[i], sequences[j], masks[i], masks[j])
    overall_diversity = np.sum(diversity_matrix) / (m ** 2)
    return overall_diversity.item()


if __name__ == "__main__":
    # ori_pdb_file = "/data/xsu2/BioScale/GenProteins/casp15/ori_pdb/T1104-D1.pdb"
    # gen_pdb_file = "/data/xsu2/BioScale/GenProteins/casp15/esm3_sm_open_v1/mcts_rollout20_depth2_posk1_sampling10_esm2_8m_esm2_8m/gen_rosettafold2/T1104-D1_idx0/models/model_00_pred.pdb"
    from biotite.database.rcsb import fetch
    import esm2

    ALL_RESIDUE_TYPES = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
    RESIDUE_TYPES_WITHOUT_CYSTEINE = copy.deepcopy(ALL_RESIDUE_TYPES)
    RESIDUE_TYPES_WITHOUT_CYSTEINE.remove("C")

    template_pdb_file = fetch("6mrs", format="pdb")
    pdb_value_str = template_pdb_file.getvalue()
    template_atoms: AtomArray = pdb_file_to_atomarray(StringIO(pdb_value_str))
    sequence_length = len(sequence_from_atomarray(template_atoms))

    # random_seq = "".join([np.random.choice(RESIDUE_TYPES_WITHOUT_CYSTEINE) for _ in range(sequence_length)])

    random_seq = [np.random.choice(RESIDUE_TYPES_WITHOUT_CYSTEINE) for _ in range(sequence_length)]
    mask_indices = np.random.choice(sequence_length, size=5, replace=False)
    for idx in mask_indices:
        random_seq[idx] = "_"
    random_seq = "".join(random_seq)

    # esmfold
    # model, alphabet = esm2.esm.pretrained.esm2_t36_3B_UR50D()
    # batch_converter = alphabet.get_batch_converter()
    # model.eval()  # disables dropout for deterministic results
    # # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
    # data = [
    #     ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
    #     ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
    #     ("protein2 with mask", "KALTARQQEVFDLIRD<mask>ISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
    #     ("protein3", "K A <mask> I S Q"),
    # ]
    # batch_labels, batch_strs, batch_tokens = batch_converter(data)
    # # alphabet.mask_idx: 32
    # # alphabet.padding_idx: 1. work well

    folding_model = esm2.esm.pretrained.esmfold_structure_module_only_3B().eval()
    output = folding_model.infer(random_seq)
    pdbs = folding_model.output_to_pdb(output)

    # metrics
    ptm = esm_to_ptm(output)
    plddt = esm_to_plddt(output)
    tm = pdb_to_tm(pdb_value_str, pdbs[0])
    crmsd = pdb_to_crmsd(pdb_value_str, pdbs[0])
    drmsd = pdb_to_drmsd(StringIO(pdb_value_str), StringIO(pdbs[0]))
    lddt = pdb_to_lddt(pdb_value_str, pdbs[0])
    hydrophobic = pdb_to_hydrophobic_score(StringIO(pdbs[0]))
    match_ss = pdb_to_match_ss_score(StringIO(pdbs[0]))
    surface_expose = pdb_to_surface_expose_score(StringIO(pdbs[0]))
    globularity = pdb_to_globularity_score(StringIO(pdbs[0]))
