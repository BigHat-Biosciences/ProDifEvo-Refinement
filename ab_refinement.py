"""Reward-guided diffusion for antibody CDR design conditional on framework regions.

Entry point analogous to refinement.py, but specialized for antibody sequences.
Framework residues are held fixed while CDR positions are iteratively designed
using reward-guided masked diffusion.

Example usage:
    python ab_refinement.py \
        --antibody_sequence "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDRLSITIRPRYYGLDVWGQGTLVTVSS" \
        --chain_type heavy \
        --cdrs_to_design H3 \
        --numbering_scheme imgt \
        --metrics_name plddt,cdr_plddt \
        --metrics_list 1,2 \
        --repeatnum 5 \
        --duplicate 10 \
        --iteration 20
"""

from evodiff.pretrained import OA_DM_38M, OA_DM_640M
from evodiff.generate import likelihood
from evodiff.generate_antibody import generate_oaardm_cdr_edit, generate_oaardm_cdr_svdd
import os
import datetime
import logging
import warnings
import numpy as np
import pandas as pd
import torch
import esm

from utils import set_seed
from ab_args_file import get_ab_args
from ab_utils import get_cdr_and_framework_indices
from reward import (
    esm_to_ptm, esm_to_plddt, esm_to_cdr_plddt,
    cdr_charge_score, cdr_hydrophobicity_score,
    pdb_to_tm, pdb_to_crmsd, pdb_to_hydrophobic_score,
    pdb_to_match_ss_score, pdb_to_surface_expose_score,
    pdb_to_globularity_score, set_diversity,
)

warnings.filterwarnings("ignore", category=UserWarning)

ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'

current_datetime = datetime.datetime.now()


class AbRewardCal:
    """Reward calculator for antibody CDR design.

    Extends the pattern from refinement.py's RewardCal with CDR-aware metrics.
    """

    def __init__(
        self,
        metrics_name,
        metrics_list,
        esm_model,
        device,
        cdr_indices=None,
        run_name="",
        pdb_save_path="ab_sc_tmp",
    ):
        self.metrics_name = metrics_name.split(",")
        metrics_list = metrics_list.split(",")
        self.metrics_list = [float(x) for x in metrics_list]
        assert len(self.metrics_name) == len(self.metrics_list), (
            f"Mismatch: {len(self.metrics_name)} metric names vs {len(self.metrics_list)} weights"
        )

        self.cdr_indices = cdr_indices if cdr_indices is not None else []
        self.pdb_save_path = pdb_save_path
        self.run_name = run_name

        # Initialize ESMFold
        self.folding_model = esm.pretrained.esmfold_v1().eval()
        self.folding_model = self.folding_model.to(device)

    def metrics_cal(self, metrics_name, ori_pdb_file=None, gen_pdb_file=None,
                    folding_results=None, protein_idx=0, save_pdb=False,
                    pdb_raw=None, sequence_str=None):
        """Calculate each requested metric."""
        all_results = []
        for metric in metrics_name:
            if metric == 'ptm':
                r = esm_to_ptm(folding_results, idx=protein_idx)
            elif metric == 'plddt':
                r = esm_to_plddt(folding_results, idx=protein_idx)
            elif metric == 'cdr_plddt':
                r = esm_to_cdr_plddt(folding_results, self.cdr_indices, idx=protein_idx)
            elif metric == 'charge_balance':
                r = cdr_charge_score(sequence_str, self.cdr_indices) if sequence_str else 0.0
            elif metric == 'cdr_hydrophobicity':
                from io import StringIO
                pdb_input = gen_pdb_file if save_pdb else StringIO(gen_pdb_file)
                r = cdr_hydrophobicity_score(pdb_input, self.cdr_indices)
            elif metric == 'tm':
                r = pdb_to_tm(ori_pdb_file, pdb_raw)
            elif metric == 'crmsd':
                r = pdb_to_crmsd(ori_pdb_file, pdb_raw)
            elif metric == 'hydrophobic':
                from io import StringIO
                r = pdb_to_hydrophobic_score(gen_pdb_file if save_pdb else StringIO(gen_pdb_file))
            elif metric == 'match_ss':
                from io import StringIO
                seq_len = len(sequence_str) if sequence_str else 0
                r, _ = pdb_to_match_ss_score(ori_pdb_file, gen_pdb_file if save_pdb else StringIO(gen_pdb_file), 1, seq_len + 1)
            elif metric == 'surface_expose':
                from io import StringIO
                r = pdb_to_surface_expose_score(gen_pdb_file if save_pdb else StringIO(gen_pdb_file))
            elif metric == 'globularity':
                from io import StringIO
                r = pdb_to_globularity_score(gen_pdb_file if save_pdb else StringIO(gen_pdb_file))
            else:
                raise NotImplementedError(f"Metric '{metric}' not implemented for antibody design.")
            all_results.append(r)
        return all_results

    def reward_metrics(self, protein_name, mask_for_loss, S_sp, ori_pdb_file,
                       save_pdb=False, add_info=""):
        """Compute reward from folded structures. Follows refinement.py RewardCal pattern."""
        sc_output_dir = os.path.join(self.pdb_save_path, self.run_name)
        os.makedirs(sc_output_dir, exist_ok=True)
        esm_input_data = []

        for _it, ssp in enumerate(S_sp):
            seq_string = "".join([
                ALPHABET[x] for _ix, x in enumerate(ssp)
                if mask_for_loss[_it][_ix] == 1
            ])
            esm_input_data.append(seq_string)

        # ESMFold forward
        output = self.folding_model.infer(esm_input_data)
        pdbs = self.folding_model.output_to_pdb(output)

        # Reward calculation
        record_reward, record_reward_agg = [], []
        for _it, pdb in enumerate(pdbs):
            if save_pdb:
                pdb_path = os.path.join(sc_output_dir, f"{protein_name}{_it}_{add_info}.pdb")
                with open(pdb_path, "w") as ff:
                    ff.write(pdb)
                fasta_path = os.path.join(sc_output_dir, f"{protein_name}{_it}_{add_info}.fasta")
                with open(fasta_path, 'w') as f:
                    f.write(f">{protein_name}\n{esm_input_data[_it]}\n")
            else:
                pdb_path = pdb

            all_reward = self.metrics_cal(
                metrics_name=self.metrics_name,
                gen_pdb_file=pdb_path,
                ori_pdb_file=ori_pdb_file,
                folding_results=output,
                protein_idx=_it,
                save_pdb=save_pdb,
                pdb_raw=pdb,
                sequence_str=esm_input_data[_it],
            )
            aggregate_reward = sum(v * w for v, w in zip(all_reward, self.metrics_list))
            record_reward_agg.append(aggregate_reward)
            record_reward.append(all_reward)

        return record_reward, record_reward_agg, 0.0


if __name__ == "__main__":
    args = get_ab_args()

    # ---- Parse CDR / Framework boundaries ----
    antibody_seq = args.antibody_sequence
    seq_len = len(antibody_seq)

    if args.cdr_indices is not None:
        manual_indices = args.cdr_indices
    elif args.auto_number == "True":
        manual_indices = None
    else:
        raise ValueError(
            "Either provide --cdr_indices or set --auto_number True (requires anarci)."
        )

    cdr_indices, framework_indices = get_cdr_and_framework_indices(
        sequence=antibody_seq,
        scheme=args.numbering_scheme,
        chain_type=args.chain_type,
        cdrs_to_design=args.cdrs_to_design,
        manual_cdr_indices=manual_indices,
    )

    # ---- Setup ----
    folder_path = os.path.join(
        "log",
        f"{current_datetime}_ab_{args.decoding}_{args.metrics_name}_{args.chain_type}_{args.cdrs_to_design}",
    )
    os.makedirs(folder_path, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=os.path.join(folder_path, "run.log"),
    )
    logging.info(args)

    set_seed(args.seed, torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load diffusion model ----
    if args.diffusion_model == "38M":
        checkpoint = OA_DM_38M()
    else:
        checkpoint = OA_DM_640M()
    model, collater, tokenizer, scheme = checkpoint
    model = model.to(device)

    # ---- Tokenize antibody sequence ----
    repeat_num = args.repeatnum
    S_initial = torch.from_numpy(tokenizer.tokenize([antibody_seq])).to(device)
    S_initial = S_initial.repeat(repeat_num, 1)

    mask_for_loss = torch.ones((repeat_num, seq_len))

    # ---- Initialize reward model ----
    ori_pdb_file_path = args.ref_pdb  # Can be None if not using structural comparison metrics

    ab_reward_model = AbRewardCal(
        metrics_name=args.metrics_name,
        metrics_list=args.metrics_list,
        esm_model=args.esm_model,
        run_name=args.run_name,
        pdb_save_path=folder_path,
        device=device,
        cdr_indices=cdr_indices,
    )

    reward_name_list = ab_reward_model.metrics_name
    protein_name = "ab_cdr_design"

    # ---- Run CDR design ----
    print(f"\n{'='*60}")
    print(f"Antibody CDR Design")
    print(f"Chain type: {args.chain_type}")
    print(f"CDRs to design: {args.cdrs_to_design}")
    print(f"CDR positions: {len(cdr_indices)} residues")
    print(f"Framework positions: {len(framework_indices)} residues (fixed)")
    print(f"Decoding: {args.decoding}")
    print(f"Iterations: {args.iteration}")
    print(f"Batch size: {repeat_num}, Candidates: {args.duplicate}")
    print(f"Metrics: {args.metrics_name} (weights: {args.metrics_list})")
    print(f"{'='*60}\n")

    if args.decoding == "SVDD_edit":
        tokenized_sample, generated_sequence = generate_oaardm_cdr_edit(
            model, tokenizer, seq_len, ab_reward_model,
            ori_pdb_file_path=ori_pdb_file_path,
            batch=protein_name,
            mask_for_loss=mask_for_loss,
            repeat_num=repeat_num,
            candidate=args.duplicate,
            folder_path=folder_path,
            device=device,
            cdr_indices=cdr_indices,
            framework_indices=framework_indices,
            iteration=args.iteration,
            edit_fraction=args.edit_fraction,
            initial_sample=S_initial,
        )
    elif args.decoding == "SVDD":
        tokenized_sample, generated_sequence = generate_oaardm_cdr_svdd(
            model, tokenizer, seq_len, ab_reward_model,
            ori_pdb_file_path=ori_pdb_file_path,
            batch=protein_name,
            mask_for_loss=mask_for_loss,
            repeat_num=repeat_num,
            candidate=args.duplicate,
            device=device,
            cdr_indices=cdr_indices,
            framework_indices=framework_indices,
            initial_sample=S_initial,
        )
    else:
        raise NotImplementedError(f"Decoding strategy '{args.decoding}' not supported for antibody design.")

    # ---- Evaluation ----
    print("\n--- Final Evaluation ---")

    # Add plddt and ptm to eval metrics if not already present
    eval_metrics_name = args.metrics_name
    eval_metrics_list = args.metrics_list
    for extra in ["plddt", "ptm"]:
        if extra not in eval_metrics_name:
            eval_metrics_name += f",{extra}"
            eval_metrics_list += ",1"

    eval_reward_model = AbRewardCal(
        metrics_name=eval_metrics_name,
        metrics_list=eval_metrics_list,
        esm_model=args.esm_model,
        run_name=args.run_name,
        pdb_save_path=folder_path,
        device=device,
        cdr_indices=cdr_indices,
    )

    eval_reward_names = eval_reward_model.metrics_name

    # Compute model likelihood
    likelihood_reward = likelihood(model, tokenizer, seq_len, tokenized_sample, repeat_num, device)

    # Compute final rewards
    cur_reward_before, cur_reward_agg, _ = eval_reward_model.reward_metrics(
        protein_name=protein_name,
        mask_for_loss=mask_for_loss,
        S_sp=tokenized_sample,
        ori_pdb_file=ori_pdb_file_path,
        save_pdb=True,
    )

    cur_reward_before = list(map(list, zip(*cur_reward_before)))

    df = pd.DataFrame(np.array(cur_reward_before).transpose(), columns=eval_reward_names)
    df['likelihood'] = likelihood_reward

    cur_reward = [sum(sublist) / len(sublist) for sublist in cur_reward_before]
    cur_reward_agg_mean = sum(cur_reward_agg) / len(cur_reward_agg)

    cur_diversity = set_diversity(
        tokenized_sample.detach().cpu().numpy(),
        mask_for_loss.detach().cpu().numpy(),
    )
    df['diversity'] = np.array([cur_diversity for _ in range(repeat_num)])

    # Save sequences
    df['sequence'] = generated_sequence
    df['cdr_indices'] = str(cdr_indices)
    df['framework_fixed'] = True

    output_csv = os.path.join(folder_path, "output.csv")
    df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")

    # Print summary
    print(f"\nMetric averages:")
    for i, name in enumerate(eval_reward_names):
        print(f"  {name}: {cur_reward[i]:.4f}")
    print(f"  aggregate_reward: {cur_reward_agg_mean:.4f}")
    print(f"  diversity: {cur_diversity:.4f}")

    # Print designed sequences
    print(f"\nDesigned sequences:")
    for i, seq in enumerate(generated_sequence):
        cdr_only = "".join([seq[j] for j in cdr_indices])
        print(f"  [{i}] CDR: {cdr_only}")
