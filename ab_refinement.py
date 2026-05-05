"""Reward-guided diffusion for antibody CDR design conditional on framework regions.

Entry point analogous to refinement.py, but specialized for antibody sequences.
Framework residues are held fixed while CDR positions are iteratively designed
using reward-guided masked diffusion.

Example usage (CDR design):
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

Example usage (binder design with ipTM):
    python ab_refinement.py \
        --antibody_sequence "EVQLVESGGGLVQPGG..." \
        --antigen_pdb datasets/pdl1.pdb \
        --antigen_chain A \
        --chain_type heavy \
        --cdrs_to_design H3 \
        --metrics_name iptm,cdr_plddt \
        --metrics_list 3,1 \
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
import time
import warnings
import numpy as np
import pandas as pd
import torch

from utils import set_seed
from ab_args_file import get_ab_args
from ab_utils import get_cdr_and_framework_indices, get_antigen_sequence
from ab_af2_reward import AbAF2RewardCal
from reward import set_diversity

warnings.filterwarnings("ignore", category=UserWarning)

ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'

current_datetime = datetime.datetime.now()


if __name__ == "__main__":
    run_t0 = time.perf_counter()
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
        args.output_root,
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

    # ---- Extract antigen sequence if provided ----
    antigen_seq = None
    if args.antigen_pdb is not None:
        antigen_chain = None
        if args.antigen_chain is not None:
            antigen_chain = args.antigen_chain.split(",") if "," in args.antigen_chain else args.antigen_chain
        antigen_seq, antigen_chain_info = get_antigen_sequence(args.antigen_pdb, antigen_chain)
        print(f"Antigen sequence ({len(antigen_seq)} aa): {antigen_seq[:50]}{'...' if len(antigen_seq) > 50 else ''}")

    # ---- Initialize reward model ----
    ori_pdb_file_path = args.ref_pdb  # Can be None if not using structural comparison metrics

    ab_reward_model = AbAF2RewardCal(
        metrics_name=args.metrics_name,
        metrics_list=args.metrics_list,
        run_name=args.run_name,
        pdb_save_path=folder_path,
        device=device,
        cdr_indices=cdr_indices,
        antigen_pdb=args.antigen_pdb,
        antigen_chain=args.antigen_chain,
        antigen_seq=antigen_seq,
        af_params_dir=args.af_params_dir,
        num_recycles=args.num_recycles,
        af_models=tuple(int(x) for x in args.af_models.split(",")),
        use_multimer=args.af_use_multimer,
        use_template=args.use_template,
        nbb2_weights_dir=args.nbb2_weights_dir,
        af_gpu_ids=[int(x) for x in args.af_gpu_ids.split(",") if x.strip()],
        seed_sequence=antibody_seq,
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
    if antigen_seq:
        print(f"Antigen: {len(antigen_seq)} residues (binder design mode)")
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

    eval_reward_model = AbAF2RewardCal(
        metrics_name=eval_metrics_name,
        metrics_list=eval_metrics_list,
        run_name=args.run_name,
        pdb_save_path=folder_path,
        device=device,
        cdr_indices=cdr_indices,
        antigen_pdb=args.antigen_pdb,
        antigen_chain=args.antigen_chain,
        antigen_seq=antigen_seq,
        af_params_dir=args.af_params_dir,
        num_recycles=args.num_recycles,
        af_models=tuple(int(x) for x in args.af_models.split(",")),
        use_multimer=args.af_use_multimer,
        use_template=args.use_template,
        nbb2_weights_dir=args.nbb2_weights_dir,
        af_gpu_ids=[int(x) for x in args.af_gpu_ids.split(",") if x.strip()],
        seed_sequence=antibody_seq,
    )

    eval_reward_names = eval_reward_model.metrics_name

    # Defensive: pin tokenized_sample back to the diffusion device. The reward
    # path can leak torch.cuda.current_device() changes across the boundary.
    tokenized_sample = tokenized_sample.to(device)

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

    # ---- Timing summary ----
    # Two distinct cost metrics:
    #   * sec_per_designed_binder = run_wall / repeat_num. The number worth quoting
    #     for campaign planning ("how long does it take to design one binder").
    #   * sec_per_af_prediction = total reward time / total AF prediction count.
    #     A throughput metric ("how fast is the AF backend per call"), useful for
    #     engineering. Note this counts every intermediate candidate scored during
    #     refinement, which is much larger than the number of designed binders.
    run_wall = time.perf_counter() - run_t0
    refine_t = ab_reward_model._timings
    eval_t = eval_reward_model._timings
    total_preds = refine_t["n_sequences"] + eval_t["n_sequences"]
    total_reward_s = refine_t["reward_seconds"] + eval_t["reward_seconds"]
    avg_s_per_pred = total_reward_s / max(total_preds, 1)
    sec_per_designed_binder = run_wall / max(repeat_num, 1)

    summary_lines = [
        "=== Run timing summary ===",
        f"Total run wall time:           {run_wall:.2f} s",
        f"Designed binders:              {repeat_num}",
        f"Sec per designed binder:       {sec_per_designed_binder:.2f} s",
        "",
        f"Refinement reward time:        {refine_t['reward_seconds']:.2f} s "
        f"({refine_t['n_sequences']} AF predictions, {refine_t['n_calls']} batched calls)",
        f"Final eval reward time:        {eval_t['reward_seconds']:.2f} s "
        f"({eval_t['n_sequences']} AF predictions, {eval_t['n_calls']} batched calls)",
        f"Total AF predictions:          {total_preds}",
        f"Avg sec per AF prediction:     {avg_s_per_pred:.3f} s",
        f"Per-iteration timings:         see {os.path.join(folder_path, 'timing.csv')}",
    ]
    print("\n" + "\n".join(summary_lines))
    with open(os.path.join(folder_path, "timing_summary.txt"), "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    logging.info("Timing summary: %s", " | ".join(summary_lines[1:]))
