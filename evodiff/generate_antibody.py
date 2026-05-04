"""CDR-constrained reward-guided diffusion generation for antibody design.

Adapts the SVDD_edit pattern from generate.py to only modify CDR positions
while keeping framework residues fixed.
"""

import csv
import logging
import os
import random
import time
import numpy as np
import torch
from tqdm import tqdm


@torch.no_grad()
def generate_oaardm_cdr_edit(
    model,
    tokenizer,
    seq_len,
    reward_model,
    ori_pdb_file_path,
    batch,
    mask_for_loss,
    repeat_num,
    candidate,
    folder_path,
    device,
    cdr_indices,
    framework_indices,
    iteration=30,
    edit_fraction=0.3,
    initial_sample=None,
):
    """Reward-guided masked diffusion that only designs CDR positions.

    Framework residues are never modified. CDR positions are iteratively
    masked and resampled with reward-weighted selection.

    Args:
        model: Pretrained EvoDiff OA-ARDM model.
        tokenizer: EvoDiff tokenizer.
        seq_len: Total sequence length.
        reward_model: RewardCal (or AbRewardCal) instance.
        ori_pdb_file_path: Path to reference PDB (or None).
        batch: Protein name string for logging.
        mask_for_loss: Tensor of shape (repeat_num, seq_len), 1s for valid positions.
        repeat_num: Batch size (number of parallel designs).
        candidate: Number of candidates to evaluate per step.
        folder_path: Output directory for trajectory logging.
        device: torch device.
        cdr_indices: List of 0-based CDR position indices to design.
        framework_indices: List of 0-based framework position indices to keep fixed.
        iteration: Number of refinement iterations.
        edit_fraction: Fraction of CDR positions to re-mask each iteration.
        initial_sample: Tokenized initial sequence tensor, shape (repeat_num, seq_len).

    Returns:
        sample: Final tokenized sequences, shape (repeat_num, seq_len).
        untokenized: List of untokenized sequence strings.
    """
    mask = tokenizer.mask_id
    num_cdr = len(cdr_indices)
    num_to_remask = max(1, int(edit_fraction * num_cdr))

    # Initialize: start from the provided antibody sequence
    sample = initial_sample.clone().to(device)

    # Store original framework tokens to enforce invariance
    framework_tokens = initial_sample[0, framework_indices].clone().to(device)

    def _enforce_framework(s):
        """Ensure framework positions are never modified."""
        s[:, framework_indices] = framework_tokens.unsqueeze(0)
        return s

    # Mask all CDR positions for initial generation
    for idx in cdr_indices:
        sample[:, idx] = mask

    sample = _enforce_framework(sample)

    pbar = tqdm(range(iteration), desc="Refinement iterations")
    timing_csv = os.path.join(folder_path, 'timing.csv')
    for ttt in pbar:
        iter_t0 = time.perf_counter()
        n_seqs_before = reward_model._timings["n_sequences"] if hasattr(reward_model, "_timings") else 0

        # Determine positions to unmask this iteration
        if ttt == 0:
            # First iteration: unmask all CDR positions from scratch
            loc_set = [list(cdr_indices) for _ in range(repeat_num)]
            length_edit = num_cdr
        # else: loc_set and length_edit are set at the end of the previous iteration

        # Inner loop: unmask one CDR position at a time
        for count in tqdm(range(length_edit), desc=f"Iter {ttt} denoising", leave=False):
            timestep = torch.tensor([0] * repeat_num).to(device)
            prediction = model(sample, timestep)  # (B, L, vocab)

            # Generate candidates
            next_candidate = []
            pes_index_list = []
            for jjj in range(candidate):
                # Choose a random masked CDR position from each sample's remaining set
                loc_list = [np.random.randint(len(loc_set[i])) for i in range(repeat_num)]
                pes_index = [loc_set[i][loc_list[i]] for i in range(repeat_num)]
                pes_index_list.append(pes_index)

                # Get model prediction at chosen position, restrict to canonical AAs
                p = torch.stack([
                    prediction[i, pes_index[i], 0:20] for i in range(repeat_num)
                ])
                p = torch.nn.functional.softmax(p, dim=1)  # (B, 20)
                p_sample = torch.multinomial(p, num_samples=1)  # (B, 1)

                sample_fake = sample.clone()
                for iii in range(repeat_num):
                    sample_fake[iii, pes_index[iii]] = p_sample.squeeze()[iii]
                sample_fake = _enforce_framework(sample_fake)
                next_candidate.append(sample_fake.clone())

            # Evaluate candidates via reward model
            reward_list = np.zeros((repeat_num, candidate))
            for jjj in range(candidate):
                pred = model(next_candidate[jjj], timestep)
                # Fill remaining masked positions with argmax for reward evaluation
                next_seq = (
                    next_candidate[jjj] * (next_candidate[jjj] != mask)
                    + torch.argmax(pred[:, :, 0:20], dim=2) * (next_candidate[jjj] == mask)
                )
                next_seq = _enforce_framework(next_seq)
                _, reward_hoge, _ = reward_model.reward_metrics(
                    protein_name=batch,
                    mask_for_loss=mask_for_loss,
                    S_sp=next_seq,
                    ori_pdb_file=ori_pdb_file_path,
                )
                reward_list[:, jjj] = reward_hoge

            # Select best candidate per sample
            next_index = np.argmax(reward_list, 1)
            next_candidate = torch.stack(next_candidate)
            sample = torch.stack([
                next_candidate[next_index[i], i, :] for i in range(repeat_num)
            ])
            sample = _enforce_framework(sample)

            # Remove the unmasked position from each sample's location set
            for jjj in range(repeat_num):
                chosen_pos = pes_index_list[next_index[jjj]][jjj]
                if chosen_pos in loc_set[jjj]:
                    loc_set[jjj].remove(chosen_pos)

        # End-of-iteration evaluation
        per_metric_rewards, reward_hoge, _ = reward_model.reward_metrics(
            protein_name=batch,
            mask_for_loss=mask_for_loss,
            S_sp=sample,
            ori_pdb_file=ori_pdb_file_path,
            save_pdb=True,
            add_info=ttt,
        )
        # per_metric_rewards: list of lists, one per sample, each inner list = per-metric values
        # reward_hoge: list of aggregate rewards, one per sample
        metric_names = reward_model.metrics_name

        # Write detailed per-iteration CSV (append mode; write header on first iteration)
        trajectory_csv = os.path.join(folder_path, 'ab_trajectory.csv')
        write_header = (ttt == 0) and not os.path.exists(trajectory_csv)
        with open(trajectory_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['iteration', 'sample_idx'] + metric_names + ['aggregate_reward'])
            for _s_idx in range(repeat_num):
                row = [ttt, _s_idx] + [f"{v:.6f}" for v in per_metric_rewards[_s_idx]] + [f"{reward_hoge[_s_idx]:.6f}"]
                writer.writerow(row)

        # Print summary
        mean_agg = np.mean(reward_hoge)
        per_metric_means = [np.mean([per_metric_rewards[s][m] for s in range(repeat_num)])
                            for m in range(len(metric_names))]
        metric_summary = ", ".join(f"{n}={v:.4f}" for n, v in zip(metric_names, per_metric_means))
        print(f"Iteration {ttt}: agg={mean_agg:.4f}, {metric_summary}")
        pbar.set_postfix_str(f"agg={mean_agg:.3f}, {metric_summary}")

        # Per-iteration timing. n_af_predictions = total candidate sequences scored
        # by AF this iteration (= num_inner_steps * candidate * repeat_num + repeat_num
        # for the end-of-iter eval). Distinct from "designed binders", which is repeat_num.
        iter_wall = time.perf_counter() - iter_t0
        n_preds_after = reward_model._timings["n_sequences"] if hasattr(reward_model, "_timings") else 0
        n_preds_iter = n_preds_after - n_seqs_before
        sec_per_pred = iter_wall / max(n_preds_iter, 1)
        logging.info(
            f"Iteration {ttt} timing: wall={iter_wall:.2f}s, "
            f"n_af_predictions={n_preds_iter}, sec_per_af_prediction={sec_per_pred:.3f}"
        )
        write_header_t = (ttt == 0) and not os.path.exists(timing_csv)
        with open(timing_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header_t:
                writer.writerow(['iteration', 'wall_seconds', 'n_af_predictions', 'sec_per_af_prediction'])
            writer.writerow([ttt, f"{iter_wall:.3f}", n_preds_iter, f"{sec_per_pred:.3f}"])

        # Return on final iteration
        if ttt == iteration - 1:
            untokenized = [tokenizer.untokenize(s) for s in sample]
            return sample, untokenized

        # Reward-weighted resampling across the population
        reward_arr = np.array(reward_hoge)
        reward_arr = np.exp(reward_arr * 5)
        reward_arr = np.nan_to_num(reward_arr, nan=0.0)
        reward_probs = reward_arr / np.sum(reward_arr)
        sampled_values = np.random.choice(
            repeat_num, size=repeat_num, p=reward_probs
        )
        sample = sample[sampled_values, :]

        # Re-mask a fraction of CDR positions for next iteration
        loc_set = [
            random.sample(list(cdr_indices), num_to_remask)
            for _ in range(repeat_num)
        ]
        length_edit = num_to_remask
        for iii in range(repeat_num):
            for pos in loc_set[iii]:
                sample[iii, pos] = mask
        sample = _enforce_framework(sample)


@torch.no_grad()
def generate_oaardm_cdr_svdd(
    model,
    tokenizer,
    seq_len,
    reward_model,
    ori_pdb_file_path,
    batch,
    mask_for_loss,
    repeat_num,
    candidate,
    device,
    cdr_indices,
    framework_indices,
    initial_sample=None,
):
    """One-pass SVDD CDR design: mask CDR positions, unmask with reward guidance.

    No iterative refinement -- a single pass of unmasking all CDR positions
    with best-of-N candidate selection at each step.

    Args:
        (Same as generate_oaardm_cdr_edit, minus iteration/edit_fraction/folder_path.)

    Returns:
        sample: Final tokenized sequences.
        untokenized: List of untokenized sequence strings.
    """
    mask = tokenizer.mask_id

    sample = initial_sample.clone().to(device)
    framework_tokens = initial_sample[0, framework_indices].clone().to(device)

    def _enforce_framework(s):
        s[:, framework_indices] = framework_tokens.unsqueeze(0)
        return s

    # Mask CDR positions
    for idx in cdr_indices:
        sample[:, idx] = mask
    sample = _enforce_framework(sample)

    # Build location sets (only CDR positions)
    loc_set = [list(cdr_indices) for _ in range(repeat_num)]

    for count in tqdm(range(len(cdr_indices)), desc="CDR SVDD denoising"):
        timestep = torch.tensor([0] * repeat_num).to(device)
        prediction = model(sample, timestep)

        next_candidate = []
        pes_index_list = []
        for jjj in range(candidate):
            loc_list = [np.random.randint(len(loc_set[i])) for i in range(repeat_num)]
            pes_index = [loc_set[i][loc_list[i]] for i in range(repeat_num)]
            pes_index_list.append(pes_index)

            p = torch.stack([
                prediction[i, pes_index[i], 0:20] for i in range(repeat_num)
            ])
            p = torch.nn.functional.softmax(p, dim=1)
            p_sample = torch.multinomial(p, num_samples=1)

            sample_fake = sample.clone()
            for iii in range(repeat_num):
                sample_fake[iii, pes_index[iii]] = p_sample.squeeze()[iii]
            sample_fake = _enforce_framework(sample_fake)
            next_candidate.append(sample_fake.clone())

        # Evaluate and select
        reward_list = np.zeros((repeat_num, candidate))
        for jjj in range(candidate):
            pred = model(next_candidate[jjj], timestep)
            next_seq = (
                next_candidate[jjj] * (next_candidate[jjj] != mask)
                + torch.argmax(pred[:, :, 0:20], dim=2) * (next_candidate[jjj] == mask)
            )
            next_seq = _enforce_framework(next_seq)
            _, reward_hoge, _ = reward_model.reward_metrics(
                protein_name=batch,
                mask_for_loss=mask_for_loss,
                S_sp=next_seq,
                ori_pdb_file=ori_pdb_file_path,
            )
            reward_list[:, jjj] = reward_hoge

        next_index = np.argmax(reward_list, 1)
        next_candidate = torch.stack(next_candidate)
        sample = torch.stack([
            next_candidate[next_index[i], i, :] for i in range(repeat_num)
        ])
        sample = _enforce_framework(sample)
        for jjj in range(repeat_num):
            loc_set[jjj].remove(pes_index_list[next_index[jjj]][jjj])

    untokenized = [tokenizer.untokenize(s) for s in sample]
    return sample, untokenized
