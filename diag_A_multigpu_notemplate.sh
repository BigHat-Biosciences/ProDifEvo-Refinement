#!/bin/bash
# Diagnostic A: multi-GPU AF dispatch, NO templating.
# Goal: confirm whether --af_gpu_ids alone parallelizes correctly vs the
# single-GPU baseline (which gave ~34s/it on this run config).
set -e

export AF_PARAMS_DIR="${AF_PARAMS_DIR:-$HOME/.mber/af_params}"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python ab_refinement.py \
    --antibody_sequence "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDRLSITIRPRYYGLDVWGQGTLVTVSS" \
    --antigen_pdb datasets/pdl1.pdb \
    --antigen_chain A \
    --chain_type heavy \
    --cdrs_to_design H3 \
    --numbering_scheme imgt \
    --metrics_name iptm,cdr_plddt,plddt \
    --metrics_list 3,1,1 \
    --repeatnum 3 \
    --duplicate 5 \
    --iteration 10 \
    --decoding SVDD_edit \
    --af_params_dir "$AF_PARAMS_DIR" \
    --num_recycles 3 \
    --af_models 0 \
    --af_gpu_ids 1,2,3 \
    --run_name diag_A_multigpu_notemplate
