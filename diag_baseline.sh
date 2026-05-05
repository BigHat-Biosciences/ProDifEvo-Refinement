#!/bin/bash
# Diagnostic baseline: NO templating, single GPU.
# Reproduces the original ~34s/it run config to anchor comparisons.
set -e

export AF_PARAMS_DIR="${AF_PARAMS_DIR:-$HOME/.mber/af_params}"
export CUDA_VISIBLE_DEVICES=0
unset XLA_PYTHON_CLIENT_PREALLOCATE

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
    --run_name diag_baseline
