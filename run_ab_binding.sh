#!/bin/bash
# AF2.3M (multimer) is now the folding backend. Set AF_PARAMS_DIR or pass
# --af_params_dir to point at the directory containing the AlphaFold2 weights.
# Default location matches mber-open's download_weights.sh: ~/.mber/af_params.
export AF_PARAMS_DIR="${AF_PARAMS_DIR:-$HOME/.mber/af_params}"

CUDA_VISIBLE_DEVICES=1 python ab_refinement.py \
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
    --af_models 0
