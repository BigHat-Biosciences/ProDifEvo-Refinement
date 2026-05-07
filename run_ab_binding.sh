#!/bin/bash
# AF2.3M (multimer) reward backend, bonobo-style: pre-made target+binder
# template PDB + per-target hotspot. Generate templates once via
# scripts/generate_template.py (see datasets/template_*.pdb for the four
# baked-in targets).
export AF_PARAMS_DIR="${AF_PARAMS_DIR:-$HOME/.mber/af_params}"

# Multi-GPU layout (g5.12xlarge: 4× A10G):
#   GPU 0 → torch (diffusion model)
#   GPU 1,2,3 → AF2 prediction workers (--af_gpu_ids 1,2,3)
# CUDA_VISIBLE_DEVICES must expose all four. JAX preallocation is disabled so
# JAX doesn't grab all of GPU 0 — we keep that for torch.
export CUDA_VISIBLE_DEVICES=0,1,2,3
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python ab_refinement.py \
    --antibody_sequence "EVQLVESGGGLVQPGGSLRLSCAASGGFTFSSYAMWFRQAPGKEREFAISGSGGSTYYNADSVKGRFTISRDNAKNTLYLQMNSLRAEDTAVYYCARLSITIRPYYGWGQGTLVTVSS" \
    --antigen_pdb datasets/pdl1.pdb \
    --antigen_chain A \
    --chain_type heavy \
    --cdrs_to_design H1,H2,H3 \
    --numbering_scheme imgt \
    --metrics_name iptm,cdr_plddt,plddt \
    --metrics_list 3,1,1 \
    --repeatnum 100 \
    --duplicate 5 \
    --iteration 10 \
    --decoding SVDD_edit \
    --af_params_dir "$AF_PARAMS_DIR" \
    --num_recycles 3 \
    --af_models 0 \
    --template_pdb datasets/template_pdl1.pdb \
    --hotspot A113 \
    --af_gpu_ids 1,2,3
