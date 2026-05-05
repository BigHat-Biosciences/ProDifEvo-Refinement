#!/bin/bash
# Entrypoint for the bh-rerd container. Activates the RERD conda env and execs
# whatever command SageMaker (or the user) passes in.
set -e

. /opt/conda/etc/profile.d/conda.sh
conda activate RERD

# Multi-GPU AF2 dispatch defaults: GPU 0 for torch (diffusion + NBB2),
# GPUs 1+ for AF workers. JAX preallocation off so it doesn't grab GPU 0.
# Caller can override either env var by setting them before launch.
: "${CUDA_VISIBLE_DEVICES:=0,1,2,3}"
: "${XLA_PYTHON_CLIENT_PREALLOCATE:=false}"
export CUDA_VISIBLE_DEVICES XLA_PYTHON_CLIENT_PREALLOCATE

# Weights baked into the image at /root/.mber. Allow override via env.
: "${AF_PARAMS_DIR:=/root/.mber/af_params}"
: "${NBB2_WEIGHTS_DIR:=/root/.mber/nbb2_weights}"
export AF_PARAMS_DIR NBB2_WEIGHTS_DIR

exec "$@"
