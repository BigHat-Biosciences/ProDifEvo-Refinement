#!/bin/bash
# Smoke test for the rerd-antibody container.
#
# Exercises every code path that matters for production:
#   * AF2 multimer complex prediction (iptm metric)
#   * NBB2 binder pre-fold + one-shot template init (--use_template)
#   * Multi-GPU AF dispatch across 3 workers (--af_gpu_ids 1,2,3)
#   * Writing artifacts to a host-mounted output dir
#
# Run from the host machine after `docker build -t rerd-antibody:test .`.
# Expected wall time: ~5-10 min (mostly first-time NBB2 + AF JIT compile).
set -euo pipefail

IMAGE="${1:-rerd-antibody:test}"
HOST_OUT="${HOST_OUT:-/tmp/rerd_smoke}"
mkdir -p "$HOST_OUT"

docker run --rm --gpus all \
    -v "$HOST_OUT":/home/output \
    "$IMAGE" \
    python ab_refinement.py \
        --antibody_sequence "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDRLSITIRPRYYGLDVWGQGTLVTVSS" \
        --antigen_pdb /home/datasets/pdl1.pdb \
        --antigen_chain A \
        --cdrs_to_design H3 \
        --metrics_name iptm,cdr_plddt,plddt \
        --metrics_list 3,1,1 \
        --repeatnum 2 \
        --duplicate 2 \
        --iteration 1 \
        --use_template \
        --af_gpu_ids 1,2,3 \
        --output_root /home/output \
        --run_name smoke

echo
echo "=== Smoke test complete ==="
echo "Outputs at $HOST_OUT:"
find "$HOST_OUT" -maxdepth 4 -type f | head -30
echo
echo "--- output.csv ---"
find "$HOST_OUT" -name output.csv -exec cat {} \;
echo
echo "--- timing_summary.txt ---"
find "$HOST_OUT" -name timing_summary.txt -exec cat {} \;
