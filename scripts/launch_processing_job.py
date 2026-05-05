"""Launch a rerd-antibody processing job on SageMaker.

Run from a checkout of the bh-ai repo (so `bh.aicore.training.sage` is importable).

Example:

    python scripts/launch_processing_job.py \\
        --antigen-s3-uri s3://332120041740-bighat-datasets/MyData/pdl1.pdb \\
        --antibody-sequence "EVQLVESGGGLVQPGG..." \\
        --cdrs-to-design H1,H2,H3 \\
        --repeatnum 100 \\
        --iteration 10

Multiple jobs can be launched by editing the parameter sweep block at the bottom.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

from sagemaker.processing import ProcessingInput, ProcessingOutput

from bh.aicore.config import SAGEMAKER_GPU_MEDIUM_INSTANCE_TYPE
from bh.aicore.training.sage import launch_container_on_sagemaker


# /opt/ml/processing/* layout follows SageMaker's default convention:
#   inputs land under /opt/ml/processing/input/<input_name>/
#   outputs are uploaded from /opt/ml/processing/output/<output_name>/
CONTAINER_INPUT_DIR = "/opt/ml/processing/input/antigen"
CONTAINER_OUTPUT_DIR = "/opt/ml/processing/output"
CONTAINER_ANTIGEN_PDB = os.path.join(CONTAINER_INPUT_DIR, "antigen.pdb")


def build_command(
    *,
    antibody_sequence: str,
    cdrs_to_design: str,
    metrics_name: str,
    metrics_list: str,
    repeatnum: int,
    duplicate: int,
    iteration: int,
    af_models: str,
    af_gpu_ids: str,
    use_template: bool,
    seed: int,
    run_name: str,
) -> list[str]:
    """Assemble the python command line for ab_refinement.py inside the container."""
    cmd = f"""
        ab_refinement.py
        --antibody_sequence {antibody_sequence}
        --antigen_pdb {CONTAINER_ANTIGEN_PDB}
        --antigen_chain A
        --chain_type heavy
        --cdrs_to_design {cdrs_to_design}
        --numbering_scheme imgt
        --metrics_name {metrics_name}
        --metrics_list {metrics_list}
        --repeatnum {repeatnum}
        --duplicate {duplicate}
        --iteration {iteration}
        --decoding SVDD_edit
        --num_recycles 3
        --af_models {af_models}
        --af_gpu_ids {af_gpu_ids}
        --seed {seed}
        --run_name {run_name}
        --output_root {CONTAINER_OUTPUT_DIR}
    """
    if use_template:
        cmd += " --use_template"
    return [seg for line in cmd.splitlines() for seg in line.strip().split(" ") if seg]


def launch_one(
    *,
    antigen_s3_uri: str,
    antibody_sequence: str,
    cdrs_to_design: str = "H1,H2,H3",
    metrics_name: str = "iptm,cdr_plddt,plddt",
    metrics_list: str = "3,1,1",
    repeatnum: int = 100,
    duplicate: int = 5,
    iteration: int = 10,
    af_models: str = "0",
    af_gpu_ids: str = "1,2,3",
    use_template: bool = True,
    seed: int = 0,
    run_name: str = "rerd_run",
    image_tag: str = "latest",
    timeout_hours: int = 24,
):
    inputs = [
        ProcessingInput(
            source=antigen_s3_uri,
            destination=CONTAINER_INPUT_DIR,
            input_name="antigen",
        ),
    ]
    outputs = [
        ProcessingOutput(
            output_name="rerd_output",
            source=CONTAINER_OUTPUT_DIR,
            s3_upload_mode="Continuous",
        ),
    ]
    cmd = build_command(
        antibody_sequence=antibody_sequence,
        cdrs_to_design=cdrs_to_design,
        metrics_name=metrics_name,
        metrics_list=metrics_list,
        repeatnum=repeatnum,
        duplicate=duplicate,
        iteration=iteration,
        af_models=af_models,
        af_gpu_ids=af_gpu_ids,
        use_template=use_template,
        seed=seed,
        run_name=run_name,
    )
    return launch_container_on_sagemaker(
        image_repo="rerd-antibody",
        image_tag=image_tag,
        command="python",
        arguments=cmd,
        inputs=inputs,
        outputs=outputs,
        instance_type=SAGEMAKER_GPU_MEDIUM_INSTANCE_TYPE,
        timeout_in_seconds=timeout_hours * 60 * 60,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--antigen-s3-uri", required=True,
                   help="s3://... path to a single-chain antigen PDB.")
    p.add_argument("--antibody-sequence", required=True,
                   help="Full heavy-chain antibody sequence to seed design from.")
    p.add_argument("--cdrs-to-design", default="H1,H2,H3")
    p.add_argument("--repeatnum", type=int, default=100)
    p.add_argument("--duplicate", type=int, default=5)
    p.add_argument("--iteration", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", default="rerd_run")
    p.add_argument("--image-tag", default="latest")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    launch_one(
        antigen_s3_uri=args.antigen_s3_uri,
        antibody_sequence=args.antibody_sequence,
        cdrs_to_design=args.cdrs_to_design,
        repeatnum=args.repeatnum,
        duplicate=args.duplicate,
        iteration=args.iteration,
        seed=args.seed,
        run_name=args.run_name,
        image_tag=args.image_tag,
    )
