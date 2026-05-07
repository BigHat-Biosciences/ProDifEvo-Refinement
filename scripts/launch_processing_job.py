"""Launch a rerd-antibody processing job on SageMaker.

Run from a checkout of the bh-ai repo (so `bh.aicore.training.sage` is importable).

Bonobo-style binder design: the antigen and a pre-made target+binder template
PDB are baked into the image (datasets/{name}.pdb and datasets/template_{name}.pdb).
The launcher auto-fills these plus the per-target hotspot when ``--antigen``
is one of the baked names.

For a custom target, pass ``--antigen s3://...`` and provide ``--template-s3-uri``
+ ``--hotspot``; both will be plumbed into the container.

Examples:

    # Use the pdl1 PDB and template baked into the image:
    python scripts/launch_processing_job.py \\
        --antigen pdl1 \\
        --antibody-sequence "EVQLVESGGGLVQPGG..." \\
        --repeatnum 100

    # Custom target:
    python scripts/launch_processing_job.py \\
        --antigen s3://332120041740-bighat-datasets/MyData/custom.pdb \\
        --template-s3-uri s3://332120041740-bighat-datasets/MyData/template_custom.pdb \\
        --hotspot A45,A46 \\
        --antibody-sequence "EVQLVESGGGLVQPGG..."
"""

from __future__ import annotations

import argparse
import os
from typing import Optional, Tuple

from sagemaker.processing import ProcessingInput, ProcessingOutput

from bh.aicore.config import SAGEMAKER_GPU_MEDIUM_INSTANCE_TYPE
from bh.aicore.training.sage import launch_container_on_sagemaker


# /opt/ml/processing/* layout follows SageMaker's default convention.
CONTAINER_INPUT_DIR = "/opt/ml/processing/input"
CONTAINER_ANTIGEN_INPUT_DIR = os.path.join(CONTAINER_INPUT_DIR, "antigen")
CONTAINER_TEMPLATE_INPUT_DIR = os.path.join(CONTAINER_INPUT_DIR, "template")
CONTAINER_OUTPUT_DIR = "/opt/ml/processing/output"
CONTAINER_ANTIGEN_PDB = os.path.join(CONTAINER_ANTIGEN_INPUT_DIR, "antigen.pdb")
CONTAINER_TEMPLATE_PDB = os.path.join(CONTAINER_TEMPLATE_INPUT_DIR, "template.pdb")

# PDBs baked into the image at build time. Keep in sync with datasets/.
BAKED_DATASETS_DIR = "/home/datasets"
BAKED_ANTIGENS = {"pdl1", "bhrf1", "il3", "il20"}

# Per-target hotspots (matches bonobo's TARGETS dict in
# eval_compiled_final_iptm.py). These bias AF interface attention toward
# experimentally-known epitope residues.
BAKED_HOTSPOTS = {
    "pdl1": "A113",
    "bhrf1": "A60,A61,A63,A71",
    "il3":  "A23,A25,A26,A31,A40,A104",
    "il20": "A58,A62,A101",
}


def resolve_antigen(antigen: str) -> Tuple[str, Optional[ProcessingInput]]:
    """Return (container_path_to_antigen_pdb, optional_processing_input)."""
    if antigen.startswith("s3://"):
        return CONTAINER_ANTIGEN_PDB, ProcessingInput(
            source=antigen,
            destination=CONTAINER_ANTIGEN_INPUT_DIR,
            input_name="antigen",
        )
    if antigen in BAKED_ANTIGENS:
        return os.path.join(BAKED_DATASETS_DIR, f"{antigen}.pdb"), None
    raise ValueError(
        f"--antigen must be an s3:// URI or one of {sorted(BAKED_ANTIGENS)}; got {antigen!r}"
    )


def resolve_template(
    antigen: str, template_s3_uri: Optional[str]
) -> Tuple[str, Optional[ProcessingInput]]:
    """Return (container_path_to_template_pdb, optional_processing_input).

    If --template-s3-uri is given, mount it. Otherwise, if the antigen is a
    baked name, use the baked template at datasets/template_<name>.pdb. Else
    error.
    """
    if template_s3_uri:
        return CONTAINER_TEMPLATE_PDB, ProcessingInput(
            source=template_s3_uri,
            destination=CONTAINER_TEMPLATE_INPUT_DIR,
            input_name="template",
        )
    if antigen in BAKED_ANTIGENS:
        return os.path.join(BAKED_DATASETS_DIR, f"template_{antigen}.pdb"), None
    raise ValueError(
        "--template-s3-uri is required when --antigen is a custom S3 PDB. "
        "(For baked targets it's auto-resolved to datasets/template_<name>.pdb.)"
    )


def resolve_hotspot(antigen: str, hotspot: Optional[str]) -> Optional[str]:
    if hotspot:
        return hotspot
    return BAKED_HOTSPOTS.get(antigen)


def build_command(
    *,
    antigen_container_path: str,
    template_container_path: str,
    hotspot: Optional[str],
    antibody_sequence: str,
    cdrs_to_design: str,
    metrics_name: str,
    metrics_list: str,
    repeatnum: int,
    duplicate: int,
    iteration: int,
    af_models: str,
    af_gpu_ids: str,
    seed: int,
    run_name: str,
) -> list[str]:
    cmd = f"""
        ab_refinement.py
        --antibody_sequence {antibody_sequence}
        --antigen_pdb {antigen_container_path}
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
        --template_pdb {template_container_path}
    """
    if hotspot:
        cmd += f" --hotspot {hotspot}"
    return [seg for line in cmd.splitlines() for seg in line.strip().split(" ") if seg]


def launch_one(
    *,
    antigen: str,
    antibody_sequence: str,
    template_s3_uri: Optional[str] = None,
    hotspot: Optional[str] = None,
    cdrs_to_design: str = "H1,H2,H3",
    metrics_name: str = "iptm,cdr_plddt,plddt",
    metrics_list: str = "3,1,1",
    repeatnum: int = 100,
    duplicate: int = 5,
    iteration: int = 10,
    af_models: str = "0",
    af_gpu_ids: str = "1,2,3",
    seed: int = 1776,
    run_name: str = "rerd_run",
    image_tag: str = "latest",
    timeout_hours: int = 24,
):
    antigen_path, antigen_input = resolve_antigen(antigen)
    template_path, template_input = resolve_template(antigen, template_s3_uri)
    resolved_hotspot = resolve_hotspot(antigen, hotspot)

    inputs = [x for x in (antigen_input, template_input) if x is not None]
    outputs = [
        ProcessingOutput(
            output_name="rerd_output",
            source=CONTAINER_OUTPUT_DIR,
            s3_upload_mode="Continuous",
        ),
    ]
    cmd = build_command(
        antigen_container_path=antigen_path,
        template_container_path=template_path,
        hotspot=resolved_hotspot,
        antibody_sequence=antibody_sequence,
        cdrs_to_design=cdrs_to_design,
        metrics_name=metrics_name,
        metrics_list=metrics_list,
        repeatnum=repeatnum,
        duplicate=duplicate,
        iteration=iteration,
        af_models=af_models,
        af_gpu_ids=af_gpu_ids,
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
    p.add_argument("--antigen", required=True,
                   help=f"Either an s3:// URI to a PDB, or a baked-in name "
                        f"({sorted(BAKED_ANTIGENS)}).")
    p.add_argument("--antibody-sequence", required=True,
                   help="Full heavy-chain antibody sequence to seed design from.")
    p.add_argument("--template-s3-uri", default=None,
                   help="s3:// URI to a pre-made target+binder template PDB. "
                        "Required when --antigen is a custom S3 PDB; ignored "
                        "for baked targets (auto-resolved).")
    p.add_argument("--hotspot", default=None,
                   help="Override the hotspot string. For baked targets this "
                        "defaults to the per-target value (e.g. pdl1 -> A113).")
    p.add_argument("--cdrs-to-design", default="H1,H2,H3")
    p.add_argument("--repeatnum", type=int, default=100)
    p.add_argument("--duplicate", type=int, default=5)
    p.add_argument("--iteration", type=int, default=10)
    p.add_argument("--seed", type=int, default=1776,
                   help="Seed passed to ab_refinement.py. Default 1776 matches "
                        "bonobo's eval_compiled_final_iptm.py for parity.")
    p.add_argument("--run-name", default="rerd_run")
    p.add_argument("--image-tag", default="latest")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    launch_one(
        antigen=args.antigen,
        antibody_sequence=args.antibody_sequence,
        template_s3_uri=args.template_s3_uri,
        hotspot=args.hotspot,
        cdrs_to_design=args.cdrs_to_design,
        repeatnum=args.repeatnum,
        duplicate=args.duplicate,
        iteration=args.iteration,
        seed=args.seed,
        run_name=args.run_name,
        image_tag=args.image_tag,
    )
