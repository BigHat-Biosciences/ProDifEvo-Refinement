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

from bh.aws_tools.sagemaker import (
    SAGEMAKER_GPU_MEDIUM_INSTANCE_TYPE,
    launch_container_on_sagemaker,
)


# /opt/ml/processing/* layout follows SageMaker's default convention.
CONTAINER_INPUT_DIR = "/opt/ml/processing/input"
CONTAINER_ANTIGEN_INPUT_DIR = os.path.join(CONTAINER_INPUT_DIR, "antigen")
CONTAINER_TEMPLATE_INPUT_DIR = os.path.join(CONTAINER_INPUT_DIR, "template")
CONTAINER_OUTPUT_DIR = "/opt/ml/processing/output"
CONTAINER_ANTIGEN_PDB = os.path.join(CONTAINER_ANTIGEN_INPUT_DIR, "antigen.pdb")
CONTAINER_TEMPLATE_PDB = os.path.join(CONTAINER_TEMPLATE_INPUT_DIR, "template.pdb")

CONTAINER_CODE_INPUT_DIR = os.path.join(CONTAINER_INPUT_DIR, "code")

# Python sources that may be overlaid onto the baked image at launch time.
# The image COPYs the repo to /home at build time, so without an overlay a code
# change needs a full CUDA image rebuild (~an hour, and cross-arch from an arm64
# laptop). Overlaying mirrors how the other baselines ship their drivers: edit,
# relaunch, no rebuild. Keep this list to interpreted sources -- anything
# compiled or pip-installed still needs a real rebuild.
OVERLAY_SOURCES = [
    "ab_refinement.py",
    "ab_args_file.py",
    "ab_af2_reward.py",
    "ab_utils.py",
    "evodiff/generate_antibody.py",
]

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


def build_overlay_input(repo_root: str, s3_prefix: str) -> ProcessingInput:
    """Upload the current working copy of OVERLAY_SOURCES and mount it.

    Uploads what is on disk, not what is committed, so an uncommitted fix still
    reaches the job -- deliberate, since this exists to shorten the edit/launch
    loop. The launcher prints the manifest so a run is always traceable to the
    exact bytes it ran.
    """
    import hashlib
    import subprocess
    import tempfile

    staged = tempfile.mkdtemp(prefix="rerd_overlay_")
    manifest = []
    for rel in OVERLAY_SOURCES:
        src = os.path.join(repo_root, rel)
        if not os.path.exists(src):
            raise FileNotFoundError(f"overlay source missing: {src}")
        dst = os.path.join(staged, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(src, "rb") as fh:
            body = fh.read()
        with open(dst, "wb") as fh:
            fh.write(body)
        manifest.append(f"{hashlib.sha256(body).hexdigest()[:12]}  {rel}")

    with open(os.path.join(staged, "OVERLAY_MANIFEST.txt"), "w") as fh:
        fh.write("\n".join(manifest) + "\n")

    # Copy the overlay over the baked tree, then exec the real command. This has
    # to be a file rather than `bash -c "..."` because SageMaker caps each
    # ContainerArguments member at 256 chars and the whole script would be one
    # member. Args after the script path arrive as "$@", each comfortably short.
    with open(os.path.join(staged, "run_rerd.sh"), "w") as fh:
        fh.write(f"""#!/bin/bash
set -euo pipefail
CODE={CONTAINER_CODE_INPUT_DIR}
if [ -f "$CODE/OVERLAY_MANIFEST.txt" ]; then
  echo "[overlay] applying code overlay:"
  cat "$CODE/OVERLAY_MANIFEST.txt"
  cd "$CODE"
  find . -name '*.py' -print0 | while IFS= read -r -d '' f; do
    mkdir -p "/home/$(dirname "$f")"
    cp "$f" "/home/$f"
    echo "[overlay]   -> /home/${{f#./}}"
  done
  cd /
else
  echo "[overlay] no manifest found; running the image as baked"
fi
cd /home
echo "[overlay] exec: $*"
exec "$@"
""")

    subprocess.run(["aws", "s3", "cp", "--recursive", "--quiet", staged, s3_prefix],
                   check=True)
    print("code overlay   : " + s3_prefix)
    for line in manifest:
        print("                 " + line)
    return ProcessingInput(source=s3_prefix, destination=CONTAINER_CODE_INPUT_DIR,
                           input_name="code")


def build_command(
    *,
    antigen_container_path: str,
    template_container_path: str,
    hotspot: Optional[str],
    antibody_sequence: str,
    cdrs_to_design: str,
    cdr_indices: Optional[str],
    metrics_name: str,
    metrics_list: str,
    repeatnum: int,
    duplicate: int,
    iteration: int,
    af_models: str,
    af_gpu_ids: str,
    seed: int,
    run_name: str,
    wallclock_seconds: int = 0,
    final_eval_reserve_seconds: int = 1800,
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
    if cdr_indices:
        cmd += f" --cdr_indices {cdr_indices}"
    if wallclock_seconds > 0:
        cmd += (f" --wallclock_seconds {wallclock_seconds}"
                f" --final_eval_reserve_seconds {final_eval_reserve_seconds}")
    return [seg for line in cmd.splitlines() for seg in line.strip().split(" ") if seg]


def launch_one(
    *,
    antigen: str,
    antibody_sequence: str,
    template_s3_uri: Optional[str] = None,
    hotspot: Optional[str] = None,
    cdrs_to_design: str = "H1,H2,H3",
    cdr_indices: Optional[str] = None,
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
    wallclock_seconds: int = 0,
    final_eval_reserve_seconds: int = 1800,
    overlay_s3_prefix: Optional[str] = None,
):
    antigen_path, antigen_input = resolve_antigen(antigen)
    template_path, template_input = resolve_template(antigen, template_s3_uri)
    resolved_hotspot = resolve_hotspot(antigen, hotspot)

    inputs = [x for x in (antigen_input, template_input) if x is not None]
    overlay_input = None
    if overlay_s3_prefix:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        overlay_input = build_overlay_input(repo_root, overlay_s3_prefix.rstrip("/"))
        inputs.append(overlay_input)
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
        cdr_indices=cdr_indices,
        metrics_name=metrics_name,
        metrics_list=metrics_list,
        repeatnum=repeatnum,
        duplicate=duplicate,
        iteration=iteration,
        af_models=af_models,
        af_gpu_ids=af_gpu_ids,
        seed=seed,
        run_name=run_name,
        wallclock_seconds=wallclock_seconds,
        final_eval_reserve_seconds=final_eval_reserve_seconds,
    )
    if wallclock_seconds > 0:
        # Belt and braces (see below). ab_refinement stops itself at the budget, but if a
        # single iteration overruns its prediction the container must still be
        # brought down cleanly: `timeout` sends SIGTERM at the budget (the driver
        # catches it and winds up at the next boundary) and SIGKILL 120s later.
        # SageMaker's own timeout sits an hour further out so it is never what
        # ends a healthy run -- a SageMaker kill is a hard stop with no chance to
        # flush artifacts.
        command = "timeout"
        arguments = ["-s", "TERM", "-k", "120s", f"{wallclock_seconds}s", "python"] + cmd
        timeout_in_seconds = wallclock_seconds + 3600
    else:
        command = "python"
        arguments = cmd
        timeout_in_seconds = timeout_hours * 60 * 60

    if overlay_input is not None:
        # Run everything through the overlay wrapper, which applies the code
        # then execs what follows (including the `timeout` prefix, so the
        # wall-clock fence still wraps the real work).
        arguments = [f"{CONTAINER_CODE_INPUT_DIR}/run_rerd.sh", command] + arguments
        command = "bash"

    return launch_container_on_sagemaker(
        image_repo="rerd-antibody",
        image_tag=image_tag,
        command=command,
        arguments=arguments,
        inputs=inputs,
        outputs=outputs,
        instance_type=SAGEMAKER_GPU_MEDIUM_INSTANCE_TYPE,
        timeout_in_seconds=timeout_in_seconds,
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
    p.add_argument("--cdr-indices", default=None,
                   help="Manual 0-based CDR indices, e.g. '26-34,47-57,95-106'. "
                        "Overrides ANARCI auto-numbering; freezes the framework to "
                        "exactly the complement of these positions.")
    p.add_argument("--repeatnum", type=int, default=100)
    p.add_argument("--duplicate", type=int, default=5)
    p.add_argument("--iteration", type=int, default=10)
    p.add_argument("--seed", type=int, default=1776,
                   help="Seed passed to ab_refinement.py. Default 1776 matches "
                        "bonobo's eval_compiled_final_iptm.py for parity.")
    p.add_argument("--run-name", default="rerd_run")
    p.add_argument("--image-tag", default="latest")
    p.add_argument("--wallclock-hours", type=float, default=0.0,
                   help="Wall-clock budget in hours. When set, the run is ended by "
                        "time rather than by --iteration (which becomes an upper "
                        "bound), matching the other baselines. 0 = disabled.")
    p.add_argument("--overlay-code-s3", default=None,
                   help="s3:// prefix to stage the current working copy of the "
                        "python sources to, and mount into the job. Lets a code "
                        "change reach the next run without rebuilding the CUDA "
                        "image. Omit to run the image exactly as baked.")
    p.add_argument("--final-eval-reserve-minutes", type=float, default=30.0,
                   help="Minutes held back from the wall-clock budget for the final "
                        "eval pass and artifact writing.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    launch_one(
        antigen=args.antigen,
        antibody_sequence=args.antibody_sequence,
        template_s3_uri=args.template_s3_uri,
        hotspot=args.hotspot,
        cdrs_to_design=args.cdrs_to_design,
        cdr_indices=args.cdr_indices,
        repeatnum=args.repeatnum,
        duplicate=args.duplicate,
        iteration=args.iteration,
        seed=args.seed,
        run_name=args.run_name,
        image_tag=args.image_tag,
        wallclock_seconds=int(args.wallclock_hours * 3600),
        final_eval_reserve_seconds=int(args.final_eval_reserve_minutes * 60),
        overlay_s3_prefix=args.overlay_code_s3,
    )
