# VHH Design with ProDifEvo-Refinement (AF2.3M backend)

This repo runs reward-guided masked-diffusion CDR design for antibodies / VHHs
(nanobodies). Framework residues are held fixed; the requested CDR positions
are iteratively re-sampled and ranked by an AlphaFold2-multimer (AF2.3M) reward.

## What changed vs upstream

The folding/scoring backend was switched from ESMFold to AF2 (multimer v3, i.e.
"AF2.3M") via colabdesign. Implementation lives in `ab_af2_reward.py`; entry
point is still `ab_refinement.py`.

For each candidate sequence in a batch, AF2 is run sequentially (no batching).
When `iptm` is in `--metrics_name`, the antibody:antigen complex is predicted
with the AF2 binder protocol; otherwise the antibody monomer is hallucinated.

## Setup

### 1. Python env

The original `RERD` conda env from `README.md` was built on py3.9 for the
ESMFold backend. The AF2 backend pins `jax==0.5.2`, which requires py≥3.10, so
the env needs to be (re)created on py3.11. If `RERD` already exists at py3.9,
remove it first with `conda env remove -n RERD`.

```bash
conda create -n RERD python=3.11 -y
conda activate RERD
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

The torch install must come before `requirements.txt` so that `deepspeed`,
`fair-esm`, and `torch_geometric` resolve against a known torch version rather
than pulling whatever default the resolver picks. On a CUDA host, install the
CUDA-matched torch wheel from the pytorch index instead, e.g. for CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Then install the CUDA-enabled jax wheel after `requirements.txt` (this adds
the `cuda12` extras to the already-pinned `jax==0.5.2`):

```bash
pip install "jax[cuda12]==0.5.2"
```

ANARCI (used for CDR numbering when `--auto_number` is True, the default)
shells out to the `hmmscan` binary from the HMMER suite, which is not pulled
in by pip. Install it via bioconda:

```bash
conda install -c bioconda hmmer -y
```

PyRosetta is only needed for structural metrics (`tm`, `crmsd`, `hydrophobic`,
`match_ss`, `surface_expose`, `globularity`, `cdr_hydrophobicity`); pure
confidence-metric runs (`plddt`, `cdr_plddt`, `ptm`, `iptm`) don't import it.

PyRosetta is not on PyPI and requires a RosettaCommons license (free for
academic use; register at https://www.pyrosetta.org/downloads). Install via
`pyrosetta-installer`, which auto-detects platform and pulls the matching
wheel:

```bash
pip install pyrosetta-installer
python -c "import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()"
# verify:
python -c "import pyrosetta; pyrosetta.init(); print('ok')"
```

The first run prompts for your RosettaCommons username/password. Linux x86_64
is supported (Ubuntu, Amazon Linux 2, Amazon Linux 2023); arm64/Graviton has
no official wheel.

### 2. AF2 weights

The AF2 backend looks in `~/.mber/af_params` by default. The minimal path is a
direct download of just the AF2 params (~3.5GB):

```bash
mkdir -p ~/.mber/af_params
cd ~/.mber/af_params
wget https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar
tar -xf alphafold_params_2022-12-06.tar && rm alphafold_params_2022-12-06.tar
```

Alternatively, the vendored `mber-open/download_weights.sh` script pulls AF2
**plus** NanoBodyBuilder2 and ESM2 weights (the latter ~5GB). This repo only
uses the AF2 weights, so prefer the direct download above. If you do use the
script, pass `--skip-esm` to skip the ESM2 download:

```bash
bash mber-open/download_weights.sh ~/.mber --skip-esm
```

Override the params location at runtime with `--af_params_dir` or the
`AF_PARAMS_DIR` env var.

### 3. Verify

Required files in `$AF_PARAMS_DIR` after the download:

- `params_model_*.npz` (monomer)
- `params_model_*_multimer_v3.npz` (multimer; AF2.3M = v3)

## Running a VHH design

VHHs are single-domain heavy chains, so use `--chain_type heavy`.

### Binder design (with antigen, ipTM-driven)

`run_ab_binding.sh` is the canonical example:

```bash
export AF_PARAMS_DIR="$HOME/.mber/af_params"

CUDA_VISIBLE_DEVICES=0 python ab_refinement.py \
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
```

### Monomer-only design (no antigen, plddt-driven)

```bash
python ab_refinement.py \
    --antibody_sequence "EVQLVESGGG..." \
    --chain_type heavy \
    --cdrs_to_design H1,H2,H3 \
    --metrics_name plddt,cdr_plddt \
    --metrics_list 1,2 \
    --repeatnum 5 --duplicate 10 --iteration 20 \
    --af_params_dir "$AF_PARAMS_DIR"
```

## Argument reference (essentials)

### Antibody / VHH

| flag | meaning |
| --- | --- |
| `--antibody_sequence` | Full heavy- or light-chain AA sequence. For VHHs, the heavy domain. |
| `--chain_type` | `heavy` (use this for VHHs) or `light`. |
| `--cdrs_to_design` | Comma-list of CDR names to design, e.g. `H3` or `H1,H2,H3`, or `all`. |
| `--numbering_scheme` | `imgt` / `chothia` / `kabat`. Used by ANARCI to locate CDRs. |
| `--cdr_indices` | Optional manual override, e.g. `26-38,55-65,104-117` (0-based). |
| `--auto_number` | `True` to use ANARCI (default), `False` to require `--cdr_indices`. |
| `--edit_fraction` | Fraction of CDR positions re-masked per refinement step. |

### Antigen (binder mode)

| flag | meaning |
| --- | --- |
| `--antigen_pdb` | Antigen structure (PDB or CIF). Required if `iptm` is in `--metrics_name`. |
| `--antigen_chain` | Chain ID(s) on the antigen. Comma-separated for multiple. |

### AF2 backend

| flag | default | meaning |
| --- | --- | --- |
| `--af_params_dir` | `$AF_PARAMS_DIR` or `~/.mber/af_params` | Directory holding AF2 .npz files. |
| `--num_recycles` | `3` | AF2 recycling iterations per prediction. |
| `--af_models` | `0` | Comma-list of model indices to ensemble (`0,1,2,3,4` for full ensemble). |
| `--af_use_multimer` | `True` | Use multimer params (required for `iptm`). |
| `--af_no_multimer` | – | Disable multimer mode. |

### Reward / decoding

| flag | meaning |
| --- | --- |
| `--metrics_name` | Comma-list of reward metrics. See below. |
| `--metrics_list` | Comma-list of weights aligned with `--metrics_name`. |
| `--decoding` | `SVDD_edit` (refinement) or `SVDD` (single-shot). |
| `--repeatnum` | Batch size (parallel designs). |
| `--duplicate` | Candidates evaluated per step. |
| `--iteration` | Number of refinement iterations (only for `SVDD_edit`). |
| `--diffusion_model` | EvoDiff checkpoint: `38M` or `640M`. |
| `--ref_pdb` | Reference structure for `tm` / `crmsd` / `match_ss`. |
| `--run_name` | Subdirectory name under `log/<timestamp>_ab_.../`. |

### Supported metric names

AF2-derived: `iptm`, `ptm`, `plddt`, `cdr_plddt`
Sequence-only: `charge_balance`
PDB-based (need pyrosetta installed): `cdr_hydrophobicity`, `tm`, `crmsd`,
`hydrophobic`, `match_ss`, `surface_expose`, `globularity`

## Outputs

A run creates `log/<timestamp>_ab_<decoding>_<metrics>_<chain>_<cdrs>/<run_name>/`
containing:

- `*.pdb` — predicted complex (or monomer) for each final candidate
- `*.fasta` — corresponding sequence
- `output.csv` — per-candidate metric values, EvoDiff likelihood, set diversity
- `run.log` — INFO-level log

## Performance notes

- AF2 is ~10–30× slower than ESMFold per prediction. Cost ≈ `repeatnum *
  duplicate * iteration` predictions; tune those down before scaling up.
- Sequences in a batch are predicted one at a time. Most XLA compile cost is
  paid once per binder length — keep the antibody length constant within a run.
- `--af_models 0` (single model) is the default. Use the full `0,1,2,3,4`
  ensemble only for the final evaluation pass, not during refinement.
- ipTM only makes sense with `--af_use_multimer`. Don't pass `--af_no_multimer`
  if `iptm` is in `--metrics_name`.
