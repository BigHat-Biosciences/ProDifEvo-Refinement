# Handoff: NeurIPS baselines (RERD + VIDD) parity work

Pointers for picking up the antibody-design baseline benchmarking. Read this
first; then the per-repo `BH-README.md` files have the deep detail. If you're
reading this with Claude Code, point it at the three `BH-README.md`s and this
file to get oriented.

## TL;DR

We benchmark two reward-guided diffusion **baselines** against our own GFlowNet
method (**bonobo**) for **CDR-only VHH (nanobody) design against a fixed
antigen, scored by an AF2-multimer (multimer-v3) ipTM reward**. Four targets:
`pdl1`, `bhrf1`, `il3`, `il20`.

- **RERD** = the `ProDifEvo-Refinement` repo (reward-guided masked diffusion, EvoDiff backbone + SVDD refinement).
- **VIDD** = the `VIDD` repo (discrete-diffusion policy distillation; fork of arXiv 2507.00445).
- **bonobo** = our method; also provides the *reference evaluator* everything is measured against.

The whole late-stage effort is **parity / reproducibility**: making the two
baselines' ipTM numbers directly comparable to bonobo's evaluator so the paper's
table is apples-to-apples.

## Repos to clone

The parity scripts assume all three live in `$HOME` (as `~/ProDifEvo-Refinement`,
`~/bonobo`, `~/VIDD`) — override with `BONOBO_REPO=` / `VIDD_REPO=` env vars.

| Repo | Branch | Role |
|---|---|---|
| `git@github.com:BigHat-Biosciences/ProDifEvo-Refinement.git` | `containerize` | RERD baseline + **parity harness** (this repo) |
| `git@github.com:BigHat-Biosciences/VIDD.git` | `ab-mvp` | VIDD baseline |
| `git@github.com:BigHat-Biosciences/bonobo.git` | `marcus-tries-ibcnn` | our method + `eval_compiled_final_iptm.py` reference evaluator |
| `git@github.com:manifoldbio/mber-open.git` | `main` | AF2 reward backend (vendored into RERD; bonobo has its own copy) |

> Confirm `~/bonobo/eval_compiled_final_iptm.py` exists on whatever bonobo
> branch you check out — the parity script depends on it.

## The parity problem (what actually matters)

Three distinct issues, all in `ProDifEvo-Refinement`:

1. **Multi-GPU race → biased ipTM.** AF2 predictions dispatched across GPUs
   1/2/3 via a thread pool could let two threads overwrite each other's
   `model.aux`, biasing design-time ipTM upward. **Fixed** (shard one task per
   worker) in `ab_af2_reward.py:_reward_metrics_parallel`. Landed in commit
   `ca94f8f`.
2. **RERD ↔ bonobo conditioning parity.** Does RERD's evaluator condition AF2
   the same way bonobo's does (template PDB, hotspot, `rm_binder`, recycles,
   multimer params, seed)? Convergence work: `0061779 bonobo parity huge
   refactor` → `b6041d4 e2e parity` → `8b58a34 sync vendored mber-open` → seed
   fixes (`84df40e`, `5e894fe`).
3. **AF2 stochasticity floor.** Even at parity AF2 isn't bit-exact.
   `scripts/eval_noise_floor.sh` characterizes the floor so you know which gaps
   are real (rule of thumb: mean |Δ| ~0.003 tight, ~0.008 loose).

### The diagnostic toolkit (`scripts/`)

| Script | Purpose |
|---|---|
| `eval_parity.sh` | Score a CSV through **all three** evaluators (RERD `eval_iptm.py`, bonobo `eval_compiled_final_iptm.py`, VIDD `scripts/eval_iptm.py`) → `comparison.csv` + 4 delta tests. No design loop. |
| `eval_parity_all.sh` | `fetch_outputs.sh` from S3 + loop `eval_parity.sh` over all 4 targets. |
| `eval_e2e_parity.sh` | Small RERD design → re-eval w/ RERD → re-eval w/ bonobo (three-way, end-to-end). |
| `diag_singlegpu_bias.sh` / `diag_multigpu_bias.sh` | Prove the race is gone: single-GPU Δ≈0, multi-GPU mean≈0 / max<0.01. |
| `eval_noise_floor.sh` | Score same seqs twice → AF2 noise floor. |

`eval_parity.sh` Step 4 emits four delta tests: **T1** design-ipTM vs RERD
re-eval (race check, ~0 = no race), **T2** RERD vs bonobo (conditioning
parity), **T3** RERD vs VIDD (reward-backend parity), **T4** VIDD vs bonobo
(end-to-end). "At parity" = all four means ≈ 0 within the noise floor.

## Current status (as of the last work session)

- **RERD design runs are in progress** still running. 
- **VIDD runs** were done locally on the GPU box for all 4 targets.
- **The parity comparison itself was mid-run and never captured.** The
  consolidated 4-target × 3-method table (the actual deliverable) does not exist
  in either repo — `eval_parity.sh` writes it to `/tmp/rerd_parity/` on the box.
- **Next step:** run `eval_parity_all.sh` on a 4-GPU box and capture the four
  `comparison.csv` + Step-4 summaries. That table is what's needed for the
  writeup / rebuttal.

### Data you need (not in the repos)

The 4 RERD design-output CSVs (`rerd_{pdl1,bhrf1,il3,il20}.csv`, `sequence` +
`iptm` columns, 100 rows) are the **inputs** to the parity eval. They live in
S3 (`s3://sagemaker-us-east-1-332120041740/…`, pulled by `fetch_outputs.sh`)
and locally on Nilay's machine. **Nilay will send them separately** — the S3
objects are from May 7 and may have aged out. Drop them at
`~/Downloads/rerd_<target>.csv` and `eval_parity_all.sh` will pick them up (or
point `eval_parity.sh --input-csv` at them directly to skip the S3 fetch).

## Setup (short version — full detail in each `BH-README.md`)

Both baseline repos share the same AF2 backend requirements:

- **Python 3.11**, `jax==0.5.2` (+ `jax[cuda12]==0.5.2` on GPU). py3.9 is stale.
- **Three conda envs**: `RERD`, `vidd`, `bonobo` (the parity script activates each).
- **AF2 weights** in `~/.mber/af_params` — needs `params_model_*_multimer_v3.npz` (ipTM requires multimer).
- **HMMER** (`conda install -c bioconda hmmer`) for ANARCI CDR numbering.
- **PyRosetta** only for structural metrics — *not* needed for ipTM-only parity runs.
- 4-GPU box (e.g. g5.12xlarge): GPU 0 = torch, GPUs 1/2/3 = AF2 workers via `--af_gpu_ids 1,2,3`.

RERD env + weights: `ProDifEvo-Refinement/BH-README.md`. VIDD: `VIDD/BH-README.md`
(or just `bash install.sh`).

## Run the parity check

```bash
# all three repos in ~, all three conda envs built, AF2 weights in place,
# the 4 rerd_*.csv in ~/Downloads
cd ~/ProDifEvo-Refinement && git pull
bash scripts/eval_parity_all.sh 2>&1 | tee /tmp/parity.log
# results: /tmp/rerd_parity/<target>/comparison.csv  + Step-4 summary per target
```

Single target / skip S3:
```bash
bash scripts/eval_parity.sh --input-csv ~/Downloads/rerd_pdl1.csv --antigen pdl1
```

## Gotchas (learned the hard way)

- **`tail -n 5`, not `tail -5`** (the short form errors in this shell).
- **Don't paste long multi-line commands** into the terminal — it garbles them.
  Put anything multi-line in a `.sh` file and `bash` it. (That's why the wrapper
  scripts exist.)
- AF2 is 10–30× slower than ESMFold per prediction. Cost ≈
  `repeatnum × duplicate × iteration`. Keep antibody length fixed within a run
  (colabdesign recompiles on length change).
- ipTM needs `--af_use_multimer`. `KeyError: 'i_ptm'` = env built without multimer.
- If a design run's ipTM looks biased vs a fresh re-eval, suspect a **stale
  SageMaker container image** that predates the race fix — Test 1 catches this.

## Fuller writeup

`NEURIPS_BASELINES_RESUME.md` (reconstructed project history, session ledger,
and the parity narrative) can be shared alongside this if you want more context.
