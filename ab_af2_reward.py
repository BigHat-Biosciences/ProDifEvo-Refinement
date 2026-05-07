"""AlphaFold2.3-multimer reward backend for antibody CDR design.

Bonobo-style: the binder template is supplied as a pre-made multi-chain PDB
(target chain + binder chain on disk). No NBB2 binder pre-folding at runtime.
Build templates offline with ``scripts/generate_template.py`` (the only place
NBB2 is invoked).

Two underlying models are constructed lazily:

* ``mk_afdesign_model(protocol="binder", use_multimer=True)`` for predicting
  the antibody:antigen complex (used when ``iptm`` is requested). Requires a
  ``template_pdb`` path containing target + binder.
* ``mk_afdesign_model(protocol="hallucination", use_multimer=False)`` for
  predicting the antibody monomer (used when no antigen is involved).

AF2 is not batchable — each candidate sequence is predicted sequentially per
worker. Multi-GPU is supported via ``af_gpu_ids``: one ``_AFWorker`` per GPU,
each holding its own colabdesign model. Sequences are sharded one task per
worker (each worker processes its shard sequentially) — this avoids the
race condition where two threads landing on the same worker could overwrite
each other's ``model.aux``.

AF2 weights are expected to live in ``~/.mber/af_params`` by default; this can
be overridden with the ``AF_PARAMS_DIR`` env var or the ``af_params_dir`` ctor
argument.
"""

from __future__ import annotations

import os
import logging
import time
from io import StringIO
from typing import List, Optional, Sequence

import numpy as np

# colabdesign is jax-based; importing it eagerly would be costly and would
# fail on environments where AF2 is not yet installed. Defer to first use.
_AF_FACTORY = None
_CLEAR_MEM = None


def _lazy_import_colabdesign():
    """Import colabdesign on first use, surfacing a clear install hint if missing."""
    global _AF_FACTORY, _CLEAR_MEM
    if _AF_FACTORY is not None:
        return _AF_FACTORY, _CLEAR_MEM
    try:
        from colabdesign import mk_afdesign_model, clear_mem
    except ImportError as e:
        raise ImportError(
            "colabdesign is required for the AF2 reward backend. Install with:\n"
            "  pip install 'colabdesign @ git+https://github.com/sokrypton/ColabDesign.git@d024c4e'\n"
            "and ensure jax/flax are installed (see requirements.txt)."
        ) from e
    _AF_FACTORY = mk_afdesign_model
    _CLEAR_MEM = clear_mem
    return _AF_FACTORY, _CLEAR_MEM


ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"

# Bonobo-style hardcoded CDR mask: covers Chothia-style CDR-H1 (H27-H35),
# CDR-H2 (H48-H58), CDR-H3 (H96-H107) — 32 positions on the binder chain. Used
# identically for rm_binder, rm_binder_seq, rm_binder_sc so AF re-hallucinates
# CDR backbone, sequence identity, and sidechains while keeping the framework
# templated. Override via ``rm_binder_positions`` ctor arg if your antibody
# scaffold uses different boundaries.
DEFAULT_RM_BINDER_POSITIONS = ",".join(
    [f"H{i}" for i in range(27, 36)]   # CDR-H1: H27..H35
    + [f"H{i}" for i in range(48, 59)] # CDR-H2: H48..H58
    + [f"H{i}" for i in range(96, 108)]# CDR-H3: H96..H107
)


# ============================================================
# Metric extraction helpers (operate on colabdesign ``aux`` dict)
# ============================================================

def _get_log(aux: dict) -> dict:
    """Return the per-prediction ``log`` dict from colabdesign aux."""
    return aux.get("log", {})


def _per_residue_plddt(aux: dict) -> np.ndarray:
    """Return per-residue pLDDT as a 1D float array on [0, 100]."""
    plddt = aux.get("plddt")
    if plddt is None:
        raise KeyError("AF2 aux missing 'plddt'")
    arr = np.asarray(plddt)
    return arr.reshape(-1) * (100.0 if arr.max() <= 1.0 else 1.0)


def af2_to_ptm(aux: dict) -> float:
    """Extract pTM from AF2 prediction (scaled to [0, 1])."""
    return float(_get_log(aux).get("ptm", 0.0))


def af2_to_iptm(aux: dict) -> float:
    """Extract interface pTM (ipTM) from AF2 multimer prediction."""
    log = _get_log(aux)
    if "i_ptm" in log:
        return float(log["i_ptm"])
    if "iptm" in log:
        return float(log["iptm"])
    raise KeyError(
        "AF2 aux['log'] does not contain i_ptm. Ensure use_multimer=True and that "
        "the binder protocol was used for complex prediction."
    )


def af2_to_plddt(aux: dict, binder_offset: int = 0, binder_len: Optional[int] = None) -> float:
    """Mean pLDDT over the binder slice, scaled to [0, 1].

    Args:
        aux: colabdesign aux dict.
        binder_offset: index where the binder starts (0 for monomer; target_len for complex).
        binder_len: number of binder residues. If None, take everything from offset onward.
    """
    pl = _per_residue_plddt(aux)
    sl = pl[binder_offset:] if binder_len is None else pl[binder_offset:binder_offset + binder_len]
    if sl.size == 0:
        return 0.0
    return float(sl.mean()) / 100.0


def af2_to_cdr_plddt(
    aux: dict,
    cdr_indices: Sequence[int],
    binder_offset: int = 0,
) -> float:
    """Mean pLDDT over CDR residue positions in the binder, scaled to [0, 1]."""
    pl = _per_residue_plddt(aux)
    if len(cdr_indices) == 0:
        return 0.0
    cdr_global = np.asarray(cdr_indices, dtype=int) + binder_offset
    cdr_global = cdr_global[cdr_global < pl.size]
    if cdr_global.size == 0:
        return 0.0
    return float(pl[cdr_global].mean()) / 100.0


# ============================================================
# Per-GPU AF worker for multi-GPU parallelism
# ============================================================

class _AFWorker:
    """A single AF2 model instance pinned to one JAX device.

    Each worker holds its own complex/monomer model handles. All JAX ops are
    executed inside ``jax.default_device(jax_device)`` so that compilation and
    forward passes are routed to that device.

    Concurrency note: workers are NOT internally thread-safe — each worker
    mutates ``model.aux`` on every predict call. The dispatcher in
    ``AbAF2RewardCal._reward_metrics_parallel`` shards sequences such that
    each worker is owned by exactly one ThreadPoolExecutor task at a time.
    """

    def __init__(
        self,
        jax_device,
        af_params_dir: str,
        num_recycles: int,
        use_multimer: bool,
        af_models: Sequence[int],
        antigen_pdb: Optional[str],
        antigen_chain: str,
        binder_chain: str = "H",
        hotspot: Optional[str] = None,
    ):
        self.jax_device = jax_device
        self.af_params_dir = af_params_dir
        self.num_recycles = num_recycles
        self.use_multimer = use_multimer
        self.af_models = list(af_models)
        self.antigen_pdb = antigen_pdb
        self.antigen_chain = antigen_chain
        self.binder_chain = binder_chain
        self.hotspot = hotspot

        self._complex_model = None
        self._complex_target_len = 0
        self._complex_binder_len = 0
        self._template_initialized = False
        self._monomer_model = None
        self._monomer_len = 0

    def _ensure_complex_model(self, ab_len: int) -> None:
        """Build the non-templated binder model (full hallucination of binder)."""
        mk_model, clear_mem = _lazy_import_colabdesign()
        if self._complex_model is None or self._complex_binder_len != ab_len:
            if self._complex_model is not None:
                clear_mem()
            self._complex_model = mk_model(
                protocol="binder",
                debug=False,
                data_dir=self.af_params_dir,
                use_multimer=self.use_multimer,
                num_recycles=self.num_recycles,
            )
            self._complex_model._prep_binder(
                pdb_filename=self.antigen_pdb,
                chain=self.antigen_chain,
                binder_len=ab_len,
                hotspot=self.hotspot,
                seed=0,
                rm_target=False,
                rm_target_seq=False,
                rm_target_sc=False,
                rm_template_ic=True,
                rm_binder=True,
                rm_binder_seq=True,
                rm_binder_sc=True,
            )
            self._complex_binder_len = ab_len
            self._complex_target_len = int(self._complex_model._target_len)

    def init_template(self, template_pdb_path: str, rm_binder_str: str) -> None:
        """One-shot template init: build a binder model with the combined target+binder
        PDB, masking the binder template at the bonobo-style CDR positions.

        ``rm_binder``, ``rm_binder_seq``, ``rm_binder_sc`` are all set to the same
        position string (per bonobo). This causes AF to re-hallucinate the CDR
        backbone, sequence identity, and sidechains while keeping the framework
        templated.
        """
        import jax
        mk_model, clear_mem = _lazy_import_colabdesign()
        with jax.default_device(self.jax_device):
            if self._complex_model is not None:
                clear_mem()
            self._complex_model = mk_model(
                protocol="binder",
                debug=False,
                data_dir=self.af_params_dir,
                use_multimer=self.use_multimer,
                num_recycles=self.num_recycles,
            )
            self._complex_model._prep_binder(
                pdb_filename=template_pdb_path,
                chain=self.antigen_chain,
                binder_chain=self.binder_chain,
                hotspot=self.hotspot,
                seed=0,
                rm_target=False,
                rm_target_seq=False,
                rm_target_sc=False,
                rm_template_ic=True,
                rm_binder=rm_binder_str,
                rm_binder_seq=rm_binder_str,
                rm_binder_sc=rm_binder_str,
            )
            self._complex_target_len = int(self._complex_model._target_len)
            self._complex_binder_len = int(self._complex_model._binder_len)
            self._template_initialized = True

    def _ensure_monomer_model(self, ab_len: int) -> None:
        mk_model, clear_mem = _lazy_import_colabdesign()
        if self._monomer_model is None or self._monomer_len != ab_len:
            if self._monomer_model is not None:
                clear_mem()
            self._monomer_model = mk_model(
                protocol="hallucination",
                use_templates=False,
                num_recycles=self.num_recycles,
                data_dir=self.af_params_dir,
                use_multimer=False,
            )
            self._monomer_model._prep_hallucination(length=ab_len)
            self._monomer_len = ab_len

    def predict_complex(self, ab_seq: str) -> tuple:
        """Run a complex prediction. Returns (aux, target_len).

        If ``init_template`` was called, the model is already templated — this
        just runs ``predict``. Otherwise the non-templated path builds a fresh
        full-hallucination model on first call.
        """
        import jax
        with jax.default_device(self.jax_device):
            if not self._template_initialized:
                self._ensure_complex_model(len(ab_seq))
            self._complex_model.predict(seq=ab_seq, models=self.af_models, verbose=False)
            aux = dict(self._complex_model.aux)
            aux["log"] = dict(self._complex_model.aux.get("log", {}))
            aux["_pdb_str"] = self._complex_model.save_pdb()
            return aux, self._complex_target_len

    def predict_monomer(self, ab_seq: str) -> dict:
        import jax
        with jax.default_device(self.jax_device):
            self._ensure_monomer_model(len(ab_seq))
            self._monomer_model.set_seq(ab_seq)
            self._monomer_model.predict(models=self.af_models, verbose=False)
            aux = dict(self._monomer_model.aux)
            aux["log"] = dict(self._monomer_model.aux.get("log", {}))
            aux["_pdb_str"] = self._monomer_model.save_pdb()
            return aux


# ============================================================
# AF2 reward calculator
# ============================================================

class AbAF2RewardCal:
    """AF2-multimer reward calculator for antibody CDR design.

    Mirrors the public surface of ``ab_refinement.AbRewardCal`` so it can be
    swapped in without changes to the diffusion driver.
    """

    def __init__(
        self,
        metrics_name: str,
        metrics_list: str,
        device,  # accepted for API parity; AF2 uses JAX/XLA, GPU selection via env
        cdr_indices: Optional[Sequence[int]] = None,
        run_name: str = "",
        pdb_save_path: str = "ab_sc_tmp",
        antigen_pdb: Optional[str] = None,
        antigen_chain: Optional[str] = None,
        antigen_seq: Optional[str] = None,  # accepted for API parity; not used
        af_params_dir: Optional[str] = None,
        num_recycles: int = 3,
        af_models: Sequence[int] = (0,),
        use_multimer: bool = True,
        template_pdb: Optional[str] = None,
        hotspot: Optional[str] = None,
        rm_binder_positions: Optional[str] = None,
        af_gpu_ids: Optional[Sequence[int]] = None,
        # ignored kwargs for backward compat with old call sites
        esm_model: Optional[str] = None,
    ):
        self.metrics_name = metrics_name.split(",")
        weights = metrics_list.split(",")
        self.metrics_list = [float(x) for x in weights]
        if len(self.metrics_name) != len(self.metrics_list):
            raise ValueError(
                f"Mismatch: {len(self.metrics_name)} metric names vs {len(self.metrics_list)} weights"
            )

        self.cdr_indices = list(cdr_indices) if cdr_indices is not None else []
        self.pdb_save_path = pdb_save_path
        self.run_name = run_name

        self.antigen_pdb = antigen_pdb
        self.antigen_chain = antigen_chain or "A"
        self.antigen_seq = antigen_seq  # informational only

        self.needs_complex = "iptm" in self.metrics_name
        if self.needs_complex and self.antigen_pdb is None:
            raise ValueError(
                "Metric 'iptm' requires an antigen structure. "
                "Provide --antigen_pdb pointing at the antigen PDB."
            )

        self.af_params_dir = os.path.expanduser(
            af_params_dir
            or os.environ.get("AF_PARAMS_DIR", "~/.mber/af_params")
        )
        self.num_recycles = num_recycles
        self.af_models = list(af_models)
        self.use_multimer = use_multimer

        # Template-based binder prediction (bonobo-style). When template_pdb is
        # set, the AF binder model is built once with that PDB as the template;
        # per-candidate predicts are pure forward passes. Required for binder
        # runs (iptm metric). The template PDB must contain the target on
        # ``antigen_chain`` and the binder on chain "H" (or override below).
        self.template_pdb: Optional[str] = template_pdb
        if self.needs_complex and self.template_pdb is None:
            raise ValueError(
                "Binder runs (iptm metric) require --template_pdb pointing at a "
                "pre-made multi-chain PDB containing target + binder. Generate "
                "one with scripts/generate_template.py."
            )
        if self.template_pdb is not None and not os.path.exists(self.template_pdb):
            raise FileNotFoundError(f"--template_pdb not found: {self.template_pdb}")

        self.hotspot: Optional[str] = hotspot
        self.rm_binder_positions: str = rm_binder_positions or DEFAULT_RM_BINDER_POSITIONS
        self._binder_chain = "H"

        # Multi-GPU AF parallelism. If af_gpu_ids has 2+ device IDs, AF
        # predictions are dispatched in parallel across one _AFWorker per GPU.
        # If 0 or 1 device IDs, the single-instance serial path is used.
        self.af_gpu_ids = list(af_gpu_ids) if af_gpu_ids else []
        self._workers: List["_AFWorker"] = []  # lazy-built on first use

        # Cumulative timing/counts. Updated by reward_metrics on every call.
        self._timings = {
            "n_sequences": 0,
            "reward_seconds": 0.0,
            "n_calls": 0,
        }

        # Single-instance handles for serial mode (only built when not using
        # multi-GPU dispatch). Templated complex model is built once on first call.
        self._complex_model = None
        self._complex_target_len = 0
        self._complex_binder_len = 0
        self._serial_template_initialized = False
        self._monomer_model = None
        self._monomer_len = 0

    # --------------------------------------------------------
    # Worker pool (multi-GPU)
    # --------------------------------------------------------
    def _ensure_workers(self) -> None:
        """Lazy-build a pool of ``_AFWorker`` instances pinned to ``af_gpu_ids``.

        When ``template_pdb`` is set, also runs the one-shot template init on
        each worker so per-candidate predicts are pure forward passes.
        """
        if self._workers or not self.af_gpu_ids:
            return
        import jax
        all_devices = jax.devices()
        try:
            picked = [all_devices[i] for i in self.af_gpu_ids]
        except IndexError as e:
            raise ValueError(
                f"--af_gpu_ids {self.af_gpu_ids} out of range; "
                f"jax.devices() has {len(all_devices)} devices."
            ) from e
        logging.info(
            f"[AF2] Building {len(picked)} AF workers on devices "
            f"{[str(d) for d in picked]}"
        )
        self._workers = [
            _AFWorker(
                jax_device=dev,
                af_params_dir=self.af_params_dir,
                num_recycles=self.num_recycles,
                use_multimer=self.use_multimer,
                af_models=self.af_models,
                antigen_pdb=self.antigen_pdb,
                antigen_chain=self.antigen_chain,
                binder_chain=self._binder_chain,
                hotspot=self.hotspot,
            )
            for dev in picked
        ]
        if self.template_pdb is not None:
            logging.info(
                f"[AF2] Templating {len(self._workers)} workers from "
                f"{self.template_pdb} (rm_binder={self.rm_binder_positions}, "
                f"hotspot={self.hotspot})"
            )
            for w in self._workers:
                w.init_template(self.template_pdb, self.rm_binder_positions)

    # --------------------------------------------------------
    # Serial-mode prediction (single-GPU / no af_gpu_ids)
    # --------------------------------------------------------
    def _ensure_serial_template(self) -> None:
        """One-shot templated binder-model build for the serial path."""
        if self._serial_template_initialized:
            return
        mk_model, clear_mem = _lazy_import_colabdesign()
        if self._complex_model is not None:
            clear_mem()
        logging.info(
            f"[AF2] Serial template init from {self.template_pdb} "
            f"(rm_binder={self.rm_binder_positions}, hotspot={self.hotspot})"
        )
        self._complex_model = mk_model(
            protocol="binder",
            debug=False,
            data_dir=self.af_params_dir,
            use_multimer=self.use_multimer,
            num_recycles=self.num_recycles,
        )
        self._complex_model._prep_binder(
            pdb_filename=self.template_pdb,
            chain=self.antigen_chain,
            binder_chain=self._binder_chain,
            hotspot=self.hotspot,
            seed=0,
            rm_target=False,
            rm_target_seq=False,
            rm_target_sc=False,
            rm_template_ic=True,
            rm_binder=self.rm_binder_positions,
            rm_binder_seq=self.rm_binder_positions,
            rm_binder_sc=self.rm_binder_positions,
        )
        self._complex_target_len = int(self._complex_model._target_len)
        self._complex_binder_len = int(self._complex_model._binder_len)
        self._serial_template_initialized = True

    def _ensure_monomer_model(self, ab_len: int) -> None:
        mk_model, clear_mem = _lazy_import_colabdesign()
        if self._monomer_model is None or self._monomer_len != ab_len:
            if self._monomer_model is not None:
                clear_mem()
            logging.info(
                f"[AF2] Building hallucination/monomer model "
                f"(recycles={self.num_recycles}, ab_len={ab_len})"
            )
            self._monomer_model = mk_model(
                protocol="hallucination",
                use_templates=False,
                num_recycles=self.num_recycles,
                data_dir=self.af_params_dir,
                use_multimer=False,
            )
            self._monomer_model._prep_hallucination(length=ab_len)
            self._monomer_len = ab_len

    def _predict_complex(self, ab_seq: str) -> dict:
        self._ensure_serial_template()
        self._complex_model.predict(seq=ab_seq, models=self.af_models, verbose=False)
        aux = dict(self._complex_model.aux)
        aux["log"] = dict(self._complex_model.aux.get("log", {}))
        aux["_pdb_str"] = self._complex_model.save_pdb()
        return aux

    def _predict_monomer(self, ab_seq: str) -> dict:
        self._ensure_monomer_model(len(ab_seq))
        self._monomer_model.set_seq(ab_seq)
        self._monomer_model.predict(models=self.af_models, verbose=False)
        aux = dict(self._monomer_model.aux)
        aux["log"] = dict(self._monomer_model.aux.get("log", {}))
        aux["_pdb_str"] = self._monomer_model.save_pdb()
        return aux

    # --------------------------------------------------------
    # Metric calculation
    # --------------------------------------------------------
    def metrics_cal(
        self,
        metrics_name: Sequence[str],
        complex_aux: Optional[dict] = None,
        monomer_aux: Optional[dict] = None,
        ori_pdb_file: Optional[str] = None,
        gen_pdb_file: Optional[str] = None,
        save_pdb: bool = False,
        sequence_str: Optional[str] = None,
        binder_offset: int = 0,
    ) -> List[float]:
        """Compute each requested metric.

        ``binder_offset`` is the residue index where the binder starts inside
        ``complex_aux`` (== target_len when complex is in use, 0 for monomer-only).
        """
        conf_aux = complex_aux if complex_aux is not None else monomer_aux
        ab_len = len(sequence_str) if sequence_str else None

        def _pdb_input():
            return gen_pdb_file if save_pdb else StringIO(gen_pdb_file)

        results: List[float] = []
        for metric in metrics_name:
            if metric == "ptm":
                results.append(af2_to_ptm(conf_aux))
            elif metric == "plddt":
                results.append(af2_to_plddt(conf_aux, binder_offset=binder_offset, binder_len=ab_len))
            elif metric == "cdr_plddt":
                results.append(af2_to_cdr_plddt(conf_aux, self.cdr_indices, binder_offset=binder_offset))
            elif metric == "iptm":
                if complex_aux is None:
                    raise ValueError("ipTM requested but no complex prediction was run.")
                results.append(af2_to_iptm(complex_aux))
            elif metric == "charge_balance":
                from reward import cdr_charge_score
                results.append(cdr_charge_score(sequence_str, self.cdr_indices) if sequence_str else 0.0)
            elif metric == "cdr_hydrophobicity":
                from reward import cdr_hydrophobicity_score
                results.append(cdr_hydrophobicity_score(_pdb_input(), self.cdr_indices))
            elif metric == "tm":
                from reward import pdb_to_tm
                results.append(pdb_to_tm(ori_pdb_file, _pdb_input()))
            elif metric == "crmsd":
                from reward import pdb_to_crmsd
                results.append(pdb_to_crmsd(ori_pdb_file, _pdb_input()))
            elif metric == "hydrophobic":
                from reward import pdb_to_hydrophobic_score
                results.append(pdb_to_hydrophobic_score(_pdb_input()))
            elif metric == "match_ss":
                from reward import pdb_to_match_ss_score
                seq_len = len(sequence_str) if sequence_str else 0
                r, _ = pdb_to_match_ss_score(ori_pdb_file, _pdb_input(), 1, seq_len + 1)
                results.append(r)
            elif metric == "surface_expose":
                from reward import pdb_to_surface_expose_score
                results.append(pdb_to_surface_expose_score(_pdb_input()))
            elif metric == "globularity":
                from reward import pdb_to_globularity_score
                results.append(pdb_to_globularity_score(_pdb_input()))
            else:
                raise NotImplementedError(f"Metric '{metric}' not implemented for AF2 antibody backend.")
        return results

    # --------------------------------------------------------
    # Reward orchestration
    # --------------------------------------------------------
    def reward_metrics(
        self,
        protein_name: str,
        mask_for_loss,
        S_sp,
        ori_pdb_file: Optional[str] = None,
        save_pdb: bool = False,
        add_info: str = "",
    ):
        """Score a batch of tokenized antibody sequences with AF2.

        Returns:
            (record_reward, record_reward_agg, 0.0)
            - record_reward: list of per-sequence per-metric scores (list[list[float]])
            - record_reward_agg: list of per-sequence weighted aggregates
        """
        t0 = time.perf_counter()
        n_seqs_this_call = 0
        try:
            sc_output_dir = os.path.join(self.pdb_save_path, self.run_name)
            os.makedirs(sc_output_dir, exist_ok=True)

            ab_sequences: List[str] = []
            for _it, ssp in enumerate(S_sp):
                seq_string = "".join(
                    ALPHABET[x] for _ix, x in enumerate(ssp) if mask_for_loss[_it][_ix] == 1
                )
                ab_sequences.append(seq_string)
            n_seqs_this_call = len(ab_sequences)

            if self.af_gpu_ids and len(self.af_gpu_ids) > 1:
                return self._reward_metrics_parallel(
                    protein_name=protein_name,
                    ab_sequences=ab_sequences,
                    ori_pdb_file=ori_pdb_file,
                    save_pdb=save_pdb,
                    add_info=add_info,
                    sc_output_dir=sc_output_dir,
                )

            return self._reward_metrics_serial(
                protein_name=protein_name,
                ab_sequences=ab_sequences,
                ori_pdb_file=ori_pdb_file,
                save_pdb=save_pdb,
                add_info=add_info,
                sc_output_dir=sc_output_dir,
            )
        finally:
            # JAX device pinning + JAX/CUDA init may shift torch.cuda.current_device()
            # at the OS level. Restore it so the diffusion model's subsequent forward
            # passes don't allocate fresh tensors on the wrong device.
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.set_device(0)
            except Exception:
                pass
            self._timings["reward_seconds"] += time.perf_counter() - t0
            self._timings["n_sequences"] += n_seqs_this_call
            self._timings["n_calls"] += 1

    def _reward_metrics_serial(
        self,
        protein_name: str,
        ab_sequences: List[str],
        ori_pdb_file: Optional[str],
        save_pdb: bool,
        add_info: str,
        sc_output_dir: str,
    ):
        record_reward: List[List[float]] = []
        record_reward_agg: List[float] = []

        for _it, ab_seq in enumerate(ab_sequences):
            complex_aux: Optional[dict] = None
            monomer_aux: Optional[dict] = None
            binder_offset = 0
            pdb_str: str

            if self.needs_complex:
                complex_aux = self._predict_complex(ab_seq)
                pdb_str = complex_aux["_pdb_str"]
                binder_offset = self._complex_target_len
            else:
                monomer_aux = self._predict_monomer(ab_seq)
                pdb_str = monomer_aux["_pdb_str"]
                binder_offset = 0

            pdb_path = os.path.join(sc_output_dir, f"{protein_name}{_it}_{add_info}.pdb")
            if save_pdb:
                with open(pdb_path, "w") as ff:
                    ff.write(pdb_str)
                fasta_path = os.path.join(sc_output_dir, f"{protein_name}{_it}_{add_info}.fasta")
                with open(fasta_path, "w") as f:
                    f.write(f">{protein_name}\n{ab_seq}\n")
                pdb_for_metrics: object = pdb_path
            else:
                pdb_for_metrics = pdb_str

            all_reward = self.metrics_cal(
                metrics_name=self.metrics_name,
                complex_aux=complex_aux,
                monomer_aux=monomer_aux,
                ori_pdb_file=ori_pdb_file,
                gen_pdb_file=pdb_for_metrics,
                save_pdb=save_pdb,
                sequence_str=ab_seq,
                binder_offset=binder_offset,
            )
            agg = sum(v * w for v, w in zip(all_reward, self.metrics_list))
            record_reward.append(all_reward)
            record_reward_agg.append(agg)

        return record_reward, record_reward_agg, 0.0

    def _reward_metrics_parallel(
        self,
        protein_name: str,
        ab_sequences: List[str],
        ori_pdb_file: Optional[str],
        save_pdb: bool,
        add_info: str,
        sc_output_dir: str,
    ):
        """Multi-GPU variant of reward_metrics. Dispatches AF predicts across workers."""
        from concurrent.futures import ThreadPoolExecutor

        self._ensure_workers()
        n_workers = len(self._workers)

        # Dispatch SHARDS (one task per worker) rather than one task per sequence.
        # Each AF model mutates internal state (model.aux) on every predict() call;
        # if two ThreadPoolExecutor threads land on the same worker concurrently,
        # one thread's aux read can capture the other thread's prediction.
        # Sharding guarantees at most one in-flight predict per worker.
        shards: List[List[int]] = [[] for _ in range(n_workers)]
        for i in range(len(ab_sequences)):
            shards[i % n_workers].append(i)

        def _shard_task(worker_idx: int, idx_list: List[int]):
            worker = self._workers[worker_idx]
            out = []
            for idx in idx_list:
                ab_seq = ab_sequences[idx]
                if self.needs_complex:
                    aux, target_len = worker.predict_complex(ab_seq)
                    out.append((idx, aux, None, target_len))
                else:
                    aux = worker.predict_monomer(ab_seq)
                    out.append((idx, None, aux, 0))
            return out

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(_shard_task, w, shards[w])
                for w in range(n_workers) if shards[w]
            ]
            all_results = []
            for f in futures:
                all_results.extend(f.result())
        results = sorted(all_results, key=lambda x: x[0])

        record_reward: List[List[float]] = []
        record_reward_agg: List[float] = []

        for idx, complex_aux, monomer_aux, target_len in results:
            ab_seq = ab_sequences[idx]
            binder_offset = target_len if self.needs_complex else 0
            pdb_str = (complex_aux or monomer_aux)["_pdb_str"]

            pdb_path = os.path.join(sc_output_dir, f"{protein_name}{idx}_{add_info}.pdb")
            if save_pdb:
                with open(pdb_path, "w") as ff:
                    ff.write(pdb_str)
                fasta_path = os.path.join(sc_output_dir, f"{protein_name}{idx}_{add_info}.fasta")
                with open(fasta_path, "w") as f:
                    f.write(f">{protein_name}\n{ab_seq}\n")
                pdb_for_metrics: object = pdb_path
            else:
                pdb_for_metrics = pdb_str

            all_reward = self.metrics_cal(
                metrics_name=self.metrics_name,
                complex_aux=complex_aux,
                monomer_aux=monomer_aux,
                ori_pdb_file=ori_pdb_file,
                gen_pdb_file=pdb_for_metrics,
                save_pdb=save_pdb,
                sequence_str=ab_seq,
                binder_offset=binder_offset,
            )
            agg = sum(v * w for v, w in zip(all_reward, self.metrics_list))
            record_reward.append(all_reward)
            record_reward_agg.append(agg)

        return record_reward, record_reward_agg, 0.0
