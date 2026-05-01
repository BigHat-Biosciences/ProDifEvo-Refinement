"""AlphaFold2.3-multimer reward backend for antibody CDR design.

Replaces the ESMFold-based ``AbRewardCal`` from ``ab_refinement.py`` with a
colabdesign-driven AF2 backend. Two underlying models are constructed lazily:

* ``mk_afdesign_model(protocol="binder", use_multimer=True)`` for predicting the
  antibody:antigen complex (used when ``iptm`` is requested). The antigen PDB
  is supplied as the target template; the antibody is hallucinated as the
  binder of length ``ab_len``.
* ``mk_afdesign_model(protocol="hallucination", use_multimer=False)`` for
  predicting the antibody monomer (used when only monomer-style metrics like
  ``plddt``, ``cdr_plddt``, ``ptm`` are requested without an antigen).

AF2 is not batchable like ESMFold — each candidate sequence is predicted
sequentially. Per-sequence metrics are extracted from ``model.aux``.

AF2 weights are expected to live in ``~/.mber/af_params`` by default; this can
be overridden with the ``AF_PARAMS_DIR`` env var or the ``af_params_dir`` ctor
argument. See ``mber-open/download_weights.sh`` for how to obtain them.
"""

from __future__ import annotations

import os
import logging
import tempfile
import time
from io import StringIO
from typing import List, Optional, Sequence

import numpy as np

# colabdesign is jax-based; importing it eagerly would be costly and would
# fail on environments where AF2 is not yet installed. Defer to first use.
_AF_FACTORY = None
_CLEAR_MEM = None
_NBB2_CLS = None


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


def _lazy_import_nbb2():
    """Import NanoBodyBuilder2 on first use."""
    global _NBB2_CLS
    if _NBB2_CLS is not None:
        return _NBB2_CLS
    try:
        from ImmuneBuilder import NanoBodyBuilder2
    except ImportError as e:
        raise ImportError(
            "ImmuneBuilder is required for --use_template (NanoBodyBuilder2 binder pre-folding). "
            "Install with: pip install ImmuneBuilder"
        ) from e
    _NBB2_CLS = NanoBodyBuilder2
    return _NBB2_CLS


ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


def _combine_target_and_binder_pdb(
    target_pdb_path: str,
    binder_pdb_str: str,
    target_chain: str,
    binder_chain: str = "H",
) -> str:
    """Concatenate target ATOMs and binder ATOMs into a single multi-chain PDB string.

    Vendored from mber-open's pdb_utils.combine_structures, simplified to operate
    on a target PDB path + a binder PDB string. The binder chain is rewritten to
    ``binder_chain`` (default 'H'); the target keeps its existing chain ID.
    """
    with open(target_pdb_path, "r") as f:
        target_pdb = f.read()

    lines = ["HEADER    PROTEIN", "TITLE     COMBINED TARGET+BINDER (NBB2 TEMPLATE)"]
    atom_count = 1

    for line in target_pdb.splitlines():
        if line.startswith("ATOM"):
            lines.append(f"ATOM  {atom_count:5d}{line[11:]}")
            atom_count += 1
    lines.append(f"TER   {atom_count:5d}      {target_chain}")

    binder_atoms = 0
    for line in binder_pdb_str.splitlines():
        if line.startswith("ATOM"):
            # Force the binder chain ID to `binder_chain` (column 22, 0-indexed 21).
            new_line = f"ATOM  {atom_count:5d}{line[11:21]}{binder_chain}{line[22:]}"
            lines.append(new_line)
            atom_count += 1
            binder_atoms += 1
    if binder_atoms > 0:
        lines.append(f"TER   {atom_count:5d}      {binder_chain}")
    lines.append("END")
    return "\n".join(lines)


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
    forward passes are routed to that device. Workers don't share state, so a
    pool of them can run concurrently from a ThreadPoolExecutor.
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
    ):
        self.jax_device = jax_device
        self.af_params_dir = af_params_dir
        self.num_recycles = num_recycles
        self.use_multimer = use_multimer
        self.af_models = list(af_models)
        self.antigen_pdb = antigen_pdb
        self.antigen_chain = antigen_chain
        self.binder_chain = binder_chain

        self._complex_model = None
        self._complex_target_len = 0
        self._complex_binder_len = 0
        self._template_initialized = False  # set True after init_template()
        self._monomer_model = None
        self._monomer_len = 0

    def _ensure_complex_model(self, ab_len: int) -> None:
        """Build the non-templated binder model. Skipped when templating is in use."""
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
                hotspot=None,
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

    def init_template(self, combined_pdb_path: str, rm_binder_str: str, ab_len: int) -> None:
        """One-shot template init: build a binder model with the combined target+binder PDB
        and bake in the CDR-position mask via ``rm_binder``. After this call, ``predict_complex``
        is just a forward pass — no per-candidate ``_prep_binder``.

        Args:
            combined_pdb_path: Path to a multi-chain PDB containing target + pre-folded binder.
            rm_binder_str: Comma-separated colabdesign positions to mask from the binder
                template (e.g. ``"H97,H98,..."`` for designed CDR positions).
            ab_len: Antibody length, used to validate model reuse.
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
                pdb_filename=combined_pdb_path,
                chain=self.antigen_chain,
                binder_chain=self.binder_chain,
                hotspot=None,
                seed=0,
                rm_target=False,
                rm_target_seq=False,
                rm_target_sc=False,
                rm_template_ic=True,
                rm_binder=rm_binder_str,
                rm_binder_seq=True,
                rm_binder_sc=True,
            )
            self._complex_binder_len = ab_len
            self._complex_target_len = int(self._complex_model._target_len)
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

        If ``init_template`` was called, the model is already templated — this just runs
        ``predict``. Otherwise the non-templated path builds a fresh model on first call.
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
        antigen_seq: Optional[str] = None,  # accepted for API parity; not used by AF2 binder protocol
        af_params_dir: Optional[str] = None,
        num_recycles: int = 3,
        af_models: Sequence[int] = (0,),
        use_multimer: bool = True,
        use_template: bool = False,
        nbb2_weights_dir: Optional[str] = None,
        af_gpu_ids: Optional[Sequence[int]] = None,
        seed_sequence: Optional[str] = None,
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

        # NBB2 templating: per-candidate NBB2 fold of the antibody is fed to
        # AF2's binder protocol as a binder-side template (backbone only).
        self.use_template = bool(use_template)
        if self.use_template and not self.needs_complex:
            raise ValueError(
                "--use_template requires complex prediction (i.e. 'iptm' in --metrics_name). "
                "Templating is only meaningful when an antigen is present."
            )
        self.nbb2_weights_dir = os.path.expanduser(
            nbb2_weights_dir
            or os.environ.get("NBB2_WEIGHTS_DIR", "~/.mber/nbb2_weights")
        )
        # Binder chain to use in the combined template PDB. NBB2 emits chain 'H'
        # for nanobody heavy chain; we keep that for clarity.
        self._binder_chain = "H"

        # Seed sequence used for the one-shot NBB2 fold when templating is enabled.
        # The CDR positions are masked out of the binder template via rm_binder, so
        # the seed sequence's CDR atoms don't matter — only the framework atoms do.
        self.seed_sequence = seed_sequence
        if self.use_template and not self.seed_sequence:
            raise ValueError(
                "--use_template requires a seed antibody sequence to fold once at startup. "
                "Pass seed_sequence=args.antibody_sequence."
            )

        # One-shot template state: the serial-mode AF model is templated lazily on first
        # call; parallel-mode workers are templated when _ensure_workers builds them.
        self._template_initialized_serial = False

        # Lazy-initialised colabdesign model handles
        self._complex_model = None
        self._monomer_model = None
        self._complex_target_len = 0   # number of antigen residues in the complex
        self._complex_binder_len = 0   # antibody length the complex was prepped for
        self._monomer_len = 0          # antibody length the monomer was prepped for

        # Lazy NBB2 model handle (only built if use_template).
        self._nbb2_model = None

        # Multi-GPU AF parallelism. If af_gpu_ids has 2+ device IDs, AF
        # predictions are dispatched in parallel across one _AFWorker per GPU.
        # If 0 or 1 device IDs, the existing serial path on the default device
        # is used (no behavior change).
        self.af_gpu_ids = list(af_gpu_ids) if af_gpu_ids else []
        self._workers: List["_AFWorker"] = []  # lazy-built on first use

        # Cumulative timing/counts. Updated by reward_metrics on every call.
        self._timings = {
            "n_sequences": 0,        # total sequences scored
            "reward_seconds": 0.0,   # total wall time inside reward_metrics
            "n_calls": 0,            # number of reward_metrics invocations
        }

    # --------------------------------------------------------
    # Model construction
    # --------------------------------------------------------
    def _ensure_complex_model(self, ab_len: int) -> None:
        mk_model, clear_mem = _lazy_import_colabdesign()
        if self._complex_model is None or self._complex_binder_len != ab_len:
            if self._complex_model is not None:
                clear_mem()
            logging.info(
                f"[AF2] Building binder/multimer model (params={self.af_params_dir}, "
                f"recycles={self.num_recycles}, multimer={self.use_multimer}, ab_len={ab_len})"
            )
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
                hotspot=None,
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

    def _ensure_monomer_model(self, ab_len: int) -> None:
        mk_model, clear_mem = _lazy_import_colabdesign()
        if self._monomer_model is None or self._monomer_len != ab_len:
            if self._monomer_model is not None:
                clear_mem()
            logging.info(
                f"[AF2] Building hallucination/monomer model (recycles={self.num_recycles}, ab_len={ab_len})"
            )
            self._monomer_model = mk_model(
                protocol="hallucination",
                use_templates=False,
                num_recycles=self.num_recycles,
                data_dir=self.af_params_dir,
                use_multimer=False,
            )
            # Hallucination protocol needs an explicit length; safe to call repeatedly.
            self._monomer_model._prep_hallucination(length=ab_len)
            self._monomer_len = ab_len

    def _ensure_workers(self) -> None:
        """Lazy-build a pool of _AFWorker instances pinned to ``af_gpu_ids``.

        When ``use_template`` is set, also runs the one-shot template init on each
        worker (NBB2-fold the seed once, combine with antigen, then init each worker).
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
            )
            for dev in picked
        ]

        if self.use_template:
            combined_pdb = self._build_template_pdb()
            rm_binder_str = self._cdr_position_string()
            ab_len = len(self.seed_sequence)
            try:
                logging.info(
                    f"[AF2] Templating {len(self._workers)} workers (rm_binder={rm_binder_str})"
                )
                for w in self._workers:
                    w.init_template(combined_pdb, rm_binder_str, ab_len)
            finally:
                try:
                    os.remove(combined_pdb)
                except OSError:
                    pass

    def _build_template_pdb(self) -> str:
        """One-shot: NBB2-fold seed sequence, combine with antigen, write to a temp PDB.

        Returns the temp file path. Caller is responsible for removing it.
        """
        if not self.seed_sequence:
            raise ValueError("Templating requires a seed sequence.")
        binder_pdb_str = self._fold_binder_with_nbb2(self.seed_sequence)
        combined_pdb_str = _combine_target_and_binder_pdb(
            target_pdb_path=self.antigen_pdb,
            binder_pdb_str=binder_pdb_str,
            target_chain=self.antigen_chain,
            binder_chain=self._binder_chain,
        )
        with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as tmp:
            tmp.write(combined_pdb_str)
            return tmp.name

    def _cdr_position_string(self) -> str:
        """colabdesign-format CDR positions for ``rm_binder``, e.g. 'H97,H98,...'.

        Indices in ``self.cdr_indices`` are 0-based positions in the antibody sequence;
        colabdesign expects ``<chain><1-based-resnum>`` strings.
        """
        if not self.cdr_indices:
            return ""
        return ",".join(f"{self._binder_chain}{i + 1}" for i in self.cdr_indices)

    def _ensure_nbb2(self) -> None:
        """Lazy-build the NBB2 model on first use."""
        if self._nbb2_model is not None:
            return
        NBB2 = _lazy_import_nbb2()
        os.makedirs(self.nbb2_weights_dir, exist_ok=True)
        logging.info(f"[NBB2] Loading NanoBodyBuilder2 from {self.nbb2_weights_dir}")
        self._nbb2_model = NBB2(numbering_scheme="raw", weights_dir=self.nbb2_weights_dir)

    def _init_serial_template(self) -> None:
        """One-shot template init for the serial-mode AF model.

        Builds ``self._complex_model`` once, NBB2-folds the seed sequence, combines with
        the antigen, and calls ``_prep_binder`` with ``rm_binder=<CDR positions>`` so the
        binder framework template is preserved while CDR positions are re-hallucinated.
        """
        if self._template_initialized_serial:
            return
        mk_model, clear_mem = _lazy_import_colabdesign()
        if self._complex_model is not None:
            clear_mem()

        ab_len = len(self.seed_sequence)
        rm_binder_str = self._cdr_position_string()
        logging.info(
            f"[AF2] One-shot template init (serial): ab_len={ab_len}, "
            f"rm_binder={rm_binder_str}"
        )
        self._complex_model = mk_model(
            protocol="binder",
            debug=False,
            data_dir=self.af_params_dir,
            use_multimer=self.use_multimer,
            num_recycles=self.num_recycles,
        )

        combined_pdb = self._build_template_pdb()
        try:
            self._complex_model._prep_binder(
                pdb_filename=combined_pdb,
                chain=self.antigen_chain,
                binder_chain=self._binder_chain,
                hotspot=None,
                seed=0,
                rm_target=False,
                rm_target_seq=False,
                rm_target_sc=False,
                rm_template_ic=True,
                rm_binder=rm_binder_str,
                rm_binder_seq=True,
                rm_binder_sc=True,
            )
        finally:
            try:
                os.remove(combined_pdb)
            except OSError:
                pass

        self._complex_binder_len = ab_len
        self._complex_target_len = int(self._complex_model._target_len)
        self._template_initialized_serial = True

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------
    def _fold_binder_with_nbb2(self, ab_seq: str) -> str:
        """Fold antibody sequence with NBB2 and return the PDB as a string."""
        self._ensure_nbb2()
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            import torch
            with torch.no_grad():
                nb = self._nbb2_model.predict({"H": ab_seq})
            nb.save(tmp_path)
            with open(tmp_path, "r") as f:
                return f.read()
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def _predict_complex(self, ab_seq: str) -> dict:
        if self.use_template:
            return self._predict_complex_templated(ab_seq)
        self._ensure_complex_model(len(ab_seq))
        self._complex_model.predict(seq=ab_seq, models=self.af_models, verbose=False)
        # Pull a shallow copy so downstream mutations don't poison the model state.
        aux = dict(self._complex_model.aux)
        aux["log"] = dict(self._complex_model.aux.get("log", {}))
        aux["_pdb_str"] = self._complex_model.save_pdb()
        return aux

    def _predict_complex_templated(self, ab_seq: str) -> dict:
        """Templated complex predict (serial mode).

        On first call, runs the one-shot template init (NBB2-fold seed + combine +
        prep_binder with CDR mask). After that, each call is just an AF2 forward pass.
        """
        self._init_serial_template()
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
        # Prefer the complex aux for confidence metrics whenever available; AF2
        # multimer plddt over the binder slice is directly comparable to monomer plddt.
        conf_aux = complex_aux if complex_aux is not None else monomer_aux
        ab_len = len(sequence_str) if sequence_str else None

        def _pdb_input():
            return gen_pdb_file if save_pdb else StringIO(gen_pdb_file)

        results: List[float] = []
        for metric in metrics_name:
            if metric == "ptm":
                # In binder protocol pTM is overall complex pTM; for monomer it's plain pTM.
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

            # Decode tokens to sequence strings using the loss mask.
            ab_sequences: List[str] = []
            for _it, ssp in enumerate(S_sp):
                seq_string = "".join(
                    ALPHABET[x] for _ix, x in enumerate(ssp) if mask_for_loss[_it][_ix] == 1
                )
                ab_sequences.append(seq_string)
            n_seqs_this_call = len(ab_sequences)

            # Multi-GPU dispatch when af_gpu_ids has 2+ devices.
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
            # JAX device pinning (with jax.default_device(devices[3])) and NBB2's
            # internal torch ops both call cudaSetDevice as a side effect, leaving
            # torch.cuda.current_device() pointing at whichever GPU the AF worker
            # last touched. Restore to 0 here so the diffusion model's subsequent
            # forward passes don't allocate fresh tensors on the wrong device.
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
        """Serial single-GPU reward path. Extracted from reward_metrics for clarity."""
        record_reward: List[List[float]] = []
        record_reward_agg: List[float] = []

        for _it, ab_seq in enumerate(ab_sequences):
            # Run AF2 prediction(s) for this single sequence.
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

            # Persist PDB output (always written; cheap and useful for debugging).
            pdb_path = os.path.join(sc_output_dir, f"{protein_name}{_it}_{add_info}.pdb")
            if save_pdb:
                with open(pdb_path, "w") as ff:
                    ff.write(pdb_str)
                fasta_path = os.path.join(sc_output_dir, f"{protein_name}{_it}_{add_info}.fasta")
                with open(fasta_path, "w") as f:
                    f.write(f">{protein_name}\n{ab_seq}\n")
                pdb_for_metrics: object = pdb_path
            else:
                # Hand the raw PDB string to PDB-based metrics; they wrap it in StringIO.
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

        # _ensure_workers does the one-shot NBB2 fold + combine + per-worker template
        # init when use_template is set, so by the time we get here every worker is
        # ready and per-candidate predicts are pure forward passes.
        self._ensure_workers()
        n_workers = len(self._workers)

        def _task(idx: int):
            worker = self._workers[idx % n_workers]
            ab_seq = ab_sequences[idx]
            if self.needs_complex:
                aux, target_len = worker.predict_complex(ab_seq)
                return idx, aux, None, target_len
            else:
                aux = worker.predict_monomer(ab_seq)
                return idx, None, aux, 0

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_task, i) for i in range(len(ab_sequences))]
            results = sorted([f.result() for f in futures], key=lambda x: x[0])

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
