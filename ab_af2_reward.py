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

        # Lazy-initialised colabdesign model handles
        self._complex_model = None
        self._monomer_model = None
        self._complex_target_len = 0   # number of antigen residues in the complex
        self._complex_binder_len = 0   # antibody length the complex was prepped for
        self._monomer_len = 0          # antibody length the monomer was prepped for

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

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------
    def _predict_complex(self, ab_seq: str) -> dict:
        self._ensure_complex_model(len(ab_seq))
        self._complex_model.predict(seq=ab_seq, models=self.af_models, verbose=False)
        # Pull a shallow copy so downstream mutations don't poison the model state.
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
        sc_output_dir = os.path.join(self.pdb_save_path, self.run_name)
        os.makedirs(sc_output_dir, exist_ok=True)

        # Decode tokens to sequence strings using the loss mask.
        ab_sequences: List[str] = []
        for _it, ssp in enumerate(S_sp):
            seq_string = "".join(
                ALPHABET[x] for _ix, x in enumerate(ssp) if mask_for_loss[_it][_ix] == 1
            )
            ab_sequences.append(seq_string)

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
