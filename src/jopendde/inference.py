"""Inference API, split into two layers.

**Torch-free predict path** -- `predict()` and `summarize()` are plain functions
of a converted jopendde model + `Features`. They import only jax/equinox/numpy,
so once you hold those two objects you can run inference in an environment
without torch or opendde installed.

**Torch-bound prep** -- `Predictor` builds+converts the model from a checkpoint
and featurizes inputs. This *unavoidably* needs torch + opendde: jopendde has no
featurizer of its own (features come from OpenDDE's data pipeline), and the
weights come from a torch checkpoint. Those imports are lazy (inside the
methods), so importing this module stays torch-free.

    import jax
    from jopendde.inference import Predictor, predict, summarize, spec_from_sequences

    p = Predictor.from_checkpoint()                          # torch: build + convert
    inp = p.featurize(spec_from_sequences(["MQIF..."], name="ubq", seed=0))
    # from here, torch-free -- needs only (model, Features, SummaryParams):
    pred = predict(p.model, inp.feat, jax.random.key(0), n_cycle=10, n_sample=5, n_step=200)
    summary = summarize(inp.feat, pred, p.summary_params, n_recycle=10)
    p.save(inp, pred, summary, "out/", pdb_id="ubq", seed=0)
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import tempfile
from typing import Any

import equinox as eqx
import jax
import numpy as np

from jopendde.features import Features, Prediction, SummaryParams


def enable_cue_kernels(model):
    """Swap the trunk's triangle-*attention* modules to the fused NVIDIA
    cuEquivariance kernel (optional; requires `cuequivariance-jax` +
    `cuequivariance-ops-jax-cu12`). Returns a new model -- the input is unchanged
    (equinox modules are immutable).

    Scope, and why it's attention-only:
    * Only `pairformer_stack` + `msa_module` (the unbatched trunk, run once per
      recycle -- where the triangle cost lives). The confidence head runs its
      pairformer under `jax.vmap` over samples and the fused primitives have no
      batching rule, so it stays on the pure path.
    * The fused *triangle-multiply* (Triton) kernel is left off by default: it
      runs slower than XLA at these token counts, cancelling the attention win.
      The triangle-*attention* kernel is a speedup that grows with token count,
      so only that one is switched on. The `use_cue_kernel` flag on
      TriangleMultiplicativeUpdate stays available for sizes where the Triton
      kernel is competitive.

    The kernel agrees with the pure path to ~1e-3."""
    from jopendde.triangular import CUE_AVAILABLE, TriangleAttention

    if not CUE_AVAILABLE:
        raise RuntimeError(
            "cuEquivariance not installed; `pip install cuequivariance-jax "
            "cuequivariance-ops-jax-cu12` (CUDA 12) to use the fused kernels."
        )

    def _is_attn(x):
        return isinstance(x, TriangleAttention)

    def _flip(subtree):
        return jax.tree_util.tree_map(
            lambda x: dataclasses.replace(x, use_cue_kernel=True) if _is_attn(x) else x,
            subtree,
            is_leaf=_is_attn,
        )

    return eqx.tree_at(
        lambda m: (m.pairformer_stack, m.msa_module),
        model,
        (_flip(model.pairformer_stack), _flip(model.msa_module)),
    )


# ===========================================================================
# Torch-free predict path -- pure jax on a converted model + Features.
# ===========================================================================

# model is an explicit arg (not captured as a large jit constant);
# n_cycle/n_sample/n_step are static (scan lengths / lowering). Matmul precision
# is left at the JAX/XLA default.
@eqx.filter_jit
def _run(model, feat, key, n_cycle, n_sample, n_step):
    return model.predict(feat, key, n_cycle=n_cycle, n_sample=n_sample, n_step=n_step)


def predict(model, feat: Features, key, *, n_cycle: int, n_sample: int, n_step: int) -> Prediction:
    """Run inference. Torch-free: given a converted jopendde model and `Features`,
    this needs no torch/opendde. `key` is a jax random key; `n_cycle`/`n_sample`/
    `n_step` are the inference budget (all required)."""
    return _run(model, feat, key, n_cycle, n_sample, n_step)


def summarize(feat: Features, pred: Prediction, params: SummaryParams, *, n_recycle: int) -> list[dict]:
    """Per-sample confidence (ipTM/pTM/pLDDT/gPDE/ranking_score/...). Torch-free:
    reads bins + clash threshold from `params`, so no opendde config needed."""
    from jopendde.confidence_summary import compute_summary_confidence

    return compute_summary_confidence(
        pae_logits=np.asarray(pred.pae),
        plddt_logits=np.asarray(pred.plddt),
        pde_logits=np.asarray(pred.pde),
        contact_probs=np.asarray(pred.contact_probs),
        token_asym_id=np.asarray(feat.asym_id),
        token_has_frame=np.asarray(feat.has_frame),
        atom_coordinate=np.asarray(pred.coordinate),
        atom_to_token_idx=np.asarray(feat.atom_to_token_idx),
        atom_is_polymer=1 - np.asarray(feat.is_ligand),
        N_recycle=n_recycle,
        plddt_bins=params.plddt_bins,
        pae_bins=params.pae_bins,
        pde_bins=params.pde_bins,
        af3_clash_threshold=params.af3_clash_threshold,
    )


# ===========================================================================
# Torch-bound prep -- build/convert the model and featurize inputs.
# ===========================================================================
@dataclasses.dataclass
class FeaturizedInput:
    """One featurized example -- torch-free. `feat` is the jopendde model input;
    `atom_array` (biotite) and `entity_poly_type` are for writing structures."""

    feat: Features
    atom_array: Any
    entity_poly_type: dict
    n_token: int


def spec_from_sequences(sequences: list, *, name: str, seed: int) -> list:
    """Build an OpenDDE/AF3-style input spec from protein sequences, so callers
    can `featurize` without knowing the JSON schema. `sequences` is always a list
    (wrap a single sequence: `["MQIF..."]`); one chain per entry, count 1."""
    return [{
        "name": name,
        "modelSeeds": [seed],
        "sequences": [{"proteinChain": {"sequence": s, "count": 1}} for s in sequences],
    }]


@contextlib.contextmanager
def _spec_tempfile(spec: list):
    """Write an in-memory input spec to a named temp JSON file (OpenDDE's
    dataloader is file-driven), removed on exit -- so callers needn't manage a
    file on disk."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(spec, tf)
        path = tf.name
    try:
        yield path
    finally:
        os.unlink(path)


def _to_numpy(v):
    # structural recursion over the featurizer's nested tensor dict (not
    # argument type-overloading): torch tensors -> numpy, recurse into dicts.
    import torch

    if isinstance(v, torch.Tensor):
        return v.detach().cpu().numpy()
    if isinstance(v, dict):
        return {k: _to_numpy(x) for k, x in v.items()}
    return v


def _set_asset_cache_dir(configs, asset_cache_dir) -> None:
    configs.load_checkpoint_dir = os.path.join(asset_cache_dir, "checkpoint")
    configs.data.ccd_components_file = os.path.join(
        asset_cache_dir, "common", "components.cif"
    )
    configs.data.ccd_components_rdkit_mol_file = os.path.join(
        asset_cache_dir, "common", "components.cif.rdkit_mol.pkl"
    )


class Predictor:
    """A loaded OpenDDE checkpoint ready for JAX inference.

    Holds the converted jopendde model (`.model`, the torch-free JAX product),
    its `.summary_params`, OpenDDE's `InferenceRunner` (`.runner`, used for
    featurization and `reference_predict`), and the resolved config
    (`.configs`). Inference itself is the module-level torch-free `predict`/
    `summarize` functions -- the Predictor only prepares their inputs."""

    def __init__(self, model, runner, configs, summary_params):
        self.model = model
        self.runner = runner
        self.configs = configs
        self.summary_params = summary_params

    @classmethod
    def from_checkpoint(
        cls,
        model_name: str = "opendde_v1",
        checkpoint_file: str | None = None,
        asset_cache_dir: str | os.PathLike | None = None,
    ) -> "Predictor":
        """Download the released checkpoint (+ CCD data) from Hugging Face, build
        the OpenDDE model + load it (fast-init), convert to jopendde, and return a
        ready Predictor. The config is pinned to the deterministic single-sequence
        path (torch triangle kernels, no MSA/template, perf fusions off);
        these are fixed, not tunable. Inference-budget knobs
        (n_cycle/n_sample/n_step) are NOT set here; they're chosen per `predict()`
        call.

        `checkpoint_file` overrides the weight file for `model_name` (e.g.
        "opendde_abag.pt"), fetched from the same HF repo. The `model_name`
        config must match the checkpoint's architecture. `asset_cache_dir`
        overrides the directory used for checkpoints and CCD data."""
        from opendde.config.dependency_url import dependency_url
        from opendde.config.inference import (
            build_inference_config,
            update_gpu_compatible_configs,
        )
        from opendde.utils.download import download_from_url, download_inference_cache
        from runner.inference import InferenceRunner

        from jopendde import convert  # noqa: F401 -- registers from_torch converters
        from jopendde.fast_init import fast_init
        from jopendde.model import OpenDDE as JaxOpenDDE

        configs = build_inference_config(model_name=model_name, fill_required_with_null=True)
        if asset_cache_dir is not None:
            _set_asset_cache_dir(configs, asset_cache_dir)
        configs.use_msa = configs.use_template = configs.use_rna_msa = False
        configs.triangle_multiplicative = configs.triangle_attention = "torch"
        configs.enable_diffusion_shared_vars_cache = False
        configs.enable_efficient_fusion = False
        configs = update_gpu_compatible_configs(configs)

        # Point at a non-default weight file (e.g. the ABAG checkpoint).
        if checkpoint_file is not None:
            ckpt_path = os.path.join(configs.load_checkpoint_dir, checkpoint_file)
            if not os.path.exists(ckpt_path):
                os.makedirs(configs.load_checkpoint_dir, exist_ok=True)
                download_from_url(dependency_url(checkpoint_file), ckpt_path)
            configs.load_checkpoint_path = ckpt_path

        # Fetch the torch checkpoint + CCD data from HF into the local cache
        # (idempotent; skips the checkpoint when load_checkpoint_path already exists).
        download_inference_cache(configs)

        with fast_init():
            runner = InferenceRunner(configs)
        model = JaxOpenDDE.from_torch(runner.model)

        c = configs.confidence
        summary_params = SummaryParams(
            plddt_bins=(c.plddt.min_bin, c.plddt.max_bin, c.plddt.no_bins),
            pae_bins=(c.pae.min_bin, c.pae.max_bin, c.pae.no_bins),
            pde_bins=(c.pde.min_bin, c.pde.max_bin, c.pde.no_bins),
            af3_clash_threshold=configs.metrics.clash.af3_clash_threshold,
        )
        return cls(model, runner, configs, summary_params)

    def _load_batch(self, spec: list):
        """Run OpenDDE's dataloader on `spec` (written to a temp file, since the
        loader is file-driven) and return the raw torch batch `data` + its
        `atom_array`, sized for this input, with the spec's seed stamped."""
        import torch
        from opendde.data.inference.infer_dataloader import get_inference_dataloader
        from runner.inference import update_inference_configs

        seed = int(spec[0]["modelSeeds"][0])
        with _spec_tempfile(spec) as json_path:
            self.configs.input_json_path = json_path
            self.configs.seeds = [seed]
            data, atom_array, err = next(iter(get_inference_dataloader(configs=self.configs)))[0]
            assert not err, err
            data["input_feature_dict"]["inference_seed"] = torch.tensor(seed, dtype=torch.long)
            self.runner.update_model_configs(
                update_inference_configs(self.configs, int(data["N_token"].item()))
            )
            return data, atom_array

    def featurize(self, spec: list) -> FeaturizedInput:
        """Featurize one input into a (torch-free) `Features`. `spec` is an
        in-memory OpenDDE/AF3 spec (a list of jobs, e.g. from
        `spec_from_sequences`); the seed comes from its `modelSeeds`."""
        import copy

        from opendde.model.opendde import update_input_feature_dict

        data, atom_array = self._load_batch(spec)
        ept = {k: v for k, v in data["entity_poly_type"].items() if v != "non-polymer"}
        raw = copy.deepcopy(data["input_feature_dict"])
        raw = self.runner.model.relative_position_encoding.generate_relp(raw, lazy=False)
        raw = update_input_feature_dict(raw)
        return FeaturizedInput(
            feat=Features.from_dict(_to_numpy(raw)),
            atom_array=atom_array,
            entity_poly_type=ept,
            n_token=int(data["N_token"].item()),
        )

    def reference_predict(self, spec: list) -> dict:
        """PyTorch reference forward off the same featurizer, for parity
        checks against `predict`. Returns OpenDDE's pred_dict
        (coordinate, summary_confidence, ...). Keeps the torch batch confined
        here so `FeaturizedInput` stays torch-free."""
        import torch

        data, _ = self._load_batch(spec)
        torch.manual_seed(int(spec[0]["modelSeeds"][0]))
        return self.runner.predict(data)

    def save(self, inp: FeaturizedInput, pred: Prediction, summary, out_dir, *,
             pdb_id: str, seed: int) -> None:
        """Write per-sample CIFs (ranked by ranking_score) + confidence JSON via
        OpenDDE's DataDumper."""
        import torch
        from runner.dumper import DataDumper

        pred_dict = {
            "coordinate": torch.tensor(np.asarray(pred.coordinate)),
            "summary_confidence": summary,
            "full_data": [{}] * len(summary),
        }
        DataDumper(base_dir=str(out_dir), sorted_by_ranking_score=True).dump(
            group_name="", pdb_id=pdb_id, seed=seed, pred_dict=pred_dict,
            atom_array=inp.atom_array, entity_poly_type=inp.entity_poly_type,
        )
