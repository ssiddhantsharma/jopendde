"""Typed `eqx.Module` structs for the model's inputs and outputs.

Attribute access on a typed module gives static field names (typos fail at
construction, not silently as a missing key) and lets the structural-token
branch rebuild via `dataclasses.replace`. `Features.from_dict` is the single
boundary adapter from a raw featurizer dict: unknown keys are ignored, a
missing required key is a KeyError at construction, and the MSA/template groups
are built only when present.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import equinox as eqx
import jax


# ---------------------------------------------------------------------------
# DenseTrunkPad -- windowed-attention padding metadata. Produced by
# rearrange_qk_to_dense_trunk or supplied in the feature dict. Only
# `mask_trunked` and `q_pad` are read.
# ---------------------------------------------------------------------------
class DenseTrunkPad(eqx.Module):
    q_pad: int          # right-pad on the query axis (always set)
    k_pad_left: int     # left-pad for the overlapping key windows (always set)
    mask_trunked: jax.Array | None = None  # None only when built with compute_mask=False

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DenseTrunkPad":
        return cls(
            q_pad=int(d["q_pad"]),
            k_pad_left=int(d["k_pad_left"]),
            mask_trunked=d.get("mask_trunked"),
        )


# ---------------------------------------------------------------------------
# Conditional feature groups -- present as a whole or not at all. Grouping them
# into sub-modules makes the optionality all-or-nothing (you can't have `msa`
# with `has_deletion=None`); within a group every field is required.
# ---------------------------------------------------------------------------
class MSAFeatures(eqx.Module):
    """MSA features. Always present: the featurizer emits at least the 1-row
    query MSA even for single-sequence (`use_msa=False`) runs."""

    msa: jax.Array            # [S, N] integer class indices (row 0 = query; rows 1.. homologs)
    has_deletion: jax.Array   # [S, N]
    deletion_value: jax.Array  # [S, N]
    # Soft one-hot of the query row (row 0), [N, 32]. The differentiable entry
    # point into the MSA stream: at inference it is exactly `one_hot(msa[0])`,
    # but a caller can substitute a soft PSSM. Only the query row is soft; the
    # homolog depth stays integer.
    query_soft: jax.Array


class TemplateFeatures(eqx.Module):
    """Template features -- present iff the run used templates. AF3 `template_*`
    keys with the prefix stripped."""

    aatype: jax.Array
    pseudo_beta_mask: jax.Array
    distogram: jax.Array
    unit_vector: jax.Array
    backbone_frame_mask: jax.Array


# ---------------------------------------------------------------------------
# Features -- the model input.
#
# Flat required fields (present in every featurization) up top; the only
# optional members are the conditional groups `msa`/`template` (present iff the
# run used them). A flat struct so the structural-token branch can rebuild via
# `dataclasses.replace`. Computed intermediates (e.g. the structural-token
# attention bias) are passed as explicit args, NOT carried on this input struct.
# ---------------------------------------------------------------------------
class Features(eqx.Module):
    # token-level metadata
    token_index: jax.Array
    residue_index: jax.Array
    asym_id: jax.Array
    entity_id: jax.Array
    sym_id: jax.Array
    restype: jax.Array
    token_bonds: jax.Array
    has_frame: jax.Array
    frame_atom_index: jax.Array
    profile: jax.Array
    deletion_mean: jax.Array
    relp: jax.Array

    # atom-level reference features
    ref_pos: jax.Array
    ref_charge: jax.Array
    ref_mask: jax.Array
    ref_element: jax.Array
    ref_atom_name_chars: jax.Array
    atom_to_token_idx: jax.Array
    atom_to_tokatom_idx: jax.Array
    is_ligand: jax.Array  # per-atom; read only by the confidence summary
    distogram_rep_atom_mask: jax.Array
    pae_rep_atom_mask: jax.Array
    d_lm: jax.Array
    v_lm: jax.Array
    pad_info: DenseTrunkPad

    # structural-token metadata (expansion is always on for opendde_v1)
    structural_token_index: jax.Array
    subtoken_role_id: jax.Array
    parent_residue_idx: jax.Array
    prev_parent_residue_idx: jax.Array
    next_parent_residue_idx: jax.Array
    atom_to_structural_token_idx: jax.Array
    atom_to_structural_tokatom_idx: jax.Array
    structural_has_frame: jax.Array
    structural_frame_atom_index: jax.Array
    structural_distogram_rep_atom_mask: jax.Array
    structural_pae_rep_atom_mask: jax.Array

    # --- optional ---
    msa: MSAFeatures | None = None
    template: TemplateFeatures | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Features":
        """Adapt a raw featurizer dict. Extra keys are ignored; a missing
        *required* key is a KeyError (loud, at construction). The MSA/template
        groups are built only when present."""
        special = {"pad_info", "msa", "template"}
        kwargs = {f.name: d[f.name] for f in dataclasses.fields(cls) if f.name not in special}
        msa = (
            MSAFeatures(
                msa=d["msa"],
                has_deletion=d["has_deletion"],
                deletion_value=d["deletion_value"],
                # query row -> soft one-hot in the restype vocabulary (32 classes).
                query_soft=jax.nn.one_hot(
                    d["msa"][0],
                    d["restype"].shape[-1],
                    dtype=d["restype"].dtype,
                ),
            )
            if "msa" in d
            else None
        )
        template = (
            TemplateFeatures(
                aatype=d["template_aatype"],
                pseudo_beta_mask=d["template_pseudo_beta_mask"],
                distogram=d["template_distogram"],
                unit_vector=d["template_unit_vector"],
                backbone_frame_mask=d["template_backbone_frame_mask"],
            )
            if "template_aatype" in d
            else None
        )
        return cls(pad_info=DenseTrunkPad.from_dict(d["pad_info"]), msa=msa, template=template, **kwargs)


# ---------------------------------------------------------------------------
# Prediction -- the top-level model output.
# ---------------------------------------------------------------------------
class Prediction(eqx.Module):
    coordinate: jax.Array
    contact_probs: jax.Array
    plddt: jax.Array
    pae: jax.Array
    pde: jax.Array
    resolved: jax.Array


# ---------------------------------------------------------------------------
# SummaryParams -- bins + clash threshold for the confidence summary, fixed by
# the checkpoint's confidence config. Plain floats/tuples so summarize() needs
# only this, not the config object.
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class SummaryParams:
    plddt_bins: tuple  # (min_bin, max_bin, no_bins)
    pae_bins: tuple
    pde_bins: tuple
    af3_clash_threshold: float = 1.1
