"""ConfidenceHead, plus a local PairformerBlock/PairformerStack it needs.

FoldCP/distributed paths are dropped; `inplace_safe`/`chunk_size` are accepted
and ignored. PairformerBlock/PairformerStack cover only ConfidenceHead's needs
(the `c_s > 0` branch: triangle updates + AttentionPairBias + single/pair
Transitions; no MSA). They map to the same torch classes as the full pairformer
module, so there must be exactly one `from_torch` registration per dotted path.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from jopendde.backend import AbstractFromTorch, Linear, LayerNorm, register_from_torch
from jopendde.features import Features
from jopendde.primitives import Transition
from jopendde.transformer import AttentionPairBias, broadcast_token_to_atom
from jopendde.triangular import TriangleAttention, TriangleMultiplicativeUpdate

Pytree = Any


def _one_hot_bins(
    x: Float[Array, "..."], lower_bins: Float[Array, "B"], upper_bins: Float[Array, "B"]
) -> Float[Array, "... B"]:
    """Bucket x into [lower_bins[b], upper_bins[b]) half-open bins (NOT a
    one-hot in the argmax sense -- if bins don't tile x's range exactly, zero
    or multiple bins can fire)."""
    return ((x[..., None] > lower_bins) & (x[..., None] < upper_bins)).astype(jnp.float32)


# ---------------------------------------------------------------------------
# PairformerBlock / PairformerStack (opendde.model.modules.pairformer) --
# only the `c_s > 0`, non-MSA, non-foldcp path used by ConfidenceHead.
# ---------------------------------------------------------------------------


@register_from_torch("opendde.model.modules.pairformer.PairformerBlock")
class PairformerBlock(AbstractFromTorch):
    tri_mul_out: TriangleMultiplicativeUpdate
    tri_mul_in: TriangleMultiplicativeUpdate
    tri_att_start: TriangleAttention
    tri_att_end: TriangleAttention
    pair_transition: Transition
    attention_pair_bias: AttentionPairBias | None
    single_transition: Transition | None
    c_s: int

    def __call__(
        self,
        s: Float[Array, "... N Cs"] | None,
        z: Float[Array, "... N N Cz"],
        pair_mask: Bool[Array, "... N N"] | None,
        extra_attn_bias: Float[Array, "..."] | None = None,
        **_ignored,  # perf/impl-select kwargs; unused
    ) -> tuple[Float[Array, "... N Cs"] | None, Float[Array, "... N N Cz"]]:
        z = z + self.tri_mul_out(z, mask=pair_mask)
        z = z + self.tri_mul_in(z, mask=pair_mask)
        z = z + self.tri_att_start(z, mask=pair_mask)
        z = jnp.swapaxes(z, -2, -3)
        pair_mask_t = None if pair_mask is None else jnp.swapaxes(pair_mask, -1, -2)
        z = z + self.tri_att_end(z, mask=pair_mask_t)
        z = jnp.swapaxes(z, -2, -3)
        z = z + self.pair_transition(z)
        if self.c_s > 0:
            s = s + self.attention_pair_bias(a=s, s=None, z=z, extra_attn_bias=extra_attn_bias)
            s = s + self.single_transition(s)
        return s, z


@register_from_torch("opendde.model.modules.pairformer.PairformerStack")
class PairformerStack(eqx.Module):
    """Homogeneous stack of `n_blocks` identical PairformerBlocks, run via
    `jax.lax.scan`. The scan body is wrapped in `jax.checkpoint` (per-block
    activation checkpointing): a no-op for inference, but under reverse-mode AD
    it rematerializes each block's activations instead of stacking all
    `n_blocks` at once, avoiding OOM on the design gradient."""

    stacked_params: PairformerBlock
    static: PairformerBlock
    n_blocks: int

    @classmethod
    def from_torch(cls, model):
        from jopendde.backend import from_torch, stack_blocks

        blocks = [from_torch(b) for b in model.blocks]
        stacked, static = stack_blocks(blocks)
        return cls(stacked_params=stacked, static=static, n_blocks=len(blocks))

    def __call__(
        self,
        s: Float[Array, "... N Cs"] | None,
        z: Float[Array, "... N N Cz"],
        pair_mask: Bool[Array, "... N N"] | None,
        extra_attn_bias: Float[Array, "..."] | None = None,
        **_ignored,  # perf/impl-select kwargs; unused
    ) -> tuple[Float[Array, "... N Cs"] | None, Float[Array, "... N N Cz"]]:
        @jax.checkpoint
        def body_fn(carry, params):
            s, z = carry
            block = eqx.combine(params, self.static)
            s, z = block(s, z, pair_mask, extra_attn_bias=extra_attn_bias)
            return (s, z), None

        (s, z), _ = jax.lax.scan(body_fn, (s, z), self.stacked_params)
        return s, z


# ---------------------------------------------------------------------------
# ConfidenceHead (Algorithm 31 in AF3)
# ---------------------------------------------------------------------------


@register_from_torch("opendde.model.modules.confidence.ConfidenceHead")
class ConfidenceHead(AbstractFromTorch):
    linear_no_bias_s1: Linear
    linear_no_bias_s2: Linear
    linear_no_bias_d: Linear
    linear_no_bias_d_wo_onehot: Linear
    pairformer_stack: PairformerStack
    linear_no_bias_pae: Linear
    linear_no_bias_pde: Linear
    plddt_weight: Float[Array, "A Cs Bplddt"]
    resolved_weight: Float[Array, "A Cs Bresolved"]
    input_strunk_ln: LayerNorm
    pae_ln: LayerNorm
    pde_ln: LayerNorm
    plddt_ln: LayerNorm
    resolved_ln: LayerNorm
    lower_bins: Float[Array, "B"]
    upper_bins: Float[Array, "B"]

    @staticmethod
    def _select_distogram_rep_atom_mask(
        feat: Features, n_token: int
    ) -> Bool[Array, "N_atom"]:
        distogram_mask = feat.distogram_rep_atom_mask.astype(bool)
        structural_mask = feat.structural_distogram_rep_atom_mask
        if structural_mask is None:
            return distogram_mask
        structural_mask = structural_mask.astype(bool)
        # Selecting on the structural mask's atom count is a data-dependent
        # branch that can't be a traced `if` under jit. `n_token` is static
        # (from a .shape), so select between the two same-shape masks with
        # jnp.where.
        use_structural = jnp.sum(structural_mask.astype(jnp.int32)) == n_token
        return jnp.where(use_structural, structural_mask, distogram_mask)

    def _memory_efficient_forward(
        self,
        s_trunk: Float[Array, "N Cs"],
        z_pair: Float[Array, "N N Cz"],
        pair_mask: Bool[Array, "N N"] | None,
        x_pred_rep_coords: Float[Array, "N 3"],
        atom_to_token_idx: Int[Array, "N_atom"],
        atom_to_tokatom_idx: Int[Array, "N_atom"],
        extra_attn_bias: Float[Array, "..."] | None,
    ):
        # Pairwise Euclidean distance via a double-`where` safe norm: `sqrt`
        # has an infinite derivative at 0, so the zero self-distance on the
        # diagonal (i==j) would backprop as 0/0 == NaN, poisoning every gradient
        # that reaches the confidence head. The value is unchanged (0 on the
        # diagonal, sqrt(u) off it) but the diagonal gradient becomes a finite 0.
        diff = x_pred_rep_coords[..., :, None, :] - x_pred_rep_coords[..., None, :, :]
        sq = jnp.sum(diff * diff, axis=-1)
        nonzero = sq > 0
        distance_pred = jnp.where(nonzero, jnp.sqrt(jnp.where(nonzero, sq, 1.0)), 0.0)

        z_pair = z_pair + self.linear_no_bias_d(
            _one_hot_bins(distance_pred, self.lower_bins, self.upper_bins)
        )
        z_pair = z_pair + self.linear_no_bias_d_wo_onehot(distance_pred[..., None])

        s_single, z_pair = self.pairformer_stack(
            s_trunk, z_pair, pair_mask, extra_attn_bias=extra_attn_bias
        )

        pae_pred = self.linear_no_bias_pae(self.pae_ln(z_pair))
        pde_pred = self.linear_no_bias_pde(
            self.pde_ln(z_pair + jnp.swapaxes(z_pair, -2, -3))
        )

        a = broadcast_token_to_atom(x_token=s_single, atom_to_token_idx=atom_to_token_idx)
        plddt_pred = jnp.einsum(
            "...nc,ncb->...nb", self.plddt_ln(a), self.plddt_weight[atom_to_tokatom_idx]
        )
        resolved_pred = jnp.einsum(
            "...nc,ncb->...nb", self.resolved_ln(a), self.resolved_weight[atom_to_tokatom_idx]
        )
        return plddt_pred, pae_pred, pde_pred, resolved_pred

    def __call__(
        self,
        feat: Features,
        s_inputs: Float[Array, "N Cs_in"],
        s_trunk: Float[Array, "N Cs"],
        z_trunk: Float[Array, "N N Cz"],
        pair_mask: Bool[Array, "N N"] | None,
        x_pred_coords: Float[Array, "N_sample N_atom 3"],
        **_ignored,  # z_trunk_spec / triangle_*_impl / inplace_safe / chunk_size: dropped
    ):
        s_trunk = self.input_strunk_ln(jnp.clip(s_trunk, -512, 512))

        n_token = s_trunk.shape[-2]
        rep_atom_mask = self._select_distogram_rep_atom_mask(feat, n_token)
        # The mask selects exactly n_token atoms (one representative per
        # token) -- boolean-mask indexing has a data-dependent output shape
        # under jit, so convert to a static-size index list instead
        # (`jnp.nonzero(..., size=n_token)` is JIT-safe: the count is fixed
        # by construction).
        rep_atom_idx = jnp.nonzero(rep_atom_mask, size=n_token)[0]
        x_pred_rep_coords = jnp.take(x_pred_coords, rep_atom_idx, axis=-2)

        z_init = (
            self.linear_no_bias_s1(s_inputs)[..., None, :, :]
            + self.linear_no_bias_s2(s_inputs)[..., None, :]
        )
        z_trunk = z_init + z_trunk

        atom_to_token_idx = feat.atom_to_token_idx.astype(jnp.int32)
        atom_to_tokatom_idx = feat.atom_to_tokatom_idx.astype(jnp.int32)
        extra_attn_bias = None  # confidence runs on the residue branch (no structural bias)

        def _per_sample(x_pred_rep_coords_i):
            return self._memory_efficient_forward(
                s_trunk,
                z_trunk,
                pair_mask,
                x_pred_rep_coords_i,
                atom_to_token_idx,
                atom_to_tokatom_idx,
                extra_attn_bias,
            )

        plddt_preds, pae_preds, pde_preds, resolved_preds = jax.vmap(_per_sample)(
            x_pred_rep_coords
        )
        return plddt_preds, pae_preds, pde_preds, resolved_preds
