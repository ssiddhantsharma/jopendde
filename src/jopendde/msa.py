"""MSA module and template embedder: MSAPairWeightedAveraging, MSAStack,
MSABlock, MSAModule, TemplateEmbedder.

Registrations use dotted-path strings resolved lazily (see
backend.register_from_torch). PairformerBlock/PairformerStack are reused from
jopendde.confidence (one from_torch registration each). FoldCP paths and
inplace_safe/chunk_size memory optimizations are not implemented.

The MSAModule output `z` is invariant to MSA-row order (OuterProductMean
averages over rows; the weighted-average / transition are per-row), so rows are
selected valid-first without shuffling. When `msa_depth < n_valid_rows`, the
first `msa_depth` valid rows are used.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from jopendde.backend import AbstractFromTorch, Linear, LayerNorm, from_torch, register_from_torch, stack_blocks
from jopendde.confidence import PairformerBlock, PairformerStack
from jopendde.features import Features, MSAFeatures
from jopendde.primitives import Transition
from jopendde.triangular import OuterProductMean

_NUM_RESTYPE_WITH_GAP = 32  # residue types including gap


# ---------------------------------------------------------------------------
# MSAPairWeightedAveraging (Algorithm 10 in AF3)
# ---------------------------------------------------------------------------


@register_from_torch("opendde.model.modules.pairformer.MSAPairWeightedAveraging")
class MSAPairWeightedAveraging(AbstractFromTorch):
    layernorm_m: LayerNorm
    linear_no_bias_mv: Linear
    layernorm_z: LayerNorm
    linear_no_bias_z: Linear
    linear_no_bias_mg: Linear
    linear_no_bias_out: Linear
    softmax_w: Any  # softmax over axis -2
    c: int
    n_heads: int

    def __call__(
        self, m: Float[Array, "... S N Cm"], z: Float[Array, "... N N Cz"]
    ) -> Float[Array, "... S N Cm"]:
        m = self.layernorm_m(m)
        v = self.linear_no_bias_mv(m)
        v = v.reshape(v.shape[:-1] + (self.n_heads, self.c))  # [..., S, N, H, c]
        g = jax.nn.sigmoid(self.linear_no_bias_mg(m))
        g = g.reshape(g.shape[:-1] + (self.n_heads, self.c))  # [..., S, N, H, c]
        b = self.linear_no_bias_z(self.layernorm_z(z))  # [..., i, j, H]
        w = self.softmax_w(b)  # softmax over j (axis=-2)
        wv = jnp.einsum("...ijh,...mjhc->...mihc", w, v)  # [..., S, N, H, c]
        o = g * wv
        o = o.reshape(o.shape[:-2] + (self.n_heads * self.c,))
        return self.linear_no_bias_out(o)


# ---------------------------------------------------------------------------
# MSAStack (Algorithm 8, lines 7-8)
# ---------------------------------------------------------------------------


@register_from_torch("opendde.model.modules.pairformer.MSAStack")
class MSAStack(AbstractFromTorch):
    msa_pair_weighted_averaging: MSAPairWeightedAveraging
    transition_m: Transition
    msa_chunk_size: int | None  # unused: msa-row chunking

    def __call__(
        self, m: Float[Array, "... S N Cm"], z: Float[Array, "... N N Cz"]
    ) -> Float[Array, "... S N Cm"]:
        m = m + self.msa_pair_weighted_averaging(m, z)
        m = m + self.transition_m(m)
        return m


# ---------------------------------------------------------------------------
# MSABlock: MSA is updated before OuterProductMean writes back to z
# ---------------------------------------------------------------------------


@register_from_torch("opendde.model.modules.pairformer.MSABlock")
class MSABlock(AbstractFromTorch):
    msa_stack: MSAStack
    outer_product_mean_msa: OuterProductMean
    pair_stack: PairformerBlock
    c_m: int
    c_z: int
    c_hidden: int
    is_last_block: bool  # m is always returned; MSAModule discards the final m

    def __call__(
        self,
        m: Float[Array, "... S N Cm"],
        z: Float[Array, "... N N Cz"],
        pair_mask: Bool[Array, "... N N"] | None = None,
        **_ignored,  # perf/impl-select kwargs; unused
    ) -> tuple[Float[Array, "... S N Cm"], Float[Array, "... N N Cz"]]:
        m = self.msa_stack(m, z)
        z = z + self.outer_product_mean_msa(m)
        _, z = self.pair_stack(s=None, z=z, pair_mask=pair_mask)
        return m, z


# ---------------------------------------------------------------------------
# MSAModule
# ---------------------------------------------------------------------------


@register_from_torch("opendde.model.modules.pairformer.MSAModule")
class MSAModule(eqx.Module):
    linear_no_bias_m: Linear
    linear_no_bias_s: Linear
    stacked_blocks: MSABlock
    static: MSABlock
    n_blocks: int
    msa_depth: int
    msa_dim: int  # one-hot classes (= 32)
    gap_token: int

    @classmethod
    def from_torch(cls, model):
        blocks = [from_torch(b) for b in model.blocks]
        stacked, static = stack_blocks(blocks)
        return cls(
            linear_no_bias_m=from_torch(model.linear_no_bias_m),
            linear_no_bias_s=from_torch(model.linear_no_bias_s),
            stacked_blocks=stacked,
            static=static,
            n_blocks=len(blocks),
            msa_depth=int(model.msa_depth),
            msa_dim=int(model.input_feature["msa"]),
            gap_token=int(model.input_feature["msa"]) - 1,
        )

    def _prepare_msa_sample(
        self, msa_feats: MSAFeatures, s_inputs: Float[Array, "N Cs_inputs"]
    ) -> Float[Array, "S N Cm"]:
        msa = msa_feats.msa
        assert msa.ndim == 2, "only the unbatched [S, N] msa case is used here"
        s_len = msa.shape[-2]
        num_msa = max(0, min(self.msa_depth, s_len))

        msa_i = msa.astype(jnp.int32)
        # valid-first selection (stable, no shuffle -- see module docstring).
        row_valid = jnp.any(msa_i != self.gap_token, axis=-1)  # [S]
        order = jnp.argsort(jnp.where(row_valid, 0, 1), stable=True)  # valid rows first
        sel = order[:num_msa]

        msa_i = msa_i[sel]
        has_deletion = msa_feats.has_deletion[sel]
        deletion_value = msa_feats.deletion_value[sel]

        msa_oh = jax.nn.one_hot(msa_i, self.msa_dim, dtype=s_inputs.dtype)  # [S, N, 32]
        # Row 0 (the query) is always valid, so the stable valid-first sort
        # keeps it at output position 0. Take it from the soft `query_soft`
        # feature so a caller can inject a differentiable PSSM.
        msa_oh = msa_oh.at[0].set(msa_feats.query_soft.astype(s_inputs.dtype))
        target = msa_oh.shape[:-1]  # [S, N]
        msa_sample = jnp.concatenate(
            [
                msa_oh,
                has_deletion.reshape(target + (1,)).astype(s_inputs.dtype),
                deletion_value.reshape(target + (1,)).astype(s_inputs.dtype),
            ],
            axis=-1,
        )  # [S, N, 34]
        msa_sample = self.linear_no_bias_m(msa_sample)
        return msa_sample + self.linear_no_bias_s(s_inputs)  # broadcast [S,N,Cm]+[N,Cm]

    def __call__(
        self,
        feat: Features,
        z: Float[Array, "N N Cz"],
        s_inputs: Float[Array, "N Cs_inputs"],
        pair_mask: Bool[Array, "N N"] | None = None,
        **_ignored,  # perf/impl-select kwargs; unused
    ) -> Float[Array, "N N Cz"]:
        if self.n_blocks < 1 or feat.msa is None:
            return z
        m = self._prepare_msa_sample(feat.msa, s_inputs)

        # Per-block activation checkpointing (see PairformerStack): a no-op for
        # inference, but keeps reverse-mode AD from stacking every MSA block's
        # activations at once.
        @jax.checkpoint
        def body_fn(carry, params):
            m, z = carry
            block = eqx.combine(params, self.static)
            m, z = block(m, z, pair_mask)
            return (m, z), None

        (m, z), _ = jax.lax.scan(body_fn, (m, z), self.stacked_blocks)
        return z


# ---------------------------------------------------------------------------
# TemplateEmbedder (Algorithm 16 in AF3)
# ---------------------------------------------------------------------------


@register_from_torch("opendde.model.modules.pairformer.TemplateEmbedder")
class TemplateEmbedder(AbstractFromTorch):
    linear_no_bias_z: Linear
    layernorm_z: LayerNorm
    linear_no_bias_a: Linear
    pairformer_stack: PairformerStack
    layernorm_v: LayerNorm
    linear_no_bias_u: Linear
    relu: Any
    n_blocks: int

    def _single_template(
        self,
        template_id: int,
        feat: Features,
        z: Float[Array, "N N C"],
        pair_mask: Float[Array, "N N"],
        multichain_mask: Float[Array, "N N"],
    ) -> Float[Array, "N N C"]:
        n = z.shape[0]
        m2 = multichain_mask[..., None] * pair_mask[..., None]

        t = feat.template
        dgram = t.distogram[template_id] * m2  # [N, N, 39]
        pseudo_beta_mask = t.pseudo_beta_mask[template_id] * multichain_mask * pair_mask
        aatype = t.aatype[template_id].astype(jnp.int32)  # [N]
        aatype = jax.nn.one_hot(aatype, _NUM_RESTYPE_WITH_GAP, dtype=z.dtype)  # [N, 32]
        # broadcast over i -> [N, N, 32]
        aatype_j = jnp.broadcast_to(aatype[None, :, :], (n, n, aatype.shape[-1]))
        # broadcast over j -> [N, N, 32]
        aatype_i = jnp.broadcast_to(aatype[:, None, :], (n, n, aatype.shape[-1]))
        unit_vector = t.unit_vector[template_id] * m2  # [N, N, 3]
        backbone_mask = t.backbone_frame_mask[template_id] * multichain_mask * pair_mask

        at = jnp.concatenate(
            [
                dgram,
                pseudo_beta_mask[..., None],
                aatype_j,
                aatype_i,
                unit_vector,
                backbone_mask[..., None],
            ],
            axis=-1,
        )
        v = self.linear_no_bias_z(z) + self.linear_no_bias_a(at)
        _, v = self.pairformer_stack(s=None, z=v, pair_mask=pair_mask)
        return self.layernorm_v(v)

    def __call__(
        self,
        feat: Features,
        z: Float[Array, "N N Cz"],
        pair_mask: Float[Array, "N N"] | None = None,
        **_ignored,  # perf/impl-select kwargs; unused
    ) -> Float[Array, "N N Cz"] | int:
        if feat.template is None or self.n_blocks < 1:
            return 0  # no templates: additive identity

        asym_id = feat.asym_id
        multichain_mask = (asym_id[:, None] == asym_id[None, :]).astype(z.dtype)
        num_templates = feat.template.aatype.shape[0]  # static

        if pair_mask is None:
            pair_mask = jnp.ones(z.shape[:-1], dtype=z.dtype)

        z = self.layernorm_z(z)
        u = 0
        for template_id in range(num_templates):
            u = u + self._single_template(
                template_id, feat, z, pair_mask, multichain_mask
            )
        u = u / (1e-7 + num_templates)
        return self.linear_no_bias_u(self.relu(u))
