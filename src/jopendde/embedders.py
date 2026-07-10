"""Fourier and relative-position embeddings."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from jopendde.backend import AbstractFromTorch, Linear, register_from_torch
from jopendde.features import Features
from jopendde.transformer import AtomAttentionEncoder


@register_from_torch("opendde.model.modules.embedders.FourierEmbedding")
class FourierEmbedding(AbstractFromTorch):
    """Algorithm 22 in AF3. Frozen random Fourier features `w`/`b`; the forward
    is an elementwise broadcast (cos of a scaled, shifted input), not a
    matmul."""

    w: Float[Array, "C"]
    b: Float[Array, "C"]

    def __call__(
        self, t_hat_noise_level: Float[Array, "... N"]
    ) -> Float[Array, "... N C"]:
        return jnp.cos(2 * jnp.pi * (t_hat_noise_level[..., None] * self.w + self.b))


@register_from_torch("opendde.model.modules.embedders.RelativePositionEncoding")
class RelativePositionEncoding(AbstractFromTorch):
    """Algorithm 3 in AF3. The forward is a linear projection of the
    precomputed dense relative-position feature grid (`relp`); the one-hot grid
    itself is built by `generate_relp`."""

    linear_no_bias: Linear
    r_max: int
    s_max: int

    def __call__(
        self, relp_feature: Float[Array, "... N N F"]
    ) -> Float[Array, "... N N Cz"]:
        return self.linear_no_bias(relp_feature)

    def generate_relp(
        self,
        asym_id: Float[Array, "... N"],
        residue_index: Float[Array, "... N"],
        entity_id: Float[Array, "... N"],
        token_index: Float[Array, "... N"],
        sym_id: Float[Array, "... N"],
    ) -> Float[Array, "... N N F"]:
        """Build the dense one-hot relative-position feature grid from
        token-index metadata (data prep, not a learned op). The residue branch's
        `relp` is shipped in the input dict; the structural-token branch
        regenerates it for the expanded token set (see the top-level model's
        `expand_to_structural_tokens`)."""
        r_max, s_max = self.r_max, self.s_max

        b_same_chain = (asym_id[..., :, None] == asym_id[..., None, :]).astype(jnp.int32)
        b_same_residue = (
            residue_index[..., :, None] == residue_index[..., None, :]
        ).astype(jnp.int32)
        b_same_entity = (entity_id[..., :, None] == entity_id[..., None, :]).astype(jnp.int32)

        d_residue = jnp.clip(
            residue_index[..., :, None] - residue_index[..., None, :] + r_max, 0, 2 * r_max
        ) * b_same_chain + (1 - b_same_chain) * (2 * r_max + 1)
        a_rel_pos = jax.nn.one_hot(d_residue.astype(jnp.int32), 2 * (r_max + 1))

        d_token = jnp.clip(
            token_index[..., :, None] - token_index[..., None, :] + r_max, 0, 2 * r_max
        ) * b_same_chain * b_same_residue + (1 - b_same_chain * b_same_residue) * (2 * r_max + 1)
        a_rel_token = jax.nn.one_hot(d_token.astype(jnp.int32), 2 * (r_max + 1))

        d_chain = jnp.clip(
            sym_id[..., :, None] - sym_id[..., None, :] + s_max, 0, 2 * s_max
        ) * b_same_entity + (1 - b_same_entity) * (2 * s_max + 1)
        a_rel_chain = jax.nn.one_hot(d_chain.astype(jnp.int32), 2 * (s_max + 1))

        return jnp.concatenate(
            [a_rel_pos, a_rel_token, b_same_entity[..., None].astype(a_rel_pos.dtype), a_rel_chain],
            axis=-1,
        ).astype(jnp.float32)


@register_from_torch("opendde.model.modules.embedders.InputFeatureEmbedder")
class InputFeatureEmbedder(AbstractFromTorch):
    """Algorithm 2 in AF3: embed per-atom features with the no-coords
    AtomAttentionEncoder, then concatenate the per-token features
    (restype/profile/deletion_mean) to form the 449-d `s_inputs`."""

    atom_attention_encoder: AtomAttentionEncoder
    c_atom: int
    c_atompair: int
    c_token: int

    def __call__(
        self, feat: Features, **_ignored
    ) -> Float[Array, "... N_token 449"]:
        # No-coords path: thread the (static) token count from the token-feature
        # shape so `aggregate_atom_to_token` doesn't fall back to the non-static
        # max(atom_to_token_idx)+1 under jit.
        n_token = feat.restype.shape[-2]
        a, _, _, _ = self.atom_attention_encoder(
            feat.atom_to_token_idx,
            feat.ref_pos,
            feat.ref_charge,
            feat.ref_mask,
            feat.ref_atom_name_chars,
            feat.ref_element,
            feat.d_lm,
            feat.v_lm,
            feat.pad_info,
            n_token=n_token,
        )
        batch_shape = feat.restype.shape[:-1]
        s_inputs = jnp.concatenate(
            [
                a,
                feat.restype.reshape(*batch_shape, 32),
                feat.profile.reshape(*batch_shape, 32),
                feat.deletion_mean.reshape(*batch_shape, 1),
            ],
            axis=-1,
        )
        return s_inputs
