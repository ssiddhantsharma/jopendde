"""StructuralTokenExpander.

Expands residue-level trunk activations (`s_inputs_res`, `s_res`, `z_res`,
one row/col per *residue*) into structural-token-level activations (one
row/col per structural *sub-token* -- e.g. protein backbone vs sidechain,
nucleotide backbone vs base). Every structural token has a `parent` residue
index and a `role` id (`STRUCTURAL_TOKEN_ROLES`:
atom=0, protein_bb=1, protein_sc=2, dna_bb=3, dna_base=4, rna_bb=5,
rna_base=6). The module:

1. Gathers each structural token's parent-residue single/pair activations
   (plain `index_select`/take -- no learned mixing across residues).
2. Adds role-specific single conditioning: an `Embedding(n_roles, ·)` lookup
   by `role`, plus (for `s`) a residual MLP (`single_split_mlp` --
   LayerNorm -> LinearNoBias(c_s,2c_s) -> SiLU -> LinearNoBias(2c_s,c_s),
   zero-init on the output projection so it starts as a no-op).
3. Adds pair conditioning in two independent pieces:
   - **`pair_project_by_role`**: a per-(row-role, col-role) *learned linear
     reprojection* of the gathered pair features. Only the "full" mode is
     implemented (from_torch asserts it): one `LinearNoBias(c_z, c_z)` per
     (role_i, role_j) combination (`n_roles**2` blocks, zero-initialized),
     selected by role. See `_pair_project_by_role_full` below.
   - **`pair_init_bias`**: a sum of 5 independent categorical-embedding
     lookups (each `Embedding(2 or 8, c_z)`, zero-init'd unless
     `init_mode="scratch"`): `same_parent_residue` (boolean: do the row/col
     tokens share a parent residue?), `same_residue_twin` (boolean: is one
     token's role a backbone and the other's a sidechain/base of the *same*
     parent -- i.e. are they the same residue's "twin" sub-tokens?),
     `prev_bb_chain`/`next_bb_chain` (boolean: are both tokens backbone and
     is one's parent the chain-adjacent (prev/next) residue of the other's?),
     and `role_pair_type` (8-way categorical: an explicit backbone/sidechain/
     base x backbone/sidechain/base relation code, default "other"=7).
4. Separately produces `structural_pair_attn_bias` -- NOT added into
   `z_struct` itself, but returned in the output dict for a downstream
   attention module (e.g. a structural-token Pairformer refiner) to consume
   as an extra additive attention bias. It's a *learned scalar* (not per-
   channel) weighted sum of the same 4 boolean pair features
   (`attn_bias_same_parent`, `attn_bias_same_residue_twin`,
   `attn_bias_prev_bb_chain`, `attn_bias_next_bb_chain` -- each a bare
   `nn.Parameter(())` scalar) plus an 8-way `attn_bias_role_pair_type`
   lookup (one scalar per `role_pair_type` category, broadcast over heads
   downstream).

Only the non-chunked path (the whole n_struct x n_struct pair grid is
materialized in one shot) is implemented; the FoldCP and `pair_chunk_size`-
chunked variants are not.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from jopendde.backend import (
    AbstractFromTorch,
    Embedding,
    Linear,
    Sequential,
    register_from_torch,
)
from jopendde.features import Features


def _isin_any(role: Int[Array, "N"], ids: tuple[int, ...]) -> Bool[Array, "N"]:
    out = role == ids[0]
    for i in ids[1:]:
        out = out | (role == i)
    return out


class _PairFeatures(eqx.Module):
    """Per-(row, col) boolean/categorical pair features the expander derives
    once and consumes in both the pair-init-bias and attention-bias heads."""

    same_parent_residue: Bool[Array, "N N"]
    same_residue_twin: Bool[Array, "N N"]
    prev_bb_chain: Bool[Array, "N N"]
    next_bb_chain: Bool[Array, "N N"]
    role_pair_type: Int[Array, "N N"]


@register_from_torch("opendde.model.modules.structural_tokens.StructuralTokenExpander")
class StructuralTokenExpander(AbstractFromTorch):
    single_split_mlp: Sequential
    single_input_role_embedding: Embedding
    single_role_embedding: Embedding
    # "full" pair_projection_mode: n_roles**2 zero-init LinearNoBias(c_z, c_z)
    # blocks, one per (role_i, role_j) combination.
    pair_block_proj: list[Linear]
    same_parent_embedding: Embedding
    same_residue_twin_embedding: Embedding
    prev_bb_chain_embedding: Embedding
    next_bb_chain_embedding: Embedding
    role_pair_type_embedding: Embedding
    attn_bias_same_parent: Float[Array, ""]
    attn_bias_same_residue_twin: Float[Array, ""]
    attn_bias_prev_bb_chain: Float[Array, ""]
    attn_bias_next_bb_chain: Float[Array, ""]
    attn_bias_role_pair_type: Float[Array, "8"]

    c_z: int
    n_roles: int
    backbone_role_ids: tuple[int, ...]
    sidechain_role_id: int
    base_role_ids: tuple[int, ...]

    # Only "full" pair projection is supported; from_torch asserts it.
    @classmethod
    def from_torch(cls, model):
        assert str(model.pair_projection_mode) == "full", (
            "jopendde hardcodes StructuralTokenExpander pair_projection_mode='full' "
            f"(the opendde_v1 mode); got {model.pair_projection_mode!r}"
        )
        return super().from_torch(model)

    def _pair_project_by_role_full(
        self, z: Float[Array, "... N N Cz"], role: Int[Array, "N"]
    ) -> Float[Array, "... N N Cz"]:
        # Stack all n_roles**2 LinearNoBias(c_z, c_z) weight matrices
        # (shape [Out, In] each, per backend.Linear convention) and gather the
        # one matching each (row_role, col_role) pair; avoids dynamic-shape
        # boolean indexing.
        stacked_weight = jnp.stack(
            [lin.weight for lin in self.pair_block_proj], axis=0
        )  # [n_roles*n_roles, Cz_out, Cz_in]
        role_pair_idx = role[:, None] * self.n_roles + role[None, :]  # [N, N]
        w = stacked_weight[role_pair_idx]  # [N, N, Cz_out, Cz_in]
        return jnp.einsum("...ijk,ijok->...ijo", z, w)

    def _build_pair_features(
        self,
        feat: Features,
        role: Int[Array, "N"],
        parent: Int[Array, "N"],
    ):
        n_struct = role.shape[-1]
        asym_id = jnp.take(feat.asym_id, parent, axis=-1)

        is_backbone = _isin_any(role, self.backbone_role_ids)
        is_sidechain = role == self.sidechain_role_id
        is_base = _isin_any(role, self.base_role_ids)

        prev_parent = feat.prev_parent_residue_idx
        next_parent = feat.next_parent_residue_idx
        if prev_parent is None:
            prev_parent = jnp.full((n_struct,), -1, dtype=parent.dtype)
        else:
            prev_parent = prev_parent.astype(parent.dtype)
        if next_parent is None:
            next_parent = jnp.full((n_struct,), -1, dtype=parent.dtype)
        else:
            next_parent = next_parent.astype(parent.dtype)

        same_parent_residue = parent[:, None] == parent[None, :]
        same_chain = asym_id[:, None] == asym_id[None, :]
        same_residue_twin = same_parent_residue & (
            (is_backbone[:, None] & (is_sidechain[None, :] | is_base[None, :]))
            | (is_backbone[None, :] & (is_sidechain[:, None] | is_base[:, None]))
        )
        prev_bb_chain = (
            is_backbone[:, None]
            & is_backbone[None, :]
            & same_chain
            & (prev_parent[:, None] == parent[None, :])
        )
        next_bb_chain = (
            is_backbone[:, None]
            & is_backbone[None, :]
            & same_chain
            & (next_parent[:, None] == parent[None, :])
        )

        bb_i, bb_j = is_backbone[:, None], is_backbone[None, :]
        sc_i, sc_j = is_sidechain[:, None], is_sidechain[None, :]
        base_i, base_j = is_base[:, None], is_base[None, :]
        role_pair_type = jnp.where(
            bb_i & bb_j,
            0,
            jnp.where(
                bb_i & sc_j,
                1,
                jnp.where(
                    sc_i & bb_j,
                    2,
                    jnp.where(
                        sc_i & sc_j,
                        3,
                        jnp.where(
                            bb_i & base_j,
                            4,
                            jnp.where(base_i & bb_j, 5, jnp.where(base_i & base_j, 6, 7)),
                        ),
                    ),
                ),
            ),
        )

        return _PairFeatures(
            same_parent_residue=same_parent_residue,
            same_residue_twin=same_residue_twin,
            prev_bb_chain=prev_bb_chain,
            next_bb_chain=next_bb_chain,
            role_pair_type=role_pair_type,
        )

    def _make_pair_init_bias(self, pf: _PairFeatures, dtype) -> Float[Array, "N N Cz"]:
        pair_bias = self.same_parent_embedding(
            pf.same_parent_residue.astype(jnp.int32)
        ).astype(dtype)
        pair_bias = pair_bias + self.same_residue_twin_embedding(
            pf.same_residue_twin.astype(jnp.int32)
        ).astype(dtype)
        pair_bias = pair_bias + self.prev_bb_chain_embedding(
            pf.prev_bb_chain.astype(jnp.int32)
        ).astype(dtype)
        pair_bias = pair_bias + self.next_bb_chain_embedding(
            pf.next_bb_chain.astype(jnp.int32)
        ).astype(dtype)
        return pair_bias + self.role_pair_type_embedding(pf.role_pair_type).astype(dtype)

    def _make_attention_bias(self, pf: _PairFeatures, dtype) -> Float[Array, "N N"]:
        role_pair_bias = self.attn_bias_role_pair_type[pf.role_pair_type].astype(dtype)
        return (
            self.attn_bias_same_parent.astype(dtype) * pf.same_parent_residue.astype(dtype)
            + self.attn_bias_same_residue_twin.astype(dtype) * pf.same_residue_twin.astype(dtype)
            + self.attn_bias_prev_bb_chain.astype(dtype) * pf.prev_bb_chain.astype(dtype)
            + self.attn_bias_next_bb_chain.astype(dtype) * pf.next_bb_chain.astype(dtype)
            + role_pair_bias
        )

    def __call__(
        self,
        feat: Features,
        s_inputs_res: Float[Array, "... N Cs_in"],
        s_res: Float[Array, "... N Cs"],
        z_res: Float[Array, "... N N Cz"],
    ) -> tuple[
        Float[Array, "... M Cs_in"],
        Float[Array, "... M Cs"],
        Float[Array, "... M M Cz"],
        Float[Array, "... M M"],
    ]:
        parent = feat.parent_residue_idx.astype(jnp.int32)
        role = feat.subtoken_role_id.astype(jnp.int32)

        s_inputs_struct = jnp.take(
            s_inputs_res, parent, axis=-2
        ) + self.single_input_role_embedding(role).astype(s_inputs_res.dtype)
        s_parent = jnp.take(s_res, parent, axis=-2)
        s_struct = (
            s_parent
            + self.single_split_mlp(s_parent)
            + self.single_role_embedding(role).astype(s_parent.dtype)
        )

        pf = self._build_pair_features(feat, role, parent)

        z_row = jnp.take(z_res, parent, axis=-3)
        z_parent = jnp.take(z_row, parent, axis=-2)

        # pair_projection_mode is always "full" (asserted at load).
        z_struct = z_parent + self._pair_project_by_role_full(z_parent, role)
        z_struct = z_struct + self._make_pair_init_bias(pf, dtype=z_parent.dtype)

        # The attention bias is the expander's only downstream output (a
        # Pairformer refiner + diffusion consume it as an additive attn bias);
        # return it directly.
        attn_bias = self._make_attention_bias(pf, dtype=z_parent.dtype)
        return s_inputs_struct, s_struct, z_struct, attn_bias
