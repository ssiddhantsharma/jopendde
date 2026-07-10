"""Diffusion transformer and atom-level attention.

Registrations use dotted-path strings resolved lazily (see
backend.register_from_torch). FoldCP paths and the inplace_safe/chunk_size
memory-optimization arguments are not implemented; only the single-device
plain path is present.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int

from jopendde.backend import (
    AbstractFromTorch,
    Identity,
    Linear,
    LayerNorm,
    from_torch,
    register_from_torch,
    stack_blocks,
)
from jopendde.features import DenseTrunkPad
from jopendde.primitives import AdaptiveLayerNorm, Transition

Pytree = Any


# ---------------------------------------------------------------------------
# Windowed local-attention indexing.
#
# Overlapping sliding windows: the query is padded on the right only; keys are
# padded on both sides so each window centers on its query block. n_queries,
# n_keys, and the sequence length are static, so the gather indices (and the
# padding mask) are precomputed with numpy and applied via jnp.take -- no
# runtime padding, masking, or dynamic shapes.
# ---------------------------------------------------------------------------


def _dense_trunk_geometry(n: int, n_queries: int, n_keys: int):
    assert n_keys >= n_queries
    assert n_queries % 2 == 0
    assert n_keys % 2 == 0
    n_trunks = -(-n // n_queries)  # ceil division, static python ints
    q_pad_length = n_trunks * n_queries - n
    pad_left = (n_keys - n_queries) // 2
    return n_trunks, q_pad_length, pad_left


def _window_query(x, dim: int, n_trunks: int, n_queries: int, q_pad_length: int):
    """Pad (zeros, right only) + reshape one axis into (n_trunks, n_queries)."""

    dim = dim % x.ndim
    if q_pad_length > 0:
        pad_width = [(0, 0)] * x.ndim
        pad_width[dim] = (0, q_pad_length)
        x = jnp.pad(x, pad_width)
    new_shape = x.shape[:dim] + (n_trunks, n_queries) + x.shape[dim + 1 :]
    return x.reshape(new_shape)


def _key_window_indices(n: int, n_trunks: int, n_queries: int, n_keys: int, pad_left: int):
    """Static (numpy) gather indices + validity mask for overlapping key windows.

    raw[trunk, k_local] = trunk * n_queries + k_local - pad_left is the source
    position in the unpadded key tensor.
    """

    trunk_idx = np.arange(n_trunks)[:, None]
    key_idx = np.arange(n_keys)[None, :]
    raw = trunk_idx * n_queries + key_idx - pad_left  # [n_trunks, n_keys]
    valid = (raw >= 0) & (raw < n)
    clipped = np.clip(raw, 0, max(n - 1, 0))
    return clipped, valid


def _window_key(x, dim: int, clipped_idx: np.ndarray, valid: np.ndarray):
    dim = dim % x.ndim
    n_trunks, n_keys = clipped_idx.shape
    gathered = jnp.take(x, jnp.asarray(clipped_idx.reshape(-1)), axis=dim)
    new_shape = gathered.shape[:dim] + (n_trunks, n_keys) + gathered.shape[dim + 1 :]
    gathered = gathered.reshape(new_shape)
    valid_shape = (1,) * dim + (n_trunks, n_keys) + (1,) * (gathered.ndim - dim - 2)
    gathered = jnp.where(jnp.asarray(valid.reshape(valid_shape)), gathered, 0)
    return gathered


def rearrange_qk_to_dense_trunk(
    qs, ks, dim_qs, dim_ks, *, n_queries: int, n_keys: int, compute_mask: bool
):
    """Build overlapping windowed query/key trunks.

    `qs`/`ks` are lists of arrays sharing one window geometry; `dim_qs`/`dim_ks`
    are their per-array axes (also lists). Returns (qs_trunked, ks_trunked,
    pad_info) with the trunked outputs as lists and pad_info a `DenseTrunkPad`
    whose `mask_trunked` is [n_trunks, n_queries, n_keys] (bool) when
    compute_mask, else None.
    """

    n = qs[0].shape[dim_qs[0] % qs[0].ndim]
    n_trunks, q_pad_length, pad_left = _dense_trunk_geometry(n, n_queries, n_keys)

    qs_trunked = [
        _window_query(x, d, n_trunks, n_queries, q_pad_length) for x, d in zip(qs, dim_qs)
    ]

    clipped_idx, valid = _key_window_indices(n, n_trunks, n_queries, n_keys, pad_left)
    ks_trunked = [_window_key(x, d, clipped_idx, valid) for x, d in zip(ks, dim_ks)]

    mask_trunked = None
    if compute_mask:
        q_idx = np.arange(n_trunks)[:, None] * n_queries + np.arange(n_queries)[None, :]
        q_valid = q_idx < n  # [n_trunks, n_queries]
        mask_np = q_valid[:, :, None] & valid[:, None, :]
        mask_trunked = jnp.asarray(mask_np)

    pad_info = DenseTrunkPad(
        mask_trunked=mask_trunked, q_pad=int(q_pad_length), k_pad_left=int(pad_left)
    )
    return qs_trunked, ks_trunked, pad_info


def _unwindow(x, dim: int, q_pad_length: int):
    """Inverse of `_window_query`: merge (n_trunks, n_queries) at `dim`,`dim+1`
    back into one axis, dropping the right-padding."""

    dim = dim % x.ndim
    shape = x.shape
    merged = shape[dim] * shape[dim + 1]
    x = x.reshape(shape[:dim] + (merged,) + shape[dim + 2 :])
    if q_pad_length > 0:
        sl = [slice(None)] * x.ndim
        sl[dim] = slice(0, merged - q_pad_length)
        x = x[tuple(sl)]
    return x


def gather_pair_embedding_in_dense_trunk(
    x: Float[Array, "... N N D"], idx_q: Int[Array, "Nb Nq"], idx_k: Int[Array, "Nb Nk"]
) -> Float[Array, "... Nb Nq Nk D"]:
    """y[..., b, i, j, :] = x[..., idx_q[b, i], idx_k[b, j], :]"""

    idx_q = idx_q.astype(jnp.int32)
    idx_k = idx_k.astype(jnp.int32)
    idx_q_e = idx_q[:, :, None]
    idx_k_e = idx_k[:, None, :]
    return x[..., idx_q_e, idx_k_e, :]


def broadcast_token_to_atom(
    x_token: Float[Array, "... N_token D"], atom_to_token_idx: Int[Array, "..."]
) -> Float[Array, "... N_atom D"]:
    atom_to_token_idx = atom_to_token_idx.astype(jnp.int32)
    if atom_to_token_idx.ndim == 1:
        return x_token[..., atom_to_token_idx, :]
    idx = atom_to_token_idx[..., :, None]
    idx = jnp.broadcast_to(idx, idx.shape[:-1] + (x_token.shape[-1],))
    return jnp.take_along_axis(x_token, idx, axis=-2)


def aggregate_atom_to_token(
    x_atom: Float[Array, "... N_atom D"],
    atom_to_token_idx: Int[Array, "N_atom"],
    n_token: int | None,
    reduce: str = "mean",
) -> Float[Array, "... N_token D"]:
    assert reduce == "mean"
    assert atom_to_token_idx.ndim == 1, "only the unbatched atom_to_token_idx case is used here"
    # When n_token is not supplied (no-coords encoder path), infer it from the
    # indices; this needs a concrete atom_to_token_idx, so callers under jit
    # thread in a static n_token instead.
    if n_token is None:
        n_token = int(jnp.max(atom_to_token_idx)) + 1
    atom_to_token_idx = atom_to_token_idx.astype(jnp.int32)
    d = x_atom.shape[-1]
    out_shape = x_atom.shape[:-2] + (n_token, d)
    sums = jnp.zeros(out_shape, dtype=x_atom.dtype).at[..., atom_to_token_idx, :].add(x_atom)
    counts = jnp.zeros((n_token,), dtype=x_atom.dtype).at[atom_to_token_idx].add(1.0)
    counts = jnp.clip(counts, 1.0, None)
    return sums / counts[:, None]


# ---------------------------------------------------------------------------
# Attention primitive, including the windowed/local-attention path
# (n_queries/n_keys). Not registered via register_from_torch: the shared
# dotted path resolves to triangular.Attention (non-windowed only), so
# AttentionPairBias converts its attention child via
# _WindowedAttention.from_torch directly.
# ---------------------------------------------------------------------------


class _WindowedAttention(AbstractFromTorch):
    linear_q: Linear
    linear_k: Linear
    linear_v: Linear
    linear_o: Linear
    linear_g: Linear | None
    sigmoid: Any | None
    num_heads: int
    c_hidden: int

    def _prep_qkv(self, q_x, kv_x):
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)
        H = self.num_heads
        q = q.reshape(q.shape[:-1] + (H, -1)) / jnp.sqrt(jnp.array(self.c_hidden, dtype=q.dtype))
        k = k.reshape(k.shape[:-1] + (H, -1))
        v = v.reshape(v.shape[:-1] + (H, -1))
        return q, k, v

    def _wrap_up(self, o, q_x):
        if self.linear_g is not None:
            g = jax.nn.sigmoid(self.linear_g(q_x))
            g = g.reshape(g.shape[:-1] + (self.num_heads, -1))
            o = o * g
        o = o.reshape(o.shape[:-2] + (-1,))
        return self.linear_o(o)

    def __call__(
        self,
        q_x: Float[Array, "... Q Cq"],
        kv_x: Float[Array, "... K Ck"],
        attn_bias: Float[Array, "..."] | None = None,
        n_queries: int | None = None,
        n_keys: int | None = None,
    ) -> Float[Array, "... Q Cq"]:
        q, k, v = self._prep_qkv(q_x, kv_x)  # [..., Q/K, H, Dh]

        if n_queries and n_keys:
            (q_blocks,), (k_blocks, v_blocks), pad_info = rearrange_qk_to_dense_trunk(
                [q], [k, v], [-3], [-3, -3], n_queries=n_queries, n_keys=n_keys, compute_mask=True
            )
            attn = jnp.einsum("...qhd,...khd->...hqk", q_blocks, k_blocks)
            mask = pad_info.mask_trunked  # [n_trunks, Q, K] bool
            attn = attn + jnp.where(mask[:, None, :, :], 0.0, -1e10)
            if attn_bias is not None:
                attn = attn + attn_bias
            attn = jax.nn.softmax(attn, axis=-1)
            o = jnp.einsum("...hqk,...khd->...qhd", attn, v_blocks)  # [..., n_trunks, Q, H, Dh]
            o = _unwindow(o, dim=-4, q_pad_length=pad_info.q_pad)
        else:
            attn = jnp.einsum("...qhd,...khd->...hqk", q, k)
            if attn_bias is not None:
                attn = attn + attn_bias
            attn = jax.nn.softmax(attn, axis=-1)
            o = jnp.einsum("...hqk,...khd->...qhd", attn, v)

        return self._wrap_up(o, q_x)


# ---------------------------------------------------------------------------
# AttentionPairBias (Algorithm 24 in AF3)
# ---------------------------------------------------------------------------


@register_from_torch("opendde.model.modules.transformer.AttentionPairBias")
class AttentionPairBias(AbstractFromTorch):
    layernorm_a: Any
    layernorm_kv: Any | None
    attention: _WindowedAttention
    layernorm_z: LayerNorm
    linear_nobias_z: Linear
    linear_a_last: Linear | None
    has_s: bool
    cross_attention_mode: bool

    @classmethod
    def from_torch(cls, model):
        # `attention` is converted via _WindowedAttention.from_torch directly,
        # since the shared dotted path resolves to triangular.Attention
        # (non-windowed only).
        return cls(
            layernorm_a=from_torch(model.layernorm_a),
            layernorm_kv=from_torch(model.layernorm_kv) if model.cross_attention_mode else None,
            attention=_WindowedAttention.from_torch(model.attention),
            layernorm_z=from_torch(model.layernorm_z),
            linear_nobias_z=from_torch(model.linear_nobias_z),
            linear_a_last=from_torch(model.linear_a_last) if model.has_s else None,
            has_s=model.has_s,
            cross_attention_mode=model.cross_attention_mode,
        )

    @staticmethod
    def _pair_bias(layernorm_z, linear_nobias_z, z):
        # [..., (n_blocks,) Q, K, C_z] -> [..., (n_blocks,) H, Q, K]
        return jnp.moveaxis(linear_nobias_z(layernorm_z(z)), -1, -3)

    def __call__(
        self,
        a: Float[Array, "... N Ca"],
        s: Float[Array, "... N Cs"],
        z: Float[Array, "..."],
        n_queries: int | None = None,
        n_keys: int | None = None,
        extra_attn_bias: Float[Array, "..."] | None = None,
        pair_bias: Float[Array, "..."] | None = None,
        **_ignored,  # perf-only kwargs; unused
    ) -> Float[Array, "... N Ca"]:
        if self.has_s:
            a_normed = self.layernorm_a(a, s)
        else:
            a_normed = self.layernorm_a(a)

        if self.cross_attention_mode:
            if self.has_s:
                kv = self.layernorm_kv(a_normed, s)
            else:
                kv = self.layernorm_kv(a_normed)
        else:
            kv = a_normed

        # `pair_bias`, when supplied, is the fully-formed attention bias for this
        # block (`_pair_bias(z)` already summed with any `extra_attn_bias`),
        # precomputed once outside the diffusion step loop since it depends only
        # on the loop-invariant `z`. Non-windowed path only.
        if pair_bias is not None:
            assert not (n_queries and n_keys), "precomputed pair_bias unsupported for windowed attention"
            out = self.attention(q_x=a_normed, kv_x=kv, attn_bias=pair_bias)
            if self.has_s:
                out = jax.nn.sigmoid(self.linear_a_last(s)) * out
            return out

        bias = self._pair_bias(self.layernorm_z, self.linear_nobias_z, z)

        if n_queries and n_keys:
            out = self.attention(q_x=a_normed, kv_x=kv, attn_bias=bias, n_queries=n_queries, n_keys=n_keys)
        else:
            if extra_attn_bias is not None:
                bias = bias + extra_attn_bias
            out = self.attention(q_x=a_normed, kv_x=kv, attn_bias=bias)

        if self.has_s:
            out = jax.nn.sigmoid(self.linear_a_last(s)) * out
        return out


# ---------------------------------------------------------------------------
# ConditionedTransitionBlock (Algorithm 25 in AF3)
# ---------------------------------------------------------------------------


@register_from_torch("opendde.model.modules.transformer.ConditionedTransitionBlock")
class ConditionedTransitionBlock(AbstractFromTorch):
    adaln: AdaptiveLayerNorm
    linear_nobias_a1: Linear
    linear_nobias_a2: Linear
    linear_nobias_b: Linear
    linear_s: Linear

    def __call__(self, a: Float[Array, "... N Ca"], s: Float[Array, "... N Cs"]) -> Float[Array, "... N Ca"]:
        a = self.adaln(a, s)
        b = jax.nn.silu(self.linear_nobias_a1(a)) * self.linear_nobias_a2(a)
        return jax.nn.sigmoid(self.linear_s(s)) * self.linear_nobias_b(b)


# ---------------------------------------------------------------------------
# DiffusionTransformerBlock / DiffusionTransformer (Algorithm 23 in AF3)
# ---------------------------------------------------------------------------


@register_from_torch("opendde.model.modules.transformer.DiffusionTransformerBlock")
class DiffusionTransformerBlock(AbstractFromTorch):
    attention_pair_bias: AttentionPairBias
    conditioned_transition_block: ConditionedTransitionBlock
    residual_path: Identity

    def __call__(
        self,
        a: Float[Array, "... N Ca"],
        s: Float[Array, "... N Cs"],
        z: Float[Array, "..."],
        n_queries: int | None = None,
        n_keys: int | None = None,
        extra_attn_bias: Float[Array, "..."] | None = None,
        pair_bias: Float[Array, "..."] | None = None,
        **_ignored,
    ) -> Float[Array, "... N Ca"]:
        attn_out = self.attention_pair_bias(
            a=a, s=s, z=z, n_queries=n_queries, n_keys=n_keys,
            extra_attn_bias=extra_attn_bias, pair_bias=pair_bias,
        )
        attn_out = attn_out + a
        ff_out = self.conditioned_transition_block(a=attn_out, s=s)
        out_a = ff_out + attn_out
        # Returns (out_a, s, z); s/z pass through unchanged.
        return out_a, s, z


@register_from_torch("opendde.model.modules.transformer.DiffusionTransformer")
class DiffusionTransformer(eqx.Module):
    """Stack of DiffusionTransformerBlock, scanned.

    Each block's AttentionPairBias owns its own layernorm_z + linear_nobias_z
    and recomputes its bias from the loop-invariant `z` every layer. Scans over
    the per-block parameters while passing `z`, `s`, `n_queries`/`n_keys`, and
    `extra_attn_bias` as loop-invariant closure args.
    """

    stacked_blocks: DiffusionTransformerBlock
    static: DiffusionTransformerBlock
    n_blocks: int

    @staticmethod
    def from_torch(m):
        blocks = [from_torch(b) for b in m.blocks]
        stacked, static = stack_blocks(blocks)
        return DiffusionTransformer(stacked_blocks=stacked, static=static, n_blocks=len(blocks))

    def precompute_pair_bias(
        self, z: Float[Array, "..."], extra_attn_bias: Float[Array, "..."] | None
    ) -> Float[Array, "n_blocks ..."]:
        """Every block's fully-formed attention bias, stacked along a leading
        n_blocks axis. Each block owns its own layernorm_z/linear_nobias_z but
        applies them to the SAME loop-invariant `z`, so in the diffusion sampler
        this is computed once and reused for all steps (see sample_diffusion).
        Non-windowed (token DiffusionTransformer) only."""
        def one(params):
            apb = eqx.combine(self.static, params).attention_pair_bias
            bias = apb._pair_bias(apb.layernorm_z, apb.linear_nobias_z, z)
            if extra_attn_bias is not None:
                bias = bias + extra_attn_bias
            return bias

        return jax.vmap(one)(self.stacked_blocks)

    def __call__(
        self,
        a: Float[Array, "... N Ca"],
        s: Float[Array, "... N Cs"],
        z: Float[Array, "..."],
        n_queries: int | None = None,
        n_keys: int | None = None,
        extra_attn_bias: Float[Array, "..."] | None = None,
        pair_bias_stack: Float[Array, "n_blocks ..."] | None = None,
        **_ignored,
    ) -> Float[Array, "... N Ca"]:
        @jax.checkpoint
        def body_fn(a_carry, xs):
            params, pair_bias = xs
            block = eqx.combine(self.static, params)
            out_a, _, _ = block(
                a_carry, s, z, n_queries=n_queries, n_keys=n_keys,
                extra_attn_bias=extra_attn_bias, pair_bias=pair_bias,
            )
            return out_a, None

        # When a precomputed per-block bias stack is supplied, scan over it
        # alongside the block params; otherwise the xs subtree is None (scan
        # treats it as an empty leaf), so each block gets pair_bias=None and
        # reprojects the bias from `z` itself.
        a, _ = jax.lax.scan(body_fn, a, (self.stacked_blocks, pair_bias_stack))
        return a


# ---------------------------------------------------------------------------
# AtomTransformer (Algorithm 7 in AF3)
# ---------------------------------------------------------------------------


@register_from_torch("opendde.model.modules.transformer.AtomTransformer")
class AtomTransformer(AbstractFromTorch):
    diffusion_transformer: DiffusionTransformer
    n_queries: int
    n_keys: int

    def __call__(
        self,
        q: Float[Array, "... N_atom Catom"],
        c: Float[Array, "... N_atom Catom"],
        p: Float[Array, "... n_blocks n_queries n_keys Catompair"],
        **_ignored,
    ) -> Float[Array, "... N_atom Catom"]:
        n_queries, n_keys = p.shape[-3], p.shape[-2]
        assert n_queries == self.n_queries
        assert n_keys == self.n_keys
        return self.diffusion_transformer(a=q, s=c, z=p, n_queries=self.n_queries, n_keys=self.n_keys)


# ---------------------------------------------------------------------------
# AtomAttentionEncoder (Algorithm 5 in AF3)
# ---------------------------------------------------------------------------


@register_from_torch("opendde.model.modules.transformer.AtomAttentionEncoder")
class AtomAttentionEncoder(AbstractFromTorch):
    linear_no_bias_ref_pos: Linear
    linear_no_bias_ref_charge: Linear
    linear_no_bias_f: Linear
    linear_no_bias_d: Linear
    linear_no_bias_invd: Linear
    linear_no_bias_v: Linear
    linear_no_bias_cl: Linear
    linear_no_bias_cm: Linear
    small_mlp: Any
    atom_transformer: AtomTransformer
    linear_no_bias_q: Linear

    layernorm_s: LayerNorm | None
    linear_no_bias_s: Linear | None
    layernorm_z: LayerNorm | None
    linear_no_bias_z: Linear | None
    linear_no_bias_r: Linear | None

    has_coords: bool
    n_queries: int
    n_keys: int

    def _add_token_pair_context_to_atom_pair(self, p_lm, z, atom_to_token_idx):
        (idx_q,), (idx_k,), _ = rearrange_qk_to_dense_trunk(
            [atom_to_token_idx],
            [atom_to_token_idx],
            [-1],
            [-1],
            n_queries=self.n_queries,
            n_keys=self.n_keys,
            compute_mask=False,
        )
        z_token_pair = gather_pair_embedding_in_dense_trunk(z, idx_q=idx_q, idx_k=idx_k)
        z_token_pair = self.linear_no_bias_z(self.layernorm_z(z_token_pair))
        # Insert the N_sample axis before the n_blocks (window) axis;
        # z_token_pair (no sample axis) broadcasts against the new leading-1
        # sample dim.
        return jnp.expand_dims(p_lm, axis=-5) + z_token_pair

    def _add_atom_single_context_and_mlp(self, p_lm, c_l_q, c_l_k):
        p_lm = (
            p_lm
            + self.linear_no_bias_cl(jax.nn.relu(c_l_q[..., None, :]))
            + self.linear_no_bias_cm(jax.nn.relu(c_l_k[..., None, :, :]))
        )
        return p_lm + self.small_mlp(p_lm)

    def prepare_cache(self, ref_pos, ref_charge, ref_mask, ref_atom_name_chars, ref_element, atom_to_token_idx, d_lm, v_lm, pad_info: DenseTrunkPad, r_l=None, z=None):
        batch_shape = ref_pos.shape[:-2]
        n_atom = ref_pos.shape[-2]
        c_l = self.linear_no_bias_ref_pos(ref_pos) + self.linear_no_bias_ref_charge(
            jnp.arcsinh(ref_charge).reshape(*batch_shape, n_atom, 1)
        )
        ref_features = jnp.concatenate(
            [
                ref_mask.reshape(*batch_shape, n_atom, 1),
                ref_element.reshape(*batch_shape, n_atom, 128),
                ref_atom_name_chars.reshape(*batch_shape, n_atom, 4 * 64),
            ],
            axis=-1,
        ).astype(c_l.dtype)
        c_l = c_l + self.linear_no_bias_f(ref_features)
        c_l = c_l * ref_mask.reshape(*batch_shape, n_atom, 1)

        mask_trunked = pad_info.mask_trunked
        p_lm = (self.linear_no_bias_d(d_lm) * v_lm) * mask_trunked[..., None]
        p_lm = p_lm + self.linear_no_bias_invd(1 / (1 + jnp.sum(d_lm**2, axis=-1, keepdims=True))) * v_lm
        p_lm = p_lm + self.linear_no_bias_v(v_lm.astype(p_lm.dtype))

        if r_l is not None:
            p_lm = self._add_token_pair_context_to_atom_pair(p_lm=p_lm, z=z, atom_to_token_idx=atom_to_token_idx)
        return p_lm, c_l

    def __call__(
        self,
        atom_to_token_idx: Int[Array, "N_atom"],
        ref_pos: Float[Array, "N_atom 3"],
        ref_charge: Float[Array, "N_atom"],
        ref_mask: Float[Array, "N_atom"],
        ref_atom_name_chars: Float[Array, "N_atom 4 64"],
        ref_element: Float[Array, "N_atom 128"],
        d_lm: Float[Array, "n_blocks n_queries n_keys 3"],
        v_lm: Float[Array, "n_blocks n_queries n_keys 1"],
        pad_info: DenseTrunkPad,
        r_l: Float[Array, "... N_atom 3"] | None = None,
        s: Float[Array, "... N_token Cs"] | None = None,
        z: Float[Array, "N_token N_token Cz"] | None = None,
        p_lm: Float[Array, "..."] | None = None,
        c_l: Float[Array, "..."] | None = None,
        n_token: int | None = None,
        **_ignored,
    ):
        if self.has_coords:
            assert r_l is not None and s is not None and z is not None

        if p_lm is None or c_l is None:
            p_lm, c_l = self.prepare_cache(
                ref_pos=ref_pos,
                ref_charge=ref_charge,
                ref_mask=ref_mask,
                ref_atom_name_chars=ref_atom_name_chars,
                ref_element=ref_element,
                atom_to_token_idx=atom_to_token_idx,
                d_lm=d_lm,
                v_lm=v_lm,
                pad_info=pad_info,
                r_l=r_l,
                z=z,
            )

        # In the coords path n_token is the (static) token count from s; in the
        # no-coords path (InputFeatureEmbedder) the caller threads a static
        # n_token in from the token-feature shape (else it falls back to
        # max(atom_to_token_idx)+1, which isn't a static jit shape).
        if r_l is not None:
            n_token = s.shape[-2]
            c_l = c_l[..., None, :, :] + broadcast_token_to_atom(
                x_token=self.linear_no_bias_s(self.layernorm_s(s)),
                atom_to_token_idx=atom_to_token_idx,
            )
            q_l = c_l + self.linear_no_bias_r(r_l)
        else:
            q_l = c_l

        (c_l_q,), (c_l_k,), _ = rearrange_qk_to_dense_trunk(
            [c_l], [c_l], [-2], [-2], n_queries=self.n_queries, n_keys=self.n_keys, compute_mask=False
        )
        p_lm = self._add_atom_single_context_and_mlp(p_lm=p_lm, c_l_q=c_l_q, c_l_k=c_l_k)

        q_l = self.atom_transformer(q_l, c_l, p_lm)

        a = aggregate_atom_to_token(
            x_atom=jax.nn.relu(self.linear_no_bias_q(q_l)),
            atom_to_token_idx=atom_to_token_idx,
            n_token=n_token,
            reduce="mean",
        )
        return a, q_l, c_l, p_lm


# ---------------------------------------------------------------------------
# AtomAttentionDecoder (Algorithm 6 in AF3)
# ---------------------------------------------------------------------------


@register_from_torch("opendde.model.modules.transformer.AtomAttentionDecoder")
class AtomAttentionDecoder(AbstractFromTorch):
    linear_no_bias_a: Linear
    layernorm_q: LayerNorm
    linear_no_bias_out: Linear
    atom_transformer: AtomTransformer

    def __call__(
        self,
        atom_to_token_idx: Int[Array, "N_atom"],
        a: Float[Array, "... N_token Ctoken"],
        q_skip: Float[Array, "... N_atom Catom"],
        c_skip: Float[Array, "... N_atom Catom"],
        p_skip: Float[Array, "... n_blocks n_queries n_keys Catompair"],
        **_ignored,
    ) -> Float[Array, "... N_atom 3"]:
        q = broadcast_token_to_atom(x_token=self.linear_no_bias_a(a), atom_to_token_idx=atom_to_token_idx) + q_skip
        q = self.atom_transformer(q, c_skip, p_skip)
        q = self.layernorm_q(q)
        return self.linear_no_bias_out(q)
