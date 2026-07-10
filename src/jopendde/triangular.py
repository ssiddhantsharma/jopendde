"""Triangle and attention modules: Attention, OuterProductMean,
TriangleMultiplicativeUpdate (outgoing/incoming), and TriangleAttention.

Registrations use dotted-path strings resolved lazily (see
backend.register_from_torch).

The local-windowed attention path (`n_queries`/`n_keys`) is not implemented
here; `Attention` below covers only the standard path and raises if windowing
args are supplied. cuEquivariance fused-kernel paths, the fp16 overflow guard,
and the in-place/chunked path are dropped.
"""

from __future__ import annotations

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float

from jopendde.backend import AbstractFromTorch, Linear, LayerNorm, from_torch, register_from_torch

# Optional fused triangle kernels (NVIDIA cuEquivariance). Import is guarded so
# jopendde stays runnable without the CUDA-only package; `use_cue_kernel` fields
# default False, so nothing changes unless a caller opts in via
# jopendde.inference.enable_cue_kernels (which also checks CUE_AVAILABLE).
try:
    import cuequivariance_jax as _cuex  # noqa: N816

    CUE_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on optional CUDA wheels
    _cuex = None
    CUE_AVAILABLE = False

# OpenfoldLinear has the same weight/bias semantics as a plain Linear (it
# differs only in init, irrelevant for loaded weights). OpenFoldLayerNorm and
# modules.primitives.Linear are registered in jopendde/primitives.py.
register_from_torch("opendde.model.triangular.layers.OpenfoldLinear")(Linear)


@register_from_torch("opendde.model.modules.primitives.Attention")
@register_from_torch("opendde.model.triangular.layers.Attention")
class Attention(AbstractFromTorch):
    """Standard multi-head attention, used both as the general-purpose
    attention in primitives.py and inside triangle attention. The math is
    q@k^T + bias -> softmax -> @v with an optional sigmoid gate. Serves two
    torch classes that differ only in config knobs (which don't affect the
    non-windowed path) and in the head-count attribute name (`num_heads` vs
    `no_heads`), handled in `from_torch`.
    """

    c_hidden: int
    no_heads: int
    linear_q: Linear
    linear_k: Linear
    linear_v: Linear
    linear_o: Linear
    linear_g: Linear | None
    sigmoid: callable

    @classmethod
    def from_torch(cls, model):
        no_heads = getattr(model, "no_heads", None)
        if no_heads is None:
            no_heads = model.num_heads
        return cls(
            c_hidden=model.c_hidden,
            no_heads=no_heads,
            linear_q=from_torch(model.linear_q),
            linear_k=from_torch(model.linear_k),
            linear_v=from_torch(model.linear_v),
            linear_o=from_torch(model.linear_o),
            linear_g=from_torch(model.linear_g) if model.linear_g is not None else None,
            sigmoid=jax.nn.sigmoid,
        )

    @staticmethod
    def _maybe_add_head_dim(bias: Float[Array, "..."], a_ndim: int) -> Float[Array, "..."]:
        # Insert a broadcastable head axis when the bias is missing exactly the
        # head dimension.
        if bias.ndim == a_ndim - 1:
            bias = bias[..., None, :, :]
        return bias

    def __call__(
        self,
        q_x: Float[Array, "... Q C_q"],
        kv_x: Float[Array, "... K C_k"],
        attn_bias: Float[Array, "..."] | None = None,
        biases: list[Float[Array, "..."]] | None = None,
        trunked_attn_bias=None,
        n_queries: int | None = None,
        n_keys: int | None = None,
        inf: float = 1e10,
        inplace_safe: bool = False,
        chunk_size: int | None = None,
        triangle_attention: str = "torch",
    ) -> Float[Array, "... Q C_q"]:
        if n_queries is not None or n_keys is not None or trunked_attn_bias is not None:
            raise NotImplementedError(
                "Local-windowed attention (n_queries/n_keys) is not implemented "
                "in this Attention."
            )
        assert chunk_size is None, "chunked inference path is not implemented"

        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)

        q = einops.rearrange(q, "... Q (H C) -> ... H Q C", H=self.no_heads)
        k = einops.rearrange(k, "... K (H C) -> ... H K C", H=self.no_heads)
        v = einops.rearrange(v, "... V (H C) -> ... H V C", H=self.no_heads)

        q = q / np.sqrt(self.c_hidden)

        a = jnp.einsum("...hqc,...hkc->...hqk", q, k)

        if attn_bias is not None:
            a = a + self._maybe_add_head_dim(attn_bias, a.ndim)
        if biases is not None:
            for bias in biases:
                a = a + self._maybe_add_head_dim(bias, a.ndim)

        a = jax.nn.softmax(a, axis=-1)

        o = jnp.einsum("...hqk,...hkc->...hqc", a, v)
        o = einops.rearrange(o, "... H Q C -> ... Q H C")

        if self.linear_g is not None:
            g = self.sigmoid(self.linear_g(q_x))
            g = einops.rearrange(g, "... (H C) -> ... H C", H=self.no_heads)
            o = o * g

        o = einops.rearrange(o, "... Q H C -> ... Q (H C)")
        return self.linear_o(o)


@register_from_torch("opendde.model.triangular.layers.OuterProductMean")
class OuterProductMean(AbstractFromTorch):
    """Algorithm 10 in AF3. Chunking is dropped."""

    layer_norm: LayerNorm
    linear_1: Linear
    linear_2: Linear
    linear_out: Linear
    eps: float

    def __call__(
        self,
        m: Float[Array, "... S N Cm"],
        mask: Bool[Array, "... S N"] | None = None,
        chunk_size: int | None = None,
        inplace_safe: bool = False,
    ) -> Float[Array, "... N N Cz"]:
        assert chunk_size is None, "chunked path is not implemented"

        if mask is None:
            mask = jnp.ones(m.shape[:-1], dtype=m.dtype)

        ln = self.layer_norm(m)

        mask = mask[..., None]
        a = self.linear_1(ln) * mask
        b = self.linear_2(ln) * mask

        # [..., N, S, C]
        a = jnp.swapaxes(a, -2, -3)
        b = jnp.swapaxes(b, -2, -3)

        outer = jnp.einsum("...bac,...dae->...bdce", a, b)
        outer = outer.reshape(outer.shape[:-2] + (-1,))
        outer = self.linear_out(outer)

        norm = jnp.einsum("...abc,...adc->...bdc", mask, mask)
        norm = norm + self.eps

        return outer / norm


@register_from_torch("opendde.model.triangular.triangular.TriangleMultiplicationOutgoing")
@register_from_torch("opendde.model.triangular.triangular.TriangleMultiplicationIncoming")
class TriangleMultiplicativeUpdate(AbstractFromTorch):
    """Algorithms 11 (outgoing) and 12 (incoming) in AF3. `_outgoing` (read
    off the torch instance) selects the contraction direction, so one class
    serves both subclasses. The cuEquivariance kernel, the fp16 overflow
    guard, and the in-place/chunked path are dropped. Both the `a`/`b` gates
    and the output gate `g` apply to the normalized input (`z` after
    `layer_norm_in`), not the raw input.
    """

    linear_g: Linear
    linear_z: Linear
    layer_norm_in: LayerNorm
    layer_norm_out: LayerNorm
    linear_a_p: Linear
    linear_a_g: Linear
    linear_b_p: Linear
    linear_b_g: Linear
    sigmoid: callable
    _outgoing: bool
    # Opt-in fused cuEquivariance kernel (static -> baked at trace time). Set by
    # jopendde.inference.enable_cue_kernels; not a torch attribute, so
    # `from_torch` injects the default rather than reading it off the module.
    use_cue_kernel: bool = eqx.field(static=True, default=False)

    @classmethod
    def from_torch(cls, model):
        return cls(
            linear_g=from_torch(model.linear_g),
            linear_z=from_torch(model.linear_z),
            layer_norm_in=from_torch(model.layer_norm_in),
            layer_norm_out=from_torch(model.layer_norm_out),
            linear_a_p=from_torch(model.linear_a_p),
            linear_a_g=from_torch(model.linear_a_g),
            linear_b_p=from_torch(model.linear_b_p),
            linear_b_g=from_torch(model.linear_b_g),
            sigmoid=jax.nn.sigmoid,
            _outgoing=model._outgoing,
        )

    def _cue_call(
        self, z: Float[Array, "... N N C"], mask: Bool[Array, "... N N"] | None
    ) -> Float[Array, "... N N C"]:
        # Fused equivalent of the pure path below. The kernel takes the input
        # projection/gating as concatenated [a; b] weights (2C, C) and the
        # output projection/gating separately; a/b ordering matches the a/b
        # split below. The projection linears are typically bias-free
        # (bias=None); the kernel
        # accepts None, so only concatenate when both halves carry a bias.
        def _cat_bias(u, v):
            return None if u is None else jnp.concatenate([u, v], 0)

        return _cuex.triangle_multiplicative_update(
            x=z,
            direction="outgoing" if self._outgoing else "incoming",
            mask=mask,
            norm_in_weight=self.layer_norm_in.weight,
            norm_in_bias=self.layer_norm_in.bias,
            p_in_weight=jnp.concatenate([self.linear_a_p.weight, self.linear_b_p.weight], 0),
            p_in_bias=_cat_bias(self.linear_a_p.bias, self.linear_b_p.bias),
            g_in_weight=jnp.concatenate([self.linear_a_g.weight, self.linear_b_g.weight], 0),
            g_in_bias=_cat_bias(self.linear_a_g.bias, self.linear_b_g.bias),
            norm_out_weight=self.layer_norm_out.weight,
            norm_out_bias=self.layer_norm_out.bias,
            p_out_weight=self.linear_z.weight,
            p_out_bias=self.linear_z.bias,
            g_out_weight=self.linear_g.weight,
            g_out_bias=self.linear_g.bias,
            eps=self.layer_norm_in.eps,
        )

    def __call__(
        self,
        z: Float[Array, "... N N C"],
        mask: Bool[Array, "... N N"] | None = None,
        inplace_safe: bool = False,
        _add_with_inplace: bool = False,
        _inplace_chunk_size: int | None = None,
        triangle_multiplicative: str = "torch",
    ) -> Float[Array, "... N N C"]:
        z_in = z

        if self.use_cue_kernel:
            x = self._cue_call(z, mask)
            return z_in + x if _add_with_inplace else x

        if mask is None:
            mask = jnp.ones(z.shape[:-1], dtype=z.dtype)
        mask = mask[..., None]

        zn = self.layer_norm_in(z)

        a = mask * self.sigmoid(self.linear_a_g(zn)) * self.linear_a_p(zn)
        b = mask * self.sigmoid(self.linear_b_g(zn)) * self.linear_b_p(zn)

        if self._outgoing:
            x = jnp.einsum("...ijc,...kjc->...ikc", a, b)
        else:
            x = jnp.einsum("...jic,...jkc->...ikc", a, b)

        x = self.layer_norm_out(x)
        x = self.linear_z(x)
        g = self.sigmoid(self.linear_g(zn))
        x = x * g

        if _add_with_inplace:
            return z_in + x
        return x


@register_from_torch("opendde.model.triangular.triangular.TriangleAttention")
class TriangleAttention(AbstractFromTorch):
    """Triangle attention (starting/ending node), AF3. The starting/ending
    distinction is implemented via transpose-before/after. `chunk_size` and
    the cuEquivariance kernel option are dropped."""

    layer_norm: LayerNorm
    linear: Linear
    mha: Attention
    starting: bool
    inf: float
    # Opt-in fused cuEquivariance kernel; see TriangleMultiplicativeUpdate.
    use_cue_kernel: bool = eqx.field(static=True, default=False)

    @classmethod
    def from_torch(cls, model):
        return cls(
            layer_norm=from_torch(model.layer_norm),
            linear=from_torch(model.linear),
            mha=from_torch(model.mha),
            starting=model.starting,
            inf=model.inf,
        )

    def _cue_core(
        self, x: Float[Array, "I J C"], mask: Bool[Array, "I J"]
    ) -> Float[Array, "I J C"]:
        # Replace only the QK^T + bias -> softmax -> @V core (the fused kernel);
        # the q/k/v/o projections and output gating stay in JAX. Kernel layout:
        # q/k/v [B, N, H, S, D], bias [B, 1, H, S, S], mask [B, N, 1, 1, S].
        # `x` here is the layer-normed, already-oriented (starting) input, rank
        # 3 [I, J, C]; the kernel path is only enabled on the (unbatched) trunk.
        assert x.ndim == 3, "cuEquivariance triangle-attention path expects [I, J, C]"
        m = self.mha
        h = m.no_heads
        q = einops.rearrange(m.linear_q(x), "i j (h c) -> i h j c", h=h)
        k = einops.rearrange(m.linear_k(x), "i j (h c) -> i h j c", h=h)
        v = einops.rearrange(m.linear_v(x), "i j (h c) -> i h j c", h=h)
        # triangle bias b[q, k] (shared across the row/batch axis) -> [1, 1, H, J, J]
        bias = einops.rearrange(self.linear(x), "q k h -> h q k")[None, None]
        kmask = mask.astype(bool)[None, :, None, None, :]
        scale = 1.0 / np.sqrt(m.c_hidden)
        out, _, _ = _cuex.triangle_attention(q[None], k[None], v[None], bias, kmask, scale)
        o = einops.rearrange(out[0], "i h j c -> i j h c")
        if m.linear_g is not None:
            g = einops.rearrange(m.sigmoid(m.linear_g(x)), "i j (h c) -> i j h c", h=h)
            o = o * g
        return m.linear_o(einops.rearrange(o, "i j h c -> i j (h c)"))

    def __call__(
        self,
        x: Float[Array, "... I J C"],
        mask: Bool[Array, "... I J"] | None = None,
        chunk_size: int | None = None,
        triangle_attention: str = "torch",
        inplace_safe: bool = False,
    ) -> Float[Array, "... I J C"]:
        assert chunk_size is None, "chunked path is not implemented"

        if mask is None:
            mask = jnp.ones(x.shape[:-1], dtype=x.dtype)

        if not self.starting:
            x = jnp.swapaxes(x, -2, -3)
            mask = jnp.swapaxes(mask, -1, -2)

        x = self.layer_norm(x)

        if self.use_cue_kernel:
            x = self._cue_core(x, mask)
        else:
            # [..., I, 1, 1, J]
            mask_bias = (self.inf * (mask - 1))[..., :, None, None, :]

            # [..., H, I, J] -> [..., 1, H, I, J]
            triangle_bias = jnp.moveaxis(self.linear(x), -1, -3)
            triangle_bias = triangle_bias[..., None, :, :, :]

            biases = [mask_bias, triangle_bias]

            x = self.mha(q_x=x, kv_x=x, biases=biases)

        if not self.starting:
            x = jnp.swapaxes(x, -2, -3)

        return x
