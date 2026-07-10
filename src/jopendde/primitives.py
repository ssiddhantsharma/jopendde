"""Leaf/primitive modules: Transition, AdaptiveLayerNorm.

Registrations use dotted-path strings resolved lazily (see
backend.register_from_torch).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from jopendde.backend import AbstractFromTorch, Linear, LayerNorm, register_from_torch

# primitives.Linear and its BiasInitLinear subclass share backend.Linear's
# weight/bias semantics (they differ only in init, irrelevant for loaded
# weights), so reuse its from_torch.
register_from_torch("opendde.model.modules.primitives.Linear")(Linear)
register_from_torch("opendde.model.modules.primitives.BiasInitLinear")(Linear)

# OpenFoldLayerNorm shares backend.LayerNorm's weight/bias/eps semantics
# (weight/bias may be None when create_scale/create_offset are False).
register_from_torch("opendde.model.triangular.layers.OpenFoldLayerNorm")(LayerNorm)


@register_from_torch("opendde.model.modules.primitives.Transition")
class Transition(AbstractFromTorch):
    """Algorithm 11 in AF3. Row-chunking is dropped."""

    layernorm1: LayerNorm
    linear_no_bias_a: Linear
    linear_no_bias_b: Linear
    linear_no_bias: Linear
    n: int
    c_in: int

    def __call__(self, x: Float[Array, "... D"]) -> Float[Array, "... D"]:
        y = self.layernorm1(x)
        a = jax.nn.silu(self.linear_no_bias_a(y))
        b = self.linear_no_bias_b(y)
        b = b * a
        return self.linear_no_bias(b)


@register_from_torch("opendde.model.modules.primitives.AdaptiveLayerNorm")
class AdaptiveLayerNorm(AbstractFromTorch):
    """Algorithm 26 in AF3."""

    layernorm_a: LayerNorm
    layernorm_s: LayerNorm
    linear_s: Linear
    linear_nobias_s: Linear

    def __call__(
        self, a: Float[Array, "... Ca"], s: Float[Array, "... Cs"]
    ) -> Float[Array, "... Ca"]:
        a = self.layernorm_a(a)
        s = self.layernorm_s(s)
        a = jax.nn.sigmoid(self.linear_s(s)) * a + self.linear_nobias_s(s)
        return a
