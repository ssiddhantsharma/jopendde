"""Distogram head."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from jopendde.backend import AbstractFromTorch, Linear, register_from_torch


@register_from_torch("opendde.model.modules.head.DistogramHead")
class DistogramHead(AbstractFromTorch):
    """Algorithm 1 [Line17] in AF3. Symmetrizes the pair logits."""

    linear: Linear
    c_z: int
    no_bins: int

    def __call__(self, z: Float[Array, "... N N Cz"]) -> Float[Array, "... N N B"]:
        logits = self.linear(z)
        logits = logits + jnp.swapaxes(logits, -2, -3)
        return logits
