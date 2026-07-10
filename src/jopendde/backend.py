"""Torch-free from_torch dispatch + AbstractFromTorch base class.

Importing this module never pulls in torch; torch-type registrations live in
`convert.py`, the only module allowed to `import torch`.
"""

from __future__ import annotations

import time
import typing
from dataclasses import fields
from functools import singledispatch

import equinox as eqx
import jax
import numpy as np
from jax import numpy as jnp
from jax import tree
from jaxtyping import Array, Float, Int


@singledispatch
def from_torch(x):
    raise NotImplementedError(f"from_torch not implemented for {type(x)}: {x}")


from_torch.register(str, lambda x: x)
from_torch.register(int, lambda x: x)
from_torch.register(float, lambda x: x)
from_torch.register(bool, lambda x: x)
from_torch.register(type(None), lambda x: x)
from_torch.register(tuple, lambda x: tuple(map(from_torch, x)))
from_torch.register(list, lambda x: [from_torch(v) for v in x])
from_torch.register(dict, lambda x: {k: from_torch(v) for k, v in x.items()})


class AbstractFromTorch(eqx.Module):
    """Default `from_torch` for equinox modules whose field names match the
    torch module's named_children()/named_parameters(recurse=False) keys."""

    @classmethod
    def from_torch(cls, model):
        # `from __future__ import annotations` in the module means field.type is
        # a *string*; resolve to real types so optionality checks work.
        try:
            resolved = typing.get_type_hints(cls)
        except Exception:
            resolved = {}
        field_to_type = {
            field.name: resolved.get(field.name, field.type) for field in fields(cls)
        }
        kwargs = {
            child: from_torch(child_module)
            for child, child_module in model.named_children()
        } | {
            parameter_name: from_torch(parameter)
            for parameter_name, parameter in model.named_parameters(recurse=False)
        }

        def _allows_none(t):
            if t is None or t is type(None) or t is typing.Any:
                return True
            return type(None) in typing.get_args(t)

        for field_name, field_type in field_to_type.items():
            if not hasattr(model, field_name):
                if not _allows_none(field_type):
                    raise ValueError(
                        f"Field {field_name} for {cls} is not optional but is "
                        f"missing from torch model {model}"
                    )
                kwargs[field_name] = None
            else:
                kwargs[field_name] = from_torch(getattr(model, field_name))

        torch_not_equinox = kwargs.keys() - field_to_type.keys()
        if torch_not_equinox:
            raise ValueError(
                f"Properties in torch model not found in equinox module {cls}: "
                f"{torch_not_equinox}"
            )

        return cls(**kwargs)


# Registry of (dotted_path, converter) pairs, resolved against real torch
# types lazily by convert.py so this module stays torch-free.
_DEFERRED_REGISTRATIONS: list[tuple[str, object]] = []


def register_from_torch(torch_dotted_path: str):
    """Class decorator: registers `cls.from_torch` against a torch type named
    by dotted path (e.g. "torch.nn.Linear"), resolved later in convert.py."""

    def decorator(cls):
        _DEFERRED_REGISTRATIONS.append((torch_dotted_path, cls.from_torch))
        return cls

    return decorator


def stack_blocks(blocks: list) -> tuple:
    """Stack a list of identical eqx.Modules into (stacked_params, static) so
    they can run with one `jax.lax.scan` instead of an unrolled Python loop.

    `stacked_params` holds every inexact-array leaf stacked along a new leading
    axis (one slice per block); `static` holds the shared non-array structure.
    A module's __call__ scans `stacked_params`, rebuilding each block with
    `eqx.combine(params_slice, static)`. Shared by PairformerStack, MSAModule,
    and DiffusionTransformer, which construct their stacks identically."""
    stacked = jax.tree.map(
        lambda *v: jnp.stack(v, 0),
        *[eqx.filter(b, eqx.is_inexact_array) for b in blocks],
    )
    _, static = eqx.partition(blocks[0], eqx.is_inexact_array)
    return stacked, static


@register_from_torch("torch.nn.Linear")
class Linear(eqx.Module):
    """Linear layer: y = x Wᵀ (+ b)."""

    weight: Float[Array, "Out In"]
    bias: Float[Array, "Out"] | None

    def __call__(self, x: Float[Array, "... In"]) -> Float[Array, "... Out"]:
        o = jnp.einsum("...i,oi->...o", x, self.weight)
        if self.bias is not None:
            o = o + jnp.broadcast_to(self.bias, x.shape[:-1] + (self.bias.shape[-1],))
        return o

    @staticmethod
    def from_torch(m):
        return Linear(weight=from_torch(m.weight), bias=from_torch(m.bias))


@register_from_torch("torch.nn.modules.linear.Identity")
class Identity(eqx.Module):
    def __call__(self, x):
        return x

    @staticmethod
    def from_torch(_):
        return Identity()


@register_from_torch("torch.nn.LayerNorm")
class LayerNorm(eqx.Module):
    """LayerNorm over the last dimension."""

    weight: Float[Array, "D"] | None
    bias: Float[Array, "D"] | None
    eps: float

    def __call__(self, x: Float[Array, "... D"]) -> Float[Array, "... D"]:
        mean = x.mean(axis=-1, keepdims=True)
        var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
        x = (x - mean) * jax.lax.rsqrt(var + self.eps)
        if self.weight is not None:
            x = x * self.weight
        if self.bias is not None:
            x = x + self.bias
        return x

    @staticmethod
    def from_torch(m):
        return LayerNorm(weight=from_torch(m.weight), bias=from_torch(m.bias), eps=m.eps)


@register_from_torch("torch.nn.Sequential")
class Sequential(eqx.Module):
    _modules: dict[str, eqx.Module]

    def __call__(self, x):
        for idx in range(len(self._modules)):
            x = self._modules[str(idx)](x)
        return x

    @staticmethod
    def from_torch(m):
        return Sequential(_modules=from_torch(dict(m._modules)))


@register_from_torch("torch.nn.modules.sparse.Embedding")
class Embedding(eqx.Module):
    weight: Float[Array, "V D"]

    def __call__(self, tokens: Int[Array, "..."]) -> Float[Array, "... D"]:
        return self.weight[tokens]

    @staticmethod
    def from_torch(m):
        return Embedding(weight=from_torch(m.weight))


class TestModule:
    """Side-by-side numerical/timing comparison of a torch module vs. its
    from_torch-converted, jit-compiled equinox module.

    Only usable from convert.py / test code that already imports torch.
    """

    def __init__(self, module):
        self.mod = module
        self.j_m = eqx.filter_jit(from_torch(module))

    def __call__(self, *args, **kwargs):
        torch_start = time.time()
        torch_output = self.mod(*args, **kwargs)
        torch_end = time.time()

        jax_start = time.time()
        with jax.default_matmul_precision("float32"):
            jax_output = self.j_m(*from_torch(args), **from_torch(kwargs))
        tree.map(lambda v: v.block_until_ready(), jax_output)
        jax_end = time.time()

        def _err(a, b):
            if not isinstance(b, jnp.ndarray):
                return None
            a = np.asarray(a)
            b = np.asarray(b)
            abs_err = np.abs(a - b).max()
            rel_err = abs_err / max(np.abs(a).max(), 1e-6)
            return abs_err, rel_err

        errors = tree.map(_err, torch_output, jax_output, is_leaf=eqx.is_inexact_array)
        print(f"errors (abs, rel) for {type(self.mod)}: {errors}")
        print(f"torch time: {torch_end - torch_start:.3f}s, jax time: {jax_end - jax_start:.3f}s")
        return torch_output
