"""Wires up from_torch registrations against real torch/opendde types.

This is the only module in jopendde that imports torch (directly or
transitively via `opendde`). Importing `jopendde` itself must stay torch-free
for torch-free inference; importing `jopendde.convert` pulls in torch/opendde
for weight conversion.
"""

import importlib
from functools import partial

import jax
import numpy as np
import torch

from jopendde.backend import _DEFERRED_REGISTRATIONS, from_torch

# Import torch-free jopendde modules so their @register_from_torch decorators
# populate _DEFERRED_REGISTRATIONS before we resolve/wire it up below.
from jopendde import (  # noqa: F401
    confidence,
    diffusion,
    embedders,
    head,
    model,
    msa,
    primitives,
    structural_tokens,
    transformer,
    triangular,
)


def _resolve(dotted_path: str):
    module_path, _, name = dotted_path.rpartition(".")
    return getattr(importlib.import_module(module_path), name)


for dotted_path, converter in _DEFERRED_REGISTRATIONS:
    from_torch.register(_resolve(dotted_path), converter)

from_torch.register(torch.Tensor, lambda t: t.detach().float().cpu().numpy())
from_torch.register(torch.nn.ModuleList, lambda x: [from_torch(m) for m in x])
from_torch.register(torch.nn.ReLU, lambda _: jax.nn.relu)
from_torch.register(torch.nn.Sigmoid, lambda _: jax.nn.sigmoid)
from_torch.register(torch.nn.SiLU, lambda _: jax.nn.silu)
from_torch.register(torch.nn.Softmax, lambda m: partial(jax.nn.softmax, axis=m.dim))
