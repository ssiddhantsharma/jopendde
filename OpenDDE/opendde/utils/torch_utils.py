# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import gc
from contextlib import contextmanager, nullcontext
from typing import Sequence

import numpy as np
import torch


def to_device(obj, device, non_blocking: bool = False):
    """Move tensor or dict of tensors to device"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, dict):
                to_device(v, device, non_blocking=non_blocking)
            elif isinstance(v, torch.Tensor):
                obj[k] = obj[k].to(device=device, non_blocking=non_blocking)
    elif isinstance(obj, torch.Tensor):
        obj = obj.to(device=device, non_blocking=non_blocking)
    else:
        raise Exception(f"type {type(obj)} not supported")
    return obj


def cleanup_cuda_memory(collect_garbage: bool = True) -> None:
    """Optionally collect Python garbage and return cached CUDA blocks to the allocator."""
    if collect_garbage:
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextmanager
def disable_cudnn_benchmark():
    """Temporarily disable cuDNN benchmark to reduce eval-time workspace spikes."""
    if not torch.cuda.is_available():
        yield
        return

    benchmark_enabled = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False
    try:
        yield
    finally:
        torch.backends.cudnn.benchmark = benchmark_enabled


def cdist(a: torch.Tensor, b: torch.Tensor | None = None):
    # for tensor shape [1, 512 * 14, 3], donot_use_mm_for_euclid_dist mode costs 0.0489s,
    # while use_mm_for_euclid_dist_if_necessary costs 0.0419s on cpu. On GPU there two costs
    # will be neglectible. So there is no need to sacrifice accuracy for speed here.
    return torch.cdist(
        a,
        b if b is not None else a,
        compute_mode="donot_use_mm_for_euclid_dist",
    )


def eye_mask(L, device=None, opposite=False):
    if opposite:
        return 1.0 - torch.eye(L, device=device)
    else:
        return torch.eye(L, device=device)


def permute_last_dims(t: torch.Tensor, dims: Sequence[int]):
    """Permute tensor on last dims, all other dims are kept unchanged.

    Args:
        t (torch.Tensor): Input tensor with at least len(dims) dimensions.
        dims: The desired ordering of dimensions, here all values should be < 0, i.e. (-1, -2) means permute last two dims.
    """
    num_dims = len(t.shape)
    prefix_dims = list(range(num_dims - len(dims)))
    last_dims = [num_dims + d for d in dims]
    return torch.permute(t, prefix_dims + last_dims)


def map_values_to_list(data, recursive=True):
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            if v.dtype == torch.bfloat16:
                v = v.float()
            data[k] = v.cpu().numpy().tolist()
        elif isinstance(v, np.ndarray):
            data[k] = v.tolist()
        elif isinstance(v, dict) and recursive:
            data[k] = map_values_to_list(v, recursive)
    return data


def round_values(data, recursive=True):
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            if v.dtype == torch.bfloat16:
                v = v.float()
            data[k] = np.round(v.cpu().numpy(), 2)
        elif isinstance(v, np.ndarray):
            data[k] = np.round(v, 2)
        elif isinstance(v, list):
            data[k] = list(np.round(np.array(v), 2))
        elif isinstance(v, dict) and recursive:
            data[k] = round_values(v, recursive)
    return data


def autocasting_disable_decorator(disable_casting):
    def func_wrapper(func):
        def new_func(*args, **kwargs):
            _amp_context = (
                torch.autocast(device_type="cuda", enabled=False)
                if disable_casting
                else nullcontext()
            )

            # Helper function to conditionally cast tensors
            def conditioned_cast(tensor):
                if (
                    disable_casting
                    and isinstance(tensor, torch.Tensor)
                    and torch.is_floating_point(tensor)
                ):
                    return tensor.to(dtype=torch.float32)
                return tensor

            with _amp_context:
                return func(
                    *(conditioned_cast(v) for v in args),
                    **{k: conditioned_cast(v) for k, v in kwargs.items()},
                )

        return new_func

    return func_wrapper


def dict_to_tensor(feature_dict):
    for k, v in feature_dict.items():
        if not isinstance(v, torch.Tensor):
            dtype = feature_dict[k].dtype
            feature_dict[k] = torch.tensor(v)

            if dtype in [np.int64, np.int32]:
                feature_dict[k] = feature_dict[k].to(torch.int64)
            elif dtype in [np.float32, np.float64]:
                feature_dict[k] = feature_dict[k].to(torch.float32)

    return feature_dict


def collate_fn_identity(x):
    return x
