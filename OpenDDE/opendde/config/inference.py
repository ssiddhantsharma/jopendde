# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import copy
import logging
from collections.abc import Mapping
from typing import Any, Optional

import torch

from opendde.config.config import parse_configs
from opendde.config.data import data_configs
from opendde.config.inference_defaults import inference_configs
from opendde.config.model_base import configs as configs_base
from opendde.config.model_registry import model_configs
from opendde.config.schema import OpenDDEConfig
from opendde.utils.environment import (
    cuda_acceleration_available,
    has_cuequivariance_packages,
    select_triangle_kernel,
)

TRIANGLE_KERNELS = ("auto", "cuequivariance", "torch")
logger = logging.getLogger(__name__)


def deep_update(configs: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively update nested config dictionaries in place."""
    for key, value in updates.items():
        if (
            isinstance(value, Mapping)
            and key in configs
            and isinstance(configs[key], Mapping)
        ):
            deep_update(configs[key], value)
        else:
            configs[key] = copy.deepcopy(value)
    return configs


def make_base_inference_config(model_name: Optional[str] = None) -> dict[str, Any]:
    """Return an isolated base inference config tree."""
    configs = {
        **copy.deepcopy(configs_base),
        "data": copy.deepcopy(data_configs),
        **copy.deepcopy(inference_configs),
    }
    if model_name is not None:
        configs["model_name"] = model_name
    return configs


def build_inference_config(
    arg_str: Optional[str] = None,
    model_name: Optional[str] = None,
    fill_required_with_null: bool = True,
) -> OpenDDEConfig:
    """
    Build inference configs with model-specific defaults and CLI overrides.

    The selected model is parsed first, model-specific defaults are merged into
    a fresh base tree, then the same arguments are parsed again so user-provided
    values keep highest priority. The fully resolved tree is wrapped in the typed
    :class:`OpenDDEConfig` view (the merge/CLI engine itself is untouched).
    """
    first_pass = parse_configs(
        configs=make_base_inference_config(model_name=model_name),
        arg_str=arg_str,
        fill_required_with_null=fill_required_with_null,
    )
    selected_model_name = first_pass.model_name

    base_configs = make_base_inference_config(model_name=model_name)
    deep_update(base_configs, model_configs[selected_model_name])
    merged = parse_configs(
        configs=base_configs,
        arg_str=arg_str,
        fill_required_with_null=fill_required_with_null,
    )
    return OpenDDEConfig.model_validate(merged.to_dict())


def validate_triangle_kernels(
    triangle_multiplicative: str, triangle_attention: str
) -> None:
    """Validate triangle kernel names used by inference."""
    if triangle_multiplicative not in TRIANGLE_KERNELS:
        raise ValueError(
            "Invalid triangle_multiplicative. Options: 'auto', 'cuequivariance', 'torch'."
        )
    if triangle_attention not in TRIANGLE_KERNELS:
        raise ValueError(
            "Invalid triangle_attention. Options: 'auto', 'cuequivariance', 'torch'."
        )


def validate_config_triangle_kernels(configs: OpenDDEConfig) -> None:
    validate_triangle_kernels(
        configs.triangle_multiplicative,
        configs.triangle_attention,
    )


def resolve_auto_triangle_kernels(configs: OpenDDEConfig) -> OpenDDEConfig:
    requested_multiplicative = configs.triangle_multiplicative
    requested_attention = configs.triangle_attention
    auto_kernel = select_triangle_kernel()

    if requested_multiplicative == "auto":
        configs.triangle_multiplicative = auto_kernel
    if requested_attention == "auto":
        configs.triangle_attention = auto_kernel

    if "auto" in {requested_multiplicative, requested_attention}:
        logger.info(
            "Resolved triangle kernels from auto to multiplicative=%s, attention=%s.",
            configs.triangle_multiplicative,
            configs.triangle_attention,
        )
    return configs


def validate_triangle_kernel_runtime(configs: OpenDDEConfig) -> None:
    if "cuequivariance" not in {
        configs.triangle_multiplicative,
        configs.triangle_attention,
    }:
        return
    if cuda_acceleration_available():
        return
    if not torch.cuda.is_available():
        raise RuntimeError(
            "cuEquivariance kernels require an NVIDIA CUDA runtime. "
            "Use CPU kernels with '--trimul_kernel torch --triatt_kernel torch' "
            "or install/run in a CUDA environment."
        )
    if not has_cuequivariance_packages():
        raise RuntimeError(
            "cuEquivariance kernels were requested, but GPU optional packages "
            "are missing. Install them with: pip install 'opendde[gpu]'."
        )


def update_gpu_compatible_configs(configs: OpenDDEConfig) -> OpenDDEConfig:
    """
    Update configurations for GPU architectures that need torch fallbacks.
    """
    validate_config_triangle_kernels(configs)
    configs = resolve_auto_triangle_kernels(configs)

    def is_gpu_capability_between_7_and_8() -> bool:
        if not torch.cuda.is_available():
            return False
        major, minor = torch.cuda.get_device_capability()
        cc = major + minor / 10.0
        return 7.0 <= cc < 8.0

    if is_gpu_capability_between_7_and_8():
        configs.dtype = "fp32"
        configs.triangle_attention = "torch"
        configs.triangle_multiplicative = "torch"
        logger.info(
            "Enforcing FP32 and torch kernels for compatibility with detected "
            "GPU (Compute Capability 7.x)."
        )
    validate_triangle_kernel_runtime(configs)
    return configs
