# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
import importlib.metadata
import importlib.util
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

from opendde.config.schema import TriangleKernel


GPU_OPTIONAL_MODULES = (
    "cuequivariance_torch",
    "cuequivariance_ops_torch",
    "triton",
)

GPU_OPTIONAL_DISTRIBUTIONS = {
    "cuequivariance_torch": "cuequivariance-torch",
    "cuequivariance_ops_torch": "cuequivariance-ops-torch-cu12",
    "triton": "triton",
}


@dataclass(frozen=True)
class TorchRuntimeInfo:
    installed: bool
    version: Optional[str] = None
    cuda_available: bool = False
    cuda_version: Optional[str] = None
    device_count: int = 0
    device_name: Optional[str] = None


def module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def distribution_version(module_name: str) -> Optional[str]:
    distribution = GPU_OPTIONAL_DISTRIBUTIONS.get(module_name, module_name)
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def get_torch_runtime_info() -> TorchRuntimeInfo:
    try:
        import torch
    except ImportError:
        return TorchRuntimeInfo(installed=False)

    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    device_name = torch.cuda.get_device_name(0) if device_count else None
    return TorchRuntimeInfo(
        installed=True,
        version=torch.__version__,
        cuda_available=cuda_available,
        cuda_version=torch.version.cuda,
        device_count=device_count,
        device_name=device_name,
    )


def has_nvidia_cuda() -> bool:
    return get_torch_runtime_info().cuda_available


def has_cuequivariance_packages() -> bool:
    return all(module_available(module_name) for module_name in GPU_OPTIONAL_MODULES)


def cuda_acceleration_available() -> bool:
    return has_nvidia_cuda() and has_cuequivariance_packages()


def select_triangle_kernel() -> TriangleKernel:
    return "cuequivariance" if cuda_acceleration_available() else "torch"


def nvidia_smi_summary() -> tuple[bool, Optional[str]]:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return False, None
    try:
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=name,driver_version",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return True, None

    summary = result.stdout.strip().splitlines()[0] if result.stdout.strip() else None
    return True, summary


def format_doctor_report() -> str:
    torch_info = get_torch_runtime_info()
    nvidia_smi_found, nvidia_smi_info = nvidia_smi_summary()
    selected_kernel = select_triangle_kernel()
    optional_lines = []
    for module_name in GPU_OPTIONAL_MODULES:
        status = "installed" if module_available(module_name) else "missing"
        version = distribution_version(module_name)
        if version:
            status = f"{status} ({version})"
        optional_lines.append(f"- {module_name}: {status}")

    if cuda_acceleration_available():
        recommendation = (
            "NVIDIA CUDA is visible and GPU optional packages are installed. "
            "This environment is ready for GPU inference."
        )
    elif torch_info.cuda_available:
        recommendation = (
            "NVIDIA CUDA is visible, but one or more GPU optional packages are "
            "missing. Install them with: pip install 'opendde[gpu]'."
        )
    else:
        recommendation = (
            "No NVIDIA CUDA runtime is visible to PyTorch. The default CPU "
            "install path for this machine is: pip install 'opendde[cpu]'."
        )

    lines = [
        "OpenDDE environment",
        f"- Python: {sys.version.split()[0]}",
        f"- Platform: {platform.platform()}",
        f"- PyTorch: {torch_info.version if torch_info.installed else 'missing'}",
        f"- torch.cuda.is_available: {torch_info.cuda_available}",
        f"- torch CUDA version: {torch_info.cuda_version or 'none'}",
        f"- CUDA device count: {torch_info.device_count}",
        f"- CUDA device 0: {torch_info.device_name or 'none'}",
        f"- nvidia-smi: {nvidia_smi_info or ('found' if nvidia_smi_found else 'not found')}",
        "- GPU optional packages:",
        *optional_lines,
        f"- Selected triangle kernel for auto mode: {selected_kernel}",
        "",
        "Install recommendation",
        "- Default release install path: pip install 'opendde[gpu]'.",
        f"- {recommendation}",
    ]
    return "\n".join(lines)
