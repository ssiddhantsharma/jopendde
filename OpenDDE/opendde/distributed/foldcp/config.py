# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Runtime flags for the Fold-CP migration.

The migration intentionally has one user-facing switch: keep the original
single-card path, or run the four-card context-parallel path once each stage is
ported. Other knobs such as dtype, MSA, templates, and chunk size stay orthogonal.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Literal

FoldCPMode = Literal["single", "distributed"]


def _as_int(value: Any, default: int) -> int:
    if value is None or value == "":
        return default
    return int(value)


@dataclass(frozen=True)
class FoldCPConfig:
    """Validated Fold-CP launch configuration."""

    mode: FoldCPMode = "single"
    size_dp: int = 1
    size_cp: int = 1
    devices: str = ""
    metrics_jsonl: str = ""

    @classmethod
    def from_runtime_args(
        cls,
        *,
        mode: str = "single",
        size_dp: int = 1,
        size_cp: int = 1,
        devices: str = "",
        metrics_jsonl: str = "",
    ) -> "FoldCPConfig":
        return cls(
            mode=mode,  # type: ignore[arg-type]
            size_dp=int(size_dp),
            size_cp=int(size_cp),
            devices=devices,
            metrics_jsonl=metrics_jsonl,
        ).validate()

    @classmethod
    def from_config(cls, configs: Any) -> "FoldCPConfig":
        return cls.from_runtime_args(
            mode=getattr(configs, "foldcp_mode", "single"),
            size_dp=_as_int(getattr(configs, "foldcp_size_dp", 1), 1),
            size_cp=_as_int(getattr(configs, "foldcp_size_cp", 1), 1),
            devices=getattr(configs, "foldcp_devices", "") or "",
            metrics_jsonl=getattr(configs, "foldcp_metrics_jsonl", "") or "",
        )

    def validate(self) -> "FoldCPConfig":
        if self.mode not in {"single", "distributed"}:
            raise ValueError("foldcp_mode must be 'single' or 'distributed'.")
        if self.size_dp < 1:
            raise ValueError("foldcp_size_dp must be >= 1.")
        if self.size_cp < 1:
            raise ValueError("foldcp_size_cp must be >= 1.")
        if self.mode == "single" and self.size_cp != 1:
            raise ValueError("foldcp_mode='single' requires foldcp_size_cp=1.")
        if self.mode == "distributed":
            sqrt_cp = math.isqrt(self.size_cp)
            if sqrt_cp * sqrt_cp != self.size_cp:
                raise ValueError(
                    "foldcp_size_cp must be a perfect square for the 2D CP mesh."
                )
            if self.size_cp == 1:
                raise ValueError(
                    "foldcp_mode='distributed' requires foldcp_size_cp > 1."
                )
        world_size = int(os.environ.get("WORLD_SIZE", "0") or "0")
        if world_size and self.size_dp * self.size_cp != world_size:
            raise ValueError(
                "foldcp_size_dp * foldcp_size_cp must equal WORLD_SIZE "
                f"when launched with torchrun; got {self.size_dp} * "
                f"{self.size_cp} != {world_size}."
            )
        return self

    @property
    def enabled(self) -> bool:
        return self.mode == "distributed"

    @property
    def cp_mesh_shape(self) -> tuple[int, int]:
        side = math.isqrt(self.size_cp)
        return (side, side)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["enabled"] = self.enabled
        data["cp_mesh_shape"] = self.cp_mesh_shape
        return data

    def launch_hint(self) -> str:
        if self.enabled:
            nproc = self.size_dp * self.size_cp
            return (
                f"torchrun --nproc_per_node {nproc} -m runner.batch_inference pred "
                f"--foldcp_mode distributed --foldcp_size_dp {self.size_dp} "
                f"--foldcp_size_cp {self.size_cp}"
            )
        return (
            "python -m runner.batch_inference pred "
            "--foldcp_mode single --foldcp_size_cp 1"
        )


def apply_foldcp_config(configs: Any, foldcp: FoldCPConfig) -> Any:
    """Attach validated Fold-CP settings to the mutable OpenDDE config object."""

    os.environ["OPENDDE_FOLDCP_MODE"] = foldcp.mode
    os.environ["OPENDDE_FOLDCP_SIZE_DP"] = str(foldcp.size_dp)
    os.environ["OPENDDE_FOLDCP_SIZE_CP"] = str(foldcp.size_cp)
    os.environ["OPENDDE_FOLDCP_DEVICES"] = foldcp.devices
    os.environ["OPENDDE_FOLDCP_METRICS_JSONL"] = foldcp.metrics_jsonl

    configs.foldcp_mode = foldcp.mode
    configs.foldcp_size_dp = foldcp.size_dp
    configs.foldcp_size_cp = foldcp.size_cp
    configs.foldcp_devices = foldcp.devices
    configs.foldcp_metrics_jsonl = foldcp.metrics_jsonl
    configs.foldcp = foldcp.to_dict()
    return configs
