# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Fold-CP distributed OuterProductMean reference implementation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from opendde.distributed.foldcp.comm import Ring2DComm
from opendde.distributed.foldcp.layout import FoldCP2DLayout
from opendde.distributed.foldcp.mesh import FoldCPProcessMesh


@dataclass(frozen=True)
class FoldCPMSAShardSpec:
    original_shape: tuple[int, ...]
    padded_shape: tuple[int, ...]
    seq_dim: int
    token_dim: int
    seq_range: tuple[int, int]
    token_range: tuple[int, int]
    mesh_shape: tuple[int, int]
    mesh_coord: tuple[int, int]


def _normalize_dim(ndim: int, dim: int) -> int:
    if dim < 0:
        dim += ndim
    if not (0 <= dim < ndim):
        raise ValueError(f"dim {dim} is outside tensor ndim={ndim}.")
    return dim


def _padded_size(size: int, mesh_side: int) -> int:
    return int(math.ceil(size / mesh_side) * mesh_side)


def _range_for(coord_value: int, total_size: int, mesh_side: int) -> tuple[int, int]:
    tile = total_size // mesh_side
    start = coord_value * tile
    return start, start + tile


def _slice_for_two_dims(
    ndim: int,
    first_dim: int,
    first_range: tuple[int, int],
    second_dim: int,
    second_range: tuple[int, int],
) -> tuple[slice, ...]:
    slices = [slice(None)] * ndim
    slices[first_dim] = slice(*first_range)
    slices[second_dim] = slice(*second_range)
    return tuple(slices)


def shard_msa_tensor_for_opm(
    tensor: torch.Tensor,
    mesh: FoldCPProcessMesh,
    seq_dim: int = 1,
    token_dim: int = 2,
    pad_value: float = 0.0,
) -> tuple[torch.Tensor, FoldCPMSAShardSpec]:
    """Shard an MSA-like tensor over sequence and token axes for OPM."""

    seq_dim = _normalize_dim(tensor.ndim, seq_dim)
    token_dim = _normalize_dim(tensor.ndim, token_dim)
    if seq_dim == token_dim:
        raise ValueError("seq_dim and token_dim must be different.")
    mesh_side = mesh.layout.shape[0]
    padded_seq = _padded_size(tensor.shape[seq_dim], mesh_side)
    padded_token = _padded_size(tensor.shape[token_dim], mesh_side)
    padded_shape = list(tensor.shape)
    padded_shape[seq_dim] = padded_seq
    padded_shape[token_dim] = padded_token

    padded = tensor.new_full(tuple(padded_shape), pad_value)
    original_slices = [slice(None)] * tensor.ndim
    original_slices[seq_dim] = slice(0, tensor.shape[seq_dim])
    original_slices[token_dim] = slice(0, tensor.shape[token_dim])
    padded[tuple(original_slices)] = tensor

    seq_range = _range_for(mesh.coord[0], padded_seq, mesh_side)
    token_range = _range_for(mesh.coord[1], padded_token, mesh_side)
    local = padded[
        _slice_for_two_dims(tensor.ndim, seq_dim, seq_range, token_dim, token_range)
    ].contiguous()
    spec = FoldCPMSAShardSpec(
        original_shape=tuple(tensor.shape),
        padded_shape=tuple(padded_shape),
        seq_dim=seq_dim,
        token_dim=token_dim,
        seq_range=seq_range,
        token_range=token_range,
        mesh_shape=mesh.layout.shape,
        mesh_coord=mesh.coord,
    )
    return local, spec


def serial_outer_product_mean(
    a: torch.Tensor,
    b: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Serial OPM reference over tensors shaped [B, S, N, C]."""

    mask = mask.to(dtype=a.dtype)
    a = a * mask.unsqueeze(-1)
    b = b * mask.unsqueeze(-1)
    outer = torch.einsum("bsic,bsjd->bijcd", a, b)
    norm = torch.einsum("bsi,bsj->bij", mask, mask).clamp_min(0.0)
    return outer / (norm[..., None, None] + eps)


def distributed_outer_product_mean(
    a_local: torch.Tensor,
    b_local: torch.Tensor,
    mask_local: torch.Tensor,
    ring: Ring2DComm,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Compute the local Fold-CP OPM tile using 2D ring communication."""

    if a_local.shape != b_local.shape:
        raise ValueError("a_local and b_local must have the same shape.")
    if mask_local.shape != a_local.shape[:-1]:
        raise ValueError("mask_local shape must match a_local without channel dim.")
    if a_local.ndim != 4:
        raise ValueError("distributed OPM expects [B, S_local, N_local, C].")

    mask_local = mask_local.to(dtype=a_local.dtype)
    a_local = a_local * mask_local.unsqueeze(-1)
    b_local = b_local * mask_local.unsqueeze(-1)

    a_ready = ring.comm_2d_trans.exchange(a_local)
    mask_a_ready = ring.comm_2d_trans.exchange(mask_local)
    a_ready = ring.comm_row_init.exchange(a_ready)
    mask_a_ready = ring.comm_row_init.exchange(mask_a_ready)
    b_ready = ring.comm_col_init.exchange(b_local)
    mask_b_ready = ring.comm_col_init.exchange(mask_local)

    batch, _, n_local, c_hidden = a_local.shape
    out = a_local.new_zeros((batch, n_local, n_local, c_hidden, c_hidden))
    norm = a_local.new_zeros((batch, n_local, n_local))

    for step in range(ring.layout.shape[1]):
        out = out + torch.einsum("bsic,bsjd->bijcd", a_ready, b_ready)
        norm = norm + torch.einsum("bsi,bsj->bij", mask_a_ready, mask_b_ready)
        if step < ring.layout.shape[1] - 1:
            a_ready = ring.comm_row.exchange(a_ready)
            mask_a_ready = ring.comm_row.exchange(mask_a_ready)
            b_ready = ring.comm_col.exchange(b_ready)
            mask_b_ready = ring.comm_col.exchange(mask_b_ready)

    return out / (norm[..., None, None] + eps)


def distributed_outer_product_sum_and_norm(
    a_local: torch.Tensor,
    b_local: torch.Tensor,
    mask_local: torch.Tensor,
    ring: Ring2DComm,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute raw Fold-CP OPM sum and normalization tiles.

    The real OpenDDE ``OuterProductMean`` applies ``linear_out`` before dividing
    by the MSA-pair norm. Returning both tensors preserves that exact order.
    """

    if a_local.shape != b_local.shape:
        raise ValueError("a_local and b_local must have the same shape.")
    if mask_local.shape != a_local.shape[:-1]:
        raise ValueError("mask_local shape must match a_local without channel dim.")
    if a_local.ndim != 4:
        raise ValueError("distributed OPM expects [B, S_local, N_local, C].")

    mask_local = mask_local.to(dtype=a_local.dtype)
    a_local = a_local * mask_local.unsqueeze(-1)
    b_local = b_local * mask_local.unsqueeze(-1)

    a_ready = ring.comm_2d_trans.exchange(a_local)
    mask_a_ready = ring.comm_2d_trans.exchange(mask_local)
    a_ready = ring.comm_row_init.exchange(a_ready)
    mask_a_ready = ring.comm_row_init.exchange(mask_a_ready)
    b_ready = ring.comm_col_init.exchange(b_local)
    mask_b_ready = ring.comm_col_init.exchange(mask_local)

    batch, _, n_local, c_hidden = a_local.shape
    out = a_local.new_zeros((batch, n_local, n_local, c_hidden, c_hidden))
    norm = a_local.new_zeros((batch, n_local, n_local))

    for step in range(ring.layout.shape[1]):
        out = out + torch.einsum("bsic,bsjd->bijcd", a_ready, b_ready)
        norm = norm + torch.einsum("bsi,bsj->bij", mask_a_ready, mask_b_ready)
        if step < ring.layout.shape[1] - 1:
            a_ready = ring.comm_row.exchange(a_ready)
            mask_a_ready = ring.comm_row.exchange(mask_a_ready)
            b_ready = ring.comm_col.exchange(b_ready)
            mask_b_ready = ring.comm_col.exchange(mask_b_ready)

    return out, norm


def opm_output_pair_layout(mesh_shape: tuple[int, int]) -> FoldCP2DLayout:
    return FoldCP2DLayout(mesh_shape)
