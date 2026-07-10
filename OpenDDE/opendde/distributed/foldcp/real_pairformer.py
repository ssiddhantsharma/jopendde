# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Fold-CP adapters for real OpenDDE Pairformer modules."""

from __future__ import annotations

import os

import torch

from opendde.distributed.foldcp.mesh import FoldCPProcessMesh
from opendde.distributed.foldcp.launch import (
    foldcp_linear_with_source_launch_shape,
    foldcp_pair_tile_linear_with_source_chunk_launch,
    foldcp_pair_row_slab_linear_with_source_grid_launch,
)
from opendde.distributed.foldcp.triangular_mult import (
    TriangleMultiplicationDirection,
    distributed_triangle_multiplication,
)
from opendde.distributed.foldcp.pair_sharding import (
    FoldCPPairShardSpec,
    gather_pair_tensor,
    shard_pair_tensor,
)
from opendde.model.modules.primitives import _attention as _single_feature_attention
from opendde.model.triangular.layers import _attention
from opendde.model.utils import permute_final_dims


_TRIATT_BIAS_SOURCE_LAUNCH_MIN_ROWS = 1_048_576
_TRIATT_BIAS_SOURCE_LAUNCH_MAX_ROWS = 1_054_729


def _triatt_bias_source_launch_boundary(source_rows: int) -> bool:
    source_rows = int(source_rows)
    return (
        _TRIATT_BIAS_SOURCE_LAUNCH_MIN_ROWS
        <= source_rows
        <= _TRIATT_BIAS_SOURCE_LAUNCH_MAX_ROWS
    )


def _triatt_query_pad_size(valid_query: int) -> int:
    min_query = 128 if valid_query <= 64 else 192
    return ((max(valid_query, min_query) + 15) // 16) * 16


def _triatt_qkv_row_pad_size(valid_rows: int, original_n: int) -> int:
    if original_n <= 512:
        return original_n
    return ((valid_rows + 15) // 16) * 16


def _triatt_attention_row_chunk_size(valid_rows: int, original_n: int) -> int:
    if original_n <= 1024:
        return valid_rows
    if original_n <= 2048:
        return min(valid_rows, 128)
    if original_n <= 4096:
        return min(valid_rows, 32)
    return min(valid_rows, 16)


def _triatt_wrap_row_chunk_size(
    out_by_row_head_query: torch.Tensor,
    x_local: torch.Tensor,
) -> int:
    value = os.environ.get("OPENDDE_FOLDCP_TRIATT_WRAP_ROW_CHUNK")
    row_chunk_size = int("4" if value is None else value)
    if row_chunk_size <= 0:
        return out_by_row_head_query.shape[0]
    return row_chunk_size


def _pair_transition_flat_chunk_size(z_local: torch.Tensor) -> int:
    value = os.environ.get("OPENDDE_FOLDCP_PAIR_TRANSITION_FLAT_CHUNK")
    return int("262144" if value is None else value)


def _pair_transition_row_pad_size(valid_rows: int, original_n: int) -> int:
    if original_n <= 256:
        return original_n
    return min(original_n, max(valid_rows, 128))


def _linear_output_slice(
    linear: torch.nn.Module,
    x: torch.Tensor,
    output_slice: slice,
) -> torch.Tensor:
    weight = linear.weight[output_slice]
    bias = None if linear.bias is None else linear.bias[output_slice]
    if getattr(linear, "precision", None) is not None:
        precision = linear.precision
        with torch.amp.autocast("cuda", enabled=False):
            x_precision = x.to(dtype=precision)
            weight_precision = weight.to(dtype=precision)
            bias = None if bias is None else bias.to(dtype=precision)
            return torch.nn.functional.linear(
                x_precision,
                weight_precision,
                bias,
            ).to(dtype=x.dtype)
    if x.dtype is torch.bfloat16:
        with torch.amp.autocast("cuda", enabled=False):
            bias = None if bias is None else bias.to(dtype=x.dtype)
            return torch.nn.functional.linear(x, weight.to(dtype=x.dtype), bias)
    return torch.nn.functional.linear(x, weight, bias)


def _linear_with_exact_source_launch_shape(
    linear: torch.nn.Module,
    x: torch.Tensor,
    *,
    source_rows: int,
) -> torch.Tensor:
    """Run a Linear with the exact source flat-row launch shape.

    This is only used at CUDA launch-family boundaries where the regular
    Fold-CP bucket is insufficient for bitwise parity. It pads owned rows with
    zeros and discards them after projection; it never gathers remote pair data.
    """

    local_rows = int(x.numel() // x.shape[-1]) if x.shape[-1] else 0
    source_rows = int(source_rows)
    if source_rows <= local_rows:
        return linear(x)
    flat = x.contiguous().reshape(local_rows, x.shape[-1])
    launch = flat.new_zeros(source_rows, flat.shape[-1])
    launch[:local_rows].copy_(flat)
    projected = linear(launch)[:local_rows]
    return projected.reshape(*x.shape[:-1], -1)


def _linear_pair_tile_with_source_grid_launch(
    linear: torch.nn.Module,
    x: torch.Tensor,
    *,
    original_n: int,
    row_start: int,
    col_start: int,
    valid_rows: int,
    valid_cols: int,
) -> torch.Tensor:
    """Project a local pair tile with the source full-pair flat launch layout."""

    if valid_rows <= 0 or valid_cols <= 0:
        return linear(x)
    source_rows = int(original_n) * int(original_n)
    flat = x.contiguous().reshape(-1, x.shape[-1])
    launch = flat.new_zeros(source_rows, flat.shape[-1])
    row_offsets = (
        (torch.arange(valid_rows, device=x.device) + int(row_start)) * int(original_n)
        + int(col_start)
    )
    source_index = (
        row_offsets[:, None]
        + torch.arange(valid_cols, device=x.device)[None, :]
    ).reshape(-1)
    tile_index = (
        torch.arange(valid_rows, device=x.device)[:, None] * x.shape[-2]
        + torch.arange(valid_cols, device=x.device)[None, :]
    ).reshape(-1)
    launch.index_copy_(0, source_index, flat.index_select(0, tile_index))
    projected = linear(launch).index_select(0, source_index)
    out = projected.new_zeros(x.shape[:-1] + (projected.shape[-1],))
    out_flat = out.reshape(-1, projected.shape[-1])
    out_flat.index_copy_(0, tile_index, projected)
    return out


def _ring_gather_by_row(
    local_tensor: torch.Tensor,
    mesh: FoldCPProcessMesh,
    dim: int,
    length: int | None = None,
) -> torch.Tensor:
    side = mesh.layout.shape[1]
    if side == 1:
        out = local_tensor
    else:
        ring = mesh.ring_comm()
        gathered: list[torch.Tensor | None] = [None for _ in range(side)]
        gathered[mesh.coord[1]] = local_tensor.contiguous()
        ready = gathered[mesh.coord[1]]
        for step in range(1, side):
            ready = ring.comm_row.exchange(ready.contiguous())
            source_col = (mesh.coord[1] + step) % side
            gathered[source_col] = ready
        if any(item is None for item in gathered):
            raise RuntimeError("failed to gather row ring blocks.")
        out = torch.cat([item for item in gathered if item is not None], dim=dim)
    if length is not None:
        dim = dim if dim >= 0 else out.ndim + dim
        out = out.narrow(dim, 0, length)
    return out.contiguous()


def _ring_gather_by_col(
    local_tensor: torch.Tensor,
    mesh: FoldCPProcessMesh,
    dim: int,
    length: int | None = None,
) -> torch.Tensor:
    side = mesh.layout.shape[0]
    if side == 1:
        out = local_tensor
    else:
        ring = mesh.ring_comm()
        gathered: list[torch.Tensor | None] = [None for _ in range(side)]
        gathered[mesh.coord[0]] = local_tensor.contiguous()
        ready = gathered[mesh.coord[0]]
        for step in range(1, side):
            ready = ring.comm_col.exchange(ready.contiguous())
            source_row = (mesh.coord[0] + step) % side
            gathered[source_row] = ready
        if any(item is None for item in gathered):
            raise RuntimeError("failed to gather column ring blocks.")
        out = torch.cat([item for item in gathered if item is not None], dim=dim)
    if length is not None:
        dim = dim if dim >= 0 else out.ndim + dim
        out = out.narrow(dim, 0, length)
    return out.contiguous()


def _triangle_source_column_chunks(
    n_token: int,
    chunk_size: int = 256,
) -> list[tuple[int, int]]:
    half_n = n_token // 2 + n_token % 2
    chunks: list[tuple[int, int]] = []
    starts = list(range(0, half_n, chunk_size))
    for start, next_start in zip(starts, starts[1:] + [half_n]):
        chunks.append((start, next_start))
    for start in range(half_n, n_token, chunk_size):
        chunks.append((start, min(start + chunk_size, n_token)))
    return chunks


def _triangle_source_matmul_row_size(valid_rows: int, original_n: int) -> int:
    if original_n <= 1024:
        return original_n
    return valid_rows




def _triangle_layer_norm_source_row_slab(
    layer_norm: torch.nn.Module,
    z_in: torch.Tensor,
    mesh: FoldCPProcessMesh,
    z_spec: FoldCPPairShardSpec,
) -> torch.Tensor:
    original_n = int(z_spec.original_shape[z_spec.pair_dims[0]])
    row_start, row_end = z_spec.row_range
    col_start, col_end = z_spec.col_range
    valid_row_end = min(row_end, original_n)
    valid_col_end = min(col_end, original_n)
    valid_rows = max(0, valid_row_end - row_start)
    valid_cols = max(0, valid_col_end - col_start)
    out = z_in.new_zeros(z_in.shape)
    if valid_rows == 0 or valid_cols == 0:
        return out
    row_slab = z_in.new_zeros(z_in.shape[:-3] + (valid_rows, original_n, z_in.shape[-1]))
    row_slab[..., :, col_start:valid_col_end, :] = z_in[..., :valid_rows, :valid_cols, :]
    normed = layer_norm(row_slab)
    out[..., :valid_rows, :valid_cols, :] = normed[..., :, col_start:valid_col_end, :]
    return out.contiguous()


def _triangle_project_source_launch(
    linear_g: torch.nn.Module,
    linear_p: torch.nn.Module,
    x: torch.Tensor,
    mask: torch.Tensor,
    *,
    source_rows: int,
    source_unbatched: bool = False,
    layer_norm: torch.nn.Module | None = None,
    original_n: int | None = None,
    row_start: int = 0,
    col_start: int = 0,
    source_chunk_rows: int | None = None,
    source_chunk_cols: int | None = None,
) -> torch.Tensor:
    if source_unbatched and x.ndim == 4 and x.shape[0] == 1:
        projected = _triangle_project_source_launch(
            linear_g,
            linear_p,
            x.squeeze(0),
            mask.squeeze(0),
            source_rows=source_rows,
            source_unbatched=False,
            layer_norm=layer_norm,
            original_n=original_n,
            row_start=row_start,
            col_start=col_start,
            source_chunk_rows=source_chunk_rows,
            source_chunk_cols=source_chunk_cols,
        )
        return projected.unsqueeze(0)
    if layer_norm is not None:
        x = layer_norm(x)
    if source_chunk_rows is not None and source_chunk_cols is not None and x.ndim in (3, 4):
        if x.ndim == 4:
            if x.shape[0] != 1:
                raise ValueError("triangle source-chunk launch expects batch size 1.")
            x_for_linear = x.squeeze(0)
            unsqueeze_batch = True
        else:
            x_for_linear = x
            unsqueeze_batch = False
        gate = foldcp_pair_tile_linear_with_source_chunk_launch(
            linear_g,
            x_for_linear,
            source_rows=source_chunk_rows,
            source_cols=source_chunk_cols,
            row_start=row_start,
            col_start=col_start,
        )
        proj = foldcp_pair_tile_linear_with_source_chunk_launch(
            linear_p,
            x_for_linear,
            source_rows=source_chunk_rows,
            source_cols=source_chunk_cols,
            row_start=row_start,
            col_start=col_start,
        )
        if unsqueeze_batch:
            gate = gate.unsqueeze(0)
            proj = proj.unsqueeze(0)
    elif original_n is not None and x.ndim in (3, 4):
        if x.ndim == 4:
            if x.shape[0] != 1:
                raise ValueError("triangle source-grid launch expects batch size 1.")
            x_for_linear = x.squeeze(0)
            unsqueeze_batch = True
        else:
            x_for_linear = x
            unsqueeze_batch = False
        gate = foldcp_pair_row_slab_linear_with_source_grid_launch(
            linear_g,
            x_for_linear,
            original_n=original_n,
            row_start=row_start,
            col_start=col_start,
        )
        proj = foldcp_pair_row_slab_linear_with_source_grid_launch(
            linear_p,
            x_for_linear,
            original_n=original_n,
            row_start=row_start,
            col_start=col_start,
        )
        if unsqueeze_batch:
            gate = gate.unsqueeze(0)
            proj = proj.unsqueeze(0)
    else:
        gate = foldcp_linear_with_source_launch_shape(
            linear_g,
            x,
            source_rows=source_rows,
        )
        proj = foldcp_linear_with_source_launch_shape(
            linear_p,
            x,
            source_rows=source_rows,
        )
    out = torch.sigmoid(gate)
    out *= proj
    out *= mask
    return out


def _triangle_a_projection_source_chunks(
    module: torch.nn.Module,
    z_norm: torch.Tensor,
    mask: torch.Tensor,
    z_spec: FoldCPPairShardSpec,
    *,
    source_unbatched: bool = False,
) -> torch.Tensor:
    original_n = int(z_spec.original_shape[z_spec.pair_dims[0]])
    row_start, row_end = z_spec.row_range
    col_start, col_end = z_spec.col_range
    valid_row_end = min(row_end, original_n)
    valid_col_end = min(col_end, original_n)
    valid_rows = max(0, valid_row_end - row_start)
    valid_cols = max(0, valid_col_end - col_start)
    out = z_norm.new_zeros(z_norm.shape[:-1] + (int(module.c_hidden),))
    if valid_rows == 0 or valid_cols == 0:
        return out

    source_row_chunk = 256
    for global_start in range(0, original_n, source_row_chunk):
        global_end = min(global_start + source_row_chunk, original_n)
        overlap_start = max(global_start, row_start)
        overlap_end = min(global_end, valid_row_end)
        if overlap_start >= overlap_end:
            continue
        local_row_slice = slice(overlap_start - row_start, overlap_end - row_start)
        source_rows = int(global_end - global_start) * int(original_n)
        projected = _triangle_project_source_launch(
            module.linear_a_g,
            module.linear_a_p,
            z_norm[..., local_row_slice, :valid_cols, :],
            mask[..., local_row_slice, :valid_cols, :],
            source_rows=source_rows,
            source_unbatched=source_unbatched,
            row_start=overlap_start - global_start,
            col_start=col_start,
            source_chunk_rows=global_end - global_start,
            source_chunk_cols=original_n,
        )
        out[..., local_row_slice, :valid_cols, :] = projected
    return out.contiguous()


def _triangle_b_projection_source_chunk(
    module: torch.nn.Module,
    z_in: torch.Tensor,
    mask: torch.Tensor,
    mesh: FoldCPProcessMesh,
    direction: TriangleMultiplicationDirection,
    z_spec: FoldCPPairShardSpec,
    *,
    source_unbatched: bool = False,
) -> torch.Tensor:
    original_n = z_spec.original_shape[z_spec.pair_dims[0]]
    row_start, row_end = z_spec.row_range
    col_start, col_end = z_spec.col_range
    valid_row_end = min(row_end, original_n)
    valid_col_end = min(col_end, original_n)
    valid_rows = max(0, valid_row_end - row_start)
    valid_cols = max(0, valid_col_end - col_start)
    out = z_in.new_zeros(z_in.shape[:-1] + (int(module.c_hidden),))
    if valid_rows == 0 or valid_cols == 0:
        return out

    if direction == TriangleMultiplicationDirection.OUTGOING:
        z_slab = _ring_gather_by_row(z_in, mesh, dim=-2, length=original_n)
        mask_slab = _ring_gather_by_row(mask, mesh, dim=-2, length=original_n)
        z_slab = z_slab[..., :valid_rows, :, :]
        mask_slab = mask_slab[..., :valid_rows, :, :]
        for global_start, global_end in _triangle_source_column_chunks(original_n):
            overlap_start = max(global_start, row_start)
            overlap_end = min(global_end, valid_row_end)
            if overlap_start >= overlap_end:
                continue
            local_row_slice = slice(overlap_start - row_start, overlap_end - row_start)
            projected = _triangle_project_source_launch(
                module.linear_b_g,
                module.linear_b_p,
                z_slab[..., local_row_slice, :, :],
                mask_slab[..., local_row_slice, :, :],
                source_rows=int(global_end - global_start) * int(original_n),
                source_unbatched=source_unbatched,
                layer_norm=module.layer_norm_in,
                row_start=overlap_start - global_start,
                col_start=0,
                source_chunk_rows=global_end - global_start,
                source_chunk_cols=original_n,
            )
            out[..., local_row_slice, :valid_cols, :] = projected[
                ..., :, col_start:valid_col_end, :
            ]
        return out.contiguous()

    if direction == TriangleMultiplicationDirection.INCOMING:
        z_slab = _ring_gather_by_col(z_in, mesh, dim=-3, length=original_n)
        mask_slab = _ring_gather_by_col(mask, mesh, dim=-3, length=original_n)
        z_slab = z_slab[..., :, :valid_cols, :]
        mask_slab = mask_slab[..., :, :valid_cols, :]
        for global_start, global_end in _triangle_source_column_chunks(original_n):
            overlap_start = max(global_start, col_start)
            overlap_end = min(global_end, valid_col_end)
            if overlap_start >= overlap_end:
                continue
            local_col_slice = slice(overlap_start - col_start, overlap_end - col_start)
            projected = _triangle_project_source_launch(
                module.linear_b_g,
                module.linear_b_p,
                z_slab[..., :, local_col_slice, :],
                mask_slab[..., :, local_col_slice, :],
                source_rows=int(original_n) * int(global_end - global_start),
                source_unbatched=source_unbatched,
                layer_norm=module.layer_norm_in,
                row_start=0,
                col_start=overlap_start - global_start,
                source_chunk_rows=original_n,
                source_chunk_cols=global_end - global_start,
            )
            out[..., :valid_rows, local_col_slice, :] = projected[
                ..., row_start:valid_row_end, :, :
            ]
        return out.contiguous()

    raise ValueError(f"unsupported direction={direction}")


def _distributed_triangle_multiplication_source_matmul(
    a_local: torch.Tensor,
    b_local: torch.Tensor,
    mesh: FoldCPProcessMesh,
    direction: TriangleMultiplicationDirection,
    z_spec: FoldCPPairShardSpec,
) -> torch.Tensor:
    """Compute a local triangle-multiplication tile with source-like matmuls.

    The ring implementation is mathematically correct but sums the K dimension
    as separate per-rank matmuls.  The OpenDDE inference path computes each
    output-column chunk with one full-K matmul.  This variant keeps the output
    sharded while gathering only the full-K projected inputs needed for the
    current local tile, preserving that source matmul shape.
    """

    if a_local.ndim != 4 or b_local.ndim != 4:
        raise ValueError("triangle multiplication expects [B, N, N, C] inputs.")
    if a_local.shape != b_local.shape:
        raise ValueError("a_local and b_local must have the same shape.")

    local_rows = a_local.shape[-3]
    local_cols = a_local.shape[-2]
    update = a_local.new_zeros(a_local.shape)
    original_n = z_spec.original_shape[z_spec.pair_dims[0]]
    row_start, row_end = z_spec.row_range
    col_start, col_end = z_spec.col_range

    if direction == TriangleMultiplicationDirection.OUTGOING:
        a_full_k = _ring_gather_by_row(a_local, mesh, dim=-2, length=original_n)
        b_trans = mesh.ring_comm().comm_2d_trans.exchange(b_local.contiguous())
        valid_row_end = min(row_end, original_n)
        valid_col_end = min(col_end, original_n)
        valid_rows = max(0, valid_row_end - row_start)
        valid_cols = max(0, valid_col_end - col_start)
        for global_start, global_end in _triangle_source_column_chunks(original_n):
            overlap_start = max(global_start, col_start)
            overlap_end = min(global_end, col_end, original_n)
            if overlap_start >= overlap_end:
                continue
            local_col_slice = slice(overlap_start - col_start, overlap_end - col_start)
            b_full_k = _ring_gather_by_col(
                b_trans[..., local_col_slice, :, :],
                mesh,
                dim=-2,
                length=original_n,
            )
            valid_row_end = min(row_end, original_n)
            valid_rows = max(0, valid_row_end - row_start)
            matmul_rows = _triangle_source_matmul_row_size(valid_rows, original_n)
            a_mat_input = a_full_k[..., :valid_rows, :, :]
            if matmul_rows != valid_rows:
                a_padded = a_full_k.new_zeros(
                    a_full_k.shape[:-3]
                    + (matmul_rows, a_full_k.shape[-2], a_full_k.shape[-1])
                )
                row_slice = (
                    slice(row_start, valid_row_end)
                    if matmul_rows == original_n
                    else slice(0, valid_rows)
                )
                a_padded[..., row_slice, :, :] = a_mat_input
                a_mat_input = a_padded
            chunk = torch.matmul(
                a_mat_input.permute(0, 3, 1, 2).contiguous(),
                b_full_k.permute(0, 3, 2, 1).contiguous(),
            ).permute(0, 2, 3, 1)
            if matmul_rows != valid_rows:
                row_slice = (
                    slice(row_start, valid_row_end)
                    if matmul_rows == original_n
                    else slice(0, valid_rows)
                )
                chunk = chunk[..., row_slice, :, :]
            update[..., :valid_rows, local_col_slice, :] = chunk
        return update.contiguous()

    if direction == TriangleMultiplicationDirection.INCOMING:
        a_trans = mesh.ring_comm().comm_2d_trans.exchange(a_local.contiguous())
        b_full_k = _ring_gather_by_col(b_local, mesh, dim=-3, length=original_n)
        valid_row_end = min(row_end, original_n)
        valid_col_end = min(col_end, original_n)
        if row_start < valid_row_end:
            a_full_k = _ring_gather_by_row(
                a_trans[..., : valid_row_end - row_start, :],
                mesh,
                dim=-3,
                length=original_n,
            )
            valid_rows = valid_row_end - row_start
            valid_cols = max(0, valid_col_end - col_start)
            for global_start, global_end in _triangle_source_column_chunks(original_n):
                overlap_start = max(global_start, col_start)
                overlap_end = min(global_end, col_end, original_n)
                if overlap_start >= overlap_end:
                    continue
                local_col_slice = slice(overlap_start - col_start, overlap_end - col_start)
                valid_rows = valid_row_end - row_start
                matmul_rows = _triangle_source_matmul_row_size(valid_rows, original_n)
                a_mat_input = a_full_k
                if matmul_rows != valid_rows:
                    a_padded = a_full_k.new_zeros(
                        a_full_k.shape[:-2] + (matmul_rows, a_full_k.shape[-1])
                    )
                    row_slice = (
                        slice(row_start, valid_row_end)
                        if matmul_rows == original_n
                        else slice(0, valid_rows)
                    )
                    a_padded[..., row_slice, :] = a_full_k
                    a_mat_input = a_padded
                chunk = torch.matmul(
                    a_mat_input.permute(0, 3, 2, 1).contiguous(),
                    b_full_k[..., local_col_slice, :].permute(0, 3, 1, 2).contiguous(),
                ).permute(0, 2, 3, 1)
                if matmul_rows != valid_rows:
                    row_slice = (
                        slice(row_start, valid_row_end)
                        if matmul_rows == original_n
                        else slice(0, valid_rows)
                    )
                    chunk = chunk[..., row_slice, :, :]
                update[..., :valid_rows, local_col_slice, :] = chunk
        return update.contiguous()

    raise ValueError(f"unsupported direction={direction}")


def _triangle_multiplication_output_norm_gate(
    module: torch.nn.Module,
    update: torch.Tensor,
    z_norm: torch.Tensor,
    mesh: FoldCPProcessMesh | None = None,
) -> torch.Tensor:
    flat_chunk_size = int(
        os.environ.get("OPENDDE_FOLDCP_TRIMUL_OUTPUT_GATE_FLAT_CHUNK", "262144")
    )
    if flat_chunk_size <= 0:
        update = module.layer_norm_out(update)
        update = module.linear_z(update)
        update = update * torch.sigmoid(module.linear_g(z_norm))
        return update

    flat_update = update.reshape(-1, update.shape[-1])
    flat_z_norm = z_norm.reshape(-1, z_norm.shape[-1])
    c_z = int(module.c_z)
    write_inplace = (not torch.is_grad_enabled()) and update.shape[-1] == c_z
    out = flat_update if write_inplace else flat_update.new_empty((flat_update.shape[0], c_z))
    for start in range(0, flat_update.shape[0], flat_chunk_size):
        end = min(start + flat_chunk_size, flat_update.shape[0])
        norm_chunk = module.layer_norm_out(flat_update[start:end])
        out_chunk = module.linear_z(norm_chunk)
        gate_chunk = torch.sigmoid(module.linear_g(flat_z_norm[start:end]))
        out[start:end] = out_chunk * gate_chunk
    if write_inplace:
        return update
    return out.reshape(update.shape[:-1] + (c_z,))


def _triangle_multiplication_output_norm_gate_source_slab(
    module: torch.nn.Module,
    update: torch.Tensor,
    z_source: torch.Tensor,
    mesh: FoldCPProcessMesh,
    z_spec: FoldCPPairShardSpec,
    *,
    source_unbatched: bool = False,
) -> torch.Tensor:
    original_n = z_spec.original_shape[z_spec.pair_dims[0]]
    row_start, row_end = z_spec.row_range
    col_start, col_end = z_spec.col_range
    valid_row_end = min(row_end, original_n)
    out = update.new_zeros(update.shape[:-1] + (int(module.c_z),))
    if row_start >= valid_row_end:
        return out

    for global_start, global_end in _triangle_source_column_chunks(original_n):
        overlap_start = max(global_start, col_start)
        overlap_end = min(global_end, col_end, original_n)
        if overlap_start >= overlap_end:
            continue
        local_col_slice = slice(overlap_start - col_start, overlap_end - col_start)
        update_slab = _ring_gather_by_col(
            update[..., :, local_col_slice, :],
            mesh,
            dim=-3,
            length=original_n,
        )
        z_slab = _ring_gather_by_col(
            z_source[..., :, local_col_slice, :],
            mesh,
            dim=-3,
            length=original_n,
        )
        if source_unbatched and update_slab.ndim == 4 and update_slab.shape[0] == 1:
            slab_3d = update_slab.squeeze(0)
            z_slab_3d = z_slab.squeeze(0)
            slab_norm = module.layer_norm_out(slab_3d)
            slab = foldcp_pair_tile_linear_with_source_chunk_launch(
                module.linear_z,
                slab_norm,
                source_rows=original_n,
                source_cols=global_end - global_start,
                row_start=0,
                col_start=overlap_start - global_start,
            )
            gate_norm = module.layer_norm_in(z_slab_3d)
            gate = foldcp_pair_tile_linear_with_source_chunk_launch(
                module.linear_g,
                gate_norm,
                source_rows=original_n,
                source_cols=global_end - global_start,
                row_start=0,
                col_start=overlap_start - global_start,
            )
            slab = (slab * torch.sigmoid(gate)).unsqueeze(0)
        else:
            slab_norm = module.layer_norm_out(update_slab)
            slab = foldcp_pair_tile_linear_with_source_chunk_launch(
                module.linear_z,
                slab_norm,
                source_rows=original_n,
                source_cols=global_end - global_start,
                row_start=0,
                col_start=overlap_start - global_start,
            )
            gate_norm = module.layer_norm_in(z_slab)
            gate = foldcp_pair_tile_linear_with_source_chunk_launch(
                module.linear_g,
                gate_norm,
                source_rows=original_n,
                source_cols=global_end - global_start,
                row_start=0,
                col_start=overlap_start - global_start,
            )
            slab = slab * torch.sigmoid(gate)
        out[..., : valid_row_end - row_start, local_col_slice, :] = slab[
            ..., row_start:valid_row_end, :, :
        ]
    return out.contiguous()

def _transpose_pair_spec(z_spec: FoldCPPairShardSpec) -> FoldCPPairShardSpec:
    row_dim, col_dim = z_spec.pair_dims
    original_shape = list(z_spec.original_shape)
    padded_shape = list(z_spec.padded_shape)
    original_shape[row_dim], original_shape[col_dim] = (
        original_shape[col_dim],
        original_shape[row_dim],
    )
    padded_shape[row_dim], padded_shape[col_dim] = (
        padded_shape[col_dim],
        padded_shape[row_dim],
    )
    return FoldCPPairShardSpec(
        original_shape=tuple(original_shape),
        padded_shape=tuple(padded_shape),
        pair_dims=z_spec.pair_dims,
        row_range=z_spec.col_range,
        col_range=z_spec.row_range,
        mesh_shape=z_spec.mesh_shape,
        mesh_coord=(z_spec.mesh_coord[1], z_spec.mesh_coord[0]),
    )


def distributed_triangle_multiplication_update(
    module: torch.nn.Module,
    z_local: torch.Tensor,
    mesh: FoldCPProcessMesh,
    mask_local: torch.Tensor | None = None,
    residual_local: torch.Tensor | None = None,
    z_spec: FoldCPPairShardSpec | None = None,
) -> torch.Tensor:
    """Run a real TriangleMultiplication module on a Fold-CP local pair tile.

    The OpenDDE module owns layernorm, projections, gates, and output projection.
    Fold-CP only replaces the full `sum_k` triangular BMM with the 2D ring
    contraction over sharded projected pair tiles.
    """

    if z_local.ndim == 3:
        z_in = z_local.unsqueeze(0)
        squeeze_batch = True
    elif z_local.ndim == 4:
        z_in = z_local
        squeeze_batch = False
    else:
        raise ValueError("z_local must be [N, N, C] or [B, N, N, C].")

    if mask_local is None:
        mask = z_in.new_ones(z_in.shape[:-1])
    elif mask_local.ndim == 2:
        mask = mask_local.unsqueeze(0)
    elif mask_local.ndim == 3:
        mask = mask_local
    else:
        raise ValueError("mask_local must be [N, N] or [B, N, N].")
    mask = mask.unsqueeze(-1)

    if z_spec is None:
        z_norm = module.layer_norm_in(z_in)
    else:
        z_norm = _triangle_layer_norm_source_row_slab(
            module.layer_norm_in,
            z_in,
            mesh,
            z_spec,
        )

    direction = (
        TriangleMultiplicationDirection.OUTGOING
        if bool(module._outgoing)
        else TriangleMultiplicationDirection.INCOMING
    )
    project_chunk_size = int(
        os.environ.get(
            "OPENDDE_FOLDCP_TRIMUL_PROJECT_CHANNEL_CHUNK",
            os.environ.get("OPENDDE_FOLDCP_TRIMUL_CHANNEL_CHUNK", "0"),
        )
    )
    source_pair_rows = None
    if z_spec is not None:
        row_dim, col_dim = z_spec.pair_dims
        source_pair_rows = (
            int(z_spec.original_shape[row_dim])
            * int(z_spec.original_shape[col_dim])
        )

    def project_linear(linear: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        if source_pair_rows is None:
            return linear(x)
        return foldcp_linear_with_source_launch_shape(
            linear,
            x,
            source_rows=source_pair_rows,
        )
    if 0 < project_chunk_size < int(module.c_hidden):
        update = z_in.new_empty(z_in.shape[:-1] + (int(module.c_hidden),))
        for channel_start in range(0, int(module.c_hidden), project_chunk_size):
            channel_end = min(channel_start + project_chunk_size, int(module.c_hidden))
            channel_slice = slice(channel_start, channel_end)
            a_local = torch.sigmoid(
                _linear_output_slice(module.linear_a_g, z_norm, channel_slice)
            )
            a_local *= _linear_output_slice(module.linear_a_p, z_norm, channel_slice)
            a_local *= mask
            b_local = torch.sigmoid(
                _linear_output_slice(module.linear_b_g, z_norm, channel_slice)
            )
            b_local *= _linear_output_slice(module.linear_b_p, z_norm, channel_slice)
            b_local *= mask
            if z_spec is None:
                update[..., channel_slice] = distributed_triangle_multiplication(
                    a_local,
                    b_local,
                    mesh.ring_comm(),
                    direction,
                )
            else:
                update[..., channel_slice] = (
                    _distributed_triangle_multiplication_source_matmul(
                        a_local,
                        b_local,
                        mesh,
                        direction,
                        z_spec,
                    )
                )
        del a_local, b_local
    else:
        if z_spec is None:
            a_local = torch.sigmoid(project_linear(module.linear_a_g, z_norm))
            a_local *= project_linear(module.linear_a_p, z_norm)
            a_local *= mask
        else:
            a_local = _triangle_a_projection_source_chunks(
                module,
                z_norm,
                mask,
                z_spec,
                source_unbatched=squeeze_batch,
            )
        if z_spec is None:
            b_local = torch.sigmoid(project_linear(module.linear_b_g, z_norm))
            b_local *= project_linear(module.linear_b_p, z_norm)
            b_local *= mask
        else:
            b_local = _triangle_b_projection_source_chunk(
                module,
                z_in,
                mask,
                mesh,
                direction,
                z_spec,
                source_unbatched=squeeze_batch,
            )
        if z_spec is None:
            update = distributed_triangle_multiplication(
                a_local,
                b_local,
                mesh.ring_comm(),
                direction,
            )
        else:
            update = _distributed_triangle_multiplication_source_matmul(
                a_local,
                b_local,
                mesh,
                direction,
                z_spec,
            )
        del a_local, b_local
    if z_spec is None:
        update = _triangle_multiplication_output_norm_gate(module, update, z_norm, mesh)
    else:
        update = _triangle_multiplication_output_norm_gate_source_slab(
            module,
            update,
            z_in,
            mesh,
            z_spec,
            source_unbatched=squeeze_batch,
        )

    if squeeze_batch:
        update = update.squeeze(0)
    if residual_local is not None:
        residual_local += update
        return residual_local.contiguous()
    return update.contiguous()


def distributed_pair_transition_update(
    transition: torch.nn.Module,
    z_local: torch.Tensor,
    mesh: FoldCPProcessMesh | None = None,
    residual_local: torch.Tensor | None = None,
    z_spec: FoldCPPairShardSpec | None = None,
) -> torch.Tensor:
    """Run the real pair transition on a local Fold-CP pair tile.

    Pair transition is pointwise over the two token axes, so no CP communication
    is required. The memory win comes from never materializing the full
    `[N, N, C] -> [N, N, n*C]` transition workspace on one rank.
    """

    if z_spec is not None and mesh is not None and not torch.is_grad_enabled():
        original_n = z_spec.original_shape[z_spec.pair_dims[0]]
        row_start, row_end = z_spec.row_range
        col_start, col_end = z_spec.col_range
        valid_rows = max(0, min(row_end, original_n) - row_start)
        valid_cols = max(0, min(col_end, original_n) - col_start)
        if valid_rows == 0 or valid_cols == 0:
            update = z_local.new_zeros(z_local.shape)
            if residual_local is not None:
                residual_local += update
                return residual_local.contiguous()
            return update

        z_row_slab = _ring_gather_by_row(z_local, mesh, dim=-2, length=original_n)
        row_pad = _pair_transition_row_pad_size(valid_rows, original_n)
        source_launch_sensitive = valid_rows * original_n >= 2_097_152
        if source_launch_sensitive and (
            row_start > 0 or col_start > 0 or valid_rows > 1024
        ):
            row_pad = max(row_pad, original_n)
        launch_row_start = row_start if row_pad == original_n else 0
        if row_pad != z_row_slab.shape[-3]:
            z_padded = z_row_slab.new_zeros(
                z_row_slab.shape[:-3]
                + (row_pad, z_row_slab.shape[-2], z_row_slab.shape[-1])
            )
            z_padded[
                ...,
                launch_row_start : launch_row_start + valid_rows,
                :,
                :,
            ] = z_row_slab[..., :valid_rows, :, :]
        else:
            z_padded = z_row_slab
        update_row_slab = transition(z_padded)
        col_start = z_spec.col_range[0]
        update = z_local.new_zeros(z_local.shape)
        update[..., :valid_rows, :valid_cols, :] = update_row_slab[
            ...,
            launch_row_start : launch_row_start + valid_rows,
            col_start : col_start + valid_cols,
            :,
        ]
        del z_row_slab, z_padded, update_row_slab
        if residual_local is not None:
            residual_local += update
            return residual_local.contiguous()
        return update

    flat_chunk_size = _pair_transition_flat_chunk_size(z_local)
    if flat_chunk_size <= 0:
        update = transition(z_local)
        if residual_local is not None:
            if torch.is_grad_enabled():
                return (residual_local + update).contiguous()
            residual_local += update
            return residual_local.contiguous()
        return update.contiguous()

    flat = z_local.reshape(-1, z_local.shape[-1])
    if residual_local is not None:
        if torch.is_grad_enabled():
            out = flat.new_empty((flat.shape[0], z_local.shape[-1]))
            for start in range(0, flat.shape[0], flat_chunk_size):
                end = min(start + flat_chunk_size, flat.shape[0])
                out[start:end] = transition(flat[start:end])
            return (residual_local + out.reshape_as(z_local)).contiguous()

        residual_flat = residual_local.reshape(-1, residual_local.shape[-1])
        for start in range(0, flat.shape[0], flat_chunk_size):
            end = min(start + flat_chunk_size, flat.shape[0])
            residual_flat[start:end] += transition(flat[start:end])
        return residual_local.contiguous()

    out = flat.new_empty((flat.shape[0], z_local.shape[-1]))
    for start in range(0, flat.shape[0], flat_chunk_size):
        end = min(start + flat_chunk_size, flat.shape[0])
        out[start:end] = transition(flat[start:end])
    return out.reshape_as(z_local).contiguous()


def _gather_single_update_by_col_ring(
    local_update: torch.Tensor,
    n_token: int,
    mesh: FoldCPProcessMesh,
) -> torch.Tensor:
    """Gather row-sharded single updates without a column all-gather collective."""

    local_update = local_update.contiguous()
    side = mesh.layout.shape[0]
    if side == 1:
        return local_update[:n_token].contiguous()

    ring = mesh.ring_comm()
    gathered: list[torch.Tensor | None] = [None for _ in range(side)]
    gathered[mesh.coord[0]] = local_update

    ready = local_update
    for step in range(1, side):
        ready = ring.comm_col.exchange(ready.contiguous())
        source_row = (mesh.coord[0] + step) % side
        gathered[source_row] = ready

    if any(item is None for item in gathered):
        raise RuntimeError("failed to collect AttentionPairBias single update.")
    full_update = torch.cat([item for item in gathered if item is not None], dim=-2)
    return full_update[:n_token].contiguous()


def _gather_single_update_by_2d_ring(
    local_update: torch.Tensor,
    n_token: int,
    mesh: FoldCPProcessMesh,
    pair_row_tile: int,
) -> torch.Tensor:
    """Gather rank-sharded single updates with Fold-CP row then column rings."""

    local_update = local_update.contiguous()
    ring = mesh.ring_comm()
    side_rows, side_cols = mesh.layout.shape

    row_chunks: list[torch.Tensor | None] = [None for _ in range(side_cols)]
    row_chunks[mesh.coord[1]] = local_update
    ready = local_update
    for step in range(1, side_cols):
        ready = ring.comm_row.exchange(ready.contiguous())
        source_col = (mesh.coord[1] + step) % side_cols
        row_chunks[source_col] = ready
    if any(item is None for item in row_chunks):
        raise RuntimeError("failed to collect single update row chunks.")
    row_block = torch.cat([item for item in row_chunks if item is not None], dim=-2)

    col_blocks: list[torch.Tensor | None] = [None for _ in range(side_rows)]
    col_blocks[mesh.coord[0]] = row_block.contiguous()
    ready = row_block
    for step in range(1, side_rows):
        ready = ring.comm_col.exchange(ready.contiguous())
        source_row = (mesh.coord[0] + step) % side_rows
        col_blocks[source_row] = ready
    if any(item is None for item in col_blocks):
        raise RuntimeError("failed to collect single update column blocks.")
    trimmed_blocks: list[torch.Tensor] = []
    for row_index, item in enumerate(col_blocks):
        if item is None:
            raise RuntimeError("failed to collect single update column block.")
        row_start = row_index * pair_row_tile
        valid_rows = max(0, min(pair_row_tile, n_token - row_start))
        trimmed_blocks.append(item[:valid_rows])
    full_update = torch.cat(trimmed_blocks, dim=-2)
    return full_update[:n_token].contiguous()


def _gather_single_rows_by_col_ring(
    local_rows: torch.Tensor,
    n_token: int,
    mesh: FoldCPProcessMesh,
    row_dim: int,
) -> torch.Tensor:
    """Gather row-sharded single-token tensors in global row order."""

    local_rows = local_rows.contiguous()
    side = mesh.layout.shape[0]
    if side == 1:
        row_dim = row_dim if row_dim >= 0 else local_rows.ndim + row_dim
        return local_rows.narrow(row_dim, 0, n_token).contiguous()

    ring = mesh.ring_comm()
    gathered: list[torch.Tensor | None] = [None for _ in range(side)]
    gathered[mesh.coord[0]] = local_rows

    ready = local_rows
    for step in range(1, side):
        ready = ring.comm_col.exchange(ready.contiguous())
        source_row = (mesh.coord[0] + step) % side
        gathered[source_row] = ready

    if any(item is None for item in gathered):
        raise RuntimeError("failed to collect row-sharded single-token tensors.")
    full_rows = torch.cat([item for item in gathered if item is not None], dim=row_dim)
    row_dim = row_dim if row_dim >= 0 else full_rows.ndim + row_dim
    return full_rows.narrow(row_dim, 0, n_token).contiguous()


def _gather_row_blocks_by_col_ring(
    local_block: torch.Tensor,
    mesh: FoldCPProcessMesh,
    cat_dim: int,
) -> torch.Tensor:
    """Collect column-sharded row blocks in global column order."""

    local_block = local_block.contiguous()
    side = mesh.layout.shape[1]
    if side == 1:
        return local_block

    ring = mesh.ring_comm()
    gathered: list[torch.Tensor | None] = [None for _ in range(side)]
    gathered[mesh.coord[1]] = local_block

    ready = local_block
    for step in range(1, side):
        ready = ring.comm_row.exchange(ready.contiguous())
        source_col = (mesh.coord[1] + step) % side
        gathered[source_col] = ready

    if any(item is None for item in gathered):
        raise RuntimeError("failed to collect AttentionPairBias row blocks.")
    return torch.cat([item for item in gathered if item is not None], dim=cat_dim).contiguous()


def _attention_pair_bias_extra_rows(
    extra_attn_bias_local: torch.Tensor | None,
    mesh: FoldCPProcessMesh,
    local_row_offset: int,
    valid_rows: int,
    n_token: int,
) -> torch.Tensor | None:
    if extra_attn_bias_local is None:
        return None
    if extra_attn_bias_local.ndim < 2:
        raise ValueError("extra attention bias must carry pair row/column dimensions.")
    extra_rows = _ring_gather_by_row(
        extra_attn_bias_local,
        mesh,
        dim=-1,
        length=n_token,
    )
    return extra_rows[..., local_row_offset : local_row_offset + valid_rows, :].contiguous()


def _single_update_rank_range(
    n_token: int,
    mesh: FoldCPProcessMesh,
    pair_row_tile: int,
) -> tuple[int, int, int]:
    tile = (pair_row_tile + mesh.layout.shape[1] - 1) // mesh.layout.shape[1]
    pair_row_start = mesh.coord[0] * pair_row_tile
    pair_row_end = min(pair_row_start + pair_row_tile, n_token)
    start = pair_row_start + mesh.coord[1] * tile
    end = min(start + tile, pair_row_end)
    return start, end, tile


def _attention_pair_bias_row_launch_size(valid_rows: int, original_n: int) -> int:
    if original_n <= 512:
        return original_n
    return min(original_n, max(valid_rows, 112))


def distributed_attention_pair_bias_update(
    attention_pair_bias: torch.nn.Module,
    a: torch.Tensor,
    z_local: torch.Tensor,
    mesh: FoldCPProcessMesh,
    z_spec: FoldCPPairShardSpec | None = None,
    extra_attn_bias_local: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run Pairformer single attention while keeping pair bias as a CP tile."""

    if getattr(attention_pair_bias, "has_s", False):
        raise ValueError("Fold-CP AttentionPairBias currently supports has_s=False.")
    if getattr(attention_pair_bias, "cross_attention_mode", False):
        raise ValueError(
            "Fold-CP AttentionPairBias currently supports self-attention only."
        )
    if a.ndim != 2 or z_local.ndim != 3:
        raise ValueError("Fold-CP AttentionPairBias expects a=[N,C] and z_local=[T,T,C].")

    n_token = a.shape[-2]
    tile = z_local.shape[-3]

    a_norm = attention_pair_bias.layernorm_a(a)
    row_start, valid_row_end, single_tile = _single_update_rank_range(
        n_token,
        mesh,
        tile,
    )
    valid_rows = max(0, valid_row_end - row_start)
    local_update = a.new_zeros((single_tile, a.shape[-1]))
    if valid_rows == 0:
        return _gather_single_update_by_2d_ring(local_update, a.shape[-2], mesh, tile)

    z_row_slab = _ring_gather_by_row(z_local, mesh, dim=-2, length=n_token)
    pair_row_start = mesh.coord[0] * tile
    local_row_offset = row_start - pair_row_start
    row_launch = _attention_pair_bias_row_launch_size(valid_rows, n_token)
    z_source_rows = z_row_slab[local_row_offset : local_row_offset + valid_rows]
    if row_launch != z_source_rows.shape[-3]:
        z_source_launch = z_row_slab.new_zeros(
            (row_launch, z_row_slab.shape[-2], z_row_slab.shape[-1])
        )
        z_source_launch[:valid_rows, :, :] = z_source_rows
    else:
        z_source_launch = z_source_rows
    bias_rows = attention_pair_bias.linear_nobias_z(
        attention_pair_bias.layernorm_z(z_source_launch)
    )
    bias_rows = permute_final_dims(bias_rows, [2, 0, 1]).contiguous()
    bias_rows = bias_rows[:, :valid_rows, :]
    extra_bias = _attention_pair_bias_extra_rows(
        extra_attn_bias_local,
        mesh,
        local_row_offset,
        valid_rows,
        n_token,
    )
    if extra_bias is not None:
        if extra_bias.ndim == 2:
            extra_bias = extra_bias.unsqueeze(0)
        bias_rows = bias_rows + extra_bias.to(dtype=bias_rows.dtype, device=bias_rows.device)

    q, k, v = attention_pair_bias.attention._prep_qkv(
        q_x=a_norm,
        kv_x=a_norm,
        apply_scale=True,
    )
    query_launch = _attention_pair_bias_row_launch_size(valid_rows, n_token)
    if query_launch == n_token:
        query_offset = row_start
    else:
        query_offset = 0
    q_source = q.new_zeros((q.shape[0], query_launch, q.shape[-1]))
    q_source[:, query_offset : query_offset + valid_rows, :] = q[
        :, row_start:valid_row_end, :
    ]
    bias_source = bias_rows.new_zeros((bias_rows.shape[0], query_launch, n_token))
    bias_source[:, query_offset : query_offset + valid_rows, :] = bias_rows
    row_out_source = _single_feature_attention(
        q=q_source.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        attn_bias=bias_source.contiguous(),
        use_efficient_implementation=attention_pair_bias.attention.use_efficient_implementation,
        inplace_safe=False,
    )
    row_out = row_out_source[:, query_offset : query_offset + valid_rows, :]
    source_shape_out = row_out.new_zeros(
        (n_token, row_out.shape[0], row_out.shape[-1])
    )
    source_shape_out[row_start:valid_row_end] = row_out.to(dtype=q.dtype).transpose(-2, -3)
    source_shape_update = attention_pair_bias.attention._wrap_up(
        source_shape_out,
        a_norm,
    )
    local_update[:valid_rows] = source_shape_update[row_start:valid_row_end]
    return _gather_single_update_by_2d_ring(local_update, a.shape[-2], mesh, tile)


def _local_triangle_bias(triangle_attention: torch.nn.Module, x_local: torch.Tensor) -> torch.Tensor:
    """Compute the real TriangleAttention pair bias for this local pair tile."""

    triangle_bias = permute_final_dims(triangle_attention.linear(x_local), (2, 0, 1))
    return triangle_bias.unsqueeze(-4).contiguous()


def _starting_triangle_bias_stack(
    local_triangle_bias: torch.Tensor,
    mesh: FoldCPProcessMesh,
) -> torch.Tensor:
    """Collect query-block bias tiles for this local key block via Fold-CP ring.

    In OpenDDE starting-node triangle attention, the pair bias used for output
    tile `(row_block=r, query_block=c)` comes from `z[query_block=c, key_block]`,
    not from `z[row_block=r, key_block]`. Each mesh column owns one key block, so
    all ranks in a column need the row-indexed bias stack.

    The previous implementation used a column `all_gather`, which became the M5
    2304 failure point under NCCL.  A Fold-CP column ring gathers the same stack
    with point-to-point exchanges and avoids the failing collective.
    """

    local_triangle_bias = local_triangle_bias.contiguous()
    side = mesh.layout.shape[0]
    if side == 1:
        return local_triangle_bias.unsqueeze(0).contiguous()

    ring = mesh.ring_comm()
    gathered: list[torch.Tensor | None] = [None for _ in range(side)]
    gathered[mesh.coord[0]] = local_triangle_bias

    ready = local_triangle_bias
    for step in range(1, side):
        ready = ring.comm_col.exchange(ready.contiguous())
        source_row = (mesh.coord[0] + step) % side
        gathered[source_row] = ready

    if any(item is None for item in gathered):
        raise RuntimeError("failed to collect starting triangle bias stack.")
    return torch.stack([item for item in gathered if item is not None], dim=0).contiguous()


def _starting_triangle_bias_full_key_from_source_slab(
    triangle_attention: torch.nn.Module,
    x_local: torch.Tensor,
    mesh: FoldCPProcessMesh,
    original_n: int,
    query_start: int,
    valid_query: int,
) -> torch.Tensor:
    """Project starting triangle bias from source-layout query-row slabs."""

    side = mesh.layout.shape[0]
    if side == 1:
        x_stack = x_local.unsqueeze(0).contiguous()
    else:
        ring = mesh.ring_comm()
        gathered: list[torch.Tensor | None] = [None for _ in range(side)]
        gathered[mesh.coord[0]] = x_local.contiguous()
        ready = gathered[mesh.coord[0]]
        for step in range(1, side):
            ready = ring.comm_col.exchange(ready.contiguous())
            source_row = (mesh.coord[0] + step) % side
            gathered[source_row] = ready
        if any(item is None for item in gathered):
            raise RuntimeError("failed to collect starting triangle bias source rows.")
        x_stack = torch.stack([item for item in gathered if item is not None], dim=0).contiguous()

    x_source_row_slab = _ring_gather_by_row(
        x_stack,
        mesh,
        dim=-2,
        length=original_n,
    )[mesh.coord[1]]
    exact_source_launch = _triatt_exact_source_launch(original_n)
    query_offset = query_start if x_source_row_slab.shape[-3] == original_n else 0
    source_rows = int(original_n) * int(original_n)
    source_launch_boundary = _triatt_bias_source_launch_boundary(source_rows)
    if x_source_row_slab.shape[-3] != original_n and (
        exact_source_launch or source_launch_boundary
    ):
        x_padded = x_source_row_slab.new_zeros(
            x_source_row_slab.shape[:-3]
            + (original_n, x_source_row_slab.shape[-2], x_source_row_slab.shape[-1])
        )
        x_padded[..., query_start : query_start + valid_query, :, :] = (
            x_source_row_slab[..., :valid_query, :, :]
        )
        x_source_row_slab = x_padded
        query_offset = query_start
    linear_bias = (
        _linear_with_exact_source_launch_shape(
            triangle_attention.linear,
            x_source_row_slab,
            source_rows=source_rows,
        )
        if exact_source_launch or source_launch_boundary
        else triangle_attention.linear(x_source_row_slab)
    )
    triangle_bias = permute_final_dims(
        linear_bias,
        (2, 0, 1),
    )
    return triangle_bias[:, query_offset : query_offset + valid_query, :].contiguous()


def _wrap_up_triangle_attention_output(
    mha: torch.nn.Module,
    out_by_row_head_query: torch.Tensor,
    x_local: torch.Tensor,
    residual_local: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run TriangleAttention MHA wrap-up in row chunks.

    If ``residual_local`` is provided, add each wrapped row chunk directly into
    that tensor. This avoids materializing a second full local ``N^2 x C`` tile
    just to immediately add it back to the Pairformer residual.
    """

    row_chunk_size = _triatt_wrap_row_chunk_size(out_by_row_head_query, x_local)
    if row_chunk_size <= 0 or out_by_row_head_query.shape[0] <= row_chunk_size:
        wrapped = mha._wrap_up(out_by_row_head_query.transpose(-2, -3), x_local)
        if residual_local is not None:
            residual_local += wrapped
            return residual_local.contiguous()
        return wrapped.contiguous()

    wrapped = residual_local if residual_local is not None else x_local.new_empty(x_local.shape)
    for row_start in range(0, out_by_row_head_query.shape[0], row_chunk_size):
        row_end = min(row_start + row_chunk_size, out_by_row_head_query.shape[0])
        row_slice = slice(row_start, row_end)
        o_chunk = out_by_row_head_query[row_slice].transpose(-2, -3)
        x_chunk = x_local[row_slice]
        if mha.linear_g is not None:
            g = mha.sigmoid(mha.linear_g(x_chunk))
            g = g.view(g.shape[:-1] + (mha.no_heads, -1))
            o_chunk = o_chunk * g
        o_chunk = o_chunk.reshape(o_chunk.shape[:-2] + (-1,))
        update_chunk = mha.linear_o(o_chunk)
        if residual_local is not None:
            wrapped[row_slice] += update_chunk
        else:
            wrapped[row_slice] = update_chunk
    return wrapped.contiguous()


def _triatt_qkv_source_rows(original_n: int) -> int | None:
    del original_n
    return None


def _triatt_exact_source_launch(original_n: int) -> bool:
    del original_n
    return False


def _prep_triangle_attention_qkv_chunks(
    mha: torch.nn.Module,
    q_x: torch.Tensor,
    kv_x: torch.Tensor,
    apply_scale: bool = True,
    source_rows: int | None = None,
    exact_source_launch: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project a TriangleAttention row/query chunk without full-tile QKV."""

    if source_rows is None:
        q = mha.linear_q(q_x)
        k = mha.linear_k(kv_x)
        v = mha.linear_v(kv_x)
    elif exact_source_launch:
        q = _linear_with_exact_source_launch_shape(
            mha.linear_q,
            q_x,
            source_rows=source_rows,
        )
        k = _linear_with_exact_source_launch_shape(
            mha.linear_k,
            kv_x,
            source_rows=source_rows,
        )
        v = _linear_with_exact_source_launch_shape(
            mha.linear_v,
            kv_x,
            source_rows=source_rows,
        )
    else:
        q = foldcp_linear_with_source_launch_shape(
            mha.linear_q,
            q_x,
            source_rows=source_rows,
        )
        k = foldcp_linear_with_source_launch_shape(
            mha.linear_k,
            kv_x,
            source_rows=source_rows,
        )
        v = foldcp_linear_with_source_launch_shape(
            mha.linear_v,
            kv_x,
            source_rows=source_rows,
        )

    q = q.view(q.shape[:-1] + (mha.no_heads, -1)).transpose(-2, -3)
    k = k.view(k.shape[:-1] + (mha.no_heads, -1)).transpose(-2, -3)
    v = v.view(v.shape[:-1] + (mha.no_heads, -1)).transpose(-2, -3)

    if apply_scale:
        q = q / (float(mha.c_hidden) ** 0.5)

    return q.contiguous(), k.contiguous(), v.contiguous()


def _prep_triangle_attention_qkv_source_grid_chunks(
    mha: torch.nn.Module,
    x_row_chunk: torch.Tensor,
    *,
    original_n: int,
    row_start: int,
    apply_scale: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid_rows = x_row_chunk.shape[-3]
    q = _linear_pair_tile_with_source_grid_launch(
        mha.linear_q,
        x_row_chunk,
        original_n=original_n,
        row_start=row_start,
        col_start=0,
        valid_rows=valid_rows,
        valid_cols=original_n,
    )
    k = _linear_pair_tile_with_source_grid_launch(
        mha.linear_k,
        x_row_chunk,
        original_n=original_n,
        row_start=row_start,
        col_start=0,
        valid_rows=valid_rows,
        valid_cols=original_n,
    )
    v = _linear_pair_tile_with_source_grid_launch(
        mha.linear_v,
        x_row_chunk,
        original_n=original_n,
        row_start=row_start,
        col_start=0,
        valid_rows=valid_rows,
        valid_cols=original_n,
    )
    q = q.view(q.shape[:-1] + (mha.no_heads, -1)).transpose(-2, -3)
    k = k.view(k.shape[:-1] + (mha.no_heads, -1)).transpose(-2, -3)
    v = v.view(v.shape[:-1] + (mha.no_heads, -1)).transpose(-2, -3)
    if apply_scale:
        q = q / (float(mha.c_hidden) ** 0.5)
    return q.contiguous(), k.contiguous(), v.contiguous()


def _wrap_up_triangle_attention_output_chunk(
    mha: torch.nn.Module,
    out_by_row_head_query: torch.Tensor,
    x_chunk: torch.Tensor,
    *,
    source_grid_launch: bool = False,
    original_n: int | None = None,
    row_start: int = 0,
    col_start: int = 0,
    valid_rows: int | None = None,
    valid_query: int | None = None,
) -> torch.Tensor:
    """Run TriangleAttention MHA wrap-up for one row/query chunk."""

    o_chunk = out_by_row_head_query.transpose(-2, -3)
    if mha.linear_g is not None:
        if source_grid_launch:
            if original_n is None or valid_rows is None or valid_query is None:
                raise ValueError("source-grid TriangleAttention wrap requires shape metadata.")
            g = mha.sigmoid(
                _linear_pair_tile_with_source_grid_launch(
                    mha.linear_g,
                    x_chunk,
                    original_n=original_n,
                    row_start=row_start,
                    col_start=col_start,
                    valid_rows=valid_rows,
                    valid_cols=valid_query,
                )
            )
        else:
            g = mha.sigmoid(mha.linear_g(x_chunk))
        g = g.view(g.shape[:-1] + (mha.no_heads, -1))
        o_chunk = o_chunk * g
    o_chunk = o_chunk.reshape(o_chunk.shape[:-2] + (-1,))
    if source_grid_launch:
        if original_n is None or valid_rows is None or valid_query is None:
            raise ValueError("source-grid TriangleAttention wrap requires shape metadata.")
        return _linear_pair_tile_with_source_grid_launch(
            mha.linear_o,
            o_chunk,
            original_n=original_n,
            row_start=row_start,
            col_start=col_start,
            valid_rows=valid_rows,
            valid_cols=valid_query,
        )
    return mha.linear_o(o_chunk)


def _distributed_triangle_attention_starting_update(
    triangle_attention: torch.nn.Module,
    z_local: torch.Tensor,
    mesh: FoldCPProcessMesh,
    mask_local: torch.Tensor | None = None,
    residual_local: torch.Tensor | None = None,
    z_spec: FoldCPPairShardSpec | None = None,
) -> torch.Tensor:
    if mask_local is None:
        mask = z_local.new_ones(z_local.shape[:-1])
    else:
        mask = mask_local

    if z_spec is None:
        x_local = triangle_attention.layer_norm(z_local)
        original_n = _ring_gather_by_row(x_local, mesh, dim=-2).shape[-2]
        row_start = mesh.coord[0] * x_local.shape[-3]
        col_start = mesh.coord[1] * x_local.shape[-2]
        valid_rows = x_local.shape[-3]
        valid_query = x_local.shape[-2]
    else:
        x_local = _triangle_layer_norm_source_row_slab(
            triangle_attention.layer_norm,
            z_local,
            mesh,
            z_spec,
        )
        original_n = z_spec.original_shape[z_spec.pair_dims[0]]
        row_start, row_end = z_spec.row_range
        col_start, col_end = z_spec.col_range
        valid_rows = max(0, min(row_end, original_n) - row_start)
        valid_query = max(0, min(col_end, original_n) - col_start)

    out_local = residual_local if residual_local is not None else x_local.new_zeros(x_local.shape)
    if valid_rows == 0 or valid_query == 0:
        return out_local.contiguous()

    triangle_bias_full_key = _starting_triangle_bias_full_key_from_source_slab(
        triangle_attention,
        x_local,
        mesh,
        original_n,
        col_start,
        valid_query,
    )

    query_pad = _triatt_query_pad_size(valid_query)
    local_query = slice(col_start, col_start + valid_query)
    triangle_bias = triangle_bias_full_key.new_zeros(
        (triangle_bias_full_key.shape[0], query_pad, original_n)
    )
    triangle_bias[:, :valid_query, :] = triangle_bias_full_key[:, :valid_query, :]
    del triangle_bias_full_key

    x_row_slab = _ring_gather_by_row(x_local, mesh, dim=-2, length=original_n)
    mask_row_slab = _ring_gather_by_row(mask, mesh, dim=-1, length=original_n)

    row_chunk_size = _triatt_attention_row_chunk_size(valid_rows, original_n)
    for valid_row_start in range(0, valid_rows, row_chunk_size):
        valid_row_end = min(valid_row_start + row_chunk_size, valid_rows)
        current_rows = valid_row_end - valid_row_start
        exact_source_launch = _triatt_exact_source_launch(original_n)
        source_grid_qkv = True
        qkv_row_pad = (
            original_n
            if exact_source_launch
            else _triatt_qkv_row_pad_size(current_rows, original_n)
        )
        row_pad = current_rows if original_n > 1024 else qkv_row_pad
        launch_row_start = row_start + valid_row_start if exact_source_launch else 0
        x_row_chunk = x_row_slab[valid_row_start:valid_row_end]
        if source_grid_qkv:
            x_row_source = x_row_chunk
            q_row, k_row, v_row = _prep_triangle_attention_qkv_source_grid_chunks(
                triangle_attention.mha,
                x_row_chunk,
                original_n=original_n,
                row_start=row_start + valid_row_start,
            )
        else:
            if qkv_row_pad != current_rows:
                x_row_source = x_row_slab.new_zeros(
                    (qkv_row_pad, x_row_slab.shape[-2], x_row_slab.shape[-1])
                )
                x_row_source[
                    launch_row_start : launch_row_start + current_rows,
                    :,
                    :,
                ] = x_row_chunk
            else:
                x_row_source = x_row_chunk
            q_row, k_row, v_row = _prep_triangle_attention_qkv_chunks(
                triangle_attention.mha,
                x_row_source,
                x_row_source,
                apply_scale=True,
                source_rows=_triatt_qkv_source_rows(original_n),
                exact_source_launch=exact_source_launch,
            )

        q_chunk = q_row.new_zeros((row_pad, q_row.shape[1], query_pad, q_row.shape[3]))
        k_chunk = k_row.new_zeros((row_pad, k_row.shape[1], original_n, k_row.shape[3]))
        v_chunk = v_row.new_zeros((row_pad, v_row.shape[1], original_n, v_row.shape[3]))
        x_chunk = x_row_slab.new_zeros((row_pad, query_pad, x_row_slab.shape[-1]))
        mask_bias = mask_row_slab.new_zeros((row_pad, 1, 1, original_n))

        q_chunk[:current_rows, :, :valid_query, :] = q_row[
            launch_row_start : launch_row_start + current_rows,
            :,
            local_query,
            :,
        ]
        k_chunk[:current_rows] = k_row[
            launch_row_start : launch_row_start + current_rows
        ]
        v_chunk[:current_rows] = v_row[
            launch_row_start : launch_row_start + current_rows
        ]
        x_chunk[:current_rows, :valid_query, :] = x_row_source[
            launch_row_start : launch_row_start + current_rows,
            local_query,
            :,
        ]
        mask_bias[:current_rows] = (
            triangle_attention.inf
            * (mask_row_slab[valid_row_start:valid_row_end] - 1)
        )[:, None, None, :]

        out_chunk = _attention(
            q_chunk.contiguous(),
            k_chunk.contiguous(),
            v_chunk.contiguous(),
            [
                mask_bias.contiguous(),
                triangle_bias.unsqueeze(0).contiguous(),
            ],
        )
        update = _wrap_up_triangle_attention_output_chunk(
            triangle_attention.mha,
            out_chunk.to(dtype=x_local.dtype),
            x_chunk.contiguous(),
            source_grid_launch=exact_source_launch or source_grid_qkv,
            original_n=original_n,
            row_start=row_start + valid_row_start,
            col_start=col_start,
            valid_rows=current_rows,
            valid_query=valid_query,
        )[:current_rows, :valid_query]
        if residual_local is not None:
            out_local[valid_row_start:valid_row_end, :valid_query] += update
        else:
            out_local[valid_row_start:valid_row_end, :valid_query] = update
    return out_local.contiguous()


def distributed_triangle_attention_update(
    triangle_attention: torch.nn.Module,
    z_local: torch.Tensor,
    mesh: FoldCPProcessMesh,
    mask_local: torch.Tensor | None = None,
    residual_local: torch.Tensor | None = None,
    z_spec: FoldCPPairShardSpec | None = None,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Run a real TriangleAttention module on a Fold-CP local pair tile.

    The implementation matches the real torch TriangleAttention path: layernorm,
    Q/K/V projections, triangle bias, gate, and output projection come from the
    OpenDDE module. Fold-CP only replaces the full `[..., H, Q, K]` softmax with
    row-ring K/V/bias rotation plus online softmax accumulation.
    """

    if getattr(triangle_attention, "starting", True):
        return _distributed_triangle_attention_starting_update(
            triangle_attention,
            z_local,
            mesh,
            mask_local,
            residual_local=residual_local,
            z_spec=z_spec,
        )

    ring = mesh.ring_comm()
    z_t_local = ring.comm_2d_trans.exchange(z_local.transpose(-2, -3).contiguous())
    mask_t_local = (
        None
        if mask_local is None
        else ring.comm_2d_trans.exchange(mask_local.transpose(-1, -2).contiguous())
    )
    out_t_local = _distributed_triangle_attention_starting_update(
        triangle_attention,
        z_t_local,
        mesh,
        mask_t_local,
        z_spec=z_spec,
    )
    out_local = ring.comm_2d_trans.exchange(out_t_local.transpose(-2, -3).contiguous())
    if residual_local is not None:
        residual_local += out_local
        return residual_local.contiguous()
    return out_local


def _distributed_pairformer_block_pair_ops(
    block: torch.nn.Module,
    z_local: torch.Tensor,
    mesh: FoldCPProcessMesh,
    pair_mask_local: torch.Tensor | None = None,
    z_spec: FoldCPPairShardSpec | None = None,
    chunk_size: int | None = None,
) -> torch.Tensor:
    z_local = distributed_triangle_multiplication_update(
        block.tri_mul_out,
        z_local,
        mesh,
        pair_mask_local,
        residual_local=z_local,
        z_spec=z_spec,
    )
    z_local = distributed_triangle_multiplication_update(
        block.tri_mul_in,
        z_local,
        mesh,
        pair_mask_local,
        residual_local=z_local,
        z_spec=z_spec,
    )
    z_local = distributed_triangle_attention_update(
        block.tri_att_start,
        z_local,
        mesh,
        pair_mask_local,
        residual_local=z_local,
        z_spec=z_spec,
        chunk_size=chunk_size,
    )

    ring = mesh.ring_comm()
    z_t_local = ring.comm_2d_trans.exchange(z_local.transpose(-2, -3).contiguous())
    mask_t_local = (
        None
        if pair_mask_local is None
        else ring.comm_2d_trans.exchange(pair_mask_local.transpose(-1, -2).contiguous())
    )
    del z_local
    if torch.is_grad_enabled():
        z_t_local = z_t_local + distributed_triangle_attention_update(
            block.tri_att_end,
            z_t_local,
            mesh,
            mask_t_local,
            z_spec=z_spec,
        )
    else:
        z_t_local = distributed_triangle_attention_update(
            block.tri_att_end,
            z_t_local,
            mesh,
            mask_t_local,
            residual_local=z_t_local,
            z_spec=z_spec,
        )
    z_local = ring.comm_2d_trans.exchange(z_t_local.transpose(-2, -3).contiguous())

    z_local = distributed_pair_transition_update(
        block.pair_transition,
        z_local,
        mesh,
        residual_local=z_local,
        z_spec=z_spec,
    )
    return z_local.contiguous()


def distributed_pairformer_block_pair_update(
    block: torch.nn.Module,
    z_local: torch.Tensor,
    mesh: FoldCPProcessMesh,
    pair_mask_local: torch.Tensor | None = None,
    z_spec: FoldCPPairShardSpec | None = None,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Run the pair-only part of a real PairformerBlock on a Fold-CP pair tile."""

    if getattr(block, "c_s", 0) != 0:
        raise ValueError("distributed_pairformer_block_pair_update currently requires c_s=0.")

    return _distributed_pairformer_block_pair_ops(
        block,
        z_local,
        mesh,
        pair_mask_local,
        z_spec,
        chunk_size,
    )


def distributed_pairformer_stack_pair_update(
    stack: torch.nn.Module,
    z_local: torch.Tensor,
    mesh: FoldCPProcessMesh,
    pair_mask_local: torch.Tensor | None = None,
    z_spec: FoldCPPairShardSpec | None = None,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Run a real c_s=0 PairformerStack while keeping pair activations sharded."""

    for block in stack.blocks:
        z_local = distributed_pairformer_block_pair_update(
            block,
            z_local,
            mesh,
            pair_mask_local,
            z_spec,
            chunk_size,
        )
    return z_local.contiguous()


def distributed_pairformer_stack_single_bridge_update(
    stack: torch.nn.Module,
    s: torch.Tensor,
    z: torch.Tensor,
    mesh: FoldCPProcessMesh,
    pair_mask: torch.Tensor | None = None,
    extra_attn_bias: torch.Tensor | None = None,
    extra_attn_bias_is_local: bool = False,
    return_local_pair: bool = False,
    z_spec: FoldCPPairShardSpec | None = None,
    chunk_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, FoldCPPairShardSpec]:
    """Run a real c_s>0 PairformerStack with CP pair ops and local pair bias.

    Pair operations stay on Fold-CP local tiles. The single update consumes the
    local pair-bias tile via row-ring attention and gathers only the N-sized
    single update, not the full pair tensor, at each block.
    """

    if not stack.blocks:
        return s, z
    if getattr(stack.blocks[0], "c_s", 0) <= 0:
        raise ValueError("single bridge requires a PairformerStack with c_s > 0.")

    if z_spec is None:
        z_local, z_spec = shard_pair_tensor(z, mesh, pair_dims=(-3, -2))
        del z
        torch.cuda.empty_cache()
    else:
        z_local = z.contiguous()
    if pair_mask is None:
        mask_local = None
    else:
        mask_local, _ = shard_pair_tensor(pair_mask, mesh, pair_dims=(-2, -1))
    if extra_attn_bias is None:
        extra_attn_bias_local = None
    elif extra_attn_bias_is_local:
        extra_attn_bias_local = extra_attn_bias.contiguous()
    else:
        extra_attn_bias_local, _ = shard_pair_tensor(
            extra_attn_bias,
            mesh,
            pair_dims=(-2, -1),
        )

    for block_index, block in enumerate(stack.blocks):
        z_local = _distributed_pairformer_block_pair_ops(
            block,
            z_local,
            mesh,
            mask_local,
            z_spec,
            chunk_size,
        )
        s = s + distributed_attention_pair_bias_update(
            block.attention_pair_bias,
            s,
            z_local,
            mesh,
            z_spec=z_spec,
            extra_attn_bias_local=extra_attn_bias_local,
        )
        s = s + block.single_transition(s)

    if return_local_pair:
        return s, z_local.contiguous(), z_spec

    z = gather_pair_tensor(z_local, z_spec, mesh.group_2d)
    return s, z.contiguous()
