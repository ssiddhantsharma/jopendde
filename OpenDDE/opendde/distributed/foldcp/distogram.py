# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Fold-CP helpers for distogram/contact probabilities."""

from __future__ import annotations

import torch
import torch.distributed as dist

from opendde.distributed.foldcp.launch import (
    foldcp_pair_row_slab_linear_with_source_launch_policy,
)
from opendde.distributed.foldcp.mesh import FoldCPProcessMesh
from opendde.distributed.foldcp.pair_sharding import (
    FoldCPPairShardSpec,
    _copy_pair_shard_into_output,
    gather_pair_tensor_like,
    shard_pair_tensor,
)


def _transpose_pair_tile_collective(
    z_pair_local: torch.Tensor,
    mesh: FoldCPProcessMesh,
) -> torch.Tensor:
    """Exchange the reciprocal pair tile without a full all-gather buffer."""

    z_pair_t_send = z_pair_local.transpose(-2, -3).contiguous()
    transposed_rank = mesh.layout.transpose_rank(mesh.coord)
    z_pair_t_recv = torch.empty_like(z_pair_t_send)

    # Broadcast one source tile at a time. Every rank participates in the same
    # collective order, but each rank only keeps the reciprocal tile it needs.
    for source_rank in range(mesh.layout.numel):
        buffer = (
            z_pair_t_send
            if mesh.cp_rank == source_rank
            else torch.empty_like(z_pair_t_send)
        )
        dist.broadcast(
            buffer,
            src=mesh.cp_global_ranks[source_rank],
            group=mesh.group_2d,
        )
        if source_rank == transposed_rank:
            z_pair_t_recv.copy_(buffer)
        if mesh.cp_rank != source_rank:
            del buffer
    return z_pair_t_recv.contiguous()


def _project_pair_row_slab_local(
    z_pair_local: torch.Tensor,
    z_pair_spec: FoldCPPairShardSpec,
    mesh: FoldCPProcessMesh,
    linear: torch.nn.Module,
) -> torch.Tensor:
    """Project this rank's pair tile using the source row-slab layout.

    The serial distogram head applies ``linear`` to the contiguous
    ``[N, N, C]`` pair tensor. Calling the same module on a local
    ``[tile, tile, C]`` slice can select a different CUDA GEMM shape and drift
    by a few ulps. Fold-CP keeps pair ownership local, but reconstructs only the
    current row block across all column tiles before projection, then slices the
    current column tile back out. This preserves the source row-major projection
    layout without materializing full pair logits on any rank.
    """

    if z_pair_local.ndim != 3:
        raise ValueError(
            "Fold-CP distogram row-slab projection expects [tile, tile, c_z]."
        )

    side = mesh.layout.shape[1]
    if side == 1:
        z_row_slab = z_pair_local.contiguous()
    else:
        ring = mesh.ring_comm()
        row_tiles: list[torch.Tensor | None] = [None for _ in range(side)]
        row_tiles[mesh.coord[1]] = z_pair_local.contiguous()
        ready = row_tiles[mesh.coord[1]]
        for step in range(1, side):
            ready = ring.comm_row.exchange(ready.contiguous())
            source_col = (mesh.coord[1] + step) % side
            row_tiles[source_col] = ready
        if any(item is None for item in row_tiles):
            raise RuntimeError("failed to collect distogram row slab.")
        z_row_slab = torch.cat([item for item in row_tiles if item is not None], dim=-2)
        z_row_slab = z_row_slab.contiguous()
        del row_tiles

    row_start, row_end = z_pair_spec.row_range
    n_token = z_pair_spec.original_shape[z_pair_spec.pair_dims[0]]
    valid_rows = max(0, min(row_end, n_token) - row_start)
    logits_row_slab = foldcp_pair_row_slab_linear_with_source_launch_policy(
        linear,
        z_row_slab,
        original_n=n_token,
        row_start=row_start,
        col_start=0,
        valid_rows=valid_rows,
        valid_cols=n_token,
    )
    del z_row_slab

    tile_col = z_pair_local.shape[-2]
    col_start = z_pair_spec.col_range[0]
    col_end = col_start + tile_col
    logits_local = logits_row_slab[..., col_start:col_end, :].contiguous()
    del logits_row_slab
    return logits_local


def _distogram_bin_tops(
    *,
    min_bin: float,
    max_bin: float,
    no_bins: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    boundaries = torch.linspace(
        min_bin,
        max_bin,
        no_bins - 1,
        device=device,
        dtype=dtype,
    )
    return torch.cat([boundaries, boundaries.new_tensor([1e8])], dim=0)


def distogram_contact_probs_local(
    *,
    z_pair_local: torch.Tensor,
    z_pair_spec: FoldCPPairShardSpec,
    mesh: FoldCPProcessMesh,
    linear: torch.nn.Module,
    min_bin: float,
    max_bin: float,
    no_bins: int,
    thres: float = 8.0,
) -> torch.Tensor:
    """Compute contact probabilities for one CP local pair tile.

    The serial DistogramHead computes ``linear(z_ij) + linear(z_ji)``. The
    reciprocal ``z_ji`` tile is obtained via the Fold-CP 2D transpose exchange,
    so no rank needs to materialize full ``[N, N, bins]`` logits.
    """

    logits_direct_local = _project_pair_row_slab_local(
        z_pair_local,
        z_pair_spec,
        mesh,
        linear,
    )
    logits_t_local = _transpose_pair_tile_collective(logits_direct_local, mesh)
    logits_local = logits_direct_local + logits_t_local
    del logits_direct_local, logits_t_local
    probs_local = torch.nn.functional.softmax(logits_local, dim=-1)
    bin_tops = _distogram_bin_tops(
        min_bin=min_bin,
        max_bin=max_bin,
        no_bins=no_bins,
        device=logits_local.device,
        dtype=logits_local.dtype,
    )
    contact_local = probs_local[..., bin_tops <= thres].sum(dim=-1)
    del logits_local, probs_local
    return contact_local.contiguous()


def distributed_distogram_contact_probs(
    *,
    z_pair_local: torch.Tensor,
    z_pair_spec: FoldCPPairShardSpec,
    mesh: FoldCPProcessMesh,
    linear: torch.nn.Module,
    min_bin: float,
    max_bin: float,
    no_bins: int,
    thres: float = 8.0,
    gather_to_rank0_only: bool = False,
) -> torch.Tensor | None:
    """Compute contact probabilities from CP-sharded pair activations."""

    contact_local = distogram_contact_probs_local(
        z_pair_local=z_pair_local,
        z_pair_spec=z_pair_spec,
        mesh=mesh,
        linear=linear,
        min_bin=min_bin,
        max_bin=max_bin,
        no_bins=no_bins,
        thres=thres,
    )
    contact_local_with_channel = contact_local.unsqueeze(dim=-1)
    if gather_to_rank0_only:
        contact = _gather_pair_like_collective_to_rank0(
            contact_local_with_channel,
            z_pair_spec,
            mesh,
        )
    else:
        contact = gather_pair_tensor_like(
            contact_local_with_channel,
            z_pair_spec,
            mesh.group_2d,
        )
    if contact is None:
        return None
    return contact.squeeze(dim=-1).contiguous()


def _gather_pair_like_collective_to_rank0(
    local_tensor: torch.Tensor,
    spec: FoldCPPairShardSpec,
    mesh: FoldCPProcessMesh,
) -> torch.Tensor | None:
    row_dim, col_dim = spec.pair_dims
    group_rank = torch.distributed.get_rank(mesh.group_2d)
    gathered = (
        [torch.empty_like(local_tensor) for _ in range(mesh.layout.numel)]
        if group_rank == 0
        else None
    )
    torch.distributed.gather(
        local_tensor.contiguous(),
        gather_list=gathered,
        dst=mesh.cp_global_ranks[0],
        group=mesh.group_2d,
    )
    if group_rank != 0:
        return None
    if gathered is None:
        raise ValueError("gathered shards must be available on the destination rank.")

    output_shape = list(local_tensor.shape)
    output_shape[row_dim] = spec.original_shape[row_dim]
    output_shape[col_dim] = spec.original_shape[col_dim]
    full = local_tensor.new_empty(tuple(output_shape))
    tile_row = local_tensor.shape[row_dim]
    tile_col = local_tensor.shape[col_dim]
    for cp_rank, shard in enumerate(gathered):
        row, col = mesh.layout.to_coord(cp_rank)
        row_range = (row * tile_row, (row + 1) * tile_row)
        col_range = (col * tile_col, (col + 1) * tile_col)
        _copy_pair_shard_into_output(
            full,
            shard,
            spec.pair_dims,
            row_range,
            col_range,
        )
    return full


def _reciprocal_tile_from_full_pair(
    z_pair: torch.Tensor,
    spec: FoldCPPairShardSpec,
    reference: torch.Tensor,
) -> torch.Tensor:
    row_start, row_end = spec.row_range
    col_start, col_end = spec.col_range
    n_token = spec.original_shape[spec.pair_dims[0]]
    valid_row_end = min(row_end, n_token)
    valid_col_end = min(col_end, n_token)
    valid_rows = max(0, valid_row_end - row_start)
    valid_cols = max(0, valid_col_end - col_start)
    reciprocal = reference.new_zeros(reference.shape)
    if valid_rows == 0 or valid_cols == 0:
        return reciprocal
    reciprocal_valid = z_pair[
        ...,
        col_start:valid_col_end,
        row_start:valid_row_end,
        :,
    ].transpose(-2, -3)
    reciprocal[..., :valid_rows, :valid_cols, :] = reciprocal_valid
    return reciprocal.contiguous()


def distributed_distogram_contact_probs_from_full_pair(
    *,
    z_pair: torch.Tensor,
    mesh: FoldCPProcessMesh,
    linear: torch.nn.Module,
    min_bin: float,
    max_bin: float,
    no_bins: int,
    thres: float = 8.0,
    gather_to_rank0_only: bool = False,
) -> torch.Tensor | None:
    """Compute contact probabilities from full pair input without full logits.

    This is the NCCL-safe main-path bridge while earlier stages still expose a
    full ``pair_z``. It slices only this rank's pair tile and reciprocal tile,
    computes local distogram logits/contact, and gathers contact probabilities.
    """

    z_pair_local, z_pair_spec = shard_pair_tensor(z_pair, mesh, pair_dims=(-3, -2))
    z_pair_t_local = _reciprocal_tile_from_full_pair(
        z_pair=z_pair,
        spec=z_pair_spec,
        reference=z_pair_local,
    )
    logits_local = linear(z_pair_local) + linear(z_pair_t_local)
    del z_pair_t_local
    probs_local = torch.nn.functional.softmax(logits_local, dim=-1)
    bin_tops = _distogram_bin_tops(
        min_bin=min_bin,
        max_bin=max_bin,
        no_bins=no_bins,
        device=logits_local.device,
        dtype=logits_local.dtype,
    )
    contact_local = probs_local[..., bin_tops <= thres].sum(dim=-1).contiguous()
    del logits_local, probs_local
    contact_local_with_channel = contact_local.unsqueeze(dim=-1)
    if gather_to_rank0_only:
        contact = _gather_pair_like_collective_to_rank0(
            contact_local_with_channel,
            z_pair_spec,
            mesh,
        )
    else:
        contact = gather_pair_tensor_like(
            contact_local_with_channel,
            z_pair_spec,
            mesh.group_2d,
        )
    if contact is None:
        return None
    return contact.squeeze(dim=-1).contiguous()
