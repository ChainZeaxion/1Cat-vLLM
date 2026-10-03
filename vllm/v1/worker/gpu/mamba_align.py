# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MRV2 GPU state migration for hybrid Mamba ``align`` prefix caching.

This is the narrow Model Runner V2 portion of upstream vLLM PR #42406.  It is
kept separate from :mod:`vllm.v1.worker.mamba_utils` because that module also
owns this tree's V1 DDTree state-selection compatibility path.
"""

import os

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.mamba_utils import MambaSpecDecodeGPUContext

# Temporal states are much larger than conv states. Splitting their byte range
# keeps block-boundary copies from serializing on one CTA at small batch sizes.
_TEMPORAL_TILES = 16

# SX_OPT_MAMBA_ALIGN_COPY_GRID (design_5 [cpu-2], default "1"; "0" = old grid).
# The pre-copy/post-process kernels run on the main stream every step, outside
# CUDA graphs. The old grid is (num_reqs, num_states, 16): ~24 x 73 x 16 = 28K
# CTAs that almost all early-exit, since a copy is needed only when a request
# crosses a state block. The grouped grid is (num_reqs, min(num_states, 8), 16):
# each CTA does the per-request early-exit loads once and then loops over the
# states of its group. Copies are the same bytes with the same single writer
# per destination (conv memmove still owned by tile 0 of the state's group), so
# the resulting state bytes are bitwise identical.
# Scope: SM70 devices only (the grid was sized and benchmarked for V100); every
# other GPU keeps the old grid. This module has no model config, so on SM70 it
# applies to every MRV2 align-mode run (outputs are byte-identical either way).
_STATE_GROUPS = 8
# Bytes per load/store step of the grouped copy loop. One grouped CTA copies
# ~num_states / 8 states serially, and the loop is latency bound, so a 4 KiB
# step keeps 4x more bytes in flight than the old grid's 1 KiB step and cuts
# the serial steps of a block-crossing row 4x. Copies stay byte-for-byte
# identical; the conv memmove stays safe for any step size because each step
# is fully loaded before the barrier and later steps only read higher source
# addresses than earlier steps wrote.
_GROUPED_COPY_BLOCK_SIZE = 4096
_USE_GROUPED_COPY_GRID = os.environ.get("SX_OPT_MAMBA_ALIGN_COPY_GRID", "1") != "0"
_GROUPED_COPY_GRID_DEVICE_OK: bool | None = None


def _grouped_copy_grid_device_ok() -> bool:
    """Resolved once, at the first launch (the worker's device is set by then)."""
    global _GROUPED_COPY_GRID_DEVICE_OK
    if _GROUPED_COPY_GRID_DEVICE_OK is None:
        try:
            from vllm.platforms import current_platform

            _GROUPED_COPY_GRID_DEVICE_OK = bool(
                current_platform.is_device_capability(70)
            )
        except Exception:
            _GROUPED_COPY_GRID_DEVICE_OK = False
    return _GROUPED_COPY_GRID_DEVICE_OK


@triton.jit
def _copy_mamba_state_block(
    state_idx,
    block_table_row,
    src_col,
    dst_col,
    token_bias,
    block_table_ptrs_ptr,
    block_table_stride_req,
    state_base_addrs_ptr,
    state_block_strides_ptr,
    state_elem_sizes_ptr,
    state_inner_sizes_ptr,
    state_conv_widths_ptr,
    state_group_indices_ptr,
    tile_idx,
    COPY_BLOCK_SIZE: tl.constexpr,
    TEMPORAL_TILES: tl.constexpr,
):
    """Copy one SD-layout conv or temporal state using V1 align semantics."""
    state_base_addr = tl.load(state_base_addrs_ptr + state_idx)
    state_block_stride = tl.load(state_block_strides_ptr + state_idx)
    state_elem_size = tl.load(state_elem_sizes_ptr + state_idx).to(tl.int64)
    state_inner_size = tl.load(state_inner_sizes_ptr + state_idx).to(tl.int64)
    conv_width = tl.load(state_conv_widths_ptr + state_idx)

    group_idx = tl.load(state_group_indices_ptr + state_idx).to(tl.int64)
    group_base_addr = tl.load(block_table_ptrs_ptr + group_idx)
    block_table = group_base_addr.to(tl.pointer_type(tl.int32))
    block_table += block_table_row * block_table_stride_req

    dst_block_id = tl.load(block_table + dst_col).to(tl.int64)
    dst_addr = state_base_addr + dst_block_id * state_block_stride
    is_conv_state = conv_width > 0

    if is_conv_state:
        # Conv states are small and may be shifted left within the same block.
        # One CTA owns the copy so a block-local barrier can make each chunk
        # memmove-safe before lower addresses overwrite overlapping input.
        if tile_idx > 0:
            return
        src_block_id = tl.load(block_table + src_col).to(tl.int64)
        token_bytes = state_inner_size * state_elem_size
        src_addr = (
            state_base_addr
            + src_block_id * state_block_stride
            + token_bias.to(tl.int64) * token_bytes
        )
        copy_size = (conv_width - token_bias).to(tl.int64) * token_bytes
        tile_start = copy_size * 0
        tile_end = copy_size
        is_left_overlap = (dst_addr < src_addr) & (dst_addr + copy_size > src_addr)
    else:
        # Temporal state chooses the accepted speculative column.
        src_block_id = tl.load(block_table + src_col + token_bias).to(tl.int64)
        src_addr = state_base_addr + src_block_id * state_block_stride
        copy_size = state_inner_size * state_elem_size
        bytes_per_tile = (copy_size + TEMPORAL_TILES - 1) // TEMPORAL_TILES
        tile_start = tile_idx.to(tl.int64) * bytes_per_tile
        tile_end = tl.minimum(tile_start + bytes_per_tile, copy_size)
        is_left_overlap = False

    offsets = tl.arange(0, COPY_BLOCK_SIZE)
    for chunk_start in range(tile_start, tile_end, COPY_BLOCK_SIZE):
        byte_offsets = chunk_start + offsets
        mask = byte_offsets < tile_end
        src = (src_addr + byte_offsets).to(tl.pointer_type(tl.uint8))
        dst = (dst_addr + byte_offsets).to(tl.pointer_type(tl.uint8))
        data = tl.load(src, mask=mask)
        if is_left_overlap:
            tl.debug_barrier()
        tl.store(dst, data, mask=mask)


@triton.jit
def preprocess_mamba_align_fused_kernel(
    idx_mapping_ptr,
    state_idx_ptr,
    num_computed_tokens_ptr,
    query_start_loc_ptr,
    num_accepted_tokens_ptr,
    src_col_ptr,
    token_bias_ptr,
    num_reqs,
    BLOCK_SIZE: tl.constexpr,
    MAMBA_BLOCK_SIZE: tl.constexpr,
):
    """Advance running state columns and emit any required pre-copy inputs."""
    rows = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = rows < num_reqs
    req_indices = tl.load(idx_mapping_ptr + rows, mask=mask, other=0)

    state_idx = tl.load(state_idx_ptr + req_indices, mask=mask, other=-1)
    num_computed = tl.load(num_computed_tokens_ptr + req_indices, mask=mask, other=0)
    # A new request is seeded here, after the resolved MambaSpec is available.
    # cache_config.block_size can be smaller than the actual Mamba block when
    # an FP16 DFlash cache shares the hybrid page pool, so add_request cannot
    # derive the state column from the global cache block size.
    seeded_state_idx = (num_computed + MAMBA_BLOCK_SIZE - 1) // MAMBA_BLOCK_SIZE - 1
    state_idx = tl.where(state_idx == -2, seeded_state_idx, state_idx)
    num_accepted = tl.load(num_accepted_tokens_ptr + req_indices, mask=mask, other=1)
    token_bias = tl.maximum(num_accepted - 1, 0)
    tl.store(src_col_ptr + req_indices, state_idx, mask=mask)
    tl.store(token_bias_ptr + req_indices, token_bias, mask=mask)

    query_start = tl.load(query_start_loc_ptr + rows, mask=mask, other=0)
    query_end = tl.load(query_start_loc_ptr + rows + 1, mask=mask, other=0)
    computed_after = num_computed + query_end - query_start
    new_state_idx = (computed_after + MAMBA_BLOCK_SIZE - 1) // MAMBA_BLOCK_SIZE - 1
    tl.store(state_idx_ptr + req_indices, new_state_idx, mask=mask)
    crossed_boundary = (state_idx >= 0) & (state_idx != new_state_idx)
    # Ordinary one-token decode reads the canonical recurrent state. After a
    # speculative step, materialize the accepted slot there even if the request
    # stays in the same Mamba block; the non-spec kernels cannot select it.
    single_token_tail = (query_end - query_start == 1) & (token_bias > 0)
    tl.store(
        num_accepted_tokens_ptr + req_indices,
        1,
        mask=mask & (crossed_boundary | single_token_tail),
    )


@triton.jit
def _precopy_mamba_align_kernel(
    state_idx_ptr,
    src_col_ptr,
    token_bias_ptr,
    block_table_ptrs_ptr,
    block_table_stride_req,
    state_base_addrs_ptr,
    state_block_strides_ptr,
    state_elem_sizes_ptr,
    state_inner_sizes_ptr,
    state_conv_widths_ptr,
    state_group_indices_ptr,
    idx_mapping_ptr,
    query_start_loc_ptr,
    num_reqs,
    COPY_BLOCK_SIZE: tl.constexpr,
    TEMPORAL_TILES: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    state_idx = tl.program_id(1)
    tile_idx = tl.program_id(2)
    if batch_idx >= num_reqs:
        return
    req_idx = tl.load(idx_mapping_ptr + batch_idx)
    if req_idx < 0:
        return

    src_col = tl.load(src_col_ptr + req_idx)
    dst_col = tl.load(state_idx_ptr + req_idx)
    token_bias = tl.load(token_bias_ptr + req_idx)
    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    single_token_tail = query_end - query_start == 1 and token_bias > 0
    if src_col < 0 or (src_col == dst_col and not single_token_tail):
        return
    _copy_mamba_state_block(
        state_idx,
        batch_idx,
        src_col,
        dst_col,
        token_bias,
        block_table_ptrs_ptr,
        block_table_stride_req,
        state_base_addrs_ptr,
        state_block_strides_ptr,
        state_elem_sizes_ptr,
        state_inner_sizes_ptr,
        state_conv_widths_ptr,
        state_group_indices_ptr,
        tile_idx,
        COPY_BLOCK_SIZE,
        TEMPORAL_TILES,
    )


@triton.jit
def _postprocess_mamba_align_kernel(
    num_accepted_snapshot_ptr,
    num_accepted_out_ptr,
    state_idx_ptr,
    new_num_computed_tokens_ptr,
    block_table_ptrs_ptr,
    block_table_stride_req,
    state_base_addrs_ptr,
    state_block_strides_ptr,
    state_elem_sizes_ptr,
    state_inner_sizes_ptr,
    state_conv_widths_ptr,
    state_group_indices_ptr,
    idx_mapping_ptr,
    num_reqs,
    MAMBA_BLOCK_SIZE: tl.constexpr,
    COPY_BLOCK_SIZE: tl.constexpr,
    TEMPORAL_TILES: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    state_meta_idx = tl.program_id(1)
    tile_idx = tl.program_id(2)
    if batch_idx >= num_reqs:
        return
    req_idx = tl.load(idx_mapping_ptr + batch_idx)
    if req_idx < 0:
        return

    num_accepted = tl.load(num_accepted_snapshot_ptr + req_idx)
    src_col = tl.load(state_idx_ptr + req_idx)
    new_num_computed = tl.load(new_num_computed_tokens_ptr + req_idx)
    num_tokens_running_state = new_num_computed - num_accepted + 1
    aligned_new_computed = (new_num_computed // MAMBA_BLOCK_SIZE) * MAMBA_BLOCK_SIZE
    if aligned_new_computed < num_tokens_running_state:
        return

    token_bias = aligned_new_computed - num_tokens_running_state
    dst_col = aligned_new_computed // MAMBA_BLOCK_SIZE - 1
    if src_col == dst_col and state_meta_idx == 0 and tile_idx == 0:
        tl.store(num_accepted_out_ptr + req_idx, 1)
    if src_col == dst_col and token_bias == 0:
        return

    _copy_mamba_state_block(
        state_meta_idx,
        batch_idx,
        src_col,
        dst_col,
        token_bias,
        block_table_ptrs_ptr,
        block_table_stride_req,
        state_base_addrs_ptr,
        state_block_strides_ptr,
        state_elem_sizes_ptr,
        state_inner_sizes_ptr,
        state_conv_widths_ptr,
        state_group_indices_ptr,
        tile_idx,
        COPY_BLOCK_SIZE,
        TEMPORAL_TILES,
    )


@triton.jit
def _copy_mamba_state_block_grouped(
    state_idx,
    block_table_row,
    src_col,
    dst_col,
    token_bias,
    block_table_ptrs_ptr,
    block_table_stride_req,
    state_base_addrs_ptr,
    state_block_strides_ptr,
    state_elem_sizes_ptr,
    state_inner_sizes_ptr,
    state_conv_widths_ptr,
    state_group_indices_ptr,
    tile_idx,
    COPY_BLOCK_SIZE: tl.constexpr,
    TEMPORAL_TILES: tl.constexpr,
):
    """``_copy_mamba_state_block`` without an early return.

    It is called in a loop over several states, so the conv-state rule "only
    tile 0 copies" becomes an empty byte range for the other tiles instead of
    a return. Addresses, byte ranges and the memmove barrier are the same as
    in ``_copy_mamba_state_block``; only the step size (COPY_BLOCK_SIZE) of
    the grouped launch differs, which does not change the copied bytes.
    """
    state_base_addr = tl.load(state_base_addrs_ptr + state_idx)
    state_block_stride = tl.load(state_block_strides_ptr + state_idx)
    state_elem_size = tl.load(state_elem_sizes_ptr + state_idx).to(tl.int64)
    state_inner_size = tl.load(state_inner_sizes_ptr + state_idx).to(tl.int64)
    conv_width = tl.load(state_conv_widths_ptr + state_idx)

    group_idx = tl.load(state_group_indices_ptr + state_idx).to(tl.int64)
    group_base_addr = tl.load(block_table_ptrs_ptr + group_idx)
    block_table = group_base_addr.to(tl.pointer_type(tl.int32))
    block_table += block_table_row * block_table_stride_req

    dst_block_id = tl.load(block_table + dst_col).to(tl.int64)
    dst_addr = state_base_addr + dst_block_id * state_block_stride
    is_conv_state = conv_width > 0

    if is_conv_state:
        # One CTA (tile 0) owns a conv-state copy so the block-local barrier
        # keeps the left-shifting memmove safe; other tiles copy nothing.
        src_block_id = tl.load(block_table + src_col).to(tl.int64)
        token_bytes = state_inner_size * state_elem_size
        src_addr = (
            state_base_addr
            + src_block_id * state_block_stride
            + token_bias.to(tl.int64) * token_bytes
        )
        copy_size = (conv_width - token_bias).to(tl.int64) * token_bytes
        tile_start = copy_size * 0
        tile_end = tl.where(tile_idx == 0, copy_size, tile_start)
        is_left_overlap = (dst_addr < src_addr) & (dst_addr + copy_size > src_addr)
    else:
        # Temporal state chooses the accepted speculative column.
        src_block_id = tl.load(block_table + src_col + token_bias).to(tl.int64)
        src_addr = state_base_addr + src_block_id * state_block_stride
        copy_size = state_inner_size * state_elem_size
        bytes_per_tile = (copy_size + TEMPORAL_TILES - 1) // TEMPORAL_TILES
        tile_start = tile_idx.to(tl.int64) * bytes_per_tile
        tile_end = tl.minimum(tile_start + bytes_per_tile, copy_size)
        is_left_overlap = False

    offsets = tl.arange(0, COPY_BLOCK_SIZE)
    for chunk_start in range(tile_start, tile_end, COPY_BLOCK_SIZE):
        byte_offsets = chunk_start + offsets
        mask = byte_offsets < tile_end
        src = (src_addr + byte_offsets).to(tl.pointer_type(tl.uint8))
        dst = (dst_addr + byte_offsets).to(tl.pointer_type(tl.uint8))
        data = tl.load(src, mask=mask)
        if is_left_overlap:
            tl.debug_barrier()
        tl.store(dst, data, mask=mask)


@triton.jit
def _precopy_mamba_align_grouped_kernel(
    state_idx_ptr,
    src_col_ptr,
    token_bias_ptr,
    block_table_ptrs_ptr,
    block_table_stride_req,
    state_base_addrs_ptr,
    state_block_strides_ptr,
    state_elem_sizes_ptr,
    state_inner_sizes_ptr,
    state_conv_widths_ptr,
    state_group_indices_ptr,
    idx_mapping_ptr,
    query_start_loc_ptr,
    num_reqs,
    num_states,
    COPY_BLOCK_SIZE: tl.constexpr,
    TEMPORAL_TILES: tl.constexpr,
    STATE_GROUPS: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    state_group = tl.program_id(1)
    tile_idx = tl.program_id(2)
    if batch_idx >= num_reqs:
        return
    req_idx = tl.load(idx_mapping_ptr + batch_idx)
    if req_idx < 0:
        return

    # Keep the per-state grid's single-token speculative tail rule: when the
    # request advanced by exactly one token and the source column is the one
    # the state still points at (src_col == dst_col), the tail column must
    # still be copied. Dropping this (as the fork base does) changes DFlash2
    # state for single-token prefill chunks, so it is preserved here.
    src_col = tl.load(src_col_ptr + req_idx)
    dst_col = tl.load(state_idx_ptr + req_idx)
    token_bias = tl.load(token_bias_ptr + req_idx)
    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    single_token_tail = query_end - query_start == 1 and token_bias > 0
    if src_col < 0 or (src_col == dst_col and not single_token_tail):
        return
    for state_idx in range(state_group, num_states, STATE_GROUPS):
        _copy_mamba_state_block_grouped(
            state_idx,
            batch_idx,
            src_col,
            dst_col,
            token_bias,
            block_table_ptrs_ptr,
            block_table_stride_req,
            state_base_addrs_ptr,
            state_block_strides_ptr,
            state_elem_sizes_ptr,
            state_inner_sizes_ptr,
            state_conv_widths_ptr,
            state_group_indices_ptr,
            tile_idx,
            COPY_BLOCK_SIZE,
            TEMPORAL_TILES,
        )


@triton.jit
def _postprocess_mamba_align_grouped_kernel(
    num_accepted_snapshot_ptr,
    num_accepted_out_ptr,
    state_idx_ptr,
    new_num_computed_tokens_ptr,
    block_table_ptrs_ptr,
    block_table_stride_req,
    state_base_addrs_ptr,
    state_block_strides_ptr,
    state_elem_sizes_ptr,
    state_inner_sizes_ptr,
    state_conv_widths_ptr,
    state_group_indices_ptr,
    idx_mapping_ptr,
    num_reqs,
    num_states,
    MAMBA_BLOCK_SIZE: tl.constexpr,
    COPY_BLOCK_SIZE: tl.constexpr,
    TEMPORAL_TILES: tl.constexpr,
    STATE_GROUPS: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    state_group = tl.program_id(1)
    tile_idx = tl.program_id(2)
    if batch_idx >= num_reqs:
        return
    req_idx = tl.load(idx_mapping_ptr + batch_idx)
    if req_idx < 0:
        return

    num_accepted = tl.load(num_accepted_snapshot_ptr + req_idx)
    src_col = tl.load(state_idx_ptr + req_idx)
    new_num_computed = tl.load(new_num_computed_tokens_ptr + req_idx)
    num_tokens_running_state = new_num_computed - num_accepted + 1
    aligned_new_computed = (new_num_computed // MAMBA_BLOCK_SIZE) * MAMBA_BLOCK_SIZE
    if aligned_new_computed < num_tokens_running_state:
        return

    token_bias = aligned_new_computed - num_tokens_running_state
    dst_col = aligned_new_computed // MAMBA_BLOCK_SIZE - 1
    # Single writer, as in the per-state grid: program (req, 0, 0).
    if src_col == dst_col and state_group == 0 and tile_idx == 0:
        tl.store(num_accepted_out_ptr + req_idx, 1)
    if src_col == dst_col and token_bias == 0:
        return

    for state_meta_idx in range(state_group, num_states, STATE_GROUPS):
        _copy_mamba_state_block_grouped(
            state_meta_idx,
            batch_idx,
            src_col,
            dst_col,
            token_bias,
            block_table_ptrs_ptr,
            block_table_stride_req,
            state_base_addrs_ptr,
            state_block_strides_ptr,
            state_elem_sizes_ptr,
            state_inner_sizes_ptr,
            state_conv_widths_ptr,
            state_group_indices_ptr,
            tile_idx,
            COPY_BLOCK_SIZE,
            TEMPORAL_TILES,
        )


def _use_grouped_grid(grouped: bool | None) -> bool:
    if grouped is not None:
        return grouped
    return _USE_GROUPED_COPY_GRID and _grouped_copy_grid_device_ok()


def run_mamba_align_precopy(
    ctx: MambaSpecDecodeGPUContext,
    num_reqs: int,
    state_idx: torch.Tensor,
    src_col: torch.Tensor,
    token_bias: torch.Tensor,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    grouped: bool | None = None,
) -> None:
    if num_reqs == 0 or not ctx.is_initialized:
        return
    if _use_grouped_grid(grouped):
        num_states = int(ctx.num_states)
        grid = (num_reqs, min(num_states, _STATE_GROUPS), _TEMPORAL_TILES)
        _precopy_mamba_align_grouped_kernel[grid](
            state_idx,
            src_col,
            token_bias,
            ctx.block_table_ptrs,
            ctx.block_table_stride_req,
            ctx.state_base_addrs,
            ctx.state_block_strides,
            ctx.state_elem_sizes,
            ctx.state_inner_sizes,
            ctx.state_conv_widths,
            ctx.state_group_indices,
            idx_mapping,
            query_start_loc,
            num_reqs,
            num_states,
            COPY_BLOCK_SIZE=_GROUPED_COPY_BLOCK_SIZE,
            TEMPORAL_TILES=_TEMPORAL_TILES,
            STATE_GROUPS=_STATE_GROUPS,
        )
        return
    grid = (num_reqs, ctx.num_states, _TEMPORAL_TILES)
    _precopy_mamba_align_kernel[grid](
        state_idx,
        src_col,
        token_bias,
        ctx.block_table_ptrs,
        ctx.block_table_stride_req,
        ctx.state_base_addrs,
        ctx.state_block_strides,
        ctx.state_elem_sizes,
        ctx.state_inner_sizes,
        ctx.state_conv_widths,
        ctx.state_group_indices,
        idx_mapping,
        query_start_loc,
        num_reqs,
        COPY_BLOCK_SIZE=1024,
        TEMPORAL_TILES=_TEMPORAL_TILES,
    )


def run_mamba_align_postprocess(
    ctx: MambaSpecDecodeGPUContext,
    num_reqs: int,
    num_accepted_tokens: torch.Tensor,
    state_idx: torch.Tensor,
    new_num_computed_tokens: torch.Tensor,
    idx_mapping: torch.Tensor,
    grouped: bool | None = None,
) -> None:
    if num_reqs == 0 or not ctx.is_initialized:
        return
    # Snapshot the decision array because one program may reset a request's
    # output count while programs copying other state tensors still read it.
    snapshot = ctx.num_accepted_tokens_out
    snapshot.copy_(num_accepted_tokens)
    if _use_grouped_grid(grouped):
        num_states = int(ctx.num_states)
        grid = (num_reqs, min(num_states, _STATE_GROUPS), _TEMPORAL_TILES)
        _postprocess_mamba_align_grouped_kernel[grid](
            snapshot,
            num_accepted_tokens,
            state_idx,
            new_num_computed_tokens,
            ctx.block_table_ptrs,
            ctx.block_table_stride_req,
            ctx.state_base_addrs,
            ctx.state_block_strides,
            ctx.state_elem_sizes,
            ctx.state_inner_sizes,
            ctx.state_conv_widths,
            ctx.state_group_indices,
            idx_mapping,
            num_reqs,
            num_states,
            MAMBA_BLOCK_SIZE=ctx.block_size,
            COPY_BLOCK_SIZE=_GROUPED_COPY_BLOCK_SIZE,
            TEMPORAL_TILES=_TEMPORAL_TILES,
            STATE_GROUPS=_STATE_GROUPS,
        )
        return
    grid = (num_reqs, ctx.num_states, _TEMPORAL_TILES)
    _postprocess_mamba_align_kernel[grid](
        snapshot,
        num_accepted_tokens,
        state_idx,
        new_num_computed_tokens,
        ctx.block_table_ptrs,
        ctx.block_table_stride_req,
        ctx.state_base_addrs,
        ctx.state_block_strides,
        ctx.state_elem_sizes,
        ctx.state_inner_sizes,
        ctx.state_conv_widths,
        ctx.state_group_indices,
        idx_mapping,
        num_reqs,
        MAMBA_BLOCK_SIZE=ctx.block_size,
        COPY_BLOCK_SIZE=1024,
        TEMPORAL_TILES=_TEMPORAL_TILES,
    )
