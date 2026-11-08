# SPDX-License-Identifier: Apache-2.0
"""Fixed per-row RMS reductions for the experimental speculative parity mode."""
import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _norm_kernel(X, R, W, Y, RO, N: tl.constexpr,
                 X_STRIDE: tl.constexpr, R_STRIDE: tl.constexpr,
                 EPS: tl.constexpr, HAS_RESIDUAL: tl.constexpr,
                 HAS_WEIGHT: tl.constexpr, GEMMA: tl.constexpr,
                 CAST_BEFORE_WEIGHT: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * X_STRIDE + cols, mask, 0).to(tl.float32)
    if HAS_RESIDUAL:
        x += tl.load(R + row * R_STRIDE + cols, mask, 0).to(tl.float32)
        tl.store(RO + row * N + cols, x, mask)
    # Keep the residual sum in FP32 through variance, matching native IR.
    variance = tl.sum(x * x, 0) / N
    normalized = x * tl.rsqrt(variance + EPS)
    if HAS_WEIGHT:
        w = tl.load(W + cols, mask, 0)
        if GEMMA:
            normalized *= w.to(tl.float32) + 1.0
        else:
            if CAST_BEFORE_WEIGHT:
                normalized = normalized.to(W.dtype.element_ty).to(tl.float32)
            normalized *= w.to(tl.float32)
    tl.store(Y + row * N + cols, normalized, mask)


def token_parity_rms_norm(x, weight, eps, residual=None, *, gemma=False):
    """Return normalized rows using the same launch layout at every batch size."""
    shape = x.shape
    rows = x.reshape(-1, shape[-1]).contiguous()
    residual_rows = residual.reshape_as(rows).contiguous() if residual is not None else rows
    out = torch.empty_like(rows)
    residual_out = torch.empty_like(rows) if residual is not None else out
    _norm_kernel[(rows.shape[0],)](
        rows, residual_rows, weight if weight is not None else rows,
        out, residual_out, rows.shape[-1], rows.stride(0), residual_rows.stride(0),
        eps, residual is not None, weight is not None, gemma,
        weight is not None and weight.dtype in (torch.bfloat16, torch.float16),
        triton.next_power_of_2(rows.shape[-1]), num_warps=4,
    )
    output = out.reshape(shape)
    return (output, residual_out.reshape(shape)) if residual is not None else output
