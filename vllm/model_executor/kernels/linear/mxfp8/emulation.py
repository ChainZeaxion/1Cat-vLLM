# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.nn.parameter import Parameter

from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    MXFP8_SCALE_DTYPE,
    dequant_mxfp8_to_bf16,
)

from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig


class EmulationMxfp8LinearKernel(Mxfp8LinearKernel):
    """Software emulation fallback for MXFP8 (dequant to BF16)."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        return True, None

    @classmethod
    def can_implement(cls, c: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data  # [N, K]
        N, K = weight.shape
        scale_k = K // MXFP8_BLOCK_SIZE

        weight_scale = layer.weight_scale.data[:N, :scale_k].contiguous()

        # Dequantize once here rather than per forward. Keeping an fp8 tensor on
        # the module would put fp8e4nv into the captured graph, which SM70
        # cannot lower; and the emulation path would otherwise rebuild the
        # whole FP16 weight matrix on every step.
        dequantized = dequant_mxfp8_to_bf16(
            weight.contiguous(), weight_scale
        ).to(torch.float16)

        layer.weight = Parameter(dequantized, requires_grad=False)
        layer.weight_scale = Parameter(weight_scale, requires_grad=False)
        layer._mxfp8_predequantized = True

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight = layer.weight
        # process_weights_after_loading already materialized the FP16 weight.
        if not getattr(layer, "_mxfp8_predequantized", False):
            weight_scale = layer.weight_scale
            if weight_scale.dtype != MXFP8_SCALE_DTYPE:
                raise ValueError(
                    f"Emulation backend requires {MXFP8_SCALE_DTYPE} "
                    f"weight_scale dtype, got {weight_scale.dtype}."
                )
            if weight_scale.ndim != 2:
                raise ValueError(
                    f"Emulation backend requires 2D weight_scale, "
                    f"got {weight_scale.ndim}D. "
                    f"Ensure process_weights_after_loading was called."
                )
            weight = dequant_mxfp8_to_bf16(weight, weight_scale)

        # Follow the activation dtype: SM70 devices run FP16, where a BF16
        # weight would not match mat1 and torch rejects the GEMM outright.
        if weight.dtype != x.dtype:
            weight = weight.to(x.dtype)
        output = torch.nn.functional.linear(x, weight, bias)
        return output.to(x.dtype)
