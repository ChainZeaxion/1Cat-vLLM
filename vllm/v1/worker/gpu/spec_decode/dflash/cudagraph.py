# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable, Mapping

import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger

logger = init_logger(__name__)
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    build_slot_mappings_by_layer,
)
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.cp_utils import prepare_dcp_local_seq_lens
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CapturedAttentionState,
    CudaGraphManager,
)
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.utils import AttentionGroup


def _prepare_dflash_inputs_to_capture(
    num_reqs: int,
    num_tokens: int,
    input_buffers: InputBuffers,
    block_tables: BlockTables,
    query_slot_mappings: torch.Tensor,
    attn_groups: list[list[AttentionGroup]],
    kv_cache_config: KVCacheConfig,
    max_model_len: int,
    skip_attn: bool,
    causal: bool | Mapping[int, bool],
) -> CapturedAttentionState:
    input_batch = InputBatch.make_dummy(num_reqs, num_tokens, input_buffers)
    input_block_tables = block_tables.get_dummy_block_tables(num_reqs)
    # DFlash owns a K+1 query batch which can be wider than the target token
    # budget. Keep its persistent graph input independent from the target
    # BlockTables slot-mapping allocation.
    query_slot_mappings.fill_(PAD_SLOT_ID)
    slot_mappings = query_slot_mappings[:, :num_tokens]
    slot_mappings_by_layer = build_slot_mappings_by_layer(
        slot_mappings, kv_cache_config
    )

    attn_metadata = None
    if not skip_attn:
        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
        dcp_local_seq_lens = None
        if block_tables.cp_size > 1:
            prepare_dcp_local_seq_lens(
                input_buffers.dcp_local_seq_lens,
                input_buffers.seq_lens,
                num_reqs,
                block_tables.cp_size,
                block_tables.cp_rank,
                block_tables.cp_interleave,
            )
            dcp_local_seq_lens = input_buffers.dcp_local_seq_lens
        attn_metadata = build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=num_tokens // num_reqs,
            seq_lens=input_batch.seq_lens,
            dcp_local_seq_lens=dcp_local_seq_lens,
            max_seq_len=max_model_len,
            block_tables=input_block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            for_cudagraph_capture=True,
            causal=causal,
        )
    return CapturedAttentionState(attn_metadata, slot_mappings_by_layer)


class DFlashCudaGraphManager(CudaGraphManager):
    """DFlash CudaGraphManager for the parallel-drafting query forward,
    building its own attention metadata from scratch."""

    def capture(
        self,
        forward_fn: Callable,
        input_buffers: InputBuffers,
        block_tables: BlockTables,
        query_slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        causal: bool | Mapping[int, bool],
        progress_bar_desc: str = "Capturing CUDA graphs",
    ) -> None:
        # [gguf-lineb] 预热 smallq metadata Triton 核：batch=1 的 FULL 图会走
        # smallq 路径，而该核此前从未 launch 时 _init_handles 在捕获流里执行
        # 会报 "operation not permitted when stream is capturing"（32K 长度
        # 下稳定复现于 7/8→8/8）。图外先 launch 一次完成句柄初始化。
        try:
            import triton
            from vllm.v1.attention.backends.flash_attn_v100 import (
                _sm70_prepare_smallq_decode_metadata,
            )

            device = torch.device("cuda", torch.cuda.current_device())
            warm_reqs, warm_tokens = 2, 8
            warm_block_table = torch.zeros(
                (warm_tokens, 1), dtype=torch.int32, device=device
            )
            warm_seq_lens = torch.full(
                (warm_reqs,), 1, dtype=torch.int32, device=device
            )
            # query_start_loc 必须是「容量长度」缓冲：核内对
            # qsl[req_idx+1] 无掩码直读（req_idx 可到 warm_reqs-1），
            # 还要预留 next_pow2(warm_reqs+1) 的掩码读区。取
            # REQ_BLOCK+1 长度、单调到 warm_tokens 的哨兵分布。
            warm_req_block = triton.next_power_of_2(warm_reqs + 1)
            warm_qsl_list = [
                min(i * (warm_tokens // warm_req_block), warm_tokens)
                for i in range(warm_req_block + 1)
            ]
            warm_qsl_list[-1] = warm_tokens
            warm_qsl = torch.tensor(
                warm_qsl_list, dtype=torch.int32, device=device
            )
            _sm70_prepare_smallq_decode_metadata(
                torch.zeros((warm_tokens, 1), dtype=torch.int32, device=device),
                torch.zeros((warm_tokens,), dtype=torch.int32, device=device),
                torch.zeros((warm_reqs + 1,), dtype=torch.int32, device=device),
                warm_block_table,
                warm_seq_lens,
                warm_qsl,
                num_reqs=warm_reqs,
                num_query_tokens=warm_tokens,
                real_num_query_tokens=warm_tokens,
            )
            # batch=1 的 dflash noncausal paged batch 路径会走
            # sm70_dflash2_split 的 part/merge Triton 核（单请求 split-KV
            # 窗口），同样是捕获期首启即崩。图外以最小输入各 launch 一次。
            try:
                from flash_attn_v100.sm70_dflash2_split import (
                    forward as _split_fwd,
                )

                # part 核 Q 布局固定为 (B, 64, 128)（8 个 query 位 ×
                # 8 头展平在第二维），核内按 ((b*8+qr)*8+h) 直读，形状
                # 给小了会越界读 → illegal access 污染上下文。
                warm_q = torch.zeros(
                    (1, 64, 128), dtype=torch.float16, device=device
                )
                warm_k = torch.zeros(
                    (1, 1, 1024, 128), dtype=torch.float16, device=device
                )
                warm_v = torch.zeros_like(warm_k)
                warm_tbl = torch.zeros((1, 1), dtype=torch.int32, device=device)
                warm_len = torch.full((1,), 8, dtype=torch.int32, device=device)
                _split_fwd(
                    warm_q, warm_k, warm_v, warm_tbl, warm_len, 0.125,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning_once(
                    "DFlash2 split-kernel warmup skipped: %s", e
                )
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001
            logger.warning_once("DFlash2 smallq metadata warmup skipped: %s", e)

        def create_forward_fn(
            desc: BatchExecutionDescriptor,
        ) -> tuple[Callable[[CUDAGraphMode], None], CapturedAttentionState]:
            num_tokens = desc.num_tokens
            num_reqs = desc.num_reqs or min(num_tokens, self.max_num_reqs)
            num_tokens_across_dp = (
                torch.full((self.dp_size,), num_tokens, dtype=torch.int32, device="cpu")
                if self.dp_size > 1
                else None
            )
            attn_state = _prepare_dflash_inputs_to_capture(
                num_reqs,
                num_tokens,
                input_buffers,
                block_tables,
                query_slot_mappings,
                attn_groups,
                kv_cache_config,
                max_model_len,
                skip_attn=(desc.cg_mode == CUDAGraphMode.PIECEWISE),
                causal=causal,
            )
            attn_metadata, slot_mappings = attn_state

            return (
                lambda cg_mode: forward_fn(
                    num_reqs,
                    num_tokens,
                    attn_metadata,
                    slot_mappings,
                    num_tokens_across_dp,
                    cg_mode,
                ),
                attn_state,
            )

        super().capture(create_forward_fn, progress_bar_desc)
