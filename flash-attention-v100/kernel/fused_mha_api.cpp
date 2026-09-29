#include <torch/extension.h>
#include <ATen/ATen.h>
#include <stdexcept>
#include "fused_mha.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "FlashAttention-2 implementation optimized for Volta";
  m.def("fwd", &flash_attention_forward,
        "FlashAttention-2 Forward Pass (Volta)");
  m.def("qk_scores_fwd", &flash_attention_qk_scores,
        "Debug FlashAttention QK score dump before softmax (Volta)");
  m.def("bwd", &flash_attention_backward,
        "FlashAttention-2 Backward Pass (Volta)");
  m.def("decode_paged_fwd", &flash_attention_decode_paged,
        "FlashAttention decode over paged KV cache (Volta)");
  m.def("decode_paged_xqa_fwd", &flash_attention_decode_paged_xqa,
        "FlashAttention XQA decode over paged KV cache (Volta)");
  m.def("decode_paged_xqa_staged_fwd", &flash_attention_decode_paged_xqa_staged,
        "Staged FlashAttention XQA decode over paged KV cache (Volta)");
  m.def("grouped_verify_paged_fwd", &flash_attention_grouped_verify_paged,
        "Exact grouped DFlash2 verification over paged KV cache (Volta)");
  m.def("grouped_verify_max_query_tokens",
        &flash_attention_grouped_verify_max_query_tokens,
        "Maximum query length supported by grouped DFlash2 verification");
  // [request-major patch] Capability probe: 1 = the grouped-verify operator
  // accepts a [batch, blocks] block_table and verifies all requests in one
  // launch (gridDim.z == batch). Older binaries lack this symbol entirely, so
  // the Python side treats its absence as "single-request only".
  m.def("grouped_verify_request_major_abi_version",
        &flash_attention_grouped_verify_request_major_abi_version,
        "ABI version of request-major grouped DFlash2 verification (0=absent)");
  // [e4m3 patch] Single-request E4M3 grouped verification over an E4M3 paged
  // KV cache, retaining FP32 numerators in the partial workspace.
  m.def("grouped_e4m3_fp32_paged_fwd", &flash_attention_grouped_e4m3_fp32_paged,
        "Exact grouped E4M3 FP32 verification over paged KV cache (Volta)");
  m.def("grouped_e4m3_fp32_precision_version",
        &flash_attention_grouped_e4m3_fp32_precision_version,
        "Workspace/precision revision of the E4M3 FP32 grouped operator");
  m.def("grouped_sparse_page4_fwd", &flash_attention_grouped_sparse_page4,
        "Grouped exact QSA page4 attention over paged KV cache (Volta)");
  // [sparse-page4 dtype patch] Capability probe: 2 = the sparse page4 entry
  // point accepts kv_cache_dtype / k_scale / v_scale. Older binaries lack the
  // symbol, so Python treats its absence as "fp16 KV only, 9-argument call".
  m.def("grouped_sparse_page4_abi_version",
        &flash_attention_grouped_sparse_page4_abi_version,
        "ABI version of grouped sparse page4 attention (0=absent)");
  m.def("grouped_sparse_page4_plan_fwd",
        &flash_attention_grouped_sparse_page4_plan,
        "Build grouped exact QSA page4 tables over paged KV cache (Volta)");
  m.def("decode_paged_wmma_fwd", &flash_attention_decode_paged_wmma,
        "FlashAttention single-query decode through paged-prefill WMMA order "
        "(Volta)");
  m.def("decode_qk_scores_fwd", &flash_attention_decode_qk_scores,
        "Debug scalar paged decode QK score dump before softmax (Volta)");
  m.def("decode_turboquant_paged_fwd", &flash_attention_turboquant_decode_paged,
        "FlashAttention decode over TurboQuant paged KV cache (Volta)");
  m.def("prefill_paged_fwd", &flash_attention_prefill_paged,
        "FlashAttention prefill over paged KV cache (Volta)");
  m.def("prefill_paged_d256_bm32_allp_pair_scratch_fwd",
        &flash_attention_prefill_paged_d256_bm32_allp_pair_scratch,
        "Fixed causal D256 BM32 ALL_P pair-scratch paged prefill (SM70)");
  m.def("prefill_paged_d256_bm32_allp_pair_scratch_splitkv3_fwd",
        &flash_attention_prefill_paged_d256_bm32_allp_pair_scratch_splitkv3,
        "Fixed causal D256 BM32 ALL_P pair-scratch three-way split-KV paged "
        "prefill (SM70)");
  m.def("prefill_paged_bfla_fwd", &flash_attention_prefill_paged_bfla,
        "BFLA sparse FlashAttention prefill over paged KV cache (Volta)");
  m.def("prefill_paged_splitkv_fwd", &flash_attention_prefill_paged_splitkv,
        "FlashAttention split-KV prefill over paged KV cache (Volta)");
  m.def("fp8_e5m2_paged_kv_to_fp16", &flash_attention_fp8_e5m2_paged_kv_to_fp16,
        "Expand paged FP8 E5M2 K/V into a preallocated FP16 paged workspace");
}
