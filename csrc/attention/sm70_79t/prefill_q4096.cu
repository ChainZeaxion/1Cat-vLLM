// SPDX-License-Identifier: BSD-3-Clause

// Build the same FP32-accumulated architecture with a native Q4096 causal
// tail.  A private namespace keeps its device globals and workspace cache
// independent from the other specializations.
//
// Why this width exists: with the opt23 auto threshold at
// 0.5 * --max-num-batched-tokens, a 8192-token step budget yields a 4096-token
// prefill chunk.  Q4096 covers the [3997, 4096] window, so that configuration
// reaches this route without having to raise the batch budget to 16384 -- and
// not raising the budget keeps the CUDA-graph memory reserve (and therefore
// the KV pool) at its smaller, faster-to-serve size.
#define FLASH_NAMESPACE onecat_79t_q4096
#define PREFIX_TORCH_QUERY_TOKENS 4096
#define PREFIX_TORCH_ARCHITECTURE_FUNCTION sm70_d256_gqa_architecture_q4096_fwd
#undef PREFIX_BATCHED_TAIL_TILE_TOKENS
#define PREFIX_BATCHED_TAIL_TILE_TOKENS 256

#include "prefill.cu"
