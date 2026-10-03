// SPDX-License-Identifier: BSD-3-Clause

// Build the same FP32-accumulated architecture with a native Q2560 causal
// tail.  A private namespace keeps its device globals and workspace cache
// independent from the other specializations.
//
// Why this width exists: the SM70 long-prefill fast route is a *fixed query
// width* kernel (the Q is a compile-time constant, the tail tile is qualified
// per width).  Before this file only 8000 and 8192 existed, so chunked prefill
// -- which produces chunks around `long_prefill_token_threshold` -- could only
// reach the route when a chunk happened to land in [8000, 8192].  Everything
// in the middle fell through to the generic path.  Q2560 covers the
// [2498, 2560] window, which is where a 2560-token chunk budget lands
// (e.g. --max-num-batched-tokens 5120 with the opt23 auto threshold).
#define FLASH_NAMESPACE onecat_79t_q2560
#define PREFIX_TORCH_QUERY_TOKENS 2560
#define PREFIX_TORCH_ARCHITECTURE_FUNCTION sm70_d256_gqa_architecture_q2560_fwd
#undef PREFIX_BATCHED_TAIL_TILE_TOKENS
#define PREFIX_BATCHED_TAIL_TILE_TOKENS 256

#include "prefill.cu"
