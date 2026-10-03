// SPDX-License-Identifier: BSD-3-Clause

// Build the same FP32-accumulated architecture with a native Q3072 causal
// tail.  A private namespace keeps its device globals and workspace cache
// independent from the other specializations.
//
// Why this width exists: it is the middle rung between Q2560 and Q4096.  Each
// width serves chunks down to 3/4 of itself, so 2560 -> (1920, 2560] and
// 4096 -> (3072, 4096] would otherwise leave (2560, 3072] unreachable and fall
// straight to the generic path.  Q3072 -> (2304, 3072] stitches the two
// together, giving continuous coverage from 1920 through 4096.
#define FLASH_NAMESPACE onecat_79t_q3072
#define PREFIX_TORCH_QUERY_TOKENS 3072
#define PREFIX_TORCH_ARCHITECTURE_FUNCTION sm70_d256_gqa_architecture_q3072_fwd
#undef PREFIX_BATCHED_TAIL_TILE_TOKENS
#define PREFIX_BATCHED_TAIL_TILE_TOKENS 256

#include "prefill.cu"
