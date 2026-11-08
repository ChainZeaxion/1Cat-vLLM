# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-side compact/reference dispatch for a uniform DFlash2 verifier."""

from typing import Any

import os

import numpy as np
import torch
import torch.distributed as dist

# Registers the SM70 sampler sidecar (probe_if_else_size2, attach_branch,
# prepare_collective) into torch.ops._C. The op probes below are meaningless
# without it, and relying on another module's import would make the gate
# silently false whenever this module is imported first.
import vllm._sm70_ops  # noqa: F401
from vllm.distributed import get_tensor_model_parallel_world_size, get_tp_group
from vllm.distributed.parallel_state import graph_capture
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.sample.ops.topk_topp_triton import sort_topk_with_vocab_ties
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    dflash2_sparse_topk_rejection_sample,
    rejection_sample,
)

logger = init_logger(__name__)

ROWS_PER_REQUEST = 8
# Each request count captures its own graphs into private pools the KV budget
# does not cover: about 170 MiB per count up to c4, and 2.7 GiB in total once
# c5-c8 are captured too, which also ran slower than the host-decision path.
MAX_GRAPH_REQUESTS = 4
MAX_GRAPHS = 8
# Capture allocates outside the budget. Serving still needs room after it: the
# host-decision path at c8 reserves about 1.3 GiB on top of the idle footprint.
CAPTURE_HEADROOM_BYTES = 5 << 29


@triton.jit
def _prepare_hidden(
    hidden,
    fixed_hidden,
    dense,
    dense_pointer,
    hidden_size: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    tl.store(
        fixed_hidden + offsets,
        tl.load(hidden + offsets, offsets < hidden_size, 0),
        offsets < hidden_size,
    )
    if tl.program_id(0) == 0:
        tl.store(dense_pointer, dense.to(tl.int64))


@triton.jit
def _prepare_request(
    inputs,
    fixed_inputs,
    positions,
    fixed_positions,
    source_mapping,
    mapping,
    expanded,
    ids,
    fixed_ids,
    values,
    fixed_values,
    sparse_size: tl.constexpr,
    source_sparse_stride: tl.constexpr,
    fixed_sparse_stride: tl.constexpr,
    rows: tl.constexpr,
    block: tl.constexpr,
):
    index = tl.program_id(0)
    request = tl.load(source_mapping + index)
    offsets = tl.arange(0, block)
    first = index * rows
    tl.store(
        fixed_ids + request * fixed_sparse_stride + offsets,
        tl.load(
            ids + request * source_sparse_stride + offsets, offsets < sparse_size, 0
        ),
        offsets < sparse_size,
    )
    tl.store(
        fixed_values + request * fixed_sparse_stride + offsets,
        tl.load(
            values + request * source_sparse_stride + offsets, offsets < sparse_size, 0
        ),
        offsets < sparse_size,
    )
    tl.store(mapping + index, request)
    tl.store(expanded + first + offsets, request, offsets < rows)
    tl.store(
        fixed_inputs + first + offsets,
        tl.load(inputs + first + offsets, offsets < rows, 0),
        offsets < rows,
    )
    tl.store(
        fixed_positions + first + offsets,
        tl.load(positions + first + offsets, offsets < rows, 0),
        offsets < rows,
    )


@triton.jit
def _copy_dense_slot(
    source_pointer,
    destination,
    mapping,
    destination_stride: tl.constexpr,
    slot_size: tl.constexpr,
    block: tl.constexpr,
):
    # The source address is refreshed with the current request's inputs.
    # Only the reference branch reads dense proposals; compact rounds incur
    # no full-vocabulary copy, and inactive request slots are never copied.
    source = tl.load(source_pointer).to(destination.dtype)
    request = tl.load(mapping + tl.program_id(0))
    offsets = tl.program_id(1) * block + tl.arange(0, block)
    values = tl.load(source + request * slot_size + offsets, offsets < slot_size, 0)
    tl.store(
        destination + request * destination_stride + offsets,
        values,
        offsets < slot_size,
    )


def _capture_headroom(device: torch.device) -> bool:
    free, _ = torch.cuda.mem_get_info(device)
    idle = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    return free + idle >= CAPTURE_HEADROOM_BYTES


# The conditional-graph branch needs TWO child bodies (reference / compact),
# i.e. cudaGraphCondTypeIf with size=2. Measured on driver API 12.5 (555.42):
# every conditional node with size >= 2 is rejected with cudaErrorInvalidValue
# while size == 1 succeeds and a size-1 branch executes end-to-end. The check
# is a real capability probe, not a version guess: a one-node If node is built
# once on a scratch graph, so a newer driver that lifts the restriction is
# picked up automatically.
_conditional_size2_cache: bool | None = None


def _conditional_size2_supported() -> bool:
    global _conditional_size2_cache
    if _conditional_size2_cache is None:
        _conditional_size2_cache = False
        # The driver capability alone does not make the two-branch sampler
        # correct: on 575.57.08 the graph builds and replays but temperatures
        # above zero degenerate into repeating filler tokens while the host
        # path stays healthy. Keep it opt-in until the numerical cause is
        # found, so a driver upgrade cannot silently corrupt serving output.
        if os.getenv("VLLM_SM70_DFLASH2_SAMPLER_GRAPH", "0") != "1":
            return _conditional_size2_cache
        if torch.cuda.get_device_properties(0).major >= 7 and hasattr(
            torch.ops._C, "sm70_sampler_graph_probe_if_else_size2"
        ):
            try:
                anchor = torch.zeros(1, dtype=torch.uint8, device="cuda")
                _conditional_size2_cache = (
                    int(
                        torch.ops._C.sm70_sampler_graph_probe_if_else_size2(anchor)
                    )
                    == 0
                )
            except Exception:
                _conditional_size2_cache = False
    return _conditional_size2_cache


def _all_ranks_support(status: int) -> bool:
    # Capability disagreement must select the existing sampler on every rank.
    # This is a capture-time CPU collective, never a per-round GPU fence.
    import os as _os
    if _os.environ.get("VLLM_SM70_DFLASH2_SAMPLER_GRAPH_TRACE") == "1":
        try:
            logger.info(
                "SGRANK enter rank=%s status=%s",
                dist.get_rank() if dist.is_initialized() else -1,
                status,
            )
        except Exception:
            pass
    with torch.inference_mode(False):
        unsupported = torch.tensor(int(status != 0), dtype=torch.int32)
        dist.all_reduce(
            unsupported, op=dist.ReduceOp.MAX, group=get_tp_group().cpu_group
        )
    if _os.environ.get("VLLM_SM70_DFLASH2_SAMPLER_GRAPH_TRACE") == "1":
        try:
            logger.info(
                "SGRANK exit  rank=%s unsupported=%s",
                dist.get_rank() if dist.is_initialized() else -1,
                int(unsupported.item()),
            )
        except Exception:
            pass
    return unsupported.item() == 0


class _RejectionGraph:
    def __init__(self, model, speculator, rejection, hidden, batch, sparse):
        self.requests = batch.num_reqs
        rows = self.requests * ROWS_PER_REQUEST
        self.hidden = torch.empty_like(hidden)
        self.inputs = torch.empty_like(batch.input_ids[:rows])
        self.positions = torch.empty_like(batch.positions[:rows])
        # MRv2 allocates expanded metadata anew every round. Keep graph inputs
        # independent of those allocations and preserve the original slot ID.
        self.mapping = batch.idx_mapping[: self.requests].clone()
        self.expanded = batch.expanded_idx_mapping[:rows].clone()
        self.local = (
            torch.arange(
                ROWS_PER_REQUEST,
                dtype=batch.expanded_local_pos.dtype,
                device=hidden.device,
            )
            .repeat(self.requests)
        )
        self.cu = batch.cu_num_logits[: self.requests + 1].clone()
        self.request_indices = batch.idx_mapping_np.copy()
        self.rejection = rejection
        self.model = model
        self.speculator = speculator
        self.sparse = tuple(torch.empty_like(t) for t in sparse)
        dense = speculator.draft_logits
        if self.requests == 1:
            # Alias the one private proposal slot for every state index while
            # preserving the original request mapping. Dense kernels accept
            # stride(0)==0; no other request slot is read.
            self.draft_slot = dense.new_empty((1, *dense.shape[1:]))
            self.draft_logits = self.draft_slot.expand(dense.shape[0], -1, -1)
        else:
            # Every replay refills the slots it reads, so graphs can share this.
            shared = getattr(rejection, "_sm70_dflash2_graph_draft_logits", None)
            if shared is None or shared.shape != dense.shape:
                shared = dense.new_empty(dense.shape)
                rejection._sm70_dflash2_graph_draft_logits = shared
            self.draft_logits = shared
        self.dense_pointer = self.inputs.new_empty(1, dtype=torch.int64)
        self.output = self.inputs.new_empty(
            (self.requests, ROWS_PER_REQUEST), dtype=torch.int64
        )
        self.count = self.inputs.new_empty(self.requests, dtype=torch.int32)
        # Sampling parameters live in UVA-backed tensors whose pool rotates
        # round-robin on every apply_staged_writes(), so states.temperature.gpu
        # is a DIFFERENT address each request. A captured graph freezes the
        # pointer it saw, which then reads a stale slot and corrupts sampling
        # (observed: temperature-dependent filler-token degeneration). Keep
        # private storage and refresh it in prepare(), same as positions.
        states = rejection.sampler.sampling_states
        self.temperature = states.temperature.gpu.new_empty(self.requests)
        self.top_p = states.top_p.gpu.new_empty(self.requests)
        self.seeds = states.seeds.gpu.new_empty(self.requests)
        self.counters = self.inputs.new_zeros(2, dtype=torch.int64)
        self.request_id = None
        self.prefix = torch.cuda.CUDAGraph(keep_graph=True)
        self.reference = torch.cuda.CUDAGraph(keep_graph=True)
        self.compact = torch.cuda.CUDAGraph(keep_graph=True)
        self.prepare(hidden, batch, sparse)

    def prepare(self, hidden, batch, sparse):
        ids, values = sparse
        fixed_ids, fixed_values = self.sparse
        _prepare_hidden[(triton.cdiv(hidden.numel(), 1024),)](
            hidden,
            self.hidden,
            self.speculator.draft_logits,
            self.dense_pointer,
            hidden.numel(),
            1024,
            num_warps=4,
        )
        _prepare_request[(self.requests,)](
            batch.input_ids,
            self.inputs,
            batch.positions,
            self.positions,
            batch.idx_mapping,
            self.mapping,
            self.expanded,
            ids,
            fixed_ids,
            values,
            fixed_values,
            ids.shape[1] * ids.shape[2],
            ids.stride(0),
            fixed_ids.stride(0),
            ROWS_PER_REQUEST,
            triton.next_power_of_2(ids.shape[1] * ids.shape[2]),
            num_warps=1,
        )
        # Refresh the private sampling parameters from the rotating UVA pool.
        # These are the live pointers the graph reads, so they must be updated
        # on every round even though the graph itself is captured once.
        states = self.rejection.sampler.sampling_states
        idx = batch.idx_mapping[: self.requests]
        self.temperature.copy_(states.temperature.gpu[idx])
        self.top_p.copy_(states.top_p.gpu[idx])
        self.seeds.copy_(states.seeds.gpu[idx])
        # Dense proposals are accessed indirectly inside a conditional body.
        # Record their consumer stream for replaced/eager tensor lifetimes.
        self.speculator.draft_logits.record_stream(torch.cuda.current_stream())

    def _probe(self):
        from .sparse_rejection import _compact_target_reference_flags

        # The default operand of getattr is evaluated eagerly, so a model that
        # lacks the fallback method raised even when the compact probe exists.
        probe = getattr(self.model, "get_compact_target_probe_with_fallback", None)
        if probe is None:
            probe = self.model.get_topk_tokens_and_logits_with_fallback
        self.ids, self.values, self.fallback = probe(self.hidden, 64)
        import os as _os
        if _os.environ.get("VLLM_GGUF_CAND_PROBE") and (not hasattr(self, "_probe_n") or self._probe_n < 3):
            self._probe_n = getattr(self, "_probe_n", 0) + 1
            try:
                import vllm.logger as _vl
                _lg = _vl.init_logger(__name__)
                _ids0 = self.ids[0].tolist()[:8]
                _vals0 = [round(v, 2) for v in self.values[0][:8].tolist()]
                _lg.info("TARGET_PROBE shape=%s ids0=%s vals0=%s nan=%s",
                         tuple(self.values.shape), _ids0, _vals0,
                         bool(torch.isnan(self.values).any().item()))
            except Exception as _e:
                import vllm.logger as _vl
                _vl.init_logger(__name__).info("TARGET_PROBE err %s", _e)
        if self.fallback is None:
            return False
        states = self.rejection.sampler.sampling_states
        self.values, self.ids = sort_topk_with_vocab_ties(
            self.values, self.ids, vocab_size=states.vocab_size, descending=True
        )
        self.flags = _compact_target_reference_flags(
            self.values, self.temperature, self.top_p, self.expanded
        )
        return True

    def _dense(self):
        slot_size = self.draft_logits.shape[1] * self.draft_logits.shape[2]
        _copy_dense_slot[(self.requests, triton.cdiv(slot_size, 4096))](
            self.dense_pointer,
            self.draft_logits,
            self.mapping,
            self.draft_logits.stride(0),
            slot_size,
            4096,
            num_warps=4,
        )
        logits = self.fallback()
        if logits is None:
            raise RuntimeError("DFlash2 graph rejection requires replicated logits")
        sampler = self.rejection.sampler
        states = sampler.sampling_states
        processed = sampler.apply_sampling_params(
            logits,
            self.expanded,
            self.request_indices,
            self.positions,
            self.inputs,
            self.local,
        )
        output, count = rejection_sample(
            processed,
            self.draft_logits,
            self.inputs,
            self.cu,
            self.positions,
            self.mapping,
            self.expanded,
            self.local,
            self.temperature,
            self.seeds,
            self.rejection.num_speculative_steps,
            use_fp64=sampler.use_fp64_gumbel,
        )
        self.output.copy_(output)
        self.count.copy_(count)

    def _compact(self):
        sampler = self.rejection.sampler
        states = sampler.sampling_states
        ids, values = self.sparse
        output, count = dflash2_sparse_topk_rejection_sample(
            self.ids,
            self.values,
            ids,
            values,
            self.inputs,
            self.cu,
            self.positions,
            self.mapping,
            self.temperature,
            self.top_p,
            self.seeds,
            self.rejection.num_speculative_steps,
            use_fp64=sampler.use_fp64_gumbel,
            target_top_k=20,
        )
        self.output.copy_(output)
        self.count.copy_(count)

    def capture(self):
        # Sampling is stateless in seed/position. Warmup writes only the private
        # outputs and does not advance request state or proposal random streams.
        with graph_capture(device=self.hidden.device):
            if not _all_ranks_support(0 if self._probe() else 801):
                return False
            self._dense()
            self._compact()
            torch.accelerator.synchronize()
            with torch.cuda.graph(self.prefix):
                self._probe()
            with torch.cuda.graph(self.reference):
                self._dense()
            with torch.cuda.graph(self.compact):
                self._compact()
        status = torch.ops._C.sm70_sampler_graph_prepare_collective(
            self.flags, self.reference.raw_cuda_graph()
        )
        if not _all_ranks_support(status):
            return False
        status = torch.ops._C.sm70_sampler_graph_attach_branch(
            self.flags,
            self.prefix.raw_cuda_graph(),
            self.reference.raw_cuda_graph(),
            self.compact.raw_cuda_graph(),
            self.counters,
        )
        if not _all_ranks_support(status):
            return False
        status = 0
        try:
            self.prefix.instantiate()
        except RuntimeError as error:
            logger.debug("CUDA rejected a DFlash2 sampler graph: %s", error)
            status = 801
        return _all_ranks_support(status)

    def replay(self, hidden, batch, sparse):
        request_id = tuple(batch.req_ids)
        if request_id != self.request_id:
            self.counters.zero_()
            self.request_id = request_id
        self.prepare(hidden, batch, sparse)
        self.prefix.replay()
        # AsyncOutput copies on another stream. Give it independent storage so
        # the next graph replay cannot overwrite a still-pending output copy.
        return SamplerOutput(
            sampled_token_ids=self.output.clone(),
            num_sampled=self.count.clone(),
            logprobs_tensors=None,
            num_nans=None,
        )


def try_graph_rejection(
    model: Any,
    speculator: Any,
    rejection: Any,
    hidden: torch.Tensor,
    batch: Any,
    sparse: tuple[torch.Tensor, torch.Tensor],
) -> SamplerOutput | None:
    """Caller has checked the positive-temperature compact sampling contract."""
    if not getattr(try_graph_rejection, "_diag", False):
        try_graph_rejection._diag = True
        try:
            logger.info(
                "SAMPLER_GRAPH_DIAG num_reqs=%s num_tokens=%s rows=%s nst=%s tp=%s "
                "has_fallback=%s has_prepare=%s cond2=%s vocab=%s contiguous=%s",
                getattr(batch, "num_reqs", None),
                getattr(batch, "num_tokens", None),
                tuple(hidden.shape), getattr(rejection, "num_speculative_steps", None),
                get_tensor_model_parallel_world_size(),
                hasattr(model, "get_topk_tokens_and_logits_with_fallback"),
                hasattr(torch.ops._C, "sm70_sampler_graph_prepare_collective"),
                _conditional_size2_supported(),
                getattr(rejection.sampler.sampling_states, "vocab_size", None),
                all(t.is_contiguous() for t in (hidden, speculator.draft_logits, *sparse)),
            )
        except Exception as ex:
            logger.info("SAMPLER_GRAPH_DIAG failed: %s", ex)
    if (
        not 1 <= getattr(batch, "num_reqs", 0) <= MAX_GRAPH_REQUESTS
        or batch.num_tokens != batch.num_reqs * ROWS_PER_REQUEST
        or hidden.shape[0] != batch.num_reqs * ROWS_PER_REQUEST
        or rejection.num_speculative_steps != ROWS_PER_REQUEST - 1
        or not np.array_equal(
            batch.cu_num_logits_np,
            np.arange(batch.num_reqs + 1) * ROWS_PER_REQUEST,
        )
        or get_tensor_model_parallel_world_size() != 4
        or not hasattr(model, "get_topk_tokens_and_logits_with_fallback")
        or not hasattr(torch.ops._C, "sm70_sampler_graph_prepare_collective")
        or not _conditional_size2_supported()
        or rejection.sampler.sampling_states.vocab_size < 32768
        or not all(
            t.is_contiguous() for t in (hidden, speculator.draft_logits, *sparse)
        )
        or sparse[0].stride(0) != sparse[1].stride(0)
    ):
        return None
    states = rejection.sampler.sampling_states
    idx = batch.idx_mapping_np
    # CPU dispatch choices are trace constants; numeric parameters and request
    # slots remain live GPU inputs. Private input storage makes cache decisions
    # independent of the allocator and the target graph's output lifetime.
    tensors = (
        hidden,
        batch.input_ids,
        batch.positions,
        speculator.draft_logits,
        *sparse,
    )
    key = (
        id(model),
        id(speculator),
        hidden.dtype,
        bool(np.any(states.temperature.np[idx] != 1.0)),
        bool(np.any(states.top_p.np[idx] != 1.0)),
        rejection.sampler.use_fp64_gumbel,
        tuple((t.dtype, tuple(t.shape), tuple(t.stride())) for t in tensors),
    )
    graphs = getattr(rejection, "_sm70_dflash2_rejection_graphs", None)
    if graphs is None:
        graphs = rejection._sm70_dflash2_rejection_graphs = {}
    if key not in graphs:
        if len(graphs) >= MAX_GRAPHS:
            # Bound graph storage for unexpected shape/contract combinations.
            return None
        if not _all_ranks_support(0 if _capture_headroom(hidden.device) else 801):
            graphs[key] = None
            return None
        candidate = _RejectionGraph(model, speculator, rejection, hidden, batch, sparse)
        graphs[key] = candidate if candidate.capture() else None
    graph = graphs[key]
    if graph is None:
        return None
    rejection.sm70_dflash2_reference_counts = graph.counters
    logger.info_once(
        "DFlash2 compact/reference rejection uses device-side graph dispatch."
    )
    return graph.replay(hidden, batch, sparse)
