# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-decode-step 分段耗时剖面（**诊断工具，默认关闭**）。

用途：把一次 decode round（execute_model + sample_tokens）拆成环节级耗时，
同时给出 CPU 侧（提交/准备）与 GPU 侧（真实计算）两条时间线。

本模块**不改变任何推理行为**：未启用时所有入口函数第一行即返回，
`gpu()` 上下文管理器退化为 nullcontext。

启用：
    VLLM_DECODE_STEP_PROFILE=1

可选：
    VLLM_DECODE_STEP_PROFILE_EVERY=<n>    每 n 步采样并打印一次（默认 20）
    VLLM_DECODE_STEP_PROFILE_WARMUP=<n>   前 n 步不采样（默认 5）
    VLLM_DECODE_STEP_PROFILE_PHASE=decode 只采样纯 decode 步（默认 decode；
                                          取值 all/decode/prefill）
    VLLM_DECODE_STEP_PROFILE_RANK=<n>     只在该 rank 打印（默认全部打印）

计时口径：
  * CPU 段用 ``time.perf_counter()``：测量"从上一个标记到本标记"的墙钟时间。
  * GPU 段用 ``torch.cuda.Event``：测量该段在 **main stream** 上的真实设备时间。
    Event 的 elapsed_time 只有在两端都完成后才有效——本 runner 在
    ``AsyncOutput.get_output()``（copy_stream.wait_stream(main_stream) + synchronize）
    之后保证了这一点，故无需额外 synchronize，不引入测量扰动。
"""

from __future__ import annotations

import contextlib
import os
import time

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _rank_tag() -> str:
    try:
        from vllm.distributed.parallel_state import (
            get_tensor_model_parallel_rank,
        )

        tp = get_tensor_model_parallel_rank()
    except Exception:
        tp = -1
    try:
        return f"tp{tp}/dev{torch.cuda.current_device()}"
    except Exception:
        return f"tp{tp}"


class DecodeStepProfiler:
    """按 step 累积 CPU/GPU 分段耗时，每 ``every`` 步打印一行。"""

    def __init__(self, tag: str = "decode") -> None:
        self.tag = tag
        self.enabled = os.getenv("VLLM_DECODE_STEP_PROFILE", "0") == "1"
        self.every = max(1, _env_int("VLLM_DECODE_STEP_PROFILE_EVERY", 20))
        self.warmup = max(0, _env_int("VLLM_DECODE_STEP_PROFILE_WARMUP", 5))
        self.phase = os.getenv("VLLM_DECODE_STEP_PROFILE_PHASE", "decode").lower()
        _rank = _env_int("VLLM_DECODE_STEP_PROFILE_RANK", -1)
        self.rank_tag = _rank_tag()
        self.silent = _rank >= 0 and not self.rank_tag.startswith(f"tp{_rank}/")

        self.step = 0
        self.sampling = False
        self.is_decode = False
        self.num_reqs = 0
        self.num_toks = 0
        self._t0 = 0.0
        self._tprev = 0.0
        self._cpu: dict[str, float] = {}
        self._gpu_ev: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]] = {}
        # 跨 execute_model / sample_tokens 两个 RPC 的累积
        self._carry_cpu: dict[str, float] = {}
        self._carry_gpu: dict[str, tuple] = {}
        self._step_started = False
        if self.enabled and not self.silent:
            logger.info(
                "DECODE_STEP_PROFILE enabled tag=%s rank=%s every=%d warmup=%d "
                "phase=%s",
                self.tag,
                self.rank_tag,
                self.every,
                self.warmup,
                self.phase,
            )

    # ---------------------------------------------------------------- step 生命周期

    def _want(self) -> bool:
        return (
            self.phase == "all"
            or (self.phase == "decode" and self.is_decode)
            or (self.phase == "prefill" and not self.is_decode)
        )

    def begin_step(self, num_reqs: int = 0, num_toks: int = 0) -> None:
        """在 execute_model 入口调用。

        阶段判据在入口只能粗估：投机解码下一个 decode 步的调度 token 数是
        ``num_reqs × (1 + 草稿数)``，远小于 chunked prefill 的 4096 budget，
        故先用一个宽比值初判，随后由 :meth:`set_decode` 用 ``input_batch``
        的权威 ``is_prefilling`` 修正（宽松初判是为了不丢掉前几段计时）。
        """
        if not self.enabled:
            return
        self.step += 1
        self.num_reqs = num_reqs
        self.num_toks = num_toks
        self.is_decode = num_reqs > 0 and num_toks <= num_reqs * 16
        self.sampling = (
            self._want()
            and self.step > self.warmup
            and self.step % self.every == 0
        )
        if not self.sampling:
            return
        now = time.perf_counter()
        self._t0 = now
        self._tprev = now
        self._cpu = {}
        self._gpu_ev = {}
        self._step_started = True

    def set_decode(self, is_decode: bool) -> None:
        """用 ``input_batch.is_prefilling`` 的权威结果修正阶段判定。"""
        if not self.sampling:
            return
        self.is_decode = is_decode
        if not self._want():
            self.sampling = False

    def begin_sample(self) -> None:
        """在 sample_tokens 入口调用（跨 RPC 重置连续计时的起点）。"""
        if not self.sampling:
            return
        self._tprev = time.perf_counter()

    def end_step(self, extra: str = "") -> None:
        """在 sample_tokens 返回前调用（此时 GPU event 已全部完成）。"""
        if not self.sampling:
            return
        now = time.perf_counter()
        self._cpu["tail"] = self._cpu.get("tail", 0.0) + (now - self._tprev) * 1000.0

        # 合并 execute 阶段结转的分段
        cpu = dict(self._carry_cpu)
        for k, v in self._cpu.items():
            cpu[k] = cpu.get(k, 0.0) + v
        gpu_ev = dict(self._carry_gpu)
        gpu_ev.update(self._gpu_ev)

        gpu: dict[str, float] = {}
        for name, (s, e) in gpu_ev.items():
            try:
                # 草稿 propose 的 event 晚于 AsyncOutput 的 copy_event，
                # 不能被 get_output() 的同步覆盖 ⇒ 逐个同步（已完成时 no-op）。
                e.synchronize()
                gpu[name] = s.elapsed_time(e)
            except Exception:
                gpu[name] = float("nan")

        cpu_total = sum(cpu.values())
        gpu_total = sum(v for v in gpu.values() if v == v)
        parts = " ".join(
            f"{k}={v:.3f}" for k, v in sorted(cpu.items(), key=lambda x: -x[1])
        )
        gparts = " ".join(
            f"{k}={v:.3f}" for k, v in sorted(gpu.items(), key=lambda x: -x[1])
        )
        if not self.silent:
            logger.info(
                "DECODE_STEP_PROFILE[%s] step=%d rank=%s decode=%s "
                "reqs=%d toks=%d "
                "cpu_total_ms=%.3f gpu_total_ms=%.3f | CPU %s | GPU %s%s",
                self.tag,
                self.step,
                self.rank_tag,
                self.is_decode,
                self.num_reqs,
                self.num_toks,
                cpu_total,
                gpu_total,
                parts,
                gparts,
                f" | {extra}" if extra else "",
            )

        self.sampling = False
        self._step_started = False
        self._carry_cpu = {}
        self._carry_gpu = {}

    def carry_over(self) -> None:
        """在 execute_model 返回前调用：把本段累积挪到 carry，供 end_step 合并。"""
        if not self.sampling:
            return
        self._carry_cpu = dict(self._cpu)
        self._carry_gpu = dict(self._gpu_ev)
        self._cpu = {}
        self._gpu_ev = {}

    # ------------------------------------------------------------------- CPU 标记

    def cpu(self, name: str) -> None:
        """记录"上一个标记 → 现在"这段墙钟时间，归到 name。"""
        if not self.sampling:
            return
        now = time.perf_counter()
        self._cpu[name] = self._cpu.get(name, 0.0) + (now - self._tprev) * 1000.0
        self._tprev = now

    # ------------------------------------------------------------------- GPU 标记

    def gpu_begin(self, name: str) -> None:
        """GPU 计时起点（用于需要包住 if/else 的段落）。"""
        if not self.sampling:
            return
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        self._gpu_ev[name] = (start, start)

    def gpu_end(self, name: str) -> None:
        if not self.sampling:
            return
        entry = self._gpu_ev.get(name)
        if entry is None:
            return
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self._gpu_ev[name] = (entry[0], end)

    @contextlib.contextmanager
    def gpu(self, name: str):
        """测量该段在 main stream 上的设备时间（同名多段合并为总跨度）。"""
        if not self.sampling:
            yield
            return
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            s_prev, _ = self._gpu_ev.get(name, (None, None))
            self._gpu_ev[name] = (s_prev or start, end)


# 每个 model runner 实例一份（由 GPUModelRunner.__init__ 创建）
_profiler: DecodeStepProfiler | None = None


def get_profiler(tag: str = "decode") -> DecodeStepProfiler:
    global _profiler
    if _profiler is None:
        _profiler = DecodeStepProfiler(tag)
    return _profiler
