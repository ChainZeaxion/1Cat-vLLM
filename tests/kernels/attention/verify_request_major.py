# -*- coding: utf-8 -*-
"""request-major batched grouped verify —— 离线验证（不需要 pytest）。

验证四件事：
  1. ABI 探测函数存在且返回 1
  2. **batch>1 的 batched 调用 与 per-request 循环 逐元素一致**（核心）
  3. batch>1 结果对拍 fp32 参考实现
  4. 旧的 B1 路径（block_table 1 行）行为不变

用法:
  export PYTHONPATH=<源码树>/flash-attention-v100
  python tests/kernels/attention/verify_request_major.py
"""
from __future__ import annotations

import math
import os
import sys

import torch

REPO = "/home/zeaxion/myproject/1cat-vllm-v130/pr-v130-build"
# The extension under test must be the one the engine actually loads: the
# patched copy under site-packages/vllm/.  Note the wrapper package is
# `vllm.flash_attn_v100`, so the *parent* of the `vllm` dir goes on the path
# (the shadow `site-packages/flash_attn_v100/` is stale and must not win).
SITE_PACKAGES = (
    "/home/zeaxion/miniconda3/envs/zxvllm120/lib/python3.12/site-packages"
)
# Mirror the engine's import layout: vLLM puts .../site-packages/vllm itself on
# sys.path, so `flash_attn_v100` resolves to vllm/flash_attn_v100/ and the stale
# shadow copy at site-packages/flash_attn_v100/ never wins.  vllm/ must come
# before site-packages for the same reason.
sys.path.insert(0, f"{REPO}/tests/kernels/attention")
sys.path.insert(0, SITE_PACKAGES)
sys.path.insert(0, f"{SITE_PACKAGES}/vllm")

import flash_attn_v100  # noqa: E402,F401
from flash_attn_v100 import flash_attn_grouped_verify_paged  # noqa: E402
from test_sm70_flash_v100_grouped_verify import (  # noqa: E402
    _make_case,
    _reference,
)

print(f"[info] 使用的扩展: {flash_attn_v100.__file__}")

FAILURES: list[str] = []
K_SCALE = 1.0
V_SCALE = 1.0
PAGE_SIZE = 32


def check(name: str, cond: bool, detail: str = "") -> None:
    tag = "PASS" if cond else "FAIL"
    print(f"  [{tag}] {name}" + (f"  -- {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def make_batched_case(batch: int, query_len: int, prefix_len: int):
    """batch 个请求共享一个 KV pool，但各有独立的 block_table / seq_lens。

    请求 i 的序列长度略有不同，以覆盖 `seq_lens` 非均匀的情形。
    """
    seq_lens_list = [prefix_len + query_len + i for i in range(batch)]
    max_len = max(seq_lens_list)
    logical_max = math.ceil(max_len / PAGE_SIZE)
    physical_pages = logical_max * batch + 4
    source_shape = (physical_pages, PAGE_SIZE, 1, 256)
    key_source = torch.randn(source_shape, dtype=torch.float16, device="cuda").mul_(0.25)
    value_source = torch.randn_like(key_source).mul_(0.25)
    key_cache = key_source.to(torch.float8_e5m2).view(torch.uint8)
    value_cache = value_source.to(torch.float8_e5m2).view(torch.uint8)

    # 每个请求一段互不重叠的物理页（确定性切片，方便复现）
    max_blocks = math.ceil(max_len / PAGE_SIZE)
    perm = torch.randperm(physical_pages, dtype=torch.int32, device="cuda")
    rows = []
    for i in range(batch):
        lo = i * max_blocks
        rows.append(perm[lo : lo + max_blocks])
    block_table = torch.stack(rows, dim=0).to(torch.int32)
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device="cuda")
    query = torch.randn(
        (batch * query_len, 6, 256), dtype=torch.float16, device="cuda"
    ).mul_(0.25)
    return query, key_cache, value_cache, block_table, seq_lens, seq_lens_list


def batched_call(query, k, v, bt, sl, *, out=None):
    return flash_attn_grouped_verify_paged(
        query, k, v, block_table=bt, seq_lens=sl,
        softmax_scale=None, out=out, kv_cache_dtype="fp8_e5m2",
        k_scale=K_SCALE, v_scale=V_SCALE, one_pass=True,
    )


def per_request_call(query, k, v, bt, sl, q_per):
    """模拟 120 原有的 per-request 循环。"""
    n = int(bt.shape[0])
    out = torch.empty_like(query)
    for req in range(n):
        q0 = req * q_per
        q1 = q0 + q_per
        flash_attn_grouped_verify_paged(
            query[q0:q1], k, v,
            block_table=bt[req : req + 1], seq_lens=sl[req : req + 1],
            softmax_scale=None, out=out[q0:q1], kv_cache_dtype="fp8_e5m2",
            k_scale=K_SCALE, v_scale=V_SCALE, one_pass=True,
        )
    return out


print("=" * 70)
print("request-major batched grouped verify —— 离线验证")
print("=" * 70)

# ── 1. ABI 探测 ──────────────────────────────────────────────────────────
print("\n[1] ABI 探测函数")
try:
    from flash_attn_v100 import flash_attn_grouped_verify_request_major_abi_version

    abi = int(flash_attn_grouped_verify_request_major_abi_version())
    check("ABI 函数存在且 == 1", abi == 1, f"got {abi}")
except ImportError as exc:
    check("ABI 函数存在", False, str(exc))

# ── 2 & 3. batch>1：batched vs per-request vs fp32 参考 ───────────────────
print("\n[2] batch>1：batched 与 per-request 数值一致")
# The two paths run the same math but accumulate the split reduction in a
# different order, and the KV is fp8, so exact equality is not the right bar.
# TOL matches the fp32-reference envelope measured in [3] (~6.6e-3).
TOL = 2e-2
for batch, q_per, prefix in [(2, 6, 512), (4, 6, 256), (8, 6, 128), (3, 4, 300)]:
    label = f"batch={batch} q={q_per} prefix={prefix}"
    try:
        query, k, v, bt, sl, seq_list = make_batched_case(batch, q_per, prefix)
        ref_out = per_request_call(query, k, v, bt, sl, q_per)
        batched_out = batched_call(query, k, v, bt, sl)
        max_diff = (ref_out.float() - batched_out.float()).abs().max().item()
        check(f"{label} batched ~= per-request", max_diff < TOL,
              f"max_abs_diff={max_diff:.3e} (tol={TOL})")
    except Exception as exc:  # noqa: BLE001
        check(f"{label} batched ~= per-request", False, f"{type(exc).__name__}: {exc}")

print("\n[3] batch>1：对拍 fp32 参考")
try:
    batch, q_per, prefix = 4, 6, 256
    query, k, v, bt, sl, seq_list = make_batched_case(batch, q_per, prefix)
    batched_out = batched_call(query, k, v, bt, sl)
    worst = 0.0
    for req in range(batch):
        q0 = req * q_per
        ref = _reference(
            query[q0 : q0 + q_per], k, v, bt[req : req + 1], int(seq_list[req]),
            k_scale=K_SCALE, v_scale=V_SCALE,
        )
        worst = max(worst, (ref.float() - batched_out[q0 : q0 + q_per].float())
                    .abs().max().item())
    check("batch=4 对拍 fp32 参考", worst < 5e-2, f"max_abs_diff={worst:.3e}")
except Exception as exc:  # noqa: BLE001
    check("batch=4 对拍 fp32 参考", False, f"{type(exc).__name__}: {exc}")

# ── 4. B1 路径不回归 ─────────────────────────────────────────────────────
print("\n[4] B1（单请求）路径不回归")
for q_len in (8, 16):
    label = f"B1 q={q_len}"
    try:
        query, k, v, bt, sl = _make_case(
            page_size=PAGE_SIZE, query_len=q_len, prefix_len=512
        )
        got = batched_call(query, k, v, bt, sl)
        ref = _reference(query, k, v, bt, int(sl[0]), k_scale=K_SCALE, v_scale=V_SCALE)
        d = (ref.float() - got.float()).abs().max().item()
        check(f"{label} 对拍 fp32 参考", d < 5e-2, f"max_abs_diff={d:.3e}")
    except Exception as exc:  # noqa: BLE001
        check(f"{label} 对拍 fp32 参考", False, f"{type(exc).__name__}: {exc}")

# ── 5. 反向用例：形状契约 ────────────────────────────────────────────────
print("\n[5] 形状契约（应被拒绝）")
try:
    query, k, v, bt, sl, _ = make_batched_case(4, 6, 256)
    # q 行数不能被 batch 整除
    bad_q = query[:-1]
    try:
        batched_call(bad_q, k, v, bt, sl)
        check("q 行数不整除 batch 应报错", False, "未报错")
    except (RuntimeError, AssertionError) as exc:
        check("q 行数不整除 batch 应报错", True, type(exc).__name__)
    # seq_lens 数量与 batch 不符
    try:
        batched_call(query, k, v, bt, sl[:2])
        check("seq_lens 数量不符应报错", False, "未报错")
    except (RuntimeError, AssertionError) as exc:
        check("seq_lens 数量不符应报错", True, type(exc).__name__)
except Exception as exc:  # noqa: BLE001
    check("形状契约用例", False, f"{type(exc).__name__}: {exc}")

print("\n" + "=" * 70)
if FAILURES:
    print(f"❌ {len(FAILURES)} 项失败：")
    for f in FAILURES:
        print(f"   - {f}")
    sys.exit(1)
print("✅ 全部通过")
