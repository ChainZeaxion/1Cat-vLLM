# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""工具解析器**免重启热重载**（2026-10-04，本地增强）。

动机：迭代工具解析器（qwen3coder / qwen3xml / utils）时，每次改动都要重启引擎
（加载模型 ~5 分钟）。

设计（**HTTP 端点触发**，不在每请求做重活）：
  - 端点 `POST /v1/internal/reload`（见 `entrypoints/serve/toolparser/api_router.py`）
    调用本模块 `reload_now(force=True)`：`importlib.reload` 相关模块 + 刷新
    `ToolParserManager` 注册表 + **版本号 +1**。
  - 每请求只做 **一次整数比较**（`current_version()` vs 本地缓存）——版本变了才重取
    解析器类。**不做 mtime stat，不在每请求 reload**（避免每请求开销与"每请求都重载"
    的不合理逻辑）。

默认无需 env；端点存在即用。⚠️ 局限：只重载本模块列出的 tool_parsers 模块；已在
飞行的请求用旧类。
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
import threading

logger = logging.getLogger(__name__)

# 重载顺序：先 utils（被解析器 import），再各解析器模块（重新 import 到新 utils 对象）。
_RELOAD_ORDER = (
    "vllm.tool_parsers.utils",
    "vllm.tool_parsers.qwen3coder_tool_parser",
    "vllm.tool_parsers.qwen3xml_tool_parser",
    # [example 区间 2026-10-18] reasoning 文本也做区间转义，纳入热重载：
    # 推理侧改动此前必须重启引擎（~5 分钟）才能验证。
    "vllm.reasoning.qwen3_reasoning_parser",
    # [本地 2026-10-12] 纳入 openai chat_completion serving —— 收尾补发口径、
    # 参数越界/分歧守卫等修复都落在该模块**类的方法**上。reload 后由
    # api_router._refresh_serving_classes 把新方法覆盖到运行实例所属的旧类，
    # 实例无需重建（它们长生命、持有 tokenizer / engine client）。
    "vllm.entrypoints.openai.chat_completion.serving",
)

_VERSION = 0
_MTIMES: dict[str, float] = {}
_LOCK = threading.Lock()


def current_version() -> int:
    """当前重载版本号。每请求比较它即可（几乎零成本）。"""
    return _VERSION


def reload_now(force: bool = True) -> dict:
    """重载工具解析器模块并刷新注册表，版本号 +1。

    Args:
        force: True 时无条件重载（端点触发用）；False 时仅当文件 mtime 变化才重载。
    Returns:
        {"reloaded": [模块名...], "version": 新版本号}
    """
    global _VERSION
    reloaded: list[str] = []
    with _LOCK:
        for name in _RELOAD_ORDER:
            mod = sys.modules.get(name)
            if mod is None:
                continue
            path = getattr(mod, "__file__", None)
            if not path or not os.path.exists(path):
                continue
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                mtime = None
            prev = _MTIMES.get(name)
            if mtime is not None:
                _MTIMES[name] = mtime
            if not force and mtime is not None and mtime == prev:
                continue
            try:
                importlib.reload(mod)
                reloaded.append(name)
            except Exception:  # 保留旧模块，勿因语法错误打挂服务
                logger.exception("tool parser hot-reload failed for %s", name)
        if reloaded:
            # 刷新注册表：lazy_parsers 的 (module_path, class_name) → 新类对象
            try:
                from vllm.tool_parsers import ToolParserManager

                for nm, (mp, cn) in list(ToolParserManager.lazy_parsers.items()):
                    m = sys.modules.get(mp)
                    cls = getattr(m, cn, None) if m is not None else None
                    if cls is not None:
                        ToolParserManager.tool_parsers[nm] = cls
            except Exception:
                logger.exception("tool parser hot-reload registry refresh failed")
            _VERSION += 1
            logger.warning(
                "tool parser hot-reloaded (v%d): %s", _VERSION, reloaded
            )
    return {"reloaded": reloaded, "version": _VERSION}
