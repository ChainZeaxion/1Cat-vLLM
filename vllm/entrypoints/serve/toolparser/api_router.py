# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""免重启热重载端点（2026-10-04，本地增强）。

`POST /v1/internal/reload` —— **一次 HTTP 请求**完成：
  1. **工具解析器**：`importlib.reload` tool_parsers 模块 + 刷新注册表，
     再把新类推送到各 serving 对象（重取 `parser_cls` / `tool_parser`）。
  2. **chat 模板**：强制重读模板源文件，并推送到各 serving 对象（`chat_template`）。

设计取舍（用户要求）：**每请求不做任何额外动作** —— 不在每请求里 stat mtime、不做
版本比较；全部生效由本端点一次完成。改解析器/模板后 **curl 一下本端点即可**，
无需重启引擎（省 ~5 分钟/次）。

⚠️ 仅本地开发/验证用；生产可保持不调用（端点存在但无副作用）。
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()

# 持有 chat 模板 / 解析器缓存的服务对象名（对齐 api_server 的 _hot_template_holders）。
_HOLDERS = (
    "openai_serving_chat",
    "openai_serving_chat_batch",
    "openai_serving_completion",
    "openai_serving_responses",
    "openai_serving_render",
    "openai_serving_tokenization",
)


def _refresh_tool_parser_holders(app) -> list[str]:
    """把重载后的解析器类推送到各 serving 对象（重取 parser_cls / tool_parser）。"""
    from vllm.parser.parser_manager import ParserManager

    done: list[str] = []
    for name in _HOLDERS:
        holder = getattr(app.state, name, None)
        if holder is None:
            continue
        tpn = getattr(holder, "_tool_parser_name", None)
        if tpn is None:
            continue  # 该对象不持有工具解析器
        try:
            holder.tool_parser = ParserManager.get_tool_parser(
                tool_parser_name=tpn,
                enable_auto_tools=getattr(holder, "enable_auto_tools", False),
                model_name=getattr(holder.model_config, "model", None),
            )
            holder.parser_cls = ParserManager.get_parser(
                tool_parser_name=tpn,
                reasoning_parser_name=getattr(holder, "_reasoning_parser_name", None),
                enable_auto_tools=getattr(holder, "enable_auto_tools", False),
                model_name=getattr(holder.model_config, "model", None),
            )
            done.append(name)
        except Exception:
            import logging

            logging.getLogger(__name__).exception(
                "reload: refresh tool parser on %s failed", name
            )
    return done


def _refresh_chat_template_holders(app) -> int:
    """强制重读 chat 模板并推送到各 serving 对象。返回推送到的对象数（0=无模板源）。"""
    from vllm.entrypoints.chat_utils import force_reload_chat_template

    template = force_reload_chat_template()
    if template is None:
        return 0
    n = 0
    for name in _HOLDERS:
        holder = getattr(app.state, name, None)
        if holder is not None and hasattr(holder, "chat_template"):
            holder.chat_template = template
            n += 1
    return n


@router.post("/v1/internal/reload")
async def reload_all(request: Request):
    """免重启热重载：工具解析器 + chat 模板（一次请求全部生效）。"""
    result: dict = {}
    try:
        from vllm.tool_parsers._hotreload import reload_now

        result["tool_parser"] = reload_now(force=True)
        result["tool_parser_holders"] = _refresh_tool_parser_holders(request.app)
    except Exception as e:  # noqa: BLE001
        result["tool_parser_error"] = str(e)
    try:
        result["chat_template_holders"] = _refresh_chat_template_holders(request.app)
    except Exception as e:  # noqa: BLE001
        result["chat_template_error"] = str(e)
    return JSONResponse(result)


def attach_router(app):
    app.include_router(router)
