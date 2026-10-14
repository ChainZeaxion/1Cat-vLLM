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


def _refresh_serving_classes(app) -> list[str]:
    """把 reload 后的 OpenAIServingChat 方法覆盖到运行实例所属的**旧类**上。

    [本地 2026-10-12] serving 实例是长生命（`app.state`），`importlib.reload`
    只会产生**新类**，实例仍绑旧类 ⇒ 改了类方法却不生效。重建实例不现实
    （持有 tokenizer / engine client / renderer 等），故直接把新类的属性覆盖
    到旧类：实例的方法查找走类属性，覆盖后立即生效，实例状态（`__dict__`）
    分毫不动。只取 `new_cls` **自身**的属性，不碰继承来的基类行为。
    """
    try:
        import vllm.entrypoints.openai.chat_completion.serving as S
    except Exception:  # noqa: BLE001
        return []
    new_cls = getattr(S, "OpenAIServingChat", None)
    if new_cls is None:
        return []
    done: list[str] = []
    for name in _HOLDERS:
        holder = getattr(app.state, name, None)
        if holder is None:
            continue
        old_cls = type(holder)
        if old_cls is new_cls:
            continue
        n = 0
        for key, val in vars(new_cls).items():
            if key.startswith("__"):
                continue
            try:
                setattr(old_cls, key, val)
                n += 1
            except Exception:  # noqa: BLE001
                pass
        if n:
            done.append(f"{name}:{old_cls.__name__}:{n}")
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
    """免重启热重载：工具解析器 + serving 类 + chat 模板（一次请求全部生效）。

    可选 body（JSON，可省略）：
      {"partial_every_n_tokens": 1}  # 1 = 逐 token 吐（当前默认，全放开）；3 = 每 3 token

    节流值在 **reload 之后**才写回：reload 会重跑 qwen3coder 模块代码、把
    `_PARTIAL_EVERY_N_TOKENS` 重置为默认/env 值，故顺序不能反。
    """
    body: dict = {}
    try:
        parsed = await request.json()
        if isinstance(parsed, dict):
            body = parsed
    except Exception:  # noqa: BLE001
        body = {}
    result: dict = {"requested": body} if body else {}
    # reload 会把 qwen3coder 的模块级 `_PARTIAL_EVERY_N_TOKENS` 重置为默认/env 值，
    # 故先记下调用前的值：body 没带该键时用它写回——否则「查询当前值」这个动作
    # 本身就会把节流设置改掉（2026-10-13 实测踩坑：查一次就从 0 变回 0.25）。
    _prev_every_n = None
    try:
        import vllm.tool_parsers.qwen3coder_tool_parser as _Q0

        _prev_every_n = getattr(_Q0, "_PARTIAL_EVERY_N_TOKENS", None)
    except Exception:  # noqa: BLE001
        pass
    try:
        from vllm.tool_parsers._hotreload import reload_now

        result["tool_parser"] = reload_now(force=True)
        result["tool_parser_holders"] = _refresh_tool_parser_holders(request.app)
        # 解析器 reload 完再刷新 serving 类（收尾补发/守卫等修复在类方法上）。
        result["serving_classes"] = _refresh_serving_classes(request.app)
    except Exception as e:  # noqa: BLE001
        result["tool_parser_error"] = str(e)
    try:
        import vllm.tool_parsers.qwen3coder_tool_parser as _Q

        # 计数制节流：每 N 个 token 发一次 partial（1 = 逐 token，3 = 每 3 token）。
        if "partial_every_n_tokens" in body:
            _Q._PARTIAL_EVERY_N_TOKENS = int(body["partial_every_n_tokens"])
        elif _prev_every_n is not None:
            _Q._PARTIAL_EVERY_N_TOKENS = _prev_every_n  # 纯查询：保持原值
        result["partial_every_n_tokens"] = getattr(
            _Q, "_PARTIAL_EVERY_N_TOKENS", None
        )
    except Exception as e:  # noqa: BLE001
        result["partial_every_n_tokens_error"] = str(e)
    try:
        result["chat_template_holders"] = _refresh_chat_template_holders(request.app)
    except Exception as e:  # noqa: BLE001
        result["chat_template_error"] = str(e)
    return JSONResponse(result)


def attach_router(app):
    app.include_router(router)
