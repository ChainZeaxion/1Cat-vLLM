# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import io
import json
import os
import re
import time
from collections.abc import AsyncGenerator, AsyncIterator, Iterable
from collections.abc import Sequence as GenericSequence
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pybase64 as base64
from fastapi import Request

from vllm.engine.protocol import EngineClient
from vllm.entrypoints.chat_utils import (
    ChatTemplateContentFormatOption,
    ConversationMessage,
    get_history_tool_calls_cnt,
    get_tool_call_id_type,
    make_tool_call_id,
)
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionLogProb,
    ChatCompletionLogProbs,
    ChatCompletionLogProbsContent,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
)
from vllm.entrypoints.openai.chat_completion.stream_harmony import (
    TokenState,
    extract_harmony_streaming_delta,
)
from vllm.entrypoints.openai.engine.protocol import (
    CompletionTokenUsageInfo,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ErrorResponse,
    FunctionCall,
    PromptTokenUsageInfo,
    RequestTimings,
    RequestResponseMetadata,
    ToolCall,
    UsageInfo,
)
from vllm.entrypoints.openai.engine.serving import (
    GenerationError,
    OpenAIServing,
    clamp_prompt_logprobs,
)
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.entrypoints.openai.parser.harmony_utils import (
    get_stop_tokens_for_assistant_actions,
    get_streamable_parser_for_assistant,
    parse_chat_output,
)
from vllm.entrypoints.openai.utils import maybe_filter_parallel_tool_calls
from vllm.entrypoints.utils import get_max_tokens, should_include_usage
from vllm.inputs import EngineInput
from vllm.logger import init_logger
from vllm.logprobs import Logprob
from vllm.outputs import CompletionOutput, RequestOutput, STREAM_KEEPALIVE
from vllm.parser import ParserManager
from vllm.parser.abstract_parser import Parser
from vllm.reasoning import ReasoningParser
from vllm.renderers import ChatParams
from vllm.sampling_params import BeamSearchParams, SamplingParams
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.utils import (
    escape_tool_like_tags,
    escape_tool_like_tags_outside_examples,
)


# [2026-10-18] 伪调用文本清洗：模型偶尔把**打算并行发起的工具**输出成纯文本
# 清单（`[Called: Bash]` 每行一条），该条 assistant 消息**没有** tool_calls。
# 这段文本回传后模型会模仿该格式 → 后续轮次全部退化成文本，且 `/compact`
# 摘要会把它抄进新上下文导致无法自愈（实测连续 8 轮失效）。
# 真实调用不会以这种纯文本形式出现，故只对「无 tool_calls 的 assistant 消息」
# 删除这些行是安全的。
_PSEUDO_CALL_LINE_RE = re.compile(r"^[ \t]*\[Called:[^\]\n]*\][ \t]*$", re.M)


def sanitize_pseudo_tool_calls(messages: list) -> list:
    """删除 assistant 历史中无 tool_calls 消息里的 `[Called: X]` 伪调用行。

    返回**新列表**（未变化时原样返回入参）。整条消息只有伪调用时丢弃该消息。
    """
    changed = False
    out: list = []
    for m in messages:
        if (
            isinstance(m, dict)
            and m.get("role") == "assistant"
            and not m.get("tool_calls")
            and not m.get("function_call")
        ):
            content = m.get("content")
            if isinstance(content, str) and "[Called:" in content:
                cleaned = _PSEUDO_CALL_LINE_RE.sub("", content)
                cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip("\n")
                if cleaned != content:
                    changed = True
                    if not cleaned:
                        continue
                    m = {**m, "content": cleaned}
        out.append(m)
    return out if changed else messages

from vllm.utils.collection_utils import as_list
from vllm.utils.mistral import is_mistral_tokenizer, is_mistral_tool_parser

if TYPE_CHECKING:
    from vllm.entrypoints.serve.render.serving import OpenAIServingRender

logger = init_logger(__name__)


def _merge_double_think_close(reasoning, content, reasoning_parser):
    """[2026-10-20] 把**第二个 `</think>`** 之前的分析段落并回 reasoning。

    成因：模板在开思考时预填「好的，」（`思考首词必须中文` 钩子），模型有时
    把它当成「已想过一句」⇒ 立刻闭合思考块，在块外继续写完整分析，最后再补
    一个 `</think>` 才发调用。实测 T=1 发生率 4/40 = 10%，集中在「先分析再执行」
    的提示词上。

    修前危害：第二次 `</think>` 之前的整段分析被当作 **content** 下发给用户，
    末尾还带着字面的 `</think>` 标签，而它本质是思考内容。

    判据只用结构不变量（不猜语义）：`content` 里若**再次出现** `</think>`，
    则第二次闭合之前的内容全部是思考 ⇒ 并回 reasoning 并丢弃多余闭合标签；
    第二次闭合之后若有内容，那才是真 content。
    """
    if content is None:
        return reasoning, content
    end_token = getattr(reasoning_parser, "end_token", None) or "</think>"
    if end_token not in content:
        return reasoning, content
    more_reasoning, _, real_content = content.partition(end_token)
    merged = (reasoning or "") + more_reasoning
    return merged or None, real_content or None


_THINK_END_TOKEN = "</think>"


def _fix_stream_double_think(delta_message):
    """[2026-10-20] 流式帧里的**第二个 `</think>`**：并回 reasoning，丢弃标签。

    成因见 `_merge_double_think_close`（模板预填「好的，」使模型提前闭合）。
    流式下第二次闭合会落在 content 帧里、以字面形式下发给用户，
    实测并发 16 路中 7 路出现，且多在该路倒数几帧。

    ⚠️ 只处理**本帧内**出现 `</think>` 的情况：跨帧的标签残片由
    reasoning parser 的暂存逻辑负责，本函数不持有跨帧状态。
    """
    if delta_message is None:
        return delta_message
    content = getattr(delta_message, "content", None)
    if not content or _THINK_END_TOKEN not in content:
        return delta_message
    head, _, tail = content.partition(_THINK_END_TOKEN)
    prev_reasoning = getattr(delta_message, "reasoning", None) or ""
    delta_message.reasoning = (prev_reasoning + head) or None
    delta_message.content = tail or None
    return delta_message


def _sx_skip_tool_parser_without_tools() -> bool:
    """SX_OPT_SKIP_TOOL_PARSER_WITHOUT_TOOLS (default "1"; "0" = old behaviour).

    When on, a streamed chat request without tools, or with tool_choice
    "none", does not run the configured tool parser over every delta. Such a
    request cannot receive tool calls, so its content passes through
    unchanged (design_5 cpu-8). The Mistral grammar path and the harmony
    path are never affected.
    """
    return os.environ.get("SX_OPT_SKIP_TOOL_PARSER_WITHOUT_TOOLS", "1") != "0"


def _count_reasoning_tokens(
    reasoning_parser: ReasoningParser | None,
    token_ids_per_choice: Iterable[GenericSequence[int]] | None,
) -> int | None:
    """Reasoning tokens among the generated ids, summed over choices.

    Returns None when there is nothing to measure — no reasoning parser, or no
    ids were retained for any choice. The caller then omits the field instead of
    reporting zero, because a zero and "we did not look" mean different things to
    whoever is reading the number.
    """
    if reasoning_parser is None or token_ids_per_choice is None:
        return None
    total = 0
    measured = False
    for token_ids in token_ids_per_choice:
        if not token_ids:
            continue
        total += reasoning_parser.count_reasoning_tokens(token_ids)
        measured = True
    return total if measured else None


def _usage_timings(metrics: object) -> RequestTimings | None:
    """[1Cat P4] Engine per-request prefill/decode breakdown for the response.

    ``metrics`` is the ``RequestStateStats`` that the engine attaches to every
    finished ``RequestOutput`` (v1/engine/output_processor.py). We only re-express
    its monotonic timestamps as durations here -- no new measurement happens in
    the serving layer. Returns None when stats were not collected (e.g. the
    engine ran with ``log_stats=False``); the field is then omitted from the
    response rather than reported as zero.
    """
    if metrics is None:
        return None
    queued = getattr(metrics, "queued_ts", 0.0)
    scheduled = getattr(metrics, "scheduled_ts", 0.0)
    first = getattr(metrics, "first_token_ts", 0.0)
    last = getattr(metrics, "last_token_ts", 0.0)
    if not (queued and scheduled and first and last):
        return None
    return RequestTimings(
        queue_s=scheduled - queued,
        prefill_s=first - scheduled,
        decode_s=last - first,
        inference_s=last - scheduled,
        # Wider than queue_s + prefill_s by the frontend->core hop.
        ttft_s=getattr(metrics, "first_token_latency", None) or None,
    )


class OpenAIServingChat(OpenAIServing):
    def __init__(
        self,
        engine_client: EngineClient,
        models: OpenAIServingModels,
        response_role: str,
        *,
        openai_serving_render: "OpenAIServingRender",
        request_logger: RequestLogger | None,
        chat_template: str | None,
        chat_template_content_format: ChatTemplateContentFormatOption,
        trust_request_chat_template: bool = False,
        return_tokens_as_token_ids: bool = False,
        reasoning_parser: str = "",
        enable_auto_tools: bool = False,
        exclude_tools_when_tool_choice_none: bool = False,
        tool_parser: str | None = None,
        enable_prompt_tokens_details: bool = False,
        enable_force_include_usage: bool = False,
        enable_log_outputs: bool = False,
        enable_log_deltas: bool = True,
        default_chat_template_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            engine_client=engine_client,
            models=models,
            request_logger=request_logger,
            return_tokens_as_token_ids=return_tokens_as_token_ids,
        )

        self.openai_serving_render = openai_serving_render
        self.response_role = response_role
        self.chat_template = chat_template
        self.chat_template_content_format: Final = chat_template_content_format
        self.trust_request_chat_template = trust_request_chat_template
        self.default_chat_template_kwargs = default_chat_template_kwargs or {}
        self.enable_log_outputs = enable_log_outputs
        self.enable_log_deltas = enable_log_deltas

        # set up reasoning parser
        self.reasoning_parser_cls = ParserManager.get_reasoning_parser(
            reasoning_parser_name=reasoning_parser
        )
        # set up tool use
        self.enable_auto_tools: bool = enable_auto_tools
        self.tool_parser = ParserManager.get_tool_parser(
            tool_parser_name=tool_parser,
            enable_auto_tools=enable_auto_tools,
            model_name=self.model_config.model,
        )
        self.parser_cls = ParserManager.get_parser(
            tool_parser_name=tool_parser,
            reasoning_parser_name=reasoning_parser,
            enable_auto_tools=enable_auto_tools,
            model_name=self.model_config.model,
        )
        # [本地 2026-10-04] 工具解析器免重启热重载：保存名字，供 `POST /v1/internal/reload` 刷新解析器类。
        self._tool_parser_name = tool_parser
        self._reasoning_parser_name = reasoning_parser
        if (
            is_mistral_tool_parser(self.tool_parser)
            and self.reasoning_parser_cls is not None
        ):
            from vllm.tool_parsers.mistral_tool_parser import MistralToolParser

            MistralToolParser.model_can_reason = True

        self.exclude_tools_when_tool_choice_none = exclude_tools_when_tool_choice_none

        self.enable_prompt_tokens_details = enable_prompt_tokens_details
        self.enable_force_include_usage = enable_force_include_usage
        self.default_sampling_params = self.model_config.get_diff_sampling_param()
        mc = self.model_config
        self.override_max_tokens = (
            self.default_sampling_params.get("max_tokens")
            if mc.generation_config not in ("auto", "vllm")
            else getattr(mc, "override_generation_config", {}).get("max_new_tokens")
        )
        self.use_harmony = self.model_config.hf_config.model_type == "gpt_oss"
        if self.use_harmony:
            if "stop_token_ids" not in self.default_sampling_params:
                self.default_sampling_params["stop_token_ids"] = []
            self.default_sampling_params["stop_token_ids"].extend(
                get_stop_tokens_for_assistant_actions()
            )

        self.tool_call_id_type = get_tool_call_id_type(self.model_config)

        # NOTE(woosuk): While OpenAI's chat completion API supports browsing
        # for some models, currently vLLM doesn't support it. Please use the
        # Responses API instead.
        self.supports_browsing = False
        self.browser_tool = None
        # NOTE(woosuk): Chat completion API does not support code interpreter.
        # Please use the Responses API instead.
        self.supports_code_interpreter = False
        self.python_tool = None

    def warmup(self) -> None:
        self.renderer.warmup(
            ChatParams(
                chat_template=self.chat_template,
                chat_template_content_format=self.chat_template_content_format,
                chat_template_kwargs=self.default_chat_template_kwargs,
            )
        )

    def _effective_chat_template_kwargs(
        self, request: ChatCompletionRequest
    ) -> dict[str, Any]:
        return (
            request.build_chat_params(
                self.chat_template,
                self.chat_template_content_format,
            )
            .with_defaults(self.default_chat_template_kwargs)
            .chat_template_kwargs
        )

    async def render_chat_request(
        self,
        request: ChatCompletionRequest,
    ) -> tuple[list[ConversationMessage], list[EngineInput]] | ErrorResponse:
        """
        Validate the model and preprocess a chat completion request.

        Delegates preprocessing logic to OpenAIServingRender, adding the
        engine-aware checks (LoRA model validation, engine health).

        Returns:
            A tuple of (conversation, engine_inputs) on success,
            or an ErrorResponse on failure.
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            logger.error("Error with model %s", error_check_ret)
            return error_check_ret

        # If the engine is dead, raise the engine's DEAD_ERROR.
        # This is required for the streaming case, where we return a
        # success status before we actually start generating text :).
        if self.engine_client.errored:
            raise self.engine_client.dead_error

        # [2026-10-18] 回传前清洗伪调用清单（见 sanitize_pseudo_tool_calls）
        _clean = sanitize_pseudo_tool_calls(request.messages)
        if _clean is not request.messages:
            request = request.model_copy(update={"messages": _clean})

        # [2026-10-20] `tool_choice="none"` ⇒ 渲染时**不把 tools 交给模板**。
        #
        # 动机（实测）：模型完全看不到 `tool_choice`——`auto` 与 `none` 下生成的
        # token_ids 前 12 位**逐位相同**（同为 `[760, 1156, 6587, …]`，总长同为 50）。
        # 模型只会照模板教的格式调用；`none` 目前全靠引擎**事后**把调用体转义成
        # `⟨'toolcall'⟩⟨'function=Bash'⟩…` 下发，结果是既不是调用、也不是回答的垃圾。
        #
        # 改为从源头不给工具：实测「完全不传 tools」时，同一提示词模型会给出
        # ```bash ls /tmp``` 这种**有用的命令示例**，而非调用体。
        #
        # **双重保险**：
        #   ① 官方开关 `--exclude-tools-when-tool-choice-none`（cli_args.py:114）
        #      已在渲染层接线（`serve/render/serving.py:228`）。开启后同样是
        #      把 `tool_dicts` 置 None，**不依赖**这段代码。
        #   ② 本段是**请求侧兜底**：不读启动参数，引擎换部署/忘开开关时仍成立，
        #      保证 `tool_choice="none"` 的语义不会因启动配置退化。
        # 两层语义相同，重复执行幂等（`tools` 已为 None 时条件不成立）。
        if getattr(request, "tool_choice", None) == "none" and request.tools:
            request = request.model_copy(update={"tools": None})
        return await self.openai_serving_render.render_chat(request)

    async def create_chat_completion(
        self,
        request: ChatCompletionRequest,
        raw_request: Request | None = None,
    ) -> AsyncGenerator[str, None] | ChatCompletionResponse | ErrorResponse:
        """
        Chat Completion API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/chat/create
        for the API specification. This API mimics the OpenAI
        Chat Completion API.
        """
        return await self._with_kv_transfer_rejection_cleanup(
            self._create_chat_completion(request, raw_request), request, raw_request
        )

    async def _create_chat_completion(
        self,
        request: ChatCompletionRequest,
        raw_request: Request | None = None,
    ) -> AsyncGenerator[str, None] | ChatCompletionResponse | ErrorResponse:
        # opt22: fix=2/3 时注入 tool_call_format → 模板渲染对应分支
        from vllm.envs import VLLM_QWEN3X_TOOL_FIX

        if VLLM_QWEN3X_TOOL_FIX == 2:
            request.chat_template_kwargs = {
                **(request.chat_template_kwargs or {}),
                "tool_call_format": "json",
            }
        elif VLLM_QWEN3X_TOOL_FIX == 3:
            request.chat_template_kwargs = {
                **(request.chat_template_kwargs or {}),
                "tool_call_format": "xml",
            }

        # Streaming response
        tokenizer = self.renderer.tokenizer
        assert tokenizer is not None
        chat_template_kwargs = self._effective_chat_template_kwargs(request)
        reasoning_parser: ReasoningParser | None = None
        if self.reasoning_parser_cls:
            reasoning_parser = self.reasoning_parser_cls(
                tokenizer,
                chat_template_kwargs=chat_template_kwargs,  # type: ignore[call-arg]
            )
        result = await self.render_chat_request(request)
        if isinstance(result, ErrorResponse):
            return result

        conversation, engine_inputs = result

        request_id = (
            f"chatcmpl-{self._base_request_id(raw_request, request.request_id)}"
        )

        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        lora_request = self._maybe_get_adapters(request, supports_default_mm_loras=True)

        model_name = self.models.model_name(lora_request)

        # Extract data_parallel_rank from header (router can inject it)
        data_parallel_rank = self._get_data_parallel_rank(raw_request)

        # Schedule the request and get the result generator.
        max_model_len = self.model_config.max_model_len
        generators: list[AsyncGenerator[RequestOutput, None]] = []
        for i, engine_input in enumerate(engine_inputs):
            prompt_token_ids = self._extract_prompt_components(engine_input).token_ids

            # If we are creating sub requests for multiple prompts, ensure that they
            # have unique request ids.
            sub_request_id = (
                request_id if len(engine_inputs) == 1 else f"{request_id}_{i}"
            )

            max_tokens = get_max_tokens(
                max_model_len,
                request.max_completion_tokens
                if request.max_completion_tokens is not None
                else request.max_tokens,
                self._extract_prompt_len(engine_input),
                self.default_sampling_params,
                self.override_max_tokens,
                truncate_prompt_tokens=request.truncate_prompt_tokens,
            )

            sampling_params: SamplingParams | BeamSearchParams
            if request.use_beam_search:
                sampling_params = request.to_beam_search_params(
                    max_tokens, self.default_sampling_params
                )
            else:
                sampling_params = request.to_sampling_params(
                    max_tokens,
                    self.default_sampling_params,
                )

            self._log_inputs(
                sub_request_id,
                engine_input,
                params=sampling_params,
                lora_request=lora_request,
            )

            trace_headers = (
                None
                if raw_request is None
                else await self._get_trace_headers(raw_request.headers)
            )

            if isinstance(sampling_params, BeamSearchParams):
                generator = self.beam_search(
                    prompt=engine_input,
                    request_id=sub_request_id,
                    params=sampling_params,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                )
            else:
                if not request.include_reasoning:
                    reasoning_ended = True
                elif request._grammar_from_tool_parser:
                    # The Mistral grammar already includes an optional
                    # `think?` rule that handles both reasoning and
                    # non-reasoning outputs.
                    reasoning_ended = True
                elif reasoning_parser:
                    reasoning_ended = reasoning_parser.is_reasoning_end(
                        prompt_token_ids or []
                    )
                else:
                    reasoning_ended = None

                generator = self.engine_client.generate(
                    engine_input,
                    sampling_params,
                    sub_request_id,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    priority=request.priority,
                    data_parallel_rank=data_parallel_rank,
                    reasoning_ended=reasoning_ended,
                    reasoning_parser_kwargs={
                        "chat_template_kwargs": chat_template_kwargs,
                    }
                    if reasoning_parser
                    else None,
                )

            generators.append(generator)

        assert len(generators) == 1
        (result_generator,) = generators

        if request.stream:
            return self.chat_completion_stream_generator(
                request,
                result_generator,
                request_id,
                model_name,
                conversation,
                tokenizer,
                request_metadata,
                reasoning_parser,
                chat_template_kwargs=chat_template_kwargs,
            )

        return await self.chat_completion_full_generator(
            request,
            result_generator,
            request_id,
            model_name,
            conversation,
            tokenizer,
            request_metadata,
            reasoning_parser,
        )

    def get_chat_request_role(self, request: ChatCompletionRequest) -> str:
        if request.add_generation_prompt:
            return self.response_role
        return request.messages[-1]["role"]

    async def chat_completion_stream_generator(
        self,
        request: ChatCompletionRequest,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
        model_name: str,
        conversation: list[ConversationMessage],
        tokenizer: TokenizerLike,
        request_metadata: RequestResponseMetadata,
        reasoning_parser: ReasoningParser | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> AsyncGenerator[str, None]:
        created_time = int(time.time())
        chunk_object_type: Final = "chat.completion.chunk"
        first_iteration = True

        # Send response for each token for each request.n (index)
        num_choices = 1 if request.n is None else request.n
        previous_num_tokens = [0] * num_choices
        finish_reason_sent = [False] * num_choices
        num_prompt_tokens = 0
        num_cached_tokens = None
        if self.use_harmony:
            harmony_parsers = [
                get_streamable_parser_for_assistant() for _ in range(num_choices)
            ]
            harmony_tools_streamed = [False] * num_choices
        tools_streamed = [False] * num_choices

        is_mistral_grammar_path = request._grammar_from_tool_parser

        if isinstance(request.tool_choice, ChatCompletionNamedToolChoiceParam):
            tool_choice_function_name = request.tool_choice.function.name
        else:
            tool_choice_function_name = None

        # Determine whether tools are in use with "auto" tool choice
        tool_choice_auto = (
            not tool_choice_function_name
            and self._should_stream_with_auto_tool_parsing(request)
        )

        all_previous_token_ids: list[list[int]] | None
        if self.tool_call_id_type == "kimi_k2":
            history_tool_call_cnt = get_history_tool_calls_cnt(conversation)
        else:
            history_tool_call_cnt = 0

        # Always track previous_texts for comprehensive output logging
        previous_texts = [""] * num_choices

        # Only one of these will be used, thus previous_texts and
        # all_previous_token_ids will not be used twice in the same iteration.
        if (
            is_mistral_grammar_path
            or tool_choice_auto
            or tool_choice_function_name
            or request.tool_choice == "required"
            or reasoning_parser
        ):
            all_previous_token_ids = [[] for _ in range(num_choices)]
            reasoning_end_arr = [False] * num_choices
            prompt_is_reasoning_end_arr: list[bool | None] = [None] * num_choices
        else:
            all_previous_token_ids = None

        try:
            if self.parser_cls is not None:
                if tokenizer is None:
                    raise ValueError(
                        "Tokenizer not available when `skip_tokenizer_init=True`"
                    )
                parsers: list[Parser | None] = [
                    self.parser_cls(
                        tokenizer,
                        request.tools,
                        chat_template_kwargs=chat_template_kwargs,
                    )
                    for _ in range(num_choices)
                ]
                for p in parsers:
                    if p is not None:
                        p._stream_state.tool_call_id_type = self.tool_call_id_type
                        p._stream_state.history_tool_call_cnt = history_tool_call_cnt
                # SX_OPT: without tools (or with tool_choice "none") the
                # tool parser can only rescan every delta. Detach it, so
                # parse_delta passes content through after any reasoning.
                # The Mistral grammar path (which asserts a tool parser and
                # sets _grammar_from_tool_parser even without tools) and the
                # harmony path keep their parser unchanged.
                if (
                    _sx_skip_tool_parser_without_tools()
                    and not is_mistral_grammar_path
                    and not self.use_harmony
                    and (not request.tools or request.tool_choice == "none")
                ):
                    for p in parsers:
                        if p is not None:
                            p.tool_parser = None
            else:
                parsers = [None] * num_choices
        except Exception as e:
            logger.exception("Error in parser creation.")
            data = self.create_streaming_error_response(e)
            yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"
            return

        stream_options = request.stream_options
        include_usage, include_continuous_usage = should_include_usage(
            stream_options, self.enable_force_include_usage
        )

        last_server_send_time = time.time()
        try:
            # [1Cat P4] Retain the last engine stats so the trailing usage chunk
            # can carry the engine-authoritative timing breakdown.
            last_metrics = None
            async for res in result_generator:
                # opt21: ignore engine keepalive (15s), handle server keepalive (120s)
                if res is STREAM_KEEPALIVE:
                    now = time.time()
                    if now - last_server_send_time >= 120:
                        yield ": keepalive\n\n"
                        last_server_send_time = now
                    continue

                last_metrics = getattr(res, "metrics", None)

                if res.prompt_token_ids is not None:
                    num_prompt_tokens = len(res.prompt_token_ids)
                    if res.encoder_prompt_token_ids is not None:
                        num_prompt_tokens += len(res.encoder_prompt_token_ids)

                # We need to do it here, because if there are exceptions in
                # the result_generator, it needs to be sent as the FIRST
                # response (by the try...catch).
                if first_iteration:
                    num_cached_tokens = res.num_cached_tokens
                    # Send first response for each request.n (index) with
                    # the role
                    role = self.get_chat_request_role(request)

                    # ``res.prompt`` is the rendered chat-templated prompt
                    prompt_text = res.prompt if request.return_prompt_text else None

                    # NOTE num_choices defaults to 1 so this usually executes
                    # once per request
                    for i in range(num_choices):
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=DeltaMessage(
                                role=role,
                                content="",
                            ),
                            logprobs=None,
                            finish_reason=None,
                        )

                        # return prompt_token_ids at the first chunk ever
                        chunk = ChatCompletionStreamResponse(
                            id=request_id,
                            object=chunk_object_type,
                            created=created_time,
                            choices=[choice_data],
                            model=model_name,
                            prompt_token_ids=(
                                res.prompt_token_ids
                                if request.return_token_ids
                                else None
                            ),
                            prompt_text=prompt_text,
                        )

                        # if continuous usage stats are requested, add it
                        if include_continuous_usage:
                            chunk.usage = UsageInfo(
                                prompt_tokens=num_prompt_tokens,
                                completion_tokens=0,
                                total_tokens=num_prompt_tokens,
                            )

                        data = chunk.model_dump_json(exclude_unset=True)
                        yield f"data: {data}\n\n"

                    # Send response to echo the input portion of the
                    # last message
                    if request.echo:
                        last_msg_content: str | list[dict[str, str]] = ""
                        if (
                            conversation
                            and "content" in conversation[-1]
                            and conversation[-1].get("role") == role
                        ):
                            last_msg_content = conversation[-1]["content"] or ""

                        if last_msg_content:
                            for i in range(num_choices):
                                choice_data = ChatCompletionResponseStreamChoice(
                                    index=i,
                                    delta=DeltaMessage(content=last_msg_content),
                                    logprobs=None,
                                    finish_reason=None,
                                )
                                chunk = ChatCompletionStreamResponse(
                                    id=request_id,
                                    object=chunk_object_type,
                                    created=created_time,
                                    choices=[choice_data],
                                    model=model_name,
                                )
                                if include_continuous_usage:
                                    chunk.usage = UsageInfo(
                                        prompt_tokens=num_prompt_tokens,
                                        completion_tokens=0,
                                        total_tokens=num_prompt_tokens,
                                    )

                                data = chunk.model_dump_json(exclude_unset=True)
                                yield f"data: {data}\n\n"
                    first_iteration = False

                for output in res.outputs:
                    i = output.index
                    parser = parsers[i]
                    tool_parser = parser.tool_parser if parser is not None else None

                    if (
                        reasoning_parser
                        and res.prompt_token_ids
                        and prompt_is_reasoning_end_arr[i] is None
                    ):
                        # only check once per choice, because prompt_token_ids
                        # are the same for all deltas in that choice
                        prompt_is_reasoning_end_arr[i] = (
                            reasoning_parser.is_reasoning_end(res.prompt_token_ids)
                        )
                    if finish_reason_sent[i]:
                        continue

                    if request.logprobs and request.top_logprobs is not None:
                        assert output.logprobs is not None, "Did not output logprobs"
                        logprobs = self._create_chat_logprobs(
                            token_ids=output.token_ids,
                            top_logprobs=output.logprobs,
                            tokenizer=tokenizer,
                            num_output_top_logprobs=request.top_logprobs,
                            return_as_token_id=request.return_tokens_as_token_ids,
                        )
                    else:
                        logprobs = None

                    if self.use_harmony:
                        harmony_parser = harmony_parsers[i]
                        prev_recipient = harmony_parser.current_recipient

                        # Track accumulated content per token with their state
                        token_states: list[TokenState] = []
                        for token_id in output.token_ids:
                            harmony_parser.process(token_id)
                            token_delta = harmony_parser.last_content_delta or ""
                            token_states.append(
                                TokenState(
                                    harmony_parser.current_channel,
                                    harmony_parser.current_recipient,
                                    token_delta,
                                )
                            )
                        delta_text = "".join(delta for _, _, delta in token_states)
                        cur_channel = harmony_parser.current_channel

                        # handle the case where several tokens where generated at once
                        # including the final token, leading to a delta in the text
                        # but the current channel to be empty (start state)
                        if not cur_channel and delta_text:
                            cur_channel = "final"
                    else:
                        delta_text = output.text

                    if (
                        not delta_text
                        and not output.token_ids
                        and not previous_num_tokens[i]
                    ):
                        # Chunked prefill case, don't return empty chunks
                        continue

                    delta_message: DeltaMessage | None

                    # just update previous_texts and previous_token_ids
                    if (
                        is_mistral_grammar_path
                        or tool_choice_auto
                        or tool_choice_function_name
                        or request.tool_choice == "required"
                        or reasoning_parser
                    ):
                        assert previous_texts is not None
                        assert all_previous_token_ids is not None
                        previous_text = previous_texts[i]
                        previous_token_ids = all_previous_token_ids[i]
                        current_text = previous_text + delta_text
                        # avoid the None + list error.
                        if previous_token_ids:
                            current_token_ids = previous_token_ids + as_list(
                                output.token_ids
                            )
                        else:
                            current_token_ids = as_list(output.token_ids)

                    if self.use_harmony:
                        delta_message, tools_streamed_flag = (
                            extract_harmony_streaming_delta(
                                harmony_parser=harmony_parser,
                                token_states=token_states,
                                prev_recipient=prev_recipient,
                                include_reasoning=request.include_reasoning,
                            )
                        )
                        harmony_tools_streamed[i] |= tools_streamed_flag
                    # Mistral grammar path: combined reasoning + tool streaming
                    elif is_mistral_grammar_path:
                        from vllm.tool_parsers.mistral_tool_parser import (
                            MistralToolParser,
                        )

                        assert tool_parser is not None
                        assert isinstance(tool_parser, MistralToolParser)
                        assert reasoning_end_arr is not None
                        output_token_ids = as_list(output.token_ids)
                        result = tool_parser.extract_maybe_reasoning_and_tool_streaming(
                            reasoning_parser=reasoning_parser,
                            previous_text=previous_text,
                            current_text=current_text,
                            delta_text=delta_text,
                            previous_token_ids=previous_token_ids,
                            current_token_ids=current_token_ids,
                            output_token_ids=output_token_ids,
                            reasoning_ended=reasoning_end_arr[i],
                            prompt_is_reasoning_end=(prompt_is_reasoning_end_arr[i]),
                            request=request,
                        )
                        delta_message = result.delta_message
                        reasoning_end_arr[i] = result.reasoning_ended
                        current_text = result.current_text
                        current_token_ids = result.current_token_ids
                        if result.tools_called:
                            tools_streamed[i] = True

                    elif parser is not None:
                        # [2026-10-20] 把「思考域已结束」这一**权威事实**下发
                        # 给 tool parser。动机：qwen3coder 的
                        # `_should_defer_parsing` 过去只看「工具是否在流式白名单
                        # 里」——思考开启时会把 Bash / Grep 等推出到 finish，首
                        # 工具帧 = 整轮生成时间。推迟的原始动机是「思考域不参与
                        # 工具解析」，思考域结束后动机即消失。G29/G27 的模板把
                        # `<think>` 放在 prompt 里、生成侧从不输出 `</think>`
                        # ⇒ 文本层判据永远不成立，只有协调器的状态可靠。
                        delta_message = parser.parse_delta(
                            delta_text=delta_text,
                            delta_token_ids=as_list(output.token_ids),
                            request=request,
                            prompt_token_ids=res.prompt_token_ids,
                        )
                        # 必须在 parse_delta **之后**取：`reasoning_ended` 正是在
                        # parse_delta 内部由 `is_reasoning_end_streaming` 置位的。
                        _coord_state = getattr(parser, "_stream_state", None)
                        if (
                            getattr(_coord_state, "reasoning_ended", False)
                            and tool_parser is not None
                            and hasattr(tool_parser, "note_thinking_ended")
                        ):
                            tool_parser.note_thinking_ended()
                        # [2026-10-20] 流式侧的双 `</think>` 还原。
                        # 非流式在 `chat_completion_full_generator` 里已接
                        # `_merge_double_think_close()`；流式此前漏网 —— 实测并发
                        # 16 路中 7 路在 content 帧里出现字面 `</think>`（多在该路
                        # 的倒数几帧），即模型第二个闭合标签直接下发给用户。
                        # 处理：content 帧出现 `</think>` 时把该标签之前的部分并到
                        # reasoning，标签本身丢弃；之后若还有内容才是真 content。
                        delta_message = _fix_stream_double_think(delta_message)
                        if delta_message and delta_message.tool_calls:
                            tools_streamed[i] = True
                    # handle streaming just a content delta (no parsers)
                    else:
                        delta_message = DeltaMessage(content=delta_text)

                    # update the previous values for the next iteration
                    if (
                        is_mistral_grammar_path
                        or tool_choice_auto
                        or tool_choice_function_name
                        or request.tool_choice == "required"
                        or reasoning_parser
                    ) and not self.use_harmony:
                        assert previous_texts is not None
                        assert all_previous_token_ids is not None
                        previous_texts[i] = current_text
                        all_previous_token_ids[i] = current_token_ids
                    else:
                        # Update for comprehensive logging even in simple case
                        assert previous_texts is not None
                        previous_texts[i] += delta_text

                    # set the previous values for the next iteration
                    previous_num_tokens[i] += len(output.token_ids)

                    # if the message delta is None (e.g. because it was a
                    # "control token" for tool calls or the parser otherwise
                    # wasn't ready to send a token, then
                    #   get the next token without streaming a chunk
                    if delta_message is None:
                        # NOTE: If return_token_ids is enabled, we still need to
                        # send a chunk with token_ids even if delta_message is None
                        # to ensure all tokens are included in the response
                        if (
                            output.finish_reason is None
                            and not request.return_token_ids
                        ):
                            continue
                        delta_message = DeltaMessage()

                    # Log streaming delta if output logging is enabled
                    if self.enable_log_outputs and self.request_logger:
                        delta_content_parts = []
                        if delta_message.content:
                            delta_content_parts.append(delta_message.content)
                        if delta_message.reasoning:
                            reasoning = delta_message.reasoning
                            delta_content_parts.append(f"[reasoning: {reasoning}]")
                        if delta_message.tool_calls:
                            tool_args = "".join(
                                tc.function.arguments
                                for tc in delta_message.tool_calls
                                if tc.function and tc.function.arguments
                            )
                            if tool_args:
                                delta_content_parts.append(f"[tool_calls: {tool_args}]")

                        if delta_content_parts and self.enable_log_deltas:
                            delta_content = " ".join(delta_content_parts)
                            self.request_logger.log_outputs(
                                request_id=request_id,
                                outputs=delta_content,
                                output_token_ids=as_list(output.token_ids),
                                finish_reason=output.finish_reason,
                                is_streaming=True,
                                delta=True,
                            )

                    if output.finish_reason is None:
                        # Send token-by-token response for each request.n
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=delta_message,
                            logprobs=logprobs,
                            finish_reason=None,
                            token_ids=(
                                as_list(output.token_ids)
                                if request.return_token_ids
                                else None
                            ),
                        )

                    # if the model is finished generating
                    else:
                        # check for error finish reason and abort streaming
                        # finish_reason='error' indicates a retryable error
                        self._raise_if_error(output.finish_reason, request_id)

                        # check to make sure we haven't "forgotten" to stream
                        #   any tokens that were generated but previously
                        #   matched by partial json parsing
                        # only happens if we are NOT using structured outputs
                        index = 0
                        auto_tools_called = False
                        # opt22d 第七轮：推迟补发是“一次性完整下发”，须跳过下方
                        # “补发未流出参数”逻辑——那套逻辑按流式解析留下的
                        # prev_tool_call_arr / streamed_args_for_tool 取索引，而被
                        # 推迟的调用从未走过流式路径、这两者皆为空 → IndexError。
                        _deferred_emitted = False
                        if tool_parser:
                            auto_tools_called = len(tool_parser.prev_tool_call_arr) > 0
                            index = (
                                len(tool_parser.prev_tool_call_arr) - 1
                                if auto_tools_called
                                else 0
                            )

                        # opt22d 第十二轮（缺陷B）：参数体内含文档示例时，收尾标签
                        # （`</parameter>`/`</function>`/`</tool_call>`）被判为**内容**
                        # 全部转义，expat 永远收不到闭合 ⇒ 流式阶段调用不入账。
                        # 消息已完整，此处强制收尾并用完整文本重解析。
                        if tool_parser and getattr(
                            tool_parser, "_needs_finish_flush", False
                        ):
                            _flush_info = tool_parser.flush_streaming_finish()
                            if _flush_info is not None and _flush_info.tool_calls:
                                if delta_message is None:
                                    delta_message = DeltaMessage()
                                _base = len(delta_message.tool_calls or [])
                                delta_message.tool_calls = list(
                                    delta_message.tool_calls or []
                                ) + [
                                    DeltaToolCall(
                                        # ⚠️ **沿用原 index**：收尾片段属于**同一个**
                                        # 已流式下发的调用（前面发过函数名与部分参数），
                                        # 重编 index 会被客户端当成新调用 ⇒ arguments 无法
                                        # 拼接、JSON 不闭合（实测线上）。
                                        index=(
                                            _tc.index
                                            if getattr(_tc, "index", None) is not None
                                            else _base + _di
                                        ),
                                        id=_tc.id,
                                        type="function",
                                        function=DeltaFunctionCall(
                                            name=_tc.function.name,
                                            arguments=_tc.function.arguments,
                                        ),
                                    )
                                    for _di, _tc in enumerate(_flush_info.tool_calls)
                                ]
                                auto_tools_called = True
                                _deferred_emitted = True
                                tools_streamed[i] = True
                            tool_parser._needs_finish_flush = False

                        # opt22d 第七轮：开启思考时，非 Write/Edit 的工具调用被推迟
                        # 到此刻用**完整文本**解析（流式阶段刻意未产出，避免 thinking
                        # 里写出的形似标签被当成真实调用）。previous_texts[i] 是本轮
                        # 累积的完整原文；先经 reasoning parser 剥出正文段，再交工具
                        # 解析器——后者复用非流式路径的全套 opt22d 防护。
                        # 注意：_reset_streaming_state() 会清空这两个标志，故先读。
                        if (
                            tool_parser
                            and previous_texts is not None
                            and reasoning_parser is not None
                            and (
                                getattr(tool_parser, "_thinking_enabled", False)
                                or getattr(tool_parser, "_coalesce_mode", False)
                            )
                            and getattr(tool_parser, "_deferred_used", False)
                        ):
                            # [本地 2026-10-13] 捕获「已流式下发」的调用数，供下方
                            # 被推迟（非流式）调用**续号**。必须在此处读：紧接着的
                            # `extract_tool_calls_deferred` 内部会调
                            # `_reset_streaming_state()` 把 current_tool_index 归零。
                            # ⚠️ 不续号的后果（实测 WebFetch(流式)+Bash/Glob(非流式)）：
                            # 非流式调用从 0 重编号 ⇒ Bash 也发到 idx0，与流式工具撞车，
                            # 客户端把两个调用的 arguments 拼进同一 index ⇒ 必然
                            # `Extra data`（用户报「bash 一堆执行失败」）。
                            _streamed_base = getattr(tool_parser, "current_tool_index", 0)
                            _, _deferred_content = reasoning_parser.extract_reasoning(
                                previous_texts[i], request=request
                            )
                            _deferred_info = tool_parser.extract_tool_calls_deferred(
                                _deferred_content or "", request
                            )
                            if _deferred_info.tools_called:
                                if delta_message is None:
                                    delta_message = DeltaMessage()
                                _base = max(
                                    _streamed_base,
                                    len(delta_message.tool_calls or []),
                                )
                                # [本地 2026-10-13] 优先用解析器记录的**原始下标**
                                # （确定性）；`_base + _di` 仅作兜底。
                                _deferred_indices = (
                                    getattr(tool_parser, "_deferred_indices", None) or []
                                )
                                delta_message.tool_calls = list(
                                    delta_message.tool_calls or []
                                ) + [
                                    DeltaToolCall(
                                        index=(
                                            _deferred_indices[_di]
                                            if _di < len(_deferred_indices)
                                            else _base + _di
                                        ),
                                        id=_tc.id,
                                        type="function",
                                        function=DeltaFunctionCall(
                                            name=_tc.function.name,
                                            arguments=_tc.function.arguments,
                                        ),
                                    )
                                    for _di, _tc in enumerate(_deferred_info.tool_calls)
                                ]
                                auto_tools_called = True
                                _deferred_emitted = True
                                tools_streamed[i] = True
                        should_check = (
                            self._should_check_for_unstreamed_tool_arg_tokens(
                                delta_message, output
                            )
                        )
                        # only check if there are any tool calls
                        # detected by partial parsing
                        # [本地 2026-10-12] 越界守卫（修 502）。
                        # 上方 955 的 flush_streaming_finish() 内部会走非流式路径
                        # extract_tool_calls()，而它会重置这两个列表
                        # （qwen3coder_tool_parser.py:2086 `.clear()` / :2025 重建）。
                        # 但 index 是 941 用**重置前的长度**算出来的；若 flush 判定
                        # 无需补发（_flush_info.tool_calls 为空 ⇒ 不进 957 分支 ⇒
                        # _deferred_emitted 仍是 False），本块照样会进来 ⇒ 1064/1074
                        # 按下标取值越界 ⇒ IndexError 被当 error chunk 下发 ⇒
                        # 客户端 502「Invalid data format received」。
                        # 故按**当前**长度复核；列表为空即无已入账调用，跳过补发。
                        # _pca = prev_tool_call_arr，_sat = streamed_args_for_tool。
                        _pca = getattr(tool_parser, "prev_tool_call_arr", ()) or ()
                        _sat = getattr(tool_parser, "streamed_args_for_tool", ()) or ()
                        # [本地 2026-10-13] 「已推迟下发」≠「已下发足够」：
                        # `_deferred_emitted` 只在 983/1025 标记**发过一次**，但
                        # 流式侧 `streamed_args_for_tool` 仍可能短于完整值
                        # （实测 r5：exp=15705 / streamed=15703，差 2 字节）
                        # ⇒ 客户端 JSON 报 `Unterminated string at char 6771`。
                        # 故 deferred 只有在 streamed 已达完整长度时才允许跳过补发。
                        _sat_short = False
                        if 0 <= index < len(_pca) and 0 <= index < len(_sat):
                            _av0 = _pca[index].get("arguments", "")
                            _exp0 = (
                                _av0
                                if isinstance(_av0, str)
                                else json.dumps(_av0, ensure_ascii=False)
                            )
                            _sat_short = bool(_exp0) and len(_sat[index]) < len(_exp0)
                        if (
                            should_check
                            and tool_parser
                            and auto_tools_called
                            and (not _deferred_emitted or _sat_short)
                            and 0 <= index < len(_pca)
                            and 0 <= index < len(_sat)
                        ):
                            latest_delta_len = 0
                            if (
                                isinstance(
                                    delta_message.tool_calls[0].function,
                                    DeltaFunctionCall,
                                )
                            ) and isinstance(
                                delta_message.tool_calls[0].function.arguments, str
                            ):
                                latest_delta_len = len(
                                    delta_message.tool_calls[0].function.arguments
                                )

                            # get the expected call based on partial JSON
                            # parsing which "autocompletes" the JSON.
                            # Tool parsers (e.g. Qwen3Coder) store
                            # arguments as a JSON string in
                            # prev_tool_call_arr. Calling json.dumps()
                            # on an already-serialized string would
                            # double-serialize it (e.g. '{"k":1}' becomes
                            # '"{\\"k\\":1}"'), which then causes the
                            # replace() below to fail and append the
                            # entire double-serialized string as a
                            # spurious final delta.
                            args = tool_parser.prev_tool_call_arr[index].get(
                                "arguments", {}
                            )
                            if isinstance(args, str):
                                expected_call = args
                            else:
                                expected_call = json.dumps(args, ensure_ascii=False)

                            # get what we've streamed so far for arguments
                            # for the current tool
                            # [opt22h 2026-10-14] 修「扣减负切片」缺陷：旧写法
                            # `actual_call[:-latest_delta_len]` 在
                            # latest_delta_len > len(actual_call) 时静默返回
                            # **空串**（Python 负切片越界不报错）。触发场景：
                            # finish 帧的 delta_message 已含 flush_streaming_finish
                            # 补发的增量（full[sent:]，如 1231B），而流式侧
                            # streamed_args_for_tool 只有部分（如 30B）——
                            # 扣减把 30B 吃成 "" ⇒ 误判「客户端什么都没收到」⇒
                            # remaining 退化成 expected 全文（1261B），且
                            # _create_remaining_args_delta 用它**整帧替换**
                            # delta_message ⇒ flush 的正确增量(1231B)被覆盖成
                            # 全量(1261B) ⇒ 客户端拼接 30+1261=1291B，头部
                            # 30B 重复、JSON 报 `Expecting ',' delimiter`
                            # （2026-10-14 活体抓帧实录，write_file 大参数
                            # max_tokens 截断场景）。修法：扣减将越界时，
                            # delta 增量属于**收尾补发**而非「本帧新发」——
                            # 客户端实收 = streamed + 该增量 ⇒ 把增量并入
                            # actual 后走 _remaining_tool_args 前缀对齐，
                            # 得 remaining=""（已完整）⇒ 不再重复补发。
                            actual_call = tool_parser.streamed_args_for_tool[index]
                            if latest_delta_len > 0:
                                _cut = len(actual_call) - latest_delta_len
                                if _cut >= 0:
                                    actual_call = actual_call[:_cut]
                                else:
                                    _flush_delta = (
                                        delta_message.tool_calls[0].function.arguments
                                        or ""
                                    )
                                    actual_call = actual_call + _flush_delta

                            # check to see if there's anything left to stream
                            # [本地 2026-10-12] 分歧守卫（与 parser 侧
                            # flush_streaming_finish 的 2026-10-05 修正同构）。
                            # 旧写法 `expected_call.replace(actual_call, "", 1)`
                            # 在 actual 不是 expected 真前缀时【匹配不到】⇒
                            # remaining 退化成 expected 全文 ⇒ 客户端已收过
                            # actual 又收一遍全文 ⇒ arguments 重复 ⇒ JSON 解析
                            # 失败（线上实测 9 次，harness 报「Edit was called
                            # with input that could not be parsed as JSON」）。
                            # 上面 1076-1078 的上游注释已指出 replace 会失败并
                            # append 整串，此处补上防御。
                            # [本地 2026-10-13] 幂等守卫（第二道）：本 index 一旦
                            # 补发完成，后续 finish output 不得再补。
                            # 触发原因：939 行把 `_deferred_emitted` 在**每个**
                            # finish output 上重置为 False，故 finish 输出不止一个
                            # 时本块会重复进入；此时 streamed 已被推进到完整值，
                            # 上面的 latest_delta_len 扣除会把尾部误判成"待补"
                            # ⇒ 尾部重复下发（实测多发 6286 B，JSON 报
                            # `Extra data: line 1 column 15914`）。
                            _synced = getattr(tool_parser, "_args_synced_idx", None)
                            if _synced is None:
                                try:
                                    _synced = set()
                                    tool_parser._args_synced_idx = _synced
                                except Exception:  # noqa: BLE001
                                    _synced = None
                            if _synced is not None and index in _synced:
                                remaining_call = ""
                            else:
                                remaining_call = self._remaining_tool_args(
                                    expected_call, actual_call
                                )
                            # [本地 2026-10-05] 收尾 remaining 诊断（env 门控，默认关）。
                            # 用于定位「流式尾部重复补发」：打印各长度与 remaining 尾部。
                            # [opt22 活体抓帧 2026-10-14] 门控扩展：除 env 外支持
                            # **开关文件** `/tmp/log/.opt23_live`（存在即开）——env
                            # 改动到不了已运行进程（unit env 历史坑），文件开关配合
                            # 热重载可**免重启**抓帧；finish 才走此块，一次 stat
                            # 开销可忽略。关=删除文件。
                            _opt23_live = (
                                os.path.exists("/tmp/log/.opt23_live")
                                or os.environ.get("VLLM_OPT23_DEBUG_LOG")
                            )
                            if _opt23_live:
                                try:
                                    with open(
                                        "/tmp/log/vllm-tool-log.txt", "a", buffering=1
                                    ) as _fh:
                                        _fh.write(
                                            "[serving-remaining] idx=%s expected=%d "
                                            "actual=%d latest_delta=%d remaining=%d "
                                            "rem_tail=%r\n"
                                            % (
                                                index,
                                                len(expected_call),
                                                len(actual_call),
                                                latest_delta_len,
                                                len(remaining_call),
                                                remaining_call[-40:],
                                            )
                                        )
                                except Exception:
                                    pass
                            # set that as a delta message
                            # [opt22h 2026-10-14] remaining 为空 ⇒ 无需补发，
                            # **保留原 delta_message**。旧写法无条件整帧替换，
                            # 把 delta_message 里已有的收尾增量（flush 补发的
                            # full[sent:]，如 1259B）替换成空串 ⇒ 客户端只收到
                            # 流式部分（30B），JSON 截断（本轮活体实录：帧3
                            # arguments=0B，拼包 len=30 报
                            # `Expecting ',' delimiter: char 30`）。
                            if remaining_call:
                                delta_message = self._create_remaining_args_delta(
                                    delta_message, remaining_call, index
                                )
                            # [本地 2026-10-12] 幂等保护：补发后把「已下发」推进到
                            # 完整值。上游本块只发 delta、**不更新任何状态**，而
                            # should_check 只要求 `output.finish_reason is not
                            # None`——finish_reason 在多个 output 上出现时本块会
                            # 被反复进入，streamed_args_for_tool 却不变 ⇒ 同一段
                            # remaining 被反复补发。实测（全文复现）：同一
                            # 13488 B 完整 arguments 等距重发 5 次，客户端拼出
                            # 62219 B，JSON 只能解析前 13488 B。
                            if remaining_call:
                                self._advance_streamed_args(
                                    tool_parser, index, expected_call
                                )
                                if _synced is not None:
                                    _synced.add(index)

                        # Send the finish response for each request.n only once
                        # In OpenAI's API, when a tool is called, the
                        # finish_reason is:
                        # "tool_calls" for "auto" or "required" tool calls,
                        # and "stop" for named tool calls.
                        if (
                            auto_tools_called
                            or (tools_streamed[i] and not tool_choice_function_name)
                            or (self.use_harmony and harmony_tools_streamed[i])
                        ):
                            finish_reason_ = "tool_calls"
                        else:
                            finish_reason_ = (
                                output.finish_reason if output.finish_reason else "stop"
                            )
                        choice_data = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=delta_message,
                            logprobs=logprobs,
                            finish_reason=finish_reason_,
                            stop_reason=output.stop_reason,
                            token_ids=(
                                as_list(output.token_ids)
                                if request.return_token_ids
                                else None
                            ),
                        )

                        finish_reason_sent[i] = True

                    choice_data = maybe_filter_parallel_tool_calls(choice_data, request)
                    chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        object=chunk_object_type,
                        created=created_time,
                        choices=[choice_data],
                        model=model_name,
                    )
                    # Stamp the fingerprint on terminal chunks only (those with
                    # finish_reason set). When ``include_usage`` is on, the
                    # trailing usage chunk below overrides this as the true
                    # final message.
                    if (
                        not include_usage
                        and self.system_fingerprint is not None
                        and choice_data.finish_reason is not None
                    ):
                        chunk.system_fingerprint = self.system_fingerprint

                    # handle usage stats if requested & if continuous
                    if include_continuous_usage:
                        completion_tokens = previous_num_tokens[i]
                        chunk.usage = UsageInfo(
                            prompt_tokens=num_prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=num_prompt_tokens + completion_tokens,
                        )

                    data = chunk.model_dump_json(exclude_unset=True)
                    yield f"data: {data}\n\n"

            # once the final token is handled, if stream_options.include_usage
            # is sent, send the usage
            if include_usage:
                completion_tokens = sum(previous_num_tokens)
                final_usage = UsageInfo(
                    prompt_tokens=num_prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=num_prompt_tokens + completion_tokens,
                )
                if self.enable_prompt_tokens_details and num_cached_tokens:
                    final_usage.prompt_tokens_details = PromptTokenUsageInfo(
                        cached_tokens=num_cached_tokens
                    )
                streamed_reasoning_tokens = _count_reasoning_tokens(
                    reasoning_parser, all_previous_token_ids
                )
                if streamed_reasoning_tokens is not None:
                    final_usage.completion_tokens_details = (
                        CompletionTokenUsageInfo(
                            reasoning_tokens=streamed_reasoning_tokens
                        )
                    )

                # [1Cat P4] Engine-authoritative per-request timings (omitted by
                # exclude_none below when stats were not collected).
                final_usage.timings = _usage_timings(last_metrics)

                final_usage_chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    object=chunk_object_type,
                    created=created_time,
                    choices=[],
                    model=model_name,
                    usage=final_usage,
                    system_fingerprint=self.system_fingerprint,
                )
                final_usage_data = final_usage_chunk.model_dump_json(
                    exclude_unset=True, exclude_none=True
                )
                yield f"data: {final_usage_data}\n\n"

            # report to FastAPI middleware aggregate usage across all choices
            num_completion_tokens = sum(previous_num_tokens)
            request_metadata.final_usage_info = UsageInfo(
                prompt_tokens=num_prompt_tokens,
                completion_tokens=num_completion_tokens,
                total_tokens=num_prompt_tokens + num_completion_tokens,
            )
            aggregate_reasoning_tokens = _count_reasoning_tokens(
                reasoning_parser, all_previous_token_ids
            )
            if aggregate_reasoning_tokens is not None:
                request_metadata.final_usage_info.completion_tokens_details = (
                    CompletionTokenUsageInfo(reasoning_tokens=aggregate_reasoning_tokens)
                )

            # Log complete streaming response if output logging is enabled
            if self.enable_log_outputs and self.request_logger:
                # Log the complete response for each choice
                for i in range(num_choices):
                    full_text = (
                        previous_texts[i]
                        if previous_texts and i < len(previous_texts)
                        else f"<streaming_complete: {previous_num_tokens[i]} tokens>"
                    )
                    self.request_logger.log_outputs(
                        request_id=request_id,
                        outputs=full_text,
                        output_token_ids=None,  # Consider also logging all token IDs
                        finish_reason="streaming_complete",
                        is_streaming=True,
                        delta=False,
                    )

        except GenerationError as e:
            yield f"data: {self._convert_generation_error_to_streaming_response(e)}\n\n"
        except Exception as e:
            logger.exception("Error in chat completion stream generator.")
            data = self.create_streaming_error_response(e)
            yield f"data: {data}\n\n"
        # Send the final done message after all response.n are finished
        yield "data: [DONE]\n\n"

    async def chat_completion_full_generator(
        self,
        request: ChatCompletionRequest,
        result_generator: AsyncIterator[RequestOutput],
        request_id: str,
        model_name: str,
        conversation: list[ConversationMessage],
        tokenizer: TokenizerLike,
        request_metadata: RequestResponseMetadata,
        reasoning_parser: ReasoningParser | None = None,
    ) -> ErrorResponse | ChatCompletionResponse:
        created_time = int(time.time())
        final_res: RequestOutput | None = None

        try:
            async for res in result_generator:
                final_res = res
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")

        if final_res is None:
            return self.create_error_response(
                "No output received from the engine.",
                err_type="InternalServerError",
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            )

        choices: list[ChatCompletionResponseChoice] = []
        if self.tool_call_id_type == "kimi_k2":
            history_tool_call_cnt = get_history_tool_calls_cnt(conversation)
        else:
            history_tool_call_cnt = 0

        role = self.get_chat_request_role(request)
        for output in final_res.outputs:
            # check for error finish reason and raise GenerationError
            # finish_reason='error' indicates a retryable request-level internal error
            self._raise_if_error(output.finish_reason, request_id)
            token_ids = output.token_ids
            out_logprobs = output.logprobs
            tool_call_info = None

            if request.logprobs and request.top_logprobs is not None:
                assert out_logprobs is not None, "Did not output logprobs"
                logprobs = self._create_chat_logprobs(
                    token_ids=token_ids,
                    top_logprobs=out_logprobs,
                    num_output_top_logprobs=request.top_logprobs,
                    tokenizer=tokenizer,
                    return_as_token_id=request.return_tokens_as_token_ids,
                )
            else:
                logprobs = None

            if self.use_harmony:
                reasoning, content, _ = parse_chat_output(token_ids)
                if not request.include_reasoning:
                    reasoning = None

                if self.tool_parser is not None:
                    if tokenizer is None:
                        raise ValueError(
                            "Tokenizer not available when `skip_tokenizer_init=True`"
                        )

                    tool_parser = self.tool_parser(tokenizer, request.tools)
                    # NOTE: We use token_ids for openai tool parser
                    tool_call_info = tool_parser.extract_tool_calls(
                        "",
                        request=request,
                        token_ids=token_ids,  # type: ignore
                    )
                    content = tool_call_info.content
                    message = ChatMessage(
                        role=role,
                        reasoning=reasoning,
                        content=content,
                        tool_calls=tool_call_info.tool_calls,
                    )
                else:
                    message = ChatMessage(
                        role=role,
                        reasoning=reasoning,
                        content=content,
                    )

                # Encode routed_experts for transport. JSON can't carry raw
                # bytes, so we write the ndarray as a ``.npy`` byte stream
                # and base64-encode it. ``pybase64`` is ~3x faster than the
                # stdlib ``base64`` on large payloads thanks to SIMD.
                routed_experts_b64 = None
                if output.routed_experts is not None:
                    buf = io.BytesIO()
                    np.save(buf, output.routed_experts)
                    routed_experts_b64 = base64.b64encode(buf.getvalue()).decode(
                        "ascii"
                    )

                choice_data = ChatCompletionResponseChoice(
                    index=output.index,
                    message=message,
                    logprobs=logprobs,
                    finish_reason=(
                        "tool_calls"
                        if (tool_call_info is not None and tool_call_info.tools_called)
                        else output.finish_reason
                        if output.finish_reason
                        else "stop"
                    ),
                    stop_reason=output.stop_reason,
                    token_ids=(
                        as_list(output.token_ids) if request.return_token_ids else None
                    ),
                    routed_experts=routed_experts_b64,
                )
                choices.append(choice_data)
                continue

            if reasoning_parser:
                # If the reasoning parser is enabled,
                # tool calls are extracted exclusively from the content.
                reasoning, content = reasoning_parser.extract_reasoning(
                    output.text, request=request
                )
                reasoning, content = _merge_double_think_close(
                    reasoning, content, reasoning_parser
                )
                if not request.include_reasoning:
                    reasoning = None
                # opt22d 第七轮：reasoning 此前无任何转义收口（R1 只覆盖 content），
                # 模型在思考里写出的形似工具调用标签会以裸形式下发，可能被下游
                # harness 误当真实调用执行。此处按与 content 相同的规则转义
                # （escape_tool_like_tags 幂等，重复施加无害）。
                # 本处一并覆盖 Anthropic 协议——anthropic/serving.py:505 复用本
                # generator，其 thinking block 取的正是这里产出的 message.reasoning。
                if reasoning:
                    # [example 区间 2026-10-18] 示例容器内原样，只转义区间外
                    reasoning = escape_tool_like_tags_outside_examples(reasoning)
            else:
                reasoning = None
                content = output.text

            auto_tools_called = False
            # if auto tools are not enabled, and a named tool choice using
            #   outlines is not being used
            tool_calls, content = self._parse_tool_calls_from_content(
                request=request,
                tokenizer=tokenizer,
                content=content,
                enable_auto_tools=self.enable_auto_tools,
                tool_parser_cls=self.tool_parser,
            )
            if is_mistral_tokenizer(tokenizer):
                from vllm.tool_parsers.mistral_tool_parser import MistralToolCall

                tool_call_class: type[ToolCall] = MistralToolCall
            else:
                tool_call_class = ToolCall

            use_mistral_tool_parser = request._grammar_from_tool_parser
            if use_mistral_tool_parser:
                from vllm.tool_parsers.mistral_tool_parser import MistralToolParser

                tool_call_items = MistralToolParser.build_non_streaming_tool_calls(
                    tool_calls
                )
                if tool_call_items:
                    auto_tools_called = (
                        request.tool_choice is None or request.tool_choice == "auto"
                    )
                message = ChatMessage(
                    role=role,
                    reasoning=reasoning,
                    content=content,
                    tool_calls=tool_call_items,
                )

            elif (not self.enable_auto_tools or not self.tool_parser) and (
                not isinstance(request.tool_choice, ChatCompletionNamedToolChoiceParam)
                and request.tool_choice != "required"
            ):
                message = ChatMessage(role=role, reasoning=reasoning, content=content)

            elif (
                request.tool_choice
                and type(request.tool_choice) is ChatCompletionNamedToolChoiceParam
            ):
                tool_call_class_items = []
                tool_calls = tool_calls or []
                for idx, tc in enumerate(tool_calls):
                    # Use native ID if available (e.g., Kimi K2),
                    # otherwise generate ID with correct id_type
                    if tc.id:
                        tool_call_class_items.append(
                            tool_call_class(id=tc.id, function=tc)
                        )
                    else:
                        # Generate ID using the correct format (kimi_k2 or random),
                        # but leave it to the class if it's Mistral to preserve
                        # 9-char IDs
                        if is_mistral_tokenizer(tokenizer):
                            tool_call_class_items.append(tool_call_class(function=tc))
                        else:
                            generated_id = make_tool_call_id(
                                id_type=self.tool_call_id_type,
                                func_name=tc.name,
                                idx=history_tool_call_cnt,
                            )
                            tool_call_class_items.append(
                                tool_call_class(id=generated_id, function=tc)
                            )
                    history_tool_call_cnt += 1
                message = ChatMessage(
                    role=role,
                    reasoning=reasoning,
                    content="",
                    tool_calls=tool_call_class_items,
                )

            elif request.tool_choice and request.tool_choice == "required":
                tool_call_class_items = []
                tool_calls = tool_calls or []
                for idx, tool_call in enumerate(tool_calls):
                    # Use native ID if available,
                    # otherwise generate ID with correct id_type
                    if tool_call.id:
                        tool_call_class_items.append(
                            tool_call_class(id=tool_call.id, function=tool_call)
                        )
                    else:
                        # Generate ID using the correct format (kimi_k2 or random),
                        # but leave it to the class if it's Mistral to preserve
                        # 9-char IDs
                        if is_mistral_tokenizer(tokenizer):
                            tool_call_class_items.append(
                                tool_call_class(function=tool_call)
                            )
                        else:
                            generated_id = make_tool_call_id(
                                id_type=self.tool_call_id_type,
                                func_name=tool_call.name,
                                idx=history_tool_call_cnt,
                            )
                            tool_call_class_items.append(
                                tool_call_class(id=generated_id, function=tool_call)
                            )
                    history_tool_call_cnt += 1
                message = ChatMessage(
                    role=role,
                    content="",
                    tool_calls=tool_call_class_items,
                    reasoning=reasoning,
                )

            # if the request doesn't use tool choice
            # OR specifies to not use a tool
            elif not request.tool_choice or request.tool_choice == "none":
                message = ChatMessage(role=role, reasoning=reasoning, content=content)

            # handle when there are tools and tool choice is auto
            elif (
                request.tools
                and (request.tool_choice == "auto" or request.tool_choice is None)
                and self.enable_auto_tools
                and self.tool_parser
            ):
                # In the OpenAI API the finish_reason is "tools_called"
                # if the tool choice is auto and the model produced a tool
                # call. The same is not true for named function calls
                auto_tools_called = tool_calls is not None and len(tool_calls) > 0
                if tool_calls:
                    tool_call_items = []
                    for idx, tc in enumerate(tool_calls):
                        # Use native ID if available (e.g., Kimi K2),
                        # otherwise generate ID with correct id_type
                        if tc.id:
                            tool_call_items.append(
                                tool_call_class(id=tc.id, function=tc)
                            )
                        else:
                            # Generate ID using the correct format (kimi_k2 or random),
                            # but leave it to the class if it's Mistral to preserve
                            # 9-char IDs
                            if is_mistral_tokenizer(tokenizer):
                                tool_call_items.append(tool_call_class(function=tc))
                            else:
                                generated_id = make_tool_call_id(
                                    id_type=self.tool_call_id_type,
                                    func_name=tc.name,
                                    idx=history_tool_call_cnt,
                                )
                                tool_call_items.append(
                                    tool_call_class(id=generated_id, function=tc)
                                )
                        history_tool_call_cnt += 1
                    message = ChatMessage(
                        role=role,
                        reasoning=reasoning,
                        content=content,
                        tool_calls=tool_call_items,
                    )

                else:
                    # FOR NOW make it a chat message; we will have to detect
                    # the type to make it later.
                    ret_content = content

                    # try to use content return from tool parser first,
                    # tool parser may do some modify for the content.
                    if content and len(content) > 0:
                        ret_content = content
                    message = ChatMessage(
                        role=role,
                        reasoning=reasoning,
                        content=ret_content,
                    )

            # undetermined case that is still important to handle
            else:
                logger.error(
                    "Error in chat_completion_full_generator - cannot determine"
                    " if tools should be extracted. Returning a standard chat "
                    "completion."
                )
                message = ChatMessage(role=role, reasoning=reasoning, content=content)

            # opt22d(R1): 上面这条 if/elif 链里，凡"跳过工具解析器"的分支
            # （tool_choice=="none" / 无 tools / 解析异常兜底）其 content 未经 parser，
            # 里面的"形似工具调用"裸标签不会被转义，会直接下发并可能被下游 harness
            # 误当真实调用执行。此处统一补一次转义。
            # 该转义是幂等的（已转义的 &lt;... 不再匹配），故对"已由 parser 处理过"
            # 的分支重复施加无害，可安全地在此单点收口。
            if message.content:
                # [example 区间 2026-10-18] `<example>`/`<etool_call>` 内是
                # 模型展示的示例，必须原样下发；只转义区间外的形似标签。
                message.content = escape_tool_like_tags_outside_examples(
                    message.content
                )
            # In OpenAI's API, when a tool is called, the finish_reason is:
            # "tool_calls" for "auto" or "required" tool calls,
            # and "stop" for named tool calls.
            is_finish_reason_tool_calls = auto_tools_called or (
                request.tool_choice
                and request.tool_choice == "required"
                and output.finish_reason == "stop"
            )

            # Encode routed_experts for transport. JSON can't carry raw
            # bytes, so we write the ndarray as a ``.npy`` byte stream
            # and base64-encode it. ``pybase64`` is ~3x faster than the
            # stdlib ``base64`` on large payloads thanks to SIMD.
            routed_experts_b64 = None
            if output.routed_experts is not None:
                buf = io.BytesIO()
                np.save(buf, output.routed_experts)
                routed_experts_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

            choice_data = ChatCompletionResponseChoice(
                index=output.index,
                message=message,
                logprobs=logprobs,
                finish_reason="tool_calls"
                if is_finish_reason_tool_calls
                else output.finish_reason
                if output.finish_reason
                else "stop",
                stop_reason=output.stop_reason,
                token_ids=(
                    as_list(output.token_ids) if request.return_token_ids else None
                ),
                routed_experts=routed_experts_b64,
            )
            choice_data = maybe_filter_parallel_tool_calls(choice_data, request)

            choices.append(choice_data)

        if request.echo:
            last_msg_content: str | list[dict[str, str]] = ""
            if (
                conversation
                and "content" in conversation[-1]
                and conversation[-1].get("role") == role
            ):
                last_msg_content = conversation[-1]["content"] or ""
            if isinstance(last_msg_content, list):
                last_msg_content = "\n".join(msg["text"] for msg in last_msg_content)

            for choice in choices:
                full_message = last_msg_content + (choice.message.content or "")
                choice.message.content = full_message

        assert final_res.prompt_token_ids is not None
        num_prompt_tokens = len(final_res.prompt_token_ids)
        if final_res.encoder_prompt_token_ids is not None:
            num_prompt_tokens += len(final_res.encoder_prompt_token_ids)
        num_generated_tokens = sum(
            len(output.token_ids) for output in final_res.outputs
        )
        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )
        # [1Cat P4] Engine-authoritative per-request timing breakdown.
        usage.timings = _usage_timings(getattr(final_res, "metrics", None))
        if self.enable_prompt_tokens_details and final_res.num_cached_tokens:
            usage.prompt_tokens_details = PromptTokenUsageInfo(
                cached_tokens=final_res.num_cached_tokens
            )
        full_reasoning_tokens = _count_reasoning_tokens(
            reasoning_parser, [output.token_ids for output in final_res.outputs]
        )
        if full_reasoning_tokens is not None:
            usage.completion_tokens_details = CompletionTokenUsageInfo(
                reasoning_tokens=full_reasoning_tokens
            )

        request_metadata.final_usage_info = usage

        # ``final_res.prompt`` is the rendered chat-templated prompt text
        prompt_text = final_res.prompt if request.return_prompt_text else None

        response = ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
            system_fingerprint=self.system_fingerprint,
            prompt_logprobs=clamp_prompt_logprobs(final_res.prompt_logprobs),
            prompt_token_ids=(
                final_res.prompt_token_ids if request.return_token_ids else None
            ),
            prompt_text=prompt_text,
            kv_transfer_params=final_res.kv_transfer_params,
        )

        # Log complete response if output logging is enabled
        if self.enable_log_outputs and self.request_logger:
            for choice in choices:
                output_text = ""
                if choice.message.content:
                    output_text = choice.message.content
                elif choice.message.tool_calls:
                    # For tool calls, log the function name and arguments
                    tool_call_descriptions = []
                    for tc in choice.message.tool_calls:  # type: ignore
                        function_call: FunctionCall = tc.function  # type: ignore
                        tool_call_descriptions.append(
                            f"{function_call.name}({function_call.arguments})"
                        )
                    tool_calls_str = ", ".join(tool_call_descriptions)
                    output_text = f"[tool_calls: {tool_calls_str}]"

                if output_text:
                    # Get the corresponding output token IDs
                    output_token_ids = None
                    if choice.index < len(final_res.outputs):
                        output_token_ids = final_res.outputs[choice.index].token_ids

                    self.request_logger.log_outputs(
                        request_id=request_id,
                        outputs=output_text,
                        output_token_ids=output_token_ids,
                        finish_reason=choice.finish_reason,
                        is_streaming=False,
                        delta=False,
                    )

        return response

    def _get_top_logprobs(
        self,
        logprobs: dict[int, Logprob],
        top_logprobs: int | None,
        tokenizer: TokenizerLike | None,
        should_return_as_token_id: bool,
    ) -> list[ChatCompletionLogProb]:
        return [
            ChatCompletionLogProb(
                token=(
                    token := self._get_decoded_token(
                        p[1],
                        p[0],
                        tokenizer,
                        return_as_token_id=should_return_as_token_id,
                    )
                ),
                logprob=max(p[1].logprob, -9999.0),
                bytes=list(token.encode("utf-8", errors="replace")),
            )
            for i, p in enumerate(logprobs.items())
            if (top_logprobs and i < top_logprobs or top_logprobs == -1)
        ]

    def _create_chat_logprobs(
        self,
        token_ids: GenericSequence[int],
        top_logprobs: GenericSequence[dict[int, Logprob] | None],
        tokenizer: TokenizerLike | None,
        num_output_top_logprobs: int | None = None,
        return_as_token_id: bool | None = None,
    ) -> ChatCompletionLogProbs:
        """Create OpenAI-style logprobs."""
        logprobs_content: list[ChatCompletionLogProbsContent] = []

        should_return_as_token_id = (
            return_as_token_id
            if return_as_token_id is not None
            else self.return_tokens_as_token_ids
        )
        for i, token_id in enumerate(token_ids):
            step_top_logprobs = top_logprobs[i]
            if step_top_logprobs is None or step_top_logprobs.get(token_id) is None:
                if should_return_as_token_id:
                    token = f"token_id:{token_id}"
                else:
                    if tokenizer is None:
                        raise ValueError(
                            "Unable to get tokenizer because `skip_tokenizer_init=True`"
                        )

                    token = tokenizer.decode(token_id)

                logprobs_content.append(
                    ChatCompletionLogProbsContent(
                        token=token,
                        bytes=list(token.encode("utf-8", errors="replace")),
                    )
                )
            else:
                step_token = step_top_logprobs[token_id]
                step_decoded = step_token.decoded_token

                logprobs_content.append(
                    ChatCompletionLogProbsContent(
                        token=self._get_decoded_token(
                            step_token,
                            token_id,
                            tokenizer,
                            should_return_as_token_id,
                        ),
                        logprob=max(step_token.logprob, -9999.0),
                        bytes=(
                            None
                            if step_decoded is None
                            else list(step_decoded.encode("utf-8", errors="replace"))
                        ),
                        top_logprobs=self._get_top_logprobs(
                            step_top_logprobs,
                            num_output_top_logprobs,
                            tokenizer,
                            should_return_as_token_id,
                        ),
                    )
                )

        return ChatCompletionLogProbs(content=logprobs_content)

    def _should_stream_with_auto_tool_parsing(self, request: ChatCompletionRequest):
        """
        Utility function to check if streamed tokens should go through the tool
        call parser that was configured.

        We only want to do this IF user-provided tools are set, a tool parser
        is configured, "auto" tool choice is enabled, and the request's tool
        choice field indicates that "auto" tool choice should be used.
        """
        return (
            request.tools
            and self.tool_parser
            and self.enable_auto_tools
            and request.tool_choice in ["auto", None]
        )

    def _should_check_for_unstreamed_tool_arg_tokens(
        self,
        delta_message: DeltaMessage | None,
        output: CompletionOutput,
    ) -> bool:
        """
        Check to see if we should check for unstreamed tool arguments tokens.
        This is only applicable when auto tool parsing is enabled, the delta
        is a tool call with arguments.
        """

        return bool(
            # if there is a delta message that includes tool calls which
            # include a function that has arguments
            output.finish_reason is not None
            and self.enable_auto_tools
            and self.tool_parser
            and delta_message
            and delta_message.tool_calls
            and delta_message.tool_calls[0]
            and delta_message.tool_calls[0].function
            and delta_message.tool_calls[0].function.arguments is not None
        )

    @staticmethod
    def _coverage_end(actual_call: str, expected_call: str) -> int | None:
        """估算 actual_call 覆盖到 expected_call 的哪个位置；无法可靠判定返回 None。

        [本地 2026-10-12] 收尾补发的定位辅助。actual 通常是 expected 的前缀
        （流式逐步下发），但也可能是它的**近似前缀**：已知 partial 路径会在
        参数值末尾多发一个 ``\\n``（a[6895:6897]='\\n'），其尾部又被截断。
        此时 actual 不是 expected 的子串，`str.replace` / `startswith` 都失效。

        办法：拿 actual 尾部的一段作探针，在 expected 中定位它，探针的结束
        位置即覆盖终点。取多个探针长度以兼顾稳健性，并用「终点与 actual 长度
        之差有限」约束排除误匹配；全部失败则返回 None（调用方保守不补）。
        """
        n = len(actual_call)
        if n == 0:
            return 0
        tol = max(16, n // 100)
        for probe in (128, 96, 64, 32, 16):
            if n < probe:
                continue
            pos = expected_call.rfind(actual_call[-probe:])
            if pos < 0:
                continue
            end = pos + probe
            if end <= len(expected_call) and abs(end - n) <= tol:
                return end
        return None

    @staticmethod
    def _remaining_tool_args(expected_call: str, actual_call: str) -> str:
        """返回仍需下发的参数尾部。

        [本地 2026-10-12] 替换原先的
        ``remaining_call = expected_call.replace(actual_call, "", 1)``。

        旧写法有两个缺陷：
        1. 当 actual 不是 expected 的子串（分歧：actual 多发字符且尾部被截断）
           时 `replace` **匹配不到** ⇒ 返回 expected **全文** ⇒ 客户端已收到
           actual 后又收一遍完整参数 ⇒ arguments 重复（实测该次 35274 字节，
           JSON 在 17611 处报错、开头片段出现两次）⇒ harness 报「Edit was
           called with input that could not be parsed as JSON」。
        2. 即使匹配上，若 actual 出现在 expected **中间**（非前缀），留下的也是
           畸形拼接（如 ``{"a":,"b":"XX"}``）。

        现改为：真前缀 ⇒ 补尾部；近似前缀 ⇒ 用覆盖终点补尾部；判定不出
        （例如 2026-10-05 那类「actual 已完整、expected 是 trim 版」，探针在
        expected 中找不到）⇒ 不补，避免重复。

        [opt22g 2026-10-14] **静默劣化缺口修复**（评估清单一.1）：上述
        「判定不出 ⇒ 不补」在 partial 多发值尾场景 = **既不重复、也不补全**
        （固件 real_incidents 起②：exp=182B / act=139B，act 头 136B 是真前缀、
        尾部多发 ``\\n"``，探针 rfind 全 -1 ⇒ remaining 空 ⇒ 缺 43B 永不补，
        客户端拿到截断 JSON）。新增**前缀对齐兜底**：
          ① 二分找最大 k 使 ``act[:k]`` 是 exp 真前缀（零猜测、O(n log n)）；
          ② 仅当 k 足够接近尾部（``k ≥ len(act) - tol``，tol 与探针同款）才可信；
          ③ **语义重叠跳位**：若多发段（``act[k:]`` rstrip 后）的尾字符与待发段
             （``exp[k:]`` lstrip 后）的首字符相同且都是 JSON 结构符（``" } ] : ,``）
             —— 即客户端已收过这个闭合符 —— 则跳过 exp 的这一个字符再补。
             固件起②：act 尾 ``\\n"`` 的 ``"`` 与 exp[k] 的 ``"`` 重叠 ⇒ 跳 1 位
             ⇒ joined=184B 合法（原本 43B 永缺）。
          ④ 重叠不成立时按 exp[k:] 全量补（值尾多发 ``\\n`` 留在字符串值内合法，
             固件起① write 案 780B 合法验证）。
        探针路径保留在最后（优先级最低），全部失败仍保守不补。
        """
        if actual_call and expected_call.startswith(actual_call):
            return expected_call[len(actual_call):]
        n = len(actual_call)
        if n == 0:
            return expected_call
        tol = max(16, n // 100)
        lo, hi = 0, n - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if expected_call.startswith(actual_call[:mid]):
                lo = mid
            else:
                hi = mid - 1
        k = lo
        # 收紧：前缀对齐只接受「尾部多发段很短」的形态（partial 值尾多发
        # 实测 1~2B，4B 为宽裕上界）。中间错位形态（值内多 \n 使后续全部
        # 错位）的 act[k:] 长达 13B，会被对齐误判 ⇒ 错位补发 ⇒ 必须拒绝
        # （t_remaining_guard 用例 4 判别力抓到）。
        if k >= n - tol and k > 0 and (n - k) <= 4:
            overflow = actual_call[k:].rstrip()
            head_raw = expected_call[k:]
            head = head_raw.lstrip()
            if (overflow and head
                    and overflow[-1] == head[0]
                    and overflow[-1] in '"}]:,'):
                skip = (len(head_raw) - len(head)) + 1
                return expected_call[k + skip:]
            return expected_call[k:]
        end = OpenAIServingChat._coverage_end(actual_call, expected_call)
        if end is None:
            return ""
        return expected_call[end:]

    @staticmethod
    def _advance_streamed_args(
        tool_parser, index: int, expected_call: str
    ) -> None:
        """补发收尾参数后，把「已下发」推进到完整值（幂等保护）。

        [本地 2026-10-12] 上游本块只发 delta、**不更新任何状态**，而补发的
        触发条件只要求 ``output.finish_reason is not None``——finish_reason 在
        多个 output 上出现时本块会被反复进入，``streamed_args_for_tool`` 却
        不变 ⇒ 同一段 remaining 被反复补发。线上实录：同一个 13488 B 的完整
        arguments 等距重发 **5 次**，客户端拼出 62219 B，JSON 只能解析前
        13488 B。推进后第二次调用 ``_remaining_tool_args`` 即返回空串。
        """
        try:
            if index < len(tool_parser.streamed_args_for_tool):
                tool_parser.streamed_args_for_tool[index] = expected_call
        except Exception:
            pass

    @staticmethod
    def _create_remaining_args_delta(
        delta_message: DeltaMessage,
        remaining_call: str,
        index: int,
    ) -> DeltaMessage:
        """
        Create a delta message for remaining tool arguments, preserving
        id/type/name from the original delta.
        """
        original_tc = next(
            (tc for tc in delta_message.tool_calls if tc.index == index),
            None,
        )
        original_fn = original_tc.function if original_tc else None
        return DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=index,
                    id=original_tc.id if original_tc else None,
                    type=original_tc.type if original_tc else None,
                    function=DeltaFunctionCall(
                        name=original_fn.name if original_fn else None,
                        arguments=remaining_call,
                    ),
                )
            ]
        )
