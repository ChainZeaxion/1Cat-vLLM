# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser
from vllm.tool_parsers.utils import (
    escape_tool_like_tags,
    split_incomplete_tool_tag_tail,
)

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.tokenizers import TokenizerLike


class Qwen3ReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for the Qwen3/Qwen3.5 model family.

    The Qwen3 model family uses <think>...</think> tokens to denote reasoning
    text. Starting with Qwen3.5, the chat template places <think> in the
    prompt so only </think> appears in the generated output. The model
    provides a strict switch to disable reasoning output via the
    'enable_thinking=False' parameter.

    When thinking is disabled, the template places <think>\\n\\n</think>\\n\\n
    in the prompt. The serving layer detects this via prompt_is_reasoning_end
    and routes deltas as content without calling the streaming parser.

    NOTE: Models up to the 2507 release (e.g., Qwen/Qwen3-235B-A22B-Instruct-2507)
    use an older chat template where the model generates <think> itself.
    This parser handles both styles: if <think> appears in the generated output
    it is stripped before extraction (non-streaming) or skipped (streaming).

    NOTE: Qwen3.5 models may emit <tool_call> inside the thinking block
    without closing </think> first. <tool_call> is treated as an implicit
    end of reasoning, matching the approach in KimiK2ReasoningParser.
    """

    def __init__(self, tokenizer: "TokenizerLike", *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        # Qwen3 defaults to thinking enabled; only treat output as
        # pure content when the user explicitly disables it.
        self.thinking_enabled = chat_kwargs.get("enable_thinking", True)

        self._tool_call_tag = "<tool_call>"
        self._tool_call_token_id = self.vocab.get(self._tool_call_tag)
        self._tool_call_end_tag = "</tool_call>"
        self._tool_call_end_token_id = self.vocab.get(self._tool_call_end_tag)
        # opt22d 第七轮：流式推理文本转义时，用于暂存“未闭合的工具标签残片”。
        self._esc_hold = ""

    @property
    def start_token(self) -> str:
        """The token that starts reasoning content."""
        return "<think>"

    @property
    def end_token(self) -> str:
        """The token that ends reasoning content."""
        return "</think>"

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        """判定推理是否已结束。**注意：本方法也用于检查 PROMPT（用户消息/历史）**，
        见 ``parser/abstract_parser.py`` 的 ``prompt_is_reasoning_checked`` 分支。

        opt22d 第九轮：``<tool_call>`` 分支加 ``not self.thinking_enabled`` 门控。
        该分支本意是识别"prompt 里已给出工具调用 ⇒ 模板已结束思考"，但**未配对的**
        ``<tool_call>`` 更多来自用户消息本身——例如用户要求"展示一个工具调用格式示例"
        时，其消息里必然含该标签。此时若判为「推理已结束」，
        ``DelegatingParser`` 会**跳过整个 reasoning 相位**、把所有输出交给工具解析器，
        导致**正文被吞**（实测：prompt 含未配对 ``<tool_call>`` 时，非流式 content 15 字符、
        流式 **0** 字符；completion_tokens 两者相同 ⇒ 生成完好、纯属解析路径丢弃）。

        与第七轮 ``is_reasoning_end_streaming`` / ``extract_reasoning_streaming``
        的门控口径保持一致：开启思考时，标签不充当推理边界信号。
        """
        start_token_id = self.start_token_id
        end_token_id = self.end_token_id
        tool_call_token_id = self._tool_call_token_id
        tool_call_end_token_id = self._tool_call_end_token_id

        for i in range(len(input_ids) - 1, -1, -1):
            token_id = input_ids[i]
            if token_id == start_token_id:
                # Found <think> before </think> or <tool_call>
                return False
            if token_id == end_token_id:
                return True
            if (
                tool_call_token_id is not None
                and not self.thinking_enabled
                and token_id == tool_call_token_id
            ):
                # Only treat as implicit reasoning end if this <tool_call>
                # is NOT followed by </tool_call>.  Paired occurrences are
                # template examples in the prompt, not model output.
                if tool_call_end_token_id is not None and any(
                    input_ids[j] == tool_call_end_token_id
                    for j in range(i + 1, len(input_ids))
                ):
                    continue
                return True
        return False

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        if super().is_reasoning_end_streaming(input_ids, delta_ids):
            # 真实 </think> 到达——这才是开启思考时唯一可信的推理边界。
            return True
        # opt22d 第七轮：开启思考时，<tool_call> 不再充当「隐式推理结束」信号。
        # 否则 thinking 里写出的调用样例会在此处切断推理——标签及其后内容被当
        # 正文送进工具解析器，误触发真实调用。
        # 实测（2026-09-18，引擎 :8000）：流式 2/2 变体复现，reasoning 被截断
        # 且 tool_calls 非空（['Write'] / ['Read','Read','Read']）；非流式因
        # extract_reasoning 优先按 </think> 切分而未复现。
        # 关闭思考时保留该兜底，与历史行为一致。
        # 「模型漏写 </think> 却真实调工具」的场景改由 serving 层 finish 时的
        # 完整文本重解析兜底（届时 "是否出现过 </think>" 可准确判定）。
        if self._tool_call_token_id is not None and not self.thinking_enabled:
            return self._tool_call_token_id in delta_ids
        return False

    def _esc_reasoning(self, text: str) -> str:
        """opt22d 第七轮：把推理文本里“形似工具调用”的标签转义为显示安全形式。

        动机：reasoning 此前无任何转义收口（opt22d R1 只覆盖 content），模型在
        思考里写出的标签会以裸形式下发，可能被下游 harness 误当真实调用执行。

        流式下标签会被切开（``<p`` + ``arameter=file_path>``），若在片段粒度上
        转义会漏出裸 ``<`` 或产出 ``⟨'/function>`` 这类畸形；故复用 opt22d L-E
        的暂存机制：未闭合尾部留在 ``_esc_hold``，等 ``>`` 到齐后整体转义。
        因入参是增量文本且暂存部分不返回，拼接后不会重复发出。

        与 content 侧取舍一致：EOS 时仍留在 ``_esc_hold`` 的残片不再补发——
        残片仅一字符量级，且推理文本的转义不承载语义。
        """
        if not text:
            return text
        head, self._esc_hold = split_incomplete_tool_tag_tail(self._esc_hold + text)
        return escape_tool_like_tags(head)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """
        Extract content token ids from the input_ids.
        """
        result = super().extract_content_ids(input_ids)
        if result:
            return result
        # Fall back: content starts at <tool_call> (implicit reasoning end).
        if (
            self._tool_call_token_id is not None
            and self._tool_call_token_id in input_ids
        ):
            tool_call_index = (
                len(input_ids) - 1 - input_ids[::-1].index(self._tool_call_token_id)
            )
            return input_ids[tool_call_index:]
        return []

    def extract_reasoning(
        self, model_output: str, request: "ChatCompletionRequest | ResponsesRequest"
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning content from the model output.

        The <think> token is placed in the prompt by the chat template,
        so typically only </think> appears in the generated output.
        If <think> is present (e.g. from a different template), it is
        stripped before extraction.

        When thinking is explicitly disabled and no </think> appears,
        returns (None, model_output) — all output is content.
        Otherwise (thinking enabled, default), a missing </think> means
        the output was truncated and everything is reasoning:
        returns (model_output, None).

        Returns:
            tuple[Optional[str], Optional[str]]: reasoning content and content
        """

        # Strip <think> if present in the generated output.
        model_output_parts = model_output.partition(self.start_token)
        model_output = (
            model_output_parts[2] if model_output_parts[1] else model_output_parts[0]
        )

        if self.end_token in model_output:
            reasoning, _, content = model_output.partition(self.end_token)
            return reasoning, content or None

        if not self.thinking_enabled:
            # Thinking explicitly disabled — treat everything as content.
            return None, model_output

        # No </think> — check for implicit reasoning end via <tool_call>.
        tool_call_index = model_output.find(self._tool_call_tag)
        if tool_call_index != -1:
            reasoning = model_output[:tool_call_index]
            content = model_output[tool_call_index:]
            return reasoning or None, content or None
        # Thinking enabled but no </think>: output was truncated.
        # Everything generated so far is reasoning.
        return model_output, None

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """
        Extract reasoning content from a streaming delta.

        Since <think> is placed in the prompt by the chat template, all
        generated tokens before </think> are reasoning and tokens after
        are content.

        NOTE: When thinking is disabled, no think tokens appear in the
        generated output. The serving layer detects this via
        prompt_is_reasoning_end and routes deltas as content without
        calling this method.
        """
        # Strip <think> from delta if present (old template / edge case
        # where the model generates <think> itself).
        if self.start_token_id in delta_token_ids:
            start_idx = delta_text.find(self.start_token)
            if start_idx >= 0:
                delta_text = delta_text[start_idx + len(self.start_token) :]

        if self.end_token_id in delta_token_ids:
            # End token in this delta: split reasoning from content.
            end_index = delta_text.find(self.end_token)
            if end_index >= 0:
                reasoning = delta_text[:end_index]
                content = delta_text[end_index + len(self.end_token) :]
                if not reasoning and not content:
                    return None
                return DeltaMessage(
                    reasoning=self._esc_reasoning(reasoning) if reasoning else None,
                    content=content if content else None,
                )
            # end_token_id in IDs but not in text (already stripped)
            return None

        # Implicit reasoning end via <tool_call>.
        # opt22d 第七轮：仅关闭思考时启用——开启时 <tool_call> 不结束推理
        # （理由与实测见 is_reasoning_end_streaming）。
        if (
            self._tool_call_token_id is not None
            and not self.thinking_enabled
            and self._tool_call_token_id in delta_token_ids
        ):
            tool_index = delta_text.find(self._tool_call_tag)
            if tool_index >= 0:
                reasoning = delta_text[:tool_index]
                content = delta_text[tool_index:]
                return DeltaMessage(
                    reasoning=self._esc_reasoning(reasoning) if reasoning else None,
                    content=content if content else None,
                )

        # No end token in this delta.
        if not delta_text:
            # Nothing left after stripping start token.
            return None
        elif self.end_token_id in previous_token_ids:
            # End token already passed: everything is content now.
            return DeltaMessage(content=delta_text)
        elif (
            self._tool_call_token_id is not None
            and not self.thinking_enabled
            and self._tool_call_token_id in previous_token_ids
        ):
            # opt22d 第七轮：同样须门控。否则上一步被跳过的 <tool_call> 会在此
            # 生效（token 已落入 previous_token_ids），把后续思考文本误判为正文。
            return DeltaMessage(content=delta_text)
        else:
            # No end token yet: still in reasoning phase.
            return DeltaMessage(reasoning=self._esc_reasoning(delta_text))
