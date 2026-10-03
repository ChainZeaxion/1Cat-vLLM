# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import time
import uuid
from collections.abc import Sequence
from typing import Any

import regex as re

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.envs import (
    VLLM_ENFORCE_STRICT_TOOL_CALLING,
    VLLM_QWEN3X_TOOL_FIX,
)
from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import (
    Tool,
    ToolParser,
)
from vllm.tool_parsers.structural_tag_registry import (
    get_enable_structured_outputs_in_reasoning,
    get_model_structural_tag,
)
from vllm.tool_parsers.utils import (
    coerce_to_schema_type,
    extract_types_from_schema,
    find_common_prefix,
    find_tool_properties,
    find_tool_required,
    request_forbids_tool_calls,
    split_incomplete_tool_tag_tail,
)

logger = init_logger(__name__)

def _normalize_qwen3_tag_names(text: str) -> str:
    """opt22: map Anthropic-style tool-call tag names to the equals form
    this parser's regexes expect (toolcall->tool_call, tool name="X"->function=X,
    param name="X"->parameter=X, closing tool/param->function/parameter). Native
    and unrelated text are left untouched.
    """
    if not text:
        return text
    e = re.sub(r'<tool name="([^"]*)">', r'<function=\1>', text)
    e = re.sub(r'<param name="([^"]*)">', r'<parameter=\1>', e)
    e = re.sub(r'<toolcall>', '<tool_call>', e)
    e = re.sub(r'</toolcall>', '</tool_call>', e)
    e = re.sub(r'</tool>', '</function>', e)
    e = re.sub(r'</param>', '</parameter>', e)
    return e


# opt22d: 预编译“形似工具调用”标签匹配（模块级；含无右尖括号的截断残片分支）
_TOOL_LIKE_TAG_RE = re.compile(
    r"</?(?:tool_call|toolcall|tool|function|parameter|param)\b[^<>]*?/?>"
    r"|</?(?:tool_call|toolcall|tool|function|parameter|param)\b[^<>]*$"
)

# opt22e：遮罩的**等长哨兵**。把工具类标签的 `<` / `>` 换成私有区单字符，
# 而非 `⟨'` / `'⟩`（2 字符）。
# **为何必须等长**：遮罩文本既用于「结构扫描」也用于「参数值切片」，而 partial
# 流式路径靠 `_partial_emit_offset` 在遮罩文本上做**跨帧偏移累计**。一旦遮罩改变
# 长度（或围栏开合导致已遮罩段回落），偏移即失配 ⇒ 重复/漏发（实测线上 args 翻倍、
# `Extra call` 内容错位）。等长后：遮罩坐标 ≡ 原文坐标，偏移天然稳定。
# 展示形（`⟨'…'⟩`）只在**最终输出**时由 `_to_display` 转换。
_SENT_LT = "\ue000"
_SENT_GT = "\ue001"


# opt22d 第七轮：抓取“已闭合”的函数名标签（归一化后形如 <function=NAME>）。
# 用于判断本轮出现的工具是否落在需要流式增量显示的白名单内。
# 要求右尖括号：分片中的 <function=Wri 不算已确名的工具，避免误判。
_FN_TAG_RE = re.compile(r"<function=([A-Za-z0-9_.\-]+)>")

# opt22d 第八轮（L-D for coder）：裸标签的行首约束。
# 同时匹配 <tool_call> 与 <function=（`>?` 兼容分片中的未闭合形式；
# 闭合与否都要整块纳入匹配，否则 `>` 会被留在原文里、产出 `⟨'function=X>` 畸形）。
_LD_TAG_RE = re.compile(r"<tool_call>|<function=[A-Za-z0-9_.\-]*>?")


# opt22e A：结构标签（闭合/开标签）——判"是否出现正文"时须先剔除，
# 否则 `</function>` / `</tool_call>` 这类**空结构**会被当成正文
# （实测：`\n<parameter=command>…</parameter>\n</function>\n` 的尾段
# `\n</function>\n` 被误判为正文 ⇒ 同轮多调用被误伤）。
_STRUCT_TAG_RE = re.compile(r"</?(?:function|tool_?call|parameter)\b[^<>]*>")


def _strip_struct_tags(text: str) -> str:
    """剔除结构标签，仅保留可能的正文。"""
    return _STRUCT_TAG_RE.sub("", text)


def _has_toplevel_prose(seg: str, depth: int) -> bool:
    """opt22e A：``seg`` 中是否存在**参数体外**（顶层）的非空白文本。

    必须逐段跟踪深度 —— 直接把整段 ``seg`` 判为"顶层文本"是错的：两次
    ``<tool_call>``/``<function=`` 匹配之间往往横跨整个参数体
    （``\n<parameter=command>\necho one\n</parameter>\n</function>``），
    那属于**参数值内容**，不是顶层正文。
    ⚠️ 踩坑：首版用「``_seg.strip()`` 非空」判定，导致**同轮多调用**被误伤 ——
    第二个调用的 ``_seg`` 含前一个调用的参数体 ⇒ 误判为"已出现正文" ⇒
    后续调用被转义、**丢掉真实调用**（实测 A2/A4 双调用只剩 1 个）。
    """
    d = depth
    k = 0
    for m in re.finditer(r"<parameter=|</parameter>", seg):
        if d == 0 and _strip_struct_tags(seg[k : m.start()]).strip():
            return True
        d += 1 if m.group(0).startswith("<parameter=") else -1
        if d < 0:
            d = 0
        k = m.end()
    return d == 0 and bool(_strip_struct_tags(seg[k:]).strip())


def _escape_non_linestart_tool_calls(text: str, *, streaming: bool = True) -> str:
    """opt22d 第八/九轮（L-D）：**双重必要条件**收紧后，把其余裸标签一律转义。

    真调用需**同时满足**：结构特征 + 位置特征。二者缺一即为展示性文本，
    转义后并入 content（用户仍可见，但下游不执行）。

    | 标签 | 位置特征 | 结构特征 |
    |---|---|---|
    | ``<tool_call>`` | 行首 | **后跟 `<function=`**（允许中间空白） |
    | 独立 ``<function=`` | 行首 | **有 `</function>` 闭合** |
    | 被 ``<tool_call>`` 包裹的 ``<function=`` | —（继承包裹的位置） | — |

    **依据（2026-09-18 实测）**：
      - chat template 的三种注入写法均为 ``\\n\\n<tool_call>\\n<function=``（行首 + 紧跟 function）；
      - 8 条真实调用提示词 → 标签 **8/8 行首**；
      - **最强同行诱导**（明确要求"说明文字与调用之间不要换行"）3 例 × 5 次
        → 真实调用 **15/15 成功**：模型受模板格式支配，即便用户要求紧凑也自动换行。

    **为何必须同时覆盖 `<function=`**：``_get_function_calls`` 有退避分支——
    找不到 ``<tool_call>`` 时把**整段输出**当候选，再用
    ``tool_call_function_regex`` 抽 ``<function=``。故围栏外的裸
    ``<function=Write>`` 同样会产出（参数为空的）调用（实测 E1 曾产出 3 个空 ``Write``）。

    **为何 `<tool_call>` 也影响正文**：``tool_call_regex`` 含 ``<tool_call>(.*?)$``
    分支（匹配到文本结尾），故**孤立、未闭合**的 ``<tool_call>`` 也会让 impl 进入
    调用解析模式、**停止输出正文**（实测：模型只输出一行 ``<tool_call>`` 时，
    非流式正常转义输出 13 字符，**流式 0 字符**）。

    展示性示例常写成行内代码或表格（``| `<tool_call>` |``），其 ``<`` 前有 ``|``
    与反引号 ⇒ 非行首 ⇒ 降级。

    Args:
        text: 待处理文本。
        streaming: 流式下标签可能尚未写全（判据未到齐），此时**暂存**该尾部
            （从输出中摘除、保持前缀关系），待判据到齐后再决定；非流式下
            全文已知，直接判定。
    """
    if not text:
        return text
    out: list[str] = []
    last = 0
    wrapper = False  # 前一个被放行的开标签是“行首的 <tool_call>”
    # opt22e：**参数体深度**。>0 表示当前处于某个 `<parameter=…>` 内部 —— 那里
    # 的内容是**文档正文**，其中的 `<function=` / `<tool_call>` 是示例，一律按
    # 展示文本转义。
    # 为何需要：本函数原判据「行首 + 有 `</function>` ⇒ 真实调用」在**参数体内
    # 同样成立** ⇒ 文档里原样写的示例（行首的 `<function=Bash>…</function>`）
    # 被放行成真实调用，产出**空参数的幽灵调用**（实测线上 1/6～3/6 采样出现
    # `{}` 参数的 Bash 调用）。参数体内不存在真实调用结构，故深度 >0 时禁用该豁免。
    _depth = 0
    _cursor = 0
    # opt22e A：**上下文判据**。「已豁免过一个真实调用」+「其后顶层出现过正文」
    # ⇒ 后续候选一律转义。
    # 为何需要：模型会先给出真实调用、**再**在正文里写「格式完全正确」的示例
    # （实测：`<tool_call><function=Write>…</tool_call>` 后接正文，再接一个
    # `<tool_call><function=Bash><parameter=command>ls -la</parameter>…`）。
    # 该示例在参数体**外**、行首、有闭合 ⇒ 全部豁免条件都满足 ⇒ 被当成第二个
    # 真实调用，**参数非空 ⇒ 下游真的会执行**（不再是"空参数失败"这种轻后果）。
    # 真调用的多个调用是**连续**输出的，中间不会夹正文；故「出现正文后」可判。
    # 边界（已接受）：模型「调用 → 说明 → 再调用」的真实场景会被误伤（罕见），
    # 而「先说明后调用」（E4）不受影响 —— 正文在**首个**调用之前，`_seen_call` 仍为假。
    _seen_call = False    # 已豁免过至少一个真实调用
    _seen_prose = False   # 其后顶层（参数体外）出现过非空白正文
    for m in _LD_TAG_RE.finditer(text):
        i = m.start()
        tok = m.group(0)
        _seg = text[_cursor:i]
        # ⚠️ 两个必要条件：
        # ① 必须在**已豁免过真实调用之后**才置位 —— 否则「先说明后调用」（E4）
        #    的开头说明会立刻置位，导致其后**所有**真实调用被转义。
        # ② 用 `_has_toplevel_prose` 逐段跟踪深度 —— `_seg` 往往横跨整个参数体，
        #    直接看整体会把它误判为顶层正文（会丢掉多调用中的后续调用）。
        if _seen_call and _has_toplevel_prose(_seg, _depth):
            _seen_prose = True
        _depth += _seg.count("<parameter=") - _seg.count("</parameter>")
        _cursor = i
        line_start = text.rfind("\n", 0, i) + 1
        at_line_start = text[line_start:i].strip() == ""
        rest = text[m.end():]
        # 参数体内、或「已有真实调用 + 其后出现正文」⇒ 不豁免。
        _in_body = _depth > 0 or (_seen_call and _seen_prose)
        if tok.startswith("<tool_call"):
            if at_line_start and not _in_body:
                if streaming and rest.strip() == "":
                    # 后续尚未到达，无法判定是否跟 <function= —— 暂存（自限：
                    # 一旦出现任何非空白字符即重新判定，不会长期扣留）。
                    return "".join(out) + text[last:i]
                if rest.lstrip().startswith("<function="):
                    wrapper = True
                    _seen_call = True
                    continue  # 双重条件满足 → 真实调用起点
                wrapper = False
            else:
                wrapper = False
        else:  # <function=
            if wrapper:
                _seen_call = True
                continue  # 被行首 <tool_call> 包裹 → 继承其真实调用身份
            if at_line_start and not _in_body:
                if "</function>" in rest:
                    _seen_call = True
                    continue  # 行首 + 有闭合 → 真实调用（退避语法）
                if streaming and rest.strip() == "":
                    return "".join(out) + text[last:i]  # 判据未到齐 → 暂存
                # 行首但无闭合 → 落空，转义
        out.append(text[last:i])
        out.append(tok.replace("<", _SENT_LT).replace(">", _SENT_GT))
        last = m.end()
    if not out:
        return text
    out.append(text[last:])
    return "".join(out)


def _escape_tool_like_tags(text: str) -> str:
    """opt22d L-B: 把文本里“形似工具调用”的裸标签换成显示安全的形式，
    使下游 harness 不再把这些文本当成真实调用执行。

    转义形如 `<function=Write>` -> `⟨'function=Write'⟩`（数学尖括号 + 成对单引号）。
    不用 `&lt;`/`&gt;` 实体的原因：① 纯文本渲染下很难看；② 若下游做 HTML 解码
    会被还原成 `<` 而失效。不用全角 `＜＞` 的原因：NFKC 会映射回 ASCII `<`/`>`。

    仅命中工具调用词族的标签名；a<b、20<3、<p> 等普通文本不受影响。

    opt22e：内部使用**等长哨兵**（见 `_SENT_LT`），不再直接用 `⟨'…'⟩`；
    展示形由 `_to_display` 在最终输出时统一转换。"""
    if not text:
        return text
    return _TOOL_LIKE_TAG_RE.sub(
        lambda m: m.group(0).replace("<", _SENT_LT).replace(">", _SENT_GT), text
    )


def _to_display(text: str) -> str:
    """opt22e：内部形态 → **用户可见的展示形**（`⟨'…'⟩`）。

    先把哨兵还原成展示形，再对**残留的真实工具标签**做展示转义
    （部分路径（如未注册调用降级）直接把原始文本转内容，未经遮罩）。
    """
    if not text:
        return text
    t = text.replace(_SENT_LT, "⟨'").replace(_SENT_GT, "'⟩")
    return _TOOL_LIKE_TAG_RE.sub(
        lambda m: m.group(0).replace("<", "⟨'").replace(">", "'⟩"), t
    )


def _to_original(text: str) -> str:
    """opt22e：内部形态 → **模型原文**（参数值专用）。

    因为哨兵**等长**，遮罩文本上的切片与原文逐字对应，仅哨兵需换回 `<`/`>`。
    参数值必须是数据本身：实测若把展示形当值，Write 写出的文件内容会变成
    `⟨'tool_call'⟩`（模型原文是 `<tool_call>`）；xml 侧同场景保留原文。
    """
    if not text or _SENT_LT not in text and _SENT_GT not in text:
        return text
    return text.replace(_SENT_LT, "<").replace(_SENT_GT, ">")


# opt22e: **真收尾三元组** —— `</parameter></function></tool_call>`（三方言）。
# 关键：它必须**结束整段文本**。文档示例的收尾（`</parameter></function>`，多为两层）
# 后面还跟着围栏 ``` 或正文，故不匹配；而真收尾恰好结束整段文本。
# ⚠️ 踩坑：曾用「后跟 `</function>` 即算结构性」，但**示例的收尾同样满足**
# ⇒ 正常成对围栏被改坏（实测 content 89→46）。
_TRAILING_CLOSE_RE = re.compile(
    r"</parameter>\s*</function>\s*</tool_?call\b[^<>]*>\s*$"
)


def _trailing_close_start(text: str) -> int | None:
    """若 `text` 以**真收尾三元组**结束，返回其 `</parameter>` 的起始位置。"""
    m = _TRAILING_CLOSE_RE.search(text)
    return m.start() if m else None


def _fenced_ranges(text: str) -> list:
    """opt22d（L-C）：返回 text 中“markdown 代码围栏”覆盖的字符区间列表。

    只认 **行首** 的三连反引号（CommonMark 语义）——行内出现的反引号不计入，
    避免正文提及反引号时误开围栏。与 qwen3xml 侧 `_scan_code_fence` 判定一致。
    """
    ranges = []
    in_fence = False
    fence_start = 0
    line_start = True
    run = 0
    for i, ch in enumerate(text):
        if ch == "\n":
            line_start = True
            run = 0
            continue
        if line_start:
            if ch in " \t":
                continue
            if ch == "`":
                run += 1
                if run == 3:
                    if in_fence:
                        in_fence = False
                        ranges.append((fence_start, i + 1))
                    else:
                        in_fence = True
                        fence_start = i + 1
                continue
            line_start = False
            run = 0
    if in_fence:
        # 流式下闭合标记可能尚未到达：未闭合围栏一直延伸到文本末尾，
        # 否则围栏内先到的工具调用会被当成真实调用解析
        ranges.append((fence_start, len(text)))
    return ranges


def _escape_fenced_tags(text: str, *, streaming: bool = True) -> str:
    """opt22d（L-C）：把代码围栏内的“形似工具调用”标签转义为实体，
    使其不再被解析成真实调用；围栏外的文本原样保留，真实调用不受影响。

    **opt22e**：非流式（含 EOS 帧）时，若围栏区间以**真收尾三元组**结束，
    则只转义它**之前**的部分、收尾原样保留。

    背景：模型写文档时，参数值（`Write.content` / `Edit.new_string`）里的
    markdown 代码块是**文档内容**。若其围栏数为**奇数**或流式下尚未闭合，
    `_fenced_ranges` 会把区间一路延伸到文本末尾 ⇒ **真收尾**被转义成 `⟨'…'⟩`
    ⇒ 收尾永远找不到 ⇒ 参数永不闭合 ⇒ **内容整段丢失**（离线 100% 复现）。

    为何只在**非流式/EOS** 豁免：流式中间态下，文档示例的三层收尾会短暂出现在
    文本末尾、与真收尾**字节同形**（XML 语义上也确实成立），无法区分 ⇒ 此时
    照常转义（内容继续按参数体处理），待 EOS 判据完整后再放行真收尾。
    """
    if not text:
        return text
    for start, end in reversed(_fenced_ranges(text)):
        seg = text[start:end]
        cut = None if streaming else _trailing_close_start(seg)
        if cut is None:
            text = text[:start] + _escape_tool_like_tags(seg) + text[end:]
        else:
            text = (
                text[:start]
                + _escape_tool_like_tags(seg[:cut])
                + seg[cut:]
                + text[end:]
            )
    return text



# opt23 debug log (disabled by default; enable via VLLM_OPT23_DEBUG_LOG=1)
_OPT23_LOG = None

def _mask_fenced_tags(text: str, *, streaming: bool = True) -> str:
    """opt22d（L-E）：围栏遮罩 + **丢弃**末尾未闭合的工具类标签片段。

    流式下标签会被切分（``<p`` + ``arameter=file_path>``、``</function`` + ``>``），
    若在片段粒度上遮罩，会漏出裸 ``<`` 或产出 ``⟨'/function>`` 这类畸形。
    这里把未闭合尾部整体**从遮罩结果中摘掉**，待其闭合后再以整体转义形式出现；
    因调用方靠“遮罩(current) − 遮罩(previous)”相减推导增量，摘掉后前缀关系仍成立。

    第八轮追加 L-D：围栏**外**的非行首 ``<tool_call>``（如行内代码 / 表格示例）
    同样转义，避免 impl 把展示示例当成真实调用、进而停止输出正文。

    第九轮追加：``streaming=False`` 用于**消息结束**时收口——此时判据（后文是否
    跟 ``<function=``）已完整，不再需要暂存，可把此前扣留的完整标签转义后释放。
    """
    head, _held = split_incomplete_tool_tag_tail(text)
    return _escape_non_linestart_tool_calls(
        _escape_fenced_tags(head, streaming=streaming), streaming=streaming
    )


def _log(msg: str, *args: Any) -> None:
    global _OPT23_LOG
    import os as _os
    if not _os.environ.get("VLLM_OPT23_DEBUG_LOG"):
        return
    if _OPT23_LOG is None:
        _OPT23_LOG = open("/tmp/log/vllm-tool-log.txt", "a", buffering=1)
    _OPT23_LOG.write(msg % args if args else msg)
    _OPT23_LOG.write("\n")


class Qwen3CoderToolParser(ToolParser):
    supports_required_and_named: bool = not VLLM_ENFORCE_STRICT_TOOL_CALLING

    # opt23 Path B fake streaming (=5): partial streaming only for
    # Write/Edit (tools with long string content params that benefit
    # from incremental display).  Path B Native (=1,2) handles all
    # XML tools via diff-based XML streaming — no whitelist needed.
    _STREAMING_TOOLS: frozenset = frozenset({"Write", "Edit"})

    @property
    def _fix_force_json(self) -> bool:
        """opt23 §11.22: fix=2 强制 JSON 真流式增量路由（XML 输出也最终按 JSON 输出）。"""
        return VLLM_QWEN3X_TOOL_FIX == 2

    @property
    def _fix_force_xml(self) -> bool:
        """opt23 §11.22: fix=3/4 强制 XML 路由（JSON 输出也走 XML 解析）。"""
        return VLLM_QWEN3X_TOOL_FIX in (3, 4)

    @property
    def _is_streaming_tool(self) -> bool:
        """Streaming optimizations apply to =1,2,5 for Write/Edit tools.

        Guards _pending_brace, partial streaming, and remaining delta
        paths for tools with long string content params that benefit
        from incremental display.
        """
        return (VLLM_QWEN3X_TOOL_FIX in (1, 2, 4, 5)
                and not self._json_tool_active
                and self.current_function_name in self._STREAMING_TOOLS)

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)

        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict] = []
        # Override base class type - we use string IDs for tool calls
        self.current_tool_id: str | None = None  # type: ignore
        self.streamed_args_for_tool: list[str] = []

        # opt22d: 被降级的（未注册）工具调用索引
        self._opt22d_demoted: set = set()
        self._opt22d_force_all: bool = False   # opt22d A/B/C：整轮降级

        # Sentinel tokens for streaming mode
        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self.tool_call_prefix: str = "<function="
        self.function_end_token: str = "</function>"
        self.parameter_prefix: str = "<parameter="
        self.parameter_end_token: str = "</parameter>"
        self.is_tool_call_started: bool = False
        self.failed_count: int = 0

        # Enhanced streaming state - reset for each new message
        self._reset_streaming_state()

        # Regex patterns
        self.tool_call_complete_regex = re.compile(
            r"<tool_call>(.*?)</tool_call>", re.DOTALL
        )
        self.tool_call_regex = re.compile(
            r"<tool_call>(.*?)</tool_call>|<tool_call>(.*?)$", re.DOTALL
        )
        self.tool_call_function_regex = re.compile(
            r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL
        )
        self.tool_call_parameter_regex = re.compile(
            r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
            re.DOTALL,
        )

        if not self.model_tokenizer:
            raise ValueError(
                "The model tokenizer must be passed to the ToolParser "
                "constructor during construction."
            )

        self.tool_call_start_token_id = self.vocab.get(self.tool_call_start_token)
        self.tool_call_end_token_id = self.vocab.get(self.tool_call_end_token)

        if self.tool_call_start_token_id is None or self.tool_call_end_token_id is None:
            raise RuntimeError(
                "Qwen3 XML Tool parser could not locate tool call start/end "
                "tokens in the tokenizer!"
            )

        logger.debug(
            "vLLM Successfully import tool parser %s !", self.__class__.__name__
        )

    def _generate_tool_call_id(self) -> str:
        """Generate a unique tool call ID."""
        return f"call_{uuid.uuid4().hex[:24]}"

    def _registered_tool_names(self) -> set:
        """已注册工具名集合。tools 未提供时为空集。"""
        names = set()
        for tool in (self.tools or []):
            fn = getattr(tool, "function", None)
            nm = getattr(fn, "name", None) if fn is not None else None
            if not nm:
                nm = getattr(tool, "name", None)
            if nm:
                names.add(nm)
        return names

    def _is_registered_tool(self, name) -> bool:
        """opt22d L-A: 名称是否在已注册工具内。
        tools 为空（非工具请求）时一律视为已注册，保持宽松、绝不误伤真实调用。"""
        if self._opt22d_force_all:
            return False  # opt22d A/B/C：整轮降级
        names = self._registered_tool_names()
        if not names:
            return True
        return name in names

    def _missing_required_args(self, tc) -> bool:
        """opt22e B：该调用的参数是否**缺少 schema 的必填字段**。

        参数无法解析（非法 JSON / 空）时也视为缺失 —— 真实调用的参数总能解析。
        `tools` 未提供或该工具无 `required` 声明时返回 False（**宽松兜底、不误伤**）。
        """
        try:
            fn = tc.function
        except Exception:
            return False
        if fn is None:
            return False
        req = find_tool_required(self.tools, fn.name)
        if not req:
            return False
        args = fn.arguments
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except Exception:
                return True          # 参数非法 ⇒ 必填必然缺失
        if not isinstance(args, dict):
            return True
        return any(k not in args or args[k] in (None, "") for k in req)

    def _opt22d_should_force_all(self, request, raw_text: str = "") -> bool:
        """opt22d A/B/C：依“请求侧信号”判断是否整轮降级为文本。

        A `tool_choice == "none"`：客户端显式禁止调用 → 形似调用必为示例
        B 请求无工具且解析器无工具：无对象可调 → 形似调用必为示例
        """
        if request is None:
            return False
        if getattr(request, "tool_choice", None) == "none":
            return True
        # opt22e C：请求侧「明确禁止调用」信号（与 xml 侧同义）。见 utils 说明。
        if request_forbids_tool_calls(request):
            return True
        if not getattr(request, "tools", None) and not self.tools:
            return True
        return False

    def _thinking_enabled_of(self, request) -> bool:
        """opt22d 第七轮：本次请求是否开启思考。

        缺省视为开启（与模板默认 enable_thinking=True 一致）；仅当客户端显式
        enable_thinking=False 时才返回 False——那时保持历史的流式解析行为。
        """
        kw = getattr(request, "chat_template_kwargs", None) or {}
        return bool(kw.get("enable_thinking", True))

    def _should_defer_parsing(self, text: str) -> bool:
        """opt22d 第七轮：是否需要把本轮工具解析推迟到 finish。

        仅当文本里已出现**闭合的**函数名标签、且其中含有 _STREAMING_TOOLS
        之外的工具时返回 True。函数名尚未闭合（仍处分片）时返回 False——
        此时 impl 不会产出工具增量，不推迟无副作用；且可避免 Write/Edit
        因分片被误判为“非白名单”而丢掉流式增量显示。
        """
        names = set(_FN_TAG_RE.findall(text))
        if not names:
            return False
        return bool(names - self._STREAMING_TOOLS)

    def _opt22d_filter_delta(self, delta):
        """opt22d 流式过滤：吞掉未注册调用的结构化增量；内容里的形似标签转义。"""
        if delta is None:
            return None
        demoted = getattr(self, "_opt22d_demoted", None)
        if demoted is None:
            demoted = self._opt22d_demoted = set()
        if delta.tool_calls:
            hit = False
            for tc in delta.tool_calls:
                fn = tc.function
                nm = fn.name if fn else None
                idx = tc.index if tc.index is not None else -1
                if nm and not self._is_registered_tool(nm):
                    demoted.add(idx)
                    # 该调用恰为最后一个时从服务层计数剔除（避免 finish_reason 误判）。
                    # 只动 prev_tool_call_arr：pop streamed_args_for_tool 会让
                    # _finish_streaming_function 的长度校验报警且索引错位。
                    if 0 <= idx == len(self.prev_tool_call_arr) - 1:
                        self.prev_tool_call_arr.pop()
                if idx in demoted:
                    hit = True
            if hit:
                return None
        if delta.content:
            esc = _to_display(delta.content)
            if esc != delta.content:
                return DeltaMessage(content=esc)
        return delta

    def _reset_streaming_state(self):
        """Reset all streaming state."""
        self.current_tool_index = 0
        self.is_tool_call_started = False
        self.header_sent = False
        self.current_tool_id = None
        self.current_function_name = None
        self.current_param_name = None
        self.current_param_value = ""
        self.param_count = 0
        self.in_param = False
        self.in_function = False
        self.accumulated_text = ""
        self.json_started = False
        self.json_closed = False
        # Store accumulated parameters for type conversion
        self.accumulated_params = {}
        self.streaming_request = None
        self.json_tool_calls_streamed = 0
        # opt23: partial streaming for long string params (write/edit content)
        self._partial_emit_offset: int = 0
        self._partial_param_start: int = -1
        self._partial_emitted: str = ""
        self._partial_prefix: str = ""
        self._last_partial_time: float = 0.0
        # opt23: defer standalone "{" to combine with first real delta
        self._pending_brace: bool = False
        # opt23: flag for model duplication of <function=...> in streaming
        self._func_dup_detected: bool = False
        # opt23 Path C: JSON-format tool call (not XML) detected in streaming
        self._json_tool_active: bool = False
        # opt23 =3 JSON path: end position of processed tool call for skip
        self._json_processed_end: int = 0
        # opt23 §11.42 (v122 框架迁入): resolved tool-call format
        # ('xml'|'json', or None when unconfigured → auto-detect);
        # config-first via request.chat_template_kwargs.tool_call_format.
        self._tool_format: str | None = None
        # opt23 =2 XML path: incomplete param saved from parsing loop
        self._stream_partial_name: str | None = None
        self._stream_partial_value: str | None = None
        # opt23 =2 优化3: 状态指纹，无变化时跳过 diff
        self._stream_last_fp: tuple | None = None
        # opt22d 第七轮：按会话 thinking 开关决定解析时机——开启思考时，
        # 非 _STREAMING_TOOLS 的工具不在流式阶段产出，统一推迟到 finish
        # 用完整文本解析（避免 thinking 内的标签被误当真实调用执行）。
        self._thinking_enabled: bool = True
        self._deferred_used: bool = False
        # opt22e：EOS 强制收尾所需状态（见 `flush_streaming_finish`）
        self._opt22e_last_text: str = ""
        self._opt22e_last_request: object | None = None
        self._needs_finish_flush: bool = False
        # 已**真正下发**的增量（按 tool index）——flush 只补缺失部分，避免重复
        self._opt22e_streamed: dict = {}
        self._opt22e_name_sent: dict = {}

    def _compute_json_diff(self, current_json: str, prev_streamed: str) -> str:
        """Safe-prefix diff on JSON strings for incremental tool args.

        Mirrors upstream ``_compute_arg_delta`` from vLLM PR #45413.
        Returns only the suffix of *current_json* that is not already
        present in *prev_streamed*, i.e. a valid JSON continuation.

        When *prev_streamed* is empty the entire *current_json* is
        returned (first emission).
        """
        if not prev_streamed:
            return current_json
        if not current_json:
            return ""
        common = find_common_prefix(prev_streamed, current_json)
        return current_json[len(common):]

    @staticmethod
    def _safe_content_length(text: str) -> int:
        """Return length of *text* prefix safe to emit as partial content.

        Detects trailing XML close-tag fragments (``</parameter>``,
        ``</function>``, ``</tool_call>``) that the model may generate
        during streaming and truncates before them.  Characters that
        look like a legitimate XML close tag are **not** emitted as
        string content — when the tag completes the XML parser will
        detect it and the streaming-param handler will emit the full
        corrected value.
        """
        _pos = text.rfind('</')
        if _pos < 0:
            return len(text)

        _after = text[_pos + 2:]       # text after '</'
        _after_clean = _after.rstrip('>')

        # Valid XML close-tag names in qwen3coder format.
        for _tag in ('parameter', 'function', 'tool_call'):
            if _tag.startswith(_after_clean) or _after_clean.startswith(_tag):
                _result = _pos
                if _result > 0 and text[_result - 1] == '\n':
                    _result -= 1
                return _result

        # '</' not followed by a known close-tag name (e.g. </div>)
        # — treat as legitimate string content.
        return len(text)

    def _is_string_param(self, param_name: str) -> bool:
        """True when *param_name* is typed as ``string`` in the tool schema.

        Used by the boundary detection in the param-completion loop to
        decide whether ``<parameter=next>`` can serve as a fallback
        delimiter — string params must wait for an explicit
        ``</parameter>`` to avoid premature completion with a partial
        value.
        """
        func = self.current_function_name
        if not func:
            return False
        _pc = find_tool_properties(self.tools, func)
        _ps = _pc.get(param_name, {})
        _pt = extract_types_from_schema(_ps)
        return bool(_pt and "string" in _pt)

    @staticmethod
    def _is_inside_json_string(json_text: str) -> bool:
        """Return True if the final character position in *json_text*
        is inside a JSON string value (between an unescaped ``"`` and
        its matching closing ``"``).

        Walks *json_text* character by character, tracking ``in_string``
        and ``escape_next`` state.  Returns the string state at the
        end of the text.
        """
        in_string = False
        escape_next = False
        for c in json_text:
            if escape_next:
                escape_next = False
                continue
            if c == '\\':
                escape_next = True
                continue
            if c == '"':
                in_string = not in_string
        return in_string

    @staticmethod
    def _unescape_display_chars(s: str) -> str:
        """Replace JSON escape sequences ``\\n``, ``\\t``, ``\\r`` with
        literal newline / tab / carriage-return characters for frontend
        display.

        Preserves structural escapes ``\\\"`` and ``\\\\`` so the
        accumulated JSON remains structurally parseable.
        """
        result: list[str] = []
        i = 0
        while i < len(s):
            c = s[i]
            if c == '\\' and i + 1 < len(s):
                nxt = s[i + 1]
                if nxt == '\\' or nxt == '"':
                    # Structural escapes — keep as-is
                    result.append(c)
                    result.append(nxt)
                    i += 2
                    continue
                elif nxt == 'n':
                    result.append('\n')
                    i += 2
                    continue
                elif nxt == 't':
                    result.append('\t')
                    i += 2
                    continue
                elif nxt == 'r':
                    result.append('\r')
                    i += 2
                    continue
            result.append(c)
            i += 1
        return ''.join(result)

    @staticmethod
    def _escape_raw_control_chars(s: str) -> str:
        """模型可能把真实换行/控制字符（XML 换行习惯）直接输出进 JSON
        字符串值，导致 JSON 非法（Write content 参数丢失根因）。仅在
        字符串值内部把控制字符转义为 JSON 合法序列（\\n → \\\\n）。"""
        out = []
        i = 0
        n = len(s)
        in_str = False
        _esc_map = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}
        while i < n:
            c = s[i]
            if c == "\\":
                out.append(c)
                if i + 1 < n:
                    out.append(s[i + 1])
                    i += 2
                else:
                    i += 1
                continue
            if c == '"':
                in_str = not in_str
                out.append(c)
                i += 1
                continue
            if in_str and ord(c) < 0x20:
                out.append(_esc_map.get(c, "\\u%04x" % ord(c)))
                i += 1
                continue
            out.append(c)
            i += 1
        return "".join(out)

    def _handle_json_tool_streaming(
        self, current_text: str, delta_text: str
    ) -> DeltaMessage | None:
        """Handle JSON-format tool calls with safe-prefix JSON diffing.

        Supported format (single tool)::

            {"name": "Write", "arguments": {"file_path": "/x", "content": "..."}}

        Each emitted delta is a valid JSON continuation of the arguments
        object, suitable for Anthropic ``input_json_delta`` accumulation.
        """
        # --- name extraction (first time only) ---
        if not self.header_sent:
            _name_kw = current_text.find('"name"')
            if _name_kw == -1:
                return None
            _colon = current_text.find(":", _name_kw)
            if _colon == -1:
                return None
            _q1 = current_text.find('"', _colon + 1)
            if _q1 == -1:
                return None
            _q2 = current_text.find('"', _q1 + 1)
            if _q2 == -1:
                return None
            name = current_text[_q1 + 1:_q2]
            if not name:
                return None

            self.current_function_name = name
            self.current_tool_id = self._generate_tool_call_id()
            self.header_sent = True
            self.in_function = True
            self.json_started = True
            self.accumulated_params = {}

            self.prev_tool_call_arr.append({"name": name, "arguments": "{}"})
            self.streamed_args_for_tool.append("")

            return DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=self.current_tool_index,
                        id=self.current_tool_id,
                        function=DeltaFunctionCall(name=name, arguments=""),
                        type="function",
                    )
                ]
            )

        # --- arguments JSON extraction ---
        _args_kw = current_text.find('"arguments"')
        if _args_kw == -1:
            return None

        _args_colon = current_text.find(":", _args_kw)
        if _args_colon == -1:
            return None

        # skip whitespace after colon
        _val_start = _args_colon + 1
        while _val_start < len(current_text) and current_text[_val_start] in " \t\n\r":
            _val_start += 1
        if _val_start >= len(current_text):
            return None
        if current_text[_val_start] != "{":
            return None

        # Walk brace depth to find the end of the arguments JSON object.
        # For incomplete streaming text the closing "}" may be absent;
        # in that case we take everything from _val_start to end-of-text.
        _depth = 0
        _in_str = False
        _esc = False
        _json_end = -1
        for i in range(_val_start, len(current_text)):
            c = current_text[i]
            if _esc:
                _esc = False
                continue
            if c == "\\":
                _esc = True
                continue
            if c == '"' and not _esc:
                _in_str = not _in_str
                continue
            if _in_str:
                continue
            if c == "{":
                _depth += 1
            elif c == "}":
                _depth -= 1
                if _depth == 0:
                    _json_end = i + 1
                    break

        if _json_end > 0:
            current_args = current_text[_val_start:_json_end]
        else:
            current_args = current_text[_val_start:]

        # opt23: 模型把真实换行/控制字符（XML 换行习惯）直接输出进 JSON
        # 字符串值时 → 字符串值内转义为 JSON 合法序列，否则参数提取
        # 非法/截断（Write content 丢失、工具调用参数不完整根因）。
        current_args = self._escape_raw_control_chars(current_args)

        # --- diff against previously streamed ---
        idx = self.current_tool_index
        prev = self.streamed_args_for_tool[idx] if idx < len(self.streamed_args_for_tool) else ""
        # opt23 §11.29: 对齐主线 hermes `_compute_args_diff`（前缀截取增量）。
        # 主线用 `args[len(prev):]`（信任流式累积，current 是 prev 的超集），
        # 替代本地 `_compute_json_diff`（find_common_prefix 手工 diff）。
        if len(current_args) <= len(prev):
            return None
        delta = current_args[len(prev):]
        if not delta:
            return None

        # Store raw (escaped) delta for future diffs and serving-layer
        # remaining-delta computation.  We emit an unescaped version
        # for display when inside a JSON string value.
        if idx < len(self.streamed_args_for_tool):
            self.streamed_args_for_tool[idx] += delta

        # Update prev_tool_call_arr when JSON is complete so the serving
        # layer can compute the correct remaining delta at stream end.
        # 优化 A: 标记参数完整。确认外层工具调用也闭合了（}} 连续）
        # 或文本正好在参数闭括号处结束（EOS 场景）。
        if (_json_end > 0 and idx < len(self.prev_tool_call_arr)
                and (_json_end == len(current_text)
                     or current_text[_json_end] == '}')):
            self.prev_tool_call_arr[idx]["arguments"] = current_args
            self.json_closed = True
            self.in_function = False
            self._json_processed_end = _json_end
            _log(
                "opt23 JSON handler 优化A: idx=%s current_args=%s "
                "prev_tool_call_arr=%s",
                idx, current_args[:200],
                self.prev_tool_call_arr)

        # opt23: 之前为前端显示打字机效果把转义序列（\n \t \r）反转义为
        # 真实控制字符——但前端累积拼接后 JSON 非法（真实换行在字符串值
        # 内），工具执行 InputValidationError（Write content 参数失败
        # 根因）。直接发射转义版增量（字面 \n），前端拼接即可 json.parse。
        display_delta = delta

        return DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=idx,
                    function=DeltaFunctionCall(arguments=display_delta),
                )
            ]
        )

    def _handle_xml_tool_streaming(
        self, current_text: str, delta_text: str
    ) -> DeltaMessage | None:
        """Path B Native: XML-native true streaming with safe-prefix diffing.

        Isomorphic to ``_handle_json_tool_streaming``.  Builds an XML
        intermediate state from the current ``tool_text`` and emits the
        diff against previously streamed XML.  Pure XML output — no JSON
        involvement.

        Algorithm:
        1. HEADER EXTRACTION — find ``<function=NAME>``, send name delta
        2. BUILD XML INTERMEDIATE STATE — parse completed + in-progress params
        3. DIFF — ``find_common_prefix(prev_xml, current_xml)``
        4. EMIT — return XML continuation delta

        Activated when ``VLLM_QWEN3X_TOOL_FIX`` is 1 or 2 and
        the model is generating XML-format tool calls.
        """

        # Find tool positions
        tool_start_positions = self._tool_start_positions(current_text)
        if self.current_tool_index >= len(tool_start_positions):
            return None

        tool_start_idx = tool_start_positions[self.current_tool_index]
        tool_end_idx = current_text.find(
            self.tool_call_end_token, tool_start_idx
        )
        if tool_end_idx == -1:
            tool_text = current_text[tool_start_idx:]
        else:
            tool_text = current_text[
                tool_start_idx:tool_end_idx + len(self.tool_call_end_token)
            ]

        # ---- Step 1: Header Extraction ----
        if not self.header_sent:
            if self.tool_call_prefix not in tool_text:
                return None
            func_start = tool_text.find(self.tool_call_prefix) + len(
                self.tool_call_prefix
            )
            func_end = tool_text.find(">", func_start)
            if func_end == -1:
                return None

            name = tool_text[func_start:func_end]
            if not name:
                return None

            self.current_function_name = name
            self.current_tool_id = self._generate_tool_call_id()
            self.header_sent = True
            self.in_function = True
            self.json_started = True
            self.accumulated_params = {}
            self._pending_brace = False

            self.prev_tool_call_arr.append(
                {"name": name, "arguments": "{}"}
            )
            self.streamed_args_for_tool.append("")

            return DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=self.current_tool_index,
                        id=self.current_tool_id,
                        function=DeltaFunctionCall(
                            name=name, arguments=""
                        ),
                        type="function",
                    )
                ]
            )

        # ---- Step 2: Build XML Intermediate State ----
        # Find all <parameter=...> positions in tool_text
        param_starts: list[int] = []
        search_idx = 0
        while True:
            search_idx = tool_text.find(
                self.parameter_prefix, search_idx
            )
            if search_idx == -1:
                break
            param_starts.append(search_idx)
            search_idx += len(self.parameter_prefix)

        # Filter out stale params before the LAST <function=...> tag
        # (model duplication artifact in streaming).
        _func_positions: list[int] = []
        _fsi = 0
        while True:
            _fsi = tool_text.find(self.tool_call_prefix, _fsi)
            if _fsi == -1:
                break
            _close = tool_text.find(
                ">", _fsi + len(self.tool_call_prefix)
            )
            if _close != -1:
                _func_positions.append(_fsi)
            _fsi += len(self.tool_call_prefix)

        if param_starts and _func_positions:
            _func_positions = [
                p for p in _func_positions if p < param_starts[0]
            ]
        if len(_func_positions) > 1:
            _last_func = _func_positions[-1]
            param_starts = [
                p for p in param_starts if p > _last_func
            ]

        # Build XML from parsed parameters
        xml_parts: list[str] = []

        for param_start_pos in param_starts:
            param_start = param_start_pos + len(self.parameter_prefix)
            remaining = tool_text[param_start:]

            if ">" not in remaining:
                break

            name_end = remaining.find(">")
            param_name = remaining[:name_end]

            value_start = param_start + name_end + 1
            value_text = tool_text[value_start:]
            if value_text.startswith("\n"):
                value_text = value_text[1:]

            # Find end of this param
            # 优化1: detect false-positive </parameter> inside content
            # (e.g. Write content that contains "</parameter>" literally).
            # Scan forward until a true close-tag is confirmed by the
            # text that follows it (another <parameter=, </function>,
            # </tool_call>, or end-of-text).
            param_end_idx = -1
            _pe_search = 0
            while _pe_search < len(value_text):
                _pe = value_text.find(self.parameter_end_token, _pe_search)
                if _pe == -1:
                    break
                _after = value_text[_pe + len(self.parameter_end_token):].lstrip()
                if (_after.startswith(self.parameter_prefix)
                        or _after.startswith(self.function_end_token)
                        or _after.startswith(self.tool_call_end_token)
                        # opt22c: double-close artifact — a bare </parameter>
                        # followed by another </parameter> closes the value.
                        or _after.startswith(self.parameter_end_token)
                        or not _after):
                    param_end_idx = _pe
                    break
                # Content contained a </parameter> fragment — keep scanning
                _pe_search = _pe + 1

            if param_end_idx != -1:
                # Complete param — include closing </parameter> tag
                param_value = value_text[:param_end_idx]
                if param_value.endswith("\n"):
                    param_value = param_value[:-1]
                xml_parts.append(
                    f"<parameter={param_name}>\n{param_value}"
                    f"\n</parameter>"
                )
                if param_name not in self.accumulated_params:
                    # opt23 =2: 对完整参数做类型转换 (bool/int/string)，
                    # 确保 json.dumps 输出正确的 JSON 类型
                    param_config = find_tool_properties(
                        self.tools, self.current_function_name or "")
                    converted = self._convert_param_value(
                        param_value, param_name,
                        param_config, self.current_function_name or "")
                    self.accumulated_params[param_name] = converted
            else:
                # Incomplete param — trim trailing XML tags from value
                next_param = value_text.find(self.parameter_prefix)
                func_end_v = value_text.find(self.function_end_token)
                tool_end_v = value_text.find(self.tool_call_end_token)

                end_candidates = []
                if next_param != -1:
                    end_candidates.append(next_param)
                if func_end_v != -1:
                    end_candidates.append(func_end_v)
                if tool_end_v != -1:
                    end_candidates.append(tool_end_v)

                if end_candidates:
                    param_value = value_text[:min(end_candidates)]
                else:
                    param_value = value_text

                if param_value.endswith("\n"):
                    param_value = param_value[:-1]

                # opt23: guard against partial XML close-tag fragments
                # (e.g. "</param", "</funct") leaking into the XML
                # intermediate state.
                _safe_len = self._safe_content_length(param_value)
                if _safe_len < len(param_value):
                    param_value = param_value[:_safe_len]
                    if param_value.endswith("\n"):
                        param_value = param_value[:-1]

                xml_parts.append(
                    f"<parameter={param_name}>\n{param_value}"
                )
                # Save incomplete param for Step 3 JSON building
                self._stream_partial_name = param_name
                self._stream_partial_value = param_value
                break  # stop at first incomplete param

        else:
            # 优化2: for-else — all params complete this cycle, clear
            # stale partial state so Step 3 doesn't attach a redundant
            # in-progress fragment that was already fully parsed.
            self._stream_partial_name = None
            self._stream_partial_value = None

        current_xml = "\n".join(xml_parts)

        # ---- Step 3: Build JSON from accumulated params ----
        # =2 and =1 XML path: XML input → JSON output (standard tool args).
        # Build partial JSON from accumulated_params + current in-progress
        # param, diff against previously streamed.
        idx = self.current_tool_index
        prev_raw = (
            self.streamed_args_for_tool[idx]
            if idx < len(self.streamed_args_for_tool)
            else ""
        )

        # 优化3: state fingerprint — skip build+diff when nothing changed
        _state_fp = (len(self.accumulated_params),
                     len(self._stream_partial_value or ""))
        if _state_fp == self._stream_last_fp:
            delta = ""
        else:
            self._stream_last_fp = _state_fp

            # Build partial JSON (no closing } — added at function-end)
            parts: list[str] = []
            first = True
            for name, value in self.accumulated_params.items():
                serialized = json.dumps(value, ensure_ascii=False)
                if first:
                    parts.append(f'"{name}": {serialized}')
                    first = False
                else:
                    parts.append(f', "{name}": {serialized}')

            # Attach the current in-progress param (trimmed by parsing loop)
            partial_name = self._stream_partial_name
            partial_value = self._stream_partial_value

            # Guard: skip stale partial when the param was fully completed
            # in this cycle (now in accumulated_params) to avoid duplicate keys.
            if partial_name is not None and partial_value is not None:
                if partial_name not in self.accumulated_params:
                    # Only stream partial values for string-type params.
                    # Bool/int params change JSON representation after type
                    # conversion (e.g. "false" → false), which breaks the
                    # safe-prefix diff and produces garbage deltas.
                    _is_string = True
                    if partial_name:
                        _pc = find_tool_properties(
                            self.tools, self.current_function_name or "")
                        _ps = _pc.get(partial_name, {})
                        _pt = extract_types_from_schema(_ps)
                        if _pt and "string" not in _pt:
                            _is_string = False
                    if _is_string:
                        escaped = json.dumps(partial_value, ensure_ascii=False)[1:-1]
                        if first:
                            parts.append(f'"{partial_name}": "{escaped}')
                        else:
                            parts.append(f', "{partial_name}": "{escaped}')

            current_partial = "{" + "".join(parts)
            delta = self._compute_json_diff(current_partial, prev_raw)

        # ---- Step 4: Detect function end ----
        func_ended = (
            self.function_end_token in tool_text
            and not self.json_closed
        )

        if func_ended:
            # Build complete JSON from accumulated_params
            full_json = json.dumps(
                self.accumulated_params, ensure_ascii=False)

            if idx < len(self.prev_tool_call_arr):
                self.prev_tool_call_arr[idx]["arguments"] = full_json

            _log(
                "opt23 Step4 func_ended: tool=%s idx=%s "
                "accumulated=%s full_json=%s prev_raw=%s delta_before=%s",
                self.current_function_name, idx,
                self.accumulated_params, full_json,
                prev_raw, delta)

            # Compute closing delta (the } suffix)
            streamed_total = prev_raw + delta if delta else prev_raw
            if full_json.startswith(streamed_total):
                delta += full_json[len(streamed_total):]
            elif full_json.startswith(prev_raw):
                delta = full_json[len(prev_raw):]

            self.json_closed = True
            self.in_function = False
            self.accumulated_params = {}

        if not delta:
            return None


        # Update streamed args
        if idx < len(self.streamed_args_for_tool):
            self.streamed_args_for_tool[idx] += delta

        return DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=idx,
                    function=DeltaFunctionCall(arguments=delta),
                )
            ]
        )

    def _emit_xml_json_diff(
        self, tool_text: str, param_starts: list[int],
        tool_start_idx: int, current_text: str,
    ) -> DeltaMessage | None:
        """Phase 3: build partial args JSON from XML params, emit via
        safe-prefix JSON-diff.

        Constructs a PARTIAL JSON string (no closing ``}``, trailing
        string value has no closing ``"``) so that each diff is a valid
        JSON continuation that can be safely accumulated by the client.

        Activated when ``VLLM_QWEN3X_TOOL_FIX == 4`` (=4 互转, 待设计).
        """
        partial_name = None
        partial_value = None

        # Extract partial param value when a param is still forming
        if self.param_count < len(param_starts):
            incomplete_start = param_starts[self.param_count]
            param_text = tool_text[
                incomplete_start + len(self.parameter_prefix):
            ]
            if ">" not in param_text:
                return None
            name_end = param_text.find(">")
            partial_name = param_text[:name_end]

            # Type gate: only string params get partial-inclusion
            _pc = find_tool_properties(
                self.tools, self.current_function_name or "",
            )
            _ps = _pc.get(partial_name, {})
            _pt = extract_types_from_schema(_ps)
            if _pt and "string" not in _pt:
                return None

            value_start = (
                incomplete_start + len(self.parameter_prefix) + name_end + 1
            )
            _abs_value_start = tool_start_idx + value_start
            if (
                _abs_value_start < len(current_text)
                and current_text[_abs_value_start:_abs_value_start + 1] == "\n"
            ):
                _abs_value_start += 1
            partial_value = current_text[_abs_value_start:]

        # --- build partial JSON string ---
        # Uses the same format as json_fragments: starts with "{",
        # completed params have fully-formed key-value pairs, and the
        # trailing string param (if any) is left open (no closing ").
        parts = []
        first = True
        for name, value in self.accumulated_params.items():
            serialized = json.dumps(value, ensure_ascii=False)
            if first:
                parts.append(f'"{name}": {serialized}')
                first = False
            else:
                parts.append(f', "{name}": {serialized}')

        if partial_name and partial_value is not None:
            escaped = json.dumps(partial_value, ensure_ascii=False)[1:-1]
            if first:
                parts.append(f'"{partial_name}": "{escaped}')
            else:
                parts.append(f', "{partial_name}": "{escaped}')

        current_partial = "{" + "".join(parts)

        # --- diff ---
        idx = self.current_tool_index
        prev = (
            self.streamed_args_for_tool[idx]
            if idx < len(self.streamed_args_for_tool) else ""
        )
        delta = self._compute_json_diff(current_partial, prev)
        if not delta:
            return None

        if idx < len(self.streamed_args_for_tool):
            self.streamed_args_for_tool[idx] += delta

        # _pending_brace is never used here — the "{" is always part
        # of the partial JSON string built above.  Mark it consumed
        # so the func-end handler won't re-emit it.
        self._pending_brace = False

        return DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=idx,
                    function=DeltaFunctionCall(arguments=delta),
                )
            ]
        )

    def _convert_param_value(
        self, param_value: str, param_name: str, param_config: dict, func_name: str
    ) -> Any:
        """Convert parameter value based on its type in the schema."""
        if not isinstance(param_value, str):
            return param_value
        # opt22e：先还原遮罩转义 —— 本方法是**三条取值路径的唯一收口**
        # （非流式 `_extract_parameters_from_text` / `_parse_xml_function_call` /
        # 流式 impl），在此还原可一次覆盖。详见 `_to_original`。
        param_value = _to_original(param_value)
        param_schema = param_config.get(param_name, {})
        param_types = extract_types_from_schema(param_schema)
        return coerce_to_schema_type(param_value, param_types)

    def _parameter_start_positions(
        self,
        text: str,
        param_config: dict,
    ) -> list[int]:
        """Find parameter tags without treating inline code as markup.

        Qwen's wire format puts parameter tags on structural boundaries. Tool
        values, especially code and JSON passed to Write/Edit, may themselves
        contain strings such as ``<parameter=x>``. A raw substring scan treats
        those literals as new arguments and corrupts the tool call.

        Unknown names at a bare line boundary are ignored when a schema is
        available, but remain accepted after a real closing tag. This preserves
        malformed-output recovery without interpreting an inline literal as a
        schema field.
        """
        positions: list[int] = []
        search_idx = 0
        while True:
            position = text.find(self.parameter_prefix, search_idx)
            if position == -1:
                break
            search_idx = position + len(self.parameter_prefix)

            name_end = text.find(">", search_idx)
            if name_end == -1:
                break
            name = text[search_idx:name_end]

            line_start = text.rfind("\n", 0, position) + 1
            at_line_boundary = not text[line_start:position].strip()
            after_close = text[:position].rstrip().endswith(self.parameter_end_token)
            after_function_header = False
            if not positions:
                function_start = text.rfind(self.tool_call_prefix, 0, position)
                if function_start != -1:
                    function_header_end = text.find(">", function_start)
                    after_function_header = (
                        function_header_end != -1
                        and not text[function_header_end + 1 : position].strip()
                    )
                elif not text[:position].strip():
                    after_function_header = True

            is_structural = at_line_boundary or after_close or after_function_header
            if not is_structural:
                continue
            if (
                param_config
                and name not in param_config
                and not after_close
                and not after_function_header
            ):
                continue
            positions.append(position)
        return positions

    def _find_structural_parameter_end(
        self,
        text: str,
        *,
        value_start: int,
        next_parameter_start: int | None,
        container_end: int | None,
        allow_text_end: bool,
    ) -> int | None:
        """Return a closing tag only when its suffix is structural.

        Literal ``</parameter>`` text is common in generated XML and source
        files. It closes the current value only when followed by the next real
        parameter, the function/tool boundary, or (for a complete non-streamed
        body) the end of the text. Streaming deliberately waits for that one-
        token lookahead instead of publishing a value that may later truncate.
        """
        search_idx = value_start
        limit = next_parameter_start
        if limit is None:
            limit = container_end
        if limit is None:
            limit = len(text)

        while True:
            position = text.find(self.parameter_end_token, search_idx)
            if position == -1 or position >= limit:
                return None
            suffix = position + len(self.parameter_end_token)
            while suffix < len(text) and text[suffix].isspace():
                suffix += 1
            # opt22c: a bare </parameter_end_t> fragment whose suffix is yet another
            # </parameter_end_t> (model double-close artifact) is structural too.
            if text.startswith(self.parameter_end_token, suffix):
                return position
            if next_parameter_start is not None and suffix == next_parameter_start:
                return position
            if container_end is not None and suffix == container_end:
                return position
            if allow_text_end and suffix == len(text):
                return position
            search_idx = position + len(self.parameter_end_token)

    def _parse_xml_function_call(self, function_call_str: str) -> ToolCall | None:
        # Extract function name
        end_index = function_call_str.find(">")
        # If there's no ">" character, this is not a valid xml function call
        if end_index == -1:
            return None
        function_name = function_call_str[:end_index]
        param_config = find_tool_properties(self.tools, function_name)
        parameters = function_call_str[end_index + 1 :]
        param_dict = {}
        param_starts = self._parameter_start_positions(parameters, param_config)
        for param_index, position in enumerate(param_starts):
            name_start = position + len(self.parameter_prefix)
            name_end = parameters.find(">", name_start)
            if name_end == -1:
                continue
            param_name = parameters[name_start:name_end]
            value_start = name_end + 1
            next_start = (
                param_starts[param_index + 1]
                if param_index + 1 < len(param_starts)
                else None
            )
            value_end = self._find_structural_parameter_end(
                parameters,
                value_start=value_start,
                next_parameter_start=next_start,
                container_end=len(parameters),
                allow_text_end=True,
            )
            if value_end is None:
                value_end = next_start if next_start is not None else len(parameters)
            param_value = parameters[value_start:value_end]
            # Remove prefix and trailing \n
            if param_value.startswith("\n"):
                param_value = param_value[1:]
            if param_value.endswith("\n"):
                param_value = param_value[:-1]

            param_dict[param_name] = self._convert_param_value(
                param_value, param_name, param_config, function_name
            )
        return ToolCall(
            type="function",
            function=FunctionCall(
                name=function_name, arguments=json.dumps(param_dict, ensure_ascii=False)
            ),
        )

    @staticmethod
    def _parse_json_function_call(payload: str) -> ToolCall | None:
        """Parse the alternate Qwen/OpenCode JSON body inside ``<tool_call>``.

        Some Qwen coding prompts teach ``{"name": ..., "arguments": ...}``
        while the native model template teaches XML ``<function=...>``.  The
        wrapper token is unambiguous, so accepting the JSON body keeps both
        clients interoperable without treating arbitrary assistant JSON as a
        tool call.
        """
        try:
            raw_call = json.loads(payload.strip())
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(raw_call, dict):
            return None

        name = raw_call.get("name")
        arguments = raw_call.get("arguments", {})
        if not isinstance(name, str) or not name:
            return None
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return None
        if not isinstance(arguments, dict):
            return None

        return ToolCall(
            type="function",
            function=FunctionCall(
                name=name,
                arguments=json.dumps(arguments, ensure_ascii=False),
            ),
        )

    def _get_json_function_calls(self, model_output: str) -> list[ToolCall]:
        calls: list[ToolCall] = []
        for match in self.tool_call_complete_regex.finditer(model_output):
            parsed = self._parse_json_function_call(match.group(1))
            if parsed is not None:
                calls.append(parsed)
        return calls

    def _get_function_calls(self, model_output: str) -> list[str]:
        # Find all tool calls
        matched_ranges = self.tool_call_regex.findall(model_output)
        raw_tool_calls = [
            match[0] if match[0] else match[1] for match in matched_ranges
        ]

        # Back-off strategy if no tool_call tags found
        if len(raw_tool_calls) == 0:
            raw_tool_calls = [model_output]

        raw_function_calls = []
        for tool_call in raw_tool_calls:
            raw_function_calls.extend(self.tool_call_function_regex.findall(tool_call))

        function_calls = [
            match[0] if match[0] else match[1] for match in raw_function_calls
        ]
        return function_calls

    def _tool_start_positions(self, text: str) -> list[int]:
        positions: list[int] = []
        idx = 0
        while True:
            idx = text.find(self.tool_call_start_token, idx)
            if idx == -1:
                break
            positions.append(idx)
            idx += len(self.tool_call_start_token)

        # Qwen3-Coder can emit a bare <function=...> block without the
        # surrounding <tool_call> tag. Non-streaming already falls back to this
        # format; keep streaming behavior consistent.
        if not positions:
            func_idx = text.find(self.tool_call_prefix)
            if func_idx != -1:
                positions.append(func_idx)
        return positions

    def _trailing_tool_marker_prefix(self, text: str) -> str:
        """Return the unfinished suffix of a possible tool-call marker."""
        markers = (self.tool_call_start_token, self.tool_call_prefix)
        for marker in markers:
            max_len = min(len(marker) - 1, len(text))
            for prefix_len in range(max_len, 0, -1):
                if text.endswith(marker[:prefix_len]):
                    return marker[:prefix_len]
        return ""

    def _pending_tool_marker_delta_prefix(
        self, previous_text: str, current_text: str, delta_text: str
    ) -> str | None:
        """Hold only an unfinished marker prefix without dropping plain text.

        A streamed ``<`` may be the first token of ``<tool_call>`` or a
        normal HTML tag. The old code held the prefix but emitted only the
        next delta when it stopped matching a marker, permanently dropping
        the ``<``. Reconstruct the previously held suffix before emitting the
        newly confirmed non-tool text.
        """
        previous_prefix = self._trailing_tool_marker_prefix(previous_text)
        current_prefix = self._trailing_tool_marker_prefix(current_text)
        if not previous_prefix and not current_prefix:
            return None

        pending_and_delta = previous_prefix + delta_text
        if current_prefix:
            return pending_and_delta[: -len(current_prefix)]
        return pending_and_delta

    def _finish_streaming_function(self, tool_text: str) -> None:
        """Finalize one streamed XML function and its JSON argument state."""
        self.json_closed = True

        func_start = tool_text.find(self.tool_call_prefix) + len(self.tool_call_prefix)
        func_content_end = tool_text.find(self.function_end_token, func_start)
        if func_content_end != -1:
            func_content = tool_text[func_start:func_content_end]
            try:
                parsed_tool = self._parse_xml_function_call(func_content)
                if parsed_tool and self.current_tool_index < len(
                    self.prev_tool_call_arr
                ):
                    self.prev_tool_call_arr[self.current_tool_index]["arguments"] = (
                        parsed_tool.function.arguments
                    )
            except Exception:
                logger.debug(
                    "Failed to parse tool call during streaming: %s",
                    tool_text,
                    exc_info=True,
                )

        if self.current_tool_index < len(self.streamed_args_for_tool):
            self.streamed_args_for_tool[self.current_tool_index] += "}"
        else:
            logger.warning(
                "streamed_args_for_tool out of sync: index=%d len=%d",
                self.current_tool_index,
                len(self.streamed_args_for_tool),
            )

        self.in_function = False
        self.accumulated_params = {}

    def _extract_json_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
    ) -> DeltaMessage | None:
        """Emit complete alternate-format JSON calls as atomic deltas.

        Buffering until ``</tool_call>`` deliberately favors correctness over
        partial display: downstream agents must never observe malformed JSON
        arguments and then persist that broken call into the next turn.
        """
        matches = list(self.tool_call_complete_regex.finditer(current_text))
        if self.json_tool_calls_streamed >= len(matches):
            return None

        deltas: list[DeltaToolCall] = []
        first_new_match = None
        for match_index in range(self.json_tool_calls_streamed, len(matches)):
            match = matches[match_index]
            parsed = self._parse_json_function_call(match.group(1))
            if parsed is None:
                continue
            if first_new_match is None:
                first_new_match = match

            call_index = len(self.prev_tool_call_arr)
            call_id = self._generate_tool_call_id()
            arguments = parsed.function.arguments
            self.prev_tool_call_arr.append(
                {
                    "name": parsed.function.name,
                    "arguments": arguments,
                }
            )
            self.streamed_args_for_tool.append(arguments)
            deltas.append(
                DeltaToolCall(
                    index=call_index,
                    id=call_id,
                    function=DeltaFunctionCall(
                        name=parsed.function.name,
                        arguments=arguments,
                    ),
                    type="function",
                )
            )

        # Complete wrappers, including malformed ones, are consumed once.
        self.json_tool_calls_streamed = len(matches)
        if not deltas:
            return None

        self.is_tool_call_started = False
        self.header_sent = False
        self.in_function = False
        self.json_started = True
        self.json_closed = True

        content = None
        if first_new_match is not None and first_new_match.start() > len(previous_text):
            content = current_text[len(previous_text) : first_new_match.start()]
            if not content:
                content = None
        return DeltaMessage(content=content, tool_calls=deltas)
    def _extract_tool_calls_json(self, model_output: str) -> list[ToolCall]:
        """opt23 fix=2: 提取 JSON 格式工具调用（<tool_call>{...}</tool_call> 或裸 JSON）。"""
        raw_calls: list[dict] = []
        for _m in re.finditer(
            r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
            model_output,
            re.DOTALL,
        ):
            try:
                raw_calls.append(json.loads(_m.group(1)))
            except Exception:
                pass
        if not raw_calls:
            try:
                _start = model_output.find("{")
                if _start != -1:
                    _d = json.loads(model_output[_start:])
                    if isinstance(_d, dict) and (
                        "name" in _d or "arguments" in _d
                    ):
                        raw_calls.append(_d)
            except Exception:
                pass
        tcs: list[ToolCall] = []
        for _d in raw_calls:
            _name = _d.get("name")
            _args = _d.get("arguments")
            if _name is None:
                continue
            if not isinstance(_args, str):
                _args = json.dumps(_args, ensure_ascii=False)
            tcs.append(
                ToolCall(
                    type="function",
                    function=FunctionCall(name=str(_name), arguments=_args),
                )
            )
        return tcs

    @staticmethod
    def _detect_tool_format(text: str) -> str:
        """Auto-detect tool-call format from generated text."""
        if text.find("<function=") != -1:
            return "xml"
        if '"name"' in text and '"arguments"' in text:
            return "json"
        return "xml"

    @staticmethod
    def _detect_tool_format_if_present(text: str) -> str | None:
        """Detect an explicit tool-call format; None when absent."""
        if text.find("<function=") != -1:
            return "xml"
        if '"name"' in text and '"arguments"' in text:
            return "json"
        return None

    def _resolve_tool_format(
        self,
        request: ChatCompletionRequest,
        text: str = "",
    ) -> str | None:
        """Resolve the client-selected tool-call format (config-first)."""
        kwargs = getattr(request, "chat_template_kwargs", None) or {}
        cfg = kwargs.get("tool_call_format")
        if cfg in ("xml", "json"):
            return cfg
        if not text:
            return None
        return self._detect_tool_format(text)

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """opt22d 包装：围栏遮罩 + L-A 未注册调用降级 + L-B 内容形似标签转义。"""
        # opt22d（L-C）：代码围栏内的标签先遮罩，使其不被当成真实调用，
        # 同时以（转义后的）文本形式留在 content 里，用户仍能看到示例
        self._opt22d_force_all = self._opt22d_should_force_all(request, model_output)
        info = self._extract_tool_calls_impl(
            _escape_non_linestart_tool_calls(
                _escape_fenced_tags(model_output), streaming=False
            ),
            request,
        )
        kept = []
        for tc in (info.tool_calls or []):
            if not (tc.function and self._is_registered_tool(tc.function.name)):
                continue
            # opt22e B：**必填参数兜底**。模型写文档时，正文里的完整示例会被
            # 文本层误判为真实调用；真实调用必然带齐 schema 的 required 参数，
            # 故「缺必填 ⇒ 不是真实调用」。这是**不依赖枚举形态**的通用判据
            # （实测线上 `('Bash', {})` 与正文示例两种形态都被它挡住）。
            if self._missing_required_args(tc):
                continue
            kept.append(tc)
        content = _to_display(info.content) if info.content else info.content
        if len(kept) != len(info.tool_calls or []):
            # 有未注册调用被降级 → 其原文并入内容（已转义），不向下游暴露假调用
            if not kept:
                content = _to_display(model_output)
            self.prev_tool_call_arr = [
                {"name": tc.function.name, "arguments": tc.function.arguments}
                for tc in kept
            ]
            return ExtractedToolCallInformation(
                tools_called=bool(kept),
                tool_calls=kept,
                content=content if content else None,
            )
        return ExtractedToolCallInformation(
            tools_called=info.tools_called,
            tool_calls=info.tool_calls,
            content=content if content else None,
        )

    def _extract_tool_calls_impl(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        # opt22: normalize Anthropic-style tag names before regexes
        model_output = _normalize_qwen3_tag_names(model_output)
        # opt23 §11.42 (v122 框架迁入): config-first 格式解析——serving 按
        # fix 注入 request.chat_template_kwargs.tool_call_format（fix=2→json、
        # fix=3→xml）优先；无配置则按输出形态检测（_detect_tool_format）。
        _fmt = self._resolve_tool_format(request, model_output)

        # Quick check to avoid unnecessary processing
        if _fmt == "json" or (_fmt is None and self.tool_call_prefix not in model_output):
            # opt23 §11.22: 所有 fix 均 fallback JSON 提取（模型输出
            # <tool_call>{"name"...}</tool_call> 或裸 JSON 时保证解析成功）
            json_tcs = self._extract_tool_calls_json(model_output)
            if json_tcs:
                self.prev_tool_call_arr = [
                    {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                    for tc in json_tcs
                ]
                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=json_tcs,
                    content=None,
                )
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

        try:
            function_calls = self._get_function_calls(model_output)
            if len(function_calls) == 0:
                return ExtractedToolCallInformation(
                    tools_called=False, tool_calls=[], content=model_output
                )

            tool_calls = [
                self._parse_xml_function_call(function_call_str)
                for function_call_str in function_calls
            ]
            # Populate prev_tool_call_arr for serving layer to set finish_reason
            self.prev_tool_call_arr.clear()  # Clear previous calls
            for tool_call in tool_calls:
                if tool_call:
                    self.prev_tool_call_arr.append(
                        {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        }
                    )

            # Extract content before tool calls
            content_index = model_output.find(self.tool_call_start_token)
            idx = model_output.find(self.tool_call_prefix)
            content_index = content_index if content_index >= 0 else idx
            content = model_output[:content_index]  # .rstrip()
            valid_tool_calls = [tc for tc in tool_calls if tc is not None]
            return ExtractedToolCallInformation(
                tools_called=(len(valid_tool_calls) > 0),
                tool_calls=valid_tool_calls,
                content=content if content else None,
            )

        except Exception:
            logger.exception("Error in extracting tool call from response.")
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )

    def extract_tool_calls_deferred(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """opt22d 第七轮：用完整文本解析“被推迟”的工具调用。

        仅在流式阶段确实发生过推迟时（_deferred_used 为真）由 serving 层在
        finish 时刻调用。复用非流式 extract_tool_calls——它自带围栏遮罩、
        L-A 白名单降级与标签转义等 opt22d 全套防护。

        结果剔除 _STREAMING_TOOLS 中的调用：那些已由流式路径即时下发，
        此处重复产出会让下游收到两份相同调用。
        """
        self._reset_streaming_state()
        info = self.extract_tool_calls(model_output, request)
        kept = [
            tc for tc in (info.tool_calls or [])
            if tc.function and tc.function.name not in self._STREAMING_TOOLS
        ]
        return ExtractedToolCallInformation(
            tools_called=bool(kept),
            tool_calls=kept,
            content=info.content,
        )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        """opt22d 包装：围栏遮罩 + 未注册调用过滤 + 内容转义。"""
        # opt22d（L-C）：先遮罩代码围栏内的标签，使 impl 不把它们当真实调用。
        # 遮罩是“文本 → 文本”的确定性函数；delta 直接由遮罩后的 current/previous
        # 相减得出（而非独立遮罩 delta_text），以保持流式前缀长度关系不错位。
        # opt22d 第九轮：消息结束时（delta_text 空、却带 token id ⇒ EOS）判据已完整，
        # 用非流式口径收口，把此前因"不知后文是否跟 <function="而暂存的完整标签
        # 转义后释放——否则模型只输出一个孤立标签时，该标签会被永久扣留、用户看不到。
        _is_eos = not delta_text and bool(delta_token_ids)
        # opt22e：**先遮罩、后归一化**（顺序很关键）。
        # `_normalize_qwen3_tag_names` 会把方言标签改名（`<toolcall>` → `<tool_call>`，
        # 长度还 +1）。若先归一化，**文档正文里**写的 `<toolcall>` 也会被改名
        # ⇒ 参数值（写入文件的内容）与模型原文不符，且与非流式路径（不归一化）
        # **产出一致性被打破**：实测 flush 的前缀比对在 `\`\`\`\n<toolcall>` 处
        # 分歧（sent 是 `<tool_call>`、full 是 `<toolcall>`）⇒ 补发错位、JSON 非法。
        # 先遮罩后：参数体/围栏内的标签已被换成哨兵（私有区字符），归一化正则
        # 匹配不到 ⇒ 文档内容原样保留；只有**结构标签**（未被遮罩）才被归一化。
        prev_m = _normalize_qwen3_tag_names(_mask_fenced_tags(previous_text))
        cur_m = _normalize_qwen3_tag_names(
            _mask_fenced_tags(current_text, streaming=not _is_eos)
        )
        if _is_eos:
            prev_f = _normalize_qwen3_tag_names(
                _mask_fenced_tags(previous_text, streaming=False)
            )
            if cur_m.startswith(prev_f) and len(cur_m) > len(prev_f):
                return self._opt22d_filter_delta(
                    DeltaMessage(content=cur_m[len(prev_f):])
                )
        if cur_m.startswith(prev_m):
            delta_m = cur_m[len(prev_m):]
        else:
            delta_m = _normalize_qwen3_tag_names(_mask_fenced_tags(delta_text))
        # opt22d 第七轮：开启思考时，非 _STREAMING_TOOLS 的工具不在流式阶段产出，
        # 推迟到 finish 用完整文本解析。拦截点放在 impl 之前——impl 的状态机
        # 不被推进，因而不会与 opt23 的 partial streaming（_partial_*）状态错位。
        self._thinking_enabled = self._thinking_enabled_of(request)
        if self._thinking_enabled and self._should_defer_parsing(cur_m):
            self._deferred_used = True
            return None
        self._opt22d_force_all = self._opt22d_should_force_all(request, current_text)
        delta = self._extract_tool_calls_streaming_impl(
            prev_m,
            cur_m,
            delta_m,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
            request,
        )
        # opt22e：记录全文/请求，并在「调用已开启但未收尾」时置标志，
        # 供 serving.py 的 finish 钩子调用 `flush_streaming_finish`。
        # 为何需要：奇数/未闭合代码围栏会让**真收尾**在流式期间被转义成 `⟨'…'⟩`
        # ⇒ 参数永不闭合 ⇒ 内容整段不下发；EOS 时判据完整（`streaming=False`
        # 会放行尾部真收尾），故在 finish 处用非流式路径补齐。
        self._opt22e_last_text = current_text
        self._opt22e_last_request = request
        if getattr(self, "in_function", False) or getattr(self, "in_param", False):
            self._needs_finish_flush = True
        _log("[opt22e] stream cur=%d in_fn=%s in_param=%s flush=%s sent=%s",
             len(current_text), getattr(self, "in_function", None),
             getattr(self, "in_param", None),
             getattr(self, "_needs_finish_flush", None),
             {k: len(v) for k, v in getattr(self, "_opt22e_streamed", {}).items()})
        _out = self._opt22d_filter_delta(delta)
        if _out and _out.tool_calls:
            for _t in _out.tool_calls:
                _i = _t.index if _t.index is not None else 0
                if _t.function:
                    if _t.function.name:
                        self._opt22e_name_sent[_i] = True
                    if _t.function.arguments:
                        self._opt22e_streamed[_i] = (
                            self._opt22e_streamed.get(_i, "") + _t.function.arguments
                        )
        return _out

    def flush_streaming_finish(self):
        """opt22e：EOS **强制收尾**（与 xml 侧同构，由 serving.py 的 finish 钩子调用）。

        奇数 / 未闭合代码围栏会让**真收尾**在流式期间被转义 ⇒ 参数永不闭合
        ⇒ 内容整段无法下发。EOS 时判据已完整，故用非流式路径重解析全文，
        把**尚未下发**的增量补发（已下发部分按前缀比对跳过，避免重复）。

        ⚠️ 名称只在**从未下发过**时补——重复下发会让客户端看到第二个同名调用
        （实测线上出现 `['Write', 'Write']`）。返回 `DeltaMessage`（`DeltaFunctionCall`
        允许 `name=None`），与 xml 侧返回类型一致。

        Returns:
            ``DeltaMessage``（仅含缺失增量）；无可补发时返回 None。
        """
        text = getattr(self, "_opt22e_last_text", "")
        req = getattr(self, "_opt22e_last_request", None)
        if not text or req is None:
            return None
        try:
            info = self.extract_tool_calls(text, req)
        except Exception:
            return None
        _log("[opt22e] flush text=%d calls=%s", len(text),
             [t.function.name for t in (info.tool_calls or [])] if info else None)
        if not info or not info.tool_calls:
            return None
        out = []
        for i, tc in enumerate(info.tool_calls):
            full = (tc.function.arguments or "") if tc.function else ""
            sent = self._opt22e_streamed.get(i, "")
            # ⚠️ 用**最长公共前缀**补缺，而非 `full.startswith(sent)`：
            # partial 流式路径会多下发 1 个尾字符（参数值末尾的 `\n`，完成路径
            # 会 trim 掉），此时 sent 比 full **长** ⇒ `startswith` 为假 ⇒ 会走
            # "覆盖式补发"把整块 JSON 再发一遍（实测线上 args 翻倍、JSON "Extra data"）。
            # LCP 只补真正缺失的尾部；sent 多出的尾字符落在 JSON 闭合之后，
            # 不影响合法性（如 `…XYZEND\n"}`）。
            # 注：`find_common_prefix` 返回**公共前缀字符串**（非长度）。
            common = find_common_prefix(sent, full) if sent else ""
            delta = full[len(common):]
            name = None
            if not self._opt22e_name_sent.get(i):
                name = tc.function.name if tc.function else None
            _log("[opt22e] flush[%d] name_sent=%s sent=%d full=%d lcp=%d delta=%d",
                 i, self._opt22e_name_sent.get(i), len(sent), len(full),
                 len(common) if isinstance(common, str) else -1, len(delta))
            _lc = len(common) if isinstance(common, str) else 0
            if _lc < min(len(sent), len(full)):
                _log("[opt22e]   diverge@%d sent[..]=%r", _lc,
                     sent[max(0, _lc - 25):_lc + 45])
                _log("[opt22e]   diverge@%d full[..]=%r", _lc,
                     full[max(0, _lc - 25):_lc + 45])
            if not delta and not name:
                continue
            out.append(
                DeltaToolCall(
                    index=i,
                    id=tc.id,
                    type="function",
                    function=DeltaFunctionCall(name=name, arguments=delta),
                )
            )
        return DeltaMessage(tool_calls=out) if out else None

    def _extract_tool_calls_streaming_impl(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        # opt22: normalize Anthropic-style tag names (consistent across all three)
        previous_text = _normalize_qwen3_tag_names(previous_text)
        current_text = _normalize_qwen3_tag_names(current_text)
        delta_text = _normalize_qwen3_tag_names(delta_text)
        # Store request for type conversion
        if not previous_text:
            self._reset_streaming_state()
            self.streaming_request = request

        # If no delta text, return None unless it's an EOS token after tools
        if not delta_text:
            # Check if this is an EOS token after all tool calls are complete
            # Check for tool calls in text even if is_tool_call_started
            # is False (might have been reset after processing all tools)
            if delta_token_ids and self.tool_call_end_token_id not in delta_token_ids:
                # Count complete tool calls
                complete_calls = len(
                    self.tool_call_complete_regex.findall(current_text)
                )

                # If we have completed tool calls and populated
                # prev_tool_call_arr
                if complete_calls > 0 and len(self.prev_tool_call_arr) > 0:
                    # Check if all tool calls are closed
                    open_calls = current_text.count(
                        self.tool_call_start_token
                    ) - current_text.count(self.tool_call_end_token)
                    if open_calls == 0:
                        # Return empty delta for finish_reason processing
                        return DeltaMessage(content="")
                elif not self.is_tool_call_started and current_text:
                    # This is a regular content response that's now complete
                    return DeltaMessage(content="")
            return None

        # Update accumulated text
        self.accumulated_text = current_text

        # Qwen/OpenCode may emit a JSON body inside the same unambiguous
        # <tool_call> wrapper used by the XML format. Handle a complete JSON
        # call before the XML state machine consumes the wrapper and waits for
        # a <function= header that will never arrive.
        json_delta = self._extract_json_tool_calls_streaming(
            previous_text,
            current_text,
        )
        if json_delta is not None:
            return json_delta

        # Check if we need to advance to next tool
        if self.json_closed and not self.in_function:
            # 优化 B: JSON 格式无 </tool_call> 标记，参数完整后直接推进
            if self._json_tool_active:
                self.current_tool_index += 1
                self.header_sent = False
                self.param_count = 0
                self.json_started = False
                self.json_closed = False
                self.accumulated_params = {}
                self.is_tool_call_started = False
                # Clear active flag so Section 3 can re-detect or release
                # subsequent content, and Section 4 won't re-enter handler
                # for an already-processed tool call. fix=2 时保持 JSON 激活。
                self._json_tool_active = self._fix_force_json
                # opt23: reset partial streaming state
                self._partial_emit_offset = 0
                self._partial_param_start = -1
                self._partial_emitted = ""
                self._partial_prefix = ""
                self._pending_brace = False
                self._func_dup_detected = False
                self._stream_last_fp = None
                self._stream_partial_name = None
                self._stream_partial_value = None
                return None

            # Check if this tool call has ended (XML format)
            tool_ends = current_text.count(self.tool_call_end_token)
            if tool_ends > self.current_tool_index:
                # This tool has ended, advance to next
                self.current_tool_index += 1
                self.header_sent = False
                self.param_count = 0
                self.json_started = False
                self.json_closed = False
                self.accumulated_params = {}
                # opt23: reset partial streaming state for next tool call
                self._partial_emit_offset = 0
                self._partial_param_start = -1
                self._partial_emitted = ""
                self._partial_prefix = ""
                self._pending_brace = False
                self._func_dup_detected = False
                self._stream_last_fp = None
                self._stream_partial_name = None
                self._stream_partial_value = None

                # Check if there are more tool calls
                tool_starts = current_text.count(self.tool_call_start_token)
                if self.current_tool_index >= tool_starts:
                    # No more tool calls
                    self.is_tool_call_started = False
                # Continue processing next tool
                return None

        # Handle normal content before tool calls
        if not self.is_tool_call_started:
            # opt23 fix=2: 强制 JSON 路由——<tool_call> 标记（模板 json 分支
            # 引导模型输出 <tool_call> 包裹 JSON）或 JSON 特征（"name"/
            # "arguments"）出现即激活。不能只等 "name" 特征：模型输出
            # <tool_call>\n{"name":...} 时 <tool_call> 先到达，XML 检测会
            # 抢先激活（expat 解析 JSON 失败吞掉工具调用）。无条件强制
            # 则会吞纯文本正文（think 后无正文根因）。<function= 表示
            # 模型实际输出 XML，不在此激活（走 XML 解析路径）。
            if self._fix_force_json:
                _json_name = current_text.find('"name"')
                _json_args = current_text.find('"arguments"')
                if (current_text.find(self.tool_call_start_token) != -1
                        or (_json_name != -1 and _json_args != -1
                            and _json_name < _json_args)):
                    self._json_tool_active = True
                    self.is_tool_call_started = True
                    return None
            # opt23 Path C: detect JSON-format tool calls before falling
            # into XML detection.  JSON format looks like:
            #   {"name": "Write", "arguments": {...}}
            # Only activate when no XML markers are present (otherwise
            # a JSON-looking substring inside XML content could trigger).
            # 优化 C: 从已处理 JSON 工具调用的结束位置之后搜索，
            # 避免重新检测到同一个已完成的工具调用。
            _json_search = max(self._json_processed_end, 0)
            _json_name = current_text.find('"name"', _json_search)
            _json_args = current_text.find('"arguments"', _json_search)
            if (_json_name != -1
                    and _json_args != -1
                    and _json_name < _json_args
                    and current_text.find(self.tool_call_prefix) == -1):
                self._json_tool_active = True
                self.is_tool_call_started = True
                return None  # routing takes effect on next call

            # Check if tool call is starting (XML)
            tool_start_candidates = [
                pos
                for pos in (
                    current_text.find(self.tool_call_start_token),
                    current_text.find(self.tool_call_prefix),
                )
                if pos != -1
            ]
            tool_start = min(tool_start_candidates) if tool_start_candidates else -1
            # opt22d 第九轮：token 层判定必须与**文本层**一致。
            # ``<tool_call>`` 是 special token（id 248058），L-C/L-D 把它在文本里
            # 转义成 ``⟨'tool_call'⟩`` 后，token id 仍留在 delta_token_ids 里
            # ⇒ 若只看 token，会把"已判定为展示示例"的内容重新当成调用起点，
            #   ``is_tool_call_started=True`` 且此后每轮都 ``return None``
            #   ⇒ **正文自该标签起全部丢失**（实测 E1 流式：944 token 全生成，
            #   却只下发 17 字符；日志显示首 4 个 delta 正常、第 5 个起恒 None）。
            # 故要求文本里确实存在**未转义**的裸标签才认。
            if tool_start != -1 or (
                self.tool_call_start_token_id in delta_token_ids
                and self.tool_call_start_token in current_text
            ):
                self.is_tool_call_started = True
                # Return any content before the tool call
                if tool_start > len(previous_text):
                    content_before = (
                        self._trailing_tool_marker_prefix(previous_text)
                        + delta_text[: tool_start - len(previous_text)]
                    )
                    if content_before:
                        return DeltaMessage(content=content_before)
                return None
            else:
                pending_prefix = self._pending_tool_marker_delta_prefix(
                    previous_text, current_text, delta_text
                )
                if pending_prefix is not None:
                    if pending_prefix:
                        return DeltaMessage(content=pending_prefix)
                    return None
                # Check if we're between tool calls - skip whitespace
                if (
                    current_text.rstrip().endswith(self.tool_call_end_token)
                    and delta_text.strip() == ""
                ):
                    # We just ended a tool call, skip whitespace
                    return None
                # Normal content, no tool call
                return DeltaMessage(content=delta_text)

        # opt23 §11.27: JSON 激活时用 safe-prefix diff 增量发射（fix=1/2 的
        # JSON 真流式——复刻 vLLM 主线 PR #45413 的 _compute_arg_delta）。
        # 退休时认为「JSON format does not occur in practice」，但前端 JSON
        # 指令会让模型输出 JSON——恢复接入，前端要流式 json 就提供增量。
        if self._json_tool_active:
            _json_delta = self._handle_json_tool_streaming(current_text, delta_text)
            if _json_delta is not None:
                return _json_delta
            # opt23 fix=2: JSON 路由激活后不得落入 original incremental
            # path——双路径竞争输出导致增量不一致（original path 未转义
            # 真实控制字符/解码 \n，Write content 参数非法根因）。
            return None

        # v1.2.2+: =1, =2, =3, =5 all use the original incremental path
        # (json_fragments loop + _is_streaming_tool enhancements for
        # Write/Edit).  The JSON-diff and XML-diff handlers are retired
        # — they emitted one combined delta per step which caused
        # CC-HAHA SSE truncation for long-content tool calls.
        # Qwen3Coder's structural tag is always XML, so JSON format
        # does not occur in practice regardless of client format.
        if VLLM_QWEN3X_TOOL_FIX != 0:
            _log(
                "opt23 original path: fix=%s json_closed=%s func=%s "
                "delta_len=%d",
                VLLM_QWEN3X_TOOL_FIX,
                self.json_closed, self.current_function_name,
                len(delta_text))

        # Check if we're between tool calls (waiting for next one)
        # Count tool calls we've seen vs processed
        tool_start_positions = self._tool_start_positions(current_text)
        tool_starts_count = len(tool_start_positions)
        if self.current_tool_index >= tool_starts_count:
            # We're past all tool calls, shouldn't be here
            return None

        # We're in a tool call, find the current tool call portion
        # Need to find the correct tool call based on current_tool_index
        tool_start_idx = tool_start_positions[self.current_tool_index]
        # Find where this tool call ends (or current position if not ended yet)
        tool_end_idx = current_text.find(self.tool_call_end_token, tool_start_idx)
        if tool_end_idx == -1:
            tool_text = current_text[tool_start_idx:]
        else:
            tool_text = current_text[
                tool_start_idx : tool_end_idx + len(self.tool_call_end_token)
            ]

        # Looking for function header
        if not self.header_sent:
            if self.tool_call_prefix in tool_text:
                func_start = tool_text.find(self.tool_call_prefix) + len(
                    self.tool_call_prefix
                )
                func_end = tool_text.find(">", func_start)

                if func_end != -1:
                    # Found complete function name
                    self.current_function_name = tool_text[func_start:func_end]
                    self.current_tool_id = self._generate_tool_call_id()
                    self.header_sent = True
                    self.in_function = True

                    # A speculative burst can finish the entire function
                    # before the parser has emitted its header. Do not rely on
                    # a later delta to revisit the already-buffered body: the
                    # next delta may only contain </tool_call> plus EOS. Parse
                    # the complete body now and emit its arguments together
                    # with the header.
                    complete_arguments: str | None = None
                    func_content_end = tool_text.find(
                        self.function_end_token, func_start
                    )
                    if func_content_end != -1:
                        parsed_tool = self._parse_xml_function_call(
                            tool_text[func_start:func_content_end]
                        )
                        if parsed_tool is not None:
                            complete_arguments = parsed_tool.function.arguments

                    # Always append — each tool call is a separate
                    # invocation even if the function name is the same
                    # (e.g. two consecutive "read" calls).
                    self.prev_tool_call_arr.append(
                        {
                            "name": self.current_function_name,
                            "arguments": complete_arguments or "{}",
                        }
                    )

                    # Initialize streamed args tracking for this tool.
                    # The serving layer reads streamed_args_for_tool to
                    # compute remaining arguments at stream end. Without
                    # this, IndexError occurs when the serving layer
                    # accesses streamed_args_for_tool[index].
                    self.streamed_args_for_tool.append(complete_arguments or "")

                    if complete_arguments is not None:
                        self.json_started = True
                        self.json_closed = True
                        self.in_function = False

                    # Send header with function info
                    return DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.current_tool_index,
                                id=self.current_tool_id,
                                function=DeltaFunctionCall(
                                    name=self.current_function_name,
                                    arguments=complete_arguments or "",
                                ),
                                type="function",
                            )
                        ]
                    )
            return None

        # We've sent header, now handle function body
        if self.in_function:
            # Always send opening brace first, regardless of whether
            # parameter_prefix is in the current delta. With speculative
            # decoding, a single delta may contain both the opening brace
            # and parameter data; skipping "{" here would desync
            # json_started from what was actually streamed.
            if not self.json_started:
                self.json_started = True
                if self._is_streaming_tool:
                    # opt23: if a parameter tag is already in tool_text,
                    # defer "{" and combine it with the first param delta
                    # so CC-HAHA receives a parseable partial_json.
                    # Otherwise fall back to bare "{" — the original
                    # behavior that the model can recover from when
                    # params arrive in a subsequent delta.
                    if tool_text.find(self.parameter_prefix) != -1:
                        self._pending_brace = True
                        # Fall through to parameter processing
                    else:
                        if self.current_tool_index < len(
                                self.streamed_args_for_tool):
                            self.streamed_args_for_tool[
                                self.current_tool_index] += "{"
                        return DeltaMessage(tool_calls=[
                            DeltaToolCall(index=self.current_tool_index,
                                function=DeltaFunctionCall(arguments="{"))
                        ])
                else:
                    if self.current_tool_index < len(self.streamed_args_for_tool):
                        self.streamed_args_for_tool[self.current_tool_index] += "{"
                    return DeltaMessage(tool_calls=[
                        DeltaToolCall(index=self.current_tool_index,
                            function=DeltaFunctionCall(arguments="{"))
                    ])

            param_config = find_tool_properties(
                self.tools, self.current_function_name or ""
            )
            param_starts = self._parameter_start_positions(tool_text, param_config)

            # opt23: detect model duplication of <function=...> within
            # the same tool_text.  Gated by env var — this is a streaming
            # artifact detection heuristic.
            _func_positions: list = []
            if VLLM_QWEN3X_TOOL_FIX:
                _known_names = {t.function.name for t in (self.tools or [])
                                if getattr(t, 'function', None)
                                and getattr(t.function, 'name', None)}
                _fsi = 0
                while True:
                    _fsi = tool_text.find(self.tool_call_prefix, _fsi)
                    if _fsi == -1:
                        break
                    _close = tool_text.find(">", _fsi + len(self.tool_call_prefix))
                    if _close != -1:
                        _name = tool_text[
                            _fsi + len(self.tool_call_prefix):_close
                        ]
                        if _name in _known_names:
                            _func_positions.append(_fsi)
                    _fsi += len(self.tool_call_prefix)

                # Exclude <function=...> tags that appear inside
                # parameter values.  Once the first <parameter= opens,
                # any subsequent "<function=" is content text, not a
                # real function tag.  Without this filter the detector
                # falsely triggers when Write/Edit content happens to
                # mention tool-call syntax (e.g. documenting tool usage).
                if param_starts:
                    _func_positions = [p for p in _func_positions
                                       if p < param_starts[0]]

                if len(_func_positions) > 1:
                    # Model re-emitted the function tag. Use the LAST
                    # <function=...> as the authoritative position and
                    # ignore parameters that appear before it (they
                    # belong to the stale/duplicated first function).
                    _last_func = _func_positions[-1]
                    param_starts = [p for p in param_starts if p > _last_func]
                    # Only reset param_count the first time we detect
                    # the duplication for this tool call.
                    if not self._func_dup_detected:
                        self._func_dup_detected = True
                        self.param_count = 0


            # Process ALL complete params in a loop (spec decode fix).
            # With speculative decoding a single delta can deliver
            # multiple complete parameters at once. The old single-pass
            # code would process one and ``return None`` if the next was
            # incomplete — skipping any already-complete params that
            # preceded it. Using a loop with ``break`` instead ensures
            # we emit every complete parameter before yielding control.
            json_fragments = []
            while not self.in_param and self.param_count < len(param_starts):
                param_idx = param_starts[self.param_count]
                param_start = param_idx + len(self.parameter_prefix)
                remaining = tool_text[param_start:]

                if ">" not in remaining:
                    break

                name_end = remaining.find(">")
                current_param_name = remaining[:name_end]

                value_start = param_start + name_end + 1
                next_param_idx = (
                    param_starts[self.param_count + 1]
                    if self.param_count + 1 < len(param_starts)
                    else None
                )
                function_end_idx = tool_text.find(self.function_end_token, value_start)
                tool_end_idx = tool_text.find(self.tool_call_end_token, value_start)
                boundaries = [
                    boundary
                    for boundary in (function_end_idx, tool_end_idx)
                    if boundary != -1
                ]
                container_end = min(boundaries) if boundaries else None
                value_end = self._find_structural_parameter_end(
                    tool_text,
                    value_start=value_start,
                    next_parameter_start=next_param_idx,
                    container_end=container_end,
                    allow_text_end=False,
                )
                if value_end is None:
                    if next_param_idx is not None:
                        # Recover a missing </parameter> before a real next tag.
                        value_end = next_param_idx
                    elif container_end is not None:
                        # Recover a missing final close before function/tool end.
                        value_end = container_end
                    else:
                        # Wait for structural lookahead. The current suffix may
                        # be literal </parameter> text inside a long value.
                        break

                param_value = tool_text[value_start:value_end]
                if param_value.startswith("\n"):
                    param_value = param_value[1:]
                if param_value.endswith("\n"):
                    param_value = param_value[:-1]

                self.current_param_name = current_param_name
                self.accumulated_params[current_param_name] = param_value

                converted_value = self._convert_param_value(
                    param_value,
                    current_param_name,
                    param_config,
                    self.current_function_name or "",
                )

                serialized_value = json.dumps(converted_value, ensure_ascii=False)

                if self.param_count == 0:
                    json_fragment = f'"{current_param_name}": {serialized_value}'
                else:
                    json_fragment = f', "{current_param_name}": {serialized_value}'

                # opt23: If this parameter was partially streamed, emit
                # only the delta (remaining content + closing quote).
                if self._partial_emitted:
                    emitted_total = len(self._partial_emitted)
                    if len(json_fragment) > emitted_total:
                        json_fragment = json_fragment[emitted_total:]
                    else:
                        # opt23 §11.40: partial 发射可能比完整参数长——
                        # XML 参数值以换行包裹（<parameter=x>\n值\n</parameter>），
                        # partial 发射保留尾部格式换行而完整参数 trim 掉，
                        # 导致 emitted_total > len(json_fragment)。此时内容
                        # 已全部发射，缺失的只是闭合引号——补发最后一个
                        # 字符（闭合 "），否则前端拼接 JSON 非法
                        # （Edit old_string 未闭合根因）。
                        json_fragment = (
                            json_fragment[-1:]
                            if json_fragment.endswith('"') else ""
                        )
                    self._partial_emitted = ""
                    self._partial_prefix = ""
                    if not json_fragment:
                        # All content was already streamed; skip
                        # duplicate emission but still advance counters.
                        self.param_count += 1
                        continue

                self.param_count += 1
                json_fragments.append(json_fragment)

            # opt23 XML→JSON 互转 (=4): build complete JSON from accumulated
            # params (+ partial value) and emit via safe-prefix diffing.
            # This bypasses the json_fragments + partial streaming paths
            # entirely, producing only valid JSON continuation deltas.
            # Only for =4 (双格式互转, 待设计), =3 is pure JSON no conversion.
            if (VLLM_QWEN3X_TOOL_FIX == 4
                    and self.current_function_name in self._STREAMING_TOOLS):
                _ph3 = self._emit_xml_json_diff(
                    tool_text, param_starts, tool_start_idx, current_text,
                )
                if _ph3 is not None:
                    return _ph3

            if json_fragments:
                combined = "".join(json_fragments)
                # opt23: when _pending_brace deferred the "{", prepend
                # it to the first json_fragment.
                if self._pending_brace:
                    combined = "{" + combined
                    self._pending_brace = False

                if self.current_tool_index < len(self.streamed_args_for_tool):
                    self.streamed_args_for_tool[self.current_tool_index] += combined
                else:
                    logger.warning(
                        "streamed_args_for_tool out of sync: index=%d len=%d",
                        self.current_tool_index,
                        len(self.streamed_args_for_tool),
                    )

                # A speculative burst (or stream_interval > 1) can complete
                # the final parameter and close the function in one parser
                # call. Returning only the parameter fragment loses the final
                # JSON brace because the next delta may be EOS/empty.
                if not self.json_closed and self.function_end_token in tool_text:
                    self._finish_streaming_function(tool_text)
                    combined += "}"

                return DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_index,
                            function=DeltaFunctionCall(arguments=combined),
                        )
                    ]
                )

            # opt23 Path B enhanced: partial streaming starts AFTER
            # file_path has been fully parsed (appears in accumulated_params).
            # This protects file_path from fragmentation regardless of
            # parameter order (e.g. Edit [replace_all, file_path, ...]).
            # Non-string params (replace_all boolean) are already blocked
            # by the type gate below.  All other string params (old_string,
            # content, new_string) get incremental streaming — even large
            # old_string values don't block the frontend.
            _param_config = find_tool_properties(
                self.tools, self.current_function_name or "",
            )
            _expected_total = len(_param_config) if _param_config else 0

            if (not json_fragments
                    and self._is_streaming_tool
                    and self.in_function
                    and self.json_started
                    and self.param_count > 0
                    and self.param_count < len(param_starts)
                    and _expected_total > 1
                    and self.accumulated_params.get("file_path") is not None
                    and self.current_tool_index < len(self.streamed_args_for_tool)):
                incomplete_start = param_starts[self.param_count]
                if incomplete_start != self._partial_param_start:
                    self._partial_param_start = incomplete_start
                    self._partial_emit_offset = 0
                    self._partial_emitted = ""
                    self._partial_prefix = ""

                param_text = tool_text[incomplete_start
                                       + len(self.parameter_prefix):]
                if ">" in param_text:
                    name_end = param_text.find(">")
                    param_name = param_text[:name_end]
                    # opt23: Partial streaming only for string-type
                    # parameters.  Boolean (replace_all), integer etc.
                    # must be fully parsed and type-coerced via
                    # _convert_param_value.  Raw streaming bypasses
                    # type coercion, producing "false" vs false
                    # mismatches that break remaining-delta computation
                    # and cause CC-HAHA to receive malformed JSON.
                    _pc = _param_config  # reuse from gate above
                    _ps = _pc.get(param_name, {})
                    _pt = extract_types_from_schema(_ps)
                    if _pt and "string" not in _pt:
                        return None

                    if not self._partial_prefix:
                        if self.param_count == 0:
                            _prefix = f'"{param_name}": "'
                        else:
                            _prefix = f', "{param_name}": "'
                        self._partial_prefix = _prefix
                    value_start = (incomplete_start
                                   + len(self.parameter_prefix)
                                   + name_end + 1)
                    # value_start is relative to tool_text. Adjust to
                    # absolute position in current_text for slicing.
                    _abs_value_start = tool_start_idx + value_start
                    if (_abs_value_start < len(current_text)
                            and current_text[_abs_value_start:_abs_value_start + 1]
                            == "\n"):
                        _abs_value_start += 1

                    # opt22e：本路径**绕过 `_convert_param_value`** 直接发射原始切片，
                    # 故须在此单独还原遮罩转义，否则流式下参数值仍是 `⟨'…'⟩`
                    # （与完成路径的还原结果不一致 ⇒ 前缀比对失配、JSON 非法）。
                    current_value = _to_original(
                        current_text[_abs_value_start:]
                    )
                    if len(current_value) > self._partial_emit_offset:
                        _now = time.monotonic()
                        if (_now - self._last_partial_time) < 0.25:
                            return None
                        _raw_new = current_value[
                            self._partial_emit_offset:]
                        # opt23: strip trailing XML close-tag fragments
                        # (</parameter>, </function>, …) so they don't
                        # leak into the streamed JSON string value.
                        _safe_len = self._safe_content_length(_raw_new)
                        if _safe_len == 0:
                            return None
                        new_content = _raw_new[:_safe_len]
                        self._partial_emit_offset += _safe_len
                        if new_content:
                            escaped = json.dumps(
                                new_content, ensure_ascii=False)[1:-1]
                            if not self._partial_emitted:
                                fragment = self._partial_prefix + escaped
                                self._partial_emitted += fragment
                            else:
                                fragment = escaped
                                self._partial_emitted += fragment
                            if self.current_tool_index < len(
                                    self.streamed_args_for_tool):
                                self.streamed_args_for_tool[
                                    self.current_tool_index] += fragment
                            self._last_partial_time = _now
                            return DeltaMessage(
                                tool_calls=[
                                    DeltaToolCall(
                                        index=self.current_tool_index,
                                        function=DeltaFunctionCall(
                                            arguments=fragment),
                                    )
                                ]
                            )
                    return None

            if (not json_fragments
                    and self.param_count < len(param_starts)
                    and self.current_tool_index < len(self.streamed_args_for_tool)):
                return None

            # Check for function end AFTER processing parameters.
            # This ordering is critical: with speculative decoding a
            # burst can deliver the final parameter value together with
            # </function>. If the close check ran first it would emit
            # "}" and set in_function=False before the parameter loop
            # ever ran, causing the parameter to be silently dropped.
            if not self.json_closed and self.function_end_token in tool_text:
                # opt23: when _pending_brace deferred "{" but no
                # <parameter=...> tags arrived, fall back to bare
                # "{" + "}" — the original behavior that the model
                # can recover from (user empirically confirmed).
                if self._pending_brace and not param_starts:
                    if self.current_tool_index < len(
                            self.streamed_args_for_tool):
                        self.streamed_args_for_tool[
                            self.current_tool_index] += "{"
                    self._pending_brace = False
                    # Fall through to emit "}" via function-end handler

                self.json_closed = True

                func_start = tool_text.find(self.tool_call_prefix) + len(
                    self.tool_call_prefix
                )
                func_content_end = tool_text.find(self.function_end_token, func_start)
                if func_content_end != -1:
                    func_content = tool_text[func_start:func_content_end]
                    try:
                        parsed_tool = self._parse_xml_function_call(
                            func_content,
                        )
                        if parsed_tool and self.current_tool_index < len(
                            self.prev_tool_call_arr
                        ):
                            args = parsed_tool.function.arguments
                            self.prev_tool_call_arr[self.current_tool_index][
                                "arguments"
                            ] = args
                            _log(
                                "opt23 main func_end: tool=%s idx=%s "
                                "args=%s streamed=%s",
                                self.current_function_name,
                                self.current_tool_index,
                                args,
                                self.streamed_args_for_tool[
                                    self.current_tool_index]
                                if self.current_tool_index < len(
                                    self.streamed_args_for_tool)
                                else "N/A")
                            # opt23 Fix B: detect empty tool calls
                            # (model emitted <function=NAME></function>
                            # with no <parameter=...> tags).
                            if args == "{}" and not self._is_streaming_tool:
                                logger.warning(
                                    "Qwen3Coder streaming: tool '%s' "
                                    "has no parameters (empty {}) — "
                                    "model error, call will fail",
                                    self.current_function_name or "?",
                                )
                    except Exception:
                        logger.debug(
                            "Failed to parse tool call during streaming: %s",
                            tool_text,
                            exc_info=True,
                        )

                # opt23: for streaming tools (Write/Edit), compute the real
                # remaining delta from the full parsed JSON minus what was
                # already streamed, instead of emitting "}" as a bare
                # punctuation delta that clients cannot parse.
                remaining = "}"
                if self._is_streaming_tool:
                    if self.current_tool_index < len(self.prev_tool_call_arr):
                        full_json = self.prev_tool_call_arr[
                            self.current_tool_index
                        ].get("arguments", "")
                        streamed = (
                            self.streamed_args_for_tool[self.current_tool_index]
                            if self.current_tool_index < len(
                                self.streamed_args_for_tool)
                            else ""
                        )
                        if full_json and full_json.startswith(streamed):
                            remaining = full_json[len(streamed):]
                self._pending_brace = False

                if self.current_tool_index < len(self.streamed_args_for_tool):
                    self.streamed_args_for_tool[self.current_tool_index] += remaining
                else:
                    logger.warning(
                        "streamed_args_for_tool out of sync: index=%d len=%d",
                        self.current_tool_index,
                        len(self.streamed_args_for_tool),
                    )

                result = DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.current_tool_index,
                            function=DeltaFunctionCall(arguments=remaining),
                        )
                    ]
                )

                self.in_function = False
                self.json_closed = True
                self.accumulated_params = {}

                return result

        return None

    def get_structural_tag(self, request: ChatCompletionRequest):
        tag = get_model_structural_tag(
            model="qwen_3_5",
            tools=request.tools,
            tool_choice=request.tool_choice,
            reasoning=get_enable_structured_outputs_in_reasoning(),
        )
        # v1.2.2+: 不再尝试检测客户端格式。Qwen3Coder 的 structural tag
        # 始终是 XML 格式（str(tag) 返回 Python repr，不含 "name"），
        # 所以 _json_tool_active 始终为 False。所有工具走原始路径。
        return tag

