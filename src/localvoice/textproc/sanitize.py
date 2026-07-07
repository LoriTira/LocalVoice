import re

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_FENCE = "```"
_CODE_MARKER = "Code omitted."

_EMOJI = re.compile(
    "["
    "\U0001f300-\U0001faff"
    "\U00002600-\U000027bf"
    "\U0001f000-\U0001f0ff"
    "⬀-⯿←-⇿️"
    "]+"
)


def _longest_suffix_prefix(text: str, token: str) -> int:
    for k in range(min(len(text), len(token) - 1), 0, -1):
        if text.endswith(token[:k]):
            return k
    return 0


_CH_OPEN = "<|channel>"
_CH_CLOSE = "<channel|>"
_CH_THOUGHT = "thought"
# A channel name is a short lowercase word; anything longer than this without
# a terminator is not a channel marker and must flow through as literal text
# rather than being held back forever.
_CH_NAME_MAX = 24


class ChannelThinkTranslator:
    """Normalize Gemma-4-style reasoning channels to canonical think tags.

    Gemma 4 templates wrap reasoning as ``<|channel>thought\\n ... <channel|>``
    (with thinking disabled the template pre-closes an empty thought channel,
    so nothing streams). The rest of the pipeline — TextFilter, the reasoning
    protocol event, transcripts — speaks only Qwen's ``<think>``/``</think>``,
    so the LLM engine runs this translator over the raw stream for
    channel-style models and downstream code stays model-agnostic.

    Non-``thought`` channels pass through untouched: they are not reasoning,
    and a future tool-calling layer will want to see them.
    """

    def __init__(self) -> None:
        self._buf = ""
        self._in_thought = False

    def feed(self, delta: str) -> str:
        self._buf += delta
        out: list[str] = []
        while True:
            if self._in_thought:
                j = self._buf.find(_CH_CLOSE)
                if j < 0:
                    hold = _longest_suffix_prefix(self._buf, _CH_CLOSE)
                    out.append(self._buf[: len(self._buf) - hold])
                    self._buf = self._buf[len(self._buf) - hold :]
                    break
                out.append(self._buf[:j])
                out.append(_THINK_CLOSE)
                self._buf = self._buf[j + len(_CH_CLOSE) :]
                self._in_thought = False
                continue

            i_open = self._buf.find(_CH_OPEN)
            i_close = self._buf.find(_CH_CLOSE)
            candidates = [(i, t) for i, t in ((i_open, _CH_OPEN), (i_close, _CH_CLOSE)) if i >= 0]
            if not candidates:
                hold = max(
                    _longest_suffix_prefix(self._buf, _CH_OPEN),
                    _longest_suffix_prefix(self._buf, _CH_CLOSE),
                )
                out.append(self._buf[: len(self._buf) - hold])
                self._buf = self._buf[len(self._buf) - hold :]
                break
            i, tok = min(candidates)
            if tok == _CH_CLOSE:  # close with no open: swallow, never speak it
                out.append(self._buf[:i])
                self._buf = self._buf[i + len(_CH_CLOSE) :]
                continue

            name_start = i + len(_CH_OPEN)
            name_end = name_start
            while name_end < len(self._buf) and self._buf[name_end].isalpha():
                name_end += 1
            if name_end == len(self._buf) and name_end - name_start <= _CH_NAME_MAX:
                # Marker present but the channel name may continue in the next
                # delta — hold from the marker on.
                out.append(self._buf[:i])
                self._buf = self._buf[i:]
                break
            name = self._buf[name_start:name_end]
            out.append(self._buf[:i])
            if name == _CH_THOUGHT:
                self._buf = self._buf[name_end:]
                if self._buf.startswith("\n"):
                    self._buf = self._buf[1:]
                out.append(_THINK_OPEN)
                self._in_thought = True
            else:  # unknown channel: emit the marker literally and move on
                out.append(self._buf[i:name_end])
                self._buf = self._buf[name_end:]
        return "".join(out)

    def finish(self) -> str:
        out, self._buf = self._buf, ""
        if self._in_thought:
            # Cancelled or truncated mid-think: close the canonical block so
            # TextFilter surfaces the partial reasoning instead of holding it.
            self._in_thought = False
            return out + _THINK_CLOSE
        return out


class TextFilter:
    def __init__(self, on_think=None) -> None:
        self._buf = ""
        self._in_think = False
        self._in_fence = False
        self._on_think = on_think
        self._think_parts: list[str] = []

    def feed(self, delta: str) -> str:
        self._buf += delta
        out: list[str] = []
        while True:
            if self._in_think or self._in_fence:
                token = _THINK_CLOSE if self._in_think else _FENCE
                j = self._buf.find(token)
                if j < 0:
                    hold = _longest_suffix_prefix(self._buf, token)
                    if self._in_think:
                        self._think_parts.append(self._buf[: len(self._buf) - hold])
                    self._buf = self._buf[len(self._buf) - hold :]
                    break
                if self._in_think:
                    self._think_parts.append(self._buf[:j])
                    self._emit_think()
                self._buf = self._buf[j + len(token) :]
                self._in_think = False
                self._in_fence = False
                continue
            i_think = self._buf.find(_THINK_OPEN)
            i_fence = self._buf.find(_FENCE)
            candidates = [(i, t) for i, t in ((i_think, _THINK_OPEN), (i_fence, _FENCE)) if i >= 0]
            if not candidates:
                hold = max(
                    _longest_suffix_prefix(self._buf, _THINK_OPEN),
                    _longest_suffix_prefix(self._buf, _FENCE),
                )
                emit_upto = len(self._buf) - hold
                out.append(self._buf[:emit_upto])
                self._buf = self._buf[emit_upto:]
                break
            i, tok = min(candidates)
            out.append(self._buf[:i])
            self._buf = self._buf[i + len(tok) :]
            if tok == _THINK_OPEN:
                self._in_think = True
            else:
                self._in_fence = True
                nl = self._buf.find("\n")
                if nl >= 0:
                    self._buf = self._buf[nl + 1 :]
                out.append(_CODE_MARKER)
        return "".join(out)

    def finish(self) -> str:
        if self._in_think or self._in_fence:
            if self._in_think:
                self._think_parts.append(self._buf)
                self._emit_think()  # unclosed think: still surface what we have
            self._buf = ""
            return ""
        out, self._buf = self._buf, ""
        return out

    def force_close_think(self) -> None:
        # A tool marker captured mid-stream inside an unclosed <think> block
        # would leave the filter stuck in think mode across the continuation
        # round, swallowing the whole next answer as reasoning. Called right
        # after a call is captured: surface the partial thought and reset think
        # state so the next feed() speaks. Mirrors finish()'s think branch; a
        # no-op (and leaves an open code fence alone) when not in think.
        if not self._in_think:
            return
        self._think_parts.append(self._buf)
        self._emit_think()
        self._buf = ""
        self._in_think = False

    def _emit_think(self) -> None:
        text = "".join(self._think_parts).strip()
        self._think_parts = []
        if self._on_think is not None and text:
            self._on_think(text)


_SPECIAL_MARKER_RE = re.compile(r"<\|[^|>]{0,32}\|?>")
_BARE_SPECIAL_MARKERS = (
    "<channel|>",
    "<tool_call|>",
    "<tool_response|>",
    "<turn|>",
    _THINK_OPEN,
    _THINK_CLOSE,
)


def strip_special_markers(text: str) -> str:
    """Strip chat-template control-token sequences from untrusted text.

    Tool results (web_search/fetch_page) carry raw page text into a
    role:"tool" message's content, which apply_chat_template renders as part
    of the prompt. Marker strings like ``<|tool_response>`` or ``<think>``
    are not template *syntax* the renderer parses -- they are literal special
    tokens in the model's vocabulary, so a hostile page could forge a fake
    tool response or open/close a reasoning block simply by including the
    right substring. This removes exactly those marker sequences (the
    ``<|...|>``/``<|...>`` family plus the bare closing forms and
    ``<think>``/``</think>``) and leaves every other character -- including
    standalone ``<``, ``|``, ``>`` -- untouched.
    """
    text = _SPECIAL_MARKER_RE.sub("", text)
    for marker in _BARE_SPECIAL_MARKERS:
        text = text.replace(marker, "")
    return text


def strip_speech_markup(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.M)
    text = re.sub(r"(\*\*|__|\*|`)", "", text)
    text = text.replace("_", " ")  # snake_case identifiers must stay speakable: my_var -> "my var"
    text = _EMOJI.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()
