from dataclasses import dataclass

from localvoice.textproc.sanitize import _longest_suffix_prefix

_TC_OPEN = "<|tool_call>"
_TC_CLOSE = "<tool_call|>"
_Q = '<|"|>'


@dataclass
class ToolCall:
    name: str
    args: dict
    raw: str  # full marker-to-marker capture, for error feedback


class ToolFormatError(Exception):
    pass


def parse_gemma_args(body: str) -> dict:
    """'query:<|"|>boston<|"|>,max_results:5' -> {"query": "boston", "max_results": 5}.

    Strings are <|"|>-quoted (may contain commas/colons/braces); bare tokens
    parse as int, float, true/false, else raise ToolFormatError.
    """
    pairs = _split_top_level(body)
    args: dict = {}
    for pair in pairs:
        key, value = _split_key_value(pair)
        args[key] = _parse_value(value)
    return args


def _split_top_level(body: str) -> list[str]:
    """Split ``body`` on top-level commas, treating ``_Q``-quoted spans as atomic."""
    parts: list[str] = []
    buf: list[str] = []
    i = 0
    n = len(body)
    while i < n:
        if body.startswith(_Q, i):
            j = body.find(_Q, i + len(_Q))
            if j < 0:
                raise ToolFormatError(f"unterminated string in args: {body!r}")
            buf.append(body[i : j + len(_Q)])
            i = j + len(_Q)
            continue
        ch = body[i]
        if ch == ",":
            parts.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    parts.append("".join(buf))
    if len(parts) == 1 and parts[0] == "":
        return []
    return parts


def _split_key_value(pair: str) -> tuple[str, str]:
    i = pair.find(":")
    if i < 0:
        raise ToolFormatError(f"missing ':' in arg pair: {pair!r}")
    return pair[:i], pair[i + 1 :]


def _parse_value(value: str):
    if value.startswith(_Q) and value.endswith(_Q) and len(value) >= 2 * len(_Q):
        return value[len(_Q) : -len(_Q)]
    if value == "true":
        return True
    if value == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    raise ToolFormatError(f"unparseable arg value: {value!r}")


def _parse_call_body(buf: str, raw: str) -> tuple[ToolCall, bool]:
    """Best-effort parse of a captured `call:NAME{...}` buffer (no markers).

    Returns the parsed (or best-effort) ``ToolCall`` plus whether the body
    was malformed. Malformed is authoritative here — never re-derived from
    ``args == {}``, since a well-formed no-arg call is a legitimate value.
    """
    body = buf
    if body.startswith("call:"):
        body = body[len("call:") :]
    brace = body.find("{")
    if brace < 0:
        return ToolCall(name=body, args={}, raw=raw), True
    name = body[:brace]
    inner = body[brace + 1 :]
    close = inner.rfind("}")
    if close < 0:
        return ToolCall(name=name, args={}, raw=raw), True
    args_body = inner[:close]
    try:
        args = parse_gemma_args(args_body)
    except ToolFormatError:
        return ToolCall(name=name, args={}, raw=raw), True
    return ToolCall(name=name, args=args, raw=raw), False


class ToolCallParser:
    """Streaming extractor for Gemma's ``<|tool_call>call:NAME{...}<tool_call|>`` markers.

    Mirrors ``ChannelThinkTranslator``'s hold-back discipline: plain text
    passes through untouched (holding back any suffix that could be a prefix
    of ``_TC_OPEN`` across delta boundaries), but once a call marker opens,
    everything — including the closing marker and any text after it — is
    swallowed. The pipeline reads the completed call from ``.call`` instead
    of the text stream.
    """

    def __init__(self) -> None:
        self._buf = ""
        self._capturing = False
        self._done = False
        self.call: ToolCall | None = None
        self.malformed: bool = False

    def feed(self, delta: str) -> str:
        if self._done:
            return ""
        self._buf += delta

        if not self._capturing:
            i = self._buf.find(_TC_OPEN)
            if i < 0:
                hold = _longest_suffix_prefix(self._buf, _TC_OPEN)
                out = self._buf[: len(self._buf) - hold]
                self._buf = self._buf[len(self._buf) - hold :]
                return out
            out = self._buf[:i]
            self._buf = self._buf[i + len(_TC_OPEN) :]
            self._capturing = True
        else:
            out = ""

        j = self._buf.find(_TC_CLOSE)
        if j < 0:
            return out
        raw = _TC_OPEN + self._buf[:j] + _TC_CLOSE
        self.call, self.malformed = _parse_call_body(self._buf[:j], raw)
        self._capturing = False
        self._done = True
        self._buf = ""
        return out

    def finish(self) -> str:
        if self._capturing:
            raw = _TC_OPEN + self._buf
            self.call, _ = _parse_call_body(self._buf, raw)
            self.malformed = True  # unclosed at end of stream is always malformed
            self._capturing = False
            self._done = True
            self._buf = ""
        return ""
