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

    def _emit_think(self) -> None:
        text = "".join(self._think_parts).strip()
        self._think_parts = []
        if self._on_think is not None and text:
            self._on_think(text)


def strip_speech_markup(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.M)
    text = re.sub(r"(\*\*|__|\*|`)", "", text)
    text = text.replace("_", " ")  # snake_case identifiers must stay speakable: my_var -> "my var"
    text = _EMOJI.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()
