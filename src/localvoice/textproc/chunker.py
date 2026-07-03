_SENTENCE = ".!?\n"
_SOFT = ",;:—"
_SENTENCE_FLOOR = 15


class ClauseChunker:
    def __init__(self, min_chars: int = 60, max_chars: int = 200) -> None:
        self.min_chars = min_chars
        self.max_chars = max_chars
        self._buf = ""

    def feed(self, delta: str) -> list[str]:
        self._buf += delta
        out: list[str] = []
        while True:
            cut = self._find_cut()
            if cut is None:
                break
            chunk = self._buf[:cut].strip()
            self._buf = self._buf[cut:].lstrip()
            if chunk:
                out.append(chunk)
        return out

    def flush(self) -> str | None:
        chunk, self._buf = self._buf.strip(), ""
        return chunk or None

    def _find_cut(self) -> int | None:
        for i, ch in enumerate(self._buf):
            if i + 1 >= len(self._buf):
                break  # boundary unconfirmed until next delta or flush()
            confirmed = self._buf[i + 1].isspace()
            if ch in _SENTENCE and (confirmed or ch == "\n") and i + 1 >= _SENTENCE_FLOOR:
                return i + 1
            if ch in _SOFT and confirmed and i + 1 >= self.min_chars:
                return i + 1
        if len(self._buf) > self.max_chars:
            space = self._buf.rfind(" ", 0, self.max_chars)
            if space > 0:
                return space + 1
            return self.max_chars
        return None
