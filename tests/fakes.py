import time
from collections.abc import Iterator

import numpy as np


class FakeSTT:
    def __init__(self, text: str = "hello") -> None:
        self.text = text

    def load(self) -> None:
        pass

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        return self.text


class FakeLLM:
    def __init__(self, deltas: list[str]) -> None:
        self.deltas = deltas
        self.last_messages: list[dict] | None = None
        self.last_think: bool | None = None
        self.last_tools: list[dict] | None = None

    def load(self) -> None:
        pass

    def stream(
        self, messages: list[dict], *, think: bool, tools: list[dict] | None = None
    ) -> Iterator[str]:
        self.last_messages = messages
        self.last_think = think
        self.last_tools = tools
        yield from self.deltas


class FakeTTS:
    def __init__(
        self, chunk_len: int = 100, chunks_per_text: int = 2, delay_s: float = 0.0
    ) -> None:
        self.chunk_len = chunk_len
        self.chunks_per_text = chunks_per_text
        self.delay_s = delay_s
        self.texts: list[str] = []

    def load(self) -> None:
        pass

    def synthesize(self, text: str) -> Iterator[np.ndarray]:
        self.texts.append(text)
        for _ in range(self.chunks_per_text):
            if self.delay_s:
                time.sleep(self.delay_s)
            yield np.full(self.chunk_len, 0.1, np.float32)


class FakePlayer:
    def __init__(self) -> None:
        self.log: list[tuple] = []
        self._tags: set[int] = set()

    def start(self) -> None:
        self.log.append(("start",))

    def stop(self) -> None:
        self.log.append(("stop",))

    def submit(self, samples: np.ndarray, tag: int) -> None:
        self.log.append(("submit", tag, len(samples)))
        self._tags.add(tag)

    def submit_raw(self, samples: np.ndarray) -> None:
        self.log.append(("submit_raw", len(samples)))

    def mark_end(self) -> None:
        self.log.append(("mark_end",))

    def flush(self) -> None:
        self.log.append(("flush",))
        self._tags = set()

    def spoken_tags(self) -> set[int]:
        return set(self._tags)
