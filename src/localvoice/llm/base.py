from collections.abc import Iterator
from typing import Protocol

Message = dict


class LLMEngine(Protocol):
    def load(self) -> None: ...

    def stream(
        self,
        messages: list[Message],
        *,
        think: bool,
        tools: list[dict] | None = None,
        image_path: str | None = None,
    ) -> Iterator[str]: ...
