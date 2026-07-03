from collections.abc import Iterator

from localvoice.config import LlmConfig
from localvoice.llm.base import Message


def plan_prompt(committed: list[Message], messages: list[Message]) -> str:
    if (
        committed
        and len(messages) == len(committed) + 1
        and messages[:-1] == committed
        and messages[-1].get("role") == "user"
    ):
        return "incremental"
    return "full"


class MlxLmEngine:
    def __init__(self, cfg: LlmConfig) -> None:
        self._cfg = cfg
        self._model = None
        self._tokenizer = None
        self._cache = None
        self._committed: list[Message] = []

    def load(self) -> None:
        from mlx_lm import load

        self._model, self._tokenizer = load(self._cfg.model)
        self._reset_cache()

    def _reset_cache(self) -> None:
        from mlx_lm.models.cache import make_prompt_cache

        self._cache = make_prompt_cache(self._model)
        self._committed = []

    def _template(self, messages: list[Message], think: bool) -> list[int]:
        try:
            return self._tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, enable_thinking=think
            )
        except TypeError:  # template without enable_thinking support
            return self._tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    def stream(self, messages: list[Message], *, think: bool) -> Iterator[str]:
        from mlx_lm import stream_generate

        if plan_prompt(self._committed, messages) == "incremental":
            prompt = self._template([messages[-1]], think)
        else:
            self._reset_cache()
            prompt = self._template(messages, think)
        parts: list[str] = []
        for response in stream_generate(
            self._model,
            self._tokenizer,
            prompt,
            max_tokens=self._cfg.max_tokens,
            prompt_cache=self._cache,
        ):
            parts.append(response.text)
            yield response.text
        self._committed = [*messages, {"role": "assistant", "content": "".join(parts)}]
