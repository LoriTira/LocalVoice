from collections.abc import Iterator

from localvoice.config import LlmConfig
from localvoice.llm.base import Message


def common_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


class MlxLmEngine:
    def __init__(self, cfg: LlmConfig) -> None:
        self._cfg = cfg
        self._model = None
        self._tokenizer = None
        self._cache = None
        # Prompt tokens we have deliberately prefilled into the cache. NEVER
        # includes generated tokens: those enter the cache during generation
        # but are removed next turn by the cache_len - common trim, together
        # with any diverged prompt tail, in one call. That is the core invariant.
        self._prompt_tokens: list[int] = []

    def load(self) -> None:
        from mlx_lm import load

        self._model, self._tokenizer = load(self._cfg.model)
        self._reset_cache()

    def _reset_cache(self) -> None:
        from mlx_lm.models.cache import make_prompt_cache

        self._cache = make_prompt_cache(self._model)
        self._prompt_tokens = []

    def _template(self, messages: list[Message], think: bool) -> list[int]:
        try:
            return self._tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, enable_thinking=think
            )
        except TypeError:  # template without enable_thinking support
            return self._tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    def stream(self, messages: list[Message], *, think: bool) -> Iterator[str]:
        from mlx_lm import stream_generate

        # Canonical full-conversation template every turn — no incremental
        # templating, so the token stream is always well-formed.
        tokens = self._template(messages, think)

        # Read how many tokens the cache actually holds (prompt + generated).
        cache = self._cache
        if cache and hasattr(cache[0], "offset"):
            cache_len = cache[0].offset
            common = common_prefix_len(tokens, self._prompt_tokens)
            to_trim = cache_len - common
            if to_trim > 0:
                from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache

                if can_trim_prompt_cache(cache):
                    trim_prompt_cache(cache, to_trim)
                else:
                    # Hybrid-attention models (e.g. Qwen3.6 family) have
                    # recurrent-state caches that cannot be trimmed; falling
                    # back to a full re-prefill is expected and correct for them.
                    self._reset_cache()
                    cache = self._cache
                    common = 0
        else:  # cache offset unavailable — reset and re-prefill everything
            self._reset_cache()
            cache = self._cache
            common = 0

        suffix = tokens[common:]
        if not suffix:  # cannot happen when messages end with a fresh user turn
            self._reset_cache()
            cache = self._cache
            suffix = tokens
            common = 0

        self._prompt_tokens = list(tokens)
        for response in stream_generate(
            self._model,
            self._tokenizer,
            suffix,
            max_tokens=self._cfg.max_tokens,
            prompt_cache=cache,
        ):
            yield response.text
