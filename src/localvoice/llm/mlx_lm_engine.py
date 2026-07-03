from collections.abc import Iterator

from localvoice.config import LlmConfig
from localvoice.llm.base import Message


def common_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def fit_messages(messages: list[Message], budget: int, count) -> list[Message]:
    """Drop the OLDEST user/assistant exchange (never the system prompt, never
    the final user message) until count(messages) fits the token budget."""
    msgs = list(messages)
    while count(msgs) > budget:
        drop = next(
            (i for i, m in enumerate(msgs[:-1]) if m.get("role") != "system"),
            None,
        )
        if drop is None:
            break  # only system + the live user turn left: nothing droppable
        end = drop + 1
        if end < len(msgs) - 1 and msgs[end].get("role") == "assistant":
            end += 1  # drop the paired assistant reply with its user turn
        del msgs[drop:end]
    return msgs


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
        # templating, so the token stream is always well-formed. Oldest turns
        # are dropped first when the prompt exceeds the context budget.
        messages = fit_messages(
            messages, self._cfg.context_tokens, lambda m: len(self._template(m, think))
        )
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
        if think and self._tokenizer.decode(tokens[-8:]).rstrip().endswith("<think>"):
            # Qwen thinking templates END with an open <think>, so the stream
            # contains reasoning with no opening tag. Prepend one synthetically
            # so the downstream TextFilter suppresses (and captures) it. Guarded
            # on the actual prompt tail so templates that ignore enable_thinking
            # never get their whole answer swallowed.
            yield "<think>"
        budget = self._cfg.max_tokens + (self._cfg.think_tokens if think else 0)
        for response in stream_generate(
            self._model,
            self._tokenizer,
            suffix,
            max_tokens=budget,
            prompt_cache=cache,
        ):
            yield response.text
