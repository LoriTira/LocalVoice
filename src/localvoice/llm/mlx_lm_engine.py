from collections.abc import Iterator

from localvoice.config import LlmConfig
from localvoice.llm.base import Message
from localvoice.textproc.sanitize import ChannelThinkTranslator


def is_channel_style(chat_template: str | None) -> bool:
    """Gemma-4-family templates express reasoning as <|channel>thought blocks
    rather than Qwen's <think> tags; the raw stream then needs translating to
    the canonical tags the rest of the pipeline filters on."""
    return bool(chat_template) and "<|channel>" in chat_template


def supports_tools(chat_template: str | None) -> bool:
    """Whether the tokenizer's chat template renders a tool-call block at all
    (i.e. accepts the `tools=` kwarg to apply_chat_template). Used to decide
    whether to offer tool schemas to a given model."""
    return bool(chat_template) and "<|tool>" in chat_template


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


def _apply_template(
    tokenizer, messages: list[Message], think: bool, tools: list[dict] | None = None
) -> list[int]:
    kwargs = {"tools": tools} if tools else {}
    try:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=think, **kwargs
        )
    except TypeError:  # template without enable_thinking support
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, **kwargs
        )


def _stream_text(
    engine, model, messages: list[Message], think: bool, tools: list[dict] | None
) -> Iterator[str]:
    """Shared text-generation algorithm for MlxLmEngine and MlxVlmEngine.

    `engine` carries the per-engine state this operates on: the mutable cache
    (`_cache`, `_prompt_tokens`), config (`_cfg`), tokenizer (`_tokenizer`),
    channel style (`_channel_style`) and a `_reset_cache()` that rebuilds the
    cache over the right module — the whole model for mlx-lm, the language
    tower for mlx-vlm. `model` is the callable `stream_generate` drives: the
    raw model for mlx-lm, the logits adapter for mlx-vlm. Keeping this one
    body means the two engines cannot drift.
    """
    from mlx_lm import stream_generate

    # Canonical full-conversation template every turn — no incremental
    # templating, so the token stream is always well-formed. Oldest turns
    # are dropped first when the prompt exceeds the context budget.
    messages = fit_messages(
        messages,
        engine._cfg.context_tokens,
        lambda m: len(_apply_template(engine._tokenizer, m, think, tools)),
    )
    tokens = _apply_template(engine._tokenizer, messages, think, tools)

    # Read how many tokens the cache actually holds (prompt + generated).
    cache = engine._cache
    if cache and hasattr(cache[0], "offset"):
        cache_len = cache[0].offset
        common = common_prefix_len(tokens, engine._prompt_tokens)
        to_trim = cache_len - common
        if to_trim > 0:
            from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache

            if can_trim_prompt_cache(cache):
                trim_prompt_cache(cache, to_trim)
            else:
                # Hybrid-attention models (e.g. Qwen3.6 family) have
                # recurrent-state caches that cannot be trimmed; falling
                # back to a full re-prefill is expected and correct for them.
                engine._reset_cache()
                cache = engine._cache
                common = 0
    else:  # cache offset unavailable — reset and re-prefill everything
        engine._reset_cache()
        cache = engine._cache
        common = 0

    suffix = tokens[common:]
    if not suffix:  # cannot happen when messages end with a fresh user turn
        engine._reset_cache()
        cache = engine._cache
        suffix = tokens
        common = 0

    engine._prompt_tokens = list(tokens)
    if think and engine._tokenizer.decode(tokens[-8:]).rstrip().endswith("<think>"):
        # Qwen thinking templates END with an open <think>, so the stream
        # contains reasoning with no opening tag. Prepend one synthetically
        # so the downstream TextFilter suppresses (and captures) it. Guarded
        # on the actual prompt tail so templates that ignore enable_thinking
        # never get their whole answer swallowed.
        yield "<think>"
    budget = engine._cfg.max_tokens + (engine._cfg.think_tokens if think else 0)
    # Channel-style models (Gemma 4) stream reasoning as
    # <|channel>thought ... <channel|>; translate to the canonical
    # <think> tags downstream filters on. Always active for such models
    # (not just when think=True): a spontaneously opened thought channel
    # must never reach the speakers either.
    translator = ChannelThinkTranslator() if engine._channel_style else None
    for response in stream_generate(
        model,
        engine._tokenizer,
        suffix,
        max_tokens=budget,
        prompt_cache=cache,
    ):
        yield translator.feed(response.text) if translator else response.text
    if translator is not None:
        tail = translator.finish()
        if tail:
            yield tail


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
        self._channel_style = is_channel_style(getattr(self._tokenizer, "chat_template", None))
        self.supports_tools = supports_tools(getattr(self._tokenizer, "chat_template", None))
        self._reset_cache()

    def _reset_cache(self) -> None:
        from mlx_lm.models.cache import make_prompt_cache

        self._cache = make_prompt_cache(self._model)
        self._prompt_tokens = []

    def stream(
        self, messages: list[Message], *, think: bool, tools: list[dict] | None = None
    ) -> Iterator[str]:
        return _stream_text(self, self._model, messages, think, tools)
