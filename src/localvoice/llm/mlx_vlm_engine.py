from collections.abc import Iterator

from localvoice.config import LlmConfig
from localvoice.llm.base import Message
from localvoice.llm.mlx_lm_engine import _stream_text, is_channel_style, supports_tools


class _TextTowerAdapter:
    """Adapt mlx-vlm's LanguageModel to what mlx_lm.stream_generate expects.

    mlx-vlm's tower returns a LanguageModelOutput dataclass; mlx-lm's
    generation loop consumes raw logits arrays. Everything else it touches
    (layers for cache construction, config/args attributes) is forwarded.
    Verified viable in the spec §11 spike: make_prompt_cache(language_model)
    already works (30 layers); only the __call__ return type mismatches.
    """

    def __init__(self, tower) -> None:
        self._tower = tower

    def __call__(self, *args, **kwargs):
        out = self._tower(*args, **kwargs)
        return out.logits if hasattr(out, "logits") else out

    def __getattr__(self, name):
        return getattr(self._tower, name)


class MlxVlmEngine:
    """Vision-capable Gemma 4 as an LLM engine via hybrid generation.

    Load once through mlx-vlm (weights include the vision tower, single
    residency) but drive TEXT turns through mlx_lm.stream_generate against
    model.language_model — so the shipped prefix-reuse / think-translation /
    tools code (the shared `_stream_text`) carries over verbatim. The one
    blocker mlx-vlm's tower presents — a LanguageModelOutput return instead of
    raw logits — is bridged by _TextTowerAdapter. IMAGE turns (`stream(...,
    image_path=...)`) go a different way: `_stream_image` drives mlx-vlm's own
    generation directly against the full model (vision tower + language
    tower), since the text-tower adapter has no way to see pixel_values. T3
    supplies the caller (a screenshot tool feeding a captured PNG back in);
    this task wires the minimal entry point it calls.
    """

    def __init__(self, cfg: LlmConfig) -> None:
        self._cfg = cfg
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._lm = None
        self._cache = None
        # Same invariant as MlxLmEngine: prompt tokens NEVER include generated
        # tokens. Those enter the cache during generation but are removed next
        # turn by the cache_len - common trim, together with any diverged
        # prompt tail, in one call.
        self._prompt_tokens: list[int] = []

    def load(self) -> None:
        from pathlib import Path

        from mlx_lm.utils import load_config, load_tokenizer
        from mlx_vlm import load as vlm_load

        self._model, self._processor = vlm_load(self._cfg.model)
        # Text turns run through mlx-lm's generation loop, so the tokenizer must
        # be built the way mlx_lm.load builds it — NOT taken from the mlx-vlm
        # processor. The processor's tokenizer exposes only <eos> (id 1), so
        # generation would never stop at Gemma's <end_of_turn> and would burn the
        # full token budget every turn; it also returns a BatchEncoding from
        # apply_chat_template rather than a token-id list. load_tokenizer with the
        # model config's full eos set ([1, 50, 106]) fixes both — true text-turn
        # parity with MlxLmEngine. self._processor is kept for T3's image path.
        model_path = Path(self._cfg.model)
        eos_token_ids = load_config(model_path).get("eos_token_id")
        self._tokenizer = load_tokenizer(model_path, eos_token_ids=eos_token_ids)
        tmpl = getattr(self._tokenizer, "chat_template", None)
        self._channel_style = is_channel_style(tmpl)
        self.supports_tools = supports_tools(tmpl)
        self._lm = _TextTowerAdapter(self._model.language_model)
        self._reset_cache()

    def _reset_cache(self) -> None:
        from mlx_lm.models.cache import make_prompt_cache

        # Cache is built over the raw tower (make_prompt_cache reads its
        # .layers, 30 for Gemma 4); generation drives the adapter, which wraps
        # the same tower object — so the cache stays compatible.
        self._cache = make_prompt_cache(self._model.language_model)
        self._prompt_tokens = []

    def stream(
        self,
        messages: list[Message],
        *,
        think: bool,
        tools: list[dict] | None = None,
        image_path: str | None = None,
    ) -> Iterator[str]:
        if image_path is not None:
            return self._stream_image(messages, think, image_path)
        return _stream_text(self, self._lm, messages, think, tools)

    def _stream_image(
        self, messages: list[Message], think: bool, image_path: str
    ) -> Iterator[str]:
        """Image turns bypass the text-tower adapter entirely.

        `self._processor` (kept at load() time for exactly this) renders the
        image placeholder through mlx-vlm's OWN chat-template convention —
        `mlx_vlm.prompt_utils.apply_chat_template` — not the mlx-lm tokenizer
        `_stream_text` uses for TEXT turns. `mlx_vlm.stream_generate` then
        drives the FULL model (`self._model`: vision tower + language tower),
        since `self._lm` only wraps the language tower and cannot consume
        pixel_values. tools are not offered to image turns (T3's screenshot
        tool result is described in a standalone turn, not a tool-calling
        round); `think` is passed to the template best-effort — Gemma 4's
        template does accept `enable_thinking`, but if a future model swap's
        template doesn't, we drop it rather than crash (documented limitation:
        such a model's image turns simply never think).

        No `prompt_cache` is passed to mlx-vlm, so it builds and discards its
        own scratch KV cache for this call alone (see mlx_vlm.generate.
        dispatch.stream_generate: absent a `prompt_cache` kwarg, it calls
        mlx_vlm's own `cache.make_prompt_cache` and never hands the result
        back to us) — `self._cache`/`self._prompt_tokens`, the persisted
        TEXT-turn state `_stream_text` trims against, are never read or
        written by an image turn; prefix reuse is simply skipped (full
        prefill), which is fine since image turns are rare. Reading gemma4's
        LanguageModel/Gemma4TextModel (mlx_vlm/models/gemma4/language.py)
        confirms its __call__ carries no cross-call mutable state of its own
        either (RoPE offset flows through as a call argument, derived from
        the cache passed in, never stored on self) — so even the shared
        language-tower object underneath both self._lm and self._model is not
        left disturbed by an image turn. The reset below is therefore
        defensive belt-and-suspenders, not a correctness fix: it is cheap
        (fresh empty KVCache objects, no compute) and keeps "the next TEXT
        turn always starts from a known-clean cache after an image turn" a
        simple, local invariant instead of one resting on the analysis above
        remaining true across future mlx-vlm versions.
        """
        from mlx_vlm import stream_generate
        from mlx_vlm.prompt_utils import apply_chat_template

        try:
            prompt = apply_chat_template(
                self._processor,
                self._model.config,
                messages,
                add_generation_prompt=True,
                num_images=1,
                enable_thinking=think,
            )
        except TypeError:  # template without enable_thinking support
            prompt = apply_chat_template(
                self._processor,
                self._model.config,
                messages,
                add_generation_prompt=True,
                num_images=1,
            )

        try:
            for result in stream_generate(
                self._model,
                self._processor,
                prompt,
                image=image_path,
                # T2 scope cut, T3 must revisit: no `+ think_tokens` budget
                # bump here (unlike _stream_text) — a thinking image turn can
                # exhaust its budget mid-reasoning and yield an empty answer.
                max_tokens=self._cfg.max_tokens,
            ):
                # T2 scope cut, T3 MUST fix before wiring this into the
                # pipeline: the raw text skips ChannelThinkTranslator, so a
                # spontaneously opened <|channel>thought block would flow to
                # callers unmarked — on a spoken path that is the reasoning-
                # read-aloud bug all over again, on image turns.
                yield result.text
        finally:
            self._reset_cache()
