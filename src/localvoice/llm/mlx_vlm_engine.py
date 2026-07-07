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
    raw logits — is bridged by _TextTowerAdapter. Image turns (T3) will use
    mlx-vlm generation directly; this engine keeps the text path at parity.
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
        self, messages: list[Message], *, think: bool, tools: list[dict] | None = None
    ) -> Iterator[str]:
        return _stream_text(self, self._lm, messages, think, tools)
