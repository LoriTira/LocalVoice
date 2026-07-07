import types

import pytest

from localvoice.config import LlmConfig
from localvoice.llm import mlx_lm_engine, mlx_vlm_engine
from localvoice.llm.mlx_lm_engine import MlxLmEngine
from localvoice.llm.mlx_vlm_engine import MlxVlmEngine, _TextTowerAdapter


class FakeTower:
    def __init__(self):
        self.layers = ["l1", "l2"]
        self.called_with = None

    def __call__(self, x, **kw):
        self.called_with = (x, kw)
        return types.SimpleNamespace(logits="LOGITS")


def test_adapter_unwraps_language_model_output():
    t = FakeTower()
    assert _TextTowerAdapter(t)("tokens", cache="c") == "LOGITS"
    assert t.called_with == ("tokens", {"cache": "c"})


def test_adapter_passes_raw_returns_through():
    class Raw:
        layers = []

        def __call__(self, x, **kw):
            return "RAW_ARRAY"

    assert _TextTowerAdapter(Raw())("x") == "RAW_ARRAY"


def test_adapter_forwards_attributes():
    assert _TextTowerAdapter(FakeTower()).layers == ["l1", "l2"]


def test_engines_share_one_stream_helper():
    # The whole point of the refactor: both engines call the exact same
    # module-level body, so text behavior cannot drift between them.
    assert mlx_vlm_engine._stream_text is mlx_lm_engine._stream_text


def test_vlm_stream_drives_the_adapter_through_shared_helper(monkeypatch):
    # The hybrid contract: stream() hands the shared algorithm the logits
    # adapter (self._lm) as the model stream_generate drives — not the raw
    # vlm model — and forwards messages/think/tools untouched.
    seen = {}

    def fake_stream_text(engine, model, messages, think, tools):
        seen["args"] = (engine, model, messages, think, tools)
        yield "ok"

    monkeypatch.setattr(mlx_vlm_engine, "_stream_text", fake_stream_text)
    eng = MlxVlmEngine(LlmConfig(model="x"))
    eng._lm = "ADAPTER"  # stand-in for the _TextTowerAdapter that load() builds
    msgs = [{"role": "user", "content": "hi"}]
    out = list(eng.stream(msgs, think=True, tools=None))
    assert out == ["ok"]
    engine, model, messages, think, tools = seen["args"]
    assert engine is eng and model == "ADAPTER"
    assert messages is msgs and think is True and tools is None


def test_mlx_lm_engine_raises_on_image_path_before_any_model_access():
    # Constructed but never load()-ed: _model/_tokenizer are still None. The
    # raise must fire from the image_path check alone, never from touching
    # that state — this engine never loads a vision tower, so honesty about
    # what it can't do is the whole contract.
    engine = MlxLmEngine(LlmConfig(model="x"))
    with pytest.raises(ValueError, match="image input requires the mlx_vlm engine"):
        engine.stream(
            [{"role": "user", "content": "what is this?"}],
            think=False,
            image_path="/tmp/does-not-matter.png",
        )


def test_vlm_stream_routes_image_path_to_dedicated_method(monkeypatch):
    # image_path must skip the shared text helper (and therefore self._lm/
    # self._cache) entirely and land on _stream_image with messages/think/
    # image_path forwarded untouched.
    seen = {}

    def fake_stream_image(self, messages, think, image_path):
        seen["args"] = (messages, think, image_path)
        yield "ok"

    monkeypatch.setattr(MlxVlmEngine, "_stream_image", fake_stream_image)
    eng = MlxVlmEngine(LlmConfig(model="x"))
    msgs = [{"role": "user", "content": "what is this?"}]
    out = list(eng.stream(msgs, think=True, image_path="/tmp/x.png"))
    assert out == ["ok"]
    assert seen["args"] == (msgs, True, "/tmp/x.png")


def test_load_resolves_hf_repo_id_to_one_local_snapshot_before_local_only_calls(
    monkeypatch, tmp_path
):
    """MlxVlmEngine.load() must resolve a model string (repo id OR local path
    -- mlx_lm.utils._download handles both, the same helper mlx_lm.load()
    itself uses) to a local snapshot path BEFORE calling mlx_lm's
    load_config/load_tokenizer, which only ever understood local directories.

    Regression guard: passing a bare HF repo id straight through as
    Path(cfg.model) let vlm_load succeed on its own (it does its own
    resolution/download internally), while load_config/load_tokenizer then
    raised FileNotFoundError against a path that was never a real directory
    -- so a repo-id `[llm].model` under the mlx_vlm engine could download
    ~15GB and still fail to boot.

    The fix must also resolve EXACTLY ONCE and feed that SAME resolved path
    to vlm_load too (not the original repo id) -- a "consistent snapshot":
    load_config/load_tokenizer and vlm_load must all read the identical
    on-disk directory rather than each doing (or skipping) their own,
    possibly-divergent resolution.
    """
    import mlx_lm.utils as mlx_lm_utils
    import mlx_vlm as mlx_vlm_pkg

    resolved = tmp_path / "snapshot"
    resolved.mkdir()
    calls: dict = {"download_n": 0}

    def fake_download(path_or_hf_repo, revision=None):
        calls["download_n"] += 1
        calls["download_arg"] = path_or_hf_repo
        return resolved

    def fake_load_config(model_path):
        calls["load_config_arg"] = model_path
        return {"eos_token_id": [1, 106, 50]}

    def fake_load_tokenizer(model_path, eos_token_ids=None):
        calls["load_tokenizer_arg"] = model_path
        calls["load_tokenizer_eos"] = eos_token_ids
        return types.SimpleNamespace(chat_template=None)

    def fake_vlm_load(path_or_hf_repo):
        calls["vlm_load_arg"] = path_or_hf_repo
        model = types.SimpleNamespace(language_model=types.SimpleNamespace(layers=[]))
        return model, "PROCESSOR"

    monkeypatch.setattr(mlx_lm_utils, "_download", fake_download)
    monkeypatch.setattr(mlx_lm_utils, "load_config", fake_load_config)
    monkeypatch.setattr(mlx_lm_utils, "load_tokenizer", fake_load_tokenizer)
    monkeypatch.setattr(mlx_vlm_pkg, "load", fake_vlm_load)

    repo_id = "mlx-community/gemma-4-26B-it-4bit"
    engine = MlxVlmEngine(LlmConfig(model=repo_id))
    engine.load()

    assert calls["download_n"] == 1, "must resolve once, not re-resolve per loader call"
    assert calls["download_arg"] == repo_id
    assert calls["load_config_arg"] == resolved
    assert calls["load_tokenizer_arg"] == resolved
    # vlm_load must receive the SAME resolved snapshot, not the raw repo id.
    assert calls["vlm_load_arg"] == str(resolved)
    assert calls["load_tokenizer_eos"] == [1, 106, 50]
    assert engine._processor == "PROCESSOR"


def test_supports_images_capability_flags():
    # Task 3 gates tool-offering on this: only an image-capable engine
    # should ever be offered an image-producing tool.
    assert MlxVlmEngine.supports_images is True
    assert MlxLmEngine.supports_images is False


def _bare_vlm_engine(cfg: LlmConfig, *, channel_style: bool) -> MlxVlmEngine:
    """Construct a MlxVlmEngine without load(): set only the private attrs
    _stream_image actually touches. White-box but honest for a wiring test
    (mirrors the FakeTower stand-ins used elsewhere in this file) --
    _model needs a `.config` (fed to apply_chat_template, itself stubbed
    below) and a `.language_model.layers` (make_prompt_cache, invoked by
    the defensive _reset_cache() in _stream_image's `finally`)."""
    eng = MlxVlmEngine(cfg)
    eng._processor = "PROCESSOR"
    eng._model = types.SimpleNamespace(
        config="CONFIG", language_model=types.SimpleNamespace(layers=[])
    )
    eng._channel_style = channel_style
    return eng


def test_stream_image_translates_channel_thought_like_text_turns(monkeypatch):
    # Regression guard for the retired T2 scope cut: a spontaneously opened
    # <|channel>thought block on an image turn must land in <think> tags
    # exactly like a text turn, never flow to callers unmarked -- on a
    # spoken path that is the reasoning-read-aloud bug all over again.
    import mlx_vlm
    import mlx_vlm.prompt_utils

    deltas = [
        types.SimpleNamespace(text="<|channel>thought\nsecret reasoning"),
        types.SimpleNamespace(text="<channel|>answer"),
    ]
    seen = {}

    def fake_apply_chat_template(processor, config, messages, **kwargs):
        return "PROMPT"

    def fake_stream_generate(model, processor, prompt, **kwargs):
        seen["kwargs"] = kwargs
        yield from deltas

    monkeypatch.setattr(mlx_vlm, "stream_generate", fake_stream_generate)
    monkeypatch.setattr(mlx_vlm.prompt_utils, "apply_chat_template", fake_apply_chat_template)

    cfg = LlmConfig(model="x", max_tokens=32, think_tokens=64)
    eng = _bare_vlm_engine(cfg, channel_style=True)

    out = "".join(
        eng._stream_image([{"role": "user", "content": "what is this?"}], True, "/fake/img.png")
    )

    assert out == "<think>secret reasoning</think>answer"
    # think=True budget bump: the other retired scope cut.
    assert seen["kwargs"]["max_tokens"] == cfg.max_tokens + cfg.think_tokens
    assert seen["kwargs"]["image"] == "/fake/img.png"


def test_stream_image_skips_translator_and_bump_when_not_thinking(monkeypatch):
    # Converse of the test above: a non-channel-style model's raw text must
    # pass through untouched (no translator instantiated at all), and
    # think=False must NOT bump the token budget.
    import mlx_vlm
    import mlx_vlm.prompt_utils

    seen = {}

    def fake_apply_chat_template(processor, config, messages, **kwargs):
        return "PROMPT"

    def fake_stream_generate(model, processor, prompt, **kwargs):
        seen["kwargs"] = kwargs
        yield types.SimpleNamespace(text="plain answer, <|channel> looks literal here")

    monkeypatch.setattr(mlx_vlm, "stream_generate", fake_stream_generate)
    monkeypatch.setattr(mlx_vlm.prompt_utils, "apply_chat_template", fake_apply_chat_template)

    cfg = LlmConfig(model="x", max_tokens=32, think_tokens=64)
    eng = _bare_vlm_engine(cfg, channel_style=False)

    out = "".join(
        eng._stream_image([{"role": "user", "content": "what is this?"}], False, "/fake/img.png")
    )

    assert out == "plain answer, <|channel> looks literal here"
    assert seen["kwargs"]["max_tokens"] == cfg.max_tokens
