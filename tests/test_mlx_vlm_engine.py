import types

from localvoice.config import LlmConfig
from localvoice.llm import mlx_lm_engine, mlx_vlm_engine
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
