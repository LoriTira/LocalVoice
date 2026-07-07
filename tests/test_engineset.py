import pytest

from localvoice.engineset import EngineSet, plan_apply


def test_plan_apply_routes_per_spec():
    plan = plan_apply({"llm.model": "x", "tts.voice": "af_bella", "audio.rebuffer_ms": 400})
    assert plan == {"reload": ["llm"], "restart_audio": True, "instant": ["tts.voice"]}
    assert plan_apply({"stt.model": "a", "tts.model": "b"})["reload"] == ["stt", "tts"]
    assert plan_apply({"llm.think": True}) == {
        "reload": [], "restart_audio": False, "instant": ["llm.think"],
    }


def test_plan_apply_routes_llm_engine_to_reload():
    # llm.engine is a `*.engine` key per the hot-apply table (docs/gui.md) —
    # switching mlx_lm <-> mlx_vlm must reload the llm engine like any other
    # engine-bound key.
    assert plan_apply({"llm.engine": "mlx_vlm"}) == {
        "reload": ["llm"], "restart_audio": False, "instant": [],
    }


class Fake:
    def __init__(self, cfg):
        self.cfg = cfg
        self.loaded = False

    def load(self):
        self.loaded = True


class Boom(Fake):
    def load(self):
        raise RuntimeError("no")


def make_set(llm_factory=Fake):
    from localvoice.config import (
        AudioConfig,
        Config,
        KeysConfig,
        LlmConfig,
        SttConfig,
        ToolsConfig,
        TtsConfig,
    )

    cfg = Config(SttConfig(), LlmConfig(), TtsConfig(), KeysConfig(), AudioConfig(), ToolsConfig())
    factories = {"stt": Fake, "llm": llm_factory, "tts": Fake}
    return cfg, EngineSet(cfg, factories=factories)


def test_load_all_reports_progress_and_loads():
    cfg, es = make_set()
    events: list = []
    es.load_all(lambda n, p, s: events.append((n, p)))
    assert es.llm.loaded and es.stt.loaded and es.tts.loaded
    assert ("llm", "start") in events and ("llm", "done") in events


def test_reload_swaps_proxy_identity_preserved():
    cfg, es = make_set()
    es.load_all(lambda *a: None)
    proxy = es.llm
    old = proxy._target
    es.reload("llm", cfg, lambda *a: None)
    assert es.llm is proxy and proxy._target is not old and proxy.loaded


def test_reload_failure_keeps_old_engine():
    cfg, es = make_set()
    es.load_all(lambda *a: None)
    old = es.llm._target
    es._factories["llm"] = Boom
    with pytest.raises(RuntimeError):
        es.reload("llm", cfg, lambda *a: None)
    assert es.llm._target is old


def test_load_all_tracks_loaded_names():
    cfg, es = make_set()
    assert es.loaded == set()
    es.load_all(lambda *a: None)
    assert es.loaded == {"stt", "llm", "tts"}


def test_reload_adds_to_loaded_on_success():
    cfg, es = make_set()
    es.load_all(lambda *a: None)
    es.loaded.discard("llm")
    es.reload("llm", cfg, lambda *a: None)
    assert "llm" in es.loaded


def test_load_all_without_on_error_still_raises_preserving_old_semantics():
    cfg, es = make_set(llm_factory=Boom)
    with pytest.raises(RuntimeError):
        es.load_all(lambda *a: None)


def test_load_all_with_on_error_collects_failure_and_continues_other_engines():
    cfg, es = make_set(llm_factory=Boom)
    errors: list = []
    es.load_all(lambda *a: None, on_error=lambda name, exc: errors.append((name, exc)))
    assert len(errors) == 1
    assert errors[0][0] == "llm"
    assert isinstance(errors[0][1], RuntimeError)
    assert es.stt.loaded and es.tts.loaded
    assert not es.llm.loaded
    assert es.loaded == {"stt", "tts"}


def _cfg_with_llm_engine(engine: str):
    """A real Config (no Fake factories) so EngineSet exercises the actual
    default llm factory — the thing under test here, not the load/reload
    mechanics covered by make_set()'s Fakes above."""
    from localvoice.config import (
        AudioConfig,
        Config,
        KeysConfig,
        LlmConfig,
        SttConfig,
        ToolsConfig,
        TtsConfig,
    )

    return Config(
        SttConfig(), LlmConfig(engine=engine), TtsConfig(), KeysConfig(), AudioConfig(),
        ToolsConfig(),
    )


def test_engineset_builds_mlx_lm_engine_by_default():
    from localvoice.llm.mlx_lm_engine import MlxLmEngine

    es = EngineSet(_cfg_with_llm_engine("mlx_lm"))
    assert isinstance(es.llm._target, MlxLmEngine)


def test_engineset_builds_mlx_vlm_engine_when_configured():
    from localvoice.llm.mlx_vlm_engine import MlxVlmEngine

    es = EngineSet(_cfg_with_llm_engine("mlx_vlm"))
    assert isinstance(es.llm._target, MlxVlmEngine)


def test_engineset_rejects_unknown_llm_engine():
    from localvoice.config import ConfigError

    with pytest.raises(ConfigError, match="bogus"):
        EngineSet(_cfg_with_llm_engine("bogus"))
