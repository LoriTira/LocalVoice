import pytest

from localvoice.engineset import EngineProxy, EngineSet, plan_apply


def test_plan_apply_routes_per_spec():
    plan = plan_apply({"llm.model": "x", "tts.voice": "af_bella", "audio.rebuffer_ms": 400})
    assert plan == {"reload": ["llm"], "restart_audio": True, "instant": ["tts.voice"]}
    assert plan_apply({"stt.model": "a", "tts.model": "b"})["reload"] == ["stt", "tts"]
    assert plan_apply({"llm.think": True}) == {
        "reload": [], "restart_audio": False, "instant": ["llm.think"],
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
    from localvoice.config import AudioConfig, Config, KeysConfig, LlmConfig, SttConfig, TtsConfig

    cfg = Config(SttConfig(), LlmConfig(), TtsConfig(), KeysConfig(), AudioConfig())
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
