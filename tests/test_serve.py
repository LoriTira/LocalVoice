import io
import json
import time
from pathlib import Path

from localvoice.config import AudioConfig, Config, KeysConfig, LlmConfig, SttConfig, TtsConfig
from localvoice.engineset import EngineSet
from localvoice.serve import Serve
from tests.fakes import FakeLLM, FakePlayer, FakeSTT, FakeTTS


class FakeCapture:
    def __init__(self):
        self.armed = False

    def start(self):
        pass

    def stop(self):
        pass

    def arm(self):
        self.armed = True

    def disarm(self):
        import numpy as np

        self.armed = False
        rng = np.random.default_rng(0)
        return (0.1 * rng.standard_normal(16000)).astype("float32")

    def discard(self):
        self.armed = False


class GatedFakeCapture(FakeCapture):
    def __init__(self):
        super().__init__()

        class _Buf:
            def __init__(self, outer):
                self.outer = outer
                self.written = []

            def write(self, audio):
                if self.outer.armed:
                    self.written.append(audio)

        self.buffer = _Buf(self)

    def disarm(self):
        import numpy as np

        self.armed = False
        if self.buffer.written:
            return np.concatenate(self.buffer.written).astype("float32")
        return np.zeros(0, np.float32)


class InstantExecutor:
    def submit(self, fn, *a, **k):
        class F:
            def __init__(self):
                fn(*a, **k)

            def done(self):
                return True

            def result(self):
                return None

        return F()

    def shutdown(self, **k):
        pass


def build(tmp_path: Path, commands: list[dict], capture=None) -> list[dict]:
    cfg = Config(SttConfig(), LlmConfig(), TtsConfig(), KeysConfig(), AudioConfig())
    cfg_path = tmp_path / "localvoice.toml"
    cfg_path.write_text("")
    factories = {
        "stt": lambda c: FakeSTT("hello there"),
        "llm": lambda c: FakeLLM(["Hi from the fake. ", "More words."]),
        "tts": lambda c: FakeTTS(),
    }
    es = EngineSet(cfg, factories=factories)
    stdin = io.StringIO("".join(json.dumps(c) + "\n" for c in commands))
    stdout = io.StringIO()
    s = Serve(
        config_path=cfg_path, cfg=cfg, allow_inject=True, stdin=stdin, stdout=stdout,
        engine_set=es, player=FakePlayer(),
        capture=capture if capture is not None else FakeCapture(),
        inference=InstantExecutor(),
    )
    s._exit = lambda code: None
    s.run()
    time.sleep(0.3)  # pipeline thread finishes
    return [json.loads(line) for line in stdout.getvalue().splitlines()]


def events_of(msgs, name):
    return [m for m in msgs if m.get("event") == name]


def test_ready_schema_and_engines_ready(tmp_path):
    msgs = build(tmp_path, [{"cmd": "shutdown"}])
    ready = events_of(msgs, "ready")[0]
    assert ready["version"] == 1
    assert ready["config"]["llm"]["max_tokens"] == 1024
    assert any(d["key"] == "llm.think" for d in ready["schema"])
    assert events_of(msgs, "engines_ready")
    assert [m for m in msgs if m.get("event") == "load_progress"]


def test_ptt_turn_emits_transcript_and_turn_done(tmp_path):
    msgs = build(
        tmp_path,
        [{"cmd": "ptt_down"}, {"cmd": "ptt_up", "held_ms": 500}, {"cmd": "shutdown"}],
    )
    assert events_of(msgs, "user_text")[0]["text"] == "hello there"
    assert "Hi from the fake." in events_of(msgs, "assistant_clause")[0]["text"]
    assert set(events_of(msgs, "turn_done")[0]["latency"]) == {
        "stt", "ttft", "first_clause", "tts_first", "total",
    }
    states = [m["state"] for m in events_of(msgs, "state")]
    assert "listening" in states and "processing" in states


def test_set_config_instant_and_overlay(tmp_path):
    msgs = build(
        tmp_path,
        [{"cmd": "set_config", "changes": {"llm.think": "true"}}, {"cmd": "shutdown"}],
    )
    applied = events_of(msgs, "config_applied")[0]
    assert applied["config"]["llm"]["think"] is True
    assert applied["reloaded"] == []
    import tomllib

    data = tomllib.loads((tmp_path / "localvoice.local.toml").read_text())
    assert data["llm"]["think"] is True


def test_set_config_reload_path(tmp_path):
    msgs = build(
        tmp_path,
        [{"cmd": "set_config", "changes": {"llm.model": "other/model"}}, {"cmd": "shutdown"}],
    )
    applied = events_of(msgs, "config_applied")[0]
    assert applied["reloaded"] == ["llm"]
    assert len([m for m in events_of(msgs, "load_progress") if m["engine"] == "llm"]) >= 4


def test_bad_config_value_yields_error_event(tmp_path):
    msgs = build(
        tmp_path,
        [{"cmd": "set_config", "changes": {"llm.max_tokens": "nope"}}, {"cmd": "shutdown"}],
    )
    assert any("llm.max_tokens" in m["message"] for m in events_of(msgs, "error"))
    assert not events_of(msgs, "config_applied")


def test_unknown_command_yields_error(tmp_path):
    msgs = build(tmp_path, [{"cmd": "dance"}, {"cmd": "shutdown"}])
    assert any("dance" in m["message"] for m in events_of(msgs, "error"))


def test_inject_audio_waits_for_arm_before_writing(tmp_path):
    """Regression guard for the arm/write race (Task 8 review finding):
    _inject must not write into the capture buffer before the orchestrator
    loop thread has actually armed it, or the frames get silently dropped."""
    import wave

    import numpy as np

    wav = tmp_path / "inject.wav"
    rng = np.random.default_rng(0)
    audio = (0.1 * rng.standard_normal(16000)).astype("float32")
    with wave.open(str(wav), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((audio * 32767).astype("int16").tobytes())
    msgs = build(
        tmp_path,
        [{"cmd": "inject_audio", "path": str(wav)}, {"cmd": "shutdown"}],
        capture=GatedFakeCapture(),
    )
    assert events_of(msgs, "user_text")
