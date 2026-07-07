import io
import json
import threading
import time
from pathlib import Path

from localvoice.config import (
    AudioConfig,
    Config,
    KeysConfig,
    LlmConfig,
    SttConfig,
    ToolsConfig,
    TtsConfig,
)
from localvoice.engineset import EngineSet
from localvoice.serve import Serve
from tests.fakes import EchoTool, FakeLLM, FakePlayer, FakeSTT, FakeTTS, ScriptedToolLLM


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


class ManualExecutor:
    """Test double for the inference ThreadPoolExecutor that defers work until
    explicitly told to run it, so a test can observe state BEFORE a submitted
    job (e.g. the boot _load_engines job) has executed."""

    def __init__(self):
        self._pending: list[tuple] = []

    def submit(self, fn, *a, **k):
        self._pending.append((fn, a, k))

        class F:
            def done(self):
                return False

            def result(self):
                return None

        return F()

    def run_all(self) -> None:
        pending, self._pending = self._pending, []
        for fn, a, k in pending:
            fn(*a, **k)

    def shutdown(self, **k):
        pass


def build(
    tmp_path: Path, commands: list[dict], capture=None, raw_lines: list[str] | None = None
) -> list[dict]:
    cfg = Config(SttConfig(), LlmConfig(), TtsConfig(), KeysConfig(), AudioConfig(), ToolsConfig())
    cfg_path = tmp_path / "localvoice.toml"
    cfg_path.write_text("")
    factories = {
        "stt": lambda c: FakeSTT("hello there"),
        "llm": lambda c: FakeLLM(["Hi from the fake. ", "More words."]),
        "tts": lambda c: FakeTTS(),
    }
    es = EngineSet(cfg, factories=factories)
    payload = "".join(json.dumps(c) + "\n" for c in commands)
    if raw_lines:
        payload += "".join(line + "\n" for line in raw_lines)
    stdin = io.StringIO(payload)
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


_CALL = '<|tool_call>call:web_search{query:<|"|>rain<|"|>}<tool_call|>'


def test_tool_events_emitted(tmp_path):
    """A turn scripted for one tool round must surface tool_call then
    tool_result on the protocol stream, in order, before turn_done — the two
    new events Task 7 adds, emitted through the same locked emitter path as
    every other pipeline callback (see on_thinking's `reasoning` wiring)."""
    cfg = Config(SttConfig(), LlmConfig(), TtsConfig(), KeysConfig(), AudioConfig(), ToolsConfig())
    cfg_path = tmp_path / "localvoice.toml"
    cfg_path.write_text("")
    tool = EchoTool()
    llm = ScriptedToolLLM([["Let me check. ", _CALL], ["It will rain at noon."]])
    factories = {
        "stt": lambda c: FakeSTT("what's the weather"),
        "llm": lambda c: llm,
        "tts": lambda c: FakeTTS(),
    }
    es = EngineSet(cfg, factories=factories)
    commands = [{"cmd": "ptt_down"}, {"cmd": "ptt_up", "held_ms": 500}, {"cmd": "shutdown"}]
    stdin = io.StringIO("".join(json.dumps(c) + "\n" for c in commands))
    stdout = io.StringIO()
    s = Serve(
        config_path=cfg_path, cfg=cfg, allow_inject=True, stdin=stdin, stdout=stdout,
        engine_set=es, player=FakePlayer(), capture=FakeCapture(),
        inference=InstantExecutor(),
        # Swaps the real registry_for (which would build the actual
        # WebSearchTool) for a fake registry offering just EchoTool, the
        # same substitution EngineSet(cfg, factories=...) does for engines.
        tools_factory=lambda tools_cfg: [tool],
    )
    s._exit = lambda code: None
    s.run()
    time.sleep(0.3)
    msgs = [json.loads(line) for line in stdout.getvalue().splitlines()]

    call_ev = events_of(msgs, "tool_call")
    result_ev = events_of(msgs, "tool_result")
    assert call_ev == [
        {"event": "tool_call", "name": "web_search", "summary": "calling web_search"}
    ]
    assert result_ev == [
        {"event": "tool_result", "name": "web_search", "ok": True, "summary": "found 1 result"}
    ]
    call_idx = msgs.index(call_ev[0])
    result_idx = msgs.index(result_ev[0])
    turn_done_idx = next(i for i, m in enumerate(msgs) if m.get("event") == "turn_done")
    assert call_idx < result_idx < turn_done_idx
    assert llm.calls[0]["tools"] and isinstance(llm.calls[0]["tools"], list)


def test_pipeline_error_surfaces_as_protocol_error_event(tmp_path):
    """A pipeline-level crash mid-turn must reach the GUI as an {"event":
    "error", ...} protocol message. Regression guard for the finding that
    serve constructed the Orchestrator with status=lambda s: None, so
    A.REPORT_ERROR's status call (the only surfacing of PIPELINE_ERROR) was
    swallowed — a real user saw the assistant silently go idle with no banner.

    The TTS raises during synthesis (same mechanism as
    test_pipeline.py::test_error_emits_pipeline_error). Uses a real inference
    thread (like production) so the pipeline runs off the orchestrator loop
    thread, and only sends shutdown AFTER the error event is observed — a
    shutdown that races the turn would cancel it and suppress the error emit
    (pipeline.py guards emit on `not cancel.is_set()`)."""
    cfg = Config(SttConfig(), LlmConfig(), TtsConfig(), KeysConfig(), AudioConfig(), ToolsConfig())
    cfg_path = tmp_path / "localvoice.toml"
    cfg_path.write_text("")

    class BoomTTS(FakeTTS):
        def synthesize(self, text):
            raise RuntimeError("kaboom")
            yield  # pragma: no cover

    factories = {
        "stt": lambda c: FakeSTT("hello there"),
        "llm": lambda c: FakeLLM(["Hi from the fake. ", "More words."]),
        "tts": lambda c: BoomTTS(),
    }
    es = EngineSet(cfg, factories=factories)

    from concurrent.futures import ThreadPoolExecutor

    emitted: list[dict] = []
    saw_error = threading.Event()

    class _CollectingStdout:
        def write(self, s: str) -> None:
            for line in s.splitlines():
                if line.strip():
                    obj = json.loads(line)
                    emitted.append(obj)
                    if obj.get("event") == "error":
                        saw_error.set()

        def flush(self) -> None:
            pass

    class _ScriptedStdin:
        """Yields ptt_down/ptt_up, then blocks until the error is observed
        before yielding shutdown, so shutdown never cancels the in-flight turn."""

        def __iter__(self):
            yield json.dumps({"cmd": "ptt_down"})
            yield json.dumps({"cmd": "ptt_up", "held_ms": 500})
            saw_error.wait(timeout=5)
            yield json.dumps({"cmd": "shutdown"})

    inference = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")
    s = Serve(
        config_path=cfg_path, cfg=cfg, allow_inject=True,
        stdin=_ScriptedStdin(), stdout=_CollectingStdout(),
        engine_set=es, player=FakePlayer(), capture=FakeCapture(),
        inference=inference,
    )
    s._exit = lambda code: None
    s.run()
    inference.shutdown(wait=True)
    assert saw_error.is_set(), f"no error event emitted: {[m.get('event') for m in emitted]}"
    errors = events_of(emitted, "error")
    assert any("kaboom" in m["message"] for m in errors), emitted


def test_tools_not_offered_without_template_support(tmp_path):
    """A model whose chat template can't render tool calls (supports_tools is
    False after load()) must never be offered tools, even with
    cfg.tools.enabled = True — Global Constraints: tools are config-gated
    AND template-gated, template support wins when the model can't accept
    them."""
    cfg = Config(SttConfig(), LlmConfig(), TtsConfig(), KeysConfig(), AudioConfig(), ToolsConfig())
    assert cfg.tools.enabled is True  # precondition: config alone would allow tools
    cfg_path = tmp_path / "localvoice.toml"
    cfg_path.write_text("")
    llm = FakeLLM(["No tools here."], supports_tools=False)
    factories = {
        "stt": lambda c: FakeSTT("hello there"),
        "llm": lambda c: llm,
        "tts": lambda c: FakeTTS(),
    }
    es = EngineSet(cfg, factories=factories)
    commands = [{"cmd": "ptt_down"}, {"cmd": "ptt_up", "held_ms": 500}, {"cmd": "shutdown"}]
    stdin = io.StringIO("".join(json.dumps(c) + "\n" for c in commands))
    stdout = io.StringIO()
    s = Serve(
        config_path=cfg_path, cfg=cfg, allow_inject=True, stdin=stdin, stdout=stdout,
        engine_set=es, player=FakePlayer(), capture=FakeCapture(),
        inference=InstantExecutor(),
    )
    s._exit = lambda code: None
    s.run()
    time.sleep(0.3)
    assert llm.last_tools is None
    assert not events_of([json.loads(line) for line in stdout.getvalue().splitlines()], "tool_call")


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


def test_reset_config_clears_overlay_but_keeps_kept_model_key(tmp_path):
    """reset_config removes every overlay key except those in `keep`: after a
    set_config that overrode both a kept model key and a non-kept instant key,
    resetting with keep=[llm.model] must leave the overlay holding only
    llm.model, and config_applied must show the base default for the cleared
    key (llm.think back to False) while keeping the model override."""
    import tomllib

    msgs = build(
        tmp_path,
        [
            {"cmd": "set_config", "changes": {"llm.model": "kept/model", "llm.think": "true"}},
            {"cmd": "reset_config", "keep": ["llm.model"]},
            {"cmd": "shutdown"},
        ],
    )
    applied = events_of(msgs, "config_applied")
    reset_applied = applied[-1]
    # Cleared instant key reverts to its dataclass default; kept model survives.
    assert reset_applied["config"]["llm"]["think"] is False
    assert reset_applied["config"]["llm"]["model"] == "kept/model"
    data = tomllib.loads((tmp_path / "localvoice.local.toml").read_text())
    assert data == {"llm": {"model": "kept/model"}}


def test_reset_config_reloads_only_engine_bound_keys_that_changed(tmp_path):
    """The reset diff routes through the same plan_apply path set_config uses:
    only engine-bound keys whose merged value actually changes trigger a
    reload. Here llm.model was overridden then reset (not kept), so it reverts
    to the base default and the llm engine must reload; the cleared instant
    key (llm.think) must NOT cause any reload."""
    msgs = build(
        tmp_path,
        [
            {"cmd": "set_config", "changes": {"llm.model": "other/model", "llm.think": "true"}},
            {"cmd": "reset_config", "keep": []},
            {"cmd": "shutdown"},
        ],
    )
    reset_applied = events_of(msgs, "config_applied")[-1]
    assert reset_applied["reloaded"] == ["llm"]
    # Back to the base default now that the override was cleared.
    assert reset_applied["config"]["llm"]["model"] == LlmConfig().model
    assert reset_applied["config"]["llm"]["think"] is False


def test_reset_config_with_empty_overlay_applies_with_no_reloads(tmp_path):
    """Resetting when the overlay is already empty (nothing was ever
    overridden) is a valid no-op-ish call: it still replies config_applied
    with the full base config and an empty reloaded list, never an error."""
    msgs = build(
        tmp_path,
        [{"cmd": "reset_config", "keep": ["llm.model"]}, {"cmd": "shutdown"}],
    )
    applied = events_of(msgs, "config_applied")
    assert len(applied) == 1
    assert applied[0]["reloaded"] == []
    # Full merged config still present; nothing changed from the defaults.
    assert applied[0]["config"]["llm"]["model"] == LlmConfig().model
    assert not events_of(msgs, "error")


def test_reset_config_defaults_keep_to_empty_when_omitted(tmp_path):
    """`keep` is optional (default empty): a reset_config with no keep field
    clears the entire overlay, same as keep=[]."""
    import tomllib

    msgs = build(
        tmp_path,
        [
            {"cmd": "set_config", "changes": {"llm.think": "true"}},
            {"cmd": "reset_config"},
            {"cmd": "shutdown"},
        ],
    )
    reset_applied = events_of(msgs, "config_applied")[-1]
    assert reset_applied["config"]["llm"]["think"] is False
    assert tomllib.loads((tmp_path / "localvoice.local.toml").read_text()) == {}


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


def test_ptt_up_none_held_ms_does_not_kill_the_loop(tmp_path):
    """Regression guard for the phase-A review dispatch-boundary finding:
    ptt_up with held_ms=None (a GUI client sending JSON null) must not raise
    inside int(msg.get("held_ms", 500)) — int(None) is a TypeError, and
    without a dispatch exception boundary that would kill the whole stdin
    loop, silently dropping every subsequent command. Either an error event
    or a handled turn is acceptable for this one message; what's asserted is
    that the loop survived to process a later list_models command."""
    msgs = build(
        tmp_path,
        [
            {"cmd": "ptt_down"},
            {"cmd": "ptt_up", "held_ms": None},
            {"cmd": "list_models"},
            {"cmd": "shutdown"},
        ],
    )
    assert events_of(msgs, "models"), "serve must survive ptt_up held_ms=None to reach list_models"


def test_set_config_system_prompt_hot_applies_to_next_turn(tmp_path):
    """llm.system_prompt is in the "instant" hot-apply group (docs/gui.md):
    a set_config call must update the live Transcript's system message so
    the very next turn's first LLM call already carries the new prompt —
    not just the Serve-level Config object, which the pipeline never reads
    directly."""
    cfg = Config(SttConfig(), LlmConfig(), TtsConfig(), KeysConfig(), AudioConfig(), ToolsConfig())
    cfg_path = tmp_path / "localvoice.toml"
    cfg_path.write_text("")
    fake_llm = FakeLLM(["Hi from the fake. ", "More words."])
    factories = {
        "stt": lambda c: FakeSTT("hello there"),
        "llm": lambda c: fake_llm,
        "tts": lambda c: FakeTTS(),
    }
    es = EngineSet(cfg, factories=factories)
    commands = [
        {"cmd": "set_config", "changes": {"llm.system_prompt": "You are a pirate."}},
        {"cmd": "ptt_down"},
        {"cmd": "ptt_up", "held_ms": 500},
        {"cmd": "shutdown"},
    ]
    stdin = io.StringIO("".join(json.dumps(c) + "\n" for c in commands))
    stdout = io.StringIO()
    s = Serve(
        config_path=cfg_path, cfg=cfg, allow_inject=True, stdin=stdin, stdout=stdout,
        engine_set=es, player=FakePlayer(), capture=FakeCapture(),
        inference=InstantExecutor(),
    )
    s._exit = lambda code: None
    s.run()
    time.sleep(0.3)
    assert fake_llm.last_messages[0] == {"role": "system", "content": "You are a pirate."}


def test_boot_load_failure_then_recovery_via_reload(tmp_path):
    """Regression guard for the phase-A review boot-load-recovery finding:
    if one engine fails to load at boot, serve must NOT emit engines_ready
    (the old all-or-nothing _load_engines either emitted engines_ready for
    a fully-loaded set or silently emitted nothing useful past one error),
    and a client that fixes the config (set_config llm.model -> reload)
    must be able to complete the boot via the normal reload path, with
    engines_ready firing once the recovered engine finishes loading."""
    cfg = Config(SttConfig(), LlmConfig(), TtsConfig(), KeysConfig(), AudioConfig(), ToolsConfig())
    cfg_path = tmp_path / "localvoice.toml"
    cfg_path.write_text("")

    load_calls = {"n": 0}

    class FlakyLLM(FakeLLM):
        def load(self) -> None:
            load_calls["n"] += 1
            if load_calls["n"] == 1:
                raise RuntimeError("model path not found")

    factories = {
        "stt": lambda c: FakeSTT("hello there"),
        "llm": lambda c: FlakyLLM(["Hi from the fake. ", "More words."]),
        "tts": lambda c: FakeTTS(),
    }
    es = EngineSet(cfg, factories=factories)
    commands = [
        {"cmd": "set_config", "changes": {"llm.model": "other/model"}},
        {"cmd": "ptt_down"},
        {"cmd": "ptt_up", "held_ms": 500},
        {"cmd": "shutdown"},
    ]
    stdin = io.StringIO("".join(json.dumps(c) + "\n" for c in commands))
    stdout = io.StringIO()
    s = Serve(
        config_path=cfg_path, cfg=cfg, allow_inject=True, stdin=stdin, stdout=stdout,
        engine_set=es, player=FakePlayer(), capture=FakeCapture(),
        inference=InstantExecutor(),
    )
    s._exit = lambda code: None
    s.run()
    time.sleep(0.3)
    msgs = [json.loads(line) for line in stdout.getvalue().splitlines()]

    # Startup happens synchronously before any command is read (InstantExecutor
    # runs _load_engines inline inside run()); the llm's first load() call raised.
    # The SECOND load_progress(engine=llm, phase=start) is the reload triggered by
    # set_config -- the boundary between "pure startup" and "processing that command".
    def _is_llm_start(m: dict) -> bool:
        return (
            m.get("event") == "load_progress"
            and m.get("engine") == "llm"
            and m.get("phase") == "start"
        )

    llm_start_indices = [i for i, m in enumerate(msgs) if _is_llm_start(m)]
    assert len(llm_start_indices) == 2, "expected one load_progress(start) per llm load() call"
    reload_start_idx = llm_start_indices[1]
    startup_msgs = msgs[:reload_start_idx]
    assert not events_of(startup_msgs, "engines_ready")
    assert any("llm" in m["message"] for m in events_of(startup_msgs, "error"))
    # The reload (triggered by set_config llm.model) completes the boot: llm's
    # second load() call succeeds, so engines_ready fires once that reload lands,
    # at or before the set_config command's config_applied reply.
    ready_idx = next(i for i, m in enumerate(msgs) if m.get("event") == "engines_ready")
    assert len(events_of(msgs, "engines_ready")) == 1  # fires at most once
    config_applied_idx = next(i for i, m in enumerate(msgs) if m.get("event") == "config_applied")
    assert ready_idx > reload_start_idx
    assert ready_idx <= config_applied_idx
    assert load_calls["n"] == 2
    # And the pipeline is now fully usable.
    assert events_of(msgs, "user_text")


def test_preview_voice_restores_original_voice_and_reaches_player(tmp_path):
    """preview_voice must synthesize through the real player and restore the
    configured voice afterward, in the ordinary (non-concurrent) case."""
    cfg = Config(SttConfig(), LlmConfig(), TtsConfig(), KeysConfig(), AudioConfig(), ToolsConfig())
    original_voice = cfg.tts.voice
    cfg_path = tmp_path / "localvoice.toml"
    cfg_path.write_text("")
    factories = {
        "stt": lambda c: FakeSTT("hello there"),
        "llm": lambda c: FakeLLM(["hi"]),
        "tts": lambda c: FakeTTS(),
    }
    es = EngineSet(cfg, factories=factories)
    player = FakePlayer()
    commands = [{"cmd": "preview_voice", "voice": "af_bella"}, {"cmd": "shutdown"}]
    stdin = io.StringIO("".join(json.dumps(c) + "\n" for c in commands))
    stdout = io.StringIO()
    s = Serve(
        config_path=cfg_path, cfg=cfg, allow_inject=True, stdin=stdin, stdout=stdout,
        engine_set=es, player=player, capture=FakeCapture(),
        inference=InstantExecutor(),
    )
    s._exit = lambda code: None
    s.run()
    time.sleep(0.3)
    assert any(entry[0] == "submit_raw" for entry in player.log)
    assert s._cfg.tts.voice == original_voice


def test_preview_voice_cas_restore_does_not_clobber_concurrent_set_config(tmp_path):
    """Regression guard for the phase-A review preview-restore race: if a
    set_config changes tts.voice WHILE a preview job is mid-synthesis, the
    preview's finally block must not blindly stomp that concurrent change
    back to the pre-preview voice. Compare-and-swap semantics: only restore
    if the live voice is still what preview itself set."""
    cfg = Config(SttConfig(), LlmConfig(), TtsConfig(), KeysConfig(), AudioConfig(), ToolsConfig())
    cfg_path = tmp_path / "localvoice.toml"
    cfg_path.write_text("")

    class MutatingTTS:
        def __init__(self, serve_holder):
            self._serve_holder = serve_holder

        def load(self) -> None:
            pass

        def synthesize(self, text: str):
            import numpy as np

            yield np.full(10, 0.1, np.float32)
            # Simulate a concurrent set_config landing mid-synthesis, on the same
            # single inference thread (a reload or another preview could not
            # interleave here in production, but a set_config's instant-path
            # write to self._cfg.tts.voice happens on the stdin-loop thread and
            # is not otherwise synchronized with this job).
            self._serve_holder["serve"]._cfg.tts.voice = "af_other"
            yield np.full(10, 0.1, np.float32)

    factories = {
        "stt": lambda c: FakeSTT("hello there"),
        "llm": lambda c: FakeLLM(["hi"]),
        "tts": lambda c: MutatingTTS(holder),
    }
    holder: dict = {}
    es = EngineSet(cfg, factories=factories)
    commands = [{"cmd": "preview_voice", "voice": "af_bella"}, {"cmd": "shutdown"}]
    stdin = io.StringIO("".join(json.dumps(c) + "\n" for c in commands))
    stdout = io.StringIO()
    s = Serve(
        config_path=cfg_path, cfg=cfg, allow_inject=True, stdin=stdin, stdout=stdout,
        engine_set=es, player=FakePlayer(), capture=FakeCapture(),
        inference=InstantExecutor(),
    )
    holder["serve"] = s
    s._exit = lambda code: None
    s.run()
    time.sleep(0.3)
    assert s._cfg.tts.voice == "af_other"


def test_ptt_down_before_engines_ready_yields_error(tmp_path):
    """Backfill regression guard: a client that races the boot sequence (sends
    ptt_down before the boot load job has even run, e.g. immediately after
    the process starts) must get the standard "engines still loading" error,
    never a crash or a silently-dropped command. Using ManualExecutor to
    defer the load job proves this holds even in the window before
    _load_engines has been executed at all -- not just while it's running."""
    cfg = Config(SttConfig(), LlmConfig(), TtsConfig(), KeysConfig(), AudioConfig(), ToolsConfig())
    cfg_path = tmp_path / "localvoice.toml"
    cfg_path.write_text("")
    factories = {
        "stt": lambda c: FakeSTT("hello there"),
        "llm": lambda c: FakeLLM(["hi"]),
        "tts": lambda c: FakeTTS(),
    }
    es = EngineSet(cfg, factories=factories)
    manual = ManualExecutor()
    commands = [{"cmd": "ptt_down"}]
    stdin = io.StringIO("".join(json.dumps(c) + "\n" for c in commands))
    stdout = io.StringIO()
    s = Serve(
        config_path=cfg_path, cfg=cfg, allow_inject=True, stdin=stdin, stdout=stdout,
        engine_set=es, player=FakePlayer(), capture=FakeCapture(),
        inference=manual,
    )
    s._exit = lambda code: None
    s.run()  # returns at stdin EOF; ptt_down was dispatched before the load job ran
    msgs = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert any("engines still loading" in m["message"] for m in events_of(msgs, "error"))
    assert not s._engines_ready
    # The deferred load job can still be run to completion cleanly afterward.
    manual.run_all()
    assert s._engines_ready


def test_restart_audio_stops_then_starts_the_player(tmp_path):
    """Backfill regression guard: set_config on an audio.* field (restart_audio
    group) must actually stop() then start() the player -- proving the hot-apply
    restart path (which now also re-arms PlaybackQueue's rebuffer gate via
    AudioPlayer.start(), item 4) really runs, not just that config_applied
    fires."""
    cfg = Config(SttConfig(), LlmConfig(), TtsConfig(), KeysConfig(), AudioConfig(), ToolsConfig())
    cfg_path = tmp_path / "localvoice.toml"
    cfg_path.write_text("")
    factories = {
        "stt": lambda c: FakeSTT("hello there"),
        "llm": lambda c: FakeLLM(["hi"]),
        "tts": lambda c: FakeTTS(),
    }
    es = EngineSet(cfg, factories=factories)
    player = FakePlayer()
    commands = [
        {"cmd": "set_config", "changes": {"audio.rebuffer_ms": 400}},
        {"cmd": "shutdown"},
    ]
    stdin = io.StringIO("".join(json.dumps(c) + "\n" for c in commands))
    stdout = io.StringIO()
    s = Serve(
        config_path=cfg_path, cfg=cfg, allow_inject=True, stdin=stdin, stdout=stdout,
        engine_set=es, player=player, capture=FakeCapture(),
        inference=InstantExecutor(),
    )
    s._exit = lambda code: None
    s.run()
    time.sleep(0.3)
    kinds = [entry[0] for entry in player.log]
    # run() calls player.start() once at boot and _shutdown() calls player.stop()
    # once at the end; _restart_audio's own stop()/start() pair sits between them.
    assert kinds == ["start", "stop", "start", "stop"], kinds


def test_dispatch_exception_boundary_survives_bad_message_shape(tmp_path):
    """A raw JSON line that decodes but isn't a dict (e.g. a bare list) must
    not kill the stdin loop: serve should emit an error naming the bad
    message and keep reading, ultimately handling shutdown cleanly."""
    msgs = build(tmp_path, [], raw_lines=["[1]", json.dumps({"cmd": "shutdown"})])
    errors = events_of(msgs, "error")
    assert any("bad message" in m["message"] for m in errors)
    # serve survived to process shutdown: shutdown() calls os._exit, which we've
    # stubbed to a no-op in build(), so the presence of the "ready" event plus no
    # hang is the practical proxy — assert the run() call returned at all by
    # checking we got output beyond just "ready".
    assert len(msgs) >= 2
