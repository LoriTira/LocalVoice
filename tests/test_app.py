import numpy as np

from localvoice.app import Orchestrator
from localvoice.config import KeysConfig, ToolsConfig
from localvoice.events import Event
from localvoice.events import EventType as E
from localvoice.events import State as S
from localvoice.transcript import Transcript
from tests.fakes import EchoTool, FakeLLM, FakePlayer, FakeSTT, FakeTTS, ScriptedToolLLM


class FakeCapture:
    def __init__(self):
        self.log = []

    def arm(self):
        self.log.append("arm")

    def disarm(self):
        self.log.append("disarm")
        rng = np.random.default_rng(1)
        return (0.1 * rng.standard_normal(16000)).astype(np.float32)

    def discard(self):
        self.log.append("discard")


def make() -> tuple[Orchestrator, FakeCapture, FakePlayer, Transcript]:
    capture, player, transcript = FakeCapture(), FakePlayer(), Transcript("sys")
    orch = Orchestrator(
        capture=capture, player=player, stt=FakeSTT("hi"), llm=FakeLLM(["Hello there. ", "More."]),
        tts=FakeTTS(), transcript=transcript, keys_cfg=KeysConfig(), status=lambda s: None,
    )
    return orch, capture, player, transcript


def drain(orch: Orchestrator) -> None:
    # process queued events until empty AND the pipeline job (if any) has finished
    import time

    for _ in range(200):
        if orch._pipeline_future is not None and not orch._pipeline_future.done():
            time.sleep(0.01)
            continue
        try:
            ev = orch._queue.get_nowait()
        except Exception:
            break
        orch.handle(ev)


def test_full_turn_reaches_idle_and_commits():
    orch, capture, player, transcript = make()
    orch.handle(Event(E.PTT_DOWN))
    assert orch.state is S.LISTENING and capture.log == ["arm"]
    orch.handle(Event(E.PTT_UP, held_ms=400))
    assert orch.state is S.PROCESSING and "disarm" in capture.log
    # pipeline thread posts FIRST_AUDIO; FakePlayer never drain-fires, so finish manually
    drain(orch)
    orch.handle(Event(E.RESPONSE_FINISHED, gen=orch._gen))
    assert orch.state is S.IDLE
    assert transcript.history()[-1]["role"] == "assistant"


def test_short_tap_discards():
    orch, capture, *_ = make()
    orch.handle(Event(E.PTT_DOWN))
    orch.handle(Event(E.PTT_UP, held_ms=50))
    assert orch.state is S.IDLE and capture.log[-1] == "discard"


def test_barge_in_truncates_and_relistens():
    orch, capture, player, transcript = make()
    orch.handle(Event(E.PTT_DOWN))
    orch.handle(Event(E.PTT_UP, held_ms=400))
    drain(orch)
    orch.handle(Event(E.FIRST_AUDIO, gen=orch._gen))
    assert orch.state is S.SPEAKING
    orch.handle(Event(E.PTT_DOWN))  # barge-in
    assert orch.state is S.LISTENING
    assert ("flush",) in player.log
    last = transcript.history()[-1]
    assert last["role"] == "assistant"
    assert last["content"].startswith("Hello there.")  # spoken clause survives truncation
    assert capture.log.count("arm") == 2


def test_stale_generation_events_are_dropped():
    orch, *_ = make()
    orch.handle(Event(E.PTT_DOWN))
    orch.handle(Event(E.PTT_UP, held_ms=400))
    drain(orch)
    before = orch.state  # drain may already have consumed FIRST_AUDIO
    orch.handle(Event(E.RESPONSE_FINISHED, gen=orch._gen - 1))
    assert orch.state is before  # stale event ignored; a live one would move to IDLE


def test_esc_during_speaking_cancels_to_idle():
    orch, capture, player, transcript = make()
    orch.handle(Event(E.PTT_DOWN))
    orch.handle(Event(E.PTT_UP, held_ms=400))
    drain(orch)
    orch.handle(Event(E.FIRST_AUDIO, gen=orch._gen))
    orch.handle(Event(E.ESC))
    assert orch.state is S.IDLE and ("flush",) in player.log


def test_on_state_callback_receives_transitions():
    seen: list = []
    capture, player, transcript = FakeCapture(), FakePlayer(), Transcript("sys")
    orch = Orchestrator(
        capture=capture, player=player, stt=FakeSTT("hi"),
        llm=FakeLLM(["Hello there."]), tts=FakeTTS(), transcript=transcript,
        keys_cfg=KeysConfig(), status=lambda s: None, on_state=seen.append,
    )
    orch.handle(Event(E.PTT_DOWN))
    assert seen[-1] is S.LISTENING
    orch.handle(Event(E.ESC))
    assert seen[-1] is S.IDLE


def test_shutdown_drains_queued_events_before_exit():
    import threading

    orch, capture, player, transcript = make()
    orch.post(Event(E.PTT_DOWN))
    orch.post(Event(E.PTT_UP, held_ms=400))
    orch.shutdown()  # sentinel queued AFTER the two events
    t = threading.Thread(target=orch.run_forever, daemon=True)
    t.start()
    t.join(timeout=5)
    assert not t.is_alive()
    assert "disarm" in capture.log  # both PTT events were processed first


def _run_one_turn(orch: Orchestrator) -> None:
    orch.handle(Event(E.PTT_DOWN))
    orch.handle(Event(E.PTT_UP, held_ms=400))
    drain(orch)
    orch.handle(Event(E.RESPONSE_FINISHED, gen=orch._gen))


def test_tools_offered_when_enabled_and_engine_supports_them():
    tool = EchoTool()
    tools_cfg = ToolsConfig(enabled=True)
    llm = ScriptedToolLLM([["Answer."]], supports_tools=True)
    capture, player, transcript = FakeCapture(), FakePlayer(), Transcript("sys")
    orch = Orchestrator(
        capture=capture, player=player, stt=FakeSTT("hi"), llm=llm, tts=FakeTTS(),
        transcript=transcript, keys_cfg=KeysConfig(), tools_cfg=tools_cfg,
        tools_factory=lambda cfg: [tool] if cfg.enabled else [], status=lambda s: None,
    )
    _run_one_turn(orch)
    assert llm.calls[0]["tools"] and isinstance(llm.calls[0]["tools"], list)


def test_tools_withheld_when_engine_does_not_support_them():
    """Global Constraints: cfg.tools.enabled alone is not sufficient — a
    model whose chat template can't render tool calls must never be offered
    tools, regardless of config."""
    tool = EchoTool()
    tools_cfg = ToolsConfig(enabled=True)
    llm = ScriptedToolLLM([["Answer."]], supports_tools=False)
    capture, player, transcript = FakeCapture(), FakePlayer(), Transcript("sys")
    orch = Orchestrator(
        capture=capture, player=player, stt=FakeSTT("hi"), llm=llm, tts=FakeTTS(),
        transcript=transcript, keys_cfg=KeysConfig(), tools_cfg=tools_cfg,
        tools_factory=lambda cfg: [tool] if cfg.enabled else [], status=lambda s: None,
    )
    _run_one_turn(orch)
    assert llm.calls[0]["tools"] is None


def test_tools_cfg_hot_applies_at_the_start_of_the_next_turn():
    """docs/gui.md's `[tools]` hot-apply paragraph: the tool registry is
    rebuilt from the live ToolsConfig at the START of every turn (inside
    _start_pipeline), not cached once at Orchestrator construction. Mutating
    the SAME ToolsConfig object serve.py's set_config would mutate (see
    Serve._apply_new_config's instant-path setattr) must change what the
    very next turn offers, with no Orchestrator reconstruction needed."""
    tool = EchoTool()
    tools_cfg = ToolsConfig(enabled=True)
    llm = ScriptedToolLLM([["First."], ["Second."]], supports_tools=True)
    capture, player, transcript = FakeCapture(), FakePlayer(), Transcript("sys")
    orch = Orchestrator(
        capture=capture, player=player, stt=FakeSTT("hi"), llm=llm, tts=FakeTTS(),
        transcript=transcript, keys_cfg=KeysConfig(), tools_cfg=tools_cfg,
        tools_factory=lambda cfg: [tool] if cfg.enabled else [], status=lambda s: None,
    )
    _run_one_turn(orch)
    assert llm.calls[0]["tools"] and isinstance(llm.calls[0]["tools"], list)
    tools_cfg.enabled = False  # mirrors set_config's live-object mutation
    _run_one_turn(orch)
    assert llm.calls[1]["tools"] is None
