import json
import threading

import numpy as np

from localvoice.events import EventType as E
from localvoice.pipeline import PipelineDeps, run_pipeline
from localvoice.transcript import Transcript
from tests.fakes import EchoTool, FakeLLM, FakePlayer, FakeSTT, FakeTTS, ScriptedToolLLM

SR = 16000


def speech(seconds: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (0.1 * rng.standard_normal(int(SR * seconds))).astype(np.float32)


def run(deps: PipelineDeps, audio: np.ndarray, cancel: threading.Event | None = None) -> list:
    events: list = []
    run_pipeline(audio, SR, deps, cancel or threading.Event(), events.append)
    return events


def make_deps(**kw) -> PipelineDeps:
    return PipelineDeps(
        stt=kw.get("stt", FakeSTT("what is two plus two")),
        llm=kw.get("llm", FakeLLM(["Four", ". ", "Easy ", "one."])),
        tts=kw.get("tts", FakeTTS()),
        player=kw.get("player", FakePlayer()),
        transcript=kw.get("transcript", Transcript("sys")),
        **{k: v for k, v in kw.items() if k in ("think", "min_rms", "min_seconds")},
    )


def test_happy_path_events_and_history():
    t = Transcript("sys")
    player = FakePlayer()
    deps = make_deps(transcript=t, player=player)
    events = run(deps, speech())
    types = [e.type for e in events]
    assert types.count(E.FIRST_AUDIO) == 1
    assert E.PIPELINE_ERROR not in types
    assert ("mark_end",) in player.log
    # commit is the orchestrator's job on RESPONSE_FINISHED; pending still open here
    t.commit()
    assert t.history()[0] == {"role": "user", "content": "what is two plus two"}
    assert "Four." in t.history()[1]["content"]


def test_quiet_audio_discards_without_stt():
    events = run(make_deps(), np.zeros(SR, np.float32))
    assert [e.type for e in events] == [E.RESPONSE_FINISHED]


def test_too_short_audio_discards():
    events = run(make_deps(), speech(0.1))
    assert [e.type for e in events] == [E.RESPONSE_FINISHED]


def test_empty_transcription_discards():
    events = run(make_deps(stt=FakeSTT("  ")), speech())
    assert [e.type for e in events] == [E.RESPONSE_FINISHED]


def test_cancel_before_start_produces_nothing():
    cancel = threading.Event()
    cancel.set()
    deps = make_deps()
    events = run(deps, speech(), cancel)
    assert events == [] and deps.player.log == []


def test_cancel_mid_llm_stops_and_stays_silent():
    class CancellingLLM(FakeLLM):
        def __init__(self, cancel):
            super().__init__(["First bit. ", "Second bit. ", "Third."])
            self._cancel = cancel

        def stream(self, messages, *, think, tools=None, image_path=None):
            for i, d in enumerate(super().stream(messages, think=think, image_path=image_path)):
                if i == 1:
                    self._cancel.set()
                yield d

    cancel = threading.Event()
    deps = make_deps(llm=CancellingLLM(cancel))
    events = run(deps, speech(), cancel)
    assert all(e.type == E.FIRST_AUDIO for e in events)  # no finish/error after cancel
    assert ("mark_end",) not in deps.player.log


def test_error_emits_pipeline_error():
    class BoomTTS(FakeTTS):
        def synthesize(self, text):
            raise RuntimeError("kaboom")
            yield  # pragma: no cover

    events = run(make_deps(tts=BoomTTS()), speech())
    assert events[-1].type == E.PIPELINE_ERROR
    assert "kaboom" in events[-1].message


def test_think_flag_reaches_llm_and_markup_is_stripped():
    llm = FakeLLM(["<think>hmm</think>", "**Bold** answer. ", "Tail"])
    tts = FakeTTS()
    deps = make_deps(llm=llm, tts=tts, think=True)
    run(deps, speech())
    assert llm.last_think is True
    assert tts.texts[0] == "Bold answer."
    assert "Tail" in tts.texts[-1]


def test_empty_llm_output_finishes_cleanly():
    t = Transcript("sys")
    events = run(make_deps(llm=FakeLLM([]), transcript=t), speech())
    assert events[-1].type == E.RESPONSE_FINISHED
    # pending was aborted, so a later commit must no-op and leave no turn behind
    t.commit()
    assert t.history() == []


def test_metrics_emitted_once_with_expected_keys():
    metrics: list[dict] = []
    deps = make_deps()
    deps.on_metrics = metrics.append
    run(deps, speech())
    assert len(metrics) == 1
    m = metrics[0]
    assert set(m) == {"stt", "ttft", "first_clause", "tts_first", "total"}
    assert all(isinstance(v, float) and v >= 0.0 for v in m.values())
    assert m["total"] >= m["stt"]


def test_metrics_not_emitted_on_quiet_discard_or_empty_output():
    metrics: list[dict] = []
    deps = make_deps()
    deps.on_metrics = metrics.append
    run(deps, np.zeros(SR, np.float32))
    deps2 = make_deps(llm=FakeLLM([]))
    deps2.on_metrics = metrics.append
    run(deps2, speech())
    assert metrics == []


def test_metrics_not_emitted_on_cancel():
    metrics: list[dict] = []
    cancel = threading.Event()
    cancel.set()
    deps = make_deps()
    deps.on_metrics = metrics.append
    run(deps, speech(), cancel)
    assert metrics == []


# --- tool rounds (Task 6) ---------------------------------------------------

_CALL = '<|tool_call>call:web_search{query:<|"|>rain<|"|>}<tool_call|>'


def test_tool_round_executes_and_speaks_continuation():
    tool = EchoTool()
    tts = FakeTTS()
    calls, results = [], []
    llm = ScriptedToolLLM([["Let me check. ", _CALL], ["It will rain at noon."]])
    deps = make_deps(llm=llm, tts=tts)
    deps.tools = [tool]
    deps.on_tool_call = lambda name, summary: calls.append((name, summary))
    deps.on_tool_result = lambda name, ok, summary: results.append((name, ok, summary))
    run(deps, speech())

    assert tool.executed == [{"query": "rain"}]
    # Both clauses spoken across the two rounds; no marker text ever reaches TTS.
    assert " ".join(tts.texts) == "Let me check. It will rain at noon."
    assert not any("tool_call" in t or "call:" in t for t in tts.texts)
    # Second stream call carries the tool result as a role:"tool" message. Its
    # content is a JSON STRING (not the raw dict): the Gemma chat template's
    # role:"tool" branch treats a dict as a sequence and iterates its keys,
    # crashing the continuation render — so the dict is JSON-encoded at the
    # message-construction site. Round-trips back to the original dict.
    second = llm.calls[1]["messages"]
    assert second[-1]["role"] == "tool"
    assert isinstance(second[-1]["content"], str)
    assert json.loads(second[-1]["content"]) == {
        "results": [{"title": "T", "url": "u", "snippet": "s"}]
    }
    assert calls == [("web_search", "Calling web_search")]
    assert results == [("web_search", True, "found 1 result")]
    # The pipeline doesn't thread image_path yet (that's T3 Task 4) -- the
    # fake must still default/record it honestly rather than erroring.
    assert llm.last_image_path is None


def test_last_round_offers_no_tools():
    llm = ScriptedToolLLM([["ok ", _CALL], ["Final answer."]])
    deps = make_deps(llm=llm, tts=FakeTTS())
    deps.tools = [EchoTool()]
    deps.max_tool_rounds = 1
    run(deps, speech())
    assert llm.calls[0]["tools"] and isinstance(llm.calls[0]["tools"], list)
    assert llm.calls[1]["tools"] is None


def test_unknown_tool_gets_error_round():
    bad = '<|tool_call>call:nope{query:<|"|>x<|"|>}<tool_call|>'
    llm = ScriptedToolLLM([["hmm ", bad], ["Recovered."]])
    results = []
    deps = make_deps(llm=llm, tts=FakeTTS())
    deps.tools = [EchoTool()]
    deps.on_tool_result = lambda name, ok, summary: results.append((name, ok, summary))
    events = run(deps, speech())
    assert E.PIPELINE_ERROR not in [e.type for e in events]
    second = llm.calls[1]["messages"]
    assert second[-1]["role"] == "tool"
    # Error results are JSON-encoded too (same template constraint as the
    # success path — both dict shapes crash the render as a raw dict).
    assert isinstance(second[-1]["content"], str)
    assert json.loads(second[-1]["content"]) == {"error": "unknown tool: nope"}
    assert results and results[0][1] is False


def test_cancel_during_tool_aborts_turn():
    class CancelTool:
        name = "web_search"
        description = "d"
        parameters = {"type": "object", "properties": {}}

        def __init__(self, cancel):
            self._cancel = cancel

        def execute(self, args, cancel):
            from localvoice.tools.base import ToolResult

            self._cancel.set()
            return ToolResult(ok=True, content={"results": []}, summary="found 0 results")

    cancel = threading.Event()
    llm = ScriptedToolLLM([["Let me check. ", _CALL], ["Should never run."]])
    deps = make_deps(llm=llm, tts=FakeTTS())
    deps.tools = [CancelTool(cancel)]
    events = run(deps, speech(), cancel)
    assert len(llm.calls) == 1  # cancelled before the second stream
    # Mirror the mid-llm cancel test: no finish/error event after cancellation.
    assert all(e.type == E.FIRST_AUDIO for e in events)
    assert ("mark_end",) not in deps.player.log


def test_no_tools_configured_is_todays_behavior():
    llm = ScriptedToolLLM([["Four", ". ", "Easy ", "one."]])
    tts = FakeTTS()
    deps = make_deps(llm=llm, tts=tts)
    deps.tools = []
    events = run(deps, speech())
    assert llm.calls[0]["tools"] is None
    assert len(llm.calls) == 1
    assert E.FIRST_AUDIO in [e.type for e in events]
    assert "".join(tts.texts).replace(" ", "") == "Four.Easyone."


def test_transcript_untouched_by_tool_traffic():
    t = Transcript("sys")
    tool = EchoTool()
    llm = ScriptedToolLLM([["Let me check. ", _CALL], ["It will rain at noon."]])
    deps = make_deps(llm=llm, tts=FakeTTS(), transcript=t)
    deps.tools = [tool]
    run(deps, speech())
    for m in t.messages():
        assert m["role"] != "tool"
        assert "tool_calls" not in m


# --- tools-T1 final-review fixes ------------------------------------------

_EMPTY_CALL = "<|tool_call>call:web_search{}<tool_call|>"


def test_tool_execute_exception_becomes_error_round_not_dead_turn():
    # A syntactically valid call missing a required arg raises KeyError inside
    # execute; that must NOT kill the turn. It should surface as a failed
    # tool_result round (ok=False) and the model's continuation is still spoken.
    class RaisingTool:
        name = "web_search"
        description = "d"
        parameters = {"type": "object", "properties": {}}

        def execute(self, args, cancel):
            return args["query"]  # KeyError: 'query' — missing required arg

    tts = FakeTTS()
    results = []
    llm = ScriptedToolLLM([["Let me look. ", _EMPTY_CALL], ["Here is the answer."]])
    deps = make_deps(llm=llm, tts=tts)
    deps.tools = [RaisingTool()]
    deps.on_tool_result = lambda name, ok, summary: results.append((name, ok, summary))
    events = run(deps, speech())

    assert E.PIPELINE_ERROR not in [e.type for e in events]  # turn survived
    assert results and results[0][1] is False  # on_tool_result fired ok=False
    # The continuation round ran and its answer was spoken.
    assert " ".join(tts.texts) == "Let me look. Here is the answer."
    # The tool message carried the exception type/message back to the model.
    second = llm.calls[1]["messages"]
    assert second[-1]["role"] == "tool"
    payload = json.loads(second[-1]["content"])
    assert "KeyError" in payload["error"]


def test_malformed_call_feedback_includes_raw():
    # The malformed tool_result must carry the captured raw text so the model
    # sees exactly what it got wrong (spec conformance).
    bad_raw = "<|tool_call>call:web_search{query}<tool_call|>"  # missing ':' -> malformed
    llm = ScriptedToolLLM([["hmm ", bad_raw], ["Recovered."]])
    calls = []
    deps = make_deps(llm=llm, tts=FakeTTS())
    deps.tools = [EchoTool()]
    deps.on_tool_call = lambda name, summary: calls.append((name, summary))
    events = run(deps, speech())

    assert E.PIPELINE_ERROR not in [e.type for e in events]
    assert calls == [("web_search", "Malformed tool call")]
    second = llm.calls[1]["messages"]
    payload = json.loads(second[-1]["content"])
    assert payload["error"] == "malformed tool call"
    assert payload["raw"] == bad_raw  # the raw capture round-trips to the model


def test_call_inside_unclosed_think_still_speaks_continuation():
    # The call marker arrives while an unclosed <think> block is open. Without
    # force_close_think the filter would stay in think mode across the
    # continuation round and swallow the whole answer as reasoning. Instead the
    # partial reasoning reaches on_thinking and the continuation IS spoken.
    thoughts = []
    tts = FakeTTS()
    llm = ScriptedToolLLM(
        [["<think>I should search for this. ", _CALL], ["It will rain at noon."]]
    )
    deps = make_deps(llm=llm, tts=tts)
    deps.tools = [EchoTool()]
    deps.on_thinking = thoughts.append
    events = run(deps, speech())

    assert E.PIPELINE_ERROR not in [e.type for e in events]
    # The continuation answer is spoken (not swallowed as reasoning), and the
    # pre-call reasoning marker text never leaks into TTS.
    assert " ".join(tts.texts) == "It will rain at noon."
    assert not any("search for this" in t for t in tts.texts)
    # The partial reasoning was surfaced to the thinking observer.
    assert any("I should search for this" in t for t in thoughts)
