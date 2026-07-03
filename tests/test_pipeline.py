import threading

import numpy as np

from localvoice.events import EventType as E
from localvoice.pipeline import PipelineDeps, run_pipeline
from localvoice.transcript import Transcript
from tests.fakes import FakeLLM, FakePlayer, FakeSTT, FakeTTS

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

        def stream(self, messages, *, think):
            for i, d in enumerate(super().stream(messages, think=think)):
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
