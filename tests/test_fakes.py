import numpy as np

from localvoice.llm.base import LLMEngine
from localvoice.stt.base import STTEngine
from localvoice.tts.base import TTSEngine
from tests.fakes import FakeLLM, FakePlayer, FakeSTT, FakeTTS


def test_fakes_satisfy_protocols():
    stt: STTEngine = FakeSTT()
    llm: LLMEngine = FakeLLM(deltas=["a", "b"])
    tts: TTSEngine = FakeTTS()
    stt.load(), llm.load(), tts.load()
    assert stt.transcribe(np.zeros(16000, np.float32), 16000) == "hello"
    assert list(llm.stream([{"role": "user", "content": "x"}], think=False)) == ["a", "b"]
    chunks = list(tts.synthesize("hi"))
    assert len(chunks) == 2 and all(c.dtype == np.float32 for c in chunks)
    p = FakePlayer()
    p.submit(chunks[0], tag=0)
    p.mark_end()
    assert p.spoken_tags() == {0}
    assert [entry[0] for entry in p.log] == ["submit", "mark_end"]
