import threading
from pathlib import Path

import pytest

from localvoice.config import LlmConfig, SttConfig, TtsConfig
from localvoice.events import EventType as E
from localvoice.pipeline import PipelineDeps, run_pipeline
from localvoice.transcript import Transcript
from tests.fakes import FakePlayer

TINY_LLM = Path.home() / ".lmstudio/models/mlx-community/Qwen3.5-0.8B-MLX-4bit"


@pytest.mark.slow
@pytest.mark.skipif(not TINY_LLM.exists(), reason="tiny local LLM not present")
def test_end_to_end_with_real_models():
    from localvoice.bench import _speech_fixture
    from localvoice.llm.mlx_lm_engine import MlxLmEngine
    from localvoice.stt.whisper_mlx import WhisperMlxEngine
    from localvoice.tts.kokoro_mlx import KokoroMlxEngine

    stt = WhisperMlxEngine(SttConfig(model="mlx-community/whisper-tiny"))
    llm = MlxLmEngine(LlmConfig(model=str(TINY_LLM), max_tokens=60))
    tts = KokoroMlxEngine(TtsConfig())
    for e in (stt, llm, tts):
        e.load()
    events: list = []
    player = FakePlayer()
    deps = PipelineDeps(
        stt=stt,
        llm=llm,
        tts=tts,
        player=player,
        transcript=Transcript("Answer in one short sentence."),
    )
    run_pipeline(_speech_fixture(), 16000, deps, threading.Event(), events.append)
    types = [e.type for e in events]
    assert E.FIRST_AUDIO in types and E.PIPELINE_ERROR not in types
    assert any(entry[0] == "submit" for entry in player.log)
    assert ("mark_end",) in player.log


@pytest.mark.slow
@pytest.mark.skipif(not TINY_LLM.exists(), reason="tiny local LLM not present")
def test_two_turn_conversation_reuses_engine():
    from localvoice.llm.mlx_lm_engine import MlxLmEngine

    llm = MlxLmEngine(LlmConfig(model=str(TINY_LLM), max_tokens=40))
    llm.load()
    system = {"role": "system", "content": "Answer in one short sentence."}
    user1 = {"role": "user", "content": "What color is the sky on a clear day?"}
    reply1 = "".join(llm.stream([system, user1], think=False))
    assert reply1.strip()
    assistant1 = {"role": "assistant", "content": "Blue."}
    user2 = {"role": "user", "content": "And at night?"}
    reply2 = "".join(llm.stream([system, user1, assistant1, user2], think=False))
    assert reply2.strip()
