import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from localvoice.audio.capture import rms
from localvoice.events import Event, EventType
from localvoice.llm.base import LLMEngine
from localvoice.stt.base import STTEngine
from localvoice.textproc.chunker import ClauseChunker
from localvoice.textproc.sanitize import TextFilter, strip_speech_markup
from localvoice.transcript import Transcript
from localvoice.tts.base import TTSEngine


@dataclass
class PipelineDeps:
    stt: STTEngine
    llm: LLMEngine
    tts: TTSEngine
    player: Any
    transcript: Transcript
    think: bool = False
    min_rms: float = 0.005
    min_seconds: float = 0.25
    on_user_text: Callable[[str], None] = field(default=lambda s: None)
    on_assistant_clause: Callable[[str], None] = field(default=lambda s: None)


def run_pipeline(
    audio: np.ndarray,
    sample_rate: int,
    deps: PipelineDeps,
    cancel: threading.Event,
    emit: Callable[[Event], None],
) -> None:
    try:
        if cancel.is_set():
            return
        too_short = len(audio) < deps.min_seconds * sample_rate
        if too_short or rms(audio) < deps.min_rms:
            emit(Event(EventType.RESPONSE_FINISHED))
            return
        user_text = deps.stt.transcribe(audio, sample_rate)
        if cancel.is_set():
            return
        if not user_text.strip():
            emit(Event(EventType.RESPONSE_FINISHED))
            return
        deps.transcript.begin_turn(user_text)
        deps.on_user_text(user_text)

        text_filter = TextFilter()
        chunker = ClauseChunker()
        first_audio_sent = False
        spoke_anything = False

        def speak(clause: str) -> bool:
            nonlocal first_audio_sent, spoke_anything
            spoken = strip_speech_markup(clause)
            if not spoken:
                return True
            tag = deps.transcript.add_clause(spoken)
            deps.on_assistant_clause(spoken)
            for chunk in deps.tts.synthesize(spoken):
                if cancel.is_set():
                    return False
                deps.player.submit(chunk, tag)
                spoke_anything = True
                if not first_audio_sent:
                    first_audio_sent = True
                    emit(Event(EventType.FIRST_AUDIO))
            return True

        for delta in deps.llm.stream(deps.transcript.messages(), think=deps.think):
            if cancel.is_set():
                return
            for clause in chunker.feed(text_filter.feed(delta)):
                if not speak(clause):
                    return
        if cancel.is_set():
            return
        tail = (chunker.flush() or "") + text_filter.finish()
        if tail.strip():
            if not speak(tail):
                return
        deps.player.mark_end()
        if not spoke_anything:
            emit(Event(EventType.RESPONSE_FINISHED))
    except Exception as exc:  # noqa: BLE001 — pipeline boundary
        if not cancel.is_set():
            emit(Event(EventType.PIPELINE_ERROR, message=str(exc)))
