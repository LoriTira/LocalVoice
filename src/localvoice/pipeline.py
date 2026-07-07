import json
import threading
import time
import traceback
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
from localvoice.textproc.toolcalls import ToolCall, ToolCallParser
from localvoice.tools.base import ToolResult, hf_tool_schema
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
    tools: list = field(default_factory=list)  # Tool instances (already config-gated)
    max_tool_rounds: int = 3
    on_user_text: Callable[[str], None] = field(default=lambda s: None)
    on_assistant_clause: Callable[[str], None] = field(default=lambda s: None)
    on_thinking: Callable[[str], None] = field(default=lambda s: None)
    on_metrics: Callable[[dict], None] = field(default=lambda m: None)
    on_tool_call: Callable[[str, str], None] = field(default=lambda name, summary: None)
    on_tool_result: Callable[[str, bool, str], None] = field(
        default=lambda name, ok, summary: None
    )


def _call_summary(call: ToolCall, malformed: bool) -> str:
    return "Malformed tool call" if malformed else f"Calling {call.name}"


def run_pipeline(
    audio: np.ndarray,
    sample_rate: int,
    deps: PipelineDeps,
    cancel: threading.Event,
    emit: Callable[[Event], None],
) -> None:
    try:
        t0 = time.perf_counter()
        if cancel.is_set():
            return
        too_short = len(audio) < deps.min_seconds * sample_rate
        if too_short or rms(audio) < deps.min_rms:
            emit(Event(EventType.RESPONSE_FINISHED))
            return
        user_text = deps.stt.transcribe(audio, sample_rate)
        t_stt = time.perf_counter()
        if cancel.is_set():
            return
        if not user_text.strip():
            emit(Event(EventType.RESPONSE_FINISHED))
            return
        deps.transcript.begin_turn(user_text)
        deps.on_user_text(user_text)

        text_filter = TextFilter(on_think=deps.on_thinking)
        chunker = ClauseChunker()
        first_audio_sent = False
        spoke_anything = False
        t_first_token = None
        t_first_clause = None
        t_first_submit = None

        def speak(clause: str) -> bool:
            nonlocal first_audio_sent, spoke_anything, t_first_clause, t_first_submit
            spoken = strip_speech_markup(clause)
            if not spoken:
                return True
            t_first_clause = t_first_clause or time.perf_counter()
            tag = deps.transcript.add_clause(spoken)
            deps.on_assistant_clause(spoken)
            for chunk in deps.tts.synthesize(spoken):
                if cancel.is_set():
                    return False
                deps.player.submit(chunk, tag)
                spoke_anything = True
                if not first_audio_sent:
                    t_first_submit = time.perf_counter()
                    first_audio_sent = True
                    emit(Event(EventType.FIRST_AUDIO))
            return True

        # Bounded tool-call rounds. The narration state (text_filter, chunker,
        # speak, and the metric timestamps above) spans the whole turn: a clause
        # spoken in an early round must not be re-flushed or duplicated later. A
        # FRESH ToolCallParser guards each round's raw stream — tool markers are
        # extracted first, then think/code filtering applies to the remainder.
        offered = [hf_tool_schema(t) for t in deps.tools] or None
        by_name = {t.name: t for t in deps.tools}
        messages = deps.transcript.messages()  # turn-local copy; Transcript untouched
        for round_no in range(deps.max_tool_rounds + 1):
            last = round_no == deps.max_tool_rounds
            parser = ToolCallParser()
            # tools=None on the last round forces a spoken answer instead of a call.
            for delta in deps.llm.stream(
                messages, think=deps.think, tools=None if last else offered
            ):
                if cancel.is_set():
                    return
                t_first_token = t_first_token or time.perf_counter()
                for clause in chunker.feed(text_filter.feed(parser.feed(delta))):
                    if not speak(clause):
                        return
                if parser.call is not None:
                    break  # captured mid-stream: stop consuming, round is over
            if cancel.is_set():
                return
            if parser.call is None:
                # Stream ended with no captured call: release any held-back
                # passthrough (a recent fix made finish() surface speech it was
                # holding as a possible marker prefix). finish() may itself set a
                # malformed call for an unclosed capture — handled as a call below.
                for clause in chunker.feed(text_filter.feed(parser.finish())):
                    if not speak(clause):
                        return
            if parser.call is None:
                break  # genuine end of turn: fall through to the single tail flush
            # On the last round (tools withheld) a marker the model emits anyway
            # is still parsed and executed even though its result cannot feed a
            # further round — a deliberate, correctness-neutral literal reading
            # of the contract, not a bug (the observer callbacks stay truthful).
            call = parser.call
            malformed = parser.malformed
            # If the marker was captured inside an unclosed <think> block, the
            # text filter is stuck in think mode; force it closed now (surfacing
            # the partial reasoning) so the continuation round's answer is spoken
            # instead of being swallowed as reasoning. No-op if not in think.
            text_filter.force_close_think()
            deps.on_tool_call(call.name, _call_summary(call, malformed))
            tool = by_name.get(call.name)
            if malformed:
                # Echo the captured raw text back so the model sees exactly what
                # it got wrong and can correct it next round.
                result = ToolResult(
                    ok=False,
                    content={"error": "malformed tool call", "raw": call.raw},
                    summary="tool call failed",
                )
            elif tool is None:
                result = ToolResult(
                    ok=False,
                    content={"error": f"unknown tool: {call.name}"},
                    summary="tool call failed",
                )
            else:
                # A syntactically valid call can still blow up inside execute
                # (e.g. a missing required arg -> KeyError). Contain it as a
                # failed tool result so the model recovers next round, exactly
                # like the unknown-tool path, instead of killing the turn.
                try:
                    result = tool.execute(call.args, cancel)
                except Exception as exc:  # noqa: BLE001 — tool boundary
                    result = ToolResult(
                        ok=False,
                        content={"error": f"{type(exc).__name__}: {exc}"},
                        summary="tool call failed",
                    )
            if cancel.is_set():
                return
            deps.on_tool_result(call.name, result.ok, result.summary)
            messages = messages + [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": f"call_{round_no}",
                            "type": "function",
                            "function": {"name": call.name, "arguments": call.args},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": f"call_{round_no}",
                    "name": call.name,
                    # JSON-encode the dict here: chat templates (e.g. Gemma's)
                    # test a role:"tool" message's `content` with `is sequence`,
                    # which is True for a dict — the template then iterates it as
                    # a list, hits its string keys, and calls .get() on a str,
                    # crashing the continuation render with UndefinedError. A JSON
                    # string renders correctly. ToolResult.content stays a dict as
                    # the tools API; this one site is where it meets the template.
                    "content": json.dumps(result.content),
                },
            ]
        if cancel.is_set():
            return
        tail = (chunker.flush() or "") + text_filter.finish()
        if tail.strip():
            if not speak(tail):
                return
        deps.player.mark_end()
        if not spoke_anything:
            # An empty reply must never commit an empty assistant turn: drop the
            # pending turn before finishing so a later commit() is a no-op.
            deps.transcript.abort_pending()
            emit(Event(EventType.RESPONSE_FINISHED))
        elif spoke_anything and t_first_submit is not None:
            deps.on_metrics(
                {
                    "stt": t_stt - t0,
                    "ttft": (t_first_token or t_stt) - t_stt,
                    "first_clause": (t_first_clause or t_first_token or t_stt)
                    - (t_first_token or t_stt),
                    "tts_first": t_first_submit - (t_first_clause or t_first_submit),
                    "total": t_first_submit - t0,
                }
            )
    except Exception as exc:  # noqa: BLE001 — pipeline boundary
        if not cancel.is_set():
            traceback.print_exc()  # the event carries only str(exc); keep the stack visible
            emit(Event(EventType.PIPELINE_ERROR, message=f"{type(exc).__name__}: {exc}"))
