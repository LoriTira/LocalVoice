import queue
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor

from localvoice.audio.earcons import EARCONS
from localvoice.config import KeysConfig
from localvoice.events import Action as A
from localvoice.events import Event
from localvoice.events import EventType as E
from localvoice.events import State as S
from localvoice.pipeline import PipelineDeps, run_pipeline
from localvoice.states import transition

_PIPELINE_EVENTS = {E.FIRST_AUDIO, E.RESPONSE_FINISHED, E.PIPELINE_ERROR}


class Orchestrator:
    def __init__(
        self,
        *,
        capture,
        player,
        stt,
        llm,
        tts,
        transcript,
        keys_cfg: KeysConfig,
        inference: ThreadPoolExecutor | None = None,
        think: bool = False,
        status: Callable[[str], None] = print,
    ) -> None:
        self.state = S.IDLE
        self._capture = capture
        self._player = player
        self._transcript = transcript
        self._keys = keys_cfg
        self._status = status
        self._deps = PipelineDeps(
            stt=stt, llm=llm, tts=tts, player=player, transcript=transcript, think=think,
            on_user_text=lambda t: status(f"you: {t}"),
            on_assistant_clause=lambda t: status(f"assistant: {t}"),
        )
        self._queue: queue.Queue[Event] = queue.Queue()
        self._cancel = threading.Event()
        # MLX streams are usable only on the thread that created them (mlx-lm builds
        # its generation stream at import time), so ALL engine imports, loads, and
        # pipeline runs must share this one persistent inference thread.
        self._inference = inference or ThreadPoolExecutor(max_workers=1)
        self._pipeline_future: Future | None = None
        self._gen = 0
        self._running = True

    def post(self, event: Event) -> None:
        self._queue.put(event)

    def shutdown(self) -> None:
        self._running = False
        self._cancel.set()
        self._queue.put(Event(E.ESC))  # wake the loop

    def run_forever(self) -> None:
        while self._running:
            self.handle(self._queue.get())

    def handle(self, event: Event) -> None:
        if event.type in _PIPELINE_EVENTS and event.gen != self._gen:
            return
        self.state, actions = transition(self.state, event, self._keys.debounce_ms)
        for action in actions:
            self._do(action, event)
        self._show_state()

    def _do(self, action: A, event: Event) -> None:
        if action is A.START_CAPTURE:
            self._capture.arm()
        elif action is A.DISCARD_CAPTURE:
            self._capture.discard()
        elif action is A.STOP_CAPTURE_AND_RUN:
            self._start_pipeline()
        elif action is A.CANCEL_PIPELINE:
            self._cancel.set()
        elif action is A.FLUSH_AUDIO:
            self._player.flush()
        elif action is A.TRUNCATE_HISTORY:
            self._transcript.truncate_commit(self._player.spoken_tags())
        elif action is A.COMMIT_TURN:
            self._transcript.commit()
        elif action is A.REPORT_ERROR:
            self._status(f"error: {event.message}")
        elif action is A.EARCON_START:
            self._player.submit_raw(EARCONS["start"])
        elif action is A.EARCON_STOP:
            self._player.submit_raw(EARCONS["stop"])
        elif action is A.EARCON_CANCEL:
            self._player.submit_raw(EARCONS["cancel"])

    def _start_pipeline(self) -> None:
        audio = self._capture.disarm()
        self._gen += 1
        gen = self._gen
        self._cancel = threading.Event()
        cancel = self._cancel

        def emit(event: Event) -> None:
            self.post(Event(event.type, event.held_ms, event.message, gen))

        self._pipeline_future = self._inference.submit(
            run_pipeline, audio, 16000, self._deps, cancel, emit
        )

    def _show_state(self) -> None:
        labels = {
            S.IDLE: "idle - hold right-command to talk, esc to stop, ctrl-c to quit",
            S.LISTENING: "listening...",
            S.PROCESSING: "thinking...",
            S.SPEAKING: "speaking... (hold key to interrupt)",
        }
        self._status(f"[{labels[self.state]}]")
