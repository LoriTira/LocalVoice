# Architecture

LocalVoice is one Python process built from plain threads and queues, not
asyncio. Every workload on the hot path — MLX inference, PortAudio I/O, the
Quartz key tap — is a blocking C/Metal call, so an event loop would add
bridging complexity without buying anything. The main thread runs a pure
state machine off an event queue; everything that can block runs on its own
thread and talks back to the state machine by posting events.

This document is a condensed, code-grounded companion to the full design
record: `docs/superpowers/specs/2026-07-03-localvoice-architecture-design.md`.
That spec captures the decisions and rationale; this page tracks what the
code actually does, with `src/localvoice/states.py` as the source of truth
for the transition table below.

## Pipeline

```
pynput key tap ──events──▶ Orchestrator (state machine)
sounddevice mic ──frames─▶   │
                             ├─▶ STTEngine.transcribe(audio) ──text──▶
                             ├─▶ LLMEngine.stream(messages) ──deltas──▶ ClauseChunker ──clauses──▶
                             ├─▶ TTSEngine.synthesize(clause) ──chunks──▶ AudioPlayer ──▶ speakers
                             └─▶ Transcript (history + barge-in truncation)
```

Each engine sits behind a small protocol (`STTEngine`, `LLMEngine`,
`TTSEngine` — `src/localvoice/stt/base.py`, `llm/base.py`, `tts/base.py`), so
the orchestrator, pipeline, and state machine are all unit-tested against
fakes (`tests/fakes.py`) without loading a single model.

## State machine

States live in `src/localvoice/events.py::State`:
`IDLE`, `LISTENING`, `PROCESSING`, `SPEAKING`. `PROCESSING` and `SPEAKING`
correspond to the spec's `RESPONDING·processing` / `RESPONDING·speaking`
split — recording has stopped and the pipeline is running in both, but
"waiting for the first sound" and "audio is flowing" warrant different
status text and, in `PROCESSING`, no audio has been queued yet to flush.

The table below is copied from `src/localvoice/states.py` (`_TABLE`, plus the
`PTT_UP` debounce branch handled in `transition()`), naming the exact
`Action` enum members each transition fires, in firing order:

| State | Event | Actions | Next state |
|---|---|---|---|
| `IDLE` | `PTT_DOWN` | `EARCON_START`, `START_CAPTURE` | `LISTENING` |
| `LISTENING` | `PTT_UP`, held ≥ `debounce_ms` (default 120 ms) | `EARCON_STOP`, `STOP_CAPTURE_AND_RUN` | `PROCESSING` |
| `LISTENING` | `PTT_UP`, held < `debounce_ms` | `DISCARD_CAPTURE` | `IDLE` |
| `LISTENING` | `ESC` | `DISCARD_CAPTURE`, `EARCON_CANCEL` | `IDLE` |
| `PROCESSING` | `FIRST_AUDIO` | — | `SPEAKING` |
| `PROCESSING` | `PTT_DOWN` | `CANCEL_PIPELINE`, `TRUNCATE_HISTORY`, `FLUSH_AUDIO`, `EARCON_START`, `START_CAPTURE` | `LISTENING` |
| `PROCESSING` | `ESC` | `CANCEL_PIPELINE`, `TRUNCATE_HISTORY`, `FLUSH_AUDIO`, `EARCON_CANCEL` | `IDLE` |
| `PROCESSING` | `RESPONSE_FINISHED` | `COMMIT_TURN` | `IDLE` |
| `PROCESSING` | `PIPELINE_ERROR` | `TRUNCATE_HISTORY`, `FLUSH_AUDIO`, `REPORT_ERROR` | `IDLE` |
| `SPEAKING` | `PTT_DOWN` | `CANCEL_PIPELINE`, `TRUNCATE_HISTORY`, `FLUSH_AUDIO`, `EARCON_START`, `START_CAPTURE` | `LISTENING` |
| `SPEAKING` | `ESC` | `CANCEL_PIPELINE`, `TRUNCATE_HISTORY`, `FLUSH_AUDIO`, `EARCON_CANCEL` | `IDLE` |
| `SPEAKING` | `RESPONSE_FINISHED` | `COMMIT_TURN` | `IDLE` |
| `SPEAKING` | `PIPELINE_ERROR` | `TRUNCATE_HISTORY`, `FLUSH_AUDIO`, `REPORT_ERROR` | `IDLE` |

Any `(state, event)` pair not listed is a no-op: the state machine returns
the current state and an empty action tuple, so unhandled events (for
example `ESC` while `IDLE`) are silently ignored rather than raising.

`transition()` is pure — no I/O, no engine calls — which is what makes it
practical to exhaustively test every state/event pair with fakes
(`tests/test_states.py`). The `Orchestrator` in `src/localvoice/app.py`
is the only thing that turns `Action` values into real work: mic
start/stop, pipeline thread spawn, player flush, transcript truncation,
earcon playback, and status printing (`_do()`).

## Threading model

- **Main thread** — `Orchestrator.run_forever()` blocks on a
  `queue.Queue[Event]`, pulls one event at a time, and runs it through
  `transition()` then `_do()`. This is the only thread that mutates
  `Orchestrator.state`.
- **Hotkey thread** — `HotkeyListener` (`src/localvoice/hotkey.py`) wraps a
  `pynput.keyboard.Listener` (a Quartz event tap) and posts `PTT_DOWN`,
  `PTT_UP` (with `held_ms`), and `ESC` events from its own callback thread.
- **Audio callback threads** — `MicCapture` and `AudioPlayer`
  (`src/localvoice/audio/capture.py`, `audio/player.py`) each run a
  `sounddevice` stream with its own PortAudio-driven callback thread.
  The mic callback writes frames into a `GatedBuffer` under a lock; the
  player callback pulls samples from a `PlaybackQueue` under a lock. Neither
  callback ever touches the state machine directly.
- **Inference thread** — one persistent single-worker executor owns every
  MLX operation for the life of the process: engine imports, model loads,
  and every `pipeline.run_pipeline()` job (`Orchestrator._start_pipeline()`
  submits to it rather than spawning threads). This is a hard requirement,
  not a style choice: MLX streams are only usable on the thread that created
  them, and mlx-lm creates its generation stream at import time — loading on
  one thread and generating on another crashes with "There is no
  Stream(gpu, N) in current thread". Colocating everything on one thread
  also serializes engine access, so a barged-in response's final MLX call
  simply finishes before the next response's STT begins. Each pipeline job
  checks a `threading.Event` cancellation flag between and inside its calls
  so a barge-in or `ESC` can stop it mid-stream. One consequence: MLX Metal
  state created on this thread can SIGBUS during normal interpreter
  teardown, so `cmd_run` exits via `os._exit(0)` after releasing OS
  resources.
- **Generation counter** — `Orchestrator._gen` increments every time a new
  pipeline job is started. Each pipeline's `emit()` closure stamps
  outgoing events with the generation it was started under
  (`src/localvoice/app.py::_start_pipeline`); `Orchestrator.handle()` drops
  any `FIRST_AUDIO` / `RESPONSE_FINISHED` / `PIPELINE_ERROR` event whose
  `gen` doesn't match the current `self._gen`. This is what makes a stale
  pipeline thread — one that hasn't yet noticed its cancellation token is
  set — harmless if it emits after a new turn has already started: the old
  generation's event is simply ignored, never merged into the new turn.

## Barge-in and truncation

Interrupting a response (pressing the hotkey again, or `ESC`) must leave the
conversation history saying only what the user actually heard, not the full
text the LLM generated, so the next turn's context reflects reality.

The mechanism, in the order it actually runs:

1. **Clause tags.** As the pipeline speaks, `pipeline.py::speak()` calls
   `Transcript.add_clause(spoken)` for each sanitized clause *before*
   synthesizing it, getting back an integer tag. That tag is passed to
   `AudioPlayer.submit(chunk, tag)` for every audio chunk of that clause.
2. **Spoken-tags accounting.** `PlaybackQueue.pull()` (the audio callback,
   called ~every 20 ms by PortAudio) adds a chunk's tag to an internal
   `_spoken: set[int]` the moment that chunk is actually pulled for
   playback — not when it's queued. A clause that was synthesized but never
   reached the speaker (queued behind a flush) never appears in
   `_spoken`.
3. **Cancel sequence: truncate before flush.** Both `_CANCEL` and
   `_BARGE_IN` action tuples in `states.py` list `TRUNCATE_HISTORY` before
   `FLUSH_AUDIO`, and `Orchestrator._do()` executes actions in that order.
   `TRUNCATE_HISTORY` calls
   `self._transcript.truncate_commit(self._player.spoken_tags())`, reading
   the spoken-tag set *before* `FLUSH_AUDIO` calls `player.flush()`, which
   clears `_spoken` back to the empty set. Truncating after the flush would
   always see an empty set and discard the entire turn. (This ordering was
   fixed explicitly — see commit `c172326`, "truncate history before
   flushing player so barge-in keeps spoken clauses" — after an earlier spec
   draft had the actions in the wrong order; commit `7196c60` corrected the
   spec and plan to match.)
4. **`Transcript.truncate_commit()`.** Filters `_pending_clauses` down to
   the ones whose index is in `spoken_tags`, joins them with spaces, and — if
   any clauses were cut — appends a trailing `"..."` marker so the model's
   own history shows it was interrupted mid-thought:

   ```python
   def truncate_commit(self, spoken_tags: set[int]) -> None:
       if self._pending_user is None:
           return
       spoken = [c for i, c in enumerate(self._pending_clauses) if i in spoken_tags]
       cut = len(spoken) < len(self._pending_clauses)
       text = " ".join(spoken)
       self._finish((text + " ..." if text else "...") if cut else text)
   ```

   If nothing was spoken yet (interrupted during `PROCESSING`, before
   `FIRST_AUDIO`), the committed assistant turn is just `"..."`. The system
   prompt tells the model such a transcript means it was interrupted, so
   it can pick up gracefully rather than treating `"..."` as a real answer.
5. **Cancellation itself.** `CANCEL_PIPELINE` sets the `threading.Event`
   that the running pipeline thread polls between LLM deltas and inside the
   TTS chunk loop (`pipeline.py::run_pipeline` and the inner `speak()`
   closure). The pipeline thread notices on its next poll and returns; it
   never emits further events for that generation, and even if it did, the
   generation-counter check in `Orchestrator.handle()` would drop them.

An uninterrupted turn instead calls `Transcript.commit()`, which joins
*all* pending clauses with no truncation marker — `RESPONSE_FINISHED` in
`PROCESSING`/`SPEAKING` fires `COMMIT_TURN`, not `TRUNCATE_HISTORY`.

## Known limitations (v1)

1. **Audio device changes do not auto-restart streams.** Swapping input or
   output device mid-session does not re-open the capture/playback streams
   (spec §5 behavior deferred).
2. **Thinking mode has no spoken filler.** With `--think`, the assistant is
   silent until reasoning completes and only then begins speaking the reply
   (spec §3 behavior deferred).
3. **Barge-in queues behind one final MLX call.** After a barge-in, the
   superseded pipeline job may complete one final MLX call before it
   observes cancellation; because all inference shares one thread, the next
   response's STT waits those few milliseconds. (Replaced the earlier
   "engine-level overlap" limitation — the single inference thread now
   serializes engine access by construction.)
4. **KV prefix-reuse fallback on hybrid-attention models.** On hybrid
   attention models (the Qwen3.6 family) the KV prefix-reuse falls back to a
   full re-prefill each turn because their recurrent-state caches cannot be
   trimmed by mlx-lm.

## Related reading

- `docs/superpowers/specs/2026-07-03-localvoice-architecture-design.md` —
  the full design record: goals, decisions with rationale, error handling,
  testing strategy, repository layout, and phased roadmap.
- `docs/models.md` — swapping models per engine.
- `docs/latency.md` — the latency budget and how to measure it on your
  machine.
