# GUI protocol

`localvoice serve` runs the same engine (orchestrator, pipeline, audio
capture/playback, MLX engines) as `localvoice run`, but instead of printing
status lines and reading a physical hotkey it speaks a line-delimited JSON
protocol over stdio. This is what `LocalVoice.app` (phase B, SwiftUI) talks
to, and it's equally usable from a shell or a test harness — the SwiftUI
app has no special access the protocol doesn't expose.

This document is the protocol reference. For the wider app architecture and
the decisions behind it, see
`docs/superpowers/specs/2026-07-03-localvoice-gui-design.md`; for the v1
terminal engine `docs/architecture.md` is the equivalent companion.

## Transport

- One JSON object per line, newline-delimited, UTF-8.
- **stdout carries protocol only.** `Serve.run()` snapshots the real stdout
  before doing anything else and then sets `sys.stdout = sys.stderr`, so any
  library `print()` (tqdm, huggingface_hub, etc.) can never corrupt a
  protocol line. Read stdout with one JSON decode per line and nothing else.
- **stderr carries logs.** Anything printed by dependencies, plus Python
  tracebacks, ends up here. Useful for debugging a stuck engine; never
  parsed by a client.
- **stdin carries commands**, one JSON object per line. Malformed JSON gets
  an `error` event back (`{"event": "error", "message": "bad json: ..."}`)
  and the loop keeps reading — a bad line never kills the process.
- The process exits via `os._exit(0)` after a `shutdown` command (or EOF on
  stdin closes the read loop without an explicit exit — send `shutdown`
  from a client that wants a clean, timely stop). There is no other exit
  path; killing the process (SIGTERM/SIGKILL) is the fallback for a client
  that gets no response.

## Startup sequence

```
ready  ──▶  load_progress (engine=stt, phase=start)
            load_progress (engine=stt, phase=done, seconds=…)
            load_progress (engine=llm, phase=start)
            load_progress (engine=llm, phase=done, seconds=…)
            load_progress (engine=tts, phase=start)
            load_progress (engine=tts, phase=done, seconds=…)
       ──▶  engines_ready
```

1. `ready` is emitted immediately — before any model touches disk — carrying
   the protocol version, the fully merged config, and the derived settings
   schema (§ below). A client can render the whole Settings pane from this
   one message.
2. Engine loads then run **asynchronously** on the process's single MLX
   inference thread (the same thread every load, reload, inference, and
   voice-preview job runs on — see `docs/architecture.md`'s threading
   model), each firing a `load_progress` pair.
3. `engines_ready` follows the last load. If a load fails (bad path, OOM,
   etc.) serve emits an `error` event instead and `engines_ready` never
   fires for that boot; a later `set_config` that fixes the offending model
   path triggers a fresh reload for just that engine.
4. **Commands that touch the orchestrator (`ptt_down`, `ptt_up`) are
   rejected with an `error` event until `engines_ready` has been seen.**
   `esc`, `set_config`, `list_models`, and `shutdown` do not require
   `engines_ready`.

## Events (engine to app)

Every event object has an `"event"` key naming it. Payload keys below are
in addition to `"event"`.

| event | payload | fires when |
|---|---|---|
| `ready` | `version` (int, `1`), `config` (full merged config, nested dict), `schema` (list of descriptors, see below) | once, at serve startup |
| `state` | `state`: `"idle"` \| `"listening"` \| `"processing"` \| `"speaking"` | every orchestrator state transition (see note below — level-triggered, not edge-triggered) |
| `user_text` | `text` | STT finishes transcribing a turn |
| `assistant_clause` | `text` | each sentence-ish clause the LLM produces, as it's chunked for TTS |
| `reasoning` | `text` | thinking-mode silent reasoning content (never spoken) |
| `tool_call` | `name` (str), `summary` (str) | pipeline tool rounds |
| `tool_result` | `name` (str), `ok` (bool), `summary` (str) | pipeline tool rounds |
| `turn_done` | `latency`: `{stt, ttft, first_clause, tts_first, total}`, all floats in seconds | once per successful turn, when generation and synthesis of the *whole* reply completes — not at first audio. Every value in `latency` is a duration frozen at or before first-audio (see the definitions below), but the event itself is emitted only after the LLM stream and clause synthesis loop finish (`pipeline.py::run_pipeline`, right after `player.mark_end()`). On a long answer, `turn_done` can arrive noticeably later than the moment the user actually started hearing the reply. |
| `level` | `rms` (float, 4 decimals) | mic input level, throttled to ≤20 Hz, only while the capture buffer is armed (state `listening`) |
| `load_progress` | `engine`: `"stt"` \| `"llm"` \| `"tts"`, `phase`: `"start"` \| `"done"`, `seconds` (float, only on `"done"`; `null` on `"start"`) | initial load and every hot-apply reload |
| `download_progress` | `repo`, `pct` (float 0-100 or `null` if size unknown), `done` (bool) | reply stream to `download_model` |
| `models` | `installed`: list of `{"id", "path", "size_gb", "kind"}` | reply to `list_models` |
| `config_applied` | `config` (full merged config, post-apply), `reloaded`: list of engine names that were reloaded | reply to `set_config` |
| `error` | `message` | any recoverable failure — bad command, bad config value, engine load/reload failure, audio unavailable, inject-audio misuse |

`turn_done.latency` keys, all seconds, all measured against pipeline-internal
timestamps captured in `pipeline.py::run_pipeline`:

- **`stt`** — transcribe duration: STT start to STT finish.
- **`ttft`** ("time to first token") — STT-done to the first LLM stream delta.
- **`first_clause`** — first LLM delta to the first speakable clause (i.e.
  the first chunk `ClauseChunker` emits that survives markup-stripping).
- **`tts_first`** — first speakable clause to the first audio chunk actually
  submitted to the player.
- **`total`** — turn start (capture released) to that same first-audio-submit
  moment.

**`state` is level-triggered, not edge-triggered.** The orchestrator emits
`state` from `_show_state()` on every `handle()` call, including
transitions whose `(state, event)` pair is a no-op in the transition table
(same state in, same state out — see `docs/architecture.md`'s state
table). A client should treat repeated `state` events carrying the same
`state` value as idempotent — safe to re-render, never a signal that
something changed — rather than assuming every `state` message represents
a genuine transition.

Example — `ready` (config/schema truncated for brevity; every `Config`
field and its schema descriptor is present in the real message):

```json
{"event": "ready", "version": 1,
 "config": {"stt": {"engine": "mlx_whisper", "model": "mlx-community/whisper-large-v3-turbo"},
            "llm": {"engine": "mlx_lm", "model": "mlx-community/Qwen3.6-35B-A3B-4bit", "think": false, "max_tokens": 1024, "...": "..."},
            "tts": {"engine": "kokoro_mlx", "model": "prince-canuma/Kokoro-82M", "voice": "af_heart", "speed": 1.0},
            "keys": {"ptt": "cmd_r", "stop": "esc", "debounce_ms": 120},
            "audio": {"input_device": "", "output_device": "", "rebuffer_ms": 300}},
 "schema": [{"key": "llm.think", "type": "bool", "value": false, "default": false,
             "section": "Language model", "label": "Thinking mode",
             "help": "Silent reasoning; shown, never spoken.", "widget": "toggle"},
            "..."]}
```

Example — `state`:

```json
{"event": "state", "state": "listening"}
```

Example — `user_text` / `assistant_clause` / `reasoning`:

```json
{"event": "user_text", "text": "What's the capital of Australia?"}
{"event": "assistant_clause", "text": "Canberra is the capital of Australia."}
{"event": "reasoning", "text": "The user is asking about Australia's capital, which is commonly confused with Sydney..."}
```

Example — `tool_call` / `tool_result` (normally one pair per tool round;
`tool_call` fires as soon as the model's call is parsed, `tool_result` after
the tool finishes executing, both strictly before that turn's `turn_done`).
A cancellation (Esc or barge-in) between the two can leave a `tool_call` with
no matching `tool_result` for that round — a client should not assume every
`tool_call` is always followed by a `tool_result`:

```json
{"event": "tool_call", "name": "web_search", "summary": "Calling web_search"}
{"event": "tool_result", "name": "web_search", "ok": true, "summary": "Found 5 results for: boston weather"}
```

Example — `turn_done`:

```json
{"event": "turn_done",
 "latency": {"stt": 0.18, "ttft": 0.24, "first_clause": 0.31, "tts_first": 0.09, "total": 0.82}}
```

Example — `level` (throttled, only while listening):

```json
{"event": "level", "rms": 0.0421}
```

Example — `load_progress`:

```json
{"event": "load_progress", "engine": "llm", "phase": "start", "seconds": null}
{"event": "load_progress", "engine": "llm", "phase": "done", "seconds": 3.42}
```

Example — `download_progress`:

```json
{"event": "download_progress", "repo": "mlx-community/whisper-tiny", "pct": 47.3, "done": false}
{"event": "download_progress", "repo": "mlx-community/whisper-tiny", "pct": 100.0, "done": true}
```

Example — `models`:

```json
{"event": "models",
 "installed": [{"id": "mlx-community/Qwen3.6-35B-A3B-4bit",
                "path": "/Users/you/.lmstudio/models/mlx-community/Qwen3.6-35B-A3B-4bit",
                "size_gb": 19.8, "kind": "mlx"}]}
```

Example — `config_applied`:

```json
{"event": "config_applied",
 "config": {"llm": {"think": true, "...": "..."}, "...": "..."},
 "reloaded": []}
```

Example — `error`:

```json
{"event": "error", "message": "engines still loading"}
```

## Commands (app to engine)

Every command object has a `"cmd"` key naming it.

| cmd | payload | behavior |
|---|---|---|
| `ptt_down` | — | posts `PTT_DOWN` into the orchestrator queue (same event the physical hotkey posts). Rejected with `error` before `engines_ready`. |
| `ptt_up` | `held_ms` (int, default `500` if omitted) | posts `PTT_UP` with the hold duration the client measured. Rejected with `error` before `engines_ready`. |
| `esc` | — | posts `ESC` (cancel/stop). Always accepted. |
| `set_config` | `changes`: `{"dotted.key": value, ...}` | coerce every value → write `localvoice.local.toml` → hot-apply (§ below) → reply `config_applied`. A bad key or bad value yields `error` and no write. |
| `reset_config` | `keep`: `["dotted.key", ...]` (optional, default `[]`) | removes every key currently set in the overlay **except** those in `keep`, rewrites `localvoice.local.toml` atomically (pruning now-empty sections; the file stays present even if empty) → hot-applies the resulting diff through the same path as `set_config` (§ below) → reply `config_applied` with the full merged config and the `reloaded` list. Only fields whose merged value actually changed are applied, so a cleared engine-bound key reloads its engine while a cleared instant key just reverts. Unknown `keep` keys are ignored; an unwritable overlay yields `error`. |
| `list_models` | — | scans `~/.lmstudio/models/*/*` plus the HF cache; replies `models`. |
| `download_model` | `repo` (HF repo id) | runs `snapshot_download` on a worker thread; streams `download_progress`. |
| `preview_voice` | `voice` (Kokoro voice id) | synthesizes a fixed sample line with the given voice through the real player, then restores the configured voice. Requires `engines_ready`. |
| `inject_audio` | `path` (WAV file path) | feeds the WAV as if it were a PTT capture — **only when serve was started with `--allow-inject`**; otherwise `error` (`"inject_audio not allowed"`). Test/automation hook, not for production GUI use. Internally waits (up to 2 s) for the capture buffer to actually arm before writing the audio in, so the frames can't race the orchestrator's `PTT_DOWN` handling; on timeout it emits `error` (`"inject_audio: capture never armed"`) instead of silently dropping the audio. |
| `shutdown` | — | stops capture/playback, joins the orchestrator loop, then `os._exit(0)`. The command loop returns after handling this, so no more events follow. |

Example — `ptt_down` / `ptt_up` / `esc`:

```json
{"cmd": "ptt_down"}
{"cmd": "ptt_up", "held_ms": 480}
{"cmd": "esc"}
```

Example — `set_config`:

```json
{"cmd": "set_config", "changes": {"llm.think": true, "tts.speed": 1.2}}
```

Example — `reset_config` (clears the overlay but keeps the four model assignments):

```json
{"cmd": "reset_config", "keep": ["llm.model", "llm.deep_model", "stt.model", "tts.model"]}
```

Example — `list_models`:

```json
{"cmd": "list_models"}
```

Example — `download_model`:

```json
{"cmd": "download_model", "repo": "mlx-community/whisper-tiny"}
```

Example — `preview_voice`:

```json
{"cmd": "preview_voice", "voice": "af_bella"}
```

Example — `inject_audio` (requires `--allow-inject`):

```json
{"cmd": "inject_audio", "path": "/tmp/question.wav"}
```

Example — `shutdown`:

```json
{"cmd": "shutdown"}
```

## Derived settings schema

`ready.schema` is produced by `src/localvoice/schema.py::build_schema` — one
descriptor per field of every `Config` section (`stt`, `llm`, `tts`, `keys`,
`audio`), so any config field added later shows up in a generic client with
no protocol change. Each descriptor:

```json
{"key": "llm.think", "type": "bool", "value": false, "default": false,
 "section": "Language model", "label": "Thinking mode",
 "help": "Silent reasoning; shown, never spoken.", "widget": "toggle"}
```

- `type` comes straight from the Python annotation: `bool`, `int`, `float`,
  `str`.
- `widget` defaults from `type` (`bool→toggle`, `int/float→number`,
  `str→text`) and is upgraded for specific keys: `*.model`/`llm.deep_model`
  → `model_picker`, `tts.voice` → `voice_picker`, `keys.ptt`/`keys.stop` →
  `key_capture`, `audio.input_device`/`output_device` → `device_picker`,
  `tts.speed`/`audio.rebuffer_ms` → `slider` (with `minimum`/`maximum`/
  `step`).
- `value` reflects the live config at the moment `ready` (or
  `config_applied`) was emitted; `default` is the dataclass field default,
  always.
- `set_config`'s `changes` values are coerced by
  `src/localvoice/schema.py::coerce` against this same schema — so
  `"true"`, `true`, `"2048"`, and `2048` are all accepted where the field
  type calls for them; an unknown key or an uncoercible value yields
  `error` with the key named in the message, and nothing is written.

## Hot-apply semantics

`set_config` always writes `localvoice.local.toml` first (atomically), then
applies the change to the running process per this table (spec §5):

| Fields | Action | Reflected in `config_applied` |
|---|---|---|
| `llm.think`, `llm.max_tokens`, `llm.context_tokens`, `llm.system_prompt`, `keys.debounce_ms` | Applied immediately on the live config object. No reload, no audio restart. A turn already in flight keeps its captured values — `llm.stream()` reads these once per turn, so an instant change here takes effect starting the *next turn*, not the one in progress. | `reloaded: []` |
| `tts.voice`, `tts.speed` | Applied immediately on the live config object. No reload, no audio restart. Unlike the `llm.*` fields above, `TTSEngine.synthesize()` reads `voice`/`speed` fresh on every call, and the pipeline calls `synthesize()` once per clause — so a change here can take effect mid-turn, at the *next clause* of a response already in flight, not just the next turn. | `reloaded: []` |
| `llm.model`, `llm.deep_model`, `stt.model`, `tts.model`, any `*.engine` (`llm.engine`: `mlx_lm` \| `mlx_vlm`) | The named engine (`stt`/`llm`/`tts`) is reloaded on the single inference thread — same thread every load and pipeline run uses — emitting `load_progress(start)`/`load_progress(done)`. Conversation history is untouched; a turn already in flight keeps running on the old engine instance until it finishes (reloads queue behind it on the same thread). | `reloaded: ["llm"]` (etc., in `stt, llm, tts` order, deduped) |
| `audio.input_device`, `audio.output_device`, `audio.rebuffer_ms` | Player and capture streams are stopped and restarted. | `reloaded: []`, but the restart itself can emit `error` if the new device is unavailable |
| `keys.ptt`, `keys.stop` | Stored on the live config for terminal-mode (`localvoice run`) use; **the GUI owns the actual key tap in Swift** and applies these client-side. `serve` does not install any hotkey listener. *Caveat: the v1 app hard-codes right-⌘/Esc in `HotkeyMonitor` and does not yet read these — configurable keys are a later phase.* | `reloaded: []` |
| `tools.enabled`, `tools.web_search`, `tools.screenshot`, `tools.max_rounds`, `tools.search_results`, `tools.page_char_cap` | Applied immediately on the live config object. No reload, no audio restart. `tools.screenshot` toggles the same way, but `look_at_screen` is only ever offered to the model when the loaded `llm` engine can consume an image (`llm.engine = "mlx_vlm"` today) — on a text-only engine the bit is stored and reflected in `config_applied` like any other value, yet the tool never appears in a turn's `tool_call`. | `reloaded: []` |

A `set_config` call can touch fields from more than one row at once (e.g.
`{"llm.think": true, "llm.model": "..."}`); each row's action runs for the
keys that match it, and `reloaded` lists every engine that got a reload out
of that one call.

`reset_config` runs this same table: after clearing the overlay it diffs the
old and new merged config and hot-applies only the keys that changed, so a
model assignment left untouched by `keep` costs no reload while a cleared
engine-bound key reloads exactly as an equivalent `set_config` would.

`[tools]` keys hot-apply per turn, no reload: the tool registry offered to
the model is rebuilt from the live config at the start of every turn (not
cached at engine-load time), so a `set_config` touching any `tools.*` field
takes effect starting the very next turn with no engine reload and no
audio restart. `tool_call`/`tool_result` events only occur when the active
model's chat template actually supports tool calling — a template without
tool-call rendering is never offered tools regardless of `tools.enabled`,
so a client should not expect these two events from every model.

A `look_at_screen` round is a special case of the same tool-round
machinery: its result carries the captured screenshot's temp-file path,
but only engine-internally (`ToolResult.image_path`, never a `tool_result`
payload field over the wire) — the *next* LLM turn receives it as an image
via the LLM engine's own image parameter, while the `role: "tool"` message
content stays the same sanitized JSON string as any other tool result, so
the path itself never appears in the prompt or on the wire. Tools are
withheld on that image turn (`tools = None`), so the model is steered to
answer about what it saw rather than chain another call — but exactly as on
the final round of `max_rounds`, a tool marker the model emits anyway is
still parsed and executed, so a client may occasionally observe a second
`tool_call`/`tool_result` pair; the loop stays bounded by `max_rounds` and
every captured PNG is still cleaned up. The temporary PNG is deleted the moment that turn ends —
normally, cancelled, or errored — so nothing about it outlives the turn
that produced it.

## Driving it manually

Start the engine with automation enabled:

```console
$ uv run localvoice serve --allow-inject
```

(Drop `--allow-inject` for anything that isn't a test harness — it's a
deliberate hole that lets a client feed a WAV file straight into the
capture buffer, bypassing the microphone and the hotkey entirely.)

Every line below is one command written to the process's stdin, one per
line, each followed by a newline:

```jsonl
{"cmd": "ptt_down"}
{"cmd": "ptt_up", "held_ms": 600}
{"cmd": "set_config", "changes": {"llm.think": true}}
{"cmd": "list_models"}
{"cmd": "inject_audio", "path": "/tmp/question.wav"}
{"cmd": "shutdown"}
```

Watch stdout for the corresponding `ready` → `load_progress`… →
`engines_ready` boot sequence, then `state`/`user_text`/`assistant_clause`/
`turn_done` for each turn.

`inject_audio` (like `ptt_down`/`ptt_up`) is rejected with `error`
(`"engines still loading"`) until `engines_ready` has actually been seen —
model loads take real time, so piping commands in immediately at process
start (e.g. via a heredoc) races the boot sequence. A client that wants to
inject as soon as possible should read stdout, wait for `{"event":
"engines_ready"}`, and only then write the `inject_audio` line —
`tests/test_smoke_slow.py::test_serve_protocol_end_to_end` is the automated
version of exactly that: it spawns `localvoice serve --config ... --allow-inject`,
reads events until it sees `engines_ready`, writes `inject_audio` at that
point, and asserts `user_text` → `assistant_clause` → `turn_done` all
arrive before sending `shutdown`.

## Client architecture (LocalVoice.app)

`LocalVoice.app` (`app/LocalVoice/`, a SwiftUI app generated via
[XcodeGen](https://github.com/yonaskolb/XcodeGen) from `project.yml`) is one
concrete client of the protocol above — everything it can do, it does
through the commands and events already documented here; it has no access
the protocol doesn't expose. Three types carry the whole client-side
contract:

- **`EngineClient`** (`app/LocalVoice/Sources/Engine/EngineClient.swift`) —
  an `actor` that owns the engine subprocess. It spawns `serve` per a
  `LaunchMode` (`app/LocalVoice/Sources/Engine/LaunchMode.swift` —
  `.devCheckout(root)` runs `uv run localvoice serve` from a repo
  checkout, the only mode the app itself uses in phase B; `.custom` is
  the test/fixture and future phase-C bundled-binary mode), decodes its
  newline-delimited stdout into `EngineEvent`s (this doc's event table),
  forwards stderr to the unified system logger, and respawns on an
  unexpected exit with capped exponential backoff (1, 2, 4, 8, 8, … seconds,
  reset to 0 on the next `ready`). Callers see a single
  `AsyncStream<ClientEvent>` (`events`) that layers three lifecycle cases
  (`.spawned`, `.exited(code:)`, `.respawning(attempt:delaySeconds:)`) over
  every decoded `.engine(EngineEvent)` — the reason the type is
  `ClientEvent` and not the engine's own `EngineEvent` verbatim is that a
  respawn banner or a "reconnecting" indicator needs to render from
  subprocess-level facts (died, restarting, gave up) that have no
  equivalent event on the wire protocol itself. `send(_:)` writes one
  `EngineCommand` (this doc's command table) as an encoded line to the
  child's stdin; `stop()` sends `shutdown`, waits up to 3 s, then SIGKILLs.
- **`AppState`** (`app/LocalVoice/Sources/State/AppState.swift`) — an
  `@Observable @MainActor` class that is the single source of truth for
  every view. Its `reduce(_ event: ClientEvent)` is a pure event-in,
  state-out step: one call per event, no async machinery of its own, which
  is what makes `AppStateTests` able to drive whole event sequences through
  it and assert on the resulting fields without touching a real
  `EngineClient`. There is exactly one long-lived consumer of
  `EngineClient.events` per app process — an unstructured `Task` started
  from `LocalVoiceApp.init()` (`app/LocalVoice/Sources/LocalVoiceApp.swift`)
  that loops `for await event in client.events { appState.reduce(event) }`
  for the life of the process, independent of any view's lifetime (closing
  LocalVoice's last window does not terminate the app or cancel this loop —
  standard SwiftUI `WindowGroup` behavior, and the reason this is a
  detached `Task` rather than a view's `.task`).
- **`HotkeyMonitor`** (`app/LocalVoice/Sources/Engine/HotkeyMonitor.swift`)
  — a `@MainActor` class wrapping a listen-only `CGEventTap` on
  `flagsChanged` (right-⌘) and `keyDown` (Esc), scoped to the current login
  session. Per this doc's "the GUI owns the actual key tap in Swift" note
  above, `serve` installs no hotkey listener of its own — `HotkeyMonitor`'s
  three callbacks (`onPttDown`, `onPttUp(heldMs:)`, `onEsc`) are wired in
  `LocalVoiceApp.init()` to send exactly the same `ptt_down` / `ptt_up` /
  `esc` commands the on-screen hold-to-talk button
  (`app/LocalVoice/Sources/Views/TalkView.swift`) sends, so the engine
  cannot distinguish a physical key press from a button click. The
  press/release/esc *decision* itself is a pure free function
  (`decide(type:keycode:commandBit:pttCurrentlyDown:)`, same file) kept
  separate from the tap callback specifically so it can be unit-tested
  (`HotkeyLogicTests`) without a live, granted tap — `CGEvent.tapCreate`
  returns `nil` silently when Input Monitoring is denied, which is not
  reproducible on demand or in CI.

Together, the reduction contract is: **events flow one way, down a single
pipe** — `serve`'s stdout → `EngineClient` (decode) → `ClientEvent` →
`AppState.reduce` (the only place protocol/lifecycle events turn into UI
state) → SwiftUI's own diffing re-renders whatever view reads the changed
`@Observable` property. Commands flow the other way, from either input
source (button or global hotkey) through the same `EngineClient.send(_:)`,
never through `AppState` — `AppState` only ever reduces, it never issues
commands itself.

The four views (`app/LocalVoice/Sources/Views/`: `TalkView`, `ModelsView`,
`SettingsView`, `SetupView`, assembled by `MainWindow`) are thin readers of
`AppState` plus direct `EngineClient.send(_:)` callers; none of them decode
protocol JSON or manage the subprocess themselves. `SettingsView`'s derived
form is built entirely from `ready.schema` (this doc's derived-settings
section) with no per-field Swift code — a new `Config` field on the Python
side needs no client change to appear in the Settings pane, only a new
`schema.py` descriptor.

## Related reading

- `docs/superpowers/specs/2026-07-03-localvoice-gui-design.md` — the full
  GUI design record: goals, architecture, the SwiftUI app plan, packaging,
  and phases.
- `docs/superpowers/plans/2026-07-03-localvoice-gui-phase-a.md` — the
  task-by-task implementation plan for the engine side of this protocol
  (`schema.py`, `overlay.py`, `engineset.py`, `modelstore.py`, `serve.py`).
- `docs/superpowers/plans/2026-07-03-localvoice-gui-phase-b.md` — the
  task-by-task implementation plan for `LocalVoice.app` itself: the Xcode
  project, `EngineClient`, `AppState`, the four views, `HotkeyMonitor`, and
  CI.
- `docs/architecture.md` — the underlying engine's state machine, threading
  model, and barge-in mechanics, all of which `serve` reuses unchanged.
