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
| `state` | `state`: `"idle"` \| `"listening"` \| `"processing"` \| `"speaking"` | every orchestrator state transition |
| `user_text` | `text` | STT finishes transcribing a turn |
| `assistant_clause` | `text` | each sentence-ish clause the LLM produces, as it's chunked for TTS |
| `reasoning` | `text` | thinking-mode silent reasoning content (never spoken) |
| `turn_done` | `latency`: `{stt, ttft, first_clause, tts_first, total}`, all floats in seconds | once per successful turn, right after the first audio is queued to the player |
| `level` | `rms` (float, 4 decimals) | mic input level, throttled to ≤20 Hz, only while the capture buffer is armed (state `listening`) |
| `load_progress` | `engine`: `"stt"` \| `"llm"` \| `"tts"`, `phase`: `"start"` \| `"done"`, `seconds` (float, only on `"done"`; `null` on `"start"`) | initial load and every hot-apply reload |
| `download_progress` | `repo`, `pct` (float 0-100 or `null` if size unknown), `done` (bool) | reply stream to `download_model` |
| `models` | `installed`: list of `{"id", "path", "size_gb", "kind"}` | reply to `list_models` |
| `config_applied` | `config` (full merged config, post-apply), `reloaded`: list of engine names that were reloaded | reply to `set_config` |
| `error` | `message` | any recoverable failure — bad command, bad config value, engine load/reload failure, audio unavailable, inject-audio misuse |

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
| `tts.voice`, `tts.speed`, `llm.think`, `llm.max_tokens`, `llm.context_tokens`, `llm.system_prompt`, `keys.debounce_ms` | Applied immediately on the live config object. No reload, no audio restart. A turn already in flight keeps its captured values — instant changes take effect at the next turn boundary. | `reloaded: []` |
| `llm.model`, `llm.deep_model`, `stt.model`, `tts.model`, any `*.engine` | The named engine (`stt`/`llm`/`tts`) is reloaded on the single inference thread — same thread every load and pipeline run uses — emitting `load_progress(start)`/`load_progress(done)`. Conversation history is untouched; a turn already in flight keeps running on the old engine instance until it finishes (reloads queue behind it on the same thread). | `reloaded: ["llm"]` (etc., in `stt, llm, tts` order, deduped) |
| `audio.input_device`, `audio.output_device`, `audio.rebuffer_ms` | Player and capture streams are stopped and restarted. | `reloaded: []`, but the restart itself can emit `error` if the new device is unavailable |
| `keys.ptt`, `keys.stop` | Stored on the live config for terminal-mode (`localvoice run`) use; **the GUI owns the actual key tap in Swift** and applies these client-side. `serve` does not install any hotkey listener. | `reloaded: []` |

A `set_config` call can touch fields from more than one row at once (e.g.
`{"llm.think": true, "llm.model": "..."}`); each row's action runs for the
keys that match it, and `reloaded` lists every engine that got a reload out
of that one call.

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

## Related reading

- `docs/superpowers/specs/2026-07-03-localvoice-gui-design.md` — the full
  GUI design record: goals, architecture, the SwiftUI app plan, packaging,
  and phases.
- `docs/superpowers/plans/2026-07-03-localvoice-gui-phase-a.md` — the
  task-by-task implementation plan for the engine side of this protocol
  (`schema.py`, `overlay.py`, `engineset.py`, `modelstore.py`, `serve.py`).
- `docs/architecture.md` — the underlying engine's state machine, threading
  model, and barge-in mechanics, all of which `serve` reuses unchanged.
