# LocalVoice

A local push-to-talk voice assistant for Apple Silicon: hold right ⌘, talk,
release, and a local LLM answers out loud in under a second — no cloud, no
voice-activity detection, no waiting for you to stop talking. It comes as a
native macOS app and a terminal CLI, both driving the same local engine.

## How it works

Hold **right ⌘** and talk. Release it, and LocalVoice transcribes what you
said, sends it to a local LLM, and speaks the answer back through your
speakers as it's generated — the first words of the reply start playing
before the model has finished thinking of the rest.

- **Esc** stops the current response immediately (or discards the current
  recording if you're still holding the key).
- **Barge-in**: press right ⌘ again while the assistant is talking (or
  still thinking) to interrupt it and start a new recording in the same
  motion — no need to hit Esc first. The interrupted reply is cut off
  instantly, and the conversation history remembers only what you actually
  heard, not the rest of what the model would have said.

There's no wake word, no "are you still there," and no silence detection —
the push-to-talk key is the entire turn-taking mechanism, which is what
makes the interaction feel instant instead of laggy.

## Requirements

- An Apple Silicon Mac (M-series). LocalVoice is MLX-based and does not run
  the model inference path on Intel Macs.
- 32 GB minimum for the default model, 48 GB or more comfortable — the
  default configuration (Qwen3.6-35B-A3B 4-bit, plus Whisper and Kokoro
  alongside it).
- [`uv`](https://docs.astral.sh/uv/) for dependency management and running
  the project.
- `espeak-ng` (via Homebrew: `brew install espeak-ng`) — a fallback
  dependency for Kokoro's grapheme-to-phoneme conversion. See
  `docs/models.md` for details.

## Install

```bash
brew install espeak-ng   # TTS phonemizer fallback (see Requirements)
git clone <this repo>
cd LocalVoice
uv sync
uv run localvoice setup
uv run localvoice
```

`uv sync` installs Python dependencies into a project-local virtual
environment. `uv run localvoice setup` checks every model configured in
`localvoice.toml`, downloads any that aren't already present locally (with
sizes shown and a confirmation prompt), and leaves already-present ones
alone. `uv run localvoice` (equivalently, `uv run localvoice run`) starts
the assistant.

## macOS permissions

LocalVoice needs two permissions to see your key presses and hear your
voice:

1. Open **System Settings → Privacy & Security → Microphone**, and enable
   your terminal application (Terminal, iTerm2, etc.).
2. Open **System Settings → Privacy & Security → Input Monitoring**, and
   enable the same terminal application. This is what lets the global
   push-to-talk hotkey work even when your terminal isn't the focused
   window.
3. **Restart your terminal application** after granting these — macOS does
   not apply newly granted Input Monitoring or Microphone access to an
   already-running process.

If either permission is missing, `localvoice run` detects it at startup
(before loading any models) and exits with the exact System Settings path
to fix it, rather than failing silently or partway through a conversation.

## Configuration

LocalVoice reads `localvoice.toml` from the current directory by default
(override with `--config path/to/file.toml`). Every key below is also
editable live from the app's Settings pane, which includes a **Restore
defaults** button (it keeps your model assignments — see
[The app](#the-app)). Every key and its default:

| Section | Key | Default | Meaning |
|---|---|---|---|
| `[stt]` | `engine` | `"mlx_whisper"` | speech-to-text backend |
| `[stt]` | `model` | `"mlx-community/whisper-large-v3-turbo"` | HF repo id or local path |
| `[llm]` | `engine` | `"mlx_lm"` | LLM backend |
| `[llm]` | `model` | `"mlx-community/Qwen3.6-35B-A3B-4bit"` | HF repo id or local path used by default |
| `[llm]` | `deep_model` | `""` (empty) | HF repo id or local path used with `--deep`; empty disables `--deep` |
| `[llm]` | `think` | `false` | enable Qwen3.6 silent reasoning by default |
| `[llm]` | `max_tokens` | `1024` | generation cap per response |
| `[llm]` | `think_tokens` | `3072` | extra generation budget granted only in thinking mode |
| `[llm]` | `context_tokens` | `8192` | prompt-token budget; oldest exchanges are dropped first when history exceeds it |
| `[llm]` | `system_prompt` | (voice-tuned default prompt) | system message prepended to every conversation |
| `[tts]` | `engine` | `"kokoro_mlx"` | text-to-speech backend |
| `[tts]` | `model` | `"prince-canuma/Kokoro-82M"` | HF repo id or local path |
| `[tts]` | `voice` | `"af_heart"` | Kokoro voice preset |
| `[tts]` | `speed` | `1.0` | Kokoro speaking-rate multiplier |
| `[keys]` | `ptt` | `"cmd_r"` | push-to-talk key (any `pynput` key name; modifier-only keys recommended) |
| `[keys]` | `stop` | `"esc"` | stop/cancel key |
| `[keys]` | `debounce_ms` | `120` | minimum hold time (ms) before a release is treated as a real recording |
| `[audio]` | `input_device` | `""` (empty) | microphone device name; empty uses the system default |
| `[audio]` | `output_device` | `""` (empty) | speaker device name; empty uses the system default |
| `[audio]` | `rebuffer_ms` | `300` | anti-stutter gate: hold response audio until this much is queued after a mid-response stall |

See `docs/models.md` for what to put in each `model` field — any HF repo id
works, and so does any local MLX-format model folder (for example, anything
already downloaded under `~/.lmstudio/models`).

### Local overlay: `localvoice.local.toml`

Drop a `localvoice.local.toml` next to `localvoice.toml` to override any
keys without touching the committed config or your shell environment.
`load_config()` reads `localvoice.toml` first, then merges
`localvoice.local.toml` on top section-by-section if it exists — you only
need to list the keys you're overriding. This file is listed in
`.gitignore`, so it's the right place for machine-specific settings like a
local model path that shouldn't be committed:

```toml
[llm]
deep_model = "/Users/you/.lmstudio/models/lmstudio-community/Qwen3.6-27B-MLX-6bit"
```

### `--deep` and `--think`

- `--deep` swaps `[llm].model` for `[llm].deep_model` for the whole session
  — use it when you want the smartest available answer and don't mind an
  extra second or so of latency. It requires `deep_model` to be set (in
  `localvoice.toml` or the local overlay); running `--deep` with an empty
  `deep_model` is a startup config error, not a silent no-op.
- `--think` enables Qwen3.6's reasoning mode for the session even if
  `[llm].think` is `false` in config (the two are OR'd together). Thinking
  is never spoken aloud; it is printed to the terminal as a `reasoning:`
  line, and the spoken reply starts only after reasoning finishes, so a
  long think means a longer stretch of silence before the assistant
  speaks. Reasoning draws on `think_tokens` of extra generation budget.
- `--model` overrides `[llm].model` for the session with any HF repo id or
  local MLX model path, beating both the config file and `--deep`.

`--deep` and `--model` apply to both `localvoice run` and `localvoice bench`; `--think` is a `run`-only flag.

## Privacy

The microphone input stream is opened once, when LocalVoice starts, and
stays open for the whole session — but audio is only ever *retained* while
the push-to-talk key is physically held down. Frames are written into a
gated buffer that discards everything unless it has been explicitly armed
by a key-down, and the buffer is cleared the instant the key comes up (or on
Esc, or on barge-in). There's no rolling pre-buffer and no background
recording between turns.

Nothing leaves your machine. Transcription, generation, and speech synthesis
all run locally via MLX; there is no network call in the conversation path.

## Latency

Voice-to-voice is the metric that matters for a push-to-talk assistant: the
time from releasing the key to hearing the first word of the reply. Measured
on an M5 Pro (64 GB): **0.82 s** with the default model
(Qwen3.6-35B-A3B 4-bit) and **1.43 s** in `--deep` mode (Qwen3.6-27B
6-bit). Target budgets per pipeline stage, the full measured table, and how
to measure on your own hardware are in [`docs/latency.md`](docs/latency.md).

## Architecture

LocalVoice is a single Python process — plain threads and queues, not
asyncio, because every hot-path call (MLX inference, PortAudio, the Quartz
key tap) blocks anyway. A pure state machine drives the whole interaction
off an event queue, with the hotkey listener, audio callbacks, and each
response's pipeline running on their own threads.

The condensed architecture writeup — pipeline diagram, the full state
transition table, the threading model, and how barge-in truncation keeps
conversation history honest — is in
[`docs/architecture.md`](docs/architecture.md). The full design record,
including all locked decisions and their rationale, error handling, testing
strategy, and the phased roadmap, is
[`docs/superpowers/specs/2026-07-03-localvoice-architecture-design.md`](docs/superpowers/specs/2026-07-03-localvoice-architecture-design.md).

## The app

`LocalVoice.app` is a native SwiftUI app (macOS 15+) that drives the same
engine as the CLI, over the line-delimited JSON protocol `localvoice serve`
speaks on stdio — a Talk view with the live transcript, reasoning, mic
level, and per-turn latency; a Settings pane generated from the engine's
own config schema (every key editable live, plus a **Restore defaults**
button that resets everything except your model assignments — the shipped
default model is a Hugging Face repo id, and resetting to it would trigger
a multi-GB download, so model choices stay put and are changed in the
Models tab instead); a Models manager for installed/downloadable models; a
Setup pane for permissions and engine status; and the same global right-⌘
push-to-talk hotkey. It lives in `app/LocalVoice/` as an Xcode project.

Build and run it:

```bash
brew install xcodegen
cd app/LocalVoice
xcodegen generate
open LocalVoice.xcodeproj   # then Run, or:
xcodebuild -scheme LocalVoice -destination 'platform=macOS' build
```

`LocalVoice.xcodeproj` is generated (via [XcodeGen](https://github.com/yonaskolb/XcodeGen)
from `app/LocalVoice/project.yml`) and gitignored, not committed — run
`xcodegen generate` once after cloning and again any time `project.yml`
changes.

**Dev-checkout mode**: in v1, the app always spawns `uv run localvoice
serve` from a repo checkout rather than bundling the engine — the same `uv
sync`'d environment the CLI uses. By default it resolves that checkout
relative to its own build location, which finds this repo automatically for
anyone building from a clone the normal way; the Setup pane lets you
override the path if you've moved the checkout or want to point the app at
a different clone. A packaged `.app` that bundles the engine and needs no
repo checkout at all is planned for phase C.

**Permissions**: the app needs the same Microphone and Input Monitoring
grants the CLI does (see [macOS permissions](#macos-permissions) above),
except the system prompts and the entries you enable in System Settings
will say **LocalVoice** (the app) instead of your terminal — grant those,
then relaunch the app the same way you'd relaunch a terminal after granting
them to it.

**If the app sits at "Connecting…" forever**, the engine process is failing
to start — almost always a stale Python environment rather than anything in
the app. Refresh it from the repo root and relaunch:

```bash
uv self update && rm -rf .venv && uv sync
```

(The first response after a rebuilt environment can take an extra minute
while TTS re-fetches a small spaCy model.)

## Development

Both halves have their own test suite; CI runs both on every push and pull
request.

```bash
uv run pytest -m "not slow"        # engine fast suite (~5 s)
uv run pytest                      # + slow end-to-end tests (loads real models)

cd app/LocalVoice && xcodegen generate && \
  xcodebuild -project LocalVoice.xcodeproj -scheme LocalVoice \
  -destination 'platform=macOS' test        # app suite
```

The JSON-lines protocol between the app and `localvoice serve` — every
event, every command, and the hot-apply semantics of each config key — is
specified in [`docs/gui.md`](docs/gui.md), which also describes the app's
client architecture.

## Roadmap

**Shipped**: the full push-to-talk engine and CLI (hold, talk, release,
answer, barge-in, Esc, earcons, configuration, the `setup` command), and
the native macOS app (live conversation view, generated settings editor
with restore-defaults, model manager, permission onboarding, global hotkey
via the app's own event tap) — with tests, docs, and CI for both.

**Next — packaging (phase C)**: a self-contained `LocalVoice.app` that
bundles the engine (PyInstaller) so no repo checkout or `uv` is needed,
distributed via GitHub Releases.

**Later**: streaming STT during the hold (transcribing as you talk rather
than after you release, for an estimated further ~0.3 s latency win), a
`parakeet-mlx` STT option, an OpenAI-compatible LLM backend so LM Studio or
Ollama can serve as the LLM engine, a spoken thinking-mode toggle, session
transcript export, a menu-bar shell, tool calling, additional TTS backends
(Chatterbox-Turbo, Qwen3-TTS), and an experiment in skipping STT entirely
via a model with native audio input (Gemma 4 E4B).

## Acknowledgments

LocalVoice is built entirely on other people's excellent local-inference
work:

- [MLX](https://github.com/ml-explore/mlx) and
  [MLX LM](https://github.com/ml-explore/mlx-lm) — Apple's array framework
  for Apple Silicon, and the LLM inference library on top of it.
- [MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper)
  and [OpenAI Whisper](https://github.com/openai/whisper) — the speech
  recognition model and its MLX port.
- [mlx-audio](https://github.com/Blaizzy/mlx-audio) — the MLX text-to-speech
  library used to run Kokoro.
- [Qwen](https://github.com/QwenLM/Qwen) — the LLM family LocalVoice ships
  as its default and deep-mode models.
- [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) — the small,
  Apache-2.0 text-to-speech model LocalVoice speaks with.

## License

MIT. See [`LICENSE`](LICENSE).
