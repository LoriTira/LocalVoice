# LocalVoice

A local push-to-talk voice assistant for Apple Silicon: hold right ⌘, talk,
release, and a local LLM answers out loud in under a second — no cloud, no
voice-activity detection, no waiting for you to stop talking.

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
- 16 GB+ unified memory for small/lightweight model configurations; 32 GB+
  recommended for the default configuration (Qwen3.6-35B-A3B 4-bit, plus
  Whisper and Kokoro alongside it).
- [`uv`](https://docs.astral.sh/uv/) for dependency management and running
  the project.
- `espeak-ng` (via Homebrew: `brew install espeak-ng`) — a fallback
  dependency for Kokoro's grapheme-to-phoneme conversion. See
  `docs/models.md` for details.

## Install

```bash
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
(override with `--config path/to/file.toml`). Every key and its default:

| Section | Key | Default | Meaning |
|---|---|---|---|
| `[stt]` | `engine` | `"mlx_whisper"` | speech-to-text backend |
| `[stt]` | `model` | `"mlx-community/whisper-large-v3-turbo"` | HF repo id or local path |
| `[llm]` | `engine` | `"mlx_lm"` | LLM backend |
| `[llm]` | `model` | `"mlx-community/Qwen3.6-35B-A3B-4bit"` | HF repo id or local path used by default |
| `[llm]` | `deep_model` | `""` (empty) | HF repo id or local path used with `--deep`; empty disables `--deep` |
| `[llm]` | `think` | `false` | enable Qwen3.6 silent reasoning by default |
| `[llm]` | `max_tokens` | `1024` | generation cap per response |
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
  runs silently — it's never spoken aloud — and if it runs long, the
  assistant speaks a short filler line while it keeps thinking, rather than
  leaving you with silence.

Both flags apply to `localvoice run` and `localvoice bench`.

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
time from releasing the key to hearing the first word of the reply. Target
budgets per pipeline stage, and how to measure them on your own hardware,
are in [`docs/latency.md`](docs/latency.md).

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

## Roadmap

**v1** (this repository) is the full push-to-talk loop: hold, talk, release,
answer, barge-in, Esc, earcons, configuration, the `setup` command, tests,
docs, and CI.

**Phase 2**: streaming STT during the hold (transcribing as you talk rather
than after you release, for an estimated further ~0.3 s latency win), a
`parakeet-mlx` STT option, an OpenAI-compatible LLM backend so LM Studio or
Ollama can serve as the LLM engine, a spoken thinking-mode toggle, and
session transcript export.

**Phase 3**: a menu-bar app shell, tool calling, additional TTS backends
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
