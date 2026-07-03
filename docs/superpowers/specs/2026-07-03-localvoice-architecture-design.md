# LocalVoice — architecture design

Date: 2026-07-03
Status: implemented (v1); amendment trail below
Target machine: MacBook Pro M5 Pro, 64 GB unified memory, macOS 26.5.1

> **Post-implementation amendments (2026-07-03).** Recorded during execution and final review; each is reflected in the code and `docs/architecture.md`:
> 1. Concurrency: threads + queues instead of asyncio (§3, amended at planning).
> 2. Cancel sequences run truncate-history BEFORE flush-player — flush clears the spoken-tag accounting (§3, found by review, fixed).
> 3. LLM prompt caching reworked to token-level prefix reuse: the full conversation is re-templated canonically each turn and only the un-cached suffix is prefilled; caches that cannot trim (hybrid-attention models like Qwen3.6) fall back to full re-prefill. Replaces the original message-list "incremental" policy, which final review found drift-defeated and template-malformed.
> 4. Deferred from §5/§3 to known limitations: audio-device-change stream restart; the spoken thinking-mode filler (thinking is silent in v1). Measured v1 latency: 0.82 s voice-to-voice (35B-A3B), 1.43 s (27B deep) — see `docs/latency.md`.
> 5. Per-response pipeline threads replaced by one persistent inference thread owning all MLX imports, loads, and generation (found in live acceptance: MLX streams are only usable on their creating thread — "There is no Stream(gpu, N) in current thread"). Engine access is now serialized by construction; `cmd_run` exits via `os._exit(0)` because worker-thread Metal state can SIGBUS in interpreter teardown.

## 1. Goal

A local, private, open-source voice assistant for macOS that feels like ChatGPT Realtime but runs entirely on-device: intelligent (reasoning-capable LLM), state-of-the-art local latency (~1 s voice-to-voice, path to ~0.6 s), with walkie-talkie push-to-talk interaction.

Interaction contract (fixed, by design):

- Hold **right ⌘** → assistant audio stops instantly, mic records.
- Release → utterance is transcribed, LLM responds, TTS speaks.
- **Esc** → stop the current response (or discard the current recording).
- No VAD, no end-of-turn detection, no echo cancellation — push-to-talk makes all three unnecessary.

### Non-goals (v1)

Automatic turn-taking/VAD, acoustic echo cancellation, tool calling, GUI, persistent cross-session memory, streaming STT during the hold (phase 2), multilingual tuning (works incidentally via Whisper/Kokoro, not optimized).

## 2. Decisions (locked with user, 2026-07-03)

| Decision | Choice | Rationale |
|---|---|---|
| Runtime | Python 3.12+/asyncio, `uv`-managed | mlx-whisper / mlx-lm / mlx-audio are first-class; fastest path to SOTA; largest contributor pool |
| Architecture | Single process, engines in-process | No IPC hops; direct prompt-cache and cancellation control |
| STT | `mlx-whisper` + whisper-large-v3-turbo (MLX build, ~1.6 GB download) | User's model choice; the existing WhisperKit CoreML files are compiled Swift artifacts and can't be reused from Python |
| LLM engine | In-process `mlx-lm` | Loads existing MLX folders in `~/.lmstudio/models`; OpenAI-compatible backend (LM Studio et al.) arrives phase 2 behind the same protocol |
| Default LLM | **Qwen3.6-35B-A3B**, 4-bit MLX (~19 GB download) | MoE ≈60–90 tok/s decode → ≈0.95 s voice-to-voice; thinking mode usable in voice |
| Deep-mode LLM | Qwen3.6-27B MLX 6-bit (already on disk) | Highest quality in family; ≈1.5 s first sound — selectable via config/flag |
| TTS | Kokoro-82M via `mlx-audio` | Apache-2.0, 24 kHz, far faster than realtime; consensus 2026 default small TTS |
| Audio I/O | `sounddevice` (PortAudio) | 16 kHz mono capture, 24 kHz playback, flushable output queue |
| Hotkey | `pynput` (Quartz event tap) | Right ⌘ = modifier-only key: safe to hold globally; requires one-time Input Monitoring permission |
| License | MIT | |

Default model pinned to `mlx-community/Qwen3.6-35B-A3B-4bit` (verified available, 64k downloads). The `-4bit-DWQ` and `-MTP-4bit` variants are candidates to benchmark later for quality and decode-speed gains respectively.

## 3. System architecture

One process, plain threads + queues (amended from asyncio at planning: every workload is a blocking C/Metal call — MLX, PortAudio, Quartz — so an event loop adds bridging complexity without benefit). Main thread runs the state machine on an event queue; the hotkey listener and audio callbacks run on their own threads; each response runs in a pipeline worker thread; a per-response cancellation token plus a generation counter make every stage abortable and stale events ignorable.

```
pynput key tap ──events──▶ Orchestrator (state machine)
sounddevice mic ──frames─▶   │
                             ├─▶ STTEngine.transcribe(audio) ──text──▶
                             ├─▶ LLMEngine.stream(messages) ──deltas──▶ ClauseChunker ──clauses──▶
                             ├─▶ TTSEngine.synthesize(clause) ──chunks──▶ AudioPlayer ──▶ speakers
                             └─▶ Transcript (history + barge-in truncation)
```

### Component interfaces

Engines are minimal protocols so every stage is swappable and the orchestrator is testable with fakes:

```python
class STTEngine(Protocol):
    def load(self) -> None: ...
    def transcribe(self, audio: NDArray[np.float32], sample_rate: int) -> str: ...

class LLMEngine(Protocol):
    def load(self) -> None: ...
    def stream(self, messages: list[Message], *, think: bool) -> Iterator[str]: ...

class TTSEngine(Protocol):
    def load(self) -> None: ...
    def synthesize(self, text: str) -> Iterator[NDArray[np.float32]]: ...  # 24 kHz chunks
```

Supporting components:

- **HotkeyListener** — Quartz tap via pynput; emits `PTT_DOWN`, `PTT_UP`, `ESC` into the orchestrator's queue. Debounce: a hold shorter than 120 ms is discarded.
- **MicCapture** — opens a 16 kHz mono stream on `PTT_DOWN`, appends frames to a buffer, returns it on `PTT_UP`.
- **ClauseChunker** — accumulates LLM deltas; emits a chunk at the first clause boundary (`.,;:—?!` or newline) after ~60 chars, hard-flushes at ~200 chars mid-clause and at stream end. Never splits mid-word.
- **Sanitizer** — strips `<think>…</think>`, markdown syntax, code fences (replaced with "code omitted"), emoji, before text reaches the chunker.
- **AudioPlayer** — 24 kHz output stream fed by a chunk queue; `flush()` empties queue and silences within one buffer (~20 ms); reports chunks actually played for truncation accounting. Ordering constraint: truncation must read `spoken_tags()` before `flush()` clears the accounting, so every cancel sequence runs truncate-then-flush.
- **Transcript** — message history; on interruption, the assistant turn is truncated to the text of chunks actually played, so the model knows where it was cut off.
- **Earcons** — short generated tones on key-down, key-up, and cancel; instant feedback that masks pipeline latency.

### State machine

States: `IDLE`, `LISTENING`, `RESPONDING` — with `RESPONDING` split internally into `processing` (release → first audio) and `speaking` (audio flowing), because cancellation and UI differ between them.

| State | Event | Action | Next |
|---|---|---|---|
| IDLE | PTT_DOWN | earcon, start mic | LISTENING |
| LISTENING | PTT_UP (≥120 ms, non-silent) | stop mic, start pipeline | RESPONDING·processing |
| LISTENING | PTT_UP (<120 ms or silent) | discard | IDLE |
| LISTENING | ESC | discard recording | IDLE |
| RESPONDING·processing | first audio chunk | — | RESPONDING·speaking |
| RESPONDING (either) | PTT_DOWN | cancel pipeline, truncate history, flush player, start mic | LISTENING |
| RESPONDING (either) | ESC | cancel, truncate, flush | IDLE |
| RESPONDING·speaking | PLAYBACK_DONE | append full turn to history | IDLE |
| any | PIPELINE_ERROR | log, brief console notice | IDLE |

The state machine is pure logic (no I/O), driven by an event queue — fully unit-testable with fake engines.

### Latency budget (targets, warm models, ~5 s utterance)

| Stage | Default (35B-A3B) | Deep mode (27B dense) |
|---|---|---|
| Release → transcript (whisper-turbo) | ~0.35 s | ~0.35 s |
| LLM first token (prompt-cached) | ~0.20 s | ~0.30 s |
| First speakable clause (~10 tok) | ~0.15 s | ~0.65 s |
| Kokoro first chunk | ~0.25 s | ~0.25 s |
| **Voice-to-voice** | **≈0.95 s** | **≈1.5 s** |

These are engineering targets, not promises: `scripts/bench.py` measures each stage on real hardware and `docs/latency.md` publishes measured numbers. Phase 2 (streaming STT during the hold) removes ~0.3 s.

Memory: whisper-turbo ~1.6 GB + 35B-A3B 4-bit ~19 GB + Kokoro ~0.5 GB + KV/overhead ~2 GB ≈ **23 GB**, comfortable within the ~48 GB default GPU wired limit on a 64 GB machine.

### Reasoning ("thinking") mode

Off by default for latency. When enabled (config or `--think`), Qwen3.6's thinking runs silently (never spoken); if thinking exceeds ~1.5 s the assistant speaks a short filler ("Let me think about that…"). Think-tags are stripped defensively in the sanitizer regardless of mode.

### Conversation shaping

System prompt tuned for speech: short sentences, contractions, no lists/markdown/emoji, answers sized for listening (with an invitation to ask for more), aware it may be interrupted mid-sentence.

## 4. Configuration

`localvoice.toml` at repo root (overridable via `--config`, individual `--flags`):

```toml
[stt]
engine = "mlx_whisper"
model = "mlx-community/whisper-large-v3-turbo"

[llm]
engine = "mlx_lm"
model = "mlx-community/Qwen3.6-35B-A3B-4bit"
deep_model = "/Users/…/.lmstudio/models/lmstudio-community/Qwen3.6-27B-MLX-6bit"
think = false
max_tokens = 1024

[tts]
engine = "kokoro_mlx"
voice = "af_heart"
speed = 1.0

[keys]
ptt = "cmd_r"      # any pynput key name; modifier-only keys recommended
stop = "esc"

[audio]
input_device = ""   # default device when empty
output_device = ""
```

## 5. Error handling

- **Input Monitoring permission missing** → pynput yields no events; on startup, a self-test detects this and prints the exact System Settings path, then exits non-zero.
- **Microphone permission missing** → capture self-test at startup; same treatment.
- **Models missing** → `localvoice setup` downloads all three (with sizes shown and confirmation); the main command refuses to start with a clear pointer to setup.
- **Engine failure mid-response** → pipeline task catches, emits `PIPELINE_ERROR`, session survives, error logged with context.
- **Audio device changes** (e.g. AirPods connect) → stream errors trigger a stream restart on the new default device.
- **Config validation** → unknown keys, bad key names, and over-memory model combinations produce actionable messages at startup, not mid-conversation.

## 6. Testing strategy

- **State machine**: scripted event sequences against fake engines assert transitions and side-effect calls (mic start/stop, flush, truncate). This is the core correctness surface — barge-in and Esc in every state.
- **ClauseChunker**: table + property tests (never splits words, respects min/max, flushes at end).
- **Sanitizer**: table tests for markdown/emoji/think-tags/code fences.
- **Truncation accounting**: interrupt after N chunks → history contains exactly the spoken prefix.
- **Slow smoke test** (`-m slow`, local only): real pipeline with tiny models (whisper-tiny + the on-disk Qwen3.5-0.8B + Kokoro) end-to-end from a WAV fixture.
- **CI**: GitHub Actions on arm64 macOS — ruff, unit tests. Model-dependent tests excluded.

## 7. Repository layout

```
LocalVoice/
├── README.md                  # demo, quickstart, permissions setup guide
├── LICENSE                    # MIT
├── pyproject.toml             # uv-managed; console script `localvoice`
├── localvoice.toml
├── src/localvoice/
│   ├── __main__.py            # CLI: run (default), setup, bench
│   ├── app.py                 # orchestrator wiring
│   ├── states.py              # state machine (pure)
│   ├── events.py
│   ├── config.py
│   ├── hotkey.py
│   ├── audio/{capture,player,earcons}.py
│   ├── stt/{base,whisper_mlx}.py
│   ├── llm/{base,mlx_lm}.py
│   ├── tts/{base,kokoro_mlx}.py
│   ├── textproc/{chunker,sanitize}.py
│   └── transcript.py
├── tests/
├── scripts/bench.py
├── docs/{architecture,latency,models}.md
└── .github/workflows/ci.yml
```

## 8. Phases

1. **v1** — full loop (hold → talk → release → answer), barge-in, Esc, earcons, config, setup command, tests, docs, CI, README.
2. **Phase 2** — streaming STT during hold (~0.3 s win), `parakeet-mlx` STT option, OpenAI-compatible LLM backend (LM Studio/Ollama), spoken thinking-mode toggle, session transcript export.
3. **Phase 3** — menu-bar app shell, tool calling, more TTS backends (Chatterbox-Turbo, Qwen3-TTS), experiment: Gemma 4 E4B native audio-in (skip STT entirely).

## 9. Risks and mitigations

- **pynput / macOS 26 event-tap changes** → startup self-test catches silently-broken taps; fallback is a direct CGEventTap via pyobjc. Right ⌘ held during other keystrokes still acts as ⌘ in the focused app — documented; key is configurable (e.g. F13).
- **mlx-audio API churn** → pin versions in `pyproject.toml`; TTS behind protocol anyway.
- **Qwen3.6 chat template / think-toggle quirks in mlx-lm** → verified at implementation; sanitizer strips think-tags defensively regardless.
- **GPU contention (LLM decode vs Kokoro)** → expected fine (Kokoro is 82M); if bench shows TTS starvation, alternate scheduling or move Kokoro to CPU via ONNX backend.
- **Latency targets miss** → bench script isolates the offending stage; fallbacks documented (parakeet STT, smaller LLM, 4-bit Kokoro).
