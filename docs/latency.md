# Latency

LocalVoice's whole reason for existing is that a local pipeline can feel as
fast as a cloud one. This page tracks the target latency budget from the
architecture spec against real measurements on your hardware.

## Target vs. measured

Targets assume warm models (already loaded, first request already run once)
and a roughly 5-second utterance, per
`docs/superpowers/specs/2026-07-03-localvoice-architecture-design.md` §3.
"Deep mode" is `--deep` / `[llm].deep_model` (Qwen3.6-27B, dense, 6-bit);
default mode is the shipped `[llm].model` (Qwen3.6-35B-A3B, MoE, 4-bit).

| Stage | Target — default (35B-A3B) | Target — deep mode (27B dense) | Measured |
|---|---|---|---|
| Release → transcript (whisper-turbo) | ~0.35 s | ~0.35 s | run `uv run localvoice bench` |
| LLM first token (prompt-cached) | ~0.20 s | ~0.30 s | run `uv run localvoice bench` |
| First speakable clause (~10 tok) | ~0.15 s | ~0.65 s | run `uv run localvoice bench` |
| Kokoro first chunk | ~0.25 s | ~0.25 s | run `uv run localvoice bench` |
| **Voice-to-voice** | **≈0.95 s** | **≈1.5 s** | run `uv run localvoice bench` |

These are engineering targets, not promises — they depend on the specific
Apple Silicon chip, unified memory bandwidth, thermal state, and which
models are actually configured. The measured column above is a placeholder;
it gets filled in once the benchmark has been run and its output recorded
here.

## Running the benchmark

```bash
uv run localvoice bench
```

`src/localvoice/bench.py::run_bench` loads all three configured engines,
synthesizes a deterministic ~4-second spoken question with the macOS `say`
command (`"What is the capital of Australia, and why is it not Sydney?"`),
and times, per run: STT transcription, time to first LLM token, time from
first token to first speakable clause (via the same `ClauseChunker` /
`TextFilter` the real pipeline uses), and time to synthesize that first
clause's audio with Kokoro. It runs `--runs` times (default 3, override with
`--runs N`) and prints the best (lowest) total as "best voice-to-voice
(excl. playback buffer)" — excluding playback buffer because the benchmark
measures pipeline stages directly rather than through the audio output
queue.

Add `--deep` to benchmark with `[llm].deep_model` instead of `[llm].model`
(same flag as `localvoice run --deep`):

```bash
uv run localvoice bench --deep
```

Use whichever `--config` overlay you'd normally run with, including
`localvoice.local.toml` if you have one — `bench` loads config the same way
`run` does.

## What moves the needle

Per the architecture spec's risk section: if a bench run misses target,
the per-stage breakdown tells you which stage to attack. Documented
fallbacks are a smaller/lighter LLM (see `docs/models.md`), the
`parakeet-mlx` STT backend (phase 2, not yet available), or moving Kokoro
off the GPU if it's starved by concurrent LLM decode (unexpected at Kokoro's
82M parameter size, but the mitigation is documented in case bench shows
otherwise). Phase 2's streaming STT during the hold is expected to remove
about 0.3 s from the default-mode total once implemented.
