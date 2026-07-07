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

| Stage | Target — default (35B-A3B) | Measured — default (35B-A3B) | Target — deep mode (27B dense) | Measured — deep mode (27B dense) |
|---|---|---|---|---|
| Release → transcript (whisper-turbo) | ~0.35 s | 0.34 s | ~0.35 s | 0.34 s |
| LLM first token (prompt-cached) | ~0.20 s | 0.31 s | ~0.30 s | 0.59 s |
| First speakable clause (~10 tok) | ~0.15 s | 0.07 s | ~0.65 s | 0.42 s |
| Kokoro first chunk | ~0.25 s | 0.10 s | ~0.25 s | 0.09 s |
| **Voice-to-voice** | **≈0.95 s** | **0.82 s** | **≈1.5 s** | **1.43 s** |

These are engineering targets, not promises — they depend on the specific
Apple Silicon chip, unified memory bandwidth, thermal state, and which
models are actually configured.

**Measured on:** Apple M5 Pro, 64 GB unified memory, macOS 26.5.1, on
2026-07-03. Default-mode LLM is the local `Qwen3.6-35B-A3B-4bit` MoE model;
deep mode is `Qwen3.6-27B-6bit` dense, both via the gitignored
`localvoice.local.toml` overlay (local on-disk paths, no download). STT and
TTS in both runs are the shipped defaults (`whisper-large-v3-turbo`,
`Kokoro-82M`). Each column is `uv run localvoice bench --runs 3` (add
`--deep` for the deep-mode column), taking the best (lowest-total) of the
three runs, matching what the bench script itself reports as "best
voice-to-voice." Run-to-run variance was mostly a cold first run (JIT /
cache warmup) followed by two stable runs within ~0.01-0.05 s of each
other; the values above are representative, not one-off outliers.

## mlx-vlm engine (Gemma 4 26B-A4B, tools phase T2)

Tools T2 (`docs/superpowers/plans/2026-07-07-localvoice-tools-t2.md`) added a
second LLM engine, `mlx_vlm`, so the user's own Gemma 4 26B-A4B model can load
its vision tower (enabling image turns for a later phase's screenshot tool)
while still driving ordinary text turns through the same shipped generation
code — `mlx_lm.stream_generate` against `model.language_model`, via a thin
logits adapter (`_TextTowerAdapter` in `src/localvoice/llm/mlx_vlm_engine.py`).
The phase gate required benching `mlx_lm` and `mlx_vlm` back-to-back on the
*same* model (the user's `gemma-4-26B-A4B-it-MLX-4bit`, via the gitignored
`localvoice.local.toml` overlay) and passing both: best voice-to-voice
**≤ 1.0 s**, and within **~15%** of the mlx_lm number measured the same
session.

| Stage | Measured — mlx_lm (Gemma 4 26B-A4B) | Measured — mlx_vlm (Gemma 4 26B-A4B) |
|---|---|---|
| Release → transcript (whisper-turbo) | 0.32 s | 0.33 s |
| LLM first token (prompt-cached) | 0.32 s | 0.32 s |
| First speakable clause (~10 tok) | 0.09 s | 0.09 s |
| Kokoro first chunk | 0.10 s | 0.10 s |
| **Voice-to-voice** | **0.82 s** | **0.84 s** |

**Gate: PASS.** 0.84 s is under the 1.0 s ceiling, and only 2.4% slower than
0.82 s — well inside the ~15% budget. Both numbers are the best (lowest-total)
of `uv run localvoice bench --runs 3`, run immediately back-to-back in the
same terminal session on the machine described below; the `mlx_vlm` run used
a scratch config overlay (`[llm].engine = "mlx_vlm"`, every other `[llm]`
value — model path, `context_tokens`, `max_tokens`, `think_tokens`,
`system_prompt` — mirrored byte-for-byte from the merged config the `mlx_lm`
run used) so the comparison is apples-to-apples. Full per-run output:

```
mlx_lm  — uv run localvoice bench --model <gemma-4-26B-A4B-it-MLX-4bit> --runs 3
  load whisper: 1.3s | load llm: 2.5s | load kokoro: 1.7s
  run 1: stt 0.34s | ttft 0.44s | clause 0.09s | tts 0.10s | total 0.96s
  run 2: stt 0.35s | ttft 0.32s | clause 0.09s | tts 0.10s | total 0.86s
  run 3: stt 0.32s | ttft 0.32s | clause 0.09s | tts 0.10s | total 0.82s
  best voice-to-voice: 0.82s

mlx_vlm — uv run localvoice bench --config <scratch.toml, engine=mlx_vlm> --runs 3
  load whisper: 1.2s | load llm: 3.8s | load kokoro: 1.5s
  run 1: stt 0.35s | ttft 0.46s | clause 0.09s | tts 0.10s | total 1.00s
  run 2: stt 0.35s | ttft 0.33s | clause 0.09s | tts 0.10s | total 0.87s
  run 3: stt 0.33s | ttft 0.32s | clause 0.09s | tts 0.10s | total 0.84s
  best voice-to-voice: 0.84s
```

LLM load time is the one stage with a real, consistent gap — 2.5 s vs. 3.8 s
— the vision tower's extra weights loading once at process startup; it does
not touch per-turn voice-to-voice latency. STT/TTS load and per-run STT/TTS
stage timings are noise-level identical between the two runs, as expected
(neither engine touches whisper or Kokoro).

Following the gate pass, the user's overlay (`localvoice.local.toml`, not
committed) now sets `[llm] engine = "mlx_vlm"`. The shipped default
(`localvoice.toml`) stays `mlx_lm` with the Qwen model — the shipped default
model has no tool/vision template support regardless of engine, so there is
nothing for `mlx_vlm` to add there.

**Measured on:** Apple M5 Pro, 64 GB unified memory, macOS 26.5.1, on
2026-07-07 — the same hardware/OS as the table above, same session for both
engine rows.

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
