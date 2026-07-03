# Models

LocalVoice has three swappable engines — STT, LLM, TTS — each configured by
a `[section]` in `localvoice.toml` (see the README's configuration table for
every key). This page covers what to put in the `model` fields and what
trade-offs to expect.

## How model loading works

Every engine's `model` value can be either a Hugging Face repo id
(`"org/name"`, containing a `/`) or a local filesystem path to an
already-downloaded model folder. `Config` doesn't distinguish between them —
loading logic does:

- `localvoice setup` (`src/localvoice/__main__.py::cmd_setup`) checks each
  configured model with `Path(model).expanduser().exists()` first. If it's a
  real path on disk, setup leaves it alone. Otherwise, if the string
  contains `/`, it's treated as an HF repo id and queued for
  `huggingface_hub.snapshot_download`. A string that's neither an existing
  path nor a repo id (no `/`) is a hard error before anything downloads.
- The engines themselves (`MlxLmEngine.load()`, `WhisperMlxEngine`,
  `KokoroMlxEngine.load()`) pass the configured string straight to
  `mlx_lm.load()` / `mlx_whisper.transcribe(path_or_hf_repo=...)` /
  `mlx_audio.tts.utils.load_model()`, all of which accept either form
  natively.

Practically: **any MLX-format model folder under `~/.lmstudio/models`
works as a local `model` value** for the LLM engine — point `[llm].model`
or `[llm].deep_model` at its path and LocalVoice loads it directly, no
download, no HF repo id needed. This is how the local overlay
(`localvoice.local.toml`, gitignored — see the README) points this machine
at on-disk LM Studio downloads.

## LLM presets

`[llm].engine = "mlx_lm"` is the only implemented LLM backend in v1 (an
OpenAI-compatible backend for LM Studio/Ollama-style servers is phase 2 —
see the architecture spec's roadmap). Three presets are relevant today:

| Preset | `model` | Role | Expected trade-off |
|---|---|---|---|
| Default | `mlx-community/Qwen3.6-35B-A3B-4bit` | shipped default in `localvoice.toml` | MoE (~3B active params), fast decode (~60-90 tok/s target), tuned for the ~0.95 s voice-to-voice target — the everyday preset |
| Deep | Qwen3.6-27B MLX 6-bit, local path via `[llm].deep_model` | opt-in via `--deep` or `localvoice.local.toml` | dense 27B, highest quality in the family, slower first token and first clause (~1.5 s voice-to-voice target) — reach for it when you want the smartest answer and don't mind the extra second |
| Lightweight | Gemma-4-E4B | not configured by default; swap `[llm].model` to try it | smallest of the three, fastest load and lowest memory, answer quality is the trade-off — useful on lower-RAM machines or when you want headroom for other apps |

`--deep` (`src/localvoice/__main__.py`) requires `[llm].deep_model` to be
non-empty — `load_config(..., deep=True)` raises a `ConfigError` otherwise
(`src/localvoice/config.py::load_config`). Setting `deep_model` in
`localvoice.toml` or the local overlay is what makes `--deep` usable.

Two variant naming conventions worth knowing about when picking a model, and
candidates to benchmark later rather than defaults today (per the
architecture spec's model decision table):

- **`-DWQ`** variants (dynamic weight quantization) — potential quality
  gains at the same bit width; unverified on this pipeline's latency budget.
- **`-MTP`** variants (multi-token prediction) — potential decode-speed
  gains; also unverified here. Neither is wired into `localvoice bench` yet;
  trying one just means pointing `[llm].model` at its path or repo id.

## STT

`[stt].engine = "mlx_whisper"`, default `model = "mlx-community/whisper-large-v3-turbo"`
(~1.6 GB). Swappable to any other MLX Whisper build the same way — set
`[stt].model` to its repo id or local path. `parakeet-mlx` is a phase-2 STT
option per the roadmap, not available in v1.

## TTS

`[tts].engine = "kokoro_mlx"`, default `model = "prince-canuma/Kokoro-82M"`.
Voice and speaking rate are separate keys — `[tts].voice` (default
`"af_heart"`) and `[tts].speed` (default `1.0`) — not part of the model
path. Other Kokoro-compatible checkpoints can be swapped in the same way as
the other engines. Additional TTS backends (Chatterbox-Turbo, Qwen3-TTS) are
phase-3 roadmap items, not available in v1.

### mlx-audio version pin

`mlx-audio` is pinned to exactly `0.4.1` in `pyproject.toml`
(`mlx-audio==0.4.1`), not a floor like the other dependencies. `0.4.4`
ships a regression in Kokoro's `SineGen` component that breaks audio length
(upstream `Blaizzy/mlx-audio` issues #784 and #803); a fix has been merged
upstream (PR #785) but had not shipped in a release as of this pin. Lift the
pin to a floor (`mlx-audio>=0.4.1` or similar) once a release containing
that fix is out and has been verified against this pipeline's TTS tests.

Kokoro's grapheme-to-phoneme conversion depends on `misaki[en]`, which is
declared directly in `pyproject.toml` as a runtime dependency (it isn't
pulled in automatically by `mlx-audio`'s own dependency list). Misaki's
fallback path uses `espeak-ng`; install it via Homebrew
(`brew install espeak-ng`) as an environment prerequisite — it's a system
binary, not a Python package, so it isn't and can't be declared in
`pyproject.toml`.
