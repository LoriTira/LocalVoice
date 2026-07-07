# LocalVoice tools — web search and screen vision — design

Date: 2026-07-07
Status: approved decisions locked; pending user review of this spec
Depends on: v1 engine + GUI phase B (both shipped)

## 1. Goal

Let the assistant act during a turn, hands-free: search the web when the
question needs current or external facts (free, no API keys), and look at
the user's screen when asked ("look at my screen — what's this error?").
Tool use is decided by the model through its native tool-calling format;
the voice interaction stays exactly as it is — hold, talk, release, answer.

Decisions locked with the user:

| Decision | Choice |
|---|---|
| Vision runtime | Same Gemma 4 26B via **mlx-vlm** for everything (text + vision), gated on a performance spike; fallback is on-demand vision load |
| Thinking-phase audio | Silence + state orb (shipped with the channel-think fix) |
| Search cost | Completely free: DuckDuckGo via `ddgs`, no keys, no paid APIs |
| Tool-call format | Gemma 4's native template markers (`<|tool_call>call:` / `<|tool_response>`) |

## 2. Why this is natural for the current stack

Verified on this machine:

- The local `gemma-4-26B-A4B-it-MLX-4bit` kept its **full vision tower**
  (`vision_config` + 358 `embed_vision.*`/vision weight keys in the
  safetensors index) — most MLX text conversions strip it; this one didn't.
- Its chat template natively formats **tool definitions** (a
  `format_parameters` macro), **tool calls** (`<|tool_call>call:`), and
  **tool responses** (`<|tool_response>`).
- `mlx-vlm` 0.6.4 ships a `gemma4` implementation and handles text-only
  generation too.

So one already-installed model can do text, tools, and screenshots. No new
model downloads are required for v1 of this feature.

## 3. Architecture

One new pipeline layer and one new engine option; everything else reuses
shipped machinery.

```
LLM raw stream
  → ChannelThinkTranslator        (shipped: reasoning → canonical <think>)
  → ToolCallParser                (new: detects <|tool_call>call:… blocks)
        │ no tool call: deltas flow to TextFilter → clauses → TTS (unchanged)
        │ tool call:
        ▼
  ToolRunner (worker thread, cancellation-aware)
        │ append tool_call + tool_response messages to the turn
        ▼
  re-stream the continuation through the same path (max `max_rounds`)
```

- **ToolCallParser** — a streaming parser in `textproc/` shaped exactly like
  `ChannelThinkTranslator`: hold-back across delta boundaries, emits
  passthrough text until a tool-call marker opens, then captures the call
  (name + JSON-ish args per the Gemma format) instead of speaking it. On a
  malformed call body, the captured text is fed back to the model as an
  error tool response (one retry), never spoken.
- **Turn loop** — `run_pipeline` gains a bounded loop: stream → (tool call?
  execute → extend messages → stream again). `max_rounds` (default 3) caps
  runaway loops. The state machine is untouched: the whole loop is one
  RESPONDING turn; barge-in and Esc cancel tool execution through the same
  cancel token the stream already honors (HTTP fetches get short timeouts
  and check the token between requests).
- **Tool registry** — `tools/` package: each tool declares name,
  description, JSON-schema parameters (rendered into the prompt by the
  template's own `tools=[…]` support via `apply_chat_template`), and an
  `execute(args, cancel) -> ToolResult`. Config can disable any tool;
  disabled tools are simply not offered to the model.
- **Protocol** — two additive events (protocol stays version 1):
  `tool_call {name, summary}` when a call starts (summary is a short
  human line, e.g. `searching: weather in Boston`), and
  `tool_result {name, ok, summary}` when it finishes. The app shows the
  summary as a live chip under the orb; the terminal prints one line.

## 4. The tools (v1)

### `web_search(query, max_results=5)`

- `ddgs` (DuckDuckGo scraping library — free, keyless) for results:
  title, URL, snippet per result.
- `fetch_page(url)` — a second tool the model may call to read one result
  more deeply: `trafilatura` fetches and extracts the main text, truncated
  to `page_char_cap` (default 8000 chars). Two small tools compose better
  than one mega-tool and each stays independently testable.
- No internet: the tool returns a structured
  `{ok: false, error: "no internet connection"}` — the model says so out
  loud naturally instead of the pipeline failing.
- In-session per-query result cache so a follow-up question doesn't re-hit
  the network for the same search.

### `look_at_screen()`

- `screencapture -x` (silent) of the main display to a temp PNG, downscaled
  to ≤1536 px on the long edge (vision-token cost control), fed to the
  model as an image message part in the continuation turn — this is what
  requires the mlx-vlm runtime.
- Requires the **Screen Recording** permission attributed to
  LocalVoice.app: the Setup pane gains a third permission card (preflight
  `CGPreflightScreenCaptureAccess()`, request
  `CGRequestScreenCaptureAccess()`, deep link, same live re-check pattern
  as the existing cards). Terminal mode attributes to the terminal app.
- The tool description steers usage ("call when the user refers to their
  screen"); the screenshot happens only when the model calls the tool —
  never ambiently.

## 5. Engine: mlx-vlm for Gemma 4

New `MlxVlmEngine` implementing the existing LLM engine protocol plus an
optional image slot on the final user message. Selected via the existing
`[llm] engine` config key (`"mlx_vlm"`), schema picker updated. All MLX
work stays on the single inference thread, unchanged.

**Spike gate (first task of the plan, before any dependent work):**
benchmark mlx-vlm text-only decode on the same prompts as mlx-lm.

- Adopt (switch the default engine for Gemma) if voice-to-voice stays
  ≤ 1.0 s on the standard bench and decode throughput is within ~15% of
  mlx-lm.
- Otherwise fall back: mlx-lm remains the text engine; `look_at_screen`
  lazily loads the same weights via mlx-vlm on first use (~15 GB extra
  unified memory while resident, freed on engine reload) and vision turns
  run there.
- Prompt-cache: verify what mlx-vlm supports; if its cache can't trim,
  full re-prefill per turn is acceptable **if** the bench stays within the
  gate (Gemma-4 MoE prefill is fast).

`mlx-vlm`'s dependency tree must resolve against the pinned
`mlx-audio==0.4.1` + overridden `mlx-lm==0.31.3`; the spike task verifies
resolution before anything else builds on it.

## 6. Prompting

The system prompt gains a tools paragraph (kept in `localvoice.toml` like
the rest): answer from knowledge when confident; search for current
events, prices, weather, or anything the user implies is recent; after
searching, speak a short synthesized answer, not a list of results; name
the source site only if the user asks where it came from.

## 7. Configuration

New `[tools]` section (auto-appears in the app's Settings via the derived
schema):

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `true` | master switch; off = no tools offered to the model |
| `web_search` | `true` | offer web_search + fetch_page |
| `screenshot` | `true` | offer look_at_screen |
| `max_rounds` | `3` | tool-call rounds per turn before forcing an answer |
| `search_results` | `5` | results per search |
| `page_char_cap` | `8000` | main-text truncation for fetch_page |

All keys hot-apply instantly: tool config is read per turn, and no engine
reload is ever needed for it.

## 8. Testing

- ToolCallParser: TDD with fixtures captured from the real template (tool
  definitions rendered + a live tool-call generation dump), split-marker
  cases, malformed-JSON case, passthrough case.
- Pipeline loop: fake LLM scripted to emit call → continuation; asserts
  tool executed once, messages extended correctly, spoken text excludes
  call payloads; cancellation mid-tool; max_rounds exhaustion.
- Tools: `web_search`/`fetch_page` unit-tested with mocked network (CI has
  no internet dependence); `look_at_screen` unit test asserts capture +
  resize (guarded to skip without Screen Recording permission).
- Slow/local-only: one live search turn e2e; one live screenshot turn
  e2e; the mlx-vlm spike bench script committed under `scripts/`.
- App: reducer tests for the two new events; Setup card logic test
  mirroring the existing permission-card tests.

## 9. Phases

- **T1 — tool framework + web search** (no engine change: works on
  today's mlx-lm): parser, turn loop, registry, web_search + fetch_page,
  protocol events, app chip, config section, docs. Ships standalone.
- **T2 — mlx-vlm spike + engine** (the gate): bench, decision, engine
  implementation behind `[llm] engine`, default flip for Gemma if adopted.
- **T3 — screen vision**: look_at_screen, Screen Recording Setup card,
  image plumbing to the vlm engine, docs + acceptance.

## 10. Non-goals (v1)

Autonomous browsing beyond one fetch per call; multi-monitor selection or
window-picking UI (main display only); OCR fallback for the non-vision
engine; tool use in `--deep` mode with a non-tool-template model (tools
are offered only when the template supports them); paid search APIs.

## 11. Amendment (2026-07-07): T2 spike pre-run — gate PASSES

The §5 spike was run ahead of planning (same model, same prompts, this
machine). Results:

| Metric | mlx-lm 0.31.3 | mlx-vlm 0.6.4 | Delta | Gate (~15%) |
|---|---|---|---|---|
| Decode | 79.2 tok/s | 76.9 tok/s | −3% | pass |
| Bulk prefill (≈1.5k tokens) | 1961 tok/s | 1720 tok/s | −12% | pass |
| Peak memory | 14.5 GB | 15.7 GB | +1.1 GB | pass |

(Short-prompt prefill shows a larger gap — fixed per-call overhead, not a
rate difference; irrelevant at conversation sizes.)

Dependency resolution: mlx-vlm 0.6.4 requires `mlx-audio>=0.4.3`,
conflicting with our `==0.4.1` pin — but **mlx-audio 0.4.3 was verified
Kokoro-clean** (the SineGen regression repro passes at short/medium/long
utterance lengths; the breakage starts at 0.4.4). T2 therefore moves the
pin to `==0.4.3`, which also natively allows `mlx-lm 0.31.3` (ephemeral
co-resolution verified), likely retiring the `[tool.uv]` override.

Implementation candidate ranked first for T2: **hybrid generation** — load
via mlx-vlm once (vision-capable, single 15 GB residency), but drive
text-only turns through `mlx_lm.stream_generate` against
`model.language_model`. Verified: the tower is exposed,
`make_prompt_cache` accepts it (30 layers) — so the shipped prefix-reuse
code carries over unchanged; the one blocker is a return-type mismatch
(`LanguageModelOutput` vs raw logits), bridged by a small `__call__`
adapter returning `.logits`. Image turns use `mlx_vlm.stream_generate`.
Fallback if the adapter fights back: plain `mlx_vlm.stream_generate` for
everything, accepting full re-prefill per turn (within gate only for
short-to-medium conversations — measure before accepting).

## 12. Risks

- **ddgs fragility** (free scraping breaks occasionally) → tool errors are
  structured and spoken gracefully; the library is swappable behind the
  registry interface.
- **mlx-vlm maturity** (cache/streaming behind mlx-lm) → the T2 spike gate
  exists precisely to catch this before committing the default engine.
- **Tool-format drift across model updates** → parser fixtures pinned from
  the real template; unknown channels/markers pass through harmlessly
  (same defensive posture as the think translator).
- **Latency of search turns** (search 1–3 s + re-prefill + regeneration) →
  visible via the app chip and orb; `max_rounds` caps the worst case; the
  in-session cache absorbs repeats.
