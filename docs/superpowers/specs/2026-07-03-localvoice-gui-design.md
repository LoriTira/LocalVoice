# LocalVoice GUI — design

Date: 2026-07-03
Status: approved design, pending final spec review
Depends on: v1 engine (shipped; see `2026-07-03-localvoice-architecture-design.md`)

## 1. Goal

Replace the terminal UI with a native macOS application. Every configuration key and CLI flag becomes editable in the GUI; the assistant's state, conversation, reasoning, and latency become visible live. The terminal mode remains supported for headless/dev use.

Decisions locked with the user:

| Decision | Choice |
|---|---|
| Stack | Native SwiftUI app (macOS 15+ deployment floor) + Python engine subprocess |
| Form factor | Main window first; menu-bar extra deferred to a later phase |
| v1 scope | Full settings editor (given) + live conversation view + model manager + voice preview & meters + permission onboarding |
| Distribution | `.app` bundle from v1 (PyInstaller-embedded engine, GitHub Releases), dev fallback to repo checkout |

## 2. Architecture

Two components, one protocol:

```
LocalVoice.app (SwiftUI)                      localvoice engine (Python)
┌─────────────────────────┐   JSON lines     ┌──────────────────────────┐
│ EngineClient (actor)    │ ──── stdin ────▶ │ serve.py: command loop   │
│ AppState (@Observable)  │ ◀─── stdout ──── │ events ← orchestrator,   │
│ Views: Talk/Models/     │      stderr →log │ pipeline, player, engines │
│   Settings/Setup        │                  │ (existing v1 internals)  │
│ HotkeyMonitor (CGEvent) │                  │ single MLX inference     │
└─────────────────────────┘                  │ thread — unchanged       │
                                             └──────────────────────────┘
```

- The app spawns the engine (`Process` + pipes) and owns its lifecycle; engine crash → auto-respawn with backoff and a UI banner. `shutdown` command on quit; SIGKILL fallback after timeout.
- Protocol: one JSON object per line. stdout carries protocol only; all logging goes to stderr. Every message carries no framing beyond the newline; `ready` carries `"version": 1`.
- In GUI mode the engine does NOT start pynput and does NOT print status lines: `localvoice serve` wires orchestrator callbacks and player/capture hooks to protocol events instead. Terminal `localvoice run` is untouched.
- The global hotkey moves into Swift (CGEventTap) for GUI sessions, so Input Monitoring and Microphone permission prompts attribute to LocalVoice.app. Key events are forwarded as `ptt_down`/`ptt_up`/`esc` commands into the same orchestrator event queue the physical pynput path uses today.
- Audio capture/playback stay in the engine process (they sit on the latency path next to inference).

## 3. Protocol (v1)

Engine → app events:

| event | payload | source |
|---|---|---|
| `ready` | `version`, merged `config`, `schema` (§4), engine paths | serve startup (implemented payload: version, config, schema — engine paths come from `list_models`) |
| `state` | `idle \| listening \| processing \| speaking` | orchestrator `_show_state` replacement |
| `user_text` | `text` | pipeline `on_user_text` |
| `assistant_clause` | `text` | pipeline `on_assistant_clause` |
| `reasoning` | `text` | pipeline `on_thinking` |
| `turn_done` | `latency: {stt, ttft, first_clause, tts_first, total}` seconds | new pipeline stage timestamps |
| `level` | `rms` float, throttled to ≤20 Hz, only while listening | GatedBuffer write hook |
| `load_progress` | `engine: stt\|llm\|tts`, `phase: start\|done`, `seconds?` | engine loads and reloads |
| `download_progress` | `repo`, `pct`, `done` | model downloads |
| `models` | `installed: [{id, path, size_gb, kind}]` | reply to `list_models` |
| `config_applied` | full merged `config`, `reloaded: [engine…]` | reply to `set_config` |
| `error` | `message` | pipeline/serve errors |

App → engine commands:

| cmd | payload | behavior |
|---|---|---|
| `ptt_down` / `ptt_up` / `esc` | — | post the corresponding Event into the orchestrator queue (`ptt_up` carries `held_ms` measured in Swift) |
| `set_config` | `changes: {dotted.key: value}` | validate → write `localvoice.local.toml` → hot-apply (§5) → `config_applied` |
| `list_models` | — | scan `~/.lmstudio/models/*/*` and the HF cache; reply `models` |
| `download_model` | `repo` | `snapshot_download` on a worker thread, `download_progress` events |
| `preview_voice` | `voice` | synthesize a fixed sample line through the real player |
| `inject_audio` | `path` (only with `serve --allow-inject`) | feed a WAV as if it were a PTT capture — test hook |
| `shutdown` | — | clean stop, then `os._exit(0)` |

## 4. Derived settings schema

`schema.py` introspects the config dataclasses and emits one descriptor per field:

```json
{"key": "llm.think", "type": "bool", "value": false, "default": false,
 "section": "Language model", "label": "Thinking mode",
 "help": "Silent reasoning; shown in the console/GUI, never spoken.",
 "widget": "toggle"}
```

- Base widget is inferred from the Python type (`bool→toggle`, `int/float→number`, `str→text`). An annotation map upgrades specific keys: `llm.model`/`llm.deep_model → model_picker`, `tts.voice → voice_picker` (with preview), `keys.ptt`/`keys.stop → key_capture`, `audio.input_device`/`output_device → device_picker` (options listed live from sounddevice), `tts.speed`/`audio.rebuffer_ms → slider` with ranges.
- SwiftUI renders the form generically from descriptors, so any config field added later in Python appears in the GUI with no Swift change (with the generic widget until annotated).
- A schema completeness unit test asserts every `Config` field yields a descriptor.

## 5. Hot-apply semantics

`set_config` writes the overlay, then applies per field group:

| Fields | Action |
|---|---|
| `tts.voice`, `tts.speed`, `llm.think*`, `llm.max_tokens`, `llm.context_tokens`, `llm.system_prompt`, `keys.debounce_ms` | apply immediately, no reload |
| `llm.model`, `llm.deep_model` (when active), `stt.model`, `tts.model`, any `*.engine` | reload that engine on the inference thread with `load_progress` events; conversation history survives |
| `audio.input_device`, `audio.output_device`, `audio.rebuffer_ms` | restart the affected stream |
| `keys.ptt`, `keys.stop` | applied in Swift (GUI owns the tap); engine stores them for terminal mode |

Reloads run as jobs on the existing single inference thread, so they serialize naturally with responses; the UI disables PTT during a reload (state banner).

## 6. SwiftUI application

- `EngineClient` (actor): process spawn, pipe I/O, Codable message types, `AsyncStream<EngineEvent>`, respawn policy, and a dev toggle to launch `uv run localvoice serve` from a configurable checkout path instead of the bundled engine.
- `AppState` (@Observable): connection state, assistant state, transcript (`[Turn]` with user/assistant/reasoning entries), mic level, last-turn latency, config snapshot, schema, model list, download and reload progress.
- Views: `NavigationSplitView` with four sections. Talk — state orb, transcript with reasoning rows, mic level bar, latency chips, and on-screen hold-to-talk button (mouse-driven PTT, same commands). Models — installed list with sizes, per-engine assignment, download field with progress. Settings — the derived form, grouped by section, each row rendering by widget kind; footer names the overlay file. Setup — permission checks with deep links (`x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent` and `?Privacy_Microphone`), live re-check on focus, engine status and versions.
- `HotkeyMonitor`: CGEventTap for the configured PTT/stop keys (default right ⌘ / Esc), measuring `held_ms` in Swift; falls back to a visible "grant Input Monitoring" state when the tap can't be created.
- The window closing hides to background (assistant keeps working); reopening restores state from `AppState`. Quit menu item shuts the engine down.

## 7. Packaging and distribution

- `scripts/build_app.sh`: PyInstaller (onedir) builds `localvoice-engine` from the repo with hooks for mlx/misaki/spacy data; output embedded at `LocalVoice.app/Contents/Resources/engine/`; `xcodebuild` produces the app; zip for release.
- GitHub Actions release workflow (tag-triggered, macos arm64): build, ad-hoc sign, attach zip to the Release. README documents right-click-Open for Gatekeeper (no paid cert; hardened-runtime notarization is a later option).
- espeak-ng: bundle its data via PyInstaller datas if feasible; otherwise the Setup pane detects its absence and shows the `brew install espeak-ng` instruction. This is the declared packaging risk with a user-visible fallback either way.
- Model weights are never bundled; the Models pane handles downloads exactly like `localvoice setup`.

## 8. Testing

- Python: `serve` loop tested with fake engines over in-memory streams — command in, JSON events out; config write + hot-apply routing; schema completeness; latency-event emission from a fake pipeline run. All fast tests.
- Slow: one test drives real `serve --allow-inject` end-to-end (inject fixture WAV → expect `user_text`, `assistant_clause`, `turn_done`).
- Swift: XCTest for message decoding and AppState reduction (event sequences → expected view state). UI tests out of scope for v1.
- CI: existing Python job unchanged; new job runs `xcodebuild -scheme LocalVoice test` on macos-latest arm64. Packaging script is exercised on release tags, not per-push.

## 9. Repository layout

```
src/localvoice/           # engine (+ serve.py, schema.py, timing additions)
app/LocalVoice/           # Xcode project (committed), SwiftUI sources, XCTests
scripts/build_app.sh      # engine bundle + app build + zip
.github/workflows/ci.yml  # + swift job
.github/workflows/release.yml
docs/gui.md               # app architecture, protocol reference, screenshots
```

## 10. Phases

- **A — engine API** (pure Python, TDD): serve loop, protocol events, schema derivation, per-turn latency timestamps, level hook, set_config + hot-apply, model list/download, voice preview, inject hook. Independently shippable; also the API for any future shell.
- **B — the app**: Xcode project, EngineClient, AppState, four views, HotkeyMonitor, dev-checkout mode. Usable end-to-end against `uv run localvoice serve`.
- **C — packaging**: PyInstaller embed, build script, release workflow, docs/screenshots, README update.

## 11. Non-goals (v1)

Menu-bar extra; notarized/signed distribution; iOS/remote clients; conversation persistence across sessions; localization; in-app model benchmarking UI (bench stays a CLI).

## 12. Risks

- PyInstaller × MLX/misaki bundling is the largest unknown → phase C isolates it; dev-checkout mode keeps the app usable regardless; worst case v1 releases document a `uv`-based install while packaging stabilizes.
- Two-language repo raises contributor friction → docs split engine/app contribution paths; engine remains fully usable without Xcode.
- CGEventTap behavior under macOS 26 permission tightening → Setup pane's live checks make failures visible; pynput terminal mode remains the fallback interaction path.
- GIL stalls delay engine events by up to ~300 ms during prefill → UI treats `state` transitions as eventually-consistent; the Swift-side key tap and on-screen button stay responsive regardless.
