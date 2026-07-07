# Tools T2 — mlx-vlm engine (vision-capable Gemma) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The same Gemma 4 26B the user already runs becomes vision-capable by swapping its runtime to mlx-vlm, with text-turn latency parity and all shipped behavior (think translation, tools, prefix reuse) intact — the enabler T3's screenshot tool plugs into.

**Architecture:** Hybrid generation, ranked first by the pre-run spike (spec §11): load once via `mlx_vlm.load` (weights include the vision tower), drive TEXT turns through `mlx_lm.stream_generate` against `model.language_model` via a tiny logits adapter — so the shipped prefix-reuse/think/tools code carries over verbatim — and expose an image-capable path through `mlx_vlm` generation for T3. Fallback if the adapter fails (hard 2-attempt cap): plain `mlx_vlm` streaming for text, accepted only if the bench gate still passes.

**Tech Stack:** mlx-vlm 0.6.4 (`gemma4` support verified), mlx-audio 0.4.3 (verified Kokoro-clean — the SineGen regression starts at 0.4.4), mlx-lm 0.31.3.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-07-localvoice-tools-design.md` §5 + §11 (spike results are binding facts). T3 items (screencapture, Screen Recording card, look_at_screen tool, injection hardening) are OUT of scope.
- Branch: `feat/tools-t2` off `main`. Baselines: Python fast **222** (+5 slow), Swift **90** (no Swift changes expected in T2 — do not run xcodebuild unless you touch `app/`).
- Bench gate (binding, from the user-approved decision): with `[llm] engine = "mlx_vlm"` on gemma-4-26B-A4B, `localvoice bench --runs 3` best voice-to-voice **≤ 1.0 s** and decode within ~15% of the mlx-lm number measured the same session.
- ALL MLX work stays on the single inference thread (the engine is called only from there — same contract as MlxLmEngine; import mlx libraries lazily inside methods, never at module import).
- NEVER change system audio (volume/mute) — the machine stays muted. Live checks use `localvoice serve --allow-inject` + `inject_audio` only.
- The user's overlay (`localvoice.local.toml`) is live user data: only Task 5 touches it (the deliberate engine flip), with a byte-backup + restore-on-failure discipline.
- Never `git add -A`; never commit `localvoice.local.toml` or `.superpowers/`.

## File Structure

```
pyproject.toml / uv.lock              # dep moves (Task 1)
src/localvoice/llm/mlx_vlm_engine.py  # new engine: adapter + hybrid stream (Task 2)
src/localvoice/llm/base.py            # optional image kwarg on the protocol (Task 4)
src/localvoice/engineset.py|app.py    # engine factory registration (Task 3 — find the map)
src/localvoice/schema.py              # llm.engine widget choices note (Task 3)
docs/models.md, docs/gui.md           # engine documentation (Task 3)
docs/latency.md                       # measured mlx-vlm numbers (Task 5)
tests/test_mlx_vlm_engine.py          # fast unit tests (Task 2/4)
tests/test_smoke_slow.py              # slow: real hybrid load/gen/prefix + image (Tasks 2/4)
```

---

### Task 1: Dependency moves

**Files:**
- Modify: `pyproject.toml`, `uv.lock`

**Interfaces:**
- Produces: `mlx-audio==0.4.3`, `mlx-vlm==0.6.4` installed; the `[tool.uv] override-dependencies` block REMOVED if 0.4.3's own constraints allow `mlx-lm==0.31.3` (spike verified they co-resolve; confirm with `uv lock` — if resolution fails without the override, keep it and say why in your report).

- [ ] **Step 1:** In `pyproject.toml`: change `"mlx-audio==0.4.1"` to `"mlx-audio==0.4.3"`; add `"mlx-vlm==0.6.4"`; update the `[tool.uv]` comment block — attempt removal of `override-dependencies` entirely (the pin comment explains it exists only for 0.4.1's exact-pin).
- [ ] **Step 2:** `uv lock && uv sync`. Confirm: `uv pip list | grep -E "^mlx"` shows mlx-audio 0.4.3, mlx-vlm 0.6.4, mlx-lm 0.31.3 (exactly one of each).
- [ ] **Step 3:** Kokoro regression gate — the reason the pin existed. Run the slow suite: `uv run pytest -m slow` → all pass (it synthesizes real audio through Kokoro; the SineGen breakage would fail here). Then the full fast suite → 222.
- [ ] **Step 4:** Boot check with the user's real config: `sleep 45 | uv run localvoice serve 2>/dev/null | grep -m1 engines_ready` → prints the event (Kokoro + Gemma load clean on the new deps).
- [ ] **Step 5:** Commit:

```bash
git add pyproject.toml uv.lock
git commit -m "chore: mlx-audio 0.4.3 (Kokoro-clean) + mlx-vlm 0.6.4, retire mlx-lm override"
```

---

### Task 2: Hybrid engine — logits adapter + text stream parity

**Files:**
- Create: `src/localvoice/llm/mlx_vlm_engine.py`
- Test: `tests/test_mlx_vlm_engine.py` (fast), `tests/test_smoke_slow.py` (slow additions)

**Interfaces:**
- Consumes: `is_channel_style`, `supports_tools`, `common_prefix_len`, `fit_messages` from `src/localvoice/llm/mlx_lm_engine.py` (import — do not duplicate); `ChannelThinkTranslator`; `LlmConfig`.
- Produces: `MlxVlmEngine(cfg: LlmConfig)` with `load()`, `stream(messages, *, think, tools=None)` — behaviorally identical to `MlxLmEngine.stream` for text (same fit/trim/prefix-reuse/think-prepend/translator/budget logic), plus attributes `supports_tools`, `_channel_style`. And `class _TextTowerAdapter` — the load-bearing new code:

```python
class _TextTowerAdapter:
    """Adapt mlx-vlm's LanguageModel to what mlx_lm.stream_generate expects.

    mlx-vlm's tower returns a LanguageModelOutput dataclass; mlx-lm's
    generation loop consumes raw logits arrays. Everything else it touches
    (layers for cache construction, config/args attributes) is forwarded.
    Verified viable in the spec §11 spike: make_prompt_cache(language_model)
    already works (30 layers); only the __call__ return type mismatches.
    """

    def __init__(self, tower) -> None:
        self._tower = tower

    def __call__(self, *args, **kwargs):
        out = self._tower(*args, **kwargs)
        return out.logits if hasattr(out, "logits") else out

    def __getattr__(self, name):
        return getattr(self._tower, name)
```

`load()` shape (lazy imports; mirror MlxLmEngine.load's structure):

```python
def load(self) -> None:
    from mlx_vlm import load as vlm_load

    self._model, self._processor = vlm_load(self._cfg.model)
    self._tokenizer = getattr(self._processor, "tokenizer", self._processor)
    tmpl = getattr(self._tokenizer, "chat_template", None)
    self._channel_style = is_channel_style(tmpl)
    self.supports_tools = supports_tools(tmpl)
    self._lm = _TextTowerAdapter(self._model.language_model)
    self._reset_cache()   # make_prompt_cache(self._model.language_model)
```

`stream()` reuses MlxLmEngine's algorithm with `self._lm` in place of the model and `self._tokenizer` for templating. IMPORTANT extraction discipline: rather than copy the ~60-line body, REFACTOR MlxLmEngine's stream body into a module-level helper both engines call (e.g. `_stream_text(engine_state, messages, think, tools)` taking the model-or-adapter, tokenizer, cache holder, cfg) — the two engines must not drift. Keep MlxLmEngine's public behavior byte-identical (its tests are the guard).

- [ ] **Step 1: Fast failing tests** (`tests/test_mlx_vlm_engine.py`):

```python
import types

from localvoice.llm.mlx_vlm_engine import _TextTowerAdapter


class FakeTower:
    def __init__(self):
        self.layers = ["l1", "l2"]
        self.called_with = None

    def __call__(self, x, **kw):
        self.called_with = (x, kw)
        return types.SimpleNamespace(logits="LOGITS")


def test_adapter_unwraps_language_model_output():
    t = FakeTower()
    assert _TextTowerAdapter(t)("tokens", cache="c") == "LOGITS"
    assert t.called_with == ("tokens", {"cache": "c"})


def test_adapter_passes_raw_returns_through():
    t = FakeTower()
    t.__call__ = None  # replaced below

    class Raw:
        layers = []
        def __call__(self, x, **kw):
            return "RAW_ARRAY"
    assert _TextTowerAdapter(Raw())("x") == "RAW_ARRAY"


def test_adapter_forwards_attributes():
    assert _TextTowerAdapter(FakeTower()).layers == ["l1", "l2"]
```

- [ ] **Step 2:** Run → ImportError. Implement the adapter + engine per the shapes above (including the shared `_stream_text` refactor in `mlx_lm_engine.py`). Run the fast suite: 222 baseline must stay green (MlxLmEngine untouched behaviorally) + your new tests.
- [ ] **Step 3: Slow proof** (append to `tests/test_smoke_slow.py`, follow its existing `@pytest.mark.slow` + skip-if-model-missing style): load `MlxVlmEngine` with the user's Gemma path (skip cleanly if absent); stream a short prompt (`think=False, max_tokens 24`) → non-empty text, no exception; SECOND stream call with the same system prompt + new user message → assert the engine's prefix-reuse path engaged (cache offset non-zero before the call, i.e. `engine._cache[0].offset > 0` — prefix reuse across turns is the hybrid's whole point). With `think=True` → reasoning arrives translated (`<think>` in the raw joined stream).
- [ ] **Step 4:** Run: `uv run pytest -m slow` → all green (including Task 1's Kokoro coverage). Fallback rule: if the adapter path fails and one focused fix attempt doesn't hold, STOP — implement plain `mlx_vlm.stream_generate` for text instead, note the prefix-reuse loss in the report, and Task 5's bench gate decides. Do not iterate past 2 attempts.
- [ ] **Step 5:** Commit: `git add src/localvoice/llm/mlx_vlm_engine.py src/localvoice/llm/mlx_lm_engine.py tests/test_mlx_vlm_engine.py tests/test_smoke_slow.py && git commit -m "feat(llm): mlx-vlm hybrid engine with text-tower adapter"`

---

### Task 3: Engine registration + docs

**Files:**
- Modify: the engine factory (grep `mlx_lm` construction — `src/localvoice/engineset.py` and/or `src/localvoice/app.py`/`__main__.py` factories), `src/localvoice/schema.py` (llm.engine help text/choices), `docs/models.md`, `docs/gui.md`
- Test: `tests/test_engineset.py`

**Interfaces:**
- Consumes: `MlxVlmEngine` (Task 2).
- Produces: `[llm] engine = "mlx_vlm"` constructs `MlxVlmEngine` everywhere engines are built (serve boot, reloads, terminal, bench); changing `llm.engine` via set_config triggers an LLM reload (verify it's already in the reload-keys routing — `*.engine` is per the original hot-apply table; add a test if untested).

- [ ] **Step 1:** Failing test in `tests/test_engineset.py` (follow its factory-test style): building with `engine="mlx_vlm"` yields an `MlxVlmEngine` instance; unknown engine string still raises the existing ConfigError.
- [ ] **Step 2:** Implement registration; update `schema.py`'s llm.engine help to name both engines (`"mlx_lm (text) or mlx_vlm (text + vision)"`); docs/models.md gains a short "Engines" paragraph (mlx_vlm requires the model to keep its vision tower for image input; text-only models also run on it); docs/gui.md hot-apply table already routes `*.engine` to reload — add "(mlx_lm | mlx_vlm)" to the row.
- [ ] **Step 3:** Fast suite green; commit `feat(llm): register mlx_vlm engine selection`.

---

### Task 4: Minimal image entry point (T3's hook)

**Files:**
- Modify: `src/localvoice/llm/base.py`, `src/localvoice/llm/mlx_vlm_engine.py`, `src/localvoice/llm/mlx_lm_engine.py`
- Test: `tests/test_mlx_vlm_engine.py` (fast), `tests/test_smoke_slow.py` (slow)

**Interfaces:**
- Produces: `stream(messages, *, think, tools=None, image_path: str | None = None)` on the protocol. `MlxLmEngine` with `image_path` → raise `ValueError("image input requires the mlx_vlm engine")` (honest, tested). `MlxVlmEngine` with `image_path` → template via mlx-vlm's image-message convention (use `mlx_vlm.prompt_utils.apply_chat_template` or the library's documented image flow — read the installed 0.6.4 source for the exact call; the image turn goes through `mlx_vlm`'s own generation, NOT the text adapter) and stream the description. Image turns may skip prefix reuse (full prefill) — acceptable; they're rare.

- [ ] **Step 1:** Fast failing tests: MlxLmEngine raises on image_path (construct with cfg, monkeypatch nothing — the raise must happen before any model access); fakes updated so FakeLLM accepts and records image_path (default None) with existing tests unmodified.
- [ ] **Step 2:** Implement. Fast suite green.
- [ ] **Step 3:** Slow test: generate a 64×64 solid-red PNG into tmp (pure-python: `struct`/`zlib` writer or `numpy` + a tiny PNG encoder — no new deps), then `MlxVlmEngine.stream([{"role":"user","content":"What color is this image? One word."}], think=False, image_path=str(png))` → joined output contains "red" case-insensitively. Skip cleanly if the model dir is absent.
- [ ] **Step 4:** `uv run pytest -m slow` green; commit `feat(llm): image entry point through the vlm engine`.

---

### Task 5: Bench gate + engine flip + acceptance

**Files:**
- Modify: `docs/latency.md` (add the measured mlx-vlm column/note), `localvoice.local.toml` (the flip — NOT committed)

- [ ] **Step 1:** Bench BOTH engines same session on the user's Gemma (`--runs 3` each): `uv run localvoice bench --model <gemma path> --runs 3` (mlx_lm baseline), then flip `[llm] engine = "mlx_vlm"` in a TEMP config overlay copy (or `--config` with a scratch toml mirroring the user's values + engine=mlx_vlm) and bench again. GATE: mlx_vlm best voice-to-voice ≤ 1.0 s AND within ~15% of the mlx_lm number. If the gate FAILS: restore everything, report BLOCKED with the numbers — the fallback decision (accept re-prefill / stay mlx_lm default) is the controller's, not yours.
- [ ] **Step 2:** On gate pass: byte-backup `localvoice.local.toml`, set `[llm] engine = "mlx_vlm"` in it (keep every other value). Boot `sleep 60 | uv run localvoice serve 2>/dev/null` → `ready` shows the engine, `engines_ready` arrives.
- [ ] **Step 3:** Live turn via serve `--allow-inject` + `inject_audio` (NO system audio changes — machine stays muted): a normal question → spoken-pipeline completes (user_text/assistant_clause/turn_done on the stream); then a search-triggering question → tool_call/tool_result events + answer (tools still work on the vlm engine). With think=true in the user's overlay, reasoning events must appear (translator parity).
- [ ] **Step 4:** Update `docs/latency.md` with the measured mlx-vlm numbers (new subsection, same table style). Fast suite once more (222+new). Commit docs: `docs(latency): mlx-vlm engine measurements`.
- [ ] **Step 5:** Report includes: both bench tables, the gate verdict, serve/inject evidence, confirmation the overlay flip is in place (and its backup path), and a human checklist (voice feel on the new engine; anything sounding different).

---

## Self-review notes (completed at plan time)

- Spec coverage: §5 engine + selection + spike gate → Tasks 2/3/5 (spike itself pre-run, §11); prompt-cache question resolved by the hybrid design (Task 2 slow test pins reuse); dependency resolution → Task 1; the §5 "default flip for Gemma if adopted" is the user-overlay flip in Task 5 (the committed default stays Qwen/mlx_lm — shipped default model has no tool/vision template support anyway, per T1's README note).
- Placeholders: Task 4 directs reading the installed mlx-vlm 0.6.4 source for the exact image-templating call rather than quoting an API I have not verified — deliberate: the spike verified load/generate/language_model, not the image-message helper; the task carries the acceptance test that pins whatever call is used.
- Type consistency: `stream(messages, *, think, tools=None, image_path=None)` introduced in Task 4 matches Task 2's signature plus one kwarg; `_TextTowerAdapter` names match between Task 2's code and tests; engine string `"mlx_vlm"` consistent across Tasks 3/5.
