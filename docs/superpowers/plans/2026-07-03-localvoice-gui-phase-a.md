# LocalVoice GUI Phase A (Engine API) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the engine a `localvoice serve` mode speaking the GUI protocol: JSON-lines events/commands, derived settings schema, per-turn latency, mic levels, hot-apply config, model scan/download, voice preview, and a test inject hook.

**Architecture:** New modules `schema.py`, `overlay.py`, `engineset.py`, `modelstore.py`, `serve.py` plus small hooks in `pipeline.py` (metrics), `audio/capture.py` (level callback), and `app.py` (state callback). `Serve` owns stdout (real stdout is captured for protocol; `sys.stdout` is redirected to stderr so library prints can never corrupt the protocol), reads commands on the main thread, and reuses the existing single-inference-thread rule for loads/reloads/preview.

**Tech Stack:** Python 3.12, stdlib json/threading, `tomli-w` (new dep, TOML writing), existing engines/orchestrator/pipeline, pytest with the existing fakes.

Spec: `docs/superpowers/specs/2026-07-03-localvoice-gui-design.md` (§3 protocol, §4 schema, §5 hot-apply are binding).

## Global Constraints

- New runtime dep: `tomli-w>=1.0` (plan amendment to the v1 dependency list). No other additions.
- Protocol event/command names and payload keys exactly as spec §3; every engine→app message includes `"event"`, every app→engine message includes `"cmd"`. `ready` carries `"version": 1`.
- Semantics refinement (documented here, applies to spec §3): `ready` is emitted immediately at serve start (config + schema); engine loads run async on the inference executor emitting `load_progress`, followed by `{"event": "engines_ready"}`. PTT commands before `engines_ready` produce an `error` event. If models are missing at startup, serve emits `error` and skips that engine's load; a later `set_config` re-pointing (or re-setting) the model triggers the load.
- stdout purity is mandatory: `Serve.run()` snapshots the real stdout for `emit()` and sets `sys.stdout = sys.stderr` before anything else.
- Terminal `localvoice run` behavior is unchanged. All mlx imports stay lazy. Ruff (incl. E501/100) clean and `pytest -m "not slow" -q` green at every commit (baseline 106). Conventional commits. No emoji.
- All engine loads/reloads/preview synthesis run on the single inference executor — never on the command loop thread.

---

### Task 1: Per-turn latency metrics from the pipeline

**Files:**
- Modify: `src/localvoice/pipeline.py`
- Test: `tests/test_pipeline.py` (append)

**Interfaces:**
- Consumes: existing `PipelineDeps`, `run_pipeline`.
- Produces: `PipelineDeps.on_metrics: Callable[[dict], None] = field(default=lambda m: None)`. Called exactly once per successful response, right after `player.mark_end()`, with `{"stt": s, "ttft": s, "first_clause": s, "tts_first": s, "total": s}` (floats, seconds): `stt` = transcribe duration; `ttft` = STT-done → first LLM delta; `first_clause` = first delta → first speak() with non-empty text; `tts_first` = first clause → first player.submit; `total` = pipeline entry → first player.submit. NOT called on discard paths, cancel, zero-clause responses, or errors.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_pipeline.py`):

```python
def test_metrics_emitted_once_with_expected_keys():
    metrics: list[dict] = []
    deps = make_deps()
    deps.on_metrics = metrics.append
    run(deps, speech())
    assert len(metrics) == 1
    m = metrics[0]
    assert set(m) == {"stt", "ttft", "first_clause", "tts_first", "total"}
    assert all(isinstance(v, float) and v >= 0.0 for v in m.values())
    assert m["total"] >= m["stt"]


def test_metrics_not_emitted_on_quiet_discard_or_empty_output():
    metrics: list[dict] = []
    deps = make_deps()
    deps.on_metrics = metrics.append
    run(deps, np.zeros(SR, np.float32))
    deps2 = make_deps(llm=FakeLLM([]))
    deps2.on_metrics = metrics.append
    run(deps2, speech())
    assert metrics == []


def test_metrics_not_emitted_on_cancel():
    metrics: list[dict] = []
    cancel = threading.Event()
    cancel.set()
    deps = make_deps()
    deps.on_metrics = metrics.append
    run(deps, speech(), cancel)
    assert metrics == []
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_pipeline.py -q` — FAIL: `PipelineDeps` has no field `on_metrics`.

- [ ] **Step 3: Implement.** In `PipelineDeps` add after `on_thinking`:

```python
    on_metrics: Callable[[dict], None] = field(default=lambda m: None)
```

In `run_pipeline`: `t0 = time.perf_counter()` as the first line of the `try`; after `user_text = deps.stt.transcribe(...)` record `t_stt = time.perf_counter()`; add `import time` to the module imports. Track first-delta and first-clause times with nonlocal-style locals: initialize `t_first_token = t_first_clause = t_first_submit = None` before the loop; in the delta loop set `t_first_token = t_first_token or time.perf_counter()` as the first statement of each iteration; in `speak()` after the empty-check set `nonlocal`-tracked `t_first_clause = t_first_clause or time.perf_counter()`; right after the first `deps.player.submit(...)` (where `first_audio_sent` flips) set `t_first_submit = time.perf_counter()`. After `deps.player.mark_end()` and the existing zero-clause branch, add:

```python
        if spoke_anything and t_first_submit is not None:
            deps.on_metrics(
                {
                    "stt": t_stt - t0,
                    "ttft": (t_first_token or t_stt) - t_stt,
                    "first_clause": (t_first_clause or t_first_token or t_stt)
                    - (t_first_token or t_stt),
                    "tts_first": t_first_submit - (t_first_clause or t_first_submit),
                    "total": t_first_submit - t0,
                }
            )
```

(Declare the tracked names in the enclosing scope and add them to `speak`'s `nonlocal` list.)

- [ ] **Step 4: Verify green** — `uv run pytest tests/test_pipeline.py -q` then full gates.
- [ ] **Step 5: Commit** — `git add src/localvoice/pipeline.py tests/test_pipeline.py && git commit -m "feat: per-turn latency metrics from the pipeline"`

---

### Task 2: Mic level callback on GatedBuffer

**Files:**
- Modify: `src/localvoice/audio/capture.py`
- Test: `tests/test_capture.py` (append)

**Interfaces:**
- Produces: `GatedBuffer(max_seconds=300.0, sr=16000, on_level: Callable[[float], None] | None = None)`; while armed, every `write()` also calls `on_level(rms(chunk))` with the chunk actually retained. `MicCapture.__init__(cfg, on_level=None)` forwards it. No throttling here (serve throttles).

- [ ] **Step 1: Failing tests** (append):

```python
def test_on_level_fires_only_while_armed():
    levels: list[float] = []
    b = GatedBuffer(on_level=levels.append)
    b.write(np.full(160, 0.5, np.float32))
    assert levels == []
    b.arm()
    b.write(np.full(160, 0.5, np.float32))
    assert len(levels) == 1 and abs(levels[0] - 0.5) < 1e-6
    b.disarm()
    b.write(np.full(160, 0.5, np.float32))
    assert len(levels) == 1
```

- [ ] **Step 2: Verify fail** — TypeError: unexpected keyword `on_level`.
- [ ] **Step 3: Implement** — store `self._on_level = on_level`; in `write()` after appending the retained `chunk`, still under the armed branch: compute level outside the lock is unsafe for ordering but fine for value — call after releasing the lock: restructure `write` to remember `retained = chunk if armed else None` inside the lock, then after the `with` block: `if retained is not None and self._on_level is not None: self._on_level(rms(retained))`. `MicCapture.__init__(self, cfg, on_level=None)` passes through to `GatedBuffer`.
- [ ] **Step 4: Gates.**
- [ ] **Step 5: Commit** — `feat: mic level callback on gated capture buffer`

---

### Task 3: Orchestrator state callback

**Files:**
- Modify: `src/localvoice/app.py`
- Test: `tests/test_app.py` (append)

**Interfaces:**
- Produces: `Orchestrator(..., on_state: Callable[[S], None] | None = None)`; `_show_state` additionally calls `on_state(self.state)` when set. Status printing unchanged.

- [ ] **Step 1: Failing test** (append):

```python
def test_on_state_callback_receives_transitions():
    seen: list = []
    capture, player, transcript = FakeCapture(), FakePlayer(), Transcript("sys")
    orch = Orchestrator(
        capture=capture, player=player, stt=FakeSTT("hi"), llm=FakeLLM(["Hello there."]),
        tts=FakeTTS(), transcript=transcript, keys_cfg=KeysConfig(),
        status=lambda s: None, on_state=seen.append,
    )
    orch.handle(Event(E.PTT_DOWN))
    assert seen[-1] is S.LISTENING
    orch.handle(Event(E.ESC))
    assert seen[-1] is S.IDLE
```

- [ ] **Step 2: Verify fail.** — TypeError.
- [ ] **Step 3: Implement** — ctor param `on_state: Callable[[S], None] | None = None`, store, and in `_show_state` after the status call: `if self._on_state is not None: self._on_state(self.state)`.
- [ ] **Step 4: Gates.**
- [ ] **Step 5: Commit** — `feat: orchestrator state callback for GUI protocol`

---

### Task 4: Derived settings schema

**Files:**
- Create: `src/localvoice/schema.py`
- Test: `tests/test_schema.py`

**Interfaces:**
- Produces:
  - `build_schema(cfg: Config) -> list[dict]` — one descriptor per field of every config section, keys: `key` ("section.field"), `type` ("bool"|"int"|"float"|"str"), `value`, `default`, `section` (human label), `label`, `help`, `widget`, and optional `minimum`/`maximum`/`step` when the annotation defines them.
  - `coerce(key: str, raw: object) -> object` — converts a JSON-decoded value to the field's Python type ("true"/True→bool, "8192"/8192→int, etc.); raises `ConfigError` naming the key for unknown keys or bad values.
- Consumes: `Config` dataclasses from `localvoice.config`.

- [ ] **Step 1: Failing tests** — `tests/test_schema.py`:

```python
from dataclasses import fields

import pytest

from localvoice.config import (
    AudioConfig,
    Config,
    ConfigError,
    KeysConfig,
    LlmConfig,
    SttConfig,
    TtsConfig,
)
from localvoice.schema import build_schema, coerce


def make_cfg() -> Config:
    return Config(SttConfig(), LlmConfig(), TtsConfig(), KeysConfig(), AudioConfig())


def test_schema_covers_every_config_field_exactly_once():
    descriptors = build_schema(make_cfg())
    keys = [d["key"] for d in descriptors]
    expected = {
        f"{section}.{f.name}"
        for section, cls in (
            ("stt", SttConfig), ("llm", LlmConfig), ("tts", TtsConfig),
            ("keys", KeysConfig), ("audio", AudioConfig),
        )
        for f in fields(cls)
    }
    assert set(keys) == expected and len(keys) == len(set(keys))


def test_descriptor_shape_and_annotations():
    d = {x["key"]: x for x in build_schema(make_cfg())}
    assert d["llm.think"]["widget"] == "toggle" and d["llm.think"]["type"] == "bool"
    assert d["llm.model"]["widget"] == "model_picker"
    assert d["tts.voice"]["widget"] == "voice_picker"
    assert d["keys.ptt"]["widget"] == "key_capture"
    assert d["audio.input_device"]["widget"] == "device_picker"
    assert d["tts.speed"]["widget"] == "slider" and d["tts.speed"]["minimum"] == 0.5
    assert d["audio.rebuffer_ms"]["widget"] == "slider"
    assert d["llm.max_tokens"]["widget"] == "number"
    assert d["llm.system_prompt"]["widget"] == "text"
    assert d["llm.think"]["section"] == "Language model"
    assert all(x["help"] is not None for x in build_schema(make_cfg()))


def test_values_reflect_config_instance():
    cfg = make_cfg()
    cfg.llm.max_tokens = 42
    d = {x["key"]: x for x in build_schema(cfg)}
    assert d["llm.max_tokens"]["value"] == 42 and d["llm.max_tokens"]["default"] == 1024


@pytest.mark.parametrize(
    "key,raw,expected",
    [
        ("llm.think", True, True),
        ("llm.think", "true", True),
        ("llm.think", "false", False),
        ("llm.max_tokens", "2048", 2048),
        ("llm.max_tokens", 2048, 2048),
        ("tts.speed", "1.2", 1.2),
        ("tts.voice", "af_bella", "af_bella"),
    ],
)
def test_coerce_valid(key, raw, expected):
    assert coerce(key, raw) == expected


def test_coerce_rejects_unknown_key_and_bad_value():
    with pytest.raises(ConfigError, match="nope.key"):
        coerce("nope.key", 1)
    with pytest.raises(ConfigError, match="llm.max_tokens"):
        coerce("llm.max_tokens", "not-a-number")
```

- [ ] **Step 2: Verify fail** — ModuleNotFoundError.
- [ ] **Step 3: Implement** — `src/localvoice/schema.py`:

```python
from dataclasses import fields
from typing import get_type_hints

from localvoice.config import (
    AudioConfig,
    Config,
    ConfigError,
    KeysConfig,
    LlmConfig,
    SttConfig,
    TtsConfig,
)

_SECTIONS = [
    ("stt", SttConfig, "Speech recognition"),
    ("llm", LlmConfig, "Language model"),
    ("tts", TtsConfig, "Speech"),
    ("keys", KeysConfig, "Keys"),
    ("audio", AudioConfig, "Audio"),
]

_TYPE_NAMES = {bool: "bool", int: "int", float: "float", str: "str"}
_DEFAULT_WIDGETS = {bool: "toggle", int: "number", float: "number", str: "text"}

_ANNOTATIONS: dict[str, dict] = {
    "stt.model": {"widget": "model_picker", "help": "HF repo id or local MLX folder."},
    "llm.model": {"widget": "model_picker", "help": "HF repo id or local MLX folder."},
    "llm.deep_model": {"widget": "model_picker", "help": "Used with --deep; empty disables it."},
    "llm.think": {"label": "Thinking mode", "help": "Silent reasoning; shown, never spoken."},
    "llm.system_prompt": {"help": "System message prepended to every conversation."},
    "tts.model": {"widget": "model_picker", "help": "Kokoro weights repo or folder."},
    "tts.voice": {"widget": "voice_picker", "help": "Kokoro voice preset."},
    "tts.speed": {
        "widget": "slider", "minimum": 0.5, "maximum": 2.0, "step": 0.1,
        "help": "Speaking-rate multiplier.",
    },
    "keys.ptt": {"widget": "key_capture", "label": "Push-to-talk key"},
    "keys.stop": {"widget": "key_capture", "label": "Stop key"},
    "keys.debounce_ms": {"help": "Presses shorter than this are ignored."},
    "audio.input_device": {"widget": "device_picker", "help": "Empty uses the system default."},
    "audio.output_device": {"widget": "device_picker", "help": "Empty uses the system default."},
    "audio.rebuffer_ms": {
        "widget": "slider", "minimum": 100, "maximum": 800, "step": 50,
        "help": "Anti-stutter gate after a mid-response stall.",
    },
}


def _field_type(cls, name: str) -> type:
    hints = get_type_hints(cls)
    t = hints[name]
    if t not in _TYPE_NAMES:
        raise ConfigError(f"unsupported config field type for {name}: {t}")
    return t


def build_schema(cfg: Config) -> list[dict]:
    out: list[dict] = []
    for section, cls, section_label in _SECTIONS:
        live = getattr(cfg, section)
        defaults = cls()
        for f in fields(cls):
            key = f"{section}.{f.name}"
            t = _field_type(cls, f.name)
            d = {
                "key": key,
                "type": _TYPE_NAMES[t],
                "value": getattr(live, f.name),
                "default": getattr(defaults, f.name),
                "section": section_label,
                "label": f.name.replace("_", " "),
                "help": "",
                "widget": _DEFAULT_WIDGETS[t],
            }
            d.update(_ANNOTATIONS.get(key, {}))
            out.append(d)
    return out


def _lookup(key: str) -> tuple[str, type, str]:
    try:
        section, name = key.split(".", 1)
    except ValueError as exc:
        raise ConfigError(f"unknown config key: {key}") from exc
    for sec, cls, _ in _SECTIONS:
        if sec == section and name in {f.name for f in fields(cls)}:
            return section, _field_type(cls, name), name
    raise ConfigError(f"unknown config key: {key}")


def coerce(key: str, raw: object) -> object:
    _, t, _ = _lookup(key)
    try:
        if t is bool:
            if isinstance(raw, bool):
                return raw
            if isinstance(raw, str) and raw.lower() in ("true", "false"):
                return raw.lower() == "true"
            raise ValueError(raw)
        return t(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"bad value for {key}: {raw!r}") from exc
```

- [ ] **Step 4: Gates.**
- [ ] **Step 5: Commit** — `feat: derived settings schema with annotations and coercion`

---

### Task 5: Overlay writer (tomli-w)

**Files:**
- Modify: `pyproject.toml` (add `"tomli-w>=1.0"` to dependencies)
- Create: `src/localvoice/overlay.py`
- Test: `tests/test_overlay.py`

**Interfaces:**
- Produces: `apply_overlay_changes(config_path: Path, changes: dict[str, object]) -> Path` — loads `<stem>.local.toml` beside `config_path` (empty dict if absent), deep-merges dotted keys (`"llm.think": True` → `{"llm": {"think": True}}`), writes atomically (write `.tmp`, `os.replace`), returns the overlay path. Values must already be coerced.

- [ ] **Step 1: Failing tests**:

```python
import tomllib
from pathlib import Path

from localvoice.overlay import apply_overlay_changes


def test_creates_overlay_and_nests_dotted_keys(tmp_path: Path):
    cfg = tmp_path / "localvoice.toml"
    cfg.write_text("")
    p = apply_overlay_changes(cfg, {"llm.think": True, "tts.speed": 1.2})
    data = tomllib.loads(p.read_text())
    assert p.name == "localvoice.local.toml"
    assert data == {"llm": {"think": True}, "tts": {"speed": 1.2}}


def test_preserves_unrelated_existing_keys(tmp_path: Path):
    cfg = tmp_path / "localvoice.toml"
    cfg.write_text("")
    (tmp_path / "localvoice.local.toml").write_text('[llm]\nmodel = "keep/me"\n')
    apply_overlay_changes(cfg, {"llm.think": False})
    data = tomllib.loads((tmp_path / "localvoice.local.toml").read_text())
    assert data["llm"] == {"model": "keep/me", "think": False}
```

- [ ] **Step 2: Verify fail.**
- [ ] **Step 3: Implement**:

```python
import os
import tomllib
from pathlib import Path

import tomli_w


def apply_overlay_changes(config_path: Path, changes: dict[str, object]) -> Path:
    overlay = config_path.with_name(config_path.stem + ".local" + config_path.suffix)
    data: dict = tomllib.loads(overlay.read_text()) if overlay.exists() else {}
    for dotted, value in changes.items():
        section, name = dotted.split(".", 1)
        data.setdefault(section, {})[name] = value
    tmp = overlay.with_suffix(".tmp")
    tmp.write_bytes(tomli_w.dumps(data).encode())
    os.replace(tmp, overlay)
    return overlay
```

Run `uv add "tomli-w>=1.0"` (updates pyproject + lock).

- [ ] **Step 4: Gates.**
- [ ] **Step 5: Commit** — `feat: atomic local-overlay writer` (include `pyproject.toml` and `uv.lock`).

---

### Task 6: Hot-apply router and EngineSet

**Files:**
- Create: `src/localvoice/engineset.py`
- Test: `tests/test_engineset.py`

**Interfaces:**
- Produces:
  - `plan_apply(changes: dict[str, object]) -> dict` — pure; returns `{"reload": [engine names], "restart_audio": bool, "instant": [keys]}` per spec §5: `stt.model`/`stt.engine`→reload "stt"; `llm.model`/`llm.engine` (and `llm.deep_model`)→reload "llm"; `tts.model`/`tts.engine`→reload "tts"; `audio.*`→restart_audio; everything else→instant. Reload list is deduped and ordered stt,llm,tts.
  - `class EngineProxy:` `__init__(self, target)`, `__getattr__` delegates to `self._target`; `swap(new)` replaces it.
  - `class EngineSet:` `__init__(self, cfg: Config, factories: dict | None = None)` builds engines via factories (default: the three real engine classes, imported lazily inside `__init__`); exposes `.stt`, `.llm`, `.tts` as `EngineProxy`s. `load_all(on_progress)` and `reload(name: str, cfg: Config, on_progress)` construct/load synchronously on the CALLING thread (serve submits them to the inference executor) and `swap` the proxy on success; `on_progress(name, phase, seconds)` fires with phase "start" then "done" (seconds set on done). On load failure the old engine stays and the exception propagates.

- [ ] **Step 1: Failing tests**:

```python
import pytest

from localvoice.engineset import EngineSet, plan_apply


def test_plan_apply_routes_per_spec():
    plan = plan_apply({"llm.model": "x", "tts.voice": "af_bella", "audio.rebuffer_ms": 400})
    assert plan == {"reload": ["llm"], "restart_audio": True, "instant": ["tts.voice"]}
    assert plan_apply({"stt.model": "a", "tts.model": "b"})["reload"] == ["stt", "tts"]
    assert plan_apply({"llm.think": True}) == {
        "reload": [], "restart_audio": False, "instant": ["llm.think"],
    }


class Fake:
    def __init__(self, cfg):
        self.cfg = cfg
        self.loaded = False

    def load(self):
        self.loaded = True


class Boom(Fake):
    def load(self):
        raise RuntimeError("no")


def make_set(llm_factory=Fake):
    from localvoice.config import AudioConfig, Config, KeysConfig, LlmConfig, SttConfig, TtsConfig

    cfg = Config(SttConfig(), LlmConfig(), TtsConfig(), KeysConfig(), AudioConfig())
    factories = {"stt": Fake, "llm": llm_factory, "tts": Fake}
    return cfg, EngineSet(cfg, factories=factories)


def test_load_all_reports_progress_and_loads():
    cfg, es = make_set()
    events: list = []
    es.load_all(lambda n, p, s: events.append((n, p)))
    assert es.llm.loaded and es.stt.loaded and es.tts.loaded
    assert ("llm", "start") in events and ("llm", "done") in events


def test_reload_swaps_proxy_identity_preserved():
    cfg, es = make_set()
    es.load_all(lambda *a: None)
    proxy = es.llm
    old = proxy._target
    es.reload("llm", cfg, lambda *a: None)
    assert es.llm is proxy and proxy._target is not old and proxy.loaded


def test_reload_failure_keeps_old_engine():
    cfg, es = make_set()
    es.load_all(lambda *a: None)
    old = es.llm._target
    es._factories["llm"] = Boom
    with pytest.raises(RuntimeError):
        es.reload("llm", cfg, lambda *a: None)
    assert es.llm._target is old
```

- [ ] **Step 2: Verify fail.**
- [ ] **Step 3: Implement**:

```python
import time
from collections.abc import Callable

from localvoice.config import Config

_RELOAD_KEYS = {
    "stt.model": "stt", "stt.engine": "stt",
    "llm.model": "llm", "llm.engine": "llm", "llm.deep_model": "llm",
    "tts.model": "tts", "tts.engine": "tts",
}
_ORDER = ["stt", "llm", "tts"]


def plan_apply(changes: dict[str, object]) -> dict:
    reload = {name for key, name in _RELOAD_KEYS.items() if key in changes}
    restart_audio = any(k.startswith("audio.") for k in changes)
    instant = [
        k for k in changes
        if k not in _RELOAD_KEYS and not k.startswith("audio.")
    ]
    return {
        "reload": [n for n in _ORDER if n in reload],
        "restart_audio": restart_audio,
        "instant": instant,
    }


class EngineProxy:
    def __init__(self, target) -> None:
        object.__setattr__(self, "_target", target)

    def swap(self, new) -> None:
        object.__setattr__(self, "_target", new)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_target"), name)


def _default_factories() -> dict:
    from localvoice.llm.mlx_lm_engine import MlxLmEngine
    from localvoice.stt.whisper_mlx import WhisperMlxEngine
    from localvoice.tts.kokoro_mlx import KokoroMlxEngine

    return {"stt": WhisperMlxEngine, "llm": MlxLmEngine, "tts": KokoroMlxEngine}


class EngineSet:
    def __init__(self, cfg: Config, factories: dict | None = None) -> None:
        self._factories = factories or _default_factories()
        self.stt = EngineProxy(self._factories["stt"](cfg.stt))
        self.llm = EngineProxy(self._factories["llm"](cfg.llm))
        self.tts = EngineProxy(self._factories["tts"](cfg.tts))

    def _cfg_for(self, name: str, cfg: Config):
        return {"stt": cfg.stt, "llm": cfg.llm, "tts": cfg.tts}[name]

    def _load(self, name: str, engine, on_progress: Callable) -> None:
        on_progress(name, "start", None)
        t0 = time.perf_counter()
        engine.load()
        on_progress(name, "done", time.perf_counter() - t0)

    def load_all(self, on_progress: Callable) -> None:
        for name in _ORDER:
            self._load(name, getattr(self, name)._target, on_progress)

    def reload(self, name: str, cfg: Config, on_progress: Callable) -> None:
        new = self._factories[name](self._cfg_for(name, cfg))
        self._load(name, new, on_progress)
        getattr(self, name).swap(new)
```

Note: `EngineProxy` satisfies the engine protocols structurally (all attribute access delegates), so the orchestrator/pipeline take proxies without change.

- [ ] **Step 4: Gates.**
- [ ] **Step 5: Commit** — `feat: hot-apply router and reloadable engine set`

---

### Task 7: Model store (scan + size math)

**Files:**
- Create: `src/localvoice/modelstore.py`
- Test: `tests/test_modelstore.py`

**Interfaces:**
- Produces:
  - `scan_models(lmstudio_root: Path | None = None) -> list[dict]` — entries `{"id": "publisher/name", "path": str, "size_gb": float (1 decimal), "kind": "mlx"|"gguf"|"other"}` from `<root>/*/*` directories (default root `~/.lmstudio/models`); kind = "mlx" if any `*.safetensors` inside, "gguf" if any `*.gguf`, else "other". Missing root → []. Additionally appends HF-cache entries via `huggingface_hub.scan_cache_dir()` wrapped in try/except (best effort, kind "mlx", id = repo_id, path = cache dir, size from the scan) — deduped by id, lmstudio entries win.
  - `download_pct(local_bytes: int, total_bytes: int | None) -> float | None` — None when total unknown, else `min(100.0, round(local/total*100, 1))`.
  - `dir_bytes(path: Path) -> int` — recursive size.
  - `download(repo: str, dest_hint: None = None, on_pct: Callable[[float | None], None] = ..., poll_s: float = 1.0) -> None` — runs `snapshot_download(repo_id=repo)` on the calling thread while a poller thread emits `download_pct(dir_bytes(cache_dir), remote_total)` every `poll_s`; remote total from `HfApi().model_info(repo, files_metadata=True)` sibling sizes (None on failure); always ends with `on_pct(100.0)` after the download returns. (Serve runs this on a worker thread.)

- [ ] **Step 1: Failing tests**:

```python
from pathlib import Path

from localvoice.modelstore import dir_bytes, download_pct, scan_models


def make_tree(tmp_path: Path) -> Path:
    root = tmp_path / "models"
    a = root / "pub" / "ModelA-MLX"
    a.mkdir(parents=True)
    (a / "model.safetensors").write_bytes(b"x" * 2048)
    b = root / "pub" / "ModelB-GGUF"
    b.mkdir(parents=True)
    (b / "weights.gguf").write_bytes(b"y" * 1024)
    return root


def test_scan_models_kinds_and_sizes(tmp_path):
    entries = {e["id"]: e for e in scan_models(lmstudio_root=make_tree(tmp_path))}
    assert entries["pub/ModelA-MLX"]["kind"] == "mlx"
    assert entries["pub/ModelB-GGUF"]["kind"] == "gguf"
    assert entries["pub/ModelA-MLX"]["size_gb"] == 0.0  # tiny fixture rounds to 0.0
    assert Path(entries["pub/ModelA-MLX"]["path"]).exists()


def test_scan_models_missing_root(tmp_path):
    assert scan_models(lmstudio_root=tmp_path / "nope") == []


def test_download_pct_math():
    assert download_pct(50, 200) == 25.0
    assert download_pct(300, 200) == 100.0
    assert download_pct(10, None) is None


def test_dir_bytes(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "f").write_bytes(b"z" * 10)
    (tmp_path / "g").write_bytes(b"z" * 5)
    assert dir_bytes(tmp_path) == 15
```

- [ ] **Step 2: Verify fail.**
- [ ] **Step 3: Implement** (HF-cache portion and `download` are thin/lazy; only the pure parts above are unit-gated):

```python
import threading
import time
from collections.abc import Callable
from pathlib import Path


def dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def download_pct(local_bytes: int, total_bytes: int | None) -> float | None:
    if not total_bytes:
        return None
    return min(100.0, round(local_bytes / total_bytes * 100, 1))


def _kind(path: Path) -> str:
    if any(path.rglob("*.safetensors")):
        return "mlx"
    if any(path.rglob("*.gguf")):
        return "gguf"
    return "other"


def scan_models(lmstudio_root: Path | None = None) -> list[dict]:
    root = Path(lmstudio_root or Path.home() / ".lmstudio/models")
    out: dict[str, dict] = {}
    if root.exists():
        for pub in sorted(p for p in root.iterdir() if p.is_dir()):
            for model in sorted(m for m in pub.iterdir() if m.is_dir()):
                mid = f"{pub.name}/{model.name}"
                out[mid] = {
                    "id": mid,
                    "path": str(model),
                    "size_gb": round(dir_bytes(model) / 1e9, 1),
                    "kind": _kind(model),
                }
    try:  # best-effort HF cache listing
        from huggingface_hub import scan_cache_dir

        for repo in scan_cache_dir().repos:
            if repo.repo_type == "model" and repo.repo_id not in out:
                out[repo.repo_id] = {
                    "id": repo.repo_id,
                    "path": str(repo.repo_path),
                    "size_gb": round(repo.size_on_disk / 1e9, 1),
                    "kind": "mlx",
                }
    except Exception:  # noqa: BLE001 — cache scan is optional
        pass
    return list(out.values())


def _remote_total(repo: str) -> int | None:
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo, files_metadata=True)
        sizes = [s.size for s in info.siblings if s.size]
        return sum(sizes) or None
    except Exception:  # noqa: BLE001 — progress falls back to indeterminate
        return None


def download(
    repo: str,
    on_pct: Callable[[float | None], None],
    poll_s: float = 1.0,
) -> None:
    from huggingface_hub import snapshot_download

    total = _remote_total(repo)
    stop = threading.Event()

    def poller() -> None:
        from huggingface_hub.constants import HF_HUB_CACHE

        repo_dir = Path(HF_HUB_CACHE) / ("models--" + repo.replace("/", "--"))
        while not stop.is_set():
            if repo_dir.exists():
                on_pct(download_pct(dir_bytes(repo_dir), total))
            time.sleep(poll_s)

    t = threading.Thread(target=poller, daemon=True)
    t.start()
    try:
        snapshot_download(repo_id=repo)
    finally:
        stop.set()
        t.join(timeout=2)
    on_pct(100.0)
```

- [ ] **Step 4: Gates.**
- [ ] **Step 5: Commit** — `feat: model store scan and download progress math`

---

### Task 8: Serve core — emit, command loop, wiring against fakes

**Files:**
- Create: `src/localvoice/serve.py`
- Test: `tests/test_serve.py`

**Interfaces:**
- Consumes: everything above; `Orchestrator(on_state=...)`; fakes from `tests/fakes.py`.
- Produces: `class Serve` with:
  - `__init__(self, config_path: Path, cfg: Config, *, allow_inject: bool = False, stdin=None, stdout=None, engine_set: EngineSet | None = None, player=None, capture=None, inference=None)` — every collaborator injectable; defaults construct the real ones (player/capture built like `cmd_run` does, inference `ThreadPoolExecutor(1)`).
  - `emit(self, obj: dict) -> None` — locked `json.dumps` + newline + flush to the REAL stdout captured at init.
  - `run(self) -> None` — first action: `self._real_stdout = self._stdout; sys.stdout = sys.stderr` (when using process stdio); emit `ready` (version 1, config as nested dict, schema); submit `engine_set.load_all` to the inference executor emitting `load_progress` then `engines_ready` (or `error` if models missing/load fails); start orchestrator loop on a thread; then read stdin lines dispatching commands until EOF/shutdown.
  - Command handlers: `ptt_down`/`ptt_up` (payload `held_ms`)/`esc` → `orch.post(Event(...))`, guarded by an `engines_ready` flag (else `error` event); `set_config` → coerce all → `apply_overlay_changes` → reload merged `Config` via `load_config` → `plan_apply` → instant keys setattr onto the LIVE section dataclasses (and `orch._deps.think` for `llm.think`) → reloads via inference executor (`load_progress` events) → audio restart via player/capture stop+start → emit `config_applied {config, reloaded}`; `list_models` → emit `models`; `shutdown` → stop streams, `os._exit(0)` (test seam: `self._exit = os._exit` injectable).
- The full merged config serializes with `dataclasses.asdict(cfg)`.

- [ ] **Step 1: Failing tests** — `tests/test_serve.py` (drive `Serve` with in-memory streams and fakes; no real audio/models):

```python
import io
import json
import threading
import time
from pathlib import Path

from localvoice.config import AudioConfig, Config, KeysConfig, LlmConfig, SttConfig, TtsConfig
from localvoice.engineset import EngineSet
from localvoice.serve import Serve
from tests.fakes import FakeLLM, FakePlayer, FakeSTT, FakeTTS


class FakeCapture:
    def __init__(self):
        self.armed = False

    def start(self):
        pass

    def stop(self):
        pass

    def arm(self):
        self.armed = True

    def disarm(self):
        import numpy as np

        self.armed = False
        rng = np.random.default_rng(0)
        return (0.1 * rng.standard_normal(16000)).astype("float32")

    def discard(self):
        self.armed = False


class InstantExecutor:
    def submit(self, fn, *a, **k):
        class F:
            def __init__(self):
                fn(*a, **k)

            def done(self):
                return True

            def result(self):
                return None

        return F()

    def shutdown(self, **k):
        pass


def build(tmp_path: Path, commands: list[dict]) -> list[dict]:
    cfg = Config(SttConfig(), LlmConfig(), TtsConfig(), KeysConfig(), AudioConfig())
    cfg_path = tmp_path / "localvoice.toml"
    cfg_path.write_text("")
    factories = {
        "stt": lambda c: FakeSTT("hello there"),
        "llm": lambda c: FakeLLM(["Hi from the fake. ", "More words."]),
        "tts": lambda c: FakeTTS(),
    }
    es = EngineSet(cfg, factories=factories)
    stdin = io.StringIO("".join(json.dumps(c) + "\n" for c in commands))
    stdout = io.StringIO()
    s = Serve(
        config_path=cfg_path, cfg=cfg, allow_inject=True, stdin=stdin, stdout=stdout,
        engine_set=es, player=FakePlayer(), capture=FakeCapture(),
        inference=InstantExecutor(),
    )
    s._exit = lambda code: None
    s.run()
    time.sleep(0.3)  # pipeline thread finishes
    return [json.loads(line) for line in stdout.getvalue().splitlines()]


def events_of(msgs, name):
    return [m for m in msgs if m.get("event") == name]


def test_ready_schema_and_engines_ready(tmp_path):
    msgs = build(tmp_path, [{"cmd": "shutdown"}])
    ready = events_of(msgs, "ready")[0]
    assert ready["version"] == 1
    assert ready["config"]["llm"]["max_tokens"] == 1024
    assert any(d["key"] == "llm.think" for d in ready["schema"])
    assert events_of(msgs, "engines_ready")
    assert [m for m in msgs if m.get("event") == "load_progress"]


def test_ptt_turn_emits_transcript_and_turn_done(tmp_path):
    msgs = build(
        tmp_path,
        [{"cmd": "ptt_down"}, {"cmd": "ptt_up", "held_ms": 500}, {"cmd": "shutdown"}],
    )
    assert events_of(msgs, "user_text")[0]["text"] == "hello there"
    assert "Hi from the fake." in events_of(msgs, "assistant_clause")[0]["text"]
    assert set(events_of(msgs, "turn_done")[0]["latency"]) == {
        "stt", "ttft", "first_clause", "tts_first", "total",
    }
    states = [m["state"] for m in events_of(msgs, "state")]
    assert "listening" in states and "processing" in states


def test_set_config_instant_and_overlay(tmp_path):
    msgs = build(
        tmp_path,
        [{"cmd": "set_config", "changes": {"llm.think": "true"}}, {"cmd": "shutdown"}],
    )
    applied = events_of(msgs, "config_applied")[0]
    assert applied["config"]["llm"]["think"] is True
    assert applied["reloaded"] == []
    import tomllib

    data = tomllib.loads((tmp_path / "localvoice.local.toml").read_text())
    assert data["llm"]["think"] is True


def test_set_config_reload_path(tmp_path):
    msgs = build(
        tmp_path,
        [{"cmd": "set_config", "changes": {"llm.model": "other/model"}}, {"cmd": "shutdown"}],
    )
    applied = events_of(msgs, "config_applied")[0]
    assert applied["reloaded"] == ["llm"]
    assert len([m for m in events_of(msgs, "load_progress") if m["engine"] == "llm"]) >= 4


def test_bad_config_value_yields_error_event(tmp_path):
    msgs = build(
        tmp_path,
        [{"cmd": "set_config", "changes": {"llm.max_tokens": "nope"}}, {"cmd": "shutdown"}],
    )
    assert any("llm.max_tokens" in m["message"] for m in events_of(msgs, "error"))
    assert not events_of(msgs, "config_applied")


def test_unknown_command_yields_error(tmp_path):
    msgs = build(tmp_path, [{"cmd": "dance"}, {"cmd": "shutdown"}])
    assert any("dance" in m["message"] for m in events_of(msgs, "error"))
```

- [ ] **Step 2: Verify fail** — ModuleNotFoundError.
- [ ] **Step 3: Implement** `src/localvoice/serve.py` (complete):

```python
import json
import os
import queue
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from localvoice.config import Config, ConfigError, load_config
from localvoice.engineset import EngineSet, plan_apply
from localvoice.events import Event, EventType, State
from localvoice.overlay import apply_overlay_changes
from localvoice.schema import build_schema, coerce

PROTOCOL_VERSION = 1
_SAMPLE_LINE = "This is what the selected voice sounds like."


class Serve:
    def __init__(
        self,
        config_path: Path,
        cfg: Config,
        *,
        allow_inject: bool = False,
        stdin=None,
        stdout=None,
        engine_set: EngineSet | None = None,
        player=None,
        capture=None,
        inference=None,
    ) -> None:
        self._config_path = Path(config_path)
        self._cfg = cfg
        self._allow_inject = allow_inject
        self._stdin = stdin if stdin is not None else sys.stdin
        self._real_stdout = stdout if stdout is not None else sys.stdout
        self._inference = inference or ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="inference"
        )
        self._engines = engine_set or EngineSet(cfg)
        self._emit_lock = threading.Lock()
        self._engines_ready = False
        self._exit = os._exit
        self._level_last = 0.0

        from localvoice.app import Orchestrator
        from localvoice.transcript import Transcript

        if player is None:
            from localvoice.audio.player import AudioPlayer

            player = AudioPlayer(cfg.audio, on_response_finished=self._on_drained)
        if capture is None:
            from localvoice.audio.capture import MicCapture

            capture = MicCapture(cfg.audio, on_level=self._on_level)
        self._player = player
        self._capture = capture
        self._orch = Orchestrator(
            capture=capture,
            player=player,
            stt=self._engines.stt,
            llm=self._engines.llm,
            tts=self._engines.tts,
            transcript=Transcript(cfg.llm.system_prompt),
            keys_cfg=cfg.keys,
            inference=self._inference,
            think=cfg.llm.think,
            status=lambda s: None,
            on_state=lambda st: self.emit({"event": "state", "state": st.name.lower()}),
        )
        d = self._orch._deps
        d.on_user_text = lambda t: self.emit({"event": "user_text", "text": t})
        d.on_assistant_clause = lambda t: self.emit({"event": "assistant_clause", "text": t})
        d.on_thinking = lambda t: self.emit({"event": "reasoning", "text": t})
        d.on_metrics = lambda m: self.emit({"event": "turn_done", "latency": m})

    def _on_drained(self) -> None:
        self._orch.post(Event(EventType.RESPONSE_FINISHED, gen=self._orch._gen))

    def _on_level(self, rms: float) -> None:
        import time

        now = time.monotonic()
        if now - self._level_last >= 0.05:  # <=20 Hz
            self._level_last = now
            self.emit({"event": "level", "rms": round(float(rms), 4)})

    def emit(self, obj: dict) -> None:
        with self._emit_lock:
            self._real_stdout.write(json.dumps(obj) + "\n")
            self._real_stdout.flush()

    def _load_engines(self) -> None:
        try:
            self._engines.load_all(
                lambda n, p, s: self.emit(
                    {"event": "load_progress", "engine": n, "phase": p, "seconds": s}
                )
            )
            self._engines_ready = True
            self.emit({"event": "engines_ready"})
        except Exception as exc:  # noqa: BLE001 — surfaced to the GUI
            self.emit({"event": "error", "message": f"engine load failed: {exc}"})

    def run(self) -> None:
        if self._real_stdout is sys.stdout:  # process mode: protect the protocol
            sys.stdout = sys.stderr
        self.emit(
            {
                "event": "ready",
                "version": PROTOCOL_VERSION,
                "config": asdict(self._cfg),
                "schema": build_schema(self._cfg),
            }
        )
        try:
            self._capture.start()
            self._player.start()
        except Exception as exc:  # noqa: BLE001 — GUI Setup pane handles it
            self.emit({"event": "error", "message": f"audio unavailable: {exc}"})
        self._inference.submit(self._load_engines)
        loop = threading.Thread(target=self._orch.run_forever, daemon=True)
        loop.start()
        for line in self._stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                self.emit({"event": "error", "message": f"bad json: {line[:80]}"})
                continue
            if self._dispatch(msg):
                break

    def _require_ready(self) -> bool:
        if not self._engines_ready:
            self.emit({"event": "error", "message": "engines still loading"})
            return False
        return True

    def _dispatch(self, msg: dict) -> bool:
        cmd = msg.get("cmd")
        if cmd == "ptt_down":
            if self._require_ready():
                self._orch.post(Event(EventType.PTT_DOWN))
        elif cmd == "ptt_up":
            if self._require_ready():
                self._orch.post(Event(EventType.PTT_UP, held_ms=int(msg.get("held_ms", 500))))
        elif cmd == "esc":
            self._orch.post(Event(EventType.ESC))
        elif cmd == "set_config":
            self._set_config(msg.get("changes", {}))
        elif cmd == "list_models":
            from localvoice.modelstore import scan_models

            self.emit({"event": "models", "installed": scan_models()})
        elif cmd == "download_model":
            self._download(msg.get("repo", ""))
        elif cmd == "preview_voice":
            self._preview(msg.get("voice", self._cfg.tts.voice))
        elif cmd == "inject_audio":
            self._inject(msg.get("path", ""))
        elif cmd == "shutdown":
            self._shutdown()
            return True
        else:
            self.emit({"event": "error", "message": f"unknown command: {cmd}"})
        return False

    def _set_config(self, raw_changes: dict) -> None:
        try:
            changes = {k: coerce(k, v) for k, v in raw_changes.items()}
        except ConfigError as exc:
            self.emit({"event": "error", "message": str(exc)})
            return
        apply_overlay_changes(self._config_path, changes)
        new_cfg = load_config(self._config_path)
        plan = plan_apply(changes)
        for key in plan["instant"]:
            section, name = key.split(".", 1)
            setattr(getattr(self._cfg, section), name, getattr(getattr(new_cfg, section), name))
        if "llm.think" in changes:
            self._orch._deps.think = self._cfg.llm.think
        for name in plan["reload"]:
            self._sync_section(name, new_cfg)
            self._inference.submit(self._reload_engine, name)
        if plan["restart_audio"]:
            self._sync_section("audio", new_cfg)
            self._restart_audio()
        self.emit(
            {
                "event": "config_applied",
                "config": asdict(self._cfg),
                "reloaded": plan["reload"],
            }
        )

    def _sync_section(self, name: str, new_cfg: Config) -> None:
        from dataclasses import fields

        live, new = getattr(self._cfg, name), getattr(new_cfg, name)
        for f in fields(type(live)):
            setattr(live, f.name, getattr(new, f.name))

    def _reload_engine(self, name: str) -> None:
        try:
            self._engines.reload(
                name,
                self._cfg,
                lambda n, p, s: self.emit(
                    {"event": "load_progress", "engine": n, "phase": p, "seconds": s}
                ),
            )
        except Exception as exc:  # noqa: BLE001
            self.emit({"event": "error", "message": f"reload {name} failed: {exc}"})

    def _restart_audio(self) -> None:
        for dev in (self._player, self._capture):
            try:
                dev.stop()
                dev.start()
            except Exception as exc:  # noqa: BLE001
                self.emit({"event": "error", "message": f"audio restart failed: {exc}"})

    def _download(self, repo: str) -> None:
        from localvoice.modelstore import download

        def job() -> None:
            try:
                download(
                    repo,
                    on_pct=lambda pct: self.emit(
                        {"event": "download_progress", "repo": repo, "pct": pct, "done": False}
                    ),
                )
                self.emit(
                    {"event": "download_progress", "repo": repo, "pct": 100.0, "done": True}
                )
            except Exception as exc:  # noqa: BLE001
                self.emit({"event": "error", "message": f"download {repo} failed: {exc}"})

        threading.Thread(target=job, daemon=True).start()

    def _preview(self, voice: str) -> None:
        if not self._require_ready():
            return

        def job() -> None:
            old = self._cfg.tts.voice
            try:
                self._cfg.tts.voice = voice
                for chunk in self._engines.tts.synthesize(_SAMPLE_LINE):
                    self._player.submit_raw(chunk)
            except Exception as exc:  # noqa: BLE001
                self.emit({"event": "error", "message": f"preview failed: {exc}"})
            finally:
                self._cfg.tts.voice = old

        self._inference.submit(job)

    def _inject(self, path: str) -> None:
        if not self._allow_inject:
            self.emit({"event": "error", "message": "inject_audio not allowed"})
            return
        if not self._require_ready():
            return
        import wave

        import numpy as np

        with wave.open(path) as w:
            audio = (
                np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32)
                / 32768.0
            )
        self._orch.post(Event(EventType.PTT_DOWN))
        self._capture.buffer.write(audio) if hasattr(self._capture, "buffer") else None
        self._orch.post(Event(EventType.PTT_UP, held_ms=int(len(audio) / 16)))

    def _shutdown(self) -> None:
        self._orch.shutdown()
        for dev in (self._player, self._capture):
            try:
                dev.stop()
            except Exception:  # noqa: BLE001
                pass
        self._inference.shutdown(wait=False, cancel_futures=True)
        self._exit(0)
```

Implementation notes for the engineer:
- The fake-capture in tests has no `.buffer`; `_inject`'s hasattr guard covers it — the real `MicCapture` path writes into the gated buffer while armed. The FakeCapture disarm() supplies audio regardless, which is what the PTT tests rely on.
- `test_ptt_turn...` relies on `InstantExecutor` running `run_pipeline` synchronously inside `orch.handle`; the orchestrator loop thread plus `time.sleep(0.3)` in the helper absorbs ordering.
- `MicCapture` gains the `on_level` pass-through in Task 2; `Orchestrator(on_state=...)` comes from Task 3.

- [ ] **Step 4: Gates** — full suite; expect prior 106 + new.
- [ ] **Step 5: Commit** — `feat: serve protocol core with command dispatch and config hot-apply`

---

### Task 9: CLI wiring, protocol doc, slow end-to-end

**Files:**
- Modify: `src/localvoice/__main__.py` (serve subcommand)
- Create: `docs/gui.md`
- Test: `tests/test_cli.py` (append), `tests/test_smoke_slow.py` (append)

**Interfaces:**
- Produces: `localvoice serve [--config PATH] [--allow-inject]`; `cmd_serve` mirrors `cmd_run`'s config load + `_defuse_tqdm_mp_lock` but NO preflight SystemExit (serve reports problems as protocol `error` events) and no hotkey listener. `docs/gui.md` documents every event/command with payload examples (source: spec §3 + this plan's semantics refinement).

- [ ] **Step 1: Failing CLI test** (append to `tests/test_cli.py`):

```python
def test_serve_parses():
    args = build_parser().parse_args(["serve", "--allow-inject"])
    assert args.command == "serve" and args.allow_inject is True
    assert build_parser().parse_args(["serve"]).allow_inject is False
```

- [ ] **Step 2: Verify fail; implement.** Subparser:

```python
    serve = sub.add_parser("serve", help="GUI/automation protocol mode (JSON lines on stdio)")
    serve.add_argument("--config", default="localvoice.toml")
    serve.add_argument("--allow-inject", action="store_true", help="enable inject_audio (tests)")
```

`cmd_serve`:

```python
def cmd_serve(args) -> None:
    cfg = _load_config(args)
    _defuse_tqdm_mp_lock()
    from localvoice.serve import Serve

    Serve(Path(args.config), cfg, allow_inject=args.allow_inject).run()
```

Register in `main()`'s dispatch dict and add `"serve"` to `_COMMANDS`.

- [ ] **Step 3: Slow end-to-end** (append to `tests/test_smoke_slow.py`):

```python
@pytest.mark.slow
@pytest.mark.skipif(not TINY_LLM.exists(), reason="tiny local LLM not present")
def test_serve_protocol_end_to_end(tmp_path):
    import json
    import subprocess
    import sys

    from localvoice.bench import _speech_fixture

    wav = tmp_path / "q.wav"
    import wave

    audio = _speech_fixture()
    with wave.open(str(wav), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((audio * 32767).astype("int16").tobytes())
    cfg = tmp_path / "localvoice.toml"
    cfg.write_text(
        f'[stt]\nmodel = "mlx-community/whisper-tiny"\n'
        f'[llm]\nmodel = "{TINY_LLM}"\nmax_tokens = 60\n'
    )
    proc = subprocess.Popen(
        [sys.executable, "-m", "localvoice", "serve", "--config", str(cfg), "--allow-inject"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, cwd=str(Path(__file__).resolve().parent.parent),
    )
    events = []
    try:
        deadline = time.time() + 300
        injected = False
        while time.time() < deadline:
            line = proc.stdout.readline()
            if not line:
                break
            ev = json.loads(line)
            events.append(ev.get("event"))
            if ev.get("event") == "engines_ready" and not injected:
                injected = True
                proc.stdin.write(json.dumps({"cmd": "inject_audio", "path": str(wav)}) + "\n")
                proc.stdin.flush()
            if ev.get("event") == "turn_done":
                break
        assert "ready" in events and "engines_ready" in events
        assert "user_text" in events and "assistant_clause" in events
        assert "turn_done" in events
    finally:
        try:
            proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
            proc.stdin.flush()
        except Exception:  # noqa: BLE001
            pass
        proc.wait(timeout=10)
```

Add `import time` to the file imports if absent. Run: `uv run pytest tests/test_smoke_slow.py -m slow -v` — all four slow tests pass.

- [ ] **Step 4: Write `docs/gui.md`** — protocol reference: transport (JSON lines; stdout=protocol, stderr=logs), startup sequence (`ready` → `load_progress`… → `engines_ready`), full event table and command table with one JSON example each (copy names/payloads from this plan exactly), hot-apply semantics table (spec §5), and a "driving it manually" snippet (`uv run localvoice serve --allow-inject` + example stdin lines).

- [ ] **Step 5: Gates + commit** — `feat: localvoice serve CLI mode with protocol docs and e2e test`

---

## Self-review notes (completed at plan time)

- Spec §3 coverage: every event/command has an implementing task (ready/schema→T4+T8, state→T3+T8, transcript/reasoning→T8 wiring, turn_done→T1+T8, level→T2+T8, load/engines_ready→T6+T8, models/download→T7+T8, config_applied→T5+T6+T8, preview/inject/shutdown→T8, CLI/docs/e2e→T9). §4→T4. §5→T6 (+T8 routing). §6-§7 are phases B/C by design.
- Type consistency: `on_progress(name, phase, seconds)` triple used identically in T6 and T8; descriptor keys of T4 match T8's `ready` payload; `plan_apply` return shape matches T8's usage; `coerce` raises `ConfigError` which T8 catches.
- Known deliberate seam: `_inject`'s hasattr guard is intentional for fake compatibility.
- Deferred by design: `download` integration (network) is exercised manually/phase B; only its pure math is unit-gated.
