# Tools T1 — tool framework + free web search — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The model can call `web_search` and `fetch_page` mid-turn — free, keyless — with tool activity visible in the app, shipping on the current mlx-lm engine.

**Architecture:** A streaming `ToolCallParser` (shaped like the shipped `ChannelThinkTranslator`) captures Gemma-format tool calls out of the LLM stream before they reach TTS; `run_pipeline` gains a bounded rounds-loop that executes the tool and re-prompts with the result; a small `tools/` registry defines the tools; `serve` emits two additive protocol events the SwiftUI app renders as a live activity chip.

**Tech Stack:** Python 3.12, ddgs (DuckDuckGo, keyless), trafilatura (page main-text extraction), existing mlx-lm engine; SwiftUI (Swift 6 strict concurrency) for the chip.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-07-localvoice-tools-design.md` §3, §4, §6, §7 (T1 scope only — no mlx-vlm, no screenshots).
- Branch: `feat/tools-t1` off `main`. Python gate: `uv run pytest -m "not slow"` (baseline **175 passed**). Swift gate: `cd app/LocalVoice && xcodegen generate && xcodebuild -project LocalVoice.xcodeproj -scheme LocalVoice -destination 'platform=macOS' test` (baseline **83 tests**).
- Completely free: no API keys, no paid endpoints. New deps allowed: `ddgs`, `trafilatura` only.
- Protocol stays **version 1**; `tool_call` / `tool_result` events are additive. docs/gui.md is the contract — update it in the same task that adds the events.
- Tool wire format is Gemma's, exactly: model emits `<|tool_call>call:NAME{args}<tool_call|>`; args use Gemma argument syntax where strings are quoted with `<|"|>` (e.g. `query:<|"|>weather boston<|"|>,max_results:5`). Tool responses are fed back as standard HF messages (assistant `tool_calls` + `role:"tool"`) — the template renders them.
- One tool call per round; `max_rounds` (config, default 3) rounds per turn; on the final round tools are not offered, forcing a spoken answer.
- Tool traffic is **turn-local**: it extends the in-flight messages list only, never the `Transcript` (history stays small; the spoken answer is what persists).
- Tools are offered only when BOTH the config enables them AND the model's template supports them (`<|tool>` present in the chat template).
- Cancellation: tool execution must observe the pipeline's `cancel: threading.Event` (checked between network requests; requests use timeouts ≤ 5 s).
- Swift: no `@unchecked Sendable`, no `DispatchQueue`. UI copy: sentence case, no emoji, no exclamation marks.
- Never `git add -A`; never commit `localvoice.local.toml` or `.superpowers/`.

## File Structure

```
src/localvoice/config.py             # + ToolsConfig, Config.tools
localvoice.toml                      # + [tools] section, system_prompt tools paragraph
src/localvoice/tools/__init__.py     # registry_for(cfg) -> list[Tool]
src/localvoice/tools/base.py         # Tool protocol, ToolResult, hf_tool_schema()
src/localvoice/tools/web.py          # WebSearchTool, FetchPageTool
src/localvoice/textproc/toolcalls.py # ToolCall, ToolCallParser, parse_gemma_args
src/localvoice/llm/base.py           # stream(..., tools=None) in protocol + fakes
src/localvoice/llm/mlx_lm_engine.py  # tools= plumbing, supports_tools
src/localvoice/pipeline.py           # rounds loop, on_tool_call/on_tool_result
src/localvoice/serve.py              # tool_call/tool_result events, registry wiring
docs/gui.md                          # two event rows + tools note
app/LocalVoice/Sources/Protocol/EngineEvent.swift   # 2 cases
app/LocalVoice/Sources/State/AppState.swift         # toolActivity
app/LocalVoice/Sources/Views/TalkView.swift         # activity chip
README.md                            # tools blurb
```

---

### Task 1: ToolsConfig + defaults + schema

**Files:**
- Modify: `src/localvoice/config.py` (add `ToolsConfig`, field on `Config`)
- Modify: `localvoice.toml` (add `[tools]`; extend `[llm] system_prompt`)
- Test: `tests/test_config.py`, `tests/test_schema.py`

**Interfaces:**
- Consumes: existing `Config` dataclass pattern (`config.py:10-58` — per-section dataclasses, `Config` aggregating them; `load_config` merges toml generically over dataclass fields).
- Produces: `ToolsConfig(enabled: bool = True, web_search: bool = True, screenshot: bool = False, max_rounds: int = 3, search_results: int = 5, page_char_cap: int = 8000)` at `cfg.tools`. NOTE `screenshot` defaults **False** in T1 (the tool ships in T3; the key exists now so the schema/settings surface is stable).

- [ ] **Step 1: Write the failing tests**

In `tests/test_config.py` add:

```python
def test_tools_config_defaults():
    from localvoice.config import ToolsConfig

    t = ToolsConfig()
    assert t.enabled is True
    assert t.web_search is True
    assert t.screenshot is False
    assert t.max_rounds == 3
    assert t.search_results == 5
    assert t.page_char_cap == 8000


def test_config_carries_tools_section(tmp_path):
    from localvoice.config import load_config

    p = tmp_path / "localvoice.toml"
    p.write_text('[tools]\nenabled = false\nmax_rounds = 5\n')
    cfg = load_config(p)
    assert cfg.tools.enabled is False
    assert cfg.tools.max_rounds == 5
    assert cfg.tools.web_search is True  # untouched keys keep defaults
```

(If `load_config` requires other sections present, mirror whatever minimal
toml the existing `load_config` tests in this file use and add `[tools]` on
top — copy their fixture style.)

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_config.py -q`
Expected: FAIL — `ImportError: cannot import name 'ToolsConfig'`.

- [ ] **Step 3: Implement**

In `src/localvoice/config.py`, after `AudioConfig`:

```python
@dataclass
class ToolsConfig:
    enabled: bool = True
    web_search: bool = True
    screenshot: bool = False  # ships in T3; key exists so settings stay stable
    max_rounds: int = 3
    search_results: int = 5
    page_char_cap: int = 8000
```

Add `tools: ToolsConfig` to `Config`, and wire construction wherever the
other five sections are constructed in `load_config` (follow the existing
generic merge exactly — read the function; if it iterates `fields(Config)`
no change may be needed beyond the dataclass and the `Config` field).

- [ ] **Step 4: Update the schema completeness expectations**

`tests/test_schema.py` has a completeness test asserting every `Config`
field yields a descriptor. Run `uv run pytest tests/test_schema.py -q`; if
it fails on a count or section list, update the expected values to include
the six `tools.*` keys (section label: `Tools`). `schema.py` derives
widgets from types automatically — `bool→toggle`, `int→number`; no
annotation entries are needed for these keys.

- [ ] **Step 5: Add the toml defaults + system prompt paragraph**

In `localvoice.toml` add (after `[audio]`):

```toml
[tools]
enabled = true
web_search = true
screenshot = false
max_rounds = 3
search_results = 5
page_char_cap = 8000
```

Extend the existing `[llm] system_prompt` value by appending this sentence
to the current string (keep everything already there):
`" You can call tools. Use web_search for current events, prices, weather, or anything the user implies is recent; after searching, speak a short synthesized answer, not a list of results, and name the source site only if asked. If a tool reports no internet connection, say so briefly."`

- [ ] **Step 6: Run the full fast suite**

Run: `uv run pytest -m "not slow" -q`
Expected: all pass (175 baseline + 2 new).

- [ ] **Step 7: Commit**

```bash
git add src/localvoice/config.py localvoice.toml tests/test_config.py tests/test_schema.py
git commit -m "feat(config): tools section with schema coverage"
```

---

### Task 2: Tool protocol, ToolResult, registry

**Files:**
- Create: `src/localvoice/tools/__init__.py`, `src/localvoice/tools/base.py`
- Test: `tests/test_tools_base.py`

**Interfaces:**
- Consumes: `ToolsConfig` from Task 1.
- Produces (exact):

```python
# tools/base.py
@dataclass
class ToolResult:
    ok: bool
    content: dict      # becomes the role:"tool" message content (template renders mappings)
    summary: str       # one human line for the protocol event, e.g. "found 5 results"

class Tool(Protocol):
    name: str
    description: str
    parameters: dict   # JSON schema: {"type": "object", "properties": {...}, "required": [...]}
    def execute(self, args: dict, cancel: threading.Event) -> ToolResult: ...

def hf_tool_schema(tool: Tool) -> dict:
    """{"type": "function", "function": {"name", "description", "parameters"}} —
    the shape apply_chat_template's tools= expects."""

# tools/__init__.py
def registry_for(cfg: ToolsConfig) -> list[Tool]:
    """Enabled tools only; empty when cfg.enabled is False. T1 registers
    WebSearchTool and FetchPageTool (both gated on cfg.web_search)."""
```

- [ ] **Step 1: Write the failing tests**

`tests/test_tools_base.py`:

```python
import threading

from localvoice.config import ToolsConfig
from localvoice.tools import registry_for
from localvoice.tools.base import ToolResult, hf_tool_schema


def test_registry_disabled_master_switch_is_empty():
    assert registry_for(ToolsConfig(enabled=False)) == []


def test_registry_web_search_gated():
    names = [t.name for t in registry_for(ToolsConfig(web_search=False))]
    assert "web_search" not in names and "fetch_page" not in names


def test_registry_default_has_both_web_tools():
    names = [t.name for t in registry_for(ToolsConfig())]
    assert names == ["web_search", "fetch_page"]


def test_hf_tool_schema_shape():
    tool = registry_for(ToolsConfig())[0]
    s = hf_tool_schema(tool)
    assert s["type"] == "function"
    assert s["function"]["name"] == "web_search"
    assert s["function"]["parameters"]["type"] == "object"
    assert "query" in s["function"]["parameters"]["properties"]


def test_tool_result_fields():
    r = ToolResult(ok=False, content={"error": "no internet connection"}, summary="search failed")
    assert r.ok is False and r.content["error"] and r.summary
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_tools_base.py -q` → ImportError.

- [ ] **Step 3: Implement** `tools/base.py` exactly per the Produces block
(`Tool` via `typing.Protocol`, `hf_tool_schema` returning the dict shown),
and `tools/__init__.py` `registry_for` importing `WebSearchTool`/
`FetchPageTool` from `localvoice.tools.web` **lazily inside the function**
(Task 3 creates them; for THIS task create `tools/web.py` with the two
minimal classes carrying only `name`/`description`/`parameters` and an
`execute` that raises `NotImplementedError` — Task 3 fills them in; keep
`parameters` real now:)

```python
# minimal tools/web.py for this task
class WebSearchTool:
    name = "web_search"
    description = (
        "Search the web. Use for current events, prices, weather, or facts "
        "you are not confident about."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "the search query"},
            "max_results": {"type": "integer", "description": "how many results (1-10)"},
        },
        "required": ["query"],
    }
    def __init__(self, cfg): self._cfg = cfg
    def execute(self, args, cancel): raise NotImplementedError


class FetchPageTool:
    name = "fetch_page"
    description = "Fetch one web page by URL and return its readable main text."
    parameters = {
        "type": "object",
        "properties": {"url": {"type": "string", "description": "absolute http(s) URL"}},
        "required": ["url"],
    }
    def __init__(self, cfg): self._cfg = cfg
    def execute(self, args, cancel): raise NotImplementedError
```

`registry_for` order is `[WebSearchTool(cfg), FetchPageTool(cfg)]`.

- [ ] **Step 4: Run** `uv run pytest tests/test_tools_base.py -q` → PASS; then full fast suite.

- [ ] **Step 5: Commit**

```bash
git add src/localvoice/tools tests/test_tools_base.py
git commit -m "feat(tools): tool protocol, result type, config-gated registry"
```

---

### Task 3: web_search + fetch_page implementations

**Files:**
- Modify: `src/localvoice/tools/web.py` (fill in `execute`)
- Modify: `pyproject.toml` (add deps), `uv.lock` (via `uv lock`)
- Test: `tests/test_tools_web.py`

**Interfaces:**
- Consumes: `ToolResult` (Task 2), `ToolsConfig.search_results` / `page_char_cap`.
- Produces: `WebSearchTool.execute({"query": str, "max_results"?: int}, cancel)` →
  ok=True `content={"results": [{"title","url","snippet"}...]}`, summary
  `f"found {n} results for: {query}"`; network failure → ok=False
  `content={"error": "no internet connection"}` summary `"search failed: no internet connection"`.
  `FetchPageTool.execute({"url": str}, cancel)` → ok=True
  `content={"url", "text"}` (text truncated to `page_char_cap`), summary
  `f"read {len(text)} chars from {domain}"`; failure → ok=False as above but
  `"could not fetch the page"`. Both check `cancel.is_set()` before each
  network call and return ok=False `content={"error":"cancelled"}` if set.
  `WebSearchTool` keeps an instance dict cache keyed `(query, n)` — a repeat
  query within the session returns the cached result without network.

- [ ] **Step 1: Add dependencies**

In `pyproject.toml` `dependencies`, add `"ddgs>=9"` and
`"trafilatura>=2"`. Run `uv lock && uv sync`; confirm both import:
`uv run python -c "import ddgs, trafilatura; print('ok')"`.
(If `ddgs>=9` does not resolve, relax to `"ddgs"` and record the resolved
version in your report.)

- [ ] **Step 2: Write the failing tests** — all network mocked via
`monkeypatch`; CI must never touch the network:

```python
# tests/test_tools_web.py
import threading

import pytest

from localvoice.config import ToolsConfig
from localvoice.tools.web import FetchPageTool, WebSearchTool

NO_CANCEL = threading.Event()


class FakeDDGS:
    calls = 0
    def __init__(self, *a, **k): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def text(self, query, max_results=5):
        FakeDDGS.calls += 1
        return [{"title": f"R{i}", "href": f"https://ex.com/{i}", "body": f"snippet {i}"}
                for i in range(max_results)]


def test_search_maps_results(monkeypatch):
    monkeypatch.setattr("localvoice.tools.web.DDGS", FakeDDGS)
    tool = WebSearchTool(ToolsConfig())
    r = tool.execute({"query": "boston weather"}, NO_CANCEL)
    assert r.ok and len(r.content["results"]) == 5
    assert set(r.content["results"][0]) == {"title", "url", "snippet"}
    assert "boston weather" in r.summary


def test_search_session_cache(monkeypatch):
    monkeypatch.setattr("localvoice.tools.web.DDGS", FakeDDGS)
    FakeDDGS.calls = 0
    tool = WebSearchTool(ToolsConfig())
    tool.execute({"query": "same"}, NO_CANCEL)
    tool.execute({"query": "same"}, NO_CANCEL)
    assert FakeDDGS.calls == 1


def test_search_offline_is_structured(monkeypatch):
    class Boom:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def text(self, *a, **k): raise OSError("network is unreachable")
    monkeypatch.setattr("localvoice.tools.web.DDGS", Boom)
    r = WebSearchTool(ToolsConfig()).execute({"query": "x"}, NO_CANCEL)
    assert r.ok is False and r.content["error"] == "no internet connection"


def test_search_cancelled_before_network(monkeypatch):
    monkeypatch.setattr("localvoice.tools.web.DDGS", FakeDDGS)
    cancel = threading.Event(); cancel.set()
    r = WebSearchTool(ToolsConfig()).execute({"query": "x"}, cancel)
    assert r.ok is False and r.content["error"] == "cancelled"


def test_fetch_page_truncates(monkeypatch):
    monkeypatch.setattr("localvoice.tools.web.fetch_url", lambda url: "<html>raw</html>")
    monkeypatch.setattr("localvoice.tools.web.extract", lambda html, **k: "words " * 4000)
    tool = FetchPageTool(ToolsConfig(page_char_cap=100))
    r = tool.execute({"url": "https://ex.com/a"}, NO_CANCEL)
    assert r.ok and len(r.content["text"]) <= 100 and r.content["url"] == "https://ex.com/a"


def test_fetch_page_failure(monkeypatch):
    monkeypatch.setattr("localvoice.tools.web.fetch_url", lambda url: None)
    r = FetchPageTool(ToolsConfig()).execute({"url": "https://ex.com"}, NO_CANCEL)
    assert r.ok is False and "could not fetch" in r.summary
```

- [ ] **Step 3: Run to verify failure** — NotImplementedError / ImportError.

- [ ] **Step 4: Implement** in `tools/web.py`. Import at module top:
`from ddgs import DDGS` and
`from trafilatura import extract, fetch_url` (module-level names so tests
can monkeypatch `localvoice.tools.web.DDGS` / `.fetch_url` / `.extract`).
`WebSearchTool.execute`: cancel check → cache check → `with DDGS() as d:
rows = d.text(query, max_results=n)` where
`n = min(int(args.get("max_results", self._cfg.search_results)), 10)` →
map `href→url`, `body→snippet` → cache + return. Catch `Exception` around
the network call; return the structured offline error (summary
`"search failed: no internet connection"`). `FetchPageTool.execute`:
cancel check → `html = fetch_url(url)`; `None`/exception → failure result →
else `text = extract(html, include_comments=False) or ""` truncated to
`self._cfg.page_char_cap`; summary uses `urllib.parse.urlparse(url).netloc`.

- [ ] **Step 5: Run** the file, then the full fast suite. Also run ruff:
`uv run ruff check src tests` — clean.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/localvoice/tools/web.py tests/test_tools_web.py
git commit -m "feat(tools): free web_search and fetch_page via ddgs and trafilatura"
```

---

### Task 4: Gemma tool-call stream parser

**Files:**
- Create: `src/localvoice/textproc/toolcalls.py`
- Test: `tests/test_toolcalls.py`

**Interfaces:**
- Consumes: nothing new (pure text processing; mirrors
  `ChannelThinkTranslator`'s hold-back style — read
  `src/localvoice/textproc/sanitize.py` first, including
  `_longest_suffix_prefix`, and reuse it by import).
- Produces (exact):

```python
@dataclass
class ToolCall:
    name: str
    args: dict
    raw: str            # full marker-to-marker capture, for error feedback

class ToolFormatError(Exception): ...

def parse_gemma_args(body: str) -> dict:
    """'query:<|\"|>boston<|\"|>,max_results:5' -> {"query": "boston", "max_results": 5}.
    Strings are <|\"|>-quoted (may contain commas/colons/braces); bare tokens
    parse as int, float, true/false, else raise ToolFormatError."""

class ToolCallParser:
    def feed(self, delta: str) -> str          # passthrough text (never call payloads)
    def finish(self) -> str
    @property
    def call(self) -> ToolCall | None          # set once a full call is captured
    # After a complete <|tool_call>call:NAME{...}<tool_call|> is captured,
    # `call` is non-None and ALL further fed text is swallowed (the round is
    # over; the pipeline stops consuming the stream). A malformed body sets
    # `call` to ToolCall(name=<parsed-or-"">, args={}, raw=...) and raises
    # nothing from feed(); the pipeline detects args=={} + raw and sends the
    # error round. Expose `self.malformed: bool` for that distinction.
```

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_toolcalls.py
import pytest

from localvoice.textproc.toolcalls import ToolCallParser, ToolFormatError, parse_gemma_args


def run(deltas):
    p = ToolCallParser()
    out = "".join(p.feed(d) for d in deltas) + p.finish()
    return out, p


def test_parse_args_strings_and_numbers():
    body = 'query:<|"|>boston, MA: weather<|"|>,max_results:5'
    assert parse_gemma_args(body) == {"query": "boston, MA: weather", "max_results": 5}


def test_parse_args_bool_float_empty():
    assert parse_gemma_args("flag:true,ratio:0.5") == {"flag": True, "ratio": 0.5}
    assert parse_gemma_args("") == {}


def test_parse_args_malformed_raises():
    with pytest.raises(ToolFormatError):
        parse_gemma_args("query:<|\"|>unterminated")


def test_plain_text_passthrough():
    out, p = run(["Hello ", "world."])
    assert out == "Hello world." and p.call is None


def test_call_captured_and_never_spoken():
    deltas = ["Let me check. ", "<|tool_call>call:web_search{query:<|\"|>rain<|\"|>}",
              "<tool_call|>", " trailing junk"]
    out, p = run(deltas)
    assert out == "Let me check. "          # payload and post-call text swallowed
    assert p.call is not None and p.call.name == "web_search"
    assert p.call.args == {"query": "rain"} and p.malformed is False


def test_call_split_across_arbitrary_boundaries():
    deltas = ["<|tool_", "call>call:fetch", "_page{url:<|\"|>https://a.b/c<|\"|>}<tool_", "call|>"]
    out, p = run(deltas)
    assert out == "" and p.call.name == "fetch_page"
    assert p.call.args == {"url": "https://a.b/c"}


def test_malformed_body_flags_not_raises():
    deltas = ["<|tool_call>call:web_search{query:<|\"|>oops", "<tool_call|>"]
    out, p = run(deltas)
    assert p.call is not None and p.malformed is True and p.call.args == {}
    assert out == ""


def test_unclosed_call_at_finish_is_malformed():
    out, p = run(["<|tool_call>call:web_search{query:<|\"|>x<|\"|>}"])
    assert out == "" and p.call is not None and p.malformed is True
```

- [ ] **Step 2: Run to verify failure** — ImportError.

- [ ] **Step 3: Implement** `textproc/toolcalls.py`. Markers:
`_TC_OPEN = "<|tool_call>"`, `_TC_CLOSE = "<tool_call|>"`, string quote
`_Q = '<|"|>'`. Parser states: passthrough (hold back suffixes that prefix
`_TC_OPEN`, exactly the `_longest_suffix_prefix` discipline in
`sanitize.py` — import it) → capturing (swallow everything into a buffer
until `_TC_CLOSE`) → done (swallow all). On close: strip the leading
`call:`, split `NAME` at the first `{`, body is up to the matching final
`}`; `parse_gemma_args(body)` — `ToolFormatError` ⇒ `malformed=True`,
`args={}`. `parse_gemma_args`: scan char-by-char; on `_Q` consume to the
closing `_Q` (no escapes — Gemma quotes are unambiguous delimiters);
top-level `,` splits pairs, first `:` splits key/value; bare values:
`true`/`false` → bool, else `int(...)`, else `float(...)`, else
`ToolFormatError`. `finish()`: if capturing, set `call` with
`malformed=True`; return "".

- [ ] **Step 4: Run** file then full fast suite. Ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/localvoice/textproc/toolcalls.py tests/test_toolcalls.py
git commit -m "feat(textproc): streaming Gemma tool-call parser"
```

---

### Task 5: Engine offers tools

**Files:**
- Modify: `src/localvoice/llm/base.py` (protocol), `src/localvoice/llm/mlx_lm_engine.py`, `tests/fakes.py` (FakeLLM signature)
- Test: `tests/test_llm_prefix.py`

**Interfaces:**
- Consumes: `hf_tool_schema` (Task 2).
- Produces: `LLMEngine.stream(messages, *, think: bool, tools: list[dict] | None = None)`
  — `tools` is a list of HF tool schemas, passed to
  `apply_chat_template(..., tools=tools)` **only when non-empty**; and
  `supports_tools(chat_template: str | None) -> bool` module function in
  `mlx_lm_engine.py` (True iff template contains `"<|tool>"`), plus
  `MlxLmEngine.supports_tools: bool` set in `load()` (mirror how
  `_channel_style` is set from `is_channel_style` at `load()`).

- [ ] **Step 1: Write the failing tests** in `tests/test_llm_prefix.py`:

```python
def test_supports_tools_detected_from_template():
    from localvoice.llm.mlx_lm_engine import supports_tools

    assert supports_tools("...{{ '<|tool>' }}...")
    assert not supports_tools("...<think>...")
    assert not supports_tools(None)
```

- [ ] **Step 2: Run to verify failure**, implement `supports_tools` beside
`is_channel_style` (same shape), set `self.supports_tools = supports_tools(...)`
in `load()` next to the `_channel_style` line.

- [ ] **Step 3: Thread `tools` through**. In `mlx_lm_engine.py`:
`_template(self, messages, think, tools=None)` — pass `tools=tools` into
BOTH `apply_chat_template` calls when `tools` is truthy (add `tools=tools`
kwarg; the except-TypeError fallback keeps working for templates without
either kwarg). `stream(..., tools=None)`: thread `tools` into both
`_template` calls (the `fit_messages` counting lambda AND the final
`tokens = self._template(messages, think, tools)`) so the token budget
accounts for the tools block. In `llm/base.py`, update the `LLMEngine`
protocol signature to `def stream(self, messages, *, think: bool, tools=None)`.
In `tests/fakes.py`, update `FakeLLM.stream` to accept and RECORD `tools`
(append to `self.calls` or a new `self.tools_seen: list` — read the fake
first and follow its existing recording style).

- [ ] **Step 4: Run the full fast suite** — existing pipeline/serve tests
must stay green (default `tools=None` keeps behavior identical).

- [ ] **Step 5: Commit**

```bash
git add src/localvoice/llm/base.py src/localvoice/llm/mlx_lm_engine.py tests/fakes.py tests/test_llm_prefix.py
git commit -m "feat(llm): offer tool schemas through the chat template"
```

---

### Task 6: Pipeline tool rounds

**Files:**
- Modify: `src/localvoice/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `ToolCallParser`/`ToolCall` (Task 4), `Tool`/`ToolResult`/`hf_tool_schema` (Task 2), `stream(..., tools=)` (Task 5).
- Produces: `PipelineDeps` gains:

```python
tools: list = field(default_factory=list)            # Tool instances (already config-gated)
max_tool_rounds: int = 3
on_tool_call: Callable[[str, str], None] = field(default=lambda name, summary: None)
on_tool_result: Callable[[str, bool, str], None] = field(default=lambda name, ok, summary: None)
```

Turn-loop contract (this is the heart of the task):

```python
offered = [hf_tool_schema(t) for t in deps.tools] or None
by_name = {t.name: t for t in deps.tools}
messages = deps.transcript.messages()          # turn-local copy; Transcript untouched
for round_no in range(deps.max_tool_rounds + 1):
    last = round_no == deps.max_tool_rounds
    parser = ToolCallParser()                  # fresh per round
    # stream with tools=None on the last round to force an answer
    for delta in deps.llm.stream(messages, think=deps.think,
                                 tools=None if last else offered):
        ...cancel checks as today...
        for clause in chunker.feed(text_filter.feed(parser.feed(delta))):
            if not speak(clause): return
        if parser.call is not None:
            break                              # stop consuming; round over
    if parser.call is None:
        # flush tail exactly as the current code does, then break
        break
    call = parser.call
    deps.on_tool_call(call.name, _call_summary(call))
    tool = by_name.get(call.name)
    if tool is None or call.malformed... -> result = ToolResult(ok=False,
        content={"error": f"unknown tool: {call.name}" or "malformed tool call"},
        summary="tool call failed")
    else: result = tool.execute(call.args, cancel)
    if cancel.is_set(): return
    deps.on_tool_result(call.name, result.ok, result.summary)
    messages = messages + [
        {"role": "assistant",
         "tool_calls": [{"id": f"call_{round_no}", "type": "function",
                          "function": {"name": call.name, "arguments": call.args}}]},
        {"role": "tool", "tool_call_id": f"call_{round_no}", "name": call.name,
         "content": result.content},
    ]
```

Notes that bind: `parser.feed` wraps OUTSIDE `text_filter.feed`? **No —
order is `text_filter.feed(parser.feed(delta))`** as shown: tool markers
are extracted from the raw stream first (they are not think-tag content),
then think/code filtering applies to what remains. `parser.malformed` on a
captured call routes to the error result (the model gets one corrective
round; `_call_summary(call)` is `f"calling {call.name}"` for well-formed,
`"malformed tool call"` otherwise). The chunker/text_filter/speak state
persists ACROSS rounds (one spoken narration per turn — a clause spoken in
round 0 like "Let me check." must not be re-flushed or duplicated later).
Metrics: unchanged code; totals naturally include tool time.

- [ ] **Step 1: Write the failing tests** in `tests/test_pipeline.py`,
following the file's existing fake/harness style (read it first — reuse its
FakeLLM/FakeTTS/player fixtures). Add a `ScriptedToolLLM` local fake whose
`stream` returns scripted delta-lists per call and records `tools`:

```python
class ScriptedToolLLM:
    def __init__(self, rounds):  # list[list[str]] — deltas per stream() call
        self.rounds = rounds; self.calls = []
    def stream(self, messages, *, think, tools=None):
        self.calls.append({"messages": list(messages), "tools": tools})
        yield from self.rounds[len(self.calls) - 1]


class EchoTool:
    name = "web_search"; description = "d"; parameters = {"type": "object", "properties": {}}
    def __init__(self): self.executed = []
    def execute(self, args, cancel):
        self.executed.append(args)
        from localvoice.tools.base import ToolResult
        return ToolResult(ok=True, content={"results": [{"title": "T", "url": "u", "snippet": "s"}]},
                          summary="found 1 result")
```

Tests (assert through the existing harness's spoken-output capture):

1. `test_tool_round_executes_and_speaks_continuation` — rounds:
   `[["Let me check. ", '<|tool_call>call:web_search{query:<|"|>rain<|"|>}<tool_call|>'], ["It will rain at noon."]]`;
   assert tool executed once with `{"query": "rain"}`; spoken output is
   `"Let me check. It will rain at noon."`-equivalent (two clauses, no
   marker text ever spoken); second `stream` call's messages end with the
   `role:"tool"` message carrying `{"results": ...}`; `on_tool_call` and
   `on_tool_result` fired once each with the summaries above.
2. `test_last_round_offers_no_tools` — `max_tool_rounds=1`, rounds scripted
   so round 0 calls the tool; assert `calls[0]["tools"]` is a non-empty
   list and `calls[1]["tools"] is None`.
3. `test_unknown_tool_gets_error_round` — call names `nope`; assert no
   crash, second round's tool message content is `{"error": "unknown tool: nope"}`,
   `on_tool_result` got `ok=False`.
4. `test_cancel_during_tool_aborts_turn` — EchoTool variant whose execute
   sets nothing but the test sets `cancel` inside `execute`; assert
   pipeline returns without a second `stream` call and without
   RESPONSE_FINISHED-vs-error inconsistency (mirror how existing cancel
   tests assert).
5. `test_no_tools_configured_is_todays_behavior` — `tools=[]`; a plain
   scripted answer; assert `calls[0]["tools"] is None` and output identical
   to the pre-change expectations (guards the default path).
6. `test_transcript_untouched_by_tool_traffic` — after test 1, assert
   `deps.transcript.messages()` contains NO `role:"tool"` entries and no
   `tool_calls` keys (turn-local guarantee).

- [ ] **Step 2: Run to verify failures.**

- [ ] **Step 3: Implement** per the contract block above. Keep the
existing single-stream code path readable: extract today's inner loop into
the rounds loop rather than duplicating it; the tail-flush
(`chunker.flush() + text_filter.finish()`) runs ONCE after the loop ends
normally (not per round). `parser.finish()` per round only when its stream
ended without a call (its unclosed-capture malformed case then feeds the
error round on the NEXT iteration — treat `parser.call` set-after-finish
identically to set-during-stream).

- [ ] **Step 4: Run** the file, then the full fast suite. Ruff clean.

- [ ] **Step 5: Commit**

```bash
git add src/localvoice/pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): bounded tool-call rounds with spoken continuity"
```

---

### Task 7: Protocol events + serve wiring + contract doc

**Files:**
- Modify: `src/localvoice/serve.py`, `src/localvoice/app.py` (only if the orchestrator constructs PipelineDeps there — read both; wire wherever PipelineDeps is built for serve AND terminal modes), `docs/gui.md`
- Test: `tests/test_serve.py`

**Interfaces:**
- Consumes: `registry_for(cfg.tools)` (Task 2), PipelineDeps fields (Task 6), engine `supports_tools` (Task 5).
- Produces: two protocol events, emitted via serve's existing locked
  emitter: `{"event": "tool_call", "name": str, "summary": str}` and
  `{"event": "tool_result", "name": str, "ok": bool, "summary": str}`.
  Tools handed to PipelineDeps are `registry_for(cfg.tools)` **when the
  active LLM engine's `supports_tools` is True**, else `[]` (a
  non-tool-template model is never offered tools — Global Constraints).
  `max_tool_rounds=cfg.tools.max_rounds`. Terminal mode (`app.py`/run):
  same wiring, printing one stderr/console line per event
  (`tool: calling web_search` / `tool: found 5 results`) — match the
  existing console line style used for `reasoning:`.

- [ ] **Step 1: Write the failing tests** in `tests/test_serve.py`
following its in-memory-stream harness style (read the file; reuse its
fake-engine setup). Two tests: `test_tool_events_emitted` — drive the
serve pipeline with a fake LLM scripted for one tool round (reuse Task 6's
`ScriptedToolLLM`/`EchoTool` pattern; the harness's fake engines live in
`tests/fakes.py` — extend there if serve's harness requires it) and assert
the stdout stream contains a `tool_call` event then a `tool_result` event
with the exact fields above, in order, before `turn_done`.
`test_tools_not_offered_without_template_support` — fake LLM with
`supports_tools = False`; assert its recorded `tools` is `None`/absent for
the turn even with `cfg.tools.enabled = true`.

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement** serve + terminal wiring per Produces.

- [ ] **Step 4: Update `docs/gui.md`**: add the two event rows to the
events table (fields exactly as above, source column: "pipeline tool
rounds"); add one paragraph under the config/hot-apply section: `[tools]`
keys hot-apply per turn, no reload; note events only occur when the model's
template supports tools.

- [ ] **Step 5: Run** the full fast suite. Commit:

```bash
git add src/localvoice/serve.py src/localvoice/app.py tests/test_serve.py tests/fakes.py docs/gui.md
git commit -m "feat(serve): tool_call and tool_result protocol events"
```

---

### Task 8: App chip + README + live acceptance

**Files:**
- Modify: `app/LocalVoice/Sources/Protocol/EngineEvent.swift`, `app/LocalVoice/Sources/State/AppState.swift`, `app/LocalVoice/Sources/Views/TalkView.swift`, `README.md`
- Test: `app/LocalVoice/Tests/ProtocolTests.swift`, `app/LocalVoice/Tests/AppStateTests.swift`

**Interfaces:**
- Consumes: the two protocol events (Task 7, exact JSON above); the app's
  existing decode pattern (read `EngineEvent.swift` — enum cases with
  snake_case CodingKeys and nil-on-unknown forward compat) and reducer
  pattern (`AppState.reduce`).
- Produces: `EngineEvent.toolCall(name: String, summary: String)` and
  `.toolResult(name: String, ok: Bool, summary: String)`;
  `AppState.toolActivity: String?` — set to `summary` on `toolCall`,
  updated to `summary` on `toolResult(ok: false)` (so failures read out),
  cleared (nil) on `toolResult(ok: true)` after the turn continues — 
  cleared on `turnDone` and on `state == "idle"` in all cases. TalkView
  renders `toolActivity` as a caption-style capsule under the orb
  (secondary color, sentence case, no emoji), only when non-nil.

- [ ] **Step 1: Failing Swift tests.** ProtocolTests: decode fixtures
`{"event":"tool_call","name":"web_search","summary":"calling web_search"}` and
`{"event":"tool_result","name":"web_search","ok":true,"summary":"found 5 results"}`
into the two cases (assert associated values). AppStateTests: sequence
toolCall → assert `toolActivity == "calling web_search"`; toolResult(ok:
true) → stays visible? **No** — per Produces: ok:true keeps the LAST
summary until `turnDone`; write the test to pin: after toolResult(ok:true)
`toolActivity == "found 5 results"`, after turnDone it is nil; separate
test: toolResult(ok:false) shows its summary and idle-state clears it.

- [ ] **Step 2: Run** (`xcodegen generate` + `xcodebuild test`) → new tests fail.

- [ ] **Step 3: Implement** the cases (follow the file's existing decode
switch and CodingKeys style exactly), the reducer lines, and the chip:

```swift
// TalkView, under the orb (inside the VStack, after `orb`):
if let activity = appState.toolActivity {
    Text(activity)
        .font(.caption)
        .padding(.horizontal, 10).padding(.vertical, 4)
        .background(Color.gray.opacity(0.15))
        .clipShape(Capsule())
        .accessibilityLabel("Tool activity: \(activity)")
}
```

- [ ] **Step 4: Run the full Swift suite** — 83 baseline + new, TEST SUCCEEDED. Python fast suite untouched-green.

- [ ] **Step 5: README** — in "The app" section's feature sentence, add
tool use: "and, when the model decides it needs them, free web-search
tool calls whose progress shows live under the state orb". In
"How it works", one sentence after the reply description: "If the model
needs current information it can search the web mid-answer (DuckDuckGo,
free, no keys) — you hear a short pause, then the answer; the app shows
what it searched."

- [ ] **Step 6: Live acceptance** (real machine, internet available):
launch the real app (repo-checkout mode, engines ready), drive one turn by
synthetic PTT + `say` asking "What is the current weather in Boston?"
(a question the model should search for). Verify: the chip appears with
the search summary (screenshot, Read it), the spoken answer arrives after,
and `localvoice.local.toml` is untouched. If the model answers WITHOUT
searching, retry once with "Search the web for the current weather in
Boston right now." — the explicit ask must trigger the tool. Record
latency chips. Kill the app after. Note: this step verifies the loop
end-to-end against the real template — it is the T1 gate; treat a
never-triggering tool as BLOCKED (report, do not paper over).

- [ ] **Step 7: Commit**

```bash
git add app/LocalVoice/Sources/Protocol/EngineEvent.swift app/LocalVoice/Sources/State/AppState.swift app/LocalVoice/Sources/Views/TalkView.swift app/LocalVoice/Tests/ProtocolTests.swift app/LocalVoice/Tests/AppStateTests.swift README.md
git commit -m "feat(app): live tool-activity chip and tools docs"
```

---

## Self-review notes (completed at plan time)

- Spec coverage (T1 slice): §3 parser/loop/registry/protocol → Tasks 4/6/2/7; §4 web tools incl. offline + cache → Task 3; §6 prompting → Task 1 step 5; §7 config table → Task 1 (screenshot key present-but-false until T3); §8 testing rows → embedded per task; app chip → Task 8. §5 engine and §4 look_at_screen are T2/T3 by design.
- Placeholder scan: pseudo-ellipses appear only inside the Task 6 contract block to mark "unchanged existing code at this spot" — the implementer is directed to the exact current code (`pipeline.py` quoted lines) rather than rewriting it blind; every NEW behavior has explicit code or exact prose semantics. No TBDs.
- Type consistency: `ToolResult(ok, content: dict, summary)` used identically in Tasks 2/3/6; parser exposes `call`/`malformed` per Task 4 and Task 6 consumes exactly those; `stream(..., tools=)` signature identical in Tasks 5/6; event field names identical in Tasks 7/8.
- Known judgment calls recorded: tool traffic turn-local (constraint block); one call per round; `text_filter.feed(parser.feed(delta))` ordering; chip persistence semantics pinned in Task 8 tests.
