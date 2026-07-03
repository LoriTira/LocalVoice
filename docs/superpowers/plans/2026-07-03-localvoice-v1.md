# LocalVoice v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local push-to-talk voice assistant for Apple Silicon: hold right ⌘ to talk, release to get a spoken answer from a local Qwen3.6 via mlx-whisper → mlx-lm → Kokoro, with instant barge-in.

**Architecture:** Single Python process, plain threads + queues. A pure state machine (IDLE/LISTENING/PROCESSING/SPEAKING) driven by an event queue; hotkey and audio callbacks post events from their own threads; each response runs in a cancellable pipeline thread that streams LLM deltas through a clause chunker into TTS and a flushable audio player. Engines sit behind three protocols so all orchestration logic is tested with fakes.

**Tech Stack:** Python ≥3.12, uv, mlx-lm, mlx-whisper, mlx-audio (Kokoro), sounddevice, pynput, numpy, pytest, ruff.

Spec: `docs/superpowers/specs/2026-07-03-localvoice-architecture-design.md` (read it first).

## Global Constraints

- Platform: Apple Silicon macOS only. Python `>=3.12`.
- Runtime deps exactly: `mlx-lm`, `mlx-whisper`, `mlx-audio`, `sounddevice`, `pynput`, `numpy`, `huggingface_hub` (used directly by `localvoice setup`). Dev deps: `pytest`, `ruff`. No additions without a plan amendment.
- Default LLM (shipped config): `mlx-community/Qwen3.6-35B-A3B-4bit`. Bootstrap on this machine (gitignored `localvoice.local.toml`): local Qwen3.6-27B-MLX-6bit path, because the 35B is still downloading.
- Audio: capture 16 kHz mono float32; playback 24 kHz mono float32.
- PTT key `cmd_r`, stop key `esc`, debounce 120 ms — config-overridable.
- All import of `mlx_*` packages happens inside `load()`/functions, never at module top level (keeps unit tests and CLI startup light).
- Every commit: `uv run ruff check .` clean and `uv run pytest -m "not slow" -q` green. Conventional-commit messages (`feat:`, `test:`, `docs:`, `chore:`).
- License MIT. No emoji anywhere in code, CLI output, or docs.

---

### Task 1: Project scaffold, CI, and repo hygiene

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `LICENSE`, `localvoice.toml`, `src/localvoice/__init__.py`, `tests/__init__.py`, `.github/workflows/ci.yml`, `README.md` (stub — full docs are Task 17)

**Interfaces:**
- Consumes: nothing.
- Produces: importable package `localvoice` with `__version__ = "0.1.0"`; `uv run pytest` / `uv run ruff check .` work; console script `localvoice` (entry point exists; `main()` lands in Task 16 — point it there now).

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "localvoice"
version = "0.1.0"
description = "Local push-to-talk voice assistant for Apple Silicon: whisper -> local LLM -> Kokoro, ~1s voice-to-voice"
readme = "README.md"
requires-python = ">=3.12"
license = { text = "MIT" }
dependencies = [
  "mlx-lm>=0.24",
  "mlx-whisper>=0.4",
  "mlx-audio>=0.2",
  "sounddevice>=0.5",
  "pynput>=1.8",
  "numpy>=1.26",
  "huggingface_hub>=0.26",
]

[project.scripts]
localvoice = "localvoice.__main__:main"

[dependency-groups]
dev = ["pytest>=8", "ruff>=0.8"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/localvoice"]

[tool.pytest.ini_options]
addopts = "-q"
testpaths = ["tests"]
markers = ["slow: needs real models and hardware; excluded in CI"]

[tool.ruff]
line-length = 100
src = ["src", "tests"]

[tool.ruff.lint]
extend-select = ["I", "E501"]
```

If `uv sync` fails on a version floor (these are post-2025 packages moving fast), relax that floor to whatever uv resolves — do not pin exact versions; `uv.lock` (committed) is the pin.

- [ ] **Step 2: Write `.gitignore`**

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
dist/
.DS_Store
localvoice.local.toml
```

- [ ] **Step 3: Write `LICENSE`** — standard MIT text, `Copyright (c) 2026 Lorenzo Tirelli`.

- [ ] **Step 4: Write `src/localvoice/__init__.py`**

```python
__version__ = "0.1.0"
```

Create empty `tests/__init__.py`. Create `src/localvoice/__main__.py` with a placeholder that Task 16 replaces:

```python
def main() -> None:
    raise SystemExit("localvoice CLI arrives in Task 16")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Write default `localvoice.toml`** (repo defaults; personal overrides go in gitignored `localvoice.local.toml`, merged on top — Task 6 implements the merge)

```toml
[stt]
engine = "mlx_whisper"
model = "mlx-community/whisper-large-v3-turbo"

[llm]
engine = "mlx_lm"
model = "mlx-community/Qwen3.6-35B-A3B-4bit"
deep_model = ""
think = false
max_tokens = 1024
system_prompt = """You are LocalVoice, a warm, sharp voice assistant running fully on this Mac. \
You hear the user through speech recognition and your replies are spoken aloud by TTS. \
Speak naturally in short sentences. Never use markdown, bullets, emoji, or code blocks; \
describe code aloud instead. Default to two or three sentences and offer to go deeper \
rather than lecturing. If the transcript shows you were interrupted mid-sentence, pick up gracefully."""

[tts]
engine = "kokoro_mlx"
model = "prince-canuma/Kokoro-82M"
voice = "af_heart"
speed = 1.0

[keys]
ptt = "cmd_r"
stop = "esc"
debounce_ms = 120

[audio]
input_device = ""
output_device = ""
```

- [ ] **Step 6: Write `.github/workflows/ci.yml`**

```yaml
name: ci
on:
  push: { branches: [main] }
  pull_request:
jobs:
  test:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv sync
      - run: uv run ruff check .
      - run: uv run pytest -m "not slow"
```

- [ ] **Step 7: Write `README.md` stub**

```markdown
# LocalVoice

Local push-to-talk voice assistant for Apple Silicon. Hold right command, talk, release —
a local LLM answers out loud in about a second. No cloud, no VAD, no waiting.

Work in progress; full docs land with v1.
```

- [ ] **Step 8: Verify** — Run: `uv sync && uv run ruff check . && uv run pytest -m "not slow"`
Expected: sync resolves, ruff clean, pytest exits 5/“no tests ran” (that's fine at this point).

- [ ] **Step 9: Commit**

```bash
git add -A && git commit -m "chore: project scaffold, uv packaging, CI"
```

---

### Task 2: Events and the pure state machine

**Files:**
- Create: `src/localvoice/events.py`, `src/localvoice/states.py`
- Test: `tests/test_states.py`

**Interfaces:**
- Consumes: nothing.
- Produces (everything downstream depends on these exact names):

```python
# events.py
class State(Enum): IDLE; LISTENING; PROCESSING; SPEAKING
class EventType(Enum): PTT_DOWN; PTT_UP; ESC; FIRST_AUDIO; RESPONSE_FINISHED; PIPELINE_ERROR
@dataclass(frozen=True) class Event: type: EventType; held_ms: int = 0; message: str = ""; gen: int = -1
class Action(Enum):
    START_CAPTURE; STOP_CAPTURE_AND_RUN; DISCARD_CAPTURE; CANCEL_PIPELINE; FLUSH_AUDIO
    TRUNCATE_HISTORY; COMMIT_TURN; REPORT_ERROR; EARCON_START; EARCON_STOP; EARCON_CANCEL
# states.py
def transition(state: State, event: Event, debounce_ms: int = 120) -> tuple[State, tuple[Action, ...]]
```

- [ ] **Step 1: Write the failing tests** — `tests/test_states.py`:

```python
import pytest

from localvoice.events import Action as A
from localvoice.events import Event, EventType as E, State as S
from localvoice.states import transition


def ev(t: E, held_ms: int = 0) -> Event:
    return Event(type=t, held_ms=held_ms)


BARGE_IN = (A.CANCEL_PIPELINE, A.FLUSH_AUDIO, A.TRUNCATE_HISTORY, A.EARCON_START, A.START_CAPTURE)
CANCEL = (A.CANCEL_PIPELINE, A.FLUSH_AUDIO, A.TRUNCATE_HISTORY, A.EARCON_CANCEL)

CASES = [
    (S.IDLE, ev(E.PTT_DOWN), S.LISTENING, (A.EARCON_START, A.START_CAPTURE)),
    (S.IDLE, ev(E.PTT_UP, 500), S.IDLE, ()),
    (S.IDLE, ev(E.ESC), S.IDLE, ()),
    (S.IDLE, ev(E.RESPONSE_FINISHED), S.IDLE, ()),
    (S.LISTENING, ev(E.PTT_UP, 500), S.PROCESSING, (A.EARCON_STOP, A.STOP_CAPTURE_AND_RUN)),
    (S.LISTENING, ev(E.PTT_UP, 80), S.IDLE, (A.DISCARD_CAPTURE,)),
    (S.LISTENING, ev(E.ESC), S.IDLE, (A.DISCARD_CAPTURE, A.EARCON_CANCEL)),
    (S.LISTENING, ev(E.PTT_DOWN), S.LISTENING, ()),
    (S.PROCESSING, ev(E.FIRST_AUDIO), S.SPEAKING, ()),
    (S.PROCESSING, ev(E.PTT_DOWN), S.LISTENING, BARGE_IN),
    (S.PROCESSING, ev(E.ESC), S.IDLE, CANCEL),
    (S.PROCESSING, ev(E.RESPONSE_FINISHED), S.IDLE, (A.COMMIT_TURN,)),
    (S.PROCESSING, ev(E.PIPELINE_ERROR), S.IDLE, (A.FLUSH_AUDIO, A.TRUNCATE_HISTORY, A.REPORT_ERROR)),
    (S.SPEAKING, ev(E.PTT_DOWN), S.LISTENING, BARGE_IN),
    (S.SPEAKING, ev(E.ESC), S.IDLE, CANCEL),
    (S.SPEAKING, ev(E.RESPONSE_FINISHED), S.IDLE, (A.COMMIT_TURN,)),
    (S.SPEAKING, ev(E.PIPELINE_ERROR), S.IDLE, (A.FLUSH_AUDIO, A.TRUNCATE_HISTORY, A.REPORT_ERROR)),
    (S.SPEAKING, ev(E.FIRST_AUDIO), S.SPEAKING, ()),
]


@pytest.mark.parametrize("state,event,next_state,actions", CASES)
def test_transition(state, event, next_state, actions):
    assert transition(state, event) == (next_state, actions)


def test_debounce_is_configurable():
    assert transition(S.LISTENING, ev(E.PTT_UP, 150), debounce_ms=200)[0] == S.IDLE
    assert transition(S.LISTENING, ev(E.PTT_UP, 150), debounce_ms=100)[0] == S.PROCESSING


def test_every_state_event_pair_is_total():
    for s in S:
        for t in E:
            next_state, actions = transition(s, Event(type=t, held_ms=500))
            assert isinstance(next_state, S) and isinstance(actions, tuple)
```

- [ ] **Step 2: Run tests to verify they fail** — Run: `uv run pytest tests/test_states.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'localvoice.events'`.

- [ ] **Step 3: Implement** — `src/localvoice/events.py`:

```python
from dataclasses import dataclass
from enum import Enum, auto


class State(Enum):
    IDLE = auto()
    LISTENING = auto()
    PROCESSING = auto()
    SPEAKING = auto()


class EventType(Enum):
    PTT_DOWN = auto()
    PTT_UP = auto()
    ESC = auto()
    FIRST_AUDIO = auto()
    RESPONSE_FINISHED = auto()
    PIPELINE_ERROR = auto()


@dataclass(frozen=True)
class Event:
    type: EventType
    held_ms: int = 0
    message: str = ""
    gen: int = -1


class Action(Enum):
    START_CAPTURE = auto()
    STOP_CAPTURE_AND_RUN = auto()
    DISCARD_CAPTURE = auto()
    CANCEL_PIPELINE = auto()
    FLUSH_AUDIO = auto()
    TRUNCATE_HISTORY = auto()
    COMMIT_TURN = auto()
    REPORT_ERROR = auto()
    EARCON_START = auto()
    EARCON_STOP = auto()
    EARCON_CANCEL = auto()
```

`src/localvoice/states.py`:

```python
from localvoice.events import Action as A
from localvoice.events import Event, EventType as E, State as S

_BARGE_IN = (A.CANCEL_PIPELINE, A.FLUSH_AUDIO, A.TRUNCATE_HISTORY, A.EARCON_START, A.START_CAPTURE)
_CANCEL = (A.CANCEL_PIPELINE, A.FLUSH_AUDIO, A.TRUNCATE_HISTORY, A.EARCON_CANCEL)
_ERROR = (A.FLUSH_AUDIO, A.TRUNCATE_HISTORY, A.REPORT_ERROR)

_TABLE: dict[tuple[S, E], tuple[S, tuple[A, ...]]] = {
    (S.IDLE, E.PTT_DOWN): (S.LISTENING, (A.EARCON_START, A.START_CAPTURE)),
    (S.LISTENING, E.ESC): (S.IDLE, (A.DISCARD_CAPTURE, A.EARCON_CANCEL)),
    (S.PROCESSING, E.FIRST_AUDIO): (S.SPEAKING, ()),
    (S.PROCESSING, E.PTT_DOWN): (S.LISTENING, _BARGE_IN),
    (S.PROCESSING, E.ESC): (S.IDLE, _CANCEL),
    (S.PROCESSING, E.RESPONSE_FINISHED): (S.IDLE, (A.COMMIT_TURN,)),
    (S.PROCESSING, E.PIPELINE_ERROR): (S.IDLE, _ERROR),
    (S.SPEAKING, E.PTT_DOWN): (S.LISTENING, _BARGE_IN),
    (S.SPEAKING, E.ESC): (S.IDLE, _CANCEL),
    (S.SPEAKING, E.RESPONSE_FINISHED): (S.IDLE, (A.COMMIT_TURN,)),
    (S.SPEAKING, E.PIPELINE_ERROR): (S.IDLE, _ERROR),
}


def transition(state: S, event: Event, debounce_ms: int = 120) -> tuple[S, tuple[A, ...]]:
    if state is S.LISTENING and event.type is E.PTT_UP:
        if event.held_ms < debounce_ms:
            return S.IDLE, (A.DISCARD_CAPTURE,)
        return S.PROCESSING, (A.EARCON_STOP, A.STOP_CAPTURE_AND_RUN)
    return _TABLE.get((state, event.type), (state, ()))
```

- [ ] **Step 4: Run tests to verify they pass** — Run: `uv run pytest tests/test_states.py -q` — Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/localvoice/events.py src/localvoice/states.py tests/test_states.py
git commit -m "feat: pure push-to-talk state machine with barge-in and debounce"
```

---

### Task 3: Clause chunker

**Files:**
- Create: `src/localvoice/textproc/__init__.py` (empty), `src/localvoice/textproc/chunker.py`
- Test: `tests/test_chunker.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `ClauseChunker(min_chars: int = 60, max_chars: int = 200)` with `feed(delta: str) -> list[str]` and `flush() -> str | None`. Semantics: sentence enders `. ! ? \n` split as soon as the buffer holds ≥15 chars; soft boundaries `, ; : —` split only at ≥`min_chars`; a boundary char only counts if followed by whitespace or buffer-end-at-flush (protects "3.14"); if the buffer exceeds `max_chars` with no boundary, split at the last space. Emitted chunks are stripped; never empty; never split mid-word.

- [ ] **Step 1: Write the failing tests** — `tests/test_chunker.py`:

```python
from localvoice.textproc.chunker import ClauseChunker


def feed_all(c: ClauseChunker, text: str, size: int = 3) -> list[str]:
    out: list[str] = []
    for i in range(0, len(text), size):
        out.extend(c.feed(text[i : i + size]))
    tail = c.flush()
    if tail:
        out.append(tail)
    return out


def test_short_sentence_emits_fast():
    c = ClauseChunker()
    out: list[str] = []
    for delta in ["Sure", ", I can", " help. ", "Second"]:
        out.extend(c.feed(delta))
    assert out == ["Sure, I can help."]


def test_soft_boundary_waits_for_min_chars():
    c = ClauseChunker(min_chars=60)
    text = "one, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve"
    chunks = feed_all(c, text)
    assert len(chunks) >= 2
    assert all(len(ch) >= 15 for ch in chunks[:-1])
    assert " ".join(" ".join(chunks).split()) == " ".join(text.split())


def test_decimal_not_split():
    c = ClauseChunker()
    chunks = feed_all(c, "Pi is 3.14159 which is handy. And that is that.")
    assert chunks[0] == "Pi is 3.14159 which is handy."


def test_hard_flush_at_max_chars_splits_on_space():
    c = ClauseChunker(min_chars=60, max_chars=80)
    chunks = feed_all(c, "word " * 40)
    assert all(len(ch) <= 80 for ch in chunks)
    assert all(not ch.startswith(" ") and not ch.endswith(" ") for ch in chunks)
    assert all("word" == w for ch in chunks for w in ch.split())


def test_flush_returns_remainder_and_empty_is_none():
    c = ClauseChunker()
    assert c.feed("tiny tail") == []
    assert c.flush() == "tiny tail"
    assert c.flush() is None


def test_newline_is_sentence_ender():
    c = ClauseChunker()
    out = c.feed("First line of the answer\nand more")
    assert out == ["First line of the answer"]
```

- [ ] **Step 2: Run tests to verify they fail** — Run: `uv run pytest tests/test_chunker.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement** — `src/localvoice/textproc/chunker.py`:

```python
_SENTENCE = ".!?\n"
_SOFT = ",;:—"
_SENTENCE_FLOOR = 15


class ClauseChunker:
    def __init__(self, min_chars: int = 60, max_chars: int = 200) -> None:
        self.min_chars = min_chars
        self.max_chars = max_chars
        self._buf = ""

    def feed(self, delta: str) -> list[str]:
        self._buf += delta
        out: list[str] = []
        while True:
            cut = self._find_cut()
            if cut is None:
                break
            chunk = self._buf[:cut].strip()
            self._buf = self._buf[cut:].lstrip()
            if chunk:
                out.append(chunk)
        return out

    def flush(self) -> str | None:
        chunk, self._buf = self._buf.strip(), ""
        return chunk or None

    def _find_cut(self) -> int | None:
        for i, ch in enumerate(self._buf):
            if i + 1 >= len(self._buf):
                break  # boundary unconfirmed until next delta or flush()
            confirmed = self._buf[i + 1].isspace()
            if ch in _SENTENCE and (confirmed or ch == "\n") and i + 1 >= _SENTENCE_FLOOR:
                return i + 1
            if ch in _SOFT and confirmed and i + 1 >= self.min_chars:
                return i + 1
        if len(self._buf) > self.max_chars:
            space = self._buf.rfind(" ", 0, self.max_chars)
            if space > 0:
                return space + 1
            return self.max_chars
        return None
```

- [ ] **Step 4: Run tests to verify they pass** — Run: `uv run pytest tests/test_chunker.py -q` — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/localvoice/textproc tests/test_chunker.py
git commit -m "feat: streaming clause chunker for low-latency TTS handoff"
```

---

### Task 4: Streaming sanitizer (think-tags, code fences, speech markup)

**Files:**
- Create: `src/localvoice/textproc/sanitize.py`
- Test: `tests/test_sanitize.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `TextFilter()` — stateful, streaming-safe: `feed(delta: str) -> str`, `finish() -> str`. Drops everything between `<think>` and `</think>` (tags may arrive split across deltas); replaces fenced code blocks (``` … ```) with the single utterance `Code omitted.`.
  - `strip_speech_markup(text: str) -> str` — pure: removes markdown emphasis/heading/link/bullet syntax (keeping readable text), inline backticks, emoji; collapses whitespace.

- [ ] **Step 1: Write the failing tests** — `tests/test_sanitize.py`:

```python
import pytest

from localvoice.textproc.sanitize import TextFilter, strip_speech_markup


def run_filter(deltas: list[str]) -> str:
    f = TextFilter()
    out = "".join(f.feed(d) for d in deltas)
    return out + f.finish()


def test_think_block_dropped_even_split_across_deltas():
    deltas = ["<th", "ink>secret ", "reasoning</thi", "nk>Hello", " there."]
    assert run_filter(deltas) == "Hello there."


def test_no_think_prefix_from_qwen_nothink_mode():
    assert run_filter(["<think>", "\n\n</think>", "\n\nHi!"]) == "\n\nHi!"


def test_unclosed_think_at_finish_yields_nothing():
    assert run_filter(["<think>still reasoning"]) == ""


def test_code_fence_replaced_with_spoken_marker():
    text = "Use this:\n```python\nprint(1)\n```\nDone."
    assert run_filter([text]) == "Use this:\nCode omitted.\nDone."


def test_plain_text_passes_through_unchanged():
    assert run_filter(["Just a normal ", "answer."]) == "Just a normal answer."


@pytest.mark.parametrize(
    "raw,clean",
    [
        ("**Bold** and *italic* text", "Bold and italic text"),
        ("# Heading\nBody", "Heading Body"),
        ("See [the docs](https://x.y) now", "See the docs now"),
        ("- item one\n- item two", "item one item two"),
        ("Inline `code` here", "Inline code here"),
        ("call my_variable_name now", "call my variable name now"),
        ("Nice \U0001f600 day ✨", "Nice day"),
        ("  spaced   out  ", "spaced out"),
    ],
)
def test_strip_speech_markup(raw, clean):
    assert strip_speech_markup(raw) == clean
```

- [ ] **Step 2: Run tests to verify they fail** — Run: `uv run pytest tests/test_sanitize.py -q`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement** — `src/localvoice/textproc/sanitize.py`:

```python
import re

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_FENCE = "```"
_CODE_MARKER = "Code omitted."

_EMOJI = re.compile(
    "["
    "\U0001f300-\U0001faff"
    "\U00002600-\U000027bf"
    "\U0001f000-\U0001f0ff"
    "⬀-⯿←-⇿️"
    "]+"
)


def _longest_suffix_prefix(text: str, token: str) -> int:
    for k in range(min(len(text), len(token) - 1), 0, -1):
        if text.endswith(token[:k]):
            return k
    return 0


class TextFilter:
    def __init__(self) -> None:
        self._buf = ""
        self._in_think = False
        self._in_fence = False

    def feed(self, delta: str) -> str:
        self._buf += delta
        out: list[str] = []
        while True:
            if self._in_think or self._in_fence:
                token = _THINK_CLOSE if self._in_think else _FENCE
                j = self._buf.find(token)
                if j < 0:
                    hold = _longest_suffix_prefix(self._buf, token)
                    self._buf = self._buf[len(self._buf) - hold :]
                    break
                self._buf = self._buf[j + len(token) :]
                self._in_think = False
                self._in_fence = False
                continue
            i_think = self._buf.find(_THINK_OPEN)
            i_fence = self._buf.find(_FENCE)
            candidates = [(i, t) for i, t in ((i_think, _THINK_OPEN), (i_fence, _FENCE)) if i >= 0]
            if not candidates:
                hold = max(
                    _longest_suffix_prefix(self._buf, _THINK_OPEN),
                    _longest_suffix_prefix(self._buf, _FENCE),
                )
                emit_upto = len(self._buf) - hold
                out.append(self._buf[:emit_upto])
                self._buf = self._buf[emit_upto:]
                break
            i, tok = min(candidates)
            out.append(self._buf[:i])
            self._buf = self._buf[i + len(tok) :]
            if tok == _THINK_OPEN:
                self._in_think = True
            else:
                self._in_fence = True
                nl = self._buf.find("\n")
                if nl >= 0:
                    self._buf = self._buf[nl + 1 :]
                out.append(_CODE_MARKER)
        return "".join(out)

    def finish(self) -> str:
        if self._in_think or self._in_fence:
            self._buf = ""
            return ""
        out, self._buf = self._buf, ""
        return out


def strip_speech_markup(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.M)
    text = re.sub(r"(\*\*|__|\*|`)", "", text)
    text = text.replace("_", " ")  # snake_case identifiers must stay speakable: my_var -> "my var"
    text = _EMOJI.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()
```

Fence-open note: the language tag (```` ```python ````) is consumed up to and including its newline when the newline has already streamed in; if it hasn't yet, the leftover language chars sit in `_buf` and are discarded by the fence-body suppression on subsequent feeds — both paths keep spoken output identical.

- [ ] **Step 4: Run tests to verify they pass** — Run: `uv run pytest tests/test_sanitize.py -q` — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/localvoice/textproc/sanitize.py tests/test_sanitize.py
git commit -m "feat: streaming sanitizer for think-tags, code fences, and speech markup"
```

---

### Task 5: Transcript with barge-in truncation

**Files:**
- Create: `src/localvoice/transcript.py`
- Test: `tests/test_transcript.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Transcript(system_prompt: str)` with:
  - `begin_turn(user_text: str) -> None`
  - `add_clause(text: str) -> int` — appends a pending assistant clause, returns its tag (0-based int, unique per response)
  - `messages() -> list[dict]` — `[{"role": "system", ...}, ...committed..., {"role": "user", <pending user>}]`; pending assistant clauses are NOT included (used to prompt the LLM)
  - `commit() -> None` — commits pending user + all pending clauses joined with a space
  - `truncate_commit(spoken_tags: set[int]) -> None` — commits pending user + only clauses whose tag is in `spoken_tags`, appending `" ..."` marker if anything was cut; if no clause was spoken, the assistant turn is recorded as `"..."` (the model should know it said nothing audible)
  - `abort_pending() -> None` — drops the pending turn entirely (silent discard)
  - `history() -> list[dict]` — committed messages without the system prompt (for display/logging)

- [ ] **Step 1: Write the failing tests** — `tests/test_transcript.py`:

```python
from localvoice.transcript import Transcript


def make() -> Transcript:
    t = Transcript(system_prompt="be brief")
    t.begin_turn("hello")
    t.add_clause("Hi there.")
    t.add_clause("How can I help?")
    return t


def test_messages_exclude_pending_assistant():
    t = make()
    assert t.messages() == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hello"},
    ]


def test_commit_joins_clauses():
    t = make()
    t.commit()
    assert t.history() == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Hi there. How can I help?"},
    ]


def test_truncate_commit_keeps_spoken_prefix_with_marker():
    t = make()
    t.truncate_commit({0})
    assert t.history()[-1] == {"role": "assistant", "content": "Hi there. ..."}


def test_truncate_commit_nothing_spoken():
    t = make()
    t.truncate_commit(set())
    assert t.history()[-1] == {"role": "assistant", "content": "..."}


def test_abort_pending_drops_turn():
    t = make()
    t.abort_pending()
    assert t.history() == []
    assert t.messages() == [{"role": "system", "content": "be brief"}]


def test_next_turn_builds_on_committed():
    t = make()
    t.commit()
    t.begin_turn("second")
    assert t.messages()[-1] == {"role": "user", "content": "second"}
    assert len(t.messages()) == 4  # system, user, assistant, user


def test_tags_are_sequential_per_response():
    t = Transcript("s")
    t.begin_turn("u")
    assert t.add_clause("a") == 0
    assert t.add_clause("b") == 1
    t.commit()
    t.begin_turn("u2")
    assert t.add_clause("c") == 0
```

- [ ] **Step 2: Run tests to verify they fail** — Run: `uv run pytest tests/test_transcript.py -q` — Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement** — `src/localvoice/transcript.py`:

```python
class Transcript:
    def __init__(self, system_prompt: str) -> None:
        self._system = system_prompt
        self._committed: list[dict] = []
        self._pending_user: str | None = None
        self._pending_clauses: list[str] = []

    def begin_turn(self, user_text: str) -> None:
        self._pending_user = user_text
        self._pending_clauses = []

    def add_clause(self, text: str) -> int:
        self._pending_clauses.append(text)
        return len(self._pending_clauses) - 1

    def messages(self) -> list[dict]:
        msgs = [{"role": "system", "content": self._system}, *self._committed]
        if self._pending_user is not None:
            msgs.append({"role": "user", "content": self._pending_user})
        return msgs

    def commit(self) -> None:
        self._finish(" ".join(self._pending_clauses))

    def truncate_commit(self, spoken_tags: set[int]) -> None:
        if self._pending_user is None:
            return
        spoken = [c for i, c in enumerate(self._pending_clauses) if i in spoken_tags]
        cut = len(spoken) < len(self._pending_clauses)
        text = " ".join(spoken)
        self._finish((text + " ..." if text else "...") if cut else text)

    def abort_pending(self) -> None:
        self._pending_user = None
        self._pending_clauses = []

    def history(self) -> list[dict]:
        return list(self._committed)

    def _finish(self, assistant_text: str) -> None:
        if self._pending_user is None:
            return
        self._committed.append({"role": "user", "content": self._pending_user})
        self._committed.append({"role": "assistant", "content": assistant_text})
        self.abort_pending()
```

- [ ] **Step 4: Run tests to verify they pass** — Run: `uv run pytest tests/test_transcript.py -q` — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/localvoice/transcript.py tests/test_transcript.py
git commit -m "feat: transcript with barge-in truncation to spoken clauses"
```

---

### Task 6: Config loading with local overlay

**Files:**
- Create: `src/localvoice/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: `localvoice.toml` schema from Task 1.
- Produces:

```python
@dataclass class SttConfig: engine: str; model: str
@dataclass class LlmConfig: engine: str; model: str; deep_model: str; think: bool; max_tokens: int; system_prompt: str
@dataclass class TtsConfig: engine: str; model: str; voice: str; speed: float
@dataclass class KeysConfig: ptt: str; stop: str; debounce_ms: int
@dataclass class AudioConfig: input_device: str; output_device: str
@dataclass class Config: stt: SttConfig; llm: LlmConfig; tts: TtsConfig; keys: KeysConfig; audio: AudioConfig
class ConfigError(Exception)
def load_config(path: Path, *, deep: bool = False) -> Config
```

  `load_config` reads `path`, then merges `<stem>.local.toml` beside it if present (local wins, per-key deep merge). `deep=True` swaps `llm.model` for `llm.deep_model` (raises `ConfigError` if empty). Unknown sections/keys raise `ConfigError` naming the key. Missing file raises `ConfigError` with the path.

- [ ] **Step 1: Write the failing tests** — `tests/test_config.py`:

```python
from pathlib import Path

import pytest

from localvoice.config import Config, ConfigError, load_config

BASE = """
[stt]
engine = "mlx_whisper"
model = "repo/whisper"
[llm]
engine = "mlx_lm"
model = "repo/big"
deep_model = "repo/deep"
think = false
max_tokens = 512
system_prompt = "be brief"
[tts]
engine = "kokoro_mlx"
model = "repo/kokoro"
voice = "af_heart"
speed = 1.0
[keys]
ptt = "cmd_r"
stop = "esc"
debounce_ms = 120
[audio]
input_device = ""
output_device = ""
"""


def write(tmp_path: Path, text: str, name: str = "localvoice.toml") -> Path:
    p = tmp_path / name
    p.write_text(text)
    return p


def test_loads_full_config(tmp_path):
    cfg = load_config(write(tmp_path, BASE))
    assert isinstance(cfg, Config)
    assert cfg.llm.model == "repo/big"
    assert cfg.keys.debounce_ms == 120
    assert cfg.tts.speed == 1.0


def test_local_overlay_wins(tmp_path):
    write(tmp_path, BASE)
    write(tmp_path, '[llm]\nmodel = "local/path"\n', "localvoice.local.toml")
    cfg = load_config(tmp_path / "localvoice.toml")
    assert cfg.llm.model == "local/path"
    assert cfg.llm.max_tokens == 512  # untouched keys survive the merge


def test_deep_flag_swaps_model(tmp_path):
    cfg = load_config(write(tmp_path, BASE), deep=True)
    assert cfg.llm.model == "repo/deep"


def test_deep_flag_without_deep_model_raises(tmp_path):
    text = BASE.replace('deep_model = "repo/deep"', 'deep_model = ""')
    with pytest.raises(ConfigError, match="deep_model"):
        load_config(write(tmp_path, text), deep=True)


def test_unknown_key_raises(tmp_path):
    with pytest.raises(ConfigError, match="volume"):
        load_config(write(tmp_path, BASE + "\n[tts2]\nvolume = 3\n").parent / "localvoice.toml")


def test_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="nope.toml"):
        load_config(tmp_path / "nope.toml")
```

(Adjust the unknown-key test: `[tts2]` is an unknown *section*; the error message must contain `tts2`. Use `match="tts2"`.)

- [ ] **Step 2: Run tests to verify they fail** — Run: `uv run pytest tests/test_config.py -q` — Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement** — `src/localvoice/config.py`:

```python
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path


class ConfigError(Exception):
    pass


@dataclass
class SttConfig:
    engine: str = "mlx_whisper"
    model: str = "mlx-community/whisper-large-v3-turbo"


@dataclass
class LlmConfig:
    engine: str = "mlx_lm"
    model: str = "mlx-community/Qwen3.6-35B-A3B-4bit"
    deep_model: str = ""
    think: bool = False
    max_tokens: int = 1024
    system_prompt: str = "You are a helpful voice assistant. Keep answers short and spoken-friendly."


@dataclass
class TtsConfig:
    engine: str = "kokoro_mlx"
    model: str = "prince-canuma/Kokoro-82M"
    voice: str = "af_heart"
    speed: float = 1.0


@dataclass
class KeysConfig:
    ptt: str = "cmd_r"
    stop: str = "esc"
    debounce_ms: int = 120


@dataclass
class AudioConfig:
    input_device: str = ""
    output_device: str = ""


@dataclass
class Config:
    stt: SttConfig
    llm: LlmConfig
    tts: TtsConfig
    keys: KeysConfig
    audio: AudioConfig


_SECTIONS = {"stt": SttConfig, "llm": LlmConfig, "tts": TtsConfig, "keys": KeysConfig, "audio": AudioConfig}


def _merge(base: dict, overlay: dict) -> dict:
    out = {k: dict(v) for k, v in base.items()}
    for section, values in overlay.items():
        out.setdefault(section, {}).update(values)
    return out


def _build(data: dict) -> Config:
    kwargs = {}
    for section, values in data.items():
        cls = _SECTIONS.get(section)
        if cls is None:
            raise ConfigError(f"unknown config section [{section}]")
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ConfigError(f"unknown key in [{section}]: {sorted(unknown)[0]}")
        kwargs[section] = cls(**values)
    for section, cls in _SECTIONS.items():
        kwargs.setdefault(section, cls())
    return Config(**kwargs)


def load_config(path: Path, *, deep: bool = False) -> Config:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config not found: {path}")
    data = tomllib.loads(path.read_text())
    local = path.with_name(path.stem + ".local" + path.suffix)
    if local.exists():
        data = _merge(data, tomllib.loads(local.read_text()))
    cfg = _build(data)
    if deep:
        if not cfg.llm.deep_model:
            raise ConfigError("--deep requested but [llm].deep_model is empty")
        cfg.llm.model = cfg.llm.deep_model
    return cfg
```

- [ ] **Step 4: Run tests to verify they pass** — Run: `uv run pytest tests/test_config.py -q` — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/localvoice/config.py tests/test_config.py
git commit -m "feat: TOML config with gitignored local overlay and deep-model flag"
```

---

### Task 7: Earcons and the playback queue (spoken-text accounting)

**Files:**
- Create: `src/localvoice/audio/__init__.py` (empty), `src/localvoice/audio/earcons.py`, `src/localvoice/audio/player.py`
- Test: `tests/test_player.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `earcons.tone(freq_hz: float, ms: int, sr: int = 24000, amp: float = 0.15) -> np.ndarray` (float32 mono, 3 ms linear fade in/out) and `EARCONS = {"start": tone(880, 60), "stop": tone(660, 50), "cancel": tone(220, 90)}`.
  - `PlaybackQueue(on_response_finished: Callable[[], None])` — pure logic, no audio device:
    - `submit(samples: np.ndarray, tag: int) -> None` (response audio, clause tag)
    - `submit_raw(samples: np.ndarray) -> None` (earcons; excluded from accounting)
    - `mark_end() -> None` — response fully submitted
    - `flush() -> None` — drop all queued response audio (earcons too), reset end mark
    - `pull(n: int) -> np.ndarray` — exactly n float32 samples (zero-padded when starved); fires `on_response_finished` exactly once when the response is fully consumed after `mark_end`
    - `spoken_tags() -> set[int]` — tags with at least one sample pulled
  - `AudioPlayer(cfg: AudioConfig, on_response_finished)` — thin sounddevice adapter around one always-open `OutputStream(samplerate=24000, channels=1, dtype="float32")` whose callback delegates to `PlaybackQueue.pull`; exposes the same submit/flush/mark_end/spoken_tags API plus `start()`/`stop()`. Not unit-tested (hardware); its callback contains zero logic beyond `outdata[:, 0] = queue.pull(frames)`.

- [ ] **Step 1: Write the failing tests** — `tests/test_player.py`:

```python
import numpy as np

from localvoice.audio.earcons import EARCONS, tone
from localvoice.audio.player import PlaybackQueue


def test_tone_shape_and_fade():
    t = tone(880, 60, sr=24000)
    assert t.dtype == np.float32 and t.ndim == 1
    assert len(t) == int(24000 * 0.060)
    assert abs(t[0]) < 1e-4 and abs(t[-1]) < 1e-4
    assert 0.05 < np.abs(t).max() <= 0.2
    assert set(EARCONS) == {"start", "stop", "cancel"}


def make(fired: list) -> PlaybackQueue:
    return PlaybackQueue(on_response_finished=lambda: fired.append(True))


def test_pull_pads_with_zeros_when_empty():
    q = make([])
    out = q.pull(64)
    assert out.shape == (64,) and not out.any()


def test_spoken_tags_track_partial_consumption():
    fired: list = []
    q = make(fired)
    q.submit(np.ones(100, np.float32), tag=0)
    q.submit(np.ones(100, np.float32), tag=1)
    q.pull(120)  # all of tag 0, some of tag 1
    assert q.spoken_tags() == {0, 1}
    q2 = make([])
    q2.submit(np.ones(100, np.float32), tag=0)
    q2.submit(np.ones(100, np.float32), tag=1)
    q2.pull(80)
    assert q2.spoken_tags() == {0}


def test_response_finished_fires_once_after_mark_end():
    fired: list = []
    q = make(fired)
    q.submit(np.ones(100, np.float32), tag=0)
    q.pull(200)
    assert fired == []  # not marked yet
    q.mark_end()
    q.pull(10)
    q.pull(10)
    assert fired == [True]


def test_flush_drops_audio_and_resets():
    fired: list = []
    q = make(fired)
    q.submit(np.ones(100, np.float32), tag=0)
    q.mark_end()
    q.flush()
    assert not q.pull(50).any()
    assert fired == []  # flushed response never "finishes"
    assert q.spoken_tags() == set()


def test_earcon_audio_plays_but_is_not_accounted():
    q = make([])
    q.submit_raw(np.full(50, 0.5, np.float32))
    out = q.pull(50)
    assert out.any()
    assert q.spoken_tags() == set()


def test_earcon_and_response_play_in_submit_order():
    q = make([])
    q.submit_raw(np.full(10, 0.5, np.float32))
    q.submit(np.full(10, 0.9, np.float32), tag=0)
    first = q.pull(10)
    second = q.pull(10)
    assert np.allclose(first, 0.5) and np.allclose(second, 0.9)
```

- [ ] **Step 2: Run tests to verify they fail** — Run: `uv run pytest tests/test_player.py -q` — Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement** — `src/localvoice/audio/earcons.py`:

```python
import numpy as np


def tone(freq_hz: float, ms: int, sr: int = 24000, amp: float = 0.15) -> np.ndarray:
    n = int(sr * ms / 1000)
    t = np.arange(n, dtype=np.float32) / sr
    wave = (amp * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
    fade = min(int(sr * 0.003), n // 2)
    if fade:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        wave[:fade] *= ramp
        wave[-fade:] *= ramp[::-1]
    return wave


EARCONS = {"start": tone(880, 60), "stop": tone(660, 50), "cancel": tone(220, 90)}
```

`src/localvoice/audio/player.py`:

```python
import threading
from collections import deque
from collections.abc import Callable

import numpy as np

from localvoice.config import AudioConfig

_RESPONSE, _RAW = 0, 1


class PlaybackQueue:
    def __init__(self, on_response_finished: Callable[[], None]) -> None:
        self._on_finished = on_response_finished
        self._lock = threading.Lock()
        self._chunks: deque[tuple[np.ndarray, int, int | None]] = deque()  # (samples, kind, tag)
        self._spoken: set[int] = set()
        self._ended = False
        self._fired = False

    def submit(self, samples: np.ndarray, tag: int) -> None:
        with self._lock:
            self._chunks.append((np.asarray(samples, np.float32), _RESPONSE, tag))

    def submit_raw(self, samples: np.ndarray) -> None:
        with self._lock:
            self._chunks.append((np.asarray(samples, np.float32), _RAW, None))

    def mark_end(self) -> None:
        with self._lock:
            self._ended = True

    def flush(self) -> None:
        with self._lock:
            self._chunks.clear()
            self._ended = False
            self._fired = False
            self._spoken = set()

    def spoken_tags(self) -> set[int]:
        with self._lock:
            return set(self._spoken)

    def pull(self, n: int) -> np.ndarray:
        out = np.zeros(n, np.float32)
        fire = False
        with self._lock:
            i = 0
            while i < n and self._chunks:
                samples, kind, tag = self._chunks[0]
                if kind == _RESPONSE and tag is not None:
                    self._spoken.add(tag)
                take = min(n - i, len(samples))
                out[i : i + take] = samples[:take]
                if take == len(samples):
                    self._chunks.popleft()
                else:
                    self._chunks[0] = (samples[take:], kind, tag)
                i += take
            response_left = any(k == _RESPONSE for _, k, _ in self._chunks)
            if self._ended and not response_left and not self._fired:
                self._fired = True
                fire = True
        if fire:
            self._on_finished()
        return out


class AudioPlayer:
    def __init__(self, cfg: AudioConfig, on_response_finished: Callable[[], None]) -> None:
        self.queue = PlaybackQueue(on_response_finished)
        self._cfg = cfg
        self._stream = None

    def start(self) -> None:
        import sounddevice as sd

        device = self._cfg.output_device or None

        def callback(outdata, frames, _time, _status):
            outdata[:, 0] = self.queue.pull(frames)

        self._stream = sd.OutputStream(
            samplerate=24000, channels=1, dtype="float32", device=device, callback=callback
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def submit(self, samples: np.ndarray, tag: int) -> None:
        self.queue.submit(samples, tag)

    def submit_raw(self, samples: np.ndarray) -> None:
        self.queue.submit_raw(samples)

    def mark_end(self) -> None:
        self.queue.mark_end()

    def flush(self) -> None:
        self.queue.flush()

    def spoken_tags(self) -> set[int]:
        return self.queue.spoken_tags()
```

Note the `test_response_finished_fires_once_after_mark_end` sequencing: `pull` after everything was already consumed but before `mark_end` must not fire; the first `pull` after `mark_end` fires. The implementation above satisfies this because the fire check runs on every pull.

- [ ] **Step 4: Run tests to verify they pass** — Run: `uv run pytest tests/test_player.py -q` — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/localvoice/audio tests/test_player.py
git commit -m "feat: playback queue with spoken-clause accounting, earcons, sounddevice adapter"
```

---

### Task 8: Gated microphone capture

**Files:**
- Create: `src/localvoice/audio/capture.py`
- Test: `tests/test_capture.py`

**Interfaces:**
- Consumes: `AudioConfig` (Task 6).
- Produces:
  - `GatedBuffer(max_seconds: float = 300.0, sr: int = 16000)` — pure: `arm()`, `disarm() -> np.ndarray` (concatenated float32 of frames received while armed), `discard()`, `write(frames: np.ndarray)` (no-op unless armed; drops frames beyond `max_seconds`).
  - `MicCapture(cfg: AudioConfig)` — thin adapter: always-open `InputStream(samplerate=16000, channels=1, dtype="float32")` whose callback does `buffer.write(indata[:, 0].copy())`; exposes `start()`, `stop()`, `arm()`, `disarm()`, `discard()`. The stream stays open for the whole session; audio is retained only between `arm()` and `disarm()` — README documents this.
  - `rms(audio: np.ndarray) -> float` helper in the same module.

- [ ] **Step 1: Write the failing tests** — `tests/test_capture.py`:

```python
import numpy as np

from localvoice.audio.capture import GatedBuffer, rms


def test_ignores_frames_when_not_armed():
    b = GatedBuffer()
    b.write(np.ones(160, np.float32))
    b.arm()
    b.write(np.full(160, 0.5, np.float32))
    audio = b.disarm()
    assert len(audio) == 160 and np.allclose(audio, 0.5)


def test_arm_clears_previous_audio():
    b = GatedBuffer()
    b.arm()
    b.write(np.ones(10, np.float32))
    b.arm()
    b.write(np.full(5, 0.2, np.float32))
    assert len(b.disarm()) == 5


def test_discard_empties():
    b = GatedBuffer()
    b.arm()
    b.write(np.ones(10, np.float32))
    b.discard()
    b.arm()
    assert len(b.disarm()) == 0


def test_caps_at_max_seconds():
    b = GatedBuffer(max_seconds=0.01, sr=16000)  # 160 samples cap
    b.arm()
    for _ in range(5):
        b.write(np.ones(100, np.float32))
    assert len(b.disarm()) == 160


def test_rms():
    assert rms(np.zeros(100, np.float32)) == 0.0
    assert abs(rms(np.full(100, 0.5, np.float32)) - 0.5) < 1e-6
    assert rms(np.array([], np.float32)) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail** — Run: `uv run pytest tests/test_capture.py -q` — Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement** — `src/localvoice/audio/capture.py`:

```python
import math
import threading

import numpy as np

from localvoice.config import AudioConfig


def rms(audio: np.ndarray) -> float:
    if len(audio) == 0:
        return 0.0
    return float(math.sqrt(float(np.mean(np.square(audio, dtype=np.float64)))))


class GatedBuffer:
    def __init__(self, max_seconds: float = 300.0, sr: int = 16000) -> None:
        self._max_samples = int(max_seconds * sr)
        self._lock = threading.Lock()
        self._armed = False
        self._frames: list[np.ndarray] = []
        self._total = 0

    def arm(self) -> None:
        with self._lock:
            self._armed = True
            self._frames = []
            self._total = 0

    def disarm(self) -> np.ndarray:
        with self._lock:
            self._armed = False
            frames, self._frames = self._frames, []
            self._total = 0
        if not frames:
            return np.zeros(0, np.float32)
        return np.concatenate(frames).astype(np.float32)

    def discard(self) -> None:
        with self._lock:
            self._armed = False
            self._frames = []
            self._total = 0

    def write(self, frames: np.ndarray) -> None:
        with self._lock:
            if not self._armed or self._total >= self._max_samples:
                return
            room = self._max_samples - self._total
            chunk = frames[:room]
            self._frames.append(np.asarray(chunk, np.float32))
            self._total += len(chunk)


class MicCapture:
    def __init__(self, cfg: AudioConfig) -> None:
        self.buffer = GatedBuffer()
        self._cfg = cfg
        self._stream = None

    def start(self) -> None:
        import sounddevice as sd

        device = self._cfg.input_device or None

        def callback(indata, _frames, _time, _status):
            self.buffer.write(indata[:, 0].copy())

        self._stream = sd.InputStream(
            samplerate=16000, channels=1, dtype="float32", device=device, callback=callback
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def arm(self) -> None:
        self.buffer.arm()

    def disarm(self) -> np.ndarray:
        return self.buffer.disarm()

    def discard(self) -> None:
        self.buffer.discard()
```

- [ ] **Step 4: Run tests to verify they pass** — Run: `uv run pytest tests/test_capture.py -q` — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/localvoice/audio/capture.py tests/test_capture.py
git commit -m "feat: gated mic capture buffer with always-open stream adapter"
```

---

### Task 9: Engine protocols and shared test fakes

**Files:**
- Create: `src/localvoice/stt/__init__.py`, `src/localvoice/stt/base.py`, `src/localvoice/llm/__init__.py`, `src/localvoice/llm/base.py`, `src/localvoice/tts/__init__.py`, `src/localvoice/tts/base.py`, `tests/fakes.py`
- Test: `tests/test_fakes.py` (tiny — fakes must satisfy the protocols)

**Interfaces:**
- Consumes: nothing.
- Produces (all later tasks use these exact signatures):

```python
# stt/base.py
class STTEngine(Protocol):
    def load(self) -> None: ...
    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str: ...
# llm/base.py
Message = dict  # {"role": str, "content": str}
class LLMEngine(Protocol):
    def load(self) -> None: ...
    def stream(self, messages: list[Message], *, think: bool) -> Iterator[str]: ...
# tts/base.py
class TTSEngine(Protocol):
    def load(self) -> None: ...
    def synthesize(self, text: str) -> Iterator[np.ndarray]: ...
```

  Fakes in `tests/fakes.py`: `FakeSTT(text="hello")`, `FakeLLM(deltas: list[str])` (records `last_messages`, `last_think`), `FakeTTS(chunk_len=100, chunks_per_text=2, delay_s=0.0)` (yields `np.full(chunk_len, 0.1)` per chunk; optional `time.sleep(delay_s)` between chunks so cancellation tests can interleave), `FakePlayer()` implementing the `AudioPlayer` surface (`submit/submit_raw/mark_end/flush/spoken_tags`) recording calls into `.log` and treating every submitted tag as spoken.

- [ ] **Step 1: Write the failing test** — `tests/test_fakes.py`:

```python
import numpy as np

from localvoice.llm.base import LLMEngine
from localvoice.stt.base import STTEngine
from localvoice.tts.base import TTSEngine
from tests.fakes import FakeLLM, FakePlayer, FakeSTT, FakeTTS


def test_fakes_satisfy_protocols():
    stt: STTEngine = FakeSTT()
    llm: LLMEngine = FakeLLM(deltas=["a", "b"])
    tts: TTSEngine = FakeTTS()
    stt.load(), llm.load(), tts.load()
    assert stt.transcribe(np.zeros(16000, np.float32), 16000) == "hello"
    assert list(llm.stream([{"role": "user", "content": "x"}], think=False)) == ["a", "b"]
    chunks = list(tts.synthesize("hi"))
    assert len(chunks) == 2 and all(c.dtype == np.float32 for c in chunks)
    p = FakePlayer()
    p.submit(chunks[0], tag=0)
    p.mark_end()
    assert p.spoken_tags() == {0}
    assert [entry[0] for entry in p.log] == ["submit", "mark_end"]
```

- [ ] **Step 2: Run test to verify it fails** — Run: `uv run pytest tests/test_fakes.py -q` — Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement** — `src/localvoice/stt/base.py`:

```python
from typing import Protocol

import numpy as np


class STTEngine(Protocol):
    def load(self) -> None: ...

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str: ...
```

`src/localvoice/llm/base.py`:

```python
from collections.abc import Iterator
from typing import Protocol

Message = dict


class LLMEngine(Protocol):
    def load(self) -> None: ...

    def stream(self, messages: list[Message], *, think: bool) -> Iterator[str]: ...
```

`src/localvoice/tts/base.py`:

```python
from collections.abc import Iterator
from typing import Protocol

import numpy as np


class TTSEngine(Protocol):
    def load(self) -> None: ...

    def synthesize(self, text: str) -> Iterator[np.ndarray]: ...
```

`tests/fakes.py`:

```python
import time
from collections.abc import Iterator

import numpy as np


class FakeSTT:
    def __init__(self, text: str = "hello") -> None:
        self.text = text

    def load(self) -> None:
        pass

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        return self.text


class FakeLLM:
    def __init__(self, deltas: list[str]) -> None:
        self.deltas = deltas
        self.last_messages: list[dict] | None = None
        self.last_think: bool | None = None

    def load(self) -> None:
        pass

    def stream(self, messages: list[dict], *, think: bool) -> Iterator[str]:
        self.last_messages = messages
        self.last_think = think
        yield from self.deltas


class FakeTTS:
    def __init__(self, chunk_len: int = 100, chunks_per_text: int = 2, delay_s: float = 0.0) -> None:
        self.chunk_len = chunk_len
        self.chunks_per_text = chunks_per_text
        self.delay_s = delay_s
        self.texts: list[str] = []

    def load(self) -> None:
        pass

    def synthesize(self, text: str) -> Iterator[np.ndarray]:
        self.texts.append(text)
        for _ in range(self.chunks_per_text):
            if self.delay_s:
                time.sleep(self.delay_s)
            yield np.full(self.chunk_len, 0.1, np.float32)


class FakePlayer:
    def __init__(self) -> None:
        self.log: list[tuple] = []
        self._tags: set[int] = set()

    def submit(self, samples: np.ndarray, tag: int) -> None:
        self.log.append(("submit", tag, len(samples)))
        self._tags.add(tag)

    def submit_raw(self, samples: np.ndarray) -> None:
        self.log.append(("submit_raw", len(samples)))

    def mark_end(self) -> None:
        self.log.append(("mark_end",))

    def flush(self) -> None:
        self.log.append(("flush",))
        self._tags = set()

    def spoken_tags(self) -> set[int]:
        return set(self._tags)
```

- [ ] **Step 4: Run test to verify it passes** — Run: `uv run pytest tests/test_fakes.py -q` — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/localvoice/stt src/localvoice/llm src/localvoice/tts tests/fakes.py tests/test_fakes.py
git commit -m "feat: engine protocols and shared test fakes"
```

---

### Task 10: Whisper STT engine

**Files:**
- Create: `src/localvoice/stt/whisper_mlx.py`

**Interfaces:**
- Consumes: `STTEngine` protocol shape, `SttConfig`.
- Produces: `WhisperMlxEngine(cfg: SttConfig)` implementing `STTEngine`. `load()` warms the model with 0.2 s of silence; `transcribe` returns stripped text.

- [ ] **Step 1: Implement** (thin hardware wrapper — no unit test; exercised by the Task 18 slow smoke test):

```python
import numpy as np

from localvoice.config import SttConfig


class WhisperMlxEngine:
    def __init__(self, cfg: SttConfig) -> None:
        self._cfg = cfg

    def load(self) -> None:
        self.transcribe(np.zeros(3200, np.float32), 16000)

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        import mlx_whisper

        assert sample_rate == 16000, "whisper path expects 16 kHz mono"
        result = mlx_whisper.transcribe(audio, path_or_hf_repo=self._cfg.model)
        return str(result["text"]).strip()
```

- [ ] **Step 2: Import check** — Run: `uv run python -c "from localvoice.stt.whisper_mlx import WhisperMlxEngine; print('ok')"`
Expected: `ok` (mlx_whisper is imported lazily, so this works without the model downloaded).

- [ ] **Step 3: Optional manual smoke (only if the whisper model is already in the HF cache)** — Run:
`uv run python -c "
import numpy as np
from localvoice.config import SttConfig
from localvoice.stt.whisper_mlx import WhisperMlxEngine
e = WhisperMlxEngine(SttConfig()); e.load(); print(repr(e.transcribe(np.zeros(16000, np.float32), 16000)))"`
Expected: empty-ish string, no crash. Skip if the model isn't downloaded yet (Task 16's `setup` handles downloads).

- [ ] **Step 4: Commit**

```bash
git add src/localvoice/stt/whisper_mlx.py
git commit -m "feat: mlx-whisper STT engine"
```

---

### Task 11: mlx-lm LLM engine with incremental prompt cache

**Files:**
- Create: `src/localvoice/llm/mlx_lm_engine.py`
- Test: `tests/test_llm_cache_plan.py`

**Interfaces:**
- Consumes: `LLMEngine` protocol, `LlmConfig`, `Message`.
- Produces: `MlxLmEngine(cfg: LlmConfig)` implementing `LLMEngine`, plus the pure function `plan_prompt(committed: list[Message], messages: list[Message]) -> str` returning `"incremental"` (messages == committed + exactly one new user message, and committed non-empty) or `"full"` otherwise. Cache policy: incremental turns reuse the persistent KV cache and template only the new user message; any history divergence (first turn, barge-in truncation, deep-model swap) resets the cache and re-prefills the full history. After each completed stream, the engine records `committed = messages + [assistant reply]`.

- [ ] **Step 1: Write the failing test** — `tests/test_llm_cache_plan.py`:

```python
from localvoice.llm.mlx_lm_engine import plan_prompt

SYS = {"role": "system", "content": "s"}
U1 = {"role": "user", "content": "one"}
A1 = {"role": "assistant", "content": "ans"}
U2 = {"role": "user", "content": "two"}


def test_first_turn_is_full():
    assert plan_prompt([], [SYS, U1]) == "full"


def test_appended_user_is_incremental():
    assert plan_prompt([SYS, U1, A1], [SYS, U1, A1, U2]) == "incremental"


def test_diverged_history_is_full():
    truncated = [SYS, U1, {"role": "assistant", "content": "ans ..."}]
    assert plan_prompt([SYS, U1, A1], truncated + [U2]) == "full"


def test_non_user_tail_is_full():
    assert plan_prompt([SYS, U1], [SYS, U1, A1]) == "full"
```

- [ ] **Step 2: Run test to verify it fails** — Run: `uv run pytest tests/test_llm_cache_plan.py -q` — Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement** — `src/localvoice/llm/mlx_lm_engine.py`:

```python
from collections.abc import Iterator

from localvoice.config import LlmConfig
from localvoice.llm.base import Message


def plan_prompt(committed: list[Message], messages: list[Message]) -> str:
    if (
        committed
        and len(messages) == len(committed) + 1
        and messages[:-1] == committed
        and messages[-1].get("role") == "user"
    ):
        return "incremental"
    return "full"


class MlxLmEngine:
    def __init__(self, cfg: LlmConfig) -> None:
        self._cfg = cfg
        self._model = None
        self._tokenizer = None
        self._cache = None
        self._committed: list[Message] = []

    def load(self) -> None:
        from mlx_lm import load

        self._model, self._tokenizer = load(self._cfg.model)
        self._reset_cache()

    def _reset_cache(self) -> None:
        from mlx_lm.models.cache import make_prompt_cache

        self._cache = make_prompt_cache(self._model)
        self._committed = []

    def _template(self, messages: list[Message], think: bool) -> list[int]:
        try:
            return self._tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, enable_thinking=think
            )
        except TypeError:  # template without enable_thinking support
            return self._tokenizer.apply_chat_template(messages, add_generation_prompt=True)

    def stream(self, messages: list[Message], *, think: bool) -> Iterator[str]:
        from mlx_lm import stream_generate

        if plan_prompt(self._committed, messages) == "incremental":
            prompt = self._template([messages[-1]], think)
        else:
            self._reset_cache()
            prompt = self._template(messages, think)
        parts: list[str] = []
        for response in stream_generate(
            self._model,
            self._tokenizer,
            prompt,
            max_tokens=self._cfg.max_tokens,
            prompt_cache=self._cache,
        ):
            parts.append(response.text)
            yield response.text
        self._committed = [*messages, {"role": "assistant", "content": "".join(parts)}]
```

Implementation notes for the engineer:
- `stream_generate` yields objects whose `.text` attribute is the *delta* for that step in current mlx-lm; if the installed version instead yields plain strings, adapt (`response if isinstance(response, str) else response.text`). Check with `uv run python -c "import mlx_lm, inspect; print(inspect.signature(mlx_lm.stream_generate))"`.
- If the stream is abandoned mid-way (barge-in), `_committed` never updates, so the next call compares against stale history, gets `"full"`, and resets the cache — that is the designed truncation-recovery path; do not "fix" it.
- The engine is not unit-tested beyond `plan_prompt` (needs a real model); the Task 18 smoke test covers it with Qwen3.5-0.8B.

- [ ] **Step 4: Run test to verify it passes** — Run: `uv run pytest tests/test_llm_cache_plan.py -q` — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/localvoice/llm/mlx_lm_engine.py tests/test_llm_cache_plan.py
git commit -m "feat: mlx-lm engine with incremental prompt cache and truncation recovery"
```

---

### Task 12: Kokoro TTS engine

**Files:**
- Create: `src/localvoice/tts/kokoro_mlx.py`

**Interfaces:**
- Consumes: `TTSEngine` protocol, `TtsConfig`.
- Produces: `KokoroMlxEngine(cfg: TtsConfig)` implementing `TTSEngine`; yields float32 mono 24 kHz chunks per generated segment.

- [ ] **Step 1: Implement** (thin hardware wrapper; smoke-tested in Task 18):

```python
from collections.abc import Iterator

import numpy as np

from localvoice.config import TtsConfig


class KokoroMlxEngine:
    def __init__(self, cfg: TtsConfig) -> None:
        self._cfg = cfg
        self._model = None

    def load(self) -> None:
        from mlx_audio.tts.utils import load_model

        self._model = load_model(self._cfg.model)
        for _ in self.synthesize("Ready."):
            pass

    def synthesize(self, text: str) -> Iterator[np.ndarray]:
        for segment in self._model.generate(
            text=text, voice=self._cfg.voice, speed=self._cfg.speed, lang_code="a"
        ):
            audio = np.asarray(segment.audio, dtype=np.float32).reshape(-1)
            if audio.size:
                yield audio
```

Implementation notes:
- Verify the mlx-audio API on the installed version before trusting this sketch: `uv run python -c "from mlx_audio.tts.utils import load_model; m = load_model('prince-canuma/Kokoro-82M'); import inspect; print(inspect.signature(m.generate))"`. If `generate` returns a single result instead of an iterator of segments, wrap it in a one-element loop. If the loader lives elsewhere (e.g. `mlx_audio.tts.generate`), adjust — the class surface (`load`/`synthesize`) must not change.
- `lang_code="a"` selects American English G2P in Kokoro; leave it hardcoded for v1.
- espeak-ng is already installed via brew (misaki fallback G2P).

- [ ] **Step 2: Import check** — Run: `uv run python -c "from localvoice.tts.kokoro_mlx import KokoroMlxEngine; print('ok')"` — Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add src/localvoice/tts/kokoro_mlx.py
git commit -m "feat: Kokoro TTS engine via mlx-audio"
```

---

### Task 13: Response pipeline

**Files:**
- Create: `src/localvoice/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: engine protocols and fakes (Task 9), `ClauseChunker` (3), `TextFilter`/`strip_speech_markup` (4), `Transcript` (5), player surface (7), `rms` (8), `Event`/`EventType` (2).
- Produces:

```python
@dataclass
class PipelineDeps:
    stt: STTEngine; llm: LLMEngine; tts: TTSEngine
    player: Any  # AudioPlayer surface: submit/submit_raw/mark_end/flush/spoken_tags
    transcript: Transcript
    think: bool = False
    min_rms: float = 0.005
    min_seconds: float = 0.25
    on_user_text: Callable[[str], None] = lambda s: None
    on_assistant_clause: Callable[[str], None] = lambda s: None

def run_pipeline(audio: np.ndarray, sample_rate: int, deps: PipelineDeps,
                 cancel: threading.Event, emit: Callable[[Event], None]) -> None
```

  Behavior: (1) too-short/quiet audio → `transcript.abort_pending()` is unnecessary (no pending yet), emit `RESPONSE_FINISHED`, return. (2) STT → empty text → same. (3) `transcript.begin_turn(text)`; stream LLM deltas through `TextFilter` then `ClauseChunker`; for each clause: `spoken = strip_speech_markup(clause)`; skip if empty; `tag = transcript.add_clause(spoken)`; synthesize; submit each TTS chunk with the tag; emit `FIRST_AUDIO` once after the first submitted chunk. (4) after the LLM ends, flush chunker+filter tail through the same path, then `player.mark_end()` (RESPONSE_FINISHED then comes from the player's drain callback — NOT from the pipeline). (5) `cancel` is checked before STT, before LLM, between deltas, and between TTS chunks; when set, return silently emitting nothing. (6) exceptions: if `cancel` is set, swallow; otherwise emit `PIPELINE_ERROR` with the message. (7) If the whole response produced zero clauses (empty LLM output), emit `RESPONSE_FINISHED` directly (the player would never drain-fire without `mark_end`+content — call `mark_end` anyway then emit; guard double-fire in orchestrator is unnecessary because SM ignores RESPONSE_FINISHED in IDLE).

- [ ] **Step 1: Write the failing tests** — `tests/test_pipeline.py`:

```python
import threading

import numpy as np

from localvoice.events import EventType as E
from localvoice.pipeline import PipelineDeps, run_pipeline
from localvoice.transcript import Transcript
from tests.fakes import FakeLLM, FakePlayer, FakeSTT, FakeTTS

SR = 16000


def speech(seconds: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (0.1 * rng.standard_normal(int(SR * seconds))).astype(np.float32)


def run(deps: PipelineDeps, audio: np.ndarray, cancel: threading.Event | None = None) -> list:
    events: list = []
    run_pipeline(audio, SR, deps, cancel or threading.Event(), events.append)
    return events


def make_deps(**kw) -> PipelineDeps:
    return PipelineDeps(
        stt=kw.get("stt", FakeSTT("what is two plus two")),
        llm=kw.get("llm", FakeLLM(["Four", ". ", "Easy ", "one."])),
        tts=kw.get("tts", FakeTTS()),
        player=kw.get("player", FakePlayer()),
        transcript=kw.get("transcript", Transcript("sys")),
        **{k: v for k, v in kw.items() if k in ("think", "min_rms", "min_seconds")},
    )


def test_happy_path_events_and_history():
    t = Transcript("sys")
    player = FakePlayer()
    deps = make_deps(transcript=t, player=player)
    events = run(deps, speech())
    types = [e.type for e in events]
    assert types.count(E.FIRST_AUDIO) == 1
    assert E.PIPELINE_ERROR not in types
    assert ("mark_end",) in player.log
    # commit is the orchestrator's job on RESPONSE_FINISHED; pending still open here
    t.commit()
    assert t.history()[0] == {"role": "user", "content": "what is two plus two"}
    assert "Four." in t.history()[1]["content"]


def test_quiet_audio_discards_without_stt():
    events = run(make_deps(), np.zeros(SR, np.float32))
    assert [e.type for e in events] == [E.RESPONSE_FINISHED]


def test_too_short_audio_discards():
    events = run(make_deps(), speech(0.1))
    assert [e.type for e in events] == [E.RESPONSE_FINISHED]


def test_empty_transcription_discards():
    events = run(make_deps(stt=FakeSTT("  ")), speech())
    assert [e.type for e in events] == [E.RESPONSE_FINISHED]


def test_cancel_before_start_produces_nothing():
    cancel = threading.Event()
    cancel.set()
    deps = make_deps()
    events = run(deps, speech(), cancel)
    assert events == [] and deps.player.log == []


def test_cancel_mid_llm_stops_and_stays_silent():
    class CancellingLLM(FakeLLM):
        def __init__(self, cancel):
            super().__init__(["First bit. ", "Second bit. ", "Third."])
            self._cancel = cancel

        def stream(self, messages, *, think):
            for i, d in enumerate(super().stream(messages, think=think)):
                if i == 1:
                    self._cancel.set()
                yield d

    cancel = threading.Event()
    deps = make_deps(llm=CancellingLLM(cancel))
    events = run(deps, speech(), cancel)
    assert all(e.type == E.FIRST_AUDIO for e in events)  # no finish/error after cancel
    assert ("mark_end",) not in deps.player.log


def test_error_emits_pipeline_error():
    class BoomTTS(FakeTTS):
        def synthesize(self, text):
            raise RuntimeError("kaboom")
            yield  # pragma: no cover

    events = run(make_deps(tts=BoomTTS()), speech())
    assert events[-1].type == E.PIPELINE_ERROR
    assert "kaboom" in events[-1].message


def test_think_flag_reaches_llm_and_markup_is_stripped():
    llm = FakeLLM(["<think>hmm</think>", "**Bold** answer. ", "Tail"])
    tts = FakeTTS()
    deps = make_deps(llm=llm, tts=tts, think=True)
    run(deps, speech())
    assert llm.last_think is True
    assert tts.texts[0] == "Bold answer."
    assert "Tail" in tts.texts[-1]


def test_empty_llm_output_finishes_cleanly():
    events = run(make_deps(llm=FakeLLM([])), speech())
    assert events[-1].type == E.RESPONSE_FINISHED
```

- [ ] **Step 2: Run tests to verify they fail** — Run: `uv run pytest tests/test_pipeline.py -q` — Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement** — `src/localvoice/pipeline.py`:

```python
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from localvoice.audio.capture import rms
from localvoice.events import Event, EventType
from localvoice.llm.base import LLMEngine
from localvoice.stt.base import STTEngine
from localvoice.textproc.chunker import ClauseChunker
from localvoice.textproc.sanitize import TextFilter, strip_speech_markup
from localvoice.transcript import Transcript
from localvoice.tts.base import TTSEngine


@dataclass
class PipelineDeps:
    stt: STTEngine
    llm: LLMEngine
    tts: TTSEngine
    player: Any
    transcript: Transcript
    think: bool = False
    min_rms: float = 0.005
    min_seconds: float = 0.25
    on_user_text: Callable[[str], None] = field(default=lambda s: None)
    on_assistant_clause: Callable[[str], None] = field(default=lambda s: None)


def run_pipeline(
    audio: np.ndarray,
    sample_rate: int,
    deps: PipelineDeps,
    cancel: threading.Event,
    emit: Callable[[Event], None],
) -> None:
    try:
        if cancel.is_set():
            return
        if len(audio) < deps.min_seconds * sample_rate or rms(audio) < deps.min_rms:
            emit(Event(EventType.RESPONSE_FINISHED))
            return
        user_text = deps.stt.transcribe(audio, sample_rate)
        if cancel.is_set():
            return
        if not user_text.strip():
            emit(Event(EventType.RESPONSE_FINISHED))
            return
        deps.transcript.begin_turn(user_text)
        deps.on_user_text(user_text)

        text_filter = TextFilter()
        chunker = ClauseChunker()
        first_audio_sent = False
        spoke_anything = False

        def speak(clause: str) -> bool:
            nonlocal first_audio_sent, spoke_anything
            spoken = strip_speech_markup(clause)
            if not spoken:
                return True
            tag = deps.transcript.add_clause(spoken)
            deps.on_assistant_clause(spoken)
            for chunk in deps.tts.synthesize(spoken):
                if cancel.is_set():
                    return False
                deps.player.submit(chunk, tag)
                spoke_anything = True
                if not first_audio_sent:
                    first_audio_sent = True
                    emit(Event(EventType.FIRST_AUDIO))
            return True

        for delta in deps.llm.stream(deps.transcript.messages(), think=deps.think):
            if cancel.is_set():
                return
            for clause in chunker.feed(text_filter.feed(delta)):
                if not speak(clause):
                    return
        if cancel.is_set():
            return
        tail = (chunker.flush() or "") + text_filter.finish()
        if tail.strip():
            if not speak(tail):
                return
        deps.player.mark_end()
        if not spoke_anything:
            emit(Event(EventType.RESPONSE_FINISHED))
    except Exception as exc:  # noqa: BLE001 — pipeline boundary
        if not cancel.is_set():
            emit(Event(EventType.PIPELINE_ERROR, message=str(exc)))
```

Note the tail handling: `chunker.flush()` may hold text whose trailing boundary was never confirmed, and `text_filter.finish()` may release held prefix chars — both go through `speak` as one final clause. Note also that on the happy path the pipeline emits no `RESPONSE_FINISHED`; the player's drain callback does (Task 7), so the turn commits only after the last sample actually plays.

- [ ] **Step 4: Run tests to verify they pass** — Run: `uv run pytest tests/test_pipeline.py -q` — Expected: PASS. Also run the full suite: `uv run pytest -m "not slow" -q`.

- [ ] **Step 5: Commit**

```bash
git add src/localvoice/pipeline.py tests/test_pipeline.py
git commit -m "feat: cancellable STT-LLM-TTS response pipeline"
```

---

### Task 14: Global hotkey listener

**Files:**
- Create: `src/localvoice/hotkey.py`
- Test: `tests/test_hotkey.py`

**Interfaces:**
- Consumes: `Event`, `EventType` (Task 2), `KeysConfig` (Task 6).
- Produces:
  - `parse_key(name: str)` — `"cmd_r"` → `pynput.keyboard.Key.cmd_r`; single chars → `KeyCode.from_char`; unknown names raise `ConfigError`.
  - `HotkeyListener(keys: KeysConfig, emit: Callable[[Event], None])` — `start()`/`stop()`; on PTT key press emits `Event(PTT_DOWN)` once (ignores repeats while held), on release emits `Event(PTT_UP, held_ms=<monotonic ms>)`; stop key press emits `Event(ESC)`. Internal handlers `_on_press(key)`/`_on_release(key)` are pure enough to test directly without a real listener.
  - `input_monitoring_ok() -> bool | None` — `True`/`False` via Quartz `CGPreflightListenEventAccess` when available (pynput already depends on pyobjc), `None` when the check is unavailable.

- [ ] **Step 1: Write the failing tests** — `tests/test_hotkey.py`:

```python
import time

import pytest
from pynput import keyboard

from localvoice.config import ConfigError, KeysConfig
from localvoice.events import EventType as E
from localvoice.hotkey import HotkeyListener, parse_key


def test_parse_named_and_char_keys():
    assert parse_key("cmd_r") == keyboard.Key.cmd_r
    assert parse_key("esc") == keyboard.Key.esc
    assert parse_key("z") == keyboard.KeyCode.from_char("z")
    with pytest.raises(ConfigError, match="no_such_key"):
        parse_key("no_such_key")


def make() -> tuple[HotkeyListener, list]:
    events: list = []
    return HotkeyListener(KeysConfig(ptt="cmd_r", stop="esc", debounce_ms=120), events.append), events


def test_press_release_emits_down_up_with_held_ms():
    listener, events = make()
    listener._on_press(keyboard.Key.cmd_r)
    time.sleep(0.05)
    listener._on_release(keyboard.Key.cmd_r)
    assert [e.type for e in events] == [E.PTT_DOWN, E.PTT_UP]
    assert 30 <= events[1].held_ms <= 500


def test_repeat_presses_while_held_are_ignored():
    listener, events = make()
    listener._on_press(keyboard.Key.cmd_r)
    listener._on_press(keyboard.Key.cmd_r)
    assert [e.type for e in events] == [E.PTT_DOWN]


def test_release_without_press_is_ignored():
    listener, events = make()
    listener._on_release(keyboard.Key.cmd_r)
    assert events == []


def test_stop_key_emits_esc_and_other_keys_ignored():
    listener, events = make()
    listener._on_press(keyboard.Key.esc)
    listener._on_press(keyboard.KeyCode.from_char("x"))
    assert [e.type for e in events] == [E.ESC]
```

- [ ] **Step 2: Run tests to verify they fail** — Run: `uv run pytest tests/test_hotkey.py -q` — Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement** — `src/localvoice/hotkey.py`:

```python
import time
from collections.abc import Callable

from pynput import keyboard

from localvoice.config import ConfigError, KeysConfig
from localvoice.events import Event, EventType


def parse_key(name: str):
    if hasattr(keyboard.Key, name):
        return getattr(keyboard.Key, name)
    if len(name) == 1:
        return keyboard.KeyCode.from_char(name)
    raise ConfigError(f"unknown key name: {name}")


def input_monitoring_ok() -> bool | None:
    try:
        from Quartz import CGPreflightListenEventAccess  # type: ignore[attr-defined]

        return bool(CGPreflightListenEventAccess())
    except Exception:
        return None


class HotkeyListener:
    def __init__(self, keys: KeysConfig, emit: Callable[[Event], None]) -> None:
        self._ptt = parse_key(keys.ptt)
        self._stop = parse_key(keys.stop)
        self._emit = emit
        self._pressed_at: float | None = None
        self._listener: keyboard.Listener | None = None

    def start(self) -> None:
        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def _on_press(self, key) -> None:
        if key == self._ptt:
            if self._pressed_at is None:
                self._pressed_at = time.monotonic()
                self._emit(Event(EventType.PTT_DOWN))
        elif key == self._stop:
            self._emit(Event(EventType.ESC))

    def _on_release(self, key) -> None:
        if key == self._ptt and self._pressed_at is not None:
            held_ms = int((time.monotonic() - self._pressed_at) * 1000)
            self._pressed_at = None
            self._emit(Event(EventType.PTT_UP, held_ms=held_ms))
```

- [ ] **Step 4: Run tests to verify they pass** — Run: `uv run pytest tests/test_hotkey.py -q` — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/localvoice/hotkey.py tests/test_hotkey.py
git commit -m "feat: global push-to-talk hotkey listener with permission preflight"
```

---

### Task 15: Orchestrator

**Files:**
- Create: `src/localvoice/app.py`
- Test: `tests/test_app.py`

**Interfaces:**
- Consumes: everything above — `transition` (2), `Event/EventType/State/Action` (2), `PipelineDeps`/`run_pipeline` (13), `Transcript` (5), `EARCONS` (7), player/capture surfaces (7/8).
- Produces:

```python
class Orchestrator:
    def __init__(self, *, capture, player, stt, llm, tts, transcript, keys_cfg, think: bool = False,
                 status: Callable[[str], None] = print) -> None
    def post(self, event: Event) -> None          # thread-safe: queue.put
    def run_forever(self) -> None                 # main loop; returns on self.shutdown()
    def shutdown(self) -> None
    # exposed for tests:
    state: State
    def handle(self, event: Event) -> None        # one dispatch step (used by run_forever)
```

  Dispatch rules: stale pipeline events (`FIRST_AUDIO`, `RESPONSE_FINISHED`, `PIPELINE_ERROR` with `event.gen != self._gen`) are dropped before touching the state machine. Actions map to: `START_CAPTURE→capture.arm()`, `DISCARD_CAPTURE→capture.discard()`, `STOP_CAPTURE_AND_RUN→` grab audio, bump `self._gen`, fresh `threading.Event` cancel token, spawn `threading.Thread(target=run_pipeline, ... , emit=<post with gen stamped>)`, `CANCEL_PIPELINE→cancel.set()`, `FLUSH_AUDIO→player.flush()`, `TRUNCATE_HISTORY→transcript.truncate_commit(player.spoken_tags())`, `COMMIT_TURN→transcript.commit()`, `REPORT_ERROR→status(...)`, `EARCON_*→player.submit_raw(EARCONS[...])`. Hotkey events carry `gen=-1` and are never dropped.

- [ ] **Step 1: Write the failing tests** — `tests/test_app.py`:

```python
import numpy as np

from localvoice.app import Orchestrator
from localvoice.config import KeysConfig
from localvoice.events import Event, EventType as E, State as S
from localvoice.transcript import Transcript
from tests.fakes import FakeLLM, FakePlayer, FakeSTT, FakeTTS


class FakeCapture:
    def __init__(self):
        self.log = []

    def arm(self):
        self.log.append("arm")

    def disarm(self):
        self.log.append("disarm")
        rng = np.random.default_rng(1)
        return (0.1 * rng.standard_normal(16000)).astype(np.float32)

    def discard(self):
        self.log.append("discard")


def make() -> tuple[Orchestrator, FakeCapture, FakePlayer, Transcript]:
    capture, player, transcript = FakeCapture(), FakePlayer(), Transcript("sys")
    orch = Orchestrator(
        capture=capture, player=player, stt=FakeSTT("hi"), llm=FakeLLM(["Hello there. ", "More."]),
        tts=FakeTTS(), transcript=transcript, keys_cfg=KeysConfig(), status=lambda s: None,
    )
    return orch, capture, player, transcript


def drain(orch: Orchestrator) -> None:
    # process queued events until empty AND pipeline thread (if any) has finished
    import time

    for _ in range(200):
        if orch._thread is not None and orch._thread.is_alive():
            time.sleep(0.01)
            continue
        try:
            ev = orch._queue.get_nowait()
        except Exception:
            break
        orch.handle(ev)


def test_full_turn_reaches_idle_and_commits():
    orch, capture, player, transcript = make()
    orch.handle(Event(E.PTT_DOWN))
    assert orch.state is S.LISTENING and capture.log == ["arm"]
    orch.handle(Event(E.PTT_UP, held_ms=400))
    assert orch.state is S.PROCESSING and "disarm" in capture.log
    drain(orch)  # pipeline thread posts FIRST_AUDIO; FakePlayer never drain-fires, so finish manually
    orch.handle(Event(E.RESPONSE_FINISHED, gen=orch._gen))
    assert orch.state is S.IDLE
    assert transcript.history()[-1]["role"] == "assistant"


def test_short_tap_discards():
    orch, capture, *_ = make()
    orch.handle(Event(E.PTT_DOWN))
    orch.handle(Event(E.PTT_UP, held_ms=50))
    assert orch.state is S.IDLE and capture.log[-1] == "discard"


def test_barge_in_truncates_and_relistens():
    orch, capture, player, transcript = make()
    orch.handle(Event(E.PTT_DOWN))
    orch.handle(Event(E.PTT_UP, held_ms=400))
    drain(orch)
    orch.handle(Event(E.FIRST_AUDIO, gen=orch._gen))
    assert orch.state is S.SPEAKING
    orch.handle(Event(E.PTT_DOWN))  # barge-in
    assert orch.state is S.LISTENING
    assert ("flush",) in player.log
    assert transcript.history() and transcript.history()[-1]["role"] == "assistant"
    assert capture.log.count("arm") == 2


def test_stale_generation_events_are_dropped():
    orch, *_ = make()
    orch.handle(Event(E.PTT_DOWN))
    orch.handle(Event(E.PTT_UP, held_ms=400))
    drain(orch)
    before = orch.state  # drain may already have consumed FIRST_AUDIO
    orch.handle(Event(E.RESPONSE_FINISHED, gen=orch._gen - 1))
    assert orch.state is before  # stale event ignored; a live one would move to IDLE


def test_esc_during_speaking_cancels_to_idle():
    orch, capture, player, transcript = make()
    orch.handle(Event(E.PTT_DOWN))
    orch.handle(Event(E.PTT_UP, held_ms=400))
    drain(orch)
    orch.handle(Event(E.FIRST_AUDIO, gen=orch._gen))
    orch.handle(Event(E.ESC))
    assert orch.state is S.IDLE and ("flush",) in player.log
```

- [ ] **Step 2: Run tests to verify they fail** — Run: `uv run pytest tests/test_app.py -q` — Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement** — `src/localvoice/app.py`:

```python
import queue
import threading
from collections.abc import Callable

from localvoice.audio.earcons import EARCONS
from localvoice.config import KeysConfig
from localvoice.events import Action as A
from localvoice.events import Event, EventType as E, State as S
from localvoice.pipeline import PipelineDeps, run_pipeline
from localvoice.states import transition

_PIPELINE_EVENTS = {E.FIRST_AUDIO, E.RESPONSE_FINISHED, E.PIPELINE_ERROR}


class Orchestrator:
    def __init__(
        self,
        *,
        capture,
        player,
        stt,
        llm,
        tts,
        transcript,
        keys_cfg: KeysConfig,
        think: bool = False,
        status: Callable[[str], None] = print,
    ) -> None:
        self.state = S.IDLE
        self._capture = capture
        self._player = player
        self._transcript = transcript
        self._keys = keys_cfg
        self._status = status
        self._deps = PipelineDeps(
            stt=stt, llm=llm, tts=tts, player=player, transcript=transcript, think=think,
            on_user_text=lambda t: status(f"you: {t}"),
            on_assistant_clause=lambda t: status(f"assistant: {t}"),
        )
        self._queue: queue.Queue[Event] = queue.Queue()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._gen = 0
        self._running = True

    def post(self, event: Event) -> None:
        self._queue.put(event)

    def shutdown(self) -> None:
        self._running = False
        self._cancel.set()
        self._queue.put(Event(E.ESC))  # wake the loop

    def run_forever(self) -> None:
        while self._running:
            self.handle(self._queue.get())

    def handle(self, event: Event) -> None:
        if event.type in _PIPELINE_EVENTS and event.gen != self._gen:
            return
        self.state, actions = transition(self.state, event, self._keys.debounce_ms)
        for action in actions:
            self._do(action, event)
        self._show_state()

    def _do(self, action: A, event: Event) -> None:
        if action is A.START_CAPTURE:
            self._capture.arm()
        elif action is A.DISCARD_CAPTURE:
            self._capture.discard()
        elif action is A.STOP_CAPTURE_AND_RUN:
            self._start_pipeline()
        elif action is A.CANCEL_PIPELINE:
            self._cancel.set()
        elif action is A.FLUSH_AUDIO:
            self._player.flush()
        elif action is A.TRUNCATE_HISTORY:
            self._transcript.truncate_commit(self._player.spoken_tags())
        elif action is A.COMMIT_TURN:
            self._transcript.commit()
        elif action is A.REPORT_ERROR:
            self._status(f"error: {event.message}")
        elif action is A.EARCON_START:
            self._player.submit_raw(EARCONS["start"])
        elif action is A.EARCON_STOP:
            self._player.submit_raw(EARCONS["stop"])
        elif action is A.EARCON_CANCEL:
            self._player.submit_raw(EARCONS["cancel"])

    def _start_pipeline(self) -> None:
        audio = self._capture.disarm()
        self._gen += 1
        gen = self._gen
        self._cancel = threading.Event()
        cancel = self._cancel

        def emit(event: Event) -> None:
            self.post(Event(event.type, event.held_ms, event.message, gen))

        self._thread = threading.Thread(
            target=run_pipeline, args=(audio, 16000, self._deps, cancel, emit), daemon=True
        )
        self._thread.start()

    def _show_state(self) -> None:
        labels = {
            S.IDLE: "idle - hold right-command to talk, esc to stop, ctrl-c to quit",
            S.LISTENING: "listening...",
            S.PROCESSING: "thinking...",
            S.SPEAKING: "speaking... (hold key to interrupt)",
        }
        self._status(f"[{labels[self.state]}]")
```

One test-vs-implementation wrinkle to expect: in `test_full_turn_reaches_idle_and_commits`, the pipeline thread posts `FIRST_AUDIO` into the queue; `drain` processes it, moving PROCESSING→SPEAKING, before the manual `RESPONSE_FINISHED`. Both orders end in IDLE with a committed turn, which is what's asserted. The player's real drain-callback path (which stamps no gen) is wired in Task 16 where the player is constructed with `on_response_finished=lambda: orch.post(Event(E.RESPONSE_FINISHED, gen=<current>))` — see that task.

- [ ] **Step 4: Run tests to verify they pass** — Run: `uv run pytest tests/test_app.py -q`, then full suite `uv run pytest -m "not slow" -q` — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/localvoice/app.py tests/test_app.py
git commit -m "feat: orchestrator wiring state machine, pipeline threads, generation guard"
```

---

### Task 16: CLI — run, setup, bench

**Files:**
- Modify: `src/localvoice/__main__.py` (replace the Task 1 placeholder entirely)
- Create: `src/localvoice/bench.py`
- Test: `tests/test_cli.py` (argument parsing only)

**Interfaces:**
- Consumes: everything.
- Produces: `main()` entry point with subcommands:
  - `localvoice` / `localvoice run [--config PATH] [--deep] [--think]` — startup sequence: load config → permission preflights → construct engines → `load()` each with timing printed → start capture/player/hotkey → orchestrator `run_forever()`; SIGINT/ctrl-c → clean shutdown (stop listener, streams).
  - `localvoice setup [--config PATH]` — for each of stt/llm/tts models that look like HF repo ids (contain `/`, not an existing local path): `huggingface_hub.snapshot_download(repo_id)` with a confirmation prompt listing what will be downloaded; local paths are checked for existence. Idempotent.
  - `localvoice bench [--config PATH] [--deep] [--runs N]` — see `bench.py` below.
  - `build_parser() -> argparse.ArgumentParser` exposed for tests.

- [ ] **Step 1: Write the failing test** — `tests/test_cli.py`:

```python
from localvoice.__main__ import build_parser


def test_default_command_is_run():
    args = build_parser().parse_args([])
    assert args.command == "run" and args.deep is False and args.think is False


def test_run_flags():
    args = build_parser().parse_args(["run", "--deep", "--think", "--config", "x.toml"])
    assert args.deep and args.think and args.config == "x.toml"


def test_setup_and_bench_parse():
    assert build_parser().parse_args(["setup"]).command == "setup"
    args = build_parser().parse_args(["bench", "--runs", "5"])
    assert args.command == "bench" and args.runs == 5
```

- [ ] **Step 2: Run test to verify it fails** — Run: `uv run pytest tests/test_cli.py -q` — Expected: FAIL (placeholder has no `build_parser`).

- [ ] **Step 3: Implement** — `src/localvoice/__main__.py`:

```python
import argparse
import sys
import time
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="localvoice", description="Local push-to-talk voice assistant")
    sub = parser.add_subparsers(dest="command")
    parser.set_defaults(command="run", config="localvoice.toml", deep=False, think=False)

    run = sub.add_parser("run", help="start the assistant (default)")
    run.add_argument("--config", default="localvoice.toml")
    run.add_argument("--deep", action="store_true", help="use [llm].deep_model")
    run.add_argument("--think", action="store_true", help="enable silent reasoning mode")

    setup = sub.add_parser("setup", help="download configured models")
    setup.add_argument("--config", default="localvoice.toml")

    bench = sub.add_parser("bench", help="measure per-stage latency")
    bench.add_argument("--config", default="localvoice.toml")
    bench.add_argument("--deep", action="store_true")
    bench.add_argument("--runs", type=int, default=3)
    return parser


def _load_config(args):
    from localvoice.config import ConfigError, load_config

    try:
        return load_config(Path(args.config), deep=getattr(args, "deep", False))
    except ConfigError as exc:
        raise SystemExit(f"config error: {exc}") from exc


def _make_engines(cfg):
    from localvoice.llm.mlx_lm_engine import MlxLmEngine
    from localvoice.stt.whisper_mlx import WhisperMlxEngine
    from localvoice.tts.kokoro_mlx import KokoroMlxEngine

    return WhisperMlxEngine(cfg.stt), MlxLmEngine(cfg.llm), KokoroMlxEngine(cfg.tts)


def _load_timed(name: str, engine) -> None:
    t0 = time.perf_counter()
    print(f"loading {name}...", end=" ", flush=True)
    engine.load()
    print(f"{time.perf_counter() - t0:.1f}s")


def cmd_run(args) -> None:
    cfg = _load_config(args)
    _preflight()
    stt, llm, tts = _make_engines(cfg)
    from localvoice.app import Orchestrator
    from localvoice.audio.capture import MicCapture
    from localvoice.audio.player import AudioPlayer
    from localvoice.events import Event, EventType
    from localvoice.hotkey import HotkeyListener
    from localvoice.transcript import Transcript

    orch_ref = {}

    def on_finished():
        orch = orch_ref.get("orch")
        if orch is not None:
            orch.post(Event(EventType.RESPONSE_FINISHED, gen=orch._gen))

    player = AudioPlayer(cfg.audio, on_response_finished=on_finished)
    capture = MicCapture(cfg.audio)
    transcript = Transcript(cfg.llm.system_prompt)
    orch = Orchestrator(
        capture=capture, player=player, stt=stt, llm=llm, tts=tts, transcript=transcript,
        keys_cfg=cfg.keys, think=args.think or cfg.llm.think,
    )
    orch_ref["orch"] = orch
    for name, engine in (("whisper", stt), ("llm", llm), ("kokoro", tts)):
        _load_timed(name, engine)
    capture.start()
    player.start()
    listener = HotkeyListener(cfg.keys, orch.post)
    listener.start()
    print("ready - hold right-command and talk; esc stops; ctrl-c quits")
    try:
        orch.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()
        capture.stop()
        player.stop()


def _preflight() -> None:
    from localvoice.hotkey import input_monitoring_ok

    ok = input_monitoring_ok()
    if ok is False:
        raise SystemExit(
            "Input Monitoring permission missing.\n"
            "Open System Settings -> Privacy & Security -> Input Monitoring and enable your terminal,\n"
            "then run localvoice again."
        )
    try:
        import sounddevice as sd

        with sd.InputStream(samplerate=16000, channels=1):
            pass
    except Exception as exc:
        raise SystemExit(
            f"microphone unavailable ({exc}).\n"
            "Open System Settings -> Privacy & Security -> Microphone and enable your terminal."
        ) from exc


def cmd_setup(args) -> None:
    cfg = _load_config(args)
    from huggingface_hub import snapshot_download

    targets = []
    for label, model in (("stt", cfg.stt.model), ("llm", cfg.llm.model), ("tts", cfg.tts.model)):
        if Path(model).expanduser().exists():
            print(f"{label}: local path present: {model}")
        elif "/" in model:
            targets.append((label, model))
        else:
            raise SystemExit(f"{label}: model is neither an existing path nor an HF repo id: {model}")
    if not targets:
        print("everything already available")
        return
    print("will download from Hugging Face:")
    for label, repo in targets:
        print(f"  {label}: {repo}")
    if input("proceed? [y/N] ").strip().lower() != "y":
        raise SystemExit("aborted")
    for label, repo in targets:
        print(f"downloading {label}: {repo}")
        snapshot_download(repo_id=repo)
    print("done")


def cmd_bench(args) -> None:
    cfg = _load_config(args)
    from localvoice.bench import run_bench

    run_bench(cfg, runs=args.runs)


def main() -> None:
    args = build_parser().parse_args()
    {"run": cmd_run, "setup": cmd_setup, "bench": cmd_bench}[args.command](args)


if __name__ == "__main__":
    main()
```

`src/localvoice/bench.py`:

```python
import time

import numpy as np

from localvoice.config import Config


def _speech_fixture(seconds: float = 4.0) -> np.ndarray:
    """Synthesize a deterministic spoken question via macOS `say` at 16 kHz mono."""
    import subprocess
    import tempfile
    import wave
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "q.wav"
        subprocess.run(
            ["say", "-o", str(path), "--data-format=LEI16@16000",
             "What is the capital of Australia, and why is it not Sydney?"],
            check=True,
        )
        with wave.open(str(path)) as w:
            raw = w.readframes(w.getnframes())
    audio = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    return audio


def run_bench(cfg: Config, runs: int = 3) -> None:
    from localvoice.llm.mlx_lm_engine import MlxLmEngine
    from localvoice.stt.whisper_mlx import WhisperMlxEngine
    from localvoice.textproc.chunker import ClauseChunker
    from localvoice.textproc.sanitize import TextFilter
    from localvoice.tts.kokoro_mlx import KokoroMlxEngine

    stt, llm, tts = WhisperMlxEngine(cfg.stt), MlxLmEngine(cfg.llm), KokoroMlxEngine(cfg.tts)
    for name, engine in (("whisper", stt), ("llm", llm), ("kokoro", tts)):
        t0 = time.perf_counter()
        engine.load()
        print(f"load {name}: {time.perf_counter() - t0:.1f}s")
    audio = _speech_fixture()
    rows = []
    for i in range(runs):
        t0 = time.perf_counter()
        text = stt.transcribe(audio, 16000)
        t_stt = time.perf_counter()
        messages = [
            {"role": "system", "content": cfg.llm.system_prompt},
            {"role": "user", "content": text},
        ]
        chunker, tfilter = ClauseChunker(), TextFilter()
        t_first_token = t_first_clause = None
        first_clause = None
        for delta in llm.stream(messages, think=False):
            if t_first_token is None:
                t_first_token = time.perf_counter()
            clauses = chunker.feed(tfilter.feed(delta))
            if clauses and t_first_clause is None:
                t_first_clause = time.perf_counter()
                first_clause = clauses[0]
                break
        if first_clause is None:
            first_clause = chunker.flush() or "Hello."
            t_first_clause = time.perf_counter()
        next(iter(tts.synthesize(first_clause)))
        t_tts = time.perf_counter()
        rows.append((t_stt - t0, t_first_token - t_stt, t_first_clause - t_first_token, t_tts - t_first_clause, t_tts - t0))
        print(f"run {i + 1}: stt {rows[-1][0]:.2f}s | ttft {rows[-1][1]:.2f}s | "
              f"clause {rows[-1][2]:.2f}s | tts {rows[-1][3]:.2f}s | total {rows[-1][4]:.2f}s")
    best = min(rows, key=lambda r: r[-1])
    print(f"best voice-to-voice (excl. playback buffer): {best[-1]:.2f}s")
```

- [ ] **Step 4: Run tests to verify they pass** — Run: `uv run pytest -m "not slow" -q` and `uv run ruff check .` — Expected: PASS/clean.

- [ ] **Step 5: Commit**

```bash
git add src/localvoice/__main__.py src/localvoice/bench.py tests/test_cli.py
git commit -m "feat: CLI with run, setup, and bench subcommands"
```

---

### Task 17: Docs, README, and this machine's bootstrap overlay

**Files:**
- Modify: `README.md` (replace stub)
- Create: `docs/architecture.md`, `docs/models.md`, `docs/latency.md`, and `localvoice.local.toml` (NOT committed — verify it is gitignored)

**Interfaces:**
- Consumes: final CLI/config surface from Tasks 6 and 16.
- Produces: user-facing docs; a local overlay pointing this machine at the on-disk 27B until the 35B download finishes.

- [ ] **Step 1: Write `localvoice.local.toml`** (gitignored; personal bootstrap for this machine)

```toml
[llm]
model = "/Users/lorenzo/.lmstudio/models/lmstudio-community/Qwen3.6-27B-MLX-6bit"
deep_model = "/Users/lorenzo/.lmstudio/models/lmstudio-community/Qwen3.6-27B-MLX-6bit"
```

When the 35B-A3B finishes downloading, delete the `model` line (repo default takes over) and keep `deep_model` pointing at the 27B. Verify gitignore: `git status --short` must not list it.

- [ ] **Step 2: Write `README.md`** — sections, in order: title + one-line pitch; how it works (hold right ⌘ → talk → release → answer; esc stops; barge-in by pressing again); requirements (Apple Silicon Mac, 16 GB+ RAM for small models / 32 GB+ recommended for the default, uv); install (`git clone`, `uv sync`, `uv run localvoice setup`, `uv run localvoice`); macOS permissions (System Settings → Privacy & Security → Microphone AND Input Monitoring, enable your terminal app, restart the terminal after granting); configuration (table of every `localvoice.toml` key with defaults, the `localvoice.local.toml` overlay pattern, `--deep`, `--think`); privacy note (mic stream open for the session but audio kept only while the key is held; everything stays on-device); latency (link `docs/latency.md`); architecture (link `docs/architecture.md` + the spec); roadmap (phase 2/3 from the spec); acknowledgments (MLX, mlx-lm, mlx-whisper, mlx-audio, OpenAI Whisper, Qwen, Kokoro); license. Write real prose — no lorem, no TODOs.

- [ ] **Step 3: Write `docs/architecture.md`** — condensed from the spec: the pipeline diagram (ASCII from spec §3), the state/transition table (from `states.py`, which is the source of truth — copy the actual table), threading model (main event loop, hotkey thread, audio callback threads, per-response pipeline thread, generation counter for stale events), and the truncation story (clause tags → spoken_tags → truncate_commit). Link back to `docs/superpowers/specs/2026-07-03-localvoice-architecture-design.md`.

- [ ] **Step 4: Write `docs/models.md`** — how to swap each engine's model via config; the three LLM presets with expected trade-offs (35B-A3B default ≈ fast, 27B `deep_model` ≈ smartest, Gemma-4-E4B ≈ lightweight); pointer that any MLX-format folder in `~/.lmstudio/models` works as a local path; note on `-DWQ`/`-MTP` variants as future bench candidates. Write `docs/latency.md` as the target-vs-measured table: targets from the spec, a "measured" column marked "run `uv run localvoice bench`" to be filled by Task 18.

- [ ] **Step 5: Verify and commit**

Run: `uv run ruff check . && uv run pytest -m "not slow" -q && git status --short` — confirm `localvoice.local.toml` absent from git status.

```bash
git add README.md docs/
git commit -m "docs: README, architecture, models, and latency docs"
```

---

### Task 18: Slow smoke test, real-model bench, manual acceptance

**Files:**
- Create: `tests/test_smoke_slow.py`
- Modify: `docs/latency.md` (fill measured column), `localvoice.toml`/`localvoice.local.toml` only if bench forces a change

**Interfaces:**
- Consumes: the whole system.
- Produces: evidence v1 works end-to-end on real models; measured latency numbers in docs.

- [ ] **Step 1: Write the slow smoke test** — `tests/test_smoke_slow.py` (runs the real pipeline with the small on-disk models; excluded from CI):

```python
import threading
from pathlib import Path

import pytest

from localvoice.config import LlmConfig, SttConfig, TtsConfig
from localvoice.events import EventType as E
from localvoice.pipeline import PipelineDeps, run_pipeline
from localvoice.transcript import Transcript
from tests.fakes import FakePlayer

TINY_LLM = Path.home() / ".lmstudio/models/mlx-community/Qwen3.5-0.8B-MLX-4bit"


@pytest.mark.slow
@pytest.mark.skipif(not TINY_LLM.exists(), reason="tiny local LLM not present")
def test_end_to_end_with_real_models():
    from localvoice.bench import _speech_fixture
    from localvoice.llm.mlx_lm_engine import MlxLmEngine
    from localvoice.stt.whisper_mlx import WhisperMlxEngine
    from localvoice.tts.kokoro_mlx import KokoroMlxEngine

    stt = WhisperMlxEngine(SttConfig(model="mlx-community/whisper-tiny"))
    llm = MlxLmEngine(LlmConfig(model=str(TINY_LLM), max_tokens=60))
    tts = KokoroMlxEngine(TtsConfig())
    for e in (stt, llm, tts):
        e.load()
    events: list = []
    player = FakePlayer()
    deps = PipelineDeps(stt=stt, llm=llm, tts=tts, player=player, transcript=Transcript("Answer in one short sentence."))
    run_pipeline(_speech_fixture(), 16000, deps, threading.Event(), events.append)
    types = [e.type for e in events]
    assert E.FIRST_AUDIO in types and E.PIPELINE_ERROR not in types
    assert any(entry[0] == "submit" for entry in player.log)
    assert ("mark_end",) in player.log
```

If `say` rejects `.wav` output on this macOS version, render to `q.aiff` instead and convert with `afconvert q.aiff -f WAVE -d LEI16@16000 q.wav` before reading (adjust `_speech_fixture` accordingly).

- [ ] **Step 2: Run it** — Run: `uv run pytest tests/test_smoke_slow.py -m slow -v`
Expected: PASS (first run downloads whisper-tiny and Kokoro, several minutes). If the mlx-audio or mlx-lm API sketches from Tasks 11/12 don't match the installed versions, this is where it surfaces — fix the engine internals (surface must not change), rerun.

- [ ] **Step 3: Real bench** — Run: `uv run localvoice setup` (downloads whisper-large-v3-turbo; LLM/TTS as configured — the local overlay means no LLM download), then `uv run localvoice bench --runs 3`. Record the printed stage numbers into `docs/latency.md` (27B column now; 35B column when its download lands).

- [ ] **Step 4: Manual acceptance checklist** (the human runs these — agent prepares and asks):

1. `uv run localvoice` — first run may hit macOS permission prompts; grant Microphone + Input Monitoring, restart terminal, rerun.
2. Hold right ⌘, ask "what's the tallest mountain in the world?", release → spoken answer starts in ~1–2 s.
3. While it's speaking, hold right ⌘ again and ask "and the second tallest?" → audio stops instantly, follow-up answer is context-aware, transcript printed shows the first answer truncated with "...".
4. Press Esc mid-answer → silence, state returns to idle.
5. Tap right ⌘ for <120 ms → nothing happens.
6. Hold the key in silence, release → no response, no error, back to idle.
7. Ctrl-C → clean exit, no stack trace.

- [ ] **Step 5: Commit**

```bash
git add tests/test_smoke_slow.py docs/latency.md
git commit -m "test: real-model smoke test; docs: measured latency numbers"
```

---

## Self-review notes (completed at plan time)

- **Spec coverage**: every spec §3–§7 element maps to a task (state machine→2, chunker/sanitizer→3/4, transcript→5, config→6, player/earcons→7, capture→8, engines→9–12, pipeline→13, hotkey→14, orchestrator→15, CLI/setup/bench→16, docs→17, smoke/acceptance→18). Spec §5 "over-memory warning" is deliberately dropped from v1 scope (models are user-chosen config; setup prints sizes implicitly via HF download) — noted here as a conscious cut, revisit in phase 2.
- **Type consistency**: `spoken_tags() -> set[int]` (player) feeds `truncate_commit(spoken_tags: set[int])` (transcript); `Event.gen` stamped only by the orchestrator's `emit` wrapper; engine `stream(messages, *, think)` signature identical in protocol, fake, and mlx implementation.
- **Known verify-at-install points** (not placeholders — the class surfaces are fixed, internals may need version adaptation): mlx-lm `stream_generate` delta attribute (Task 11 note), mlx-audio Kokoro loader/generate shape (Task 12 note), `enable_thinking` template kwarg (Task 11 try/except).
