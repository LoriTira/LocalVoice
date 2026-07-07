import json
import threading
import time
from pathlib import Path

import pytest

from localvoice.config import LlmConfig, SttConfig, TtsConfig
from localvoice.events import EventType as E
from localvoice.pipeline import PipelineDeps, run_pipeline
from localvoice.transcript import Transcript
from tests.fakes import FakePlayer

TINY_LLM = Path.home() / ".lmstudio/models/mlx-community/Qwen3.5-0.8B-MLX-4bit"
GEMMA_LLM = Path.home() / ".lmstudio/models/lmstudio-community/gemma-4-26B-A4B-it-MLX-4bit"


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
    deps = PipelineDeps(
        stt=stt,
        llm=llm,
        tts=tts,
        player=player,
        transcript=Transcript("Answer in one short sentence."),
    )
    run_pipeline(_speech_fixture(), 16000, deps, threading.Event(), events.append)
    types = [e.type for e in events]
    assert E.FIRST_AUDIO in types and E.PIPELINE_ERROR not in types
    assert any(entry[0] == "submit" for entry in player.log)
    assert ("mark_end",) in player.log


@pytest.mark.slow
@pytest.mark.skipif(not TINY_LLM.exists(), reason="tiny local LLM not present")
def test_two_turn_conversation_reuses_engine():
    from localvoice.llm.mlx_lm_engine import MlxLmEngine

    llm = MlxLmEngine(LlmConfig(model=str(TINY_LLM), max_tokens=40))
    llm.load()
    system = {"role": "system", "content": "Answer in one short sentence."}
    user1 = {"role": "user", "content": "What color is the sky on a clear day?"}
    reply1 = "".join(llm.stream([system, user1], think=False))
    assert reply1.strip()
    assistant1 = {"role": "assistant", "content": "Blue."}
    user2 = {"role": "user", "content": "And at night?"}
    reply2 = "".join(llm.stream([system, user1, assistant1, user2], think=False))
    assert reply2.strip()


@pytest.mark.slow
@pytest.mark.skipif(not TINY_LLM.exists(), reason="tiny local LLM not present")
def test_pipeline_on_dedicated_inference_thread():
    """Production threading shape: engines load AND run on one worker thread.

    Regression guard for the MLX stream-affinity crash ("There is no Stream(gpu, N)
    in current thread"): loading engines on the main thread and generating on another
    thread breaks mlx-lm. cmd_run colocates everything on one executor thread; this
    test mirrors that wiring with real models.

    Runs in a fresh subprocess: earlier tests in this module import mlx_lm on
    pytest's main thread, which would poison the executor-threaded wiring in a way
    production (imports only ever on the inference thread) cannot experience.
    """
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        f"""
        import threading
        from concurrent.futures import ThreadPoolExecutor

        from localvoice.bench import _speech_fixture
        from localvoice.config import LlmConfig, SttConfig, TtsConfig
        from localvoice.events import EventType as E
        from localvoice.llm.mlx_lm_engine import MlxLmEngine
        from localvoice.pipeline import PipelineDeps, run_pipeline
        from localvoice.stt.whisper_mlx import WhisperMlxEngine
        from localvoice.transcript import Transcript
        from localvoice.tts.kokoro_mlx import KokoroMlxEngine
        from tests.fakes import FakePlayer

        inference = ThreadPoolExecutor(max_workers=1)
        stt = WhisperMlxEngine(SttConfig(model="mlx-community/whisper-tiny"))
        llm = MlxLmEngine(LlmConfig(model={str(TINY_LLM)!r}, max_tokens=60))
        tts = KokoroMlxEngine(TtsConfig())
        for e in (stt, llm, tts):
            inference.submit(e.load).result()
        events = []
        player = FakePlayer()
        deps = PipelineDeps(
            stt=stt, llm=llm, tts=tts, player=player,
            transcript=Transcript("Answer in one short sentence."),
        )
        inference.submit(
            run_pipeline, _speech_fixture(), 16000, deps, threading.Event(), events.append
        ).result()
        types = [ev.type for ev in events]
        assert E.PIPELINE_ERROR not in types, f"pipeline error: {{events}}"
        assert E.FIRST_AUDIO in types
        assert ("mark_end",) in player.log
        print("inference-thread pipeline OK", flush=True)
        import os

        os._exit(0)  # MLX Metal teardown of worker-thread state can SIGBUS at exit
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    assert proc.returncode == 0, f"subprocess failed:\n{proc.stdout}\n{proc.stderr}"
    assert "inference-thread pipeline OK" in proc.stdout


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


@pytest.mark.slow
@pytest.mark.skipif(not GEMMA_LLM.exists(), reason="Gemma tool-capable model not present")
def test_gemma_template_renders_tool_round_continuation():
    """The regression test that would have caught the tool-round crash: load
    the REAL Gemma chat template and render a full tool-round message sequence
    (system / user / assistant-with-tool_calls / tool) with tools= schemas.

    The bug: pipeline.py fed the role:"tool" message a dict `content`; Gemma's
    template tests `content is sequence` (True for a dict), iterates it as a
    list, hits a string key, calls .get() on that str, and raises
    UndefinedError. The fix JSON-encodes the content to a string. The fakes in
    test_pipeline.py never render a real template, so only this test exercises
    the actual template constraint end-to-end.
    """
    from transformers import AutoTokenizer

    from localvoice.tools.base import hf_tool_schema

    class _WebSearch:
        name = "web_search"
        description = "Search the web for current information."
        parameters = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }

    tok = AutoTokenizer.from_pretrained(str(GEMMA_LLM))
    tools = [hf_tool_schema(_WebSearch())]
    result_content = {"results": [{"title": "T", "url": "u", "snippet": "s"}]}

    def sequence(content):
        # Mirrors pipeline.py's exact role:"tool" message shape.
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the current weather in Boston?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": {"query": "weather Boston"},
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_0", "name": "web_search", "content": content},
        ]

    # The fix: content is a JSON string (what pipeline.py now builds) -> renders.
    out = tok.apply_chat_template(
        sequence(json.dumps(result_content)),
        tools=tools,
        add_generation_prompt=True,
        tokenize=False,
    )
    assert isinstance(out, str) and out

    # Guard the regression: the pre-fix raw-dict content still crashes this
    # template, so this test genuinely exercises the bug it protects against.
    with pytest.raises(Exception):
        tok.apply_chat_template(
            sequence(result_content), tools=tools, add_generation_prompt=True, tokenize=False
        )


@pytest.mark.slow
@pytest.mark.skipif(not GEMMA_LLM.exists(), reason="Gemma vision-capable model not present")
def test_mlx_vlm_hybrid_text_parity_and_prefix_reuse():
    """The hybrid engine's whole reason to exist: load Gemma 4 through mlx-vlm
    (vision-capable) yet drive TEXT turns through mlx_lm.stream_generate against
    the language tower via _TextTowerAdapter — with prefix reuse across turns and
    channel->think translation intact. Proves, on the real model:

      1. hybrid generation produces real text (the adapter bridges the tower's
         LanguageModelOutput to the raw logits the mlx-lm loop consumes),
      2. the tower's prompt cache actually advances and is reused next turn
         (cache offset > 0 before turn 2, with a real common prefix to reuse) —
         the latency win the hybrid is chosen for,
      3. reasoning streamed as Gemma's <|channel>thought is surfaced as the
         canonical <think> the downstream TextFilter speaks.
    """
    from localvoice.llm.mlx_lm_engine import _apply_template, common_prefix_len
    from localvoice.llm.mlx_vlm_engine import MlxVlmEngine

    engine = MlxVlmEngine(LlmConfig(model=str(GEMMA_LLM), max_tokens=24))
    engine.load()

    system = {"role": "system", "content": "Answer in one short sentence."}
    user1 = {"role": "user", "content": "What color is the sky on a clear day?"}

    # Turn 1: fresh prefill through the hybrid path must yield real text.
    reply1 = "".join(engine.stream([system, user1], think=False))
    assert reply1.strip(), "hybrid turn 1 produced no text"

    # The tower's cache now holds turn-1 prompt + generated tokens. Prefix reuse
    # can only engage if that offset actually advanced — the load-bearing proof
    # that the adapter drove real generation over a shared cache.
    assert engine._cache[0].offset > 0

    assistant1 = {"role": "assistant", "content": reply1.strip()}
    user2 = {"role": "user", "content": "And at night?"}
    msgs2 = [system, user1, assistant1, user2]

    # Turn 2 shares a real prefix (system + first exchange) with what the cache
    # already holds, so _stream_text trims (offset - common) and re-prefills only
    # the diverged tail instead of the whole conversation.
    turn2_tokens = _apply_template(engine._tokenizer, msgs2, False, None)
    reused = common_prefix_len(turn2_tokens, engine._prompt_tokens)
    assert reused > 0, "no reusable prefix — the hybrid would re-prefill every turn"
    assert engine._cache[0].offset > 0  # prefix-reuse path is armed before the call

    reply2 = "".join(engine.stream(msgs2, think=False))
    assert reply2.strip(), "hybrid turn 2 produced no text"

    # Think turn: Gemma streams reasoning as <|channel>thought…; the engine's
    # always-on ChannelThinkTranslator must surface it as a canonical <think>
    # in the raw stream. Break as soon as the tag appears (it opens near the
    # start) so we never generate the full think budget.
    think_sys = {"role": "system", "content": "Reason step by step before answering."}
    think_user = {"role": "user", "content": "Is 91 a prime number?"}
    joined = ""
    for i, delta in enumerate(engine.stream([think_sys, think_user], think=True)):
        joined += delta
        if "<think>" in joined or i > 80:
            break
    assert "<think>" in joined, f"no translated reasoning in stream: {joined!r}"


def _write_solid_png(path: Path, width: int, height: int, rgb: tuple[int, int, int]) -> None:
    """Pure-python PNG writer for a single solid-color RGB image.

    No PIL/numpy needed for a fixture this trivial: one IHDR (8-bit truecolor,
    no palette/interlace), one IDAT (zlib-compressed scanlines, each prefixed
    with filter-type 0 = None — correct and simplest for a flat color), one
    IEND. Every multi-byte PNG field is big-endian per spec; struct.pack(">..")
    handles that, zlib.compress gives the zlib-wrapped (not raw) deflate
    stream IDAT requires, and zlib.crc32 is the exact CRC-32 variant PNG
    chunks use.
    """
    import struct
    import zlib

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    row = bytes([0]) + bytes(rgb) * width  # filter-type byte + width RGB pixels
    raw = row * height
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # depth 8, color type 2 (RGB)
    png = (
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(
            b"IEND", b""
        )
    )
    path.write_bytes(png)


@pytest.mark.slow
@pytest.mark.skipif(not GEMMA_LLM.exists(), reason="Gemma vision-capable model not present")
def test_mlx_vlm_image_turn_describes_color_and_leaves_text_cache_clean(tmp_path):
    """Task 4's minimal image entry point (T3's hook), proven on the real model.

    1. stream(..., image_path=...) drives mlx-vlm's own generation
       (_stream_image, NOT the text-tower adapter) against a real solid-red
       fixture image, end-to-end through the real Gemma 4 vision tower.
    2. The cache-interaction decision documented in _stream_image's docstring
       is verified live, not just by source-reading: mlx-vlm never receives
       self._cache for an image turn (it builds and discards its own scratch
       cache), so the persisted TEXT-turn cache is untouched by the image
       call other than the engine's own defensive reset — and a plain TEXT
       turn run immediately afterward still produces real text through the
       ordinary hybrid (_TextTowerAdapter) path, proving the image turn left
       no wreckage for it to trip over.
    """
    from localvoice.llm.mlx_vlm_engine import MlxVlmEngine

    png = tmp_path / "red.png"
    _write_solid_png(png, 64, 64, (255, 0, 0))

    engine = MlxVlmEngine(LlmConfig(model=str(GEMMA_LLM), max_tokens=32))
    engine.load()
    assert engine._cache[0].offset == 0  # nothing prefilled yet

    reply = "".join(
        engine.stream(
            [{"role": "user", "content": "What color is this image? Answer with one word."}],
            think=False,
            image_path=str(png),
        )
    )
    assert "red" in reply.lower(), f"expected 'red' in the model's answer, got: {reply!r}"

    # The persisted TEXT-turn cache is left at a clean zero offset — either
    # because the image turn never touched it in the first place (it isn't
    # passed to mlx-vlm at all) or via the engine's own defensive reset.
    assert engine._cache[0].offset == 0

    # A plain TEXT turn right after must still work through the ordinary
    # hybrid path — proof the image turn didn't wedge the engine.
    text_reply = "".join(
        engine.stream([{"role": "user", "content": "Say hello in one word."}], think=False)
    )
    assert text_reply.strip(), "text turn after an image turn produced no text"
    assert engine._cache[0].offset > 0  # the text turn prefilled normally
