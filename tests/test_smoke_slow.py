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
