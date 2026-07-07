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
            [
                "say",
                "-o",
                str(path),
                "--data-format=LEI16@16000",
                "What is the capital of Australia, and why is it not Sydney?",
            ],
            check=True,
        )
        with wave.open(str(path)) as w:
            raw = w.readframes(w.getnframes())
    audio = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
    return audio


def run_bench(cfg: Config, runs: int = 3) -> None:
    from localvoice.llm import build_llm_engine
    from localvoice.stt.whisper_mlx import WhisperMlxEngine
    from localvoice.textproc.chunker import ClauseChunker
    from localvoice.textproc.sanitize import TextFilter
    from localvoice.tts.kokoro_mlx import KokoroMlxEngine

    stt, llm, tts = WhisperMlxEngine(cfg.stt), build_llm_engine(cfg.llm), KokoroMlxEngine(cfg.tts)
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
        if t_first_token is None:  # degenerate zero-delta stream
            t_first_token = t_first_clause
        next(iter(tts.synthesize(first_clause)))
        t_tts = time.perf_counter()
        rows.append(
            (
                t_stt - t0,
                t_first_token - t_stt,
                t_first_clause - t_first_token,
                t_tts - t_first_clause,
                t_tts - t0,
            )
        )
        print(
            f"run {i + 1}: stt {rows[-1][0]:.2f}s | ttft {rows[-1][1]:.2f}s | "
            f"clause {rows[-1][2]:.2f}s | tts {rows[-1][3]:.2f}s | total {rows[-1][4]:.2f}s"
        )
    if not rows:
        print("no runs executed (check --runs)")
        return
    best = min(rows, key=lambda r: r[-1])
    print(f"best voice-to-voice (excl. playback buffer): {best[-1]:.2f}s")
