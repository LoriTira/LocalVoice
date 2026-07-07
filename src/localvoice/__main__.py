import argparse
import sys
import time
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="localvoice", description="Local push-to-talk voice assistant"
    )
    sub = parser.add_subparsers(dest="command")
    parser.set_defaults(
        command="run", config="localvoice.toml", deep=False, think=False, model=None
    )

    run = sub.add_parser("run", help="start the assistant (default)")
    run.add_argument("--config", default="localvoice.toml")
    run.add_argument("--deep", action="store_true", help="use [llm].deep_model")
    run.add_argument("--think", action="store_true", help="enable silent reasoning mode")
    run.add_argument("--model", default=None, help="override [llm].model (HF repo id or path)")

    setup = sub.add_parser("setup", help="download configured models")
    setup.add_argument("--config", default="localvoice.toml")

    bench = sub.add_parser("bench", help="measure per-stage latency")
    bench.add_argument("--config", default="localvoice.toml")
    bench.add_argument("--deep", action="store_true")
    bench.add_argument("--model", default=None, help="override [llm].model (HF repo id or path)")
    bench.add_argument("--runs", type=int, default=3)

    serve = sub.add_parser("serve", help="GUI/automation protocol mode (JSON lines on stdio)")
    serve.add_argument("--config", default="localvoice.toml")
    serve.add_argument("--allow-inject", action="store_true", help="enable inject_audio (tests)")
    return parser


def _load_config(args):
    from localvoice.config import ConfigError, load_config

    try:
        cfg = load_config(Path(args.config), deep=getattr(args, "deep", False))
    except ConfigError as exc:
        raise SystemExit(f"config error: {exc}") from exc
    if getattr(args, "model", None):
        cfg.llm.model = args.model  # explicit CLI override beats config and --deep
    return cfg


def _make_engines(cfg):
    from localvoice.llm import build_llm_engine
    from localvoice.stt.whisper_mlx import WhisperMlxEngine
    from localvoice.tts.kokoro_mlx import KokoroMlxEngine

    return WhisperMlxEngine(cfg.stt), build_llm_engine(cfg.llm), KokoroMlxEngine(cfg.tts)


def _load_timed(name: str, engine, inference=None) -> None:
    t0 = time.perf_counter()
    print(f"loading {name}...", end=" ", flush=True)
    if inference is not None:
        inference.submit(engine.load).result()  # load on the inference thread
    else:
        engine.load()
    print(f"{time.perf_counter() - t0:.1f}s")


def _missing_models(cfg) -> list[tuple[str, str]]:
    """Return (label, model) for each engine model that is neither an existing
    local path nor an already-cached HF snapshot. A bare string that is neither
    a path nor a repo id raises SystemExit (unrecoverable misconfiguration)."""
    from huggingface_hub import snapshot_download

    missing: list[tuple[str, str]] = []
    for label, model in (("stt", cfg.stt.model), ("llm", cfg.llm.model), ("tts", cfg.tts.model)):
        if Path(model).expanduser().exists():
            continue
        if "/" not in model:
            raise SystemExit(
                f"{label}: model is neither an existing path nor an HF repo id: {model}"
            )
        try:
            snapshot_download(repo_id=model, local_files_only=True)
        except Exception:  # noqa: BLE001 — not in local HF cache
            missing.append((label, model))
    return missing


def _defuse_tqdm_mp_lock() -> None:
    """mlx-whisper's progress bar makes tqdm allocate a global multiprocessing
    RLock; because cmd_run exits via os._exit, the resource tracker reports that
    semaphore as leaked on every exit. We never fork, so pre-setting mp_lock=None
    (a state tqdm itself uses when the lock can't be created) skips it entirely."""
    try:
        from tqdm import std as tqdm_std

        if not hasattr(tqdm_std.TqdmDefaultWriteLock, "mp_lock"):
            tqdm_std.TqdmDefaultWriteLock.mp_lock = None
    except Exception:  # noqa: BLE001 — cosmetic-only guard; never block startup
        pass


def cmd_run(args) -> None:
    cfg = _load_config(args)
    _defuse_tqdm_mp_lock()
    _preflight()
    missing = _missing_models(cfg)
    if missing:
        listing = "\n".join(f"  {label}: {model}" for label, model in missing)
        raise SystemExit(
            "missing models (not a local path and not in the HF cache):\n"
            f"{listing}\n"
            "run `uv run localvoice setup` to download them first."
        )
    stt, llm, tts = _make_engines(cfg)
    from localvoice.app import Orchestrator
    from localvoice.audio.capture import MicCapture
    from localvoice.audio.player import AudioPlayer
    from localvoice.events import Event, EventType
    from localvoice.hotkey import HotkeyListener
    from localvoice.transcript import Transcript

    orch_ref = {}

    def on_finished():
        # Fire-time gen read: a drain callback stalled across a full barge-in +
        # re-release (microsecond window stretched over >300ms of human action)
        # could stamp the new generation. Judged unreachable in practice; if it
        # ever bites, plumb mark_end(gen) through PlaybackQueue instead.
        orch = orch_ref.get("orch")
        if orch is not None:
            orch.post(Event(EventType.RESPONSE_FINISHED, gen=orch._gen))

    player = AudioPlayer(cfg.audio, on_response_finished=on_finished)
    capture = MicCapture(cfg.audio)
    transcript = Transcript(cfg.llm.system_prompt)
    from concurrent.futures import ThreadPoolExecutor

    # One persistent thread owns every MLX import, load, and inference call:
    # MLX streams (e.g. mlx-lm's import-time generation stream) are only usable
    # on the thread that created them, so loads and pipelines must colocate.
    inference = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")
    orch = Orchestrator(
        capture=capture,
        player=player,
        stt=stt,
        llm=llm,
        tts=tts,
        transcript=transcript,
        keys_cfg=cfg.keys,
        tools_cfg=cfg.tools,
        inference=inference,
        think=args.think or cfg.llm.think,
    )
    orch_ref["orch"] = orch
    for name, engine in (("whisper", stt), ("llm", llm), ("kokoro", tts)):
        _load_timed(name, engine, inference)
    listener = HotkeyListener(cfg.keys, orch.post)
    try:
        capture.start()
        player.start()
        listener.start()
        print("ready - hold right-command and talk; esc stops; ctrl-c quits")
        orch.run_forever()
    except KeyboardInterrupt:
        pass
    finally:
        listener.stop()
        capture.stop()
        player.stop()
        inference.shutdown(wait=False, cancel_futures=True)
        # MLX Metal state created on the inference thread can SIGBUS during normal
        # interpreter teardown. Everything above already released OS resources, so
        # skip teardown entirely rather than crash on exit.
        import os

        os._exit(0)


def _preflight() -> None:
    from localvoice.hotkey import input_monitoring_ok

    ok = input_monitoring_ok()
    if ok is False:
        raise SystemExit(
            "Input Monitoring permission missing.\n"
            "Open System Settings -> Privacy & Security -> Input Monitoring and enable "
            "your terminal,\n"
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

    targets = _missing_models(cfg)
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


def cmd_serve(args) -> None:
    cfg = _load_config(args)
    _defuse_tqdm_mp_lock()
    from localvoice.serve import Serve

    Serve(Path(args.config), cfg, allow_inject=args.allow_inject).run()


_COMMANDS = {"run", "setup", "bench", "serve"}


def _normalize_argv(argv: list[str]) -> list[str]:
    """`localvoice --think` must mean `localvoice run --think`: flags belong to
    subparsers, so inject the default subcommand when none was given (but let
    bare -h/--help reach the top-level parser)."""
    if argv and argv[0] in ("-h", "--help"):
        return argv
    if not argv or argv[0] not in _COMMANDS:
        return ["run", *argv]
    return argv


def parse_cli(argv: list[str] | None = None) -> argparse.Namespace:
    if argv is None:
        argv = sys.argv[1:]
    return build_parser().parse_args(_normalize_argv(argv))


def main() -> None:
    args = parse_cli()
    {"run": cmd_run, "setup": cmd_setup, "bench": cmd_bench, "serve": cmd_serve}[args.command](
        args
    )


if __name__ == "__main__":
    main()
