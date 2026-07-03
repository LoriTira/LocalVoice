import argparse
import time
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="localvoice", description="Local push-to-talk voice assistant"
    )
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


def cmd_run(args) -> None:
    cfg = _load_config(args)
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
    orch = Orchestrator(
        capture=capture,
        player=player,
        stt=stt,
        llm=llm,
        tts=tts,
        transcript=transcript,
        keys_cfg=cfg.keys,
        think=args.think or cfg.llm.think,
    )
    orch_ref["orch"] = orch
    for name, engine in (("whisper", stt), ("llm", llm), ("kokoro", tts)):
        _load_timed(name, engine)
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


def main() -> None:
    args = build_parser().parse_args()
    {"run": cmd_run, "setup": cmd_setup, "bench": cmd_bench}[args.command](args)


if __name__ == "__main__":
    main()
