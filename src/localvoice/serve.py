import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from localvoice.config import Config, ConfigError, load_config
from localvoice.engineset import EngineSet, plan_apply
from localvoice.events import Event, EventType
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
        self._state_cv = threading.Condition()
        self._last_state = None

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
            on_state=self._on_state,
        )
        d = self._orch._deps
        d.on_user_text = lambda t: self.emit({"event": "user_text", "text": t})
        d.on_assistant_clause = lambda t: self.emit({"event": "assistant_clause", "text": t})
        d.on_thinking = lambda t: self.emit({"event": "reasoning", "text": t})
        d.on_metrics = lambda m: self.emit({"event": "turn_done", "latency": m})

    def _on_drained(self) -> None:
        self._orch.post(Event(EventType.RESPONSE_FINISHED, gen=self._orch._gen))

    def _on_state(self, st) -> None:
        with self._state_cv:
            self._last_state = st
            self._state_cv.notify_all()
        self.emit({"event": "state", "state": st.name.lower()})

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
        self._loop = threading.Thread(target=self._orch.run_forever, daemon=True)
        self._loop.start()
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
        from localvoice.events import State

        self._orch.post(Event(EventType.PTT_DOWN))
        with self._state_cv:
            armed = self._state_cv.wait_for(
                lambda: self._last_state is State.LISTENING, timeout=2.0
            )
        if not armed:
            self.emit({"event": "error", "message": "inject_audio: capture never armed"})
            return
        if hasattr(self._capture, "buffer"):
            self._capture.buffer.write(audio)
        self._orch.post(Event(EventType.PTT_UP, held_ms=int(len(audio) / 16)))

    def _shutdown(self) -> None:
        self._orch.shutdown()
        if getattr(self, "_loop", None) is not None:
            self._loop.join(timeout=5)
        for dev in (self._player, self._capture):
            try:
                dev.stop()
            except Exception:  # noqa: BLE001
                pass
        self._inference.shutdown(wait=False, cancel_futures=True)
        self._exit(0)
