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
from localvoice.overlay import apply_overlay_changes, remove_overlay_keys
from localvoice.schema import build_schema, coerce

PROTOCOL_VERSION = 1
_SAMPLE_LINE = "This is what the selected voice sounds like."


def _changed_keys(old: Config, new: Config) -> set:
    """Dotted keys whose value differs between two merged configs. Used by
    reset_config to feed plan_apply only the fields that actually changed, so
    clearing an overlay key that already equalled its base default triggers no
    reload/restart."""
    from dataclasses import fields

    changed: set = set()
    for section in fields(Config):
        old_sec, new_sec = getattr(old, section.name), getattr(new, section.name)
        for f in fields(type(old_sec)):
            if getattr(old_sec, f.name) != getattr(new_sec, f.name):
                changed.add(f"{section.name}.{f.name}")
    return changed


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

    def _on_load_error(self, name: str, exc: Exception) -> None:
        self.emit({"event": "error", "message": f"{name} load failed: {exc}"})

    def _load_engines(self) -> None:
        self._engines.load_all(
            lambda n, p, s: self.emit(
                {"event": "load_progress", "engine": n, "phase": p, "seconds": s}
            ),
            on_error=self._on_load_error,
        )
        self._maybe_engines_ready()

    def _maybe_engines_ready(self) -> None:
        if not self._engines_ready and self._engines.loaded == {"stt", "llm", "tts"}:
            self._engines_ready = True
            self.emit({"event": "engines_ready"})

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
        shutdown_requested = False
        for line in self._stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                self.emit({"event": "error", "message": f"bad json: {line[:80]}"})
                continue
            if not isinstance(msg, dict):
                self.emit({"event": "error", "message": f"bad message: {line[:80]}"})
                continue
            try:
                if self._dispatch(msg):
                    shutdown_requested = True
                    break
            except Exception as exc:  # noqa: BLE001 — protocol boundary: a command must never kill serve
                self.emit({"event": "error", "message": f"{type(exc).__name__}: {exc}"})
        if not shutdown_requested:
            # stdin EOF with no explicit shutdown means the GUI process died (pipe
            # closed): release the mic/player and exit rather than hang forever.
            self._shutdown()

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
                held_ms = int(msg.get("held_ms") or 500)
                self._orch.post(Event(EventType.PTT_UP, held_ms=held_ms))
        elif cmd == "esc":
            self._orch.post(Event(EventType.ESC))
        elif cmd == "set_config":
            self._set_config(msg.get("changes", {}))
        elif cmd == "reset_config":
            self._reset_config(msg.get("keep") or [])
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
        # set_config's changed-key set is exactly what the client asked to
        # change; plan_apply keys off dotted names, so the coerced values
        # themselves are irrelevant to the plan.
        self._apply_new_config(new_cfg, set(changes))

    def _reset_config(self, keep: list) -> None:
        """Clear every overlay key except those in `keep`, then route the
        resulting config diff through the SAME apply path set_config uses.
        Only keys whose merged value actually changed feed plan_apply, so
        cleared engine-bound keys reload engines, cleared audio keys restart
        streams, and cleared instant keys apply — while untouched keys cost
        nothing. Replies config_applied with the full new merged config."""
        old_cfg = load_config(self._config_path)
        try:
            remove_overlay_keys(self._config_path, keep)
        except OSError as exc:
            self.emit({"event": "error", "message": f"reset_config failed: {exc}"})
            return
        new_cfg = load_config(self._config_path)
        changed = _changed_keys(old_cfg, new_cfg)
        self._apply_new_config(new_cfg, changed)

    def _apply_new_config(self, new_cfg: Config, changed: set) -> None:
        """Shared back half of set_config/reset_config: given the freshly
        merged config and the set of dotted keys that changed, run the
        instant/reload/audio-restart plan and emit config_applied."""
        plan = plan_apply({k: None for k in changed})
        for key in plan["instant"]:
            section, name = key.split(".", 1)
            setattr(getattr(self._cfg, section), name, getattr(getattr(new_cfg, section), name))
        if "llm.think" in changed:
            self._orch._deps.think = self._cfg.llm.think
        if "llm.system_prompt" in changed:
            self._orch._transcript.set_system_prompt(self._cfg.llm.system_prompt)
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
            return
        self._maybe_engines_ready()

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
                # Compare-and-swap: only restore if nothing else (a concurrent
                # set_config) changed the voice while we were synthesizing —
                # otherwise we'd clobber that newer, intentional change.
                if self._cfg.tts.voice == voice:
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
