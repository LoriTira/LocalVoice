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
    from localvoice.llm import build_llm_engine
    from localvoice.stt.whisper_mlx import WhisperMlxEngine
    from localvoice.tts.kokoro_mlx import KokoroMlxEngine

    return {"stt": WhisperMlxEngine, "llm": build_llm_engine, "tts": KokoroMlxEngine}


class EngineSet:
    def __init__(self, cfg: Config, factories: dict | None = None) -> None:
        self._factories = factories or _default_factories()
        self.stt = EngineProxy(self._factories["stt"](cfg.stt))
        self.llm = EngineProxy(self._factories["llm"](cfg.llm))
        self.tts = EngineProxy(self._factories["tts"](cfg.tts))
        self.loaded: set[str] = set()

    def _cfg_for(self, name: str, cfg: Config):
        return {"stt": cfg.stt, "llm": cfg.llm, "tts": cfg.tts}[name]

    def _load(self, name: str, engine, on_progress: Callable) -> None:
        on_progress(name, "start", None)
        t0 = time.perf_counter()
        engine.load()
        on_progress(name, "done", time.perf_counter() - t0)
        self.loaded.add(name)

    def load_all(self, on_progress: Callable, on_error: Callable | None = None) -> None:
        for name in _ORDER:
            try:
                self._load(name, getattr(self, name)._target, on_progress)
            except Exception as exc:  # noqa: BLE001 — routed to on_error or re-raised below
                if on_error is not None:
                    on_error(name, exc)
                    continue
                raise

    def reload(self, name: str, cfg: Config, on_progress: Callable) -> None:
        new = self._factories[name](self._cfg_for(name, cfg))
        self._load(name, new, on_progress)
        getattr(self, name).swap(new)
