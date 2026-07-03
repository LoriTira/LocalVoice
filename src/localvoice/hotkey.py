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
