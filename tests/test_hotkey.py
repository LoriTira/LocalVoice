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
    keys = KeysConfig(ptt="cmd_r", stop="esc", debounce_ms=120)
    return HotkeyListener(keys, events.append), events


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
