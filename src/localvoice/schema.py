from dataclasses import fields
from typing import get_type_hints

from localvoice.config import (
    AudioConfig,
    Config,
    ConfigError,
    KeysConfig,
    LlmConfig,
    SttConfig,
    TtsConfig,
)

_SECTIONS = [
    ("stt", SttConfig, "Speech recognition"),
    ("llm", LlmConfig, "Language model"),
    ("tts", TtsConfig, "Speech"),
    ("keys", KeysConfig, "Keys"),
    ("audio", AudioConfig, "Audio"),
]

_TYPE_NAMES = {bool: "bool", int: "int", float: "float", str: "str"}
_DEFAULT_WIDGETS = {bool: "toggle", int: "number", float: "number", str: "text"}

_ANNOTATIONS: dict[str, dict] = {
    "stt.model": {"widget": "model_picker", "help": "HF repo id or local MLX folder."},
    "llm.model": {"widget": "model_picker", "help": "HF repo id or local MLX folder."},
    "llm.deep_model": {"widget": "model_picker", "help": "Used with --deep; empty disables it."},
    "llm.think": {"label": "Thinking mode", "help": "Silent reasoning; shown, never spoken."},
    "llm.system_prompt": {"help": "System message prepended to every conversation."},
    "tts.model": {"widget": "model_picker", "help": "Kokoro weights repo or folder."},
    "tts.voice": {"widget": "voice_picker", "help": "Kokoro voice preset."},
    "tts.speed": {
        "widget": "slider", "minimum": 0.5, "maximum": 2.0, "step": 0.1,
        "help": "Speaking-rate multiplier.",
    },
    "keys.ptt": {"widget": "key_capture", "label": "Push-to-talk key"},
    "keys.stop": {"widget": "key_capture", "label": "Stop key"},
    "keys.debounce_ms": {"help": "Presses shorter than this are ignored."},
    "audio.input_device": {"widget": "device_picker", "help": "Empty uses the system default."},
    "audio.output_device": {"widget": "device_picker", "help": "Empty uses the system default."},
    "audio.rebuffer_ms": {
        "widget": "slider", "minimum": 100, "maximum": 800, "step": 50,
        "help": "Anti-stutter gate after a mid-response stall.",
    },
}


def _field_type(cls, name: str) -> type:
    hints = get_type_hints(cls)
    t = hints[name]
    if t not in _TYPE_NAMES:
        raise ConfigError(f"unsupported config field type for {name}: {t}")
    return t


def build_schema(cfg: Config) -> list[dict]:
    out: list[dict] = []
    for section, cls, section_label in _SECTIONS:
        live = getattr(cfg, section)
        defaults = cls()
        for f in fields(cls):
            key = f"{section}.{f.name}"
            t = _field_type(cls, f.name)
            d = {
                "key": key,
                "type": _TYPE_NAMES[t],
                "value": getattr(live, f.name),
                "default": getattr(defaults, f.name),
                "section": section_label,
                "label": f.name.replace("_", " "),
                "help": "",
                "widget": _DEFAULT_WIDGETS[t],
            }
            d.update(_ANNOTATIONS.get(key, {}))
            out.append(d)
    return out


def _lookup(key: str) -> tuple[str, type, str]:
    try:
        section, name = key.split(".", 1)
    except ValueError as exc:
        raise ConfigError(f"unknown config key: {key}") from exc
    for sec, cls, _ in _SECTIONS:
        if sec == section and name in {f.name for f in fields(cls)}:
            return section, _field_type(cls, name), name
    raise ConfigError(f"unknown config key: {key}")


def coerce(key: str, raw: object) -> object:
    _, t, _ = _lookup(key)
    try:
        if t is bool:
            if isinstance(raw, bool):
                return raw
            if isinstance(raw, str) and raw.lower() in ("true", "false"):
                return raw.lower() == "true"
            raise ValueError(raw)
        return t(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"bad value for {key}: {raw!r}") from exc
