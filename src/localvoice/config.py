import tomllib
from dataclasses import dataclass, fields
from pathlib import Path


class ConfigError(Exception):
    pass


@dataclass
class SttConfig:
    engine: str = "mlx_whisper"
    model: str = "mlx-community/whisper-large-v3-turbo"


@dataclass
class LlmConfig:
    engine: str = "mlx_lm"
    model: str = "mlx-community/Qwen3.6-35B-A3B-4bit"
    deep_model: str = ""
    think: bool = False
    max_tokens: int = 1024
    system_prompt: str = "You are a helpful voice assistant. Keep answers short and spoken-friendly."


@dataclass
class TtsConfig:
    engine: str = "kokoro_mlx"
    model: str = "prince-canuma/Kokoro-82M"
    voice: str = "af_heart"
    speed: float = 1.0


@dataclass
class KeysConfig:
    ptt: str = "cmd_r"
    stop: str = "esc"
    debounce_ms: int = 120


@dataclass
class AudioConfig:
    input_device: str = ""
    output_device: str = ""


@dataclass
class Config:
    stt: SttConfig
    llm: LlmConfig
    tts: TtsConfig
    keys: KeysConfig
    audio: AudioConfig


_SECTIONS = {"stt": SttConfig, "llm": LlmConfig, "tts": TtsConfig, "keys": KeysConfig, "audio": AudioConfig}


def _merge(base: dict, overlay: dict) -> dict:
    out = {k: dict(v) for k, v in base.items()}
    for section, values in overlay.items():
        out.setdefault(section, {}).update(values)
    return out


def _build(data: dict) -> Config:
    kwargs = {}
    for section, values in data.items():
        cls = _SECTIONS.get(section)
        if cls is None:
            raise ConfigError(f"unknown config section [{section}]")
        known = {f.name for f in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ConfigError(f"unknown key in [{section}]: {sorted(unknown)[0]}")
        kwargs[section] = cls(**values)
    for section, cls in _SECTIONS.items():
        kwargs.setdefault(section, cls())
    return Config(**kwargs)


def load_config(path: Path, *, deep: bool = False) -> Config:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config not found: {path}")
    data = tomllib.loads(path.read_text())
    local = path.with_name(path.stem + ".local" + path.suffix)
    if local.exists():
        data = _merge(data, tomllib.loads(local.read_text()))
    cfg = _build(data)
    if deep:
        if not cfg.llm.deep_model:
            raise ConfigError("--deep requested but [llm].deep_model is empty")
        cfg.llm.model = cfg.llm.deep_model
    return cfg
