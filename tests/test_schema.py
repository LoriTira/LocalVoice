from dataclasses import fields

import pytest

from localvoice.config import (
    AudioConfig,
    Config,
    ConfigError,
    KeysConfig,
    LlmConfig,
    SttConfig,
    ToolsConfig,
    TtsConfig,
)
from localvoice.schema import build_schema, coerce


def make_cfg() -> Config:
    return Config(SttConfig(), LlmConfig(), TtsConfig(), KeysConfig(), AudioConfig(), ToolsConfig())


def test_schema_covers_every_config_field_exactly_once():
    descriptors = build_schema(make_cfg())
    keys = [d["key"] for d in descriptors]
    expected = {
        f"{section}.{f.name}"
        for section, cls in (
            ("stt", SttConfig), ("llm", LlmConfig), ("tts", TtsConfig),
            ("keys", KeysConfig), ("audio", AudioConfig), ("tools", ToolsConfig),
        )
        for f in fields(cls)
    }
    assert set(keys) == expected and len(keys) == len(set(keys))


def test_descriptor_shape_and_annotations():
    d = {x["key"]: x for x in build_schema(make_cfg())}
    assert d["llm.think"]["widget"] == "toggle" and d["llm.think"]["type"] == "bool"
    assert d["llm.model"]["widget"] == "model_picker"
    assert d["tts.voice"]["widget"] == "voice_picker"
    assert d["keys.ptt"]["widget"] == "key_capture"
    assert d["audio.input_device"]["widget"] == "device_picker"
    assert d["tts.speed"]["widget"] == "slider" and d["tts.speed"]["minimum"] == 0.5
    assert d["audio.rebuffer_ms"]["widget"] == "slider"
    assert d["llm.max_tokens"]["widget"] == "number"
    assert d["llm.system_prompt"]["widget"] == "text"
    assert d["llm.think"]["section"] == "Language model"
    assert all(x["help"] is not None for x in build_schema(make_cfg()))


def test_values_reflect_config_instance():
    cfg = make_cfg()
    cfg.llm.max_tokens = 42
    d = {x["key"]: x for x in build_schema(cfg)}
    assert d["llm.max_tokens"]["value"] == 42 and d["llm.max_tokens"]["default"] == 1024


@pytest.mark.parametrize(
    "key,raw,expected",
    [
        ("llm.think", True, True),
        ("llm.think", "true", True),
        ("llm.think", "false", False),
        ("llm.max_tokens", "2048", 2048),
        ("llm.max_tokens", 2048, 2048),
        ("tts.speed", "1.2", 1.2),
        ("tts.voice", "af_bella", "af_bella"),
    ],
)
def test_coerce_valid(key, raw, expected):
    assert coerce(key, raw) == expected


def test_coerce_rejects_unknown_key_and_bad_value():
    with pytest.raises(ConfigError, match="nope.key"):
        coerce("nope.key", 1)
    with pytest.raises(ConfigError, match="llm.max_tokens"):
        coerce("llm.max_tokens", "not-a-number")
