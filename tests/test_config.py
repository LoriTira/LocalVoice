from pathlib import Path

import pytest

from localvoice.config import Config, ConfigError, load_config

BASE = """
[stt]
engine = "mlx_whisper"
model = "repo/whisper"
[llm]
engine = "mlx_lm"
model = "repo/big"
deep_model = "repo/deep"
think = false
max_tokens = 512
system_prompt = "be brief"
[tts]
engine = "kokoro_mlx"
model = "repo/kokoro"
voice = "af_heart"
speed = 1.0
[keys]
ptt = "cmd_r"
stop = "esc"
debounce_ms = 120
[audio]
input_device = ""
output_device = ""
"""


def write(tmp_path: Path, text: str, name: str = "localvoice.toml") -> Path:
    p = tmp_path / name
    p.write_text(text)
    return p


def test_loads_full_config(tmp_path):
    cfg = load_config(write(tmp_path, BASE))
    assert isinstance(cfg, Config)
    assert cfg.llm.model == "repo/big"
    assert cfg.keys.debounce_ms == 120
    assert cfg.tts.speed == 1.0


def test_local_overlay_wins(tmp_path):
    write(tmp_path, BASE)
    write(tmp_path, '[llm]\nmodel = "local/path"\n', "localvoice.local.toml")
    cfg = load_config(tmp_path / "localvoice.toml")
    assert cfg.llm.model == "local/path"
    assert cfg.llm.max_tokens == 512  # untouched keys survive the merge


def test_deep_flag_swaps_model(tmp_path):
    cfg = load_config(write(tmp_path, BASE), deep=True)
    assert cfg.llm.model == "repo/deep"


def test_deep_flag_without_deep_model_raises(tmp_path):
    text = BASE.replace('deep_model = "repo/deep"', 'deep_model = ""')
    with pytest.raises(ConfigError, match="deep_model"):
        load_config(write(tmp_path, text), deep=True)


def test_unknown_key_raises(tmp_path):
    with pytest.raises(ConfigError, match="tts2"):
        load_config(write(tmp_path, BASE + "\n[tts2]\nvolume = 3\n").parent / "localvoice.toml")


def test_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="nope.toml"):
        load_config(tmp_path / "nope.toml")


def test_tools_config_defaults():
    from localvoice.config import ToolsConfig

    t = ToolsConfig()
    assert t.enabled is True
    assert t.web_search is True
    assert t.screenshot is False
    assert t.max_rounds == 3
    assert t.search_results == 5
    assert t.page_char_cap == 8000


def test_config_carries_tools_section(tmp_path):
    from localvoice.config import load_config

    p = tmp_path / "localvoice.toml"
    p.write_text('[tools]\nenabled = false\nmax_rounds = 5\n')
    cfg = load_config(p)
    assert cfg.tools.enabled is False
    assert cfg.tools.max_rounds == 5
    assert cfg.tools.web_search is True  # untouched keys keep defaults
