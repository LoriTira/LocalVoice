import tomllib
from pathlib import Path

from localvoice.overlay import apply_overlay_changes


def test_creates_overlay_and_nests_dotted_keys(tmp_path: Path):
    cfg = tmp_path / "localvoice.toml"
    cfg.write_text("")
    p = apply_overlay_changes(cfg, {"llm.think": True, "tts.speed": 1.2})
    data = tomllib.loads(p.read_text())
    assert p.name == "localvoice.local.toml"
    assert data == {"llm": {"think": True}, "tts": {"speed": 1.2}}


def test_preserves_unrelated_existing_keys(tmp_path: Path):
    cfg = tmp_path / "localvoice.toml"
    cfg.write_text("")
    (tmp_path / "localvoice.local.toml").write_text('[llm]\nmodel = "keep/me"\n')
    apply_overlay_changes(cfg, {"llm.think": False})
    data = tomllib.loads((tmp_path / "localvoice.local.toml").read_text())
    assert data["llm"] == {"model": "keep/me", "think": False}
