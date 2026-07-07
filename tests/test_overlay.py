import tomllib
from pathlib import Path

from localvoice.overlay import apply_overlay_changes, remove_overlay_keys


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


def test_remove_overlay_keys_clears_all_except_kept(tmp_path: Path):
    cfg = tmp_path / "localvoice.toml"
    cfg.write_text("")
    overlay = tmp_path / "localvoice.local.toml"
    overlay.write_text(
        '[llm]\nmodel = "keep/me"\nthink = true\nmax_tokens = 9000\n[tts]\nspeed = 1.4\n'
    )
    removed = remove_overlay_keys(cfg, keep=["llm.model"])
    data = tomllib.loads(overlay.read_text())
    # Only the kept key survives; every other overlay key is gone.
    assert data == {"llm": {"model": "keep/me"}}
    assert sorted(removed) == ["llm.max_tokens", "llm.think", "tts.speed"]


def test_remove_overlay_keys_prunes_now_empty_sections(tmp_path: Path):
    cfg = tmp_path / "localvoice.toml"
    cfg.write_text("")
    overlay = tmp_path / "localvoice.local.toml"
    # [tts] holds only a key that is cleared, so the whole section must be pruned.
    overlay.write_text('[llm]\nmodel = "keep/me"\n[tts]\nvoice = "af_bella"\n')
    remove_overlay_keys(cfg, keep=["llm.model"])
    data = tomllib.loads(overlay.read_text())
    assert data == {"llm": {"model": "keep/me"}}
    assert "tts" not in data


def test_remove_overlay_keys_missing_file_is_noop(tmp_path: Path):
    cfg = tmp_path / "localvoice.toml"
    cfg.write_text("")
    # No overlay on disk at all: removing keys must not create one or raise.
    removed = remove_overlay_keys(cfg, keep=["llm.model"])
    assert removed == []
    assert not (tmp_path / "localvoice.local.toml").exists()


def test_remove_overlay_keys_empty_keep_clears_everything_keeping_file(tmp_path: Path):
    cfg = tmp_path / "localvoice.toml"
    cfg.write_text("")
    overlay = tmp_path / "localvoice.local.toml"
    overlay.write_text('[llm]\nthink = true\n[tts]\nspeed = 1.4\n')
    removed = remove_overlay_keys(cfg, keep=[])
    # With nothing kept, all sections are pruned but the file stays present.
    assert overlay.exists()
    assert tomllib.loads(overlay.read_text()) == {}
    assert sorted(removed) == ["llm.think", "tts.speed"]


def test_remove_overlay_keys_unknown_keep_keys_are_ignored(tmp_path: Path):
    cfg = tmp_path / "localvoice.toml"
    cfg.write_text("")
    overlay = tmp_path / "localvoice.local.toml"
    overlay.write_text('[llm]\nthink = true\n')
    # A keep key that isn't in the overlay (or names a nonexistent section) is
    # harmless: it just doesn't protect anything.
    removed = remove_overlay_keys(cfg, keep=["llm.model", "stt.model", "bogus.key"])
    assert tomllib.loads(overlay.read_text()) == {}
    assert removed == ["llm.think"]


def test_remove_overlay_keys_writes_atomically(tmp_path: Path, monkeypatch):
    """A crash mid-write must never leave a half-written overlay: the helper
    writes a temp file and os.replace()s it into place, same as
    apply_overlay_changes. Prove the temp path is used (not written in-place)
    by making os.replace raise and asserting the original overlay is intact."""
    import os

    cfg = tmp_path / "localvoice.toml"
    cfg.write_text("")
    overlay = tmp_path / "localvoice.local.toml"
    original = '[llm]\nmodel = "keep/me"\nthink = true\n'
    overlay.write_text(original)

    def boom(src, dst):
        raise OSError("simulated crash during replace")

    monkeypatch.setattr(os, "replace", boom)
    try:
        remove_overlay_keys(cfg, keep=["llm.model"])
    except OSError:
        pass
    # The real overlay is byte-for-byte untouched — the write went to a temp path.
    assert overlay.read_text() == original
