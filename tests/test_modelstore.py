from pathlib import Path

from localvoice.modelstore import dir_bytes, download_pct, scan_models


def make_tree(tmp_path: Path) -> Path:
    root = tmp_path / "models"
    a = root / "pub" / "ModelA-MLX"
    a.mkdir(parents=True)
    (a / "model.safetensors").write_bytes(b"x" * 2048)
    b = root / "pub" / "ModelB-GGUF"
    b.mkdir(parents=True)
    (b / "weights.gguf").write_bytes(b"y" * 1024)
    return root


def test_scan_models_kinds_and_sizes(tmp_path):
    entries = {e["id"]: e for e in scan_models(lmstudio_root=make_tree(tmp_path))}
    assert entries["pub/ModelA-MLX"]["kind"] == "mlx"
    assert entries["pub/ModelB-GGUF"]["kind"] == "gguf"
    assert entries["pub/ModelA-MLX"]["size_gb"] == 0.0  # tiny fixture rounds to 0.0
    assert Path(entries["pub/ModelA-MLX"]["path"]).exists()


def test_scan_models_missing_root(tmp_path):
    assert scan_models(lmstudio_root=tmp_path / "nope") == []


def test_download_pct_math():
    assert download_pct(50, 200) == 25.0
    assert download_pct(300, 200) == 100.0
    assert download_pct(10, None) is None


def test_dir_bytes(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "f").write_bytes(b"z" * 10)
    (tmp_path / "g").write_bytes(b"z" * 5)
    assert dir_bytes(tmp_path) == 15
