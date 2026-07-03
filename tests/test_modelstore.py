import time
from pathlib import Path

from localvoice import modelstore
from localvoice.modelstore import dir_bytes, download, download_pct, scan_models


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


def test_dir_bytes_tolerates_file_deleted_between_rglob_and_stat(tmp_path, monkeypatch):
    """A racing writer/cleanup thread (e.g. huggingface_hub's own downloader
    pruning partial files) can delete a path between rglob() yielding it and
    stat() reading it. dir_bytes must skip that entry, not blow up the whole
    scan -- it's called from the download progress poller on a live,
    concurrently-written cache directory."""
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "f").write_bytes(b"z" * 10)
    doomed = tmp_path / "g"
    doomed.write_bytes(b"z" * 5)

    real_stat = Path.stat

    def flaky_stat(self, *a, **k):
        if self == doomed:
            raise OSError("No such file or directory (simulated race)")
        return real_stat(self, *a, **k)

    monkeypatch.setattr(Path, "stat", flaky_stat)
    assert dir_bytes(tmp_path) == 10  # only "sub/f" counted; "g" skipped, no raise


def test_download_poller_tolerates_oserror_and_keeps_polling(tmp_path, monkeypatch):
    """The download() progress poller reads a live, concurrently-written HF
    cache directory every poll_s; a transient OSError while computing the
    percentage (e.g. a file vanishing mid-scan, racing the exact snapshot
    layout the real download is writing) must not kill the poller thread --
    it should skip that tick and keep polling, so progress resumes on the
    next tick instead of silently freezing at the last good percentage."""
    repo = "org/repo"
    cache = tmp_path / "hub"
    repo_dir = cache / ("models--" + repo.replace("/", "--"))
    repo_dir.mkdir(parents=True)

    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache))

    calls = {"n": 0}

    def flaky_dir_bytes(path):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("simulated race mid-scan")
        return 42

    monkeypatch.setattr(modelstore, "dir_bytes", flaky_dir_bytes)

    def fake_snapshot_download(*, repo_id):
        time.sleep(0.25)  # let the poller tick at least twice (poll_s=0.05 below)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(modelstore, "_remote_total", lambda repo: 100)

    pcts: list = []
    download(repo, on_pct=pcts.append, poll_s=0.05)
    # First tick raised (skipped, no crash); a later tick succeeded with the fixed
    # dir_bytes return value 42 -> download_pct(42, 100) == 42.0. The final on_pct(100.0)
    # after snapshot_download always fires too.
    assert calls["n"] >= 2
    assert 42.0 in pcts
    assert pcts[-1] == 100.0
