import threading
import time
from collections.abc import Callable
from pathlib import Path


def dir_bytes(path: Path) -> int:
    total = 0
    for p in Path(path).rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue  # racing writer (e.g. a download) deleted/replaced p; skip it
    return total


def download_pct(local_bytes: int, total_bytes: int | None) -> float | None:
    if not total_bytes:
        return None
    return min(100.0, round(local_bytes / total_bytes * 100, 1))


def _kind(path: Path) -> str:
    if any(path.rglob("*.safetensors")):
        return "mlx"
    if any(path.rglob("*.gguf")):
        return "gguf"
    return "other"


def scan_models(lmstudio_root: Path | None = None) -> list[dict]:
    is_default_root = lmstudio_root is None
    root = Path(lmstudio_root or Path.home() / ".lmstudio/models")
    out: dict[str, dict] = {}
    if root.exists():
        for pub in sorted(p for p in root.iterdir() if p.is_dir()):
            for model in sorted(m for m in pub.iterdir() if m.is_dir()):
                mid = f"{pub.name}/{model.name}"
                out[mid] = {
                    "id": mid,
                    "path": str(model),
                    "size_gb": round(dir_bytes(model) / 1e9, 1),
                    "kind": _kind(model),
                }
    if is_default_root:
        try:  # best-effort HF cache listing
            from huggingface_hub import scan_cache_dir

            for repo in scan_cache_dir().repos:
                if repo.repo_type == "model" and repo.repo_id not in out:
                    out[repo.repo_id] = {
                        "id": repo.repo_id,
                        "path": str(repo.repo_path),
                        "size_gb": round(repo.size_on_disk / 1e9, 1),
                        "kind": "mlx",
                    }
        except Exception:  # noqa: BLE001 — cache scan is optional
            pass
    return list(out.values())


def _remote_total(repo: str) -> int | None:
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo, files_metadata=True)
        sizes = [s.size for s in info.siblings if s.size]
        return sum(sizes) or None
    except Exception:  # noqa: BLE001 — progress falls back to indeterminate
        return None


def download(
    repo: str,
    on_pct: Callable[[float | None], None],
    poll_s: float = 1.0,
) -> None:
    from huggingface_hub import snapshot_download

    total = _remote_total(repo)
    stop = threading.Event()

    def poller() -> None:
        from huggingface_hub.constants import HF_HUB_CACHE

        repo_dir = Path(HF_HUB_CACHE) / ("models--" + repo.replace("/", "--"))
        while not stop.is_set():
            try:
                if repo_dir.exists():
                    on_pct(download_pct(dir_bytes(repo_dir), total))
            except OSError:
                pass  # racing writer mid-tick; skip this tick, try again next poll
            time.sleep(poll_s)

    t = threading.Thread(target=poller, daemon=True)
    t.start()
    try:
        snapshot_download(repo_id=repo)
    finally:
        stop.set()
        t.join(timeout=2)
    on_pct(100.0)
