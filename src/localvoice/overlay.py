import os
import tomllib
from pathlib import Path

import tomli_w


def apply_overlay_changes(config_path: Path, changes: dict[str, object]) -> Path:
    overlay = config_path.with_name(
        config_path.stem + ".local" + config_path.suffix
    )
    data: dict = tomllib.loads(overlay.read_text()) if overlay.exists() else {}
    for dotted, value in changes.items():
        section, name = dotted.split(".", 1)
        data.setdefault(section, {})[name] = value
    tmp = overlay.with_suffix(".tmp")
    tmp.write_bytes(tomli_w.dumps(data).encode())
    os.replace(tmp, overlay)
    return overlay
