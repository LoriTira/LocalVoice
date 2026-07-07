import os
import tomllib
from pathlib import Path

import tomli_w


def _overlay_path(config_path: Path) -> Path:
    return config_path.with_name(config_path.stem + ".local" + config_path.suffix)


def apply_overlay_changes(config_path: Path, changes: dict[str, object]) -> Path:
    overlay = _overlay_path(config_path)
    data: dict = tomllib.loads(overlay.read_text()) if overlay.exists() else {}
    for dotted, value in changes.items():
        section, name = dotted.split(".", 1)
        data.setdefault(section, {})[name] = value
    tmp = overlay.with_suffix(".tmp")
    tmp.write_bytes(tomli_w.dumps(data).encode())
    os.replace(tmp, overlay)
    return overlay


def remove_overlay_keys(config_path: Path, keep: list[str]) -> list[str]:
    """Remove every dotted key currently set in the overlay except those in
    `keep`, rewriting it atomically. Prunes sections left empty; keeps the
    file present even when nothing remains (writes an empty document rather
    than deleting it) so the "your overlay is <name>" contract still holds.

    Returns the list of dotted keys actually removed. Unknown `keep` entries
    (keys or whole sections not present in the overlay) are ignored
    harmlessly. A missing overlay file is a no-op: nothing to clear, so
    nothing is written and no file is created.
    """
    overlay = _overlay_path(config_path)
    if not overlay.exists():
        return []
    data: dict = tomllib.loads(overlay.read_text())
    keep_set = set(keep)
    removed: list[str] = []
    new_data: dict = {}
    for section, values in data.items():
        kept_section: dict = {}
        for name, value in values.items():
            dotted = f"{section}.{name}"
            if dotted in keep_set:
                kept_section[name] = value
            else:
                removed.append(dotted)
        if kept_section:  # drop now-empty sections
            new_data[section] = kept_section
    tmp = overlay.with_suffix(".tmp")
    tmp.write_bytes(tomli_w.dumps(new_data).encode())
    os.replace(tmp, overlay)
    return removed
