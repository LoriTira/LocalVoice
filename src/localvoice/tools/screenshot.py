import os
import subprocess
import tempfile
import threading
from pathlib import Path

from localvoice.tools.base import ToolResult

# screencapture/sips are each a single quick, local, non-interactive call;
# 10s is generous headroom without letting one wedge the one MLX inference
# thread for anywhere near as long as a barge-in needs to stay responsive.
_TIMEOUT_S = 10

# A denied screen-recording permission doesn't always make screencapture
# exit non-zero -- it can silently write a near-empty file instead. A real
# screen capture, even of a small display, is comfortably over this floor.
_MIN_VALID_BYTES = 1024


def _cancelled() -> ToolResult:
    return ToolResult(ok=False, content={"error": "cancelled"}, summary="cancelled")


def _capture_failed() -> ToolResult:
    return ToolResult(
        ok=False,
        content={"error": "screen recording permission not granted or capture failed"},
        summary="Could not capture the screen",
    )


class ScreenshotTool:
    name = "look_at_screen"
    description = "Capture the user's screen when they ask about something visible on it."
    parameters = {"type": "object", "properties": {}, "required": []}
    # Gating flag consulted at the offer site (app.py's per-turn tools
    # computation): a tool that produces an image must not be offered to an
    # engine that cannot consume one (getattr(llm, "supports_images", False)).
    needs_image_engine = True

    def __init__(self, cfg):
        self._cfg = cfg

    def execute(self, args: dict, cancel: threading.Event) -> ToolResult:
        if cancel.is_set():
            return _cancelled()

        fd, tmp_png = tempfile.mkstemp(suffix=".png")
        os.close(fd)  # screencapture writes to the path itself, not this fd

        capture = subprocess.run(
            ["screencapture", "-x", tmp_png],  # -x: silent, no camera-shutter sound
            capture_output=True,
            timeout=_TIMEOUT_S,
            check=False,
        )
        path = Path(tmp_png)
        size = path.stat().st_size if path.exists() else 0
        if capture.returncode != 0 or size < _MIN_VALID_BYTES:
            return _capture_failed()

        # Best-effort downscale so the image doesn't blow the vision engine's
        # token budget; the full-size capture already passed the checks above,
        # so a downscale hiccup is not treated as a capture failure.
        subprocess.run(
            ["sips", "--resampleHeightWidthMax", "1536", tmp_png],
            capture_output=True,
            timeout=_TIMEOUT_S,
            check=False,
        )

        return ToolResult(
            ok=True,
            content={"status": "screenshot captured"},
            summary="Captured the screen",
            image_path=tmp_png,
        )
