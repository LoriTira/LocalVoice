import subprocess
import tempfile
import threading
from pathlib import Path

import pytest

from localvoice.config import ToolsConfig
from localvoice.tools import registry_for
from localvoice.tools.base import ToolResult, hf_tool_schema
from localvoice.tools.screenshot import ScreenshotTool

NO_CANCEL = threading.Event()


@pytest.fixture(autouse=True)
def _confine_tempfiles_to_tmp_path(tmp_path, monkeypatch):
    # ScreenshotTool.execute() calls tempfile.mkstemp() with no dir=, which
    # defaults to the real system tmp dir -- fine in production, but tests
    # that make it write real bytes (the fake screencapture below) would
    # otherwise leak small *.png files there run after run. Confine every
    # test in this module to pytest's own auto-cleaned tmp_path instead.
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))


def _completed(returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode)


def _write_capture(cmd, size: int) -> None:
    # screencapture's last arg is the destination path -- write `size` bytes
    # there, standing in for the real screen image the binary would produce.
    with open(cmd[-1], "wb") as f:
        f.write(b"\0" * size)


def test_screenshot_success_carries_image_path(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[0] == "screencapture":
            _write_capture(cmd, 2048)  # comfortably over the 1 KB floor
        return _completed(0)

    monkeypatch.setattr("localvoice.tools.screenshot.subprocess.run", fake_run)
    tool = ScreenshotTool(ToolsConfig())
    r = tool.execute({}, NO_CANCEL)

    assert isinstance(r, ToolResult)
    assert r.ok is True
    assert r.content == {"status": "screenshot captured"}
    assert r.summary == "Captured the screen"
    assert r.image_path is not None and r.image_path.endswith(".png")
    assert [c[0] for c in calls] == ["screencapture", "sips"]
    assert calls[0] == ["screencapture", "-x", r.image_path]
    assert calls[1] == ["sips", "--resampleHeightWidthMax", "1536", r.image_path]


def test_screenshot_failure_exit_code_is_reported(monkeypatch):
    def fake_run(cmd, **kwargs):
        return _completed(1)  # e.g. screen-recording permission denied

    monkeypatch.setattr("localvoice.tools.screenshot.subprocess.run", fake_run)
    tool = ScreenshotTool(ToolsConfig())
    r = tool.execute({}, NO_CANCEL)

    assert r.ok is False
    assert r.content == {
        "error": "screen recording permission not granted or capture failed"
    }
    assert r.summary == "Could not capture the screen"
    assert r.image_path is None


def test_screenshot_tiny_file_is_treated_as_failure(monkeypatch):
    # A zero-exit screencapture that produces a near-empty file is exactly
    # what a denied screen-recording permission looks like on macOS -- the
    # binary doesn't always fail loudly.
    def fake_run(cmd, **kwargs):
        if cmd[0] == "screencapture":
            _write_capture(cmd, 10)  # well under the 1 KB floor
        return _completed(0)

    monkeypatch.setattr("localvoice.tools.screenshot.subprocess.run", fake_run)
    tool = ScreenshotTool(ToolsConfig())
    r = tool.execute({}, NO_CANCEL)

    assert r.ok is False
    assert r.summary == "Could not capture the screen"
    assert r.image_path is None


def test_screenshot_cancelled_before_capture(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "localvoice.tools.screenshot.subprocess.run",
        lambda cmd, **kwargs: calls.append(cmd) or _completed(0),
    )
    cancel = threading.Event()
    cancel.set()
    tool = ScreenshotTool(ToolsConfig())
    r = tool.execute({}, cancel)

    assert r.ok is False and r.content == {"error": "cancelled"}
    assert calls == []  # screencapture/sips never invoked


def _pngs_in(tmp_path) -> list:
    return list(tmp_path.glob("*.png"))


def test_screenshot_failure_unlinks_its_temp_file(monkeypatch, tmp_path):
    # A denied/failed capture must not leak the temp file it created: nothing
    # downstream ever learns the path on the failure branch, so the tool is
    # the only thing that can clean it up.
    def fake_run(cmd, **kwargs):
        if cmd[0] == "screencapture":
            _write_capture(cmd, 10)  # tiny -> treated as failure
        return _completed(0)

    monkeypatch.setattr("localvoice.tools.screenshot.subprocess.run", fake_run)
    r = ScreenshotTool(ToolsConfig()).execute({}, NO_CANCEL)

    assert r.ok is False and r.image_path is None
    assert _pngs_in(tmp_path) == []  # no orphaned capture left behind


def test_screenshot_timeout_unlinks_its_temp_file(monkeypatch, tmp_path):
    # A subprocess timeout mid-capture must also clean up rather than escape
    # as an uncaught exception dragging a leaked file with it.
    def fake_run(cmd, **kwargs):
        _write_capture(cmd, 4096)  # file exists on disk when the timeout hits
        raise subprocess.TimeoutExpired(cmd, _kwargs_timeout(kwargs))

    monkeypatch.setattr("localvoice.tools.screenshot.subprocess.run", fake_run)
    r = ScreenshotTool(ToolsConfig()).execute({}, NO_CANCEL)

    assert r.ok is False and r.summary == "Could not capture the screen"
    assert _pngs_in(tmp_path) == []


def _kwargs_timeout(kwargs):
    return kwargs.get("timeout", 0)


def test_screenshot_success_keeps_its_temp_file(monkeypatch, tmp_path):
    # The mirror image: the success path must NOT unlink -- it hands the file
    # downstream via image_path for the pipeline to consume then delete.
    def fake_run(cmd, **kwargs):
        if cmd[0] == "screencapture":
            _write_capture(cmd, 2048)
        return _completed(0)

    monkeypatch.setattr("localvoice.tools.screenshot.subprocess.run", fake_run)
    r = ScreenshotTool(ToolsConfig()).execute({}, NO_CANCEL)

    assert r.ok is True and r.image_path is not None
    assert _pngs_in(tmp_path) == [Path(r.image_path)]


def test_screenshot_tool_schema_and_gating_flag():
    tool = ScreenshotTool(ToolsConfig())
    assert tool.name == "look_at_screen"
    assert tool.needs_image_engine is True  # consulted at app.py's offer site

    schema = hf_tool_schema(tool)
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "look_at_screen"
    assert schema["function"]["parameters"] == {
        "type": "object",
        "properties": {},
        "required": [],
    }


def test_registry_includes_screenshot_when_enabled():
    names = [t.name for t in registry_for(ToolsConfig(screenshot=True))]
    assert "look_at_screen" in names


def test_registry_excludes_screenshot_by_default():
    # ToolsConfig()'s screenshot default is False -- unchanged by T3.
    names = [t.name for t in registry_for(ToolsConfig())]
    assert "look_at_screen" not in names


def test_registry_excludes_screenshot_when_master_switch_off():
    names = [t.name for t in registry_for(ToolsConfig(enabled=False, screenshot=True))]
    assert names == []
