import threading
from dataclasses import dataclass
from typing import Protocol


@dataclass
class ToolResult:
    ok: bool
    content: dict      # becomes the role:"tool" message content (template renders mappings)
    summary: str       # one human line for the protocol event, e.g. "found 5 results"
    # Set on success by image-producing tools (e.g. ScreenshotTool/look_at_screen);
    # None for every other tool and for any failed result. A tool that sets this
    # only ever creates the file -- ownership of deleting it belongs to whatever
    # feeds it into the LLM engine downstream. Default keeps every existing
    # ToolResult(...) construction (web.py, pipeline.py's error paths) valid.
    image_path: str | None = None


class Tool(Protocol):
    name: str
    description: str
    parameters: dict   # JSON schema: {"type": "object", "properties": {...}, "required": [...]}
    def execute(self, args: dict, cancel: threading.Event) -> ToolResult: ...


def hf_tool_schema(tool: Tool) -> dict:
    """{"type": "function", "function": {"name", "description", "parameters"}} —
    the shape apply_chat_template's tools= expects."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }
