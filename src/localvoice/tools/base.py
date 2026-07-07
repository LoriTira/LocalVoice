import threading
from dataclasses import dataclass
from typing import Protocol


@dataclass
class ToolResult:
    ok: bool
    content: dict      # becomes the role:"tool" message content (template renders mappings)
    summary: str       # one human line for the protocol event, e.g. "found 5 results"


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
