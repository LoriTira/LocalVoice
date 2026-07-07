from localvoice.config import ToolsConfig
from localvoice.tools import registry_for
from localvoice.tools.base import ToolResult, hf_tool_schema


def test_registry_disabled_master_switch_is_empty():
    assert registry_for(ToolsConfig(enabled=False)) == []


def test_registry_web_search_gated():
    names = [t.name for t in registry_for(ToolsConfig(web_search=False))]
    assert "web_search" not in names and "fetch_page" not in names


def test_registry_default_has_both_web_tools():
    names = [t.name for t in registry_for(ToolsConfig())]
    assert names == ["web_search", "fetch_page"]


def test_hf_tool_schema_shape():
    tool = registry_for(ToolsConfig())[0]
    s = hf_tool_schema(tool)
    assert s["type"] == "function"
    assert s["function"]["name"] == "web_search"
    assert s["function"]["parameters"]["type"] == "object"
    assert "query" in s["function"]["parameters"]["properties"]


def test_tool_result_fields():
    r = ToolResult(ok=False, content={"error": "no internet connection"}, summary="search failed")
    assert r.ok is False and r.content["error"] and r.summary
