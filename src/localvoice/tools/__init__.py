from localvoice.config import ToolsConfig
from localvoice.tools.web import FetchPageTool, WebSearchTool


def registry_for(cfg: ToolsConfig) -> list:
    """Enabled tools only; empty when cfg.enabled is False. T1 registers
    WebSearchTool and FetchPageTool (both gated on cfg.web_search)."""
    if not cfg.enabled:
        return []

    tools = []

    if cfg.web_search:
        tools.append(WebSearchTool(cfg))
        tools.append(FetchPageTool(cfg))

    return tools
