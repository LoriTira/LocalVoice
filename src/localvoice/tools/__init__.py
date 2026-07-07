from localvoice.config import ToolsConfig


def registry_for(cfg: ToolsConfig) -> list:
    """Enabled tools only; empty when cfg.enabled is False. T1 registers
    WebSearchTool and FetchPageTool (both gated on cfg.web_search)."""
    if not cfg.enabled:
        return []

    tools = []

    if cfg.web_search:
        # Imported here, not at module top: web.py carries the network
        # dependencies (ddgs, trafilatura), which must not load just because
        # config code touched the registry with tools disabled.
        from localvoice.tools.web import FetchPageTool, WebSearchTool

        tools.append(WebSearchTool(cfg))
        tools.append(FetchPageTool(cfg))

    return tools
