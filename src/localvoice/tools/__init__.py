from localvoice.config import ToolsConfig


def registry_for(cfg: ToolsConfig) -> list:
    """Enabled tools only; empty when cfg.enabled is False. T1 registers
    WebSearchTool and FetchPageTool (both gated on cfg.web_search). T3 adds
    ScreenshotTool (gated on cfg.screenshot); it's filtered back out at the
    offer site in app.py for engines that can't consume the image it produces."""
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

    if cfg.screenshot:
        # Lazy import for the same reason as above: screencapture/sips are
        # macOS-only tools, kept out of the import graph when disabled.
        from localvoice.tools.screenshot import ScreenshotTool

        tools.append(ScreenshotTool(cfg))

    return tools
