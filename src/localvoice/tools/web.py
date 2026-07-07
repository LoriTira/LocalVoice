import threading
import urllib.parse

from ddgs import DDGS
from trafilatura import extract, fetch_url

from localvoice.tools.base import ToolResult


def _cancelled() -> ToolResult:
    return ToolResult(ok=False, content={"error": "cancelled"}, summary="cancelled")


class WebSearchTool:
    name = "web_search"
    description = (
        "Search the web. Use for current events, prices, weather, or facts "
        "you are not confident about."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "the search query"},
            "max_results": {"type": "integer", "description": "how many results (1-10)"},
        },
        "required": ["query"],
    }

    def __init__(self, cfg):
        self._cfg = cfg
        self._cache: dict[tuple[str, int], ToolResult] = {}

    def execute(self, args, cancel: threading.Event) -> ToolResult:
        if cancel.is_set():
            return _cancelled()

        query = args["query"]
        n = min(int(args.get("max_results", self._cfg.search_results)), 10)

        key = (query, n)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        try:
            with DDGS() as d:
                rows = d.text(query, max_results=n)
        except Exception:
            return ToolResult(
                ok=False,
                content={"error": "no internet connection"},
                summary="search failed: no internet connection",
            )

        results = [
            {"title": row["title"], "url": row["href"], "snippet": row["body"]}
            for row in rows
        ]
        result = ToolResult(
            ok=True,
            content={"results": results},
            summary=f"found {len(results)} results for: {query}",
        )
        self._cache[key] = result
        return result


class FetchPageTool:
    name = "fetch_page"
    description = "Fetch one web page by URL and return its readable main text."
    parameters = {
        "type": "object",
        "properties": {"url": {"type": "string", "description": "absolute http(s) URL"}},
        "required": ["url"],
    }

    def __init__(self, cfg):
        self._cfg = cfg

    def execute(self, args, cancel: threading.Event) -> ToolResult:
        if cancel.is_set():
            return _cancelled()

        url = args["url"]

        try:
            html = fetch_url(url)
        except Exception:
            html = None

        if html is None:
            return ToolResult(
                ok=False,
                content={"error": "could not fetch the page"},
                summary="could not fetch the page",
            )

        text = extract(html, include_comments=False) or ""
        text = text[: self._cfg.page_char_cap]
        domain = urllib.parse.urlparse(url).netloc
        return ToolResult(
            ok=True,
            content={"url": url, "text": text},
            summary=f"read {len(text)} chars from {domain}",
        )
