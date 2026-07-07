import threading
import urllib.parse

from ddgs import DDGS
from ddgs.exceptions import DDGSException
from trafilatura import extract, fetch_url
from trafilatura.settings import use_config

from localvoice.tools.base import ToolResult

# Shared across every WebSearchTool instance: app.py rebuilds the tool registry
# every turn, so a per-instance cache would never survive to the follow-up
# question it exists for (spec §4). Keyed (query, n). Capped FIFO so a
# pathological session can't grow it without bound.
_SEARCH_CACHE: dict[tuple[str, int], ToolResult] = {}
_CACHE_MAX = 32

# trafilatura's default DOWNLOAD_TIMEOUT is 30s; the binding caps request
# timeouts at 5s (one MLX inference thread must not wedge for ~90s across
# rounds against a barge-in). use_config() returns a fresh copy of
# DEFAULT_CONFIG, so mutating it here leaves trafilatura's global untouched.
_FETCH_CONFIG = use_config()
_FETCH_CONFIG.set("DEFAULT", "DOWNLOAD_TIMEOUT", "5")


def _cancelled() -> ToolResult:
    return ToolResult(ok=False, content={"error": "cancelled"}, summary="cancelled")


def _offline() -> ToolResult:
    return ToolResult(
        ok=False,
        content={"error": "no internet connection"},
        summary="Search failed: no internet connection",
    )


def _cache_put(key: tuple[str, int], result: ToolResult) -> None:
    if key not in _SEARCH_CACHE and len(_SEARCH_CACHE) >= _CACHE_MAX:
        _SEARCH_CACHE.pop(next(iter(_SEARCH_CACHE)))  # FIFO: drop the oldest
    _SEARCH_CACHE[key] = result


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

    def execute(self, args, cancel: threading.Event) -> ToolResult:
        if cancel.is_set():
            return _cancelled()

        query = args["query"]
        n = min(int(args.get("max_results", self._cfg.search_results)), 10)

        key = (query, n)
        cached = _SEARCH_CACHE.get(key)
        if cached is not None:
            return cached

        try:
            with DDGS() as d:
                rows = d.text(query, max_results=n)
        except DDGSException as exc:
            # ddgs raises a plain DDGSException("No results found.") on a
            # zero-hit query — an honest empty result, not an outage. Report it
            # as such so the model isn't told the network is down when it just
            # found nothing. A rate-limit or timeout (DDGSException subclasses)
            # or an HTTP-error DDGSException is a real failure and falls through
            # to the offline message below.
            if type(exc) is DDGSException and "No results found" in str(exc):
                result = ToolResult(
                    ok=True,
                    content={"results": []},
                    summary=f"Found 0 results for: {query}",
                )
                _cache_put(key, result)
                return result
            return _offline()
        except Exception:
            # Anything else (OSError, connection failures) is a real outage.
            return _offline()

        results = [
            {"title": row["title"], "url": row["href"], "snippet": row["body"]}
            for row in rows
        ]
        result = ToolResult(
            ok=True,
            content={"results": results},
            summary=f"Found {len(results)} results for: {query}",
        )
        _cache_put(key, result)
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
            html = fetch_url(url, config=_FETCH_CONFIG)
        except Exception:
            html = None

        if html is None:
            return ToolResult(
                ok=False,
                content={"error": "could not fetch the page"},
                summary="Could not fetch the page",
            )

        text = extract(html, include_comments=False) or ""
        text = text[: self._cfg.page_char_cap]
        domain = urllib.parse.urlparse(url).netloc
        return ToolResult(
            ok=True,
            content={"url": url, "text": text},
            summary=f"Read {len(text)} chars from {domain}",
        )
