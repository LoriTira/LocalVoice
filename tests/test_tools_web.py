import threading

from ddgs.exceptions import DDGSException

from localvoice.config import ToolsConfig
from localvoice.tools import web
from localvoice.tools.web import FetchPageTool, WebSearchTool

NO_CANCEL = threading.Event()


class FakeDDGS:
    calls = 0
    def __init__(self, *a, **k): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def text(self, query, max_results=5):
        FakeDDGS.calls += 1
        return [{"title": f"R{i}", "href": f"https://ex.com/{i}", "body": f"snippet {i}"}
                for i in range(max_results)]


def _clear_cache():
    web._SEARCH_CACHE.clear()


def test_search_maps_results(monkeypatch):
    _clear_cache()
    monkeypatch.setattr("localvoice.tools.web.DDGS", FakeDDGS)
    tool = WebSearchTool(ToolsConfig())
    r = tool.execute({"query": "boston weather"}, NO_CANCEL)
    assert r.ok and len(r.content["results"]) == 5
    assert set(r.content["results"][0]) == {"title", "url", "snippet"}
    assert r.summary == "Found 5 results for: boston weather"


def test_search_session_cache(monkeypatch):
    _clear_cache()
    monkeypatch.setattr("localvoice.tools.web.DDGS", FakeDDGS)
    FakeDDGS.calls = 0
    tool = WebSearchTool(ToolsConfig())
    tool.execute({"query": "same"}, NO_CANCEL)
    tool.execute({"query": "same"}, NO_CANCEL)
    assert FakeDDGS.calls == 1


def test_search_cache_shared_across_instances(monkeypatch):
    # app.py rebuilds tools every turn, so the cache the search tool exists
    # for (a follow-up question reusing the same query) only pays off if it
    # survives instance churn — it lives at module scope, keyed (query, n).
    _clear_cache()
    monkeypatch.setattr("localvoice.tools.web.DDGS", FakeDDGS)
    FakeDDGS.calls = 0
    WebSearchTool(ToolsConfig()).execute({"query": "same"}, NO_CANCEL)
    r = WebSearchTool(ToolsConfig()).execute({"query": "same"}, NO_CANCEL)
    assert FakeDDGS.calls == 1  # second, fresh instance hit the shared cache
    assert r.ok and len(r.content["results"]) == 5


def test_search_cache_is_bounded(monkeypatch):
    # A pathological session must not grow the cache without limit; it keeps
    # only the last _CACHE_MAX distinct (query, n) keys, FIFO.
    _clear_cache()
    monkeypatch.setattr("localvoice.tools.web.DDGS", FakeDDGS)
    tool = WebSearchTool(ToolsConfig())
    for i in range(web._CACHE_MAX + 5):
        tool.execute({"query": f"q{i}"}, NO_CANCEL)
    assert len(web._SEARCH_CACHE) == web._CACHE_MAX
    # The earliest keys were evicted; the most recent survive.
    assert ("q0", 5) not in web._SEARCH_CACHE
    assert (f"q{web._CACHE_MAX + 4}", 5) in web._SEARCH_CACHE


def test_search_offline_is_structured(monkeypatch):
    _clear_cache()
    class Boom:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def text(self, *a, **k): raise OSError("network is unreachable")
    monkeypatch.setattr("localvoice.tools.web.DDGS", Boom)
    r = WebSearchTool(ToolsConfig()).execute({"query": "x"}, NO_CANCEL)
    assert r.ok is False and r.content["error"] == "no internet connection"
    assert r.summary == "Search failed: no internet connection"


def test_search_no_results_is_ok_not_offline(monkeypatch):
    # ddgs raises DDGSException("No results found.") on a zero-hit query —
    # that is an honest empty result, not an offline failure. It must report
    # ok=True with an empty list, distinct from the network-down message.
    _clear_cache()
    class NoHits:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def text(self, *a, **k): raise DDGSException("No results found.")
    monkeypatch.setattr("localvoice.tools.web.DDGS", NoHits)
    r = WebSearchTool(ToolsConfig()).execute({"query": "zxqwv"}, NO_CANCEL)
    assert r.ok is True and r.content["results"] == []
    assert r.summary == "Found 0 results for: zxqwv"


def test_search_timeout_subclass_is_offline_not_no_results(monkeypatch):
    # A TimeoutException/RatelimitException (DDGSException *subclasses*), or an
    # HTTP-error DDGSException, is a genuine failure — it must NOT be reported
    # as "found 0 results". Only the plain "No results found." case is empty.
    from ddgs.exceptions import TimeoutException

    _clear_cache()
    class TimedOut:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def text(self, *a, **k): raise TimeoutException("request timed out")
    monkeypatch.setattr("localvoice.tools.web.DDGS", TimedOut)
    r = WebSearchTool(ToolsConfig()).execute({"query": "x"}, NO_CANCEL)
    assert r.ok is False and r.content["error"] == "no internet connection"


def test_search_http_error_ddgs_exception_is_offline(monkeypatch):
    # A bare DDGSException that is NOT the "No results found." sentinel (e.g. an
    # HTTP error) is a real failure, not an empty result set.
    _clear_cache()
    class HttpFail:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def text(self, *a, **k): raise DDGSException("Failed to fetch: HTTP 503")
    monkeypatch.setattr("localvoice.tools.web.DDGS", HttpFail)
    r = WebSearchTool(ToolsConfig()).execute({"query": "x"}, NO_CANCEL)
    assert r.ok is False and r.content["error"] == "no internet connection"


def test_search_cancelled_before_network(monkeypatch):
    _clear_cache()
    monkeypatch.setattr("localvoice.tools.web.DDGS", FakeDDGS)
    cancel = threading.Event()
    cancel.set()
    r = WebSearchTool(ToolsConfig()).execute({"query": "x"}, cancel)
    assert r.ok is False and r.content["error"] == "cancelled"


def test_fetch_page_truncates(monkeypatch):
    monkeypatch.setattr(
        "localvoice.tools.web.fetch_url", lambda url, config=None: "<html>raw</html>"
    )
    monkeypatch.setattr("localvoice.tools.web.extract", lambda html, **k: "words " * 4000)
    tool = FetchPageTool(ToolsConfig(page_char_cap=100))
    r = tool.execute({"url": "https://ex.com/a"}, NO_CANCEL)
    assert r.ok and len(r.content["text"]) <= 100 and r.content["url"] == "https://ex.com/a"
    assert r.summary.startswith("Read ") and "ex.com" in r.summary


def test_fetch_page_uses_5s_timeout(monkeypatch):
    # The binding requires request timeouts ≤5s; trafilatura's default is 30s.
    # fetch_url must be handed a config carrying DOWNLOAD_TIMEOUT=5.
    captured = {}

    def spy(url, config=None):
        captured["config"] = config
        return "<html>x</html>"

    monkeypatch.setattr("localvoice.tools.web.fetch_url", spy)
    monkeypatch.setattr("localvoice.tools.web.extract", lambda html, **k: "text")
    FetchPageTool(ToolsConfig()).execute({"url": "https://ex.com"}, NO_CANCEL)
    cfg = captured["config"]
    assert cfg is not None
    assert cfg.getint("DEFAULT", "DOWNLOAD_TIMEOUT") == 5


def test_fetch_page_failure(monkeypatch):
    monkeypatch.setattr("localvoice.tools.web.fetch_url", lambda url, config=None: None)
    r = FetchPageTool(ToolsConfig()).execute({"url": "https://ex.com"}, NO_CANCEL)
    assert r.ok is False and "Could not fetch the page" == r.summary
