import threading

from localvoice.config import ToolsConfig
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


def test_search_maps_results(monkeypatch):
    monkeypatch.setattr("localvoice.tools.web.DDGS", FakeDDGS)
    tool = WebSearchTool(ToolsConfig())
    r = tool.execute({"query": "boston weather"}, NO_CANCEL)
    assert r.ok and len(r.content["results"]) == 5
    assert set(r.content["results"][0]) == {"title", "url", "snippet"}
    assert "boston weather" in r.summary


def test_search_session_cache(monkeypatch):
    monkeypatch.setattr("localvoice.tools.web.DDGS", FakeDDGS)
    FakeDDGS.calls = 0
    tool = WebSearchTool(ToolsConfig())
    tool.execute({"query": "same"}, NO_CANCEL)
    tool.execute({"query": "same"}, NO_CANCEL)
    assert FakeDDGS.calls == 1


def test_search_offline_is_structured(monkeypatch):
    class Boom:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def text(self, *a, **k): raise OSError("network is unreachable")
    monkeypatch.setattr("localvoice.tools.web.DDGS", Boom)
    r = WebSearchTool(ToolsConfig()).execute({"query": "x"}, NO_CANCEL)
    assert r.ok is False and r.content["error"] == "no internet connection"


def test_search_cancelled_before_network(monkeypatch):
    monkeypatch.setattr("localvoice.tools.web.DDGS", FakeDDGS)
    cancel = threading.Event()
    cancel.set()
    r = WebSearchTool(ToolsConfig()).execute({"query": "x"}, cancel)
    assert r.ok is False and r.content["error"] == "cancelled"


def test_fetch_page_truncates(monkeypatch):
    monkeypatch.setattr("localvoice.tools.web.fetch_url", lambda url: "<html>raw</html>")
    monkeypatch.setattr("localvoice.tools.web.extract", lambda html, **k: "words " * 4000)
    tool = FetchPageTool(ToolsConfig(page_char_cap=100))
    r = tool.execute({"url": "https://ex.com/a"}, NO_CANCEL)
    assert r.ok and len(r.content["text"]) <= 100 and r.content["url"] == "https://ex.com/a"


def test_fetch_page_failure(monkeypatch):
    monkeypatch.setattr("localvoice.tools.web.fetch_url", lambda url: None)
    r = FetchPageTool(ToolsConfig()).execute({"url": "https://ex.com"}, NO_CANCEL)
    assert r.ok is False and "could not fetch" in r.summary
