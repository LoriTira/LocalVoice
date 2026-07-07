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
    def __init__(self, cfg): self._cfg = cfg
    def execute(self, args, cancel): raise NotImplementedError


class FetchPageTool:
    name = "fetch_page"
    description = "Fetch one web page by URL and return its readable main text."
    parameters = {
        "type": "object",
        "properties": {"url": {"type": "string", "description": "absolute http(s) URL"}},
        "required": ["url"],
    }
    def __init__(self, cfg): self._cfg = cfg
    def execute(self, args, cancel): raise NotImplementedError
