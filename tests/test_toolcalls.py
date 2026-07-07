import pytest

from localvoice.textproc.toolcalls import ToolCallParser, ToolFormatError, parse_gemma_args


def run(deltas):
    p = ToolCallParser()
    out = "".join(p.feed(d) for d in deltas) + p.finish()
    return out, p


def test_parse_args_strings_and_numbers():
    body = 'query:<|"|>boston, MA: weather<|"|>,max_results:5'
    assert parse_gemma_args(body) == {"query": "boston, MA: weather", "max_results": 5}


def test_parse_args_bool_float_empty():
    assert parse_gemma_args("flag:true,ratio:0.5") == {"flag": True, "ratio": 0.5}
    assert parse_gemma_args("") == {}


def test_parse_args_malformed_raises():
    with pytest.raises(ToolFormatError):
        parse_gemma_args("query:<|\"|>unterminated")


def test_plain_text_passthrough():
    out, p = run(["Hello ", "world."])
    assert out == "Hello world." and p.call is None


def test_call_captured_and_never_spoken():
    deltas = ["Let me check. ", "<|tool_call>call:web_search{query:<|\"|>rain<|\"|>}",
              "<tool_call|>", " trailing junk"]
    out, p = run(deltas)
    assert out == "Let me check. "          # payload and post-call text swallowed
    assert p.call is not None and p.call.name == "web_search"
    assert p.call.args == {"query": "rain"} and p.malformed is False


def test_call_split_across_arbitrary_boundaries():
    deltas = ["<|tool_", "call>call:fetch", "_page{url:<|\"|>https://a.b/c<|\"|>}<tool_", "call|>"]
    out, p = run(deltas)
    assert out == "" and p.call.name == "fetch_page"
    assert p.call.args == {"url": "https://a.b/c"}


def test_malformed_body_flags_not_raises():
    deltas = ["<|tool_call>call:web_search{query:<|\"|>oops", "<tool_call|>"]
    out, p = run(deltas)
    assert p.call is not None and p.malformed is True and p.call.args == {}
    assert out == ""


def test_unclosed_call_at_finish_is_malformed():
    out, p = run(["<|tool_call>call:web_search{query:<|\"|>x<|\"|>}"])
    assert out == "" and p.call is not None and p.malformed is True
