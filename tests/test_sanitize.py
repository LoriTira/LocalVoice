import pytest

from localvoice.textproc.sanitize import TextFilter, strip_speech_markup


def run_filter(deltas: list[str]) -> str:
    f = TextFilter()
    out = "".join(f.feed(d) for d in deltas)
    return out + f.finish()


def test_think_block_dropped_even_split_across_deltas():
    deltas = ["<th", "ink>secret ", "reasoning</thi", "nk>Hello", " there."]
    assert run_filter(deltas) == "Hello there."


def test_no_think_prefix_from_qwen_nothink_mode():
    assert run_filter(["<think>", "\n\n</think>", "\n\nHi!"]) == "\n\nHi!"


def test_unclosed_think_at_finish_yields_nothing():
    assert run_filter(["<think>still reasoning"]) == ""


def test_code_fence_replaced_with_spoken_marker():
    text = "Use this:\n```python\nprint(1)\n```\nDone."
    assert run_filter([text]) == "Use this:\nCode omitted.\nDone."


def test_plain_text_passes_through_unchanged():
    assert run_filter(["Just a normal ", "answer."]) == "Just a normal answer."


@pytest.mark.parametrize(
    "raw,clean",
    [
        ("**Bold** and *italic* text", "Bold and italic text"),
        ("# Heading\nBody", "Heading Body"),
        ("See [the docs](https://x.y) now", "See the docs now"),
        ("- item one\n- item two", "item one item two"),
        ("Inline `code` here", "Inline code here"),
        ("call my_variable_name now", "call my variable name now"),
        ("Nice \U0001f600 day ✨", "Nice day"),
        ("  spaced   out  ", "spaced out"),
    ],
)
def test_strip_speech_markup(raw, clean):
    assert strip_speech_markup(raw) == clean


def test_on_think_collects_across_split_deltas():
    got: list[str] = []
    f = TextFilter(on_think=got.append)
    out = "".join(f.feed(d) for d in ["<th", "ink>step one ", "and two</thi", "nk>Answer."])
    out += f.finish()
    assert out == "Answer."
    assert got == ["step one and two"]


def test_on_think_not_called_without_think_block():
    got: list[str] = []
    f = TextFilter(on_think=got.append)
    f.feed("plain text only")
    f.finish()
    assert got == []


def test_on_think_delivers_unclosed_think_at_finish():
    got: list[str] = []
    f = TextFilter(on_think=got.append)
    f.feed("<think>partial reasoning that never closes")
    assert f.finish() == ""
    assert got == ["partial reasoning that never closes"]
