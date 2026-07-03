from localvoice.textproc.chunker import ClauseChunker


def feed_all(c: ClauseChunker, text: str, size: int = 3) -> list[str]:
    out: list[str] = []
    for i in range(0, len(text), size):
        out.extend(c.feed(text[i : i + size]))
    tail = c.flush()
    if tail:
        out.append(tail)
    return out


def test_short_sentence_emits_fast():
    c = ClauseChunker()
    out: list[str] = []
    for delta in ["Sure", ", I can", " help. ", "Second"]:
        out.extend(c.feed(delta))
    assert out == ["Sure, I can help."]


def test_soft_boundary_waits_for_min_chars():
    c = ClauseChunker(min_chars=60)
    text = "one, two, three, four, five, six, seven, eight, nine, ten, eleven, twelve"
    chunks = feed_all(c, text)
    assert len(chunks) >= 2
    assert all(len(ch) >= 15 for ch in chunks[:-1])
    assert " ".join(" ".join(chunks).split()) == " ".join(text.split())


def test_decimal_not_split():
    c = ClauseChunker()
    chunks = feed_all(c, "Pi is 3.14159 which is handy. And that is that.")
    assert chunks[0] == "Pi is 3.14159 which is handy."


def test_hard_flush_at_max_chars_splits_on_space():
    c = ClauseChunker(min_chars=60, max_chars=80)
    chunks = feed_all(c, "word " * 40)
    assert all(len(ch) <= 80 for ch in chunks)
    assert all(not ch.startswith(" ") and not ch.endswith(" ") for ch in chunks)
    assert all("word" == w for ch in chunks for w in ch.split())


def test_flush_returns_remainder_and_empty_is_none():
    c = ClauseChunker()
    assert c.feed("tiny tail") == []
    assert c.flush() == "tiny tail"
    assert c.flush() is None


def test_newline_is_sentence_ender():
    c = ClauseChunker()
    out = c.feed("First line of the answer\nand more")
    assert out == ["First line of the answer"]
