from localvoice.transcript import Transcript


def make() -> Transcript:
    t = Transcript(system_prompt="be brief")
    t.begin_turn("hello")
    t.add_clause("Hi there.")
    t.add_clause("How can I help?")
    return t


def test_messages_exclude_pending_assistant():
    t = make()
    assert t.messages() == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hello"},
    ]


def test_commit_joins_clauses():
    t = make()
    t.commit()
    assert t.history() == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Hi there. How can I help?"},
    ]


def test_truncate_commit_keeps_spoken_prefix_with_marker():
    t = make()
    t.truncate_commit({0})
    assert t.history()[-1] == {"role": "assistant", "content": "Hi there. ..."}


def test_truncate_commit_nothing_spoken():
    t = make()
    t.truncate_commit(set())
    assert t.history()[-1] == {"role": "assistant", "content": "..."}


def test_abort_pending_drops_turn():
    t = make()
    t.abort_pending()
    assert t.history() == []
    assert t.messages() == [{"role": "system", "content": "be brief"}]


def test_next_turn_builds_on_committed():
    t = make()
    t.commit()
    t.begin_turn("second")
    assert t.messages()[-1] == {"role": "user", "content": "second"}
    assert len(t.messages()) == 4  # system, user, assistant, user


def test_tags_are_sequential_per_response():
    t = Transcript("s")
    t.begin_turn("u")
    assert t.add_clause("a") == 0
    assert t.add_clause("b") == 1
    t.commit()
    t.begin_turn("u2")
    assert t.add_clause("c") == 0
