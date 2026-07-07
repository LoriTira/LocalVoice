from localvoice.llm.mlx_lm_engine import common_prefix_len, is_channel_style


def test_channel_style_detected_from_template():
    assert is_channel_style("...{{ '<|channel>thought\\n<channel|>' }}...")
    assert not is_channel_style("...{% if enable_thinking %}<think>{% endif %}...")
    assert not is_channel_style(None)
    assert not is_channel_style("")


def test_equal_lists():
    assert common_prefix_len([1, 2, 3], [1, 2, 3]) == 3


def test_one_is_prefix_of_other():
    assert common_prefix_len([1, 2], [1, 2, 3, 4]) == 2
    assert common_prefix_len([1, 2, 3, 4], [1, 2]) == 2


def test_divergence_at_zero():
    assert common_prefix_len([9, 1, 2], [1, 2, 3]) == 0


def test_both_empty():
    assert common_prefix_len([], []) == 0


def count_chars(msgs):
    return sum(len(m["content"]) for m in msgs)


def test_fit_messages_noop_under_budget():
    from localvoice.llm.mlx_lm_engine import fit_messages

    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]
    assert fit_messages(msgs, 100, count_chars) == msgs


def test_fit_messages_drops_oldest_pair_first():
    from localvoice.llm.mlx_lm_engine import fit_messages

    msgs = [
        {"role": "system", "content": "ss"},
        {"role": "user", "content": "aaaa"},
        {"role": "assistant", "content": "bbbb"},
        {"role": "user", "content": "cccc"},
        {"role": "assistant", "content": "dddd"},
        {"role": "user", "content": "ee"},
    ]
    out = fit_messages(msgs, 12, count_chars)
    assert out == [
        {"role": "system", "content": "ss"},
        {"role": "user", "content": "cccc"},
        {"role": "assistant", "content": "dddd"},
        {"role": "user", "content": "ee"},
    ]


def test_fit_messages_never_drops_system_or_live_user():
    from localvoice.llm.mlx_lm_engine import fit_messages

    msgs = [
        {"role": "system", "content": "x" * 50},
        {"role": "user", "content": "y" * 50},
    ]
    out = fit_messages(msgs, 10, count_chars)  # over budget but nothing droppable
    assert out == msgs
