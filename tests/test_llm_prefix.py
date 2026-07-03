from localvoice.llm.mlx_lm_engine import common_prefix_len


def test_equal_lists():
    assert common_prefix_len([1, 2, 3], [1, 2, 3]) == 3


def test_one_is_prefix_of_other():
    assert common_prefix_len([1, 2], [1, 2, 3, 4]) == 2
    assert common_prefix_len([1, 2, 3, 4], [1, 2]) == 2


def test_divergence_at_zero():
    assert common_prefix_len([9, 1, 2], [1, 2, 3]) == 0


def test_both_empty():
    assert common_prefix_len([], []) == 0
