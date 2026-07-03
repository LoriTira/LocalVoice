from localvoice.llm.mlx_lm_engine import plan_prompt

SYS = {"role": "system", "content": "s"}
U1 = {"role": "user", "content": "one"}
A1 = {"role": "assistant", "content": "ans"}
U2 = {"role": "user", "content": "two"}


def test_first_turn_is_full():
    assert plan_prompt([], [SYS, U1]) == "full"


def test_appended_user_is_incremental():
    assert plan_prompt([SYS, U1, A1], [SYS, U1, A1, U2]) == "incremental"


def test_diverged_history_is_full():
    truncated = [SYS, U1, {"role": "assistant", "content": "ans ..."}]
    assert plan_prompt([SYS, U1, A1], truncated + [U2]) == "full"


def test_non_user_tail_is_full():
    assert plan_prompt([SYS, U1], [SYS, U1, A1]) == "full"
