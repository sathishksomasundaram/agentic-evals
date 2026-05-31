"""Unit tests for exp-003 thinking-block handling (Rule 4: one positive, one negative).

Pure-logic, no model loading — runs instantly in CI.
"""

from agentic_evals.experiments.exp003.thinking import had_thinking, strip_thinking


def test_strip_returns_answer_after_thinking_block() -> None:
    """Positive: code after </think> is returned; draft fences inside think are dropped."""
    reply = (
        "<think>Let me draft:\n```python\nreturn 0  # wrong\n```\n</think>\n"
        "```python\ndef f():\n    return 1\n```"
    )
    answer = strip_thinking(reply)
    assert had_thinking(reply) is True
    assert "return 0" not in answer  # the draft inside <think> is gone
    assert "return 1" in answer


def test_strip_is_noop_without_thinking_block() -> None:
    """Negative: with no </think> marker the text is returned unchanged."""
    reply = "```python\ndef f():\n    return 1\n```"
    assert had_thinking(reply) is False
    assert strip_thinking(reply) == reply
