"""Unit tests for the generic harness (Rule 4: one positive, one negative).

Pure-logic tests only — no model loading — so they run instantly in CI.
"""

from agentic_evals.harness import buffer_for_ratio, grade


def test_correct_tool_call_grades_pass_and_buffer_ratio_is_exact() -> None:
    """Positive: a correct tool-call JSON grades PASS; ratio sizing is exact."""
    output = '{"tool": "web_search", "args": {"query": "weather in San Francisco"}}'
    assert grade(output) == "PASS"
    assert buffer_for_ratio(0.5, 1000) == 500


def test_fluent_non_tool_answer_is_not_pass_and_buffer_floor_holds() -> None:
    """Negative: a fluent answer with no tool call is not PASS; min-buffer floor holds."""
    assert grade("The weather in San Francisco is sunny today.") != "PASS"
    assert buffer_for_ratio(0.01, 100) == 128  # floor enforced over the tiny ratio
