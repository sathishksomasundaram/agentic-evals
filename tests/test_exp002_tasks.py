"""Unit tests for the exp-002 coding battery (Rule 4: one positive, one negative).

Pure-logic: exercises code extraction + subprocess execution grading without
loading any model, so it runs instantly in CI.
"""

from agentic_evals.experiments.exp002.tasks import TASKS, extract_code, run_task

_PALINDROME = next(t for t in TASKS if t.name == "is_palindrome")


def test_correct_solution_passes_and_code_is_extracted_from_fence() -> None:
    """Positive: a correct fenced solution extracts cleanly and passes its asserts."""
    reply = (
        "Here you go:\n```python\n"
        "def is_palindrome(s: str) -> bool:\n"
        "    cleaned = [c.lower() for c in s if c.isalnum()]\n"
        "    return cleaned == cleaned[::-1]\n"
        "```\n"
    )
    assert "def is_palindrome" in extract_code(reply)
    passed, detail = run_task(_PALINDROME, reply)
    assert passed is True
    assert detail == "ok"


def test_wrong_solution_fails_grading() -> None:
    """Negative: a solution that ignores cleaning fails the asserts (not PASS)."""
    reply = "```python\ndef is_palindrome(s: str) -> bool:\n    return s == s[::-1]\n```"
    passed, _ = run_task(_PALINDROME, reply)
    assert passed is False
