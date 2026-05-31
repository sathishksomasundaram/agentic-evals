"""Unit tests for the generic harness (Rule 4: one positive, one negative).

Pure-logic tests only — no model loading — so they run instantly in CI.
"""

import json
from pathlib import Path

from agentic_evals.harness import buffer_for_ratio, grade, write_run


def test_correct_tool_call_grades_pass_and_buffer_ratio_is_exact() -> None:
    """Positive: a correct tool-call JSON grades PASS; ratio sizing is exact."""
    output = '{"tool": "web_search", "args": {"query": "weather in San Francisco"}}'
    assert grade(output) == "PASS"
    assert buffer_for_ratio(0.5, 1000) == 500


def test_fluent_non_tool_answer_is_not_pass_and_buffer_floor_holds() -> None:
    """Negative: a fluent answer with no tool call is not PASS; min-buffer floor holds."""
    assert grade("The weather in San Francisco is sunny today.") != "PASS"
    assert buffer_for_ratio(0.01, 100) == 128  # floor enforced over the tiny ratio


def test_write_run_archives_each_run_and_updates_latest(tmp_path: Path) -> None:
    """Positive: every run leaves an immutable archive; raw-results.json is latest."""
    a1, latest = write_run(tmp_path, {"value": 1})
    a2, _ = write_run(tmp_path, {"value": 2})

    # Both per-run archives survive (proof trail), with distinct ids.
    archives = sorted((tmp_path / "runs").glob("*.json"))
    assert len(archives) == 2
    assert a1 != a2
    assert json.loads(a1.read_text())["value"] == 1
    assert json.loads(a1.read_text())["run_id"] == a1.stem

    # raw-results.json reflects the most recent run.
    assert json.loads(latest.read_text())["value"] == 2


def test_write_run_injects_run_id_and_timestamp(tmp_path: Path) -> None:
    """Negative-ish: stored payload always carries run_id + created_utc metadata."""
    _, latest = write_run(tmp_path, {"value": 1})
    stored = json.loads(latest.read_text())
    assert "run_id" in stored and "created_utc" in stored
