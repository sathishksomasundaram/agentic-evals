"""Pure-logic tests for the exp-005 vendored router (no models)."""

from __future__ import annotations

import json
from pathlib import Path

from agentic_evals.experiments.exp005.iris_router import (
    classify_keyword,
    parse_llm_route,
    route_llm_with_fallback,
)

VALID_REPLY = json.dumps(
    {"intent": "calendar", "agent_type": "calendar", "is_multi_step": False, "confidence": 0.95}
)


def test_parse_accepts_valid_route() -> None:
    parsed = parse_llm_route(VALID_REPLY)
    assert parsed == {"intent": "calendar", "agent": "calendar", "source": "llm"}


def test_parse_strips_think_blocks() -> None:
    raw = f"<think>let me reason about this...</think>\n{VALID_REPLY}"
    assert parse_llm_route(raw) is not None


def test_parse_rejects_unterminated_think_block() -> None:
    raw = "<think>thinking forever, budget exhausted, no json"
    assert parse_llm_route(raw) is None


def test_parse_rejects_intent_agent_mismatch() -> None:
    raw = json.dumps({"intent": "code_exec", "agent_type": "rag"})
    assert parse_llm_route(raw) is None


def test_parse_rejects_unknown_vocab() -> None:
    raw = json.dumps({"intent": "shopping", "agent_type": "system"})
    assert parse_llm_route(raw) is None


def test_fallback_uses_keyword_classifier() -> None:
    routed = route_llm_with_fallback("not json at all", "check my inbox")
    assert routed["intent"] == "communication"
    assert routed["source"] == "fallback-keyword"


def test_keyword_matches_phase2_baseline_behavior() -> None:
    # The Phase 1 live misroute, frozen as adv-01.
    routed = classify_keyword("Please reply with just the single word: hello")
    assert routed["intent"] == "communication"  # the known trap, by design
    # The cal-05 fix is part of the snapshot.
    routed = classify_keyword("set up a recurring standup every weekday at 9")
    assert routed["intent"] == "calendar"


def test_golden_set_is_complete() -> None:
    golden = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "src"
            / "agentic_evals"
            / "experiments"
            / "exp005"
            / "golden_set.json"
        ).read_text()
    )
    cases = golden["cases"]
    assert len(cases) == 77
    assert all({"id", "utterance", "expect_intent", "expect_agent"} <= set(c) for c in cases)
