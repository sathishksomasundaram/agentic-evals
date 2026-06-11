"""Spike 0002 — keyword-first hybrid: LLM only on keyword-fallback cases.

Production ``KeywordFirstClassifier`` semantics: when a keyword rule fires
(confidence ≥ 0.8) its answer is final; only no-rule fallback cases go to the
LLM (which itself keeps the guarded parser + keyword fallback). On this golden
set the keyword stage answers 73/77 cases, so the hybrid can only differ from
pure-keyword on the remaining 4 — this spike measures whether the LLM lifts
or hurts exactly those.

    uv run agentic-evals exp-005 hybrid-sweep
    EXP005_MODEL=qwen2.5:7b-instruct EXP005_RUNS=3 uv run agentic-evals exp-005 hybrid-sweep
"""

from __future__ import annotations

import json
import os
from typing import Any

from agentic_evals.experiments.exp005 import iris_router, ollama_client
from agentic_evals.experiments.exp005.model_sweep import (
    DEFAULT_MODELS,
    EXP_DIR,
    RESULTS_DIR,
    grade_config,
    load_cases,
)
from agentic_evals.harness.report import append_jsonl, write_run

OUT_DIR = EXP_DIR / "spikes" / "0002-hybrid-sweep"
KEYWORD_THRESHOLD = 0.8


def make_hybrid_route(model: str) -> Any:
    def route(utterance: str) -> dict[str, Any]:
        keyword = iris_router.classify_keyword(utterance)
        if float(str(keyword["confidence"])) >= KEYWORD_THRESHOLD:
            keyword["latency_ms"] = 0.0
            return keyword
        reply = ollama_client.chat(model, iris_router.ROUTER_SYSTEM_PROMPT, utterance)
        routed = iris_router.route_llm_with_fallback(reply["content"], utterance)
        routed["latency_ms"] = reply["latency_ms"]
        routed["source"] = f"hybrid-{routed['source']}"
        return routed

    return route


def main() -> None:
    single_model = os.getenv("EXP005_MODEL") or os.getenv("OMB_MODEL")
    runs = int(os.getenv("EXP005_RUNS", "1"))

    cases = load_cases()
    models = [single_model] if single_model else DEFAULT_MODELS
    summaries: list[dict[str, Any]] = []

    for model in models:
        print(f"\n--- warming {model} ---")
        ollama_client.warm(model)
        for run_index in range(runs):
            graded = grade_config(
                f"hybrid:{model}", cases, make_hybrid_route(model), run_index
            )
            summaries.append(graded["summary"])
            append_jsonl(RESULTS_DIR / "raw-results.jsonl", graded["rows"])
            print(json.dumps(graded["summary"], indent=2))

    write_run(OUT_DIR, {"spike": "0002-hybrid-sweep", "summaries": summaries})
    print("\nHybrid leaderboard (accuracy / p50 ms):")
    for summary in sorted(summaries, key=lambda s: -s["accuracy"]):
        print(
            f"  {summary['config']:32s} run={summary['run']} "
            f"{summary['accuracy']:.1%}  p50={summary['latency_p50_ms']}ms"
        )


if __name__ == "__main__":
    main()
