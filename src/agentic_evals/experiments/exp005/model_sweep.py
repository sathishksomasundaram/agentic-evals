"""Spike 0001 — model sweep: every candidate model on the routing golden set.

Each model gets the exact production router prompt and guarded parser
(keyword fallback on parse/vocab/consistency failure — production semantics).
Pure-keyword is scored alongside as the baseline configuration.

    uv run agentic-evals exp-005 model-sweep
    EXP005_MODEL=llama3.2:3b EXP005_RUNS=3 uv run agentic-evals exp-005 model-sweep

Config via env (the CLI dispatcher owns argv): EXP005_MODEL (single model
instead of the sweep; --model also works), EXP005_RUNS, EXP005_SKIP_KEYWORD.
"""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path
from typing import Any

from agentic_evals.experiments.exp005 import iris_router, ollama_client
from agentic_evals.harness.report import append_jsonl, write_run

DEFAULT_MODELS = [
    "llama3.2:3b",
    "qwen3.5:4b-q4_K_M",
    "granite4:latest",
    "qwen2.5:7b-instruct",
    "gemma2:9b",
]

EXP_DIR = (
    Path(__file__).resolve().parents[4]
    / "docs"
    / "experiments"
    / "exp-005-router-shootout"
)
OUT_DIR = EXP_DIR / "spikes" / "0001-model-sweep"
RESULTS_DIR = EXP_DIR / "results"


def load_cases() -> list[dict[str, Any]]:
    golden = json.loads((Path(__file__).parent / "golden_set.json").read_text())
    return list(golden["cases"])


def grade_config(
    label: str,
    cases: list[dict[str, Any]],
    route_fn: Any,
    run_index: int = 0,
) -> dict[str, Any]:
    """Route every case, return summary + per-case rows."""
    rows: list[dict[str, Any]] = []
    for case in cases:
        routed = route_fn(case["utterance"])
        ok = (
            routed["intent"] == case["expect_intent"]
            and routed["agent"] == case["expect_agent"]
        )
        rows.append(
            {
                "config": label,
                "run": run_index,
                "case_id": case["id"],
                "class": case.get("class", "clear"),
                "pass": ok,
                "expected": f"{case['expect_intent']}/{case['expect_agent']}",
                "actual": f"{routed['intent']}/{routed['agent']}",
                "source": routed.get("source", ""),
                "latency_ms": round(float(routed.get("latency_ms", 0.0)), 1),
            }
        )

    latencies = sorted(r["latency_ms"] for r in rows)
    by_class: dict[str, list[bool]] = {}
    for row in rows:
        by_class.setdefault(row["class"], []).append(row["pass"])
    fallbacks = sum(1 for r in rows if str(r["source"]).startswith("fallback-"))
    summary = {
        "config": label,
        "run": run_index,
        "cases": len(rows),
        "accuracy": round(sum(r["pass"] for r in rows) / len(rows), 4),
        "accuracy_by_class": {
            cls: round(sum(vals) / len(vals), 4) for cls, vals in sorted(by_class.items())
        },
        "fallback_rate": round(fallbacks / len(rows), 4),
        "latency_p50_ms": round(statistics.median(latencies), 1),
        "latency_p95_ms": round(latencies[int(len(latencies) * 0.95) - 1], 1),
        "latency_max_ms": round(latencies[-1], 1),
    }
    return {"summary": summary, "rows": rows}


def keyword_route(utterance: str) -> dict[str, Any]:
    result = iris_router.classify_keyword(utterance)
    result["latency_ms"] = 0.0
    return result


def make_llm_route(model: str) -> Any:
    def route(utterance: str) -> dict[str, Any]:
        reply = ollama_client.chat(model, iris_router.ROUTER_SYSTEM_PROMPT, utterance)
        routed = iris_router.route_llm_with_fallback(reply["content"], utterance)
        routed["latency_ms"] = reply["latency_ms"]
        routed["raw"] = reply["content"][:200]
        return routed

    return route


def main() -> None:
    single_model = os.getenv("EXP005_MODEL") or os.getenv("OMB_MODEL")
    runs = int(os.getenv("EXP005_RUNS", "1"))
    skip_keyword = os.getenv("EXP005_SKIP_KEYWORD", "").strip() == "1"
    # Distinguishes spike-0003 variants (no-think, big-budget) in the results.
    label_suffix = os.getenv("EXP005_LABEL_SUFFIX", "")

    cases = load_cases()
    models = [single_model] if single_model else DEFAULT_MODELS
    all_summaries: list[dict[str, Any]] = []

    if not skip_keyword:
        graded = grade_config("keyword", cases, keyword_route)
        all_summaries.append(graded["summary"])
        append_jsonl(RESULTS_DIR / "raw-results.jsonl", graded["rows"])
        print(json.dumps(graded["summary"], indent=2))

    for model in models:
        print(f"\n--- warming {model} ---")
        ollama_client.warm(model)
        for run_index in range(runs):
            graded = grade_config(
                f"llm:{model}{label_suffix}", cases, make_llm_route(model), run_index
            )
            all_summaries.append(graded["summary"])
            append_jsonl(RESULTS_DIR / "raw-results.jsonl", graded["rows"])
            print(json.dumps(graded["summary"], indent=2))

    write_run(OUT_DIR, {"spike": "0001-model-sweep", "summaries": all_summaries})
    print("\nLeaderboard (accuracy / p50 ms):")
    for summary in sorted(all_summaries, key=lambda s: -s["accuracy"]):
        print(
            f"  {summary['config']:28s} run={summary['run']} "
            f"{summary['accuracy']:.1%}  p50={summary['latency_p50_ms']}ms "
            f"fallback={summary['fallback_rate']:.0%}"
        )


if __name__ == "__main__":
    main()
