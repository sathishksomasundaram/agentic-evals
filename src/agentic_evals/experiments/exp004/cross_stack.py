"""Spike 2 — same model, three serving stacks: Ollama vs llama.cpp vs MLX.

Two workloads per (stack, model):

- **Routing** — the 77-case golden set, unconstrained, with the exp-005
  guarded parser (accuracy, p50, fallback). Sanity expectation: accuracy
  is a model property and should match across stacks; latency is a stack
  property and should differ.
- **Throughput probe** — 3 fixed ~256-token generations; decode tok/s from
  server-reported stats where available (Ollama eval_count/eval_duration,
  llama.cpp timings), wall-clock otherwise (MLX).

Stack endpoints via env:
    EXP004_STACK   = ollama | llamacpp | mlx
    EXP004_MODEL   = model name (Ollama name, or ignored for single-model servers)
    EXP004_BASE_URL= server base (default: ollama :11434, others :8088/:8090)
    EXP004_LABEL   = results label (default "<stack>:<model>")

    uv run agentic-evals exp-004 cross-stack
"""

from __future__ import annotations

import json
import os
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any

from agentic_evals.experiments.exp005.iris_router import (
    ROUTER_SYSTEM_PROMPT,
    route_llm_with_fallback,
)
from agentic_evals.experiments.exp005.model_sweep import load_cases
from agentic_evals.harness.report import append_jsonl, write_run

EXP_DIR = (
    Path(__file__).resolve().parents[4]
    / "docs"
    / "experiments"
    / "exp-004-serving-stack-bakeoff"
)
OUT_DIR = EXP_DIR / "spikes" / "0002-cross-stack"
RESULTS_DIR = EXP_DIR / "results"

THROUGHPUT_PROMPT = (
    "Write a detailed 250-word essay about the benefits of running AI models "
    "locally on personal hardware, covering privacy, cost and reliability."
)
DEFAULT_URLS = {
    "ollama": "http://localhost:11434",
    "llamacpp": "http://localhost:8088",
    "mlx": "http://localhost:8090",
}


def _post(url: str, body: dict[str, Any], *, timeout: float = 300.0) -> dict[str, Any]:
    request = urllib.request.Request(  # noqa: S310 - localhost only
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def chat_ollama(base: str, model: str, system: str, user: str, max_tokens: int) -> dict[str, Any]:
    start = time.monotonic()
    body = _post(
        f"{base}/api/chat",
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": 0.1, "num_ctx": 1024, "num_predict": max_tokens},
        },
    )
    latency_ms = (time.monotonic() - start) * 1000.0
    eval_count = body.get("eval_count") or 0
    eval_ns = body.get("eval_duration") or 0
    return {
        "content": body.get("message", {}).get("content", ""),
        "latency_ms": latency_ms,
        "decode_tps": (eval_count / (eval_ns / 1e9)) if eval_ns else None,
        "completion_tokens": eval_count,
    }


def chat_openai(base: str, model: str, system: str, user: str, max_tokens: int) -> dict[str, Any]:
    start = time.monotonic()
    body = _post(
        f"{base}/v1/chat/completions",
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.1,
            "max_tokens": max_tokens,
            "stream": False,
        },
    )
    latency_ms = (time.monotonic() - start) * 1000.0
    completion_tokens = body.get("usage", {}).get("completion_tokens") or 0
    timings = body.get("timings") or {}  # llama.cpp extension
    decode_tps = timings.get("predicted_per_second")
    if decode_tps is None and completion_tokens:
        decode_tps = completion_tokens / (latency_ms / 1000.0)  # wall-clock floor
    return {
        "content": body["choices"][0]["message"]["content"],
        "latency_ms": latency_ms,
        "decode_tps": decode_tps,
        "completion_tokens": completion_tokens,
    }


def main() -> None:
    stack = os.getenv("EXP004_STACK", "ollama").strip().lower()
    model = os.getenv("EXP004_MODEL", "qwen2.5:7b-instruct")
    base = os.getenv("EXP004_BASE_URL", DEFAULT_URLS[stack])
    label = os.getenv("EXP004_LABEL", f"{stack}:{model}")
    chat = chat_ollama if stack == "ollama" else chat_openai

    # Warm-up (model load + prompt-prefix cache).
    chat(base, model, ROUTER_SYSTEM_PROMPT, "ok?", 16)

    rows: list[dict[str, Any]] = []
    for case in load_cases():
        reply = chat(base, model, ROUTER_SYSTEM_PROMPT, case["utterance"], 256)
        routed = route_llm_with_fallback(reply["content"], case["utterance"])
        rows.append(
            {
                "config": label,
                "run": 0,
                "workload": "routing",
                "case_id": case["id"],
                "class": case.get("class", "clear"),
                "pass": routed["intent"] == case["expect_intent"]
                and routed["agent"] == case["expect_agent"],
                "expected": f"{case['expect_intent']}/{case['expect_agent']}",
                "actual": f"{routed['intent']}/{routed['agent']}",
                "source": routed.get("source", ""),
                "latency_ms": round(reply["latency_ms"], 1),
            }
        )

    throughput: list[float] = []
    for _ in range(3):
        reply = chat(base, model, "You are a helpful writer.", THROUGHPUT_PROMPT, 256)
        if reply["decode_tps"]:
            throughput.append(float(reply["decode_tps"]))

    latencies = sorted(r["latency_ms"] for r in rows)
    fallbacks = sum(1 for r in rows if str(r["source"]).startswith("fallback-"))
    summary = {
        "config": label,
        "stack": stack,
        "model": model,
        "routing_accuracy": round(sum(r["pass"] for r in rows) / len(rows), 4),
        "routing_p50_ms": round(statistics.median(latencies), 1),
        "routing_p95_ms": round(latencies[int(len(latencies) * 0.95) - 1], 1),
        "fallback_rate": round(fallbacks / len(rows), 4),
        "decode_tps_mean": round(statistics.mean(throughput), 1) if throughput else None,
        "decode_tps_runs": [round(t, 1) for t in throughput],
    }
    append_jsonl(RESULTS_DIR / "raw-results.jsonl", rows)
    write_run(OUT_DIR, {"spike": "0002-cross-stack", "summary": summary})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
