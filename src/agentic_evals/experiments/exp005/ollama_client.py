"""Minimal stdlib Ollama client for exp-005 — no new dependencies.

Talks to the native ``/api/chat`` endpoint with the production router tier's
sampling options (temperature 0.1, num_ctx 1024). Returns the reply text and
wall-clock latency.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from typing import Any

BASE_URL = "http://localhost:11434"
# EXP005_NUM_PREDICT overrides the production 256-token output budget —
# used by the spike-0003 fairness variant for thinking-by-default models,
# which consume the whole production budget inside <think> blocks.
ROUTER_OPTIONS = {
    "temperature": 0.1,
    "num_ctx": int(os.getenv("EXP005_NUM_CTX", "1024")),
    "num_predict": int(os.getenv("EXP005_NUM_PREDICT", "256")),
}


def chat(model: str, system: str, user: str, *, timeout: float = 60.0) -> dict[str, Any]:
    """One chat completion. Returns {content, latency_ms, eval_count}.

    EXP005_THINK=false adds Ollama's ``think: false`` (>=0.30) to disable
    thinking on thinking-by-default models — the spike-0003 no-think variant.
    """
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": ROUTER_OPTIONS,
    }
    if os.getenv("EXP005_THINK", "").strip().lower() == "false":
        body["think"] = False
    payload = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(  # noqa: S310 - localhost only
        f"{BASE_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    start = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        body = json.loads(response.read().decode("utf-8"))
    latency_ms = (time.monotonic() - start) * 1000.0
    return {
        "content": body.get("message", {}).get("content", ""),
        "latency_ms": latency_ms,
        "eval_count": body.get("eval_count"),
    }


def warm(model: str) -> None:
    """One throwaway call so model-load time doesn't pollute case latencies."""
    chat(model, "Reply with the word ok.", "ok?", timeout=300.0)
