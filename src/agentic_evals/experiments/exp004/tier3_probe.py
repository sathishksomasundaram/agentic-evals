"""Spike 4A — Tier-3 MoE throughput/memory: Ollama vs in-process MLX.

The Tier-3 migration question reduces to: same MoE (Qwen3.6-35B-A3B,
4-bit), does MLX's published advantage (exp-002/003: 100+ tok/s, ~20 GB)
hold against Ollama serving the same architecture — and how does the
current dense incumbent (qwen3.6:27b) compare?

Ollama side: server eval stats (eval_count/eval_duration) + `ollama ps`
footprint. MLX side: in-process ``mlx_lm.stream_generate`` with native
generation stats + ``mx.get_peak_memory()`` — the exp-002/003 method, so
numbers are directly comparable with those reports.

    EXP004_T3=ollama:qwen3.6:35b-a3b uv run agentic-evals exp-004 tier3-probe
    EXP004_T3=mlx:mlx-community/Qwen3.6-35B-A3B-4bit uv run agentic-evals exp-004 tier3-probe
"""

from __future__ import annotations

import json
import os
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any

from agentic_evals.harness.report import write_run

EXP_DIR = (
    Path(__file__).resolve().parents[4]
    / "docs"
    / "experiments"
    / "exp-004-serving-stack-bakeoff"
)
OUT_DIR = EXP_DIR / "spikes" / "0004-tier3-moe"

PROMPT = (
    "Write a detailed 250-word essay about the benefits of running AI models "
    "locally on personal hardware, covering privacy, cost and reliability."
)
RUNS = 3
MAX_TOKENS = 256


def probe_ollama(model: str) -> dict[str, Any]:
    decode: list[float] = []
    walls: list[float] = []
    for _ in range(RUNS + 1):  # first run is warm-up (model load)
        start = time.monotonic()
        request = urllib.request.Request(  # noqa: S310 - localhost only
            "http://localhost:11434/api/generate",
            data=json.dumps(
                {
                    "model": model,
                    "prompt": PROMPT,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": MAX_TOKENS},
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=600) as response:  # noqa: S310
            body = json.loads(response.read().decode("utf-8"))
        wall = time.monotonic() - start
        if _ == 0:
            continue
        walls.append(wall)
        if body.get("eval_duration"):
            decode.append(body["eval_count"] / (body["eval_duration"] / 1e9))

    with urllib.request.urlopen("http://localhost:11434/api/ps", timeout=10) as resp:  # noqa: S310
        ps = json.loads(resp.read().decode("utf-8"))
    size_gb = next(
        (m.get("size", 0) / 1e9 for m in ps.get("models", []) if m["name"] == model), None
    )
    return {
        "stack": "ollama",
        "model": model,
        "decode_tps_mean": round(statistics.mean(decode), 1),
        "decode_tps_runs": [round(d, 1) for d in decode],
        "wall_s_mean": round(statistics.mean(walls), 2),
        "footprint_gb": round(size_gb, 1) if size_gb else None,
        "memory_method": "ollama ps (model+kv resident)",
    }


def probe_mlx(model_id: str) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(model_id)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}], add_generation_prompt=True
    )
    sampler = make_sampler(temp=0.1)
    decode: list[float] = []
    walls: list[float] = []
    mx.reset_peak_memory()
    for run_index in range(RUNS + 1):  # first run warms compile caches
        start = time.monotonic()
        last: Any = None
        for resp in stream_generate(
            model, tokenizer, prompt, max_tokens=MAX_TOKENS, sampler=sampler
        ):
            last = resp
        wall = time.monotonic() - start
        if run_index == 0:
            continue
        walls.append(wall)
        decode.append(float(last.generation_tps))
    return {
        "stack": "mlx",
        "model": model_id,
        "decode_tps_mean": round(statistics.mean(decode), 1),
        "decode_tps_runs": [round(d, 1) for d in decode],
        "wall_s_mean": round(statistics.mean(walls), 2),
        "footprint_gb": round(mx.get_peak_memory() / 1e9, 1),
        "memory_method": "mx.get_peak_memory()",
    }


def main() -> None:
    spec = os.getenv("EXP004_T3", "ollama:qwen3.6:35b-a3b")
    stack, _, model = spec.partition(":")
    result = probe_ollama(model) if stack == "ollama" else probe_mlx(model)
    write_run(OUT_DIR, {"spike": "0004-tier3-moe", "summary": result})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
