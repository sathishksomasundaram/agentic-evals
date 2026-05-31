"""Spike 0001 — coding viability + headline throughput/memory.

Verifies the article's central LLM claim on real hardware: a 4-bit MoE LLM that
(a) runs in ~19-22 GB, (b) decodes at ~90-130 tok/s, and (c) is "competitive
enough for day-to-day coding". We measure prefill/decode throughput and peak
memory with `mlx-lm`'s own per-generation stats, and grade coding quality by
*executing* the generated code against asserts (see :mod:`.tasks`).

One model per process; pass it with ``--model`` (or set ``OMB_MODEL``):
    uv run agentic-evals exp-002 coding-viability \\
        --model mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit
"""

from __future__ import annotations

import os
import statistics
from pathlib import Path
from typing import Any

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

from agentic_evals.experiments.exp002.tasks import TASKS, run_task
from agentic_evals.harness.report import write_run

DEFAULT_MODEL = "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"
MAX_TOKENS = 1024

# The article's claimed bands (M5 Max, 36 GB).
ARTICLE_DECODE_TPS = (90.0, 130.0)
ARTICLE_PEAK_GB = (19.0, 22.0)

OUT_DIR = (
    Path(__file__).resolve().parents[4]
    / "docs"
    / "experiments"
    / "exp-002-mlx-moe-coding-viability"
    / "spikes"
    / "0001-coding-viability"
)


def _generate(model: Any, tokenizer: Any, prompt: str) -> dict[str, Any]:
    """Run one greedy generation, returning text + mlx-lm throughput/mem stats."""
    sampler = make_sampler(temp=0.0)
    text = ""
    last: Any = None
    for resp in stream_generate(model, tokenizer, prompt, max_tokens=MAX_TOKENS, sampler=sampler):
        text += resp.text
        last = resp
    return {
        "output": text,
        "prompt_tokens": int(last.prompt_tokens),
        "prompt_tps": round(float(last.prompt_tps), 2),
        "generation_tokens": int(last.generation_tokens),
        "generation_tps": round(float(last.generation_tps), 2),
        "peak_mem_gb": round(float(last.peak_memory), 3),
        "finish_reason": last.finish_reason,
    }


def _in_band(value: float, band: tuple[float, float]) -> str:
    lo, hi = band
    if value < lo:
        return "below"
    if value > hi:
        return "above"
    return "in-band"


def main() -> None:
    model_id = os.environ.get("OMB_MODEL", "").strip() or DEFAULT_MODEL
    print(f"==> exp-002 spike 0001 — coding viability for {model_id}")
    print("    (downloads ~17 GB on first run)")

    model, tokenizer = load(model_id)  # type: ignore[misc]

    rows: list[dict[str, Any]] = []
    for task in TASKS:
        messages = [{"role": "user", "content": task.prompt}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        stats = _generate(model, tokenizer, prompt)
        passed, detail = run_task(task, stats["output"])
        row = {
            "task": task.name,
            "difficulty": task.difficulty,
            "verdict": "PASS" if passed else "FAIL",
            "detail": detail,
            **{k: v for k, v in stats.items() if k != "output"},
        }
        rows.append(row)
        print(
            f"    {task.name:18s} [{task.difficulty:4s}] {row['verdict']:4s} "
            f"({detail:24.24s}) "
            f"decode={stats['generation_tps']:6.1f} tok/s  "
            f"peak={stats['peak_mem_gb']:5.1f} GB"
        )

    decode = statistics.median(r["generation_tps"] for r in rows)
    prefill = statistics.median(r["prompt_tps"] for r in rows)
    peak = max(r["peak_mem_gb"] for r in rows)
    passes = sum(r["verdict"] == "PASS" for r in rows)

    def _pass_rate(difficulty: str) -> str:
        sub = [r for r in rows if r["difficulty"] == difficulty]
        return f"{sum(r['verdict'] == 'PASS' for r in sub)}/{len(sub)}"

    summary = {
        "model_id": model_id,
        "max_tokens": MAX_TOKENS,
        "n_tasks": len(rows),
        "n_pass": passes,
        "pass_easy": _pass_rate("easy"),
        "pass_hard": _pass_rate("hard"),
        "median_decode_tps": round(decode, 2),
        "median_prefill_tps": round(prefill, 2),
        "peak_mem_gb": round(peak, 3),
        "decode_vs_article": _in_band(decode, ARTICLE_DECODE_TPS),
        "peak_mem_vs_article": _in_band(peak, ARTICLE_PEAK_GB),
        "rows": rows,
    }

    print("\n==> Summary vs article claims (M5 Max → measured on this Mac)")
    print(
        f"    decode  : {decode:6.1f} tok/s  "
        f"(article {ARTICLE_DECODE_TPS[0]:.0f}-{ARTICLE_DECODE_TPS[1]:.0f}) "
        f"=> {summary['decode_vs_article']}"
    )
    print(f"    prefill : {prefill:6.1f} tok/s  (article ~1851)")
    print(
        f"    peak mem: {peak:6.1f} GB    "
        f"(article {ARTICLE_PEAK_GB[0]:.0f}-{ARTICLE_PEAK_GB[1]:.0f}) "
        f"=> {summary['peak_mem_vs_article']}"
    )
    print(
        f"    coding  : {passes}/{len(rows)} PASS  "
        f"(easy {summary['pass_easy']}, hard {summary['pass_hard']})"
    )

    archived, latest = write_run(OUT_DIR, summary)
    print(f"\n==> Wrote {latest}")
    print(f"    archived run: {archived}")


if __name__ == "__main__":
    main()
