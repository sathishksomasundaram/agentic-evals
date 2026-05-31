"""Spike 0001 — the article's real model on the exp-002 coding battery.

Re-runs exp-002's headline check (decode tok/s, peak memory, executable coding
pass-rate) on the article's *actual* model `mlx-community/Qwen3.6-35B-A3B-4bit`,
so the result is directly comparable to the exp-002 substitute.

Two differences from exp-002 spike 0001:
  1. **Thinking-by-default** — the model reasons inside `<think>…</think>` before
     answering, so we give it a large token budget and grade only the post-think
     answer (see :mod:`.thinking`).
  2. **Bigger model** — ~18-20 GB 4-bit; the first run downloads it. We record peak
     memory to judge the article's "19-22 GB" claim on the real model.

Greedy / deterministic. n=1 (headline spike).

    uv run agentic-evals exp-003 coding-viability
"""

from __future__ import annotations

import os
import statistics
from pathlib import Path
from typing import Any

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

from agentic_evals.experiments.exp002.tasks import TASKS, run_task
from agentic_evals.experiments.exp003.thinking import had_thinking, strip_thinking
from agentic_evals.harness.report import write_run

DEFAULT_MODEL = "mlx-community/Qwen3.6-35B-A3B-4bit"
MAX_TOKENS = 3072  # generous: thinking + the actual answer
ARTICLE_DECODE_TPS = (90.0, 130.0)
ARTICLE_PEAK_GB = (19.0, 22.0)

OUT_DIR = (
    Path(__file__).resolve().parents[4]
    / "docs"
    / "experiments"
    / "exp-003-qwen36-35b-coding-viability"
    / "spikes"
    / "0001-coding-viability"
)


def _in_band(value: float, band: tuple[float, float]) -> bool:
    return band[0] <= value <= band[1]


def _generate(model: Any, tokenizer: Any, prompt_text: str, *, sampler: Any) -> dict[str, Any]:
    """Greedy-generate a full reply, returning text + throughput/memory stats."""
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    text = ""
    last: Any = None
    for resp in stream_generate(model, tokenizer, prompt, max_tokens=MAX_TOKENS, sampler=sampler):
        text += resp.text
        last = resp
    return {
        "text": text,
        "prompt_tps": round(float(last.prompt_tps), 2),
        "generation_tps": round(float(last.generation_tps), 2),
        "generation_tokens": int(last.generation_tokens),
        "peak_mem_gb": round(float(last.peak_memory), 3),
        "finish_reason": last.finish_reason,
    }


def _pass_rate(rows: list[dict[str, Any]], difficulty: str) -> str:
    subset = [r for r in rows if r["difficulty"] == difficulty]
    n_pass = sum(r["verdict"] == "PASS" for r in subset)
    return f"{n_pass}/{len(subset)}"


def main() -> None:
    model_id = os.environ.get("OMB_MODEL", "").strip() or DEFAULT_MODEL
    print(f"==> exp-003 spike 0001 — real-model coding viability for {model_id}")

    model, tokenizer = load(model_id)  # type: ignore[misc]
    greedy = make_sampler(temp=0.0)

    rows: list[dict[str, Any]] = []
    for task in TASKS:
        gen = _generate(model, tokenizer, task.prompt, sampler=greedy)
        answer = strip_thinking(gen["text"])
        passed, detail = run_task(task, answer)
        row = {
            "task": task.name,
            "difficulty": task.difficulty,
            "verdict": "PASS" if passed else "FAIL",
            "detail": detail,
            "thinking": had_thinking(gen["text"]),
            "generation_tps": gen["generation_tps"],
            "generation_tokens": gen["generation_tokens"],
            "peak_mem_gb": gen["peak_mem_gb"],
            "finish_reason": gen["finish_reason"],
        }
        rows.append(row)
        print(
            f"    {task.name:18s} {task.difficulty:4s} {row['verdict']:4s} "
            f"decode={row['generation_tps']:6.1f} tok/s  gen={row['generation_tokens']:4d} "
            f"think={row['thinking']!s:5s} peak={row['peak_mem_gb']:5.1f} GB  ({detail})"
        )

    decodes = [r["generation_tps"] for r in rows]
    median_decode = round(statistics.median(decodes), 2)
    peak = max(r["peak_mem_gb"] for r in rows)
    n_pass = sum(r["verdict"] == "PASS" for r in rows)

    summary = {
        "model_id": model_id,
        "max_tokens": MAX_TOKENS,
        "decoding": "greedy",
        "article_decode_tps": list(ARTICLE_DECODE_TPS),
        "article_peak_gb": list(ARTICLE_PEAK_GB),
        "median_decode_tps": median_decode,
        "decode_in_band": _in_band(median_decode, ARTICLE_DECODE_TPS),
        "peak_mem_gb": peak,
        "peak_in_band": _in_band(peak, ARTICLE_PEAK_GB),
        "n_pass": n_pass,
        "n_tasks": len(rows),
        "pass_easy": _pass_rate(rows, "easy"),
        "pass_hard": _pass_rate(rows, "hard"),
        "rows": rows,
    }

    print(
        f"\n==> coding {n_pass}/{len(rows)} (easy {summary['pass_easy']}, "
        f"hard {summary['pass_hard']})  median decode {median_decode} tok/s "
        f"(in-band={summary['decode_in_band']})  peak {peak} GB "
        f"(in-band={summary['peak_in_band']})"
    )

    archived, latest = write_run(OUT_DIR, summary)
    print(f"\n==> Wrote {latest}")
    print(f"    archived run: {archived}")


if __name__ == "__main__":
    main()
