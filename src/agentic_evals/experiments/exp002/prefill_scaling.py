"""Spike 0002 — long-context prefill throughput.

Spike 0001 measured prefill on 56-88-token prompts, where prefill tok/s is noisy
and not comparable to the article's "~1851 tokens/sec prefill" figure (almost
certainly a long-context number). This spike sweeps prompt length and records
prefill (prompt) tok/s at each, so the ~1851 claim can be judged fairly.

Generation is capped at a few tokens so the prompt-eval phase dominates the
measurement. Greedy / deterministic.

    uv run agentic-evals exp-002 prefill-scaling
    uv run agentic-evals exp-002 prefill-scaling --model <other-hf-id>
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

from agentic_evals.harness.report import write_run

DEFAULT_MODEL = "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"
GEN_TOKENS = 8  # keep generation tiny so prefill dominates the timing
TARGET_PROMPT_TOKENS: tuple[int, ...] = (512, 1024, 2048, 4096, 8192)
ARTICLE_PREFILL_TPS = 1851.0

# A realistic ~code-shaped filler unit, repeated to reach a target token count.
_FILLER_UNIT = (
    "def process_batch(records: list[dict], *, retries: int = 3) -> list[dict]:\n"
    "    results = []\n"
    "    for record in records:\n"
    "        for attempt in range(retries):\n"
    "            try:\n"
    "                results.append(transform(record))\n"
    "                break\n"
    "            except TransientError:\n"
    "                continue\n"
    "    return results\n\n"
)

OUT_DIR = (
    Path(__file__).resolve().parents[4]
    / "docs"
    / "experiments"
    / "exp-002-mlx-moe-coding-viability"
    / "spikes"
    / "0002-prefill-scaling"
)


def _build_prompt(tokenizer: Any, target_tokens: int) -> tuple[str, int]:
    """Build a chat prompt of approximately `target_tokens` tokens.

    Returns (prompt_text, actual_token_count). We grow a code-shaped filler until
    the templated prompt reaches the target, then report the true token count.
    """
    instruction = "\n\nIn one short sentence, summarize what the code above does."
    reps = 1
    while True:
        body = _FILLER_UNIT * reps + instruction
        messages = [{"role": "user", "content": body}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        n = len(tokenizer.encode(prompt))
        if n >= target_tokens:
            return prompt, n
        # Estimate how many more reps are needed; always advance by at least one.
        per_rep = max(1, n // reps)
        reps += max(1, (target_tokens - n) // per_rep)


def main() -> None:
    model_id = os.environ.get("OMB_MODEL", "").strip() or DEFAULT_MODEL
    print(f"==> exp-002 spike 0002 — prefill scaling for {model_id}")

    model, tokenizer = load(model_id)  # type: ignore[misc]
    sampler = make_sampler(temp=0.0)

    rows: list[dict[str, Any]] = []
    for target in TARGET_PROMPT_TOKENS:
        prompt, _ = _build_prompt(tokenizer, target)
        last: Any = None
        for resp in stream_generate(
            model, tokenizer, prompt, max_tokens=GEN_TOKENS, sampler=sampler
        ):
            last = resp
        row = {
            "target_tokens": target,
            "prompt_tokens": int(last.prompt_tokens),
            "prompt_tps": round(float(last.prompt_tps), 2),
            "generation_tps": round(float(last.generation_tps), 2),
            "peak_mem_gb": round(float(last.peak_memory), 3),
        }
        rows.append(row)
        print(
            f"    prompt~{target:5d} (actual {row['prompt_tokens']:5d} tok)  "
            f"prefill={row['prompt_tps']:8.1f} tok/s  peak={row['peak_mem_gb']:5.1f} GB"
        )

    best_prefill = max(r["prompt_tps"] for r in rows)
    summary = {
        "model_id": model_id,
        "gen_tokens": GEN_TOKENS,
        "article_prefill_tps": ARTICLE_PREFILL_TPS,
        "best_prefill_tps": best_prefill,
        "fraction_of_article": round(best_prefill / ARTICLE_PREFILL_TPS, 3),
        "rows": rows,
    }

    print("\n==> Prefill vs article claim")
    print(
        f"    best measured prefill: {best_prefill:8.1f} tok/s  "
        f"(article ~{ARTICLE_PREFILL_TPS:.0f}) "
        f"=> {summary['fraction_of_article']:.0%} of claim"
    )

    archived, latest = write_run(OUT_DIR, summary)
    print(f"\n==> Wrote {latest}")
    print(f"    archived run: {archived}")


if __name__ == "__main__":
    main()
