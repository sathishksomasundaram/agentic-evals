"""Spike 0002 — re-grade exp-003's coding battery with a *fair* token budget.

Spike 0001 graded the real model (`mlx-community/Qwen3.6-35B-A3B-4bit`) at a flat
`max_tokens=3072` and 3/8 tasks hit that cap (`finish_reason=length`): the answer was
truncated mid-code, or the model was still *inside* `<think>` when the budget ran out.
That makes the 6/8 a budget artifact, not a coding score.

This spike fixes the mechanism, not the model. Two changes:

  1. **Split budgets.** A generous *thinking* budget plus a *separate* answer budget,
     so a long reasoning trace can no longer starve the answer. We watch the stream for
     ``</think>`` and only then start counting answer tokens — capping each phase
     independently (total cap = ``THINK_BUDGET + ANSWER_BUDGET``).
  2. **Honest finish reason.** We record *why* each generation ended:
     ``stop`` (natural EOS — a clean result), ``answer_cap`` (answer ran long — a real
     failure to be concise, not a confound), or ``think_cap`` (still thinking at the
     total cap — the model never produced an answer).

If the article's "good for day-to-day coding" claim holds for the real model, the
battery should now mostly finish at ``stop`` and pass on its merits.

Greedy / deterministic. n=1 (headline re-grade).

    uv run agentic-evals exp-003 coding-budget
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
THINK_BUDGET = 6144  # room for a long reasoning trace
ANSWER_BUDGET = 2048  # room for the answer *after* </think>
MAX_TOKENS = THINK_BUDGET + ANSWER_BUDGET  # hard ceiling handed to the sampler
ARTICLE_DECODE_TPS = (90.0, 130.0)
ARTICLE_PEAK_GB = (19.0, 22.0)

OUT_DIR = (
    Path(__file__).resolve().parents[4]
    / "docs"
    / "experiments"
    / "exp-003-qwen36-35b-coding-viability"
    / "spikes"
    / "0002-token-budget"
)


def _in_band(value: float, band: tuple[float, float]) -> bool:
    return band[0] <= value <= band[1]


def _generate(model: Any, tokenizer: Any, prompt_text: str, *, sampler: Any) -> dict[str, Any]:
    """Greedy-generate with split thinking/answer budgets and an honest finish reason."""
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    text = ""
    last: Any = None
    in_answer = False
    think_tokens = 0
    answer_tokens = 0
    answer_capped = False
    for step, resp in enumerate(
        stream_generate(model, tokenizer, prompt, max_tokens=MAX_TOKENS, sampler=sampler), start=1
    ):
        text += resp.text
        last = resp
        if not in_answer:
            if had_thinking(text):
                in_answer = True
                think_tokens = step
        else:
            answer_tokens += 1
            if answer_tokens >= ANSWER_BUDGET:
                answer_capped = True
                break

    if answer_capped:
        finish = "answer_cap"
    elif not in_answer:
        # never emitted </think>; if the sampler also hit MAX_TOKENS it's a think cap
        finish = "think_cap" if last.finish_reason == "length" else last.finish_reason
        finish = finish or "think_cap"
    else:
        finish = last.finish_reason  # natural completion after the answer ("stop")

    return {
        "text": text,
        "prompt_tps": round(float(last.prompt_tps), 2),
        "generation_tps": round(float(last.generation_tps), 2),
        "generation_tokens": int(last.generation_tokens),
        "think_tokens": think_tokens,
        "answer_tokens": answer_tokens,
        "peak_mem_gb": round(float(last.peak_memory), 3),
        "finish_reason": finish,
    }


def _pass_rate(rows: list[dict[str, Any]], difficulty: str) -> str:
    subset = [r for r in rows if r["difficulty"] == difficulty]
    n_pass = sum(r["verdict"] == "PASS" for r in subset)
    return f"{n_pass}/{len(subset)}"


def main() -> None:
    model_id = os.environ.get("OMB_MODEL", "").strip() or DEFAULT_MODEL
    print(f"==> exp-003 spike 0002 — fair-budget coding re-grade for {model_id}")
    print(f"    think budget={THINK_BUDGET}  answer budget={ANSWER_BUDGET}  (greedy)")

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
            "think_tokens": gen["think_tokens"],
            "answer_tokens": gen["answer_tokens"],
            "peak_mem_gb": gen["peak_mem_gb"],
            "finish_reason": gen["finish_reason"],
        }
        rows.append(row)
        print(
            f"    {task.name:18s} {task.difficulty:4s} {row['verdict']:4s} "
            f"decode={row['generation_tps']:6.1f} tok/s  "
            f"think={row['think_tokens']:4d} ans={row['answer_tokens']:4d}  "
            f"finish={row['finish_reason']:9s} peak={row['peak_mem_gb']:5.1f} GB  ({detail})"
        )

    decodes = [r["generation_tps"] for r in rows]
    median_decode = round(statistics.median(decodes), 2)
    peak = max(r["peak_mem_gb"] for r in rows)
    n_pass = sum(r["verdict"] == "PASS" for r in rows)
    n_clean_stop = sum(r["finish_reason"] == "stop" for r in rows)
    n_truncated = sum(r["finish_reason"] in ("answer_cap", "think_cap") for r in rows)

    summary = {
        "model_id": model_id,
        "think_budget": THINK_BUDGET,
        "answer_budget": ANSWER_BUDGET,
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
        "n_clean_stop": n_clean_stop,
        "n_truncated": n_truncated,
        "pass_easy": _pass_rate(rows, "easy"),
        "pass_hard": _pass_rate(rows, "hard"),
        "rows": rows,
    }

    print(
        f"\n==> coding {n_pass}/{len(rows)} (easy {summary['pass_easy']}, "
        f"hard {summary['pass_hard']})  clean-stop {n_clean_stop}/{len(rows)} "
        f"truncated {n_truncated}/{len(rows)}  median decode {median_decode} tok/s "
        f"(in-band={summary['decode_in_band']})  peak {peak} GB "
        f"(in-band={summary['peak_in_band']})"
    )

    archived, latest = write_run(OUT_DIR, summary)
    print(f"\n==> Wrote {latest}")
    print(f"    archived run: {archived}")


if __name__ == "__main__":
    main()
