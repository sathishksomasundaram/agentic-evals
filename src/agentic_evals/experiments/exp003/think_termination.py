"""Spike 0003 — does the runaway thinking ever terminate with a bigger budget?

Spike 0002 found two hard tasks (`min_stack`, `fix_binary_search`) that generated the
*entire* 8192-token budget without ever closing ``</think>`` (`think_cap`). `min_stack`
passed by luck (a valid fence sat inside the unfinished draft); `fix_binary_search`
failed (its draft still had the infinite-loop bug). The open question, which n=1 at 8192
could not answer: is that non-termination **intrinsic**, or would a much larger budget
eventually let the model close `</think>` and answer cleanly?

This spike re-runs *only* those two tasks at a **16384** thinking budget (2x spike 0002,
~5x spike 0001) plus the same 2048 answer budget, and records whether `</think>` ever
appears (`closed_think`). If it still never closes, non-termination is intrinsic to this
task on this model — not a budget artifact. Greedy / deterministic. n=1.

    uv run agentic-evals exp-003 think-termination
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

from agentic_evals.experiments.exp002.tasks import TASKS, run_task
from agentic_evals.experiments.exp003.thinking import had_thinking, strip_thinking
from agentic_evals.harness.report import write_run

DEFAULT_MODEL = "mlx-community/Qwen3.6-35B-A3B-4bit"
THINK_BUDGET = 16384  # 2x spike 0002 — the whole point of this probe
ANSWER_BUDGET = 2048
MAX_TOKENS = THINK_BUDGET + ANSWER_BUDGET
PROBE_TASKS = ("min_stack", "fix_binary_search")  # the two think_cap tasks from 0002

OUT_DIR = (
    Path(__file__).resolve().parents[4]
    / "docs"
    / "experiments"
    / "exp-003-qwen36-35b-coding-viability"
    / "spikes"
    / "0003-think-termination"
)


def _generate(model: Any, tokenizer: Any, prompt_text: str, *, sampler: Any) -> dict[str, Any]:
    """Greedy-generate with a large thinking budget; report whether </think> closes."""
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
        finish = "think_cap" if last.finish_reason == "length" else last.finish_reason
        finish = finish or "think_cap"
    else:
        finish = last.finish_reason

    return {
        "text": text,
        "closed_think": in_answer,
        "generation_tps": round(float(last.generation_tps), 2),
        "generation_tokens": int(last.generation_tokens),
        "think_tokens": think_tokens,
        "answer_tokens": answer_tokens,
        "peak_mem_gb": round(float(last.peak_memory), 3),
        "finish_reason": finish,
    }


def main() -> None:
    model_id = os.environ.get("OMB_MODEL", "").strip() or DEFAULT_MODEL
    print(f"==> exp-003 spike 0003 — think-termination probe for {model_id}")
    print(f"    think budget={THINK_BUDGET}  answer budget={ANSWER_BUDGET}  (greedy)")

    model, tokenizer = load(model_id)  # type: ignore[misc]
    greedy = make_sampler(temp=0.0)

    tasks = [t for t in TASKS if t.name in PROBE_TASKS]
    rows: list[dict[str, Any]] = []
    for task in tasks:
        gen = _generate(model, tokenizer, task.prompt, sampler=greedy)
        answer = strip_thinking(gen["text"])
        passed, detail = run_task(task, answer)
        row = {
            "task": task.name,
            "difficulty": task.difficulty,
            "verdict": "PASS" if passed else "FAIL",
            "detail": detail,
            "closed_think": gen["closed_think"],
            "generation_tps": gen["generation_tps"],
            "generation_tokens": gen["generation_tokens"],
            "think_tokens": gen["think_tokens"],
            "answer_tokens": gen["answer_tokens"],
            "peak_mem_gb": gen["peak_mem_gb"],
            "finish_reason": gen["finish_reason"],
        }
        rows.append(row)
        print(
            f"    {task.name:18s} {row['verdict']:4s} closed_think={row['closed_think']!s:5s} "
            f"gen={row['generation_tokens']:5d} think={row['think_tokens']:5d} "
            f"ans={row['answer_tokens']:4d} finish={row['finish_reason']:9s} "
            f"peak={row['peak_mem_gb']:5.1f} GB  ({detail})"
        )

    n_closed = sum(r["closed_think"] for r in rows)
    summary = {
        "model_id": model_id,
        "think_budget": THINK_BUDGET,
        "answer_budget": ANSWER_BUDGET,
        "max_tokens": MAX_TOKENS,
        "decoding": "greedy",
        "probe_tasks": list(PROBE_TASKS),
        "n_closed_think": n_closed,
        "n_tasks": len(rows),
        "rows": rows,
    }

    print(
        f"\n==> {n_closed}/{len(rows)} closed </think> at {THINK_BUDGET}-token budget "
        f"(if 0/{len(rows)}: non-termination is intrinsic, not a budget artifact)"
    )

    archived, latest = write_run(OUT_DIR, summary)
    print(f"\n==> Wrote {latest}")
    print(f"    archived run: {archived}")


if __name__ == "__main__":
    main()
