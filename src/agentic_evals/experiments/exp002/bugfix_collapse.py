"""Spike 0003 — replicate the bug-fix language collapse, and find its cause.

Spike 0001 saw `Qwen3-Coder-30B-A3B-Instruct-4bit` begin a correct Python fix on
the `fix_binary_search` bug-fix task, then code-switch to Java mid-token. This
spike asks two things:

  1. **Is it reproducible (n>=3)?** Greedy is deterministic, so we anchor with one
     greedy run and add seeded *sampled* runs (temp>0) to estimate a real failure
     rate rather than a single greedy-path artifact.
  2. **What causes it?** Three prompt variants isolate the trigger:
       - `echo_fix_binsearch`  — the original (echo buggy code, ask for a fix)
       - `scratch_binsearch`   — same algorithm, written from scratch (no echo)
       - `echo_fix_factorial`  — echo-then-fix on a *different* buggy function
     If only the echo-then-fix variants collapse, the prompt structure is the
     cause; if only binary-search variants collapse, the content is.

Collapse = the extracted "Python" code contains Java/C-style markers. Greedy /
seeded-sampled, so the whole spike reproduces.

    uv run agentic-evals exp-002 bugfix-collapse
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

from agentic_evals.experiments.exp002.tasks import CodingTask, extract_code, run_task
from agentic_evals.harness.report import write_run

DEFAULT_MODEL = "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"
MAX_TOKENS = 512
SAMPLED_SEEDS: tuple[int, ...] = (1, 2, 3, 4)  # + 1 greedy run => n=5 per variant
SAMPLED_TEMP = 0.8
SAMPLED_TOP_P = 0.95

OUT_DIR = (
    Path(__file__).resolve().parents[4]
    / "docs"
    / "experiments"
    / "exp-002-mlx-moe-coding-viability"
    / "spikes"
    / "0003-bugfix-collapse"
)

# Markers that should never appear in correct Python — their presence signals a
# code-switch to Java/C-family syntax.
_NON_PYTHON = re.compile(
    r"\b(public|private|static|void|System\.out|println|#include)\b|int\[\]|;\s*\n\s*\}"
)

_BUGGY_BINSEARCH = CodingTask(
    name="echo_fix_binsearch",
    difficulty="hard",
    prompt=(
        "The function below should return the index of `target` in the sorted list "
        "`nums`, or -1 if it is absent, using binary search. It has a bug. Return a "
        "corrected version with the same signature. Respond with a single Python "
        "code block and no prose.\n\n"
        "```python\n"
        "def binary_search(nums, target):\n"
        "    lo, hi = 0, len(nums)\n"
        "    while lo < hi:\n"
        "        mid = (lo + hi) // 2\n"
        "        if nums[mid] == target:\n"
        "            return mid\n"
        "        elif nums[mid] < target:\n"
        "            lo = mid\n"
        "        else:\n"
        "            hi = mid\n"
        "    return -1\n"
        "```"
    ),
    test=(
        "assert binary_search([1,2,3,4,5], 4) == 3\n"
        "assert binary_search([1,2,3,4,5], 1) == 0\n"
        "assert binary_search([1,2,3,4,5], 6) == -1\n"
        "assert binary_search([], 1) == -1\n"
        "assert binary_search([5], 5) == 0\n"
    ),
)

_SCRATCH_BINSEARCH = CodingTask(
    name="scratch_binsearch",
    difficulty="hard",
    prompt=(
        "Write a Python function `binary_search(nums, target)` that returns the "
        "index of `target` in the sorted list `nums`, or -1 if it is absent, using "
        "binary search. Respond with a single Python code block and no prose."
    ),
    test=_BUGGY_BINSEARCH.test,
)

_ECHO_FIX_FACTORIAL = CodingTask(
    name="echo_fix_factorial",
    difficulty="hard",
    prompt=(
        "The function below should return n! (the factorial of n). It has a bug. "
        "Return a corrected version with the same signature. Respond with a single "
        "Python code block and no prose.\n\n"
        "```python\n"
        "def factorial(n):\n"
        "    result = 0\n"
        "    for i in range(1, n + 1):\n"
        "        result *= i\n"
        "    return result\n"
        "```"
    ),
    test=(
        "assert factorial(0) == 1\n"
        "assert factorial(1) == 1\n"
        "assert factorial(5) == 120\n"
        "assert factorial(6) == 720\n"
    ),
)

VARIANTS: tuple[CodingTask, ...] = (
    _BUGGY_BINSEARCH,
    _SCRATCH_BINSEARCH,
    _ECHO_FIX_FACTORIAL,
)


def _looks_non_python(code: str) -> bool:
    return _NON_PYTHON.search(code) is not None


def _run_once(model: Any, tokenizer: Any, task: CodingTask, *, sampler: Any) -> dict[str, Any]:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": task.prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    text = ""
    last: Any = None
    for resp in stream_generate(model, tokenizer, prompt, max_tokens=MAX_TOKENS, sampler=sampler):
        text += resp.text
        last = resp
    code = extract_code(text)
    passed, detail = run_task(task, text)
    return {
        "verdict": "PASS" if passed else "FAIL",
        "detail": detail,
        "collapsed": _looks_non_python(code),
        "finish_reason": last.finish_reason,
        "snippet": code[:200],
    }


def main() -> None:
    model_id = os.environ.get("OMB_MODEL", "").strip() or DEFAULT_MODEL
    print(f"==> exp-002 spike 0003 — bug-fix language-collapse replication for {model_id}")

    model, tokenizer = load(model_id)  # type: ignore[misc]
    greedy = make_sampler(temp=0.0)
    sampled = make_sampler(temp=SAMPLED_TEMP, top_p=SAMPLED_TOP_P)

    variant_rows: list[dict[str, Any]] = []
    for task in VARIANTS:
        runs: list[dict[str, Any]] = []

        greedy_run = _run_once(model, tokenizer, task, sampler=greedy)
        greedy_run.update({"kind": "greedy", "seed": None})
        runs.append(greedy_run)

        for seed in SAMPLED_SEEDS:
            mx.random.seed(seed)
            run = _run_once(model, tokenizer, task, sampler=sampled)
            run.update({"kind": "sampled", "seed": seed})
            runs.append(run)

        n = len(runs)
        n_pass = sum(r["verdict"] == "PASS" for r in runs)
        n_collapse = sum(r["collapsed"] for r in runs)
        variant_rows.append(
            {
                "variant": task.name,
                "n": n,
                "n_pass": n_pass,
                "n_collapse": n_collapse,
                "runs": runs,
            }
        )
        print(
            f"    {task.name:20s} PASS {n_pass}/{n}   collapse {n_collapse}/{n}   "
            f"(greedy: {greedy_run['verdict']}, collapsed={greedy_run['collapsed']})"
        )

    summary = {
        "model_id": model_id,
        "max_tokens": MAX_TOKENS,
        "sampled_temp": SAMPLED_TEMP,
        "sampled_top_p": SAMPLED_TOP_P,
        "sampled_seeds": list(SAMPLED_SEEDS),
        "runs_per_variant": 1 + len(SAMPLED_SEEDS),
        "variants": variant_rows,
    }

    archived, latest = write_run(OUT_DIR, summary)
    print(f"\n==> Wrote {latest}")
    print(f"    archived run: {archived}")


if __name__ == "__main__":
    main()
