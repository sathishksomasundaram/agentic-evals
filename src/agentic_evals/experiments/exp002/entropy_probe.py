"""Spike 0004 — instrument the `mid = (` break point with per-token entropy.

Spike 0003 confirmed (n=5) that the echo-and-fix-this-buggy-binary-search prompt
derails at the *identical* token — the `(` in `mid = (` — and hypothesized that
`mid = (` is a *maximum-entropy branch point*: the model is torn between faithfully
echoing the buggy arithmetic and "fixing" it, and the next-token distribution after
`(` is near-uniform, so greedy/sampling scatters into Java, stray imports, etc.

This spike tests that hypothesis directly. `mlx-lm`'s `GenerationResponse.logprobs`
is the full log-probability vector over the vocabulary at each step, so we can
compute the Shannon entropy (in bits) of every next-token decision and locate the
decision *after* `mid = (`.

Prediction (if the hypothesis holds): in `echo_fix_binsearch` the post-`(` decision
has anomalously high entropy — near the max over the whole generation — whereas in
`scratch_binsearch`, which writes the *same* `mid = (` text but passes, that decision
is low-entropy/confident. Greedy/deterministic, so the whole spike reproduces.

    uv run agentic-evals exp-002 entropy-probe
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

from agentic_evals.experiments.exp002.bugfix_collapse import (
    _BUGGY_BINSEARCH,
    _ECHO_FIX_FACTORIAL,
    _SCRATCH_BINSEARCH,
)
from agentic_evals.experiments.exp002.tasks import CodingTask
from agentic_evals.harness.report import write_run

DEFAULT_MODEL = "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"
MAX_TOKENS = 160  # enough to pass `mid = (` and reveal the collapse onset
TOP_K = 5  # candidate tokens recorded at each step
BRANCH_MARKER = "mid = ("
_LN2 = math.log(2.0)

VARIANTS: tuple[CodingTask, ...] = (
    _BUGGY_BINSEARCH,
    _SCRATCH_BINSEARCH,
    _ECHO_FIX_FACTORIAL,
)

OUT_DIR = (
    Path(__file__).resolve().parents[4]
    / "docs"
    / "experiments"
    / "exp-002-mlx-moe-coding-viability"
    / "spikes"
    / "0004-entropy-probe"
)


def _entropy_bits(logprobs: mx.array) -> float:
    """Shannon entropy (bits) of a vocabulary distribution given natural log-probs."""
    nats = -mx.sum(mx.exp(logprobs) * logprobs)
    return float(nats.item()) / _LN2


def _top_k(tokenizer: Any, logprobs: mx.array, k: int) -> list[dict[str, Any]]:
    """The k most probable next tokens as {token, prob} (descending)."""
    order = mx.argsort(logprobs)[-k:][::-1]  # ascending → take top, reverse
    ids = [int(order[j].item()) for j in range(order.shape[0])]
    return [
        {"token": tokenizer.decode([i]), "prob": round(float(mx.exp(logprobs[i]).item()), 4)}
        for i in ids
    ]


def _probe(model: Any, tokenizer: Any, task: CodingTask, *, sampler: Any) -> dict[str, Any]:
    """Greedy-decode `task`, recording per-step entropy and top-k candidates."""
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": task.prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    steps: list[dict[str, Any]] = []
    for resp in stream_generate(model, tokenizer, prompt, max_tokens=MAX_TOKENS, sampler=sampler):
        steps.append(
            {
                "i": len(steps),
                "token": resp.text,
                "entropy_bits": round(_entropy_bits(resp.logprobs), 3),
                "top_k": _top_k(tokenizer, resp.logprobs, TOP_K),
            }
        )

    full = "".join(s["token"] for s in steps)
    entropies = [s["entropy_bits"] for s in steps]
    max_step = max(steps, key=lambda s: s["entropy_bits"]) if steps else None

    branch = _locate_branch(steps, full)
    return {
        "variant": task.name,
        "n_steps": len(steps),
        "text": full[:600],
        "entropy_max_bits": max(entropies) if entropies else None,
        "entropy_median_bits": round(_median(entropies), 3) if entropies else None,
        "max_entropy_step": max_step,
        "branch": branch,
        "window": _window(steps, branch["step"]) if branch else None,
    }


def _locate_branch(steps: list[dict[str, Any]], full: str) -> dict[str, Any] | None:
    """Find the decision *after* the first `mid = (`; return its step + entropy + rank."""
    pos = full.find(BRANCH_MARKER)
    if pos == -1:
        return None
    end = pos + len(BRANCH_MARKER)
    cursor = 0
    for s in steps:
        start = cursor
        cursor += len(s["token"])
        if start >= end:  # first token that begins at/after the `(`
            ranked = sorted(steps, key=lambda x: x["entropy_bits"], reverse=True)
            rank = next(i for i, x in enumerate(ranked) if x["i"] == s["i"]) + 1
            return {
                "step": s["i"],
                "after_marker": BRANCH_MARKER,
                "entropy_bits": s["entropy_bits"],
                "entropy_rank": rank,  # 1 == highest-entropy step in the generation
                "n_steps": len(steps),
                "top_k": s["top_k"],
            }
    return None


def _window(steps: list[dict[str, Any]], center: int, radius: int = 3) -> list[dict[str, Any]]:
    """A small slice of steps around `center` (entropy + chosen token only)."""
    lo = max(0, center - radius)
    hi = min(len(steps), center + radius + 1)
    return [
        {"i": s["i"], "token": s["token"], "entropy_bits": s["entropy_bits"]} for s in steps[lo:hi]
    ]


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def main() -> None:
    model_id = os.environ.get("OMB_MODEL", "").strip() or DEFAULT_MODEL
    print(f"==> exp-002 spike 0004 — entropy probe of the `mid = (` break point for {model_id}")

    model, tokenizer = load(model_id)  # type: ignore[misc]
    greedy = make_sampler(temp=0.0)

    rows: list[dict[str, Any]] = []
    for task in VARIANTS:
        row = _probe(model, tokenizer, task, sampler=greedy)
        rows.append(row)
        b = row["branch"]
        if b:
            print(
                f"    {task.name:20s} branch `{BRANCH_MARKER}` entropy="
                f"{b['entropy_bits']:6.3f} bits  rank {b['entropy_rank']}/{b['n_steps']}  "
                f"(max {row['entropy_max_bits']:.3f}, median {row['entropy_median_bits']:.3f})"
            )
        else:
            print(
                f"    {task.name:20s} no `{BRANCH_MARKER}` emitted  "
                f"(max {row['entropy_max_bits']:.3f}, median {row['entropy_median_bits']:.3f})"
            )

    summary = {
        "model_id": model_id,
        "max_tokens": MAX_TOKENS,
        "top_k": TOP_K,
        "branch_marker": BRANCH_MARKER,
        "decoding": "greedy",
        "variants": rows,
    }

    archived, latest = write_run(OUT_DIR, summary)
    print(f"\n==> Wrote {latest}")
    print(f"    archived run: {archived}")


if __name__ == "__main__":
    main()
