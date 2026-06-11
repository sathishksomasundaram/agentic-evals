"""Spike 4B — the exp-001 buffer cliff on an IRIS-shaped prompt.

exp-001 pinned the TurboQuant cliff (buffer_ratio ≈ 0.50, safe ≥ 0.55) on a
weather tool-calling prompt. The published rule has never been tested on
the prompt shape IRIS actually serves. This runner builds an IRIS-shaped
long prompt — the production router system prompt diluted with filler to
~2,000 tokens, exp-001's methodology — and grades *routing JSON* across
buffer ratios on either side of the published cliff.

Model: Qwen2.5-7B-Instruct-4bit (exp-001's pinned-cliff model, IRIS Tier 2).
Greedy sampling, deterministic.

    uv run agentic-evals exp-004 cliff-in-harness
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler

from agentic_evals.experiments.exp005.iris_router import (
    ROUTER_SYSTEM_PROMPT,
    parse_llm_route,
)
from agentic_evals.harness.report import write_run
from agentic_evals.harness.runtime import buffer_for_ratio, load_model, make_tq_cache
from agentic_evals.harness.tasks import FILLER_PARAGRAPH

EXP_DIR = (
    Path(__file__).resolve().parents[4]
    / "docs"
    / "experiments"
    / "exp-004-serving-stack-bakeoff"
)
OUT_DIR = EXP_DIR / "spikes" / "0004-tier3-moe"

MODEL_ID = "mlx-community/Qwen2.5-7B-Instruct-4bit"
TARGET_PROMPT_TOKENS = 2000
RATIOS = [0.45, 0.50, 0.55, 1.0]
# Five clear golden-set cases, one per major route.
CASES = [
    ("schedule a meeting with Alex tomorrow at 3pm", "calendar", "calendar"),
    ("summarize my inbox from today", "communication", "email"),
    ("write a python function to parse csv files", "coding", "coding_agent"),
    ("what's the weather in chennai today?", "weather", "system"),
    ("look up the latest mlx release notes", "search", "rag"),
]


def build_prompt(tokenizer: Any, utterance: str) -> tuple[str, int]:
    """Router system prompt + filler to ~2k tokens + utterance, chat-templated."""
    filler = ""
    while True:
        system = ROUTER_SYSTEM_PROMPT + "\n\n# Workspace context (reference only)\n" + filler
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": utterance},
        ]
        tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        if len(tokens) >= TARGET_PROMPT_TOKENS:
            return tokenizer.decode(tokens), len(tokens)
        filler += FILLER_PARAGRAPH


def main() -> None:
    model, tokenizer = load_model(MODEL_ID, patch_for_tq=True)
    sampler = make_sampler(temp=0.0)  # greedy — exp-001 determinism
    grid: list[dict[str, Any]] = []
    for ratio in RATIOS:
        for utterance, expect_intent, expect_agent in CASES:
            prompt, prompt_tokens = build_prompt(tokenizer, utterance)
            cache = make_tq_cache(
                model, buffer_size=buffer_for_ratio(ratio, prompt_tokens)
            )
            output = generate(
                model,
                tokenizer,
                prompt,
                max_tokens=128,
                sampler=sampler,
                prompt_cache=cache,
            )
            parsed = parse_llm_route(output)
            verdict = (
                "PASS"
                if parsed
                and parsed["intent"] == expect_intent
                and parsed["agent"] == expect_agent
                else ("WRONG_ROUTE" if parsed else "BROKEN_OUTPUT")
            )
            grid.append(
                {
                    "ratio": ratio,
                    "utterance": utterance,
                    "prompt_tokens": prompt_tokens,
                    "verdict": verdict,
                    "output_head": output[:120],
                }
            )
            print(f"ratio={ratio} {verdict:13s} {utterance[:40]!r}")

    summary = {
        "model": MODEL_ID,
        "ratios": {
            str(r): {
                "pass": sum(1 for g in grid if g["ratio"] == r and g["verdict"] == "PASS"),
                "wrong_route": sum(
                    1 for g in grid if g["ratio"] == r and g["verdict"] == "WRONG_ROUTE"
                ),
                "broken": sum(
                    1 for g in grid if g["ratio"] == r and g["verdict"] == "BROKEN_OUTPUT"
                ),
            }
            for r in RATIOS
        },
        "grid": grid,
    }
    write_run(OUT_DIR, {"spike": "0004-cliff-in-harness", "summary": summary})
    print(json.dumps(summary["ratios"], indent=2))


if __name__ == "__main__":
    main()
