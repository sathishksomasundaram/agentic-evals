"""Spike 0003 — TurboQuant bit-width sweep on Qwen-2.5-7B-Instruct.

Per ADR 0011: switch v0 primary to a non-reasoning instruct model. Per spike 0002
REPORT: bits=3 symmetric collapsed the R1-Distill model. This spike finds the
TurboQuant quality cliff on a real instruct model across [None (baseline), 4, 3, 2].

Run with:
    uv run agentic-evals exp-001 bit-width-sweep
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mlx_lm import generate, load
from mlx_turboquant.cache import TurboQuantKVCache  # type: ignore[import-untyped]

from agentic_evals.harness.runtime import measure, probe_arch

MODEL_ID = "mlx-community/Qwen2.5-7B-Instruct-4bit"
MAX_TOKENS = 300  # bumped from 200 — give the instruct model room to also explain if it wants
BIT_WIDTHS: tuple[int | None, ...] = (None, 4, 3, 2)  # None = baseline mlx-lm cache

SYSTEM_PROMPT = """You are a helpful assistant. You have access to one tool:

web_search(query: str, time_range: str | None = None) -> str
  Description: Run a DuckDuckGo search and return results.
  Parameters:
    query (required, string): the search query
    time_range (optional, string): one of "d" (day), "w" (week), "m" (month),
      "y" (year), or empty string. Defaults to inferred-from-query when omitted.

When you need to use the tool, respond with EXACTLY this JSON format and nothing else:

  {"tool": "web_search", "args": {"query": "<your query>", "time_range": "<bucket-or-null>"}}

When you have the answer, respond normally without JSON."""

USER_PROMPT = "What's the weather in San Francisco right now?"

OUT_DIR = (
    Path(__file__).resolve().parents[4]
    / "docs"
    / "experiments"
    / "exp-001-mlx-kv-compression-toolcalling"
    / "spikes"
    / "0003-bit-width-sweep"
)


def main() -> None:
    print(f"==> Loading model: {MODEL_ID}")
    model, tokenizer = load(MODEL_ID)  # type: ignore[misc]
    head_dim, num_layers = probe_arch(model)
    print(f"    head_dim={head_dim}, num_layers={num_layers}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"==> Prompt ({len(prompt)} chars)")

    runs: list[dict[str, Any]] = []
    for bits in BIT_WIDTHS:
        label = "baseline" if bits is None else f"tq_bits{bits}"
        print(f"\n==> RUN: {label}")
        if bits is None:
            run = measure(
                label,
                lambda: generate(model, tokenizer, prompt, max_tokens=MAX_TOKENS, verbose=False),
            )
        else:
            cache = [TurboQuantKVCache(bits=bits, head_dim=head_dim) for _ in range(num_layers)]

            def _run(c: list[Any] = cache) -> str:
                return generate(
                    model, tokenizer, prompt, max_tokens=MAX_TOKENS, prompt_cache=c, verbose=False
                )

            run = measure(label, _run)
        print(f"    time={run['time_s']}s  peak_mem={run['peak_mem_gb']} GB")
        # Print first 240 chars so the terminal stays readable
        out_preview = run["output"][:240].replace("\n", "\\n")
        print(f"    output[:240]: {out_preview!r}")
        runs.append(run)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "model_id": MODEL_ID,
        "max_tokens": MAX_TOKENS,
        "head_dim": head_dim,
        "num_layers": num_layers,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": USER_PROMPT,
        "bit_widths_swept": [b if b is not None else "baseline" for b in BIT_WIDTHS],
        "runs": runs,
    }
    out_path = OUT_DIR / "raw-results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n==> Wrote {out_path}")


if __name__ == "__main__":
    main()
