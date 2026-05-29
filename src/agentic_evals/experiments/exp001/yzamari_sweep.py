"""Spike 0005 — yzamari/mlx-turboquant cross-port validation.

Spike 0004 Part B showed rachittshah's PolarQuant cannot reconstruct real Qwen-2.5
K/V (per-row cos = 0.33). Spike 0005 tests yzamari's port, which uses the SAME
algorithm but has two architectural differences:

  1. Asymmetric K/V by default (key_bits=3, value_bits=2 — matches POC's recommendation)
  2. buffer_size workaround: keep the most recent N tokens UNCOMPRESSED for quality
     (the "needle retrieval FAILS on compressed tokens" workaround the README admits)
  3. Metal kernels in scaled_dot_product_attention (patch_model() installs them)

Configs tested on Qwen-2.5-7B-Instruct-4bit with the same tool-calling prompt:

  - baseline (default mlx-lm cache, no TQ) — sanity check
  - yzamari default (K3/V2, buffer=128) — should work due to buffer
  - yzamari buffer=0 — disables the workaround, should fail like rachittshah
  - yzamari aggressive (K2/V2, buffer=128) — push the bit budget

Run with:
    uv run agentic-evals exp-001 yzamari-sweep
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mlx_lm import generate, load
from mlx_turboquant import make_turboquant_cache, patch_model  # type: ignore[import-untyped]

from agentic_evals.harness.runtime import measure, probe_arch

MODEL_ID = "mlx-community/Qwen2.5-7B-Instruct-4bit"
MAX_TOKENS = 300

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
    / "0005-yzamari-port"
)


CONFIGS: list[dict[str, Any]] = [
    {"label": "baseline_no_tq", "tq": False},
    {
        "label": "yzamari_default_K3V2_buffer128",
        "tq": True,
        "key_bits": 3,
        "value_bits": 2,
        "buffer_size": 128,
    },
    {
        "label": "yzamari_K2V2_buffer128",
        "tq": True,
        "key_bits": 2,
        "value_bits": 2,
        "buffer_size": 128,
    },
    {
        # Force compression to actually happen by shrinking the buffer well
        # below our prompt length — exercises the compressed-attention path.
        "label": "yzamari_K3V2_buffer32_forces_compression",
        "tq": True,
        "key_bits": 3,
        "value_bits": 2,
        "buffer_size": 32,
    },
]
# NOTE: yzamari with buffer_size=0 crashes with "range() arg 3 must not be zero"
# inside its _flush() method. Skipped here; documented in the REPORT as a port bug
# (NOT an algorithm finding — it's the chunk_size = flush_batch_size or buffer_size
# fallback ending up as 0).


def main() -> None:
    print(f"==> Loading model: {MODEL_ID}")
    model, tokenizer = load(MODEL_ID)  # type: ignore[misc]
    head_dim, num_layers = probe_arch(model)
    print(f"    head_dim={head_dim}, num_layers={num_layers}")

    # IMPORTANT: yzamari requires patch_model() to install the Metal-kernel attention.
    # Without it, the cache returns the buffer but attention doesn't know how to
    # consume compressed entries — undefined behavior.
    print("==> Patching model (installs Metal-kernel attention + cache factory)")
    model = patch_model(model)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"==> Prompt ({len(prompt)} chars)")

    runs: list[dict[str, Any]] = []
    for cfg in CONFIGS:
        print(f"\n==> RUN: {cfg['label']}")
        if not cfg["tq"]:
            run = measure(
                str(cfg["label"]),
                lambda: generate(model, tokenizer, prompt, max_tokens=MAX_TOKENS, verbose=False),
            )
        else:
            cache = make_turboquant_cache(
                model,
                key_bits=int(cfg["key_bits"]),
                value_bits=int(cfg["value_bits"]),
                buffer_size=int(cfg["buffer_size"]),
            )

            def _run(c: list[Any] = cache) -> str:
                return generate(
                    model, tokenizer, prompt, max_tokens=MAX_TOKENS, prompt_cache=c, verbose=False
                )

            run = measure(str(cfg["label"]), _run)
        print(f"    time={run['time_s']}s  peak_mem={run['peak_mem_gb']} GB")
        out_preview = run["output"][:240].replace("\n", "\\n")
        print(f"    output[:240]: {out_preview!r}")
        run["config"] = cfg
        runs.append(run)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "model_id": MODEL_ID,
        "max_tokens": MAX_TOKENS,
        "head_dim": head_dim,
        "num_layers": num_layers,
        "port": "yzamari/mlx-turboquant",
        "runs": runs,
    }
    out_path = OUT_DIR / "raw-results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n==> Wrote {out_path}")


if __name__ == "__main__":
    main()
