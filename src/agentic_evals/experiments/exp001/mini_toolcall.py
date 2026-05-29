"""Mini-spike 0002 — DeepSeek-R1-Distill-Qwen-7B-4bit on MLX, TQ on/off.

Per ADR 0009: v0 wedge axis 2 is TurboQuant K4/V2 on/off. This mini-spike
proves the test apparatus produces interpretable signal before scaling to
the full 18-30-row v0 matrix.

NOTE on K4/V2: rachittshah/mlx-turboquant supports symmetric N-bit only
(bits=2, 3, 3.5, 4) — no native K-vs-V asymmetry. We use bits=3 as the
closest symmetric proxy. See spike 0002's SETUP.md.

Run with:
    uv run agentic-evals exp-001 mini-toolcall
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm import generate, load
from mlx_turboquant.cache import TurboQuantKVCache  # type: ignore[import-untyped]

MODEL_ID = "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit"
MAX_TOKENS = 200

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
    / "0002-mini-toolcall-run"
)


def _probe_arch(model: Any) -> tuple[int, int]:
    """Return (head_dim, num_layers) by best-effort inspection of the loaded model."""
    # mlx-lm exposes the underlying model under .model on most arches; layers vary.
    candidates_layers = [
        getattr(model, "layers", None),
        getattr(getattr(model, "model", None), "layers", None),
    ]
    layers = next((c for c in candidates_layers if c is not None), None)
    num_layers = len(layers) if layers is not None else -1

    # head_dim: prefer config.head_dim, fall back to hidden_size / num_attention_heads
    args = getattr(model, "args", None) or getattr(model, "config", None)
    head_dim = getattr(args, "head_dim", None) if args is not None else None
    if head_dim is None and args is not None:
        hidden = getattr(args, "hidden_size", None)
        heads = getattr(args, "num_attention_heads", None)
        if hidden and heads:
            head_dim = hidden // heads
    return int(head_dim or 128), int(num_layers or -1)


def _measure(label: str, fn: Any) -> dict[str, Any]:
    """Time + memory-measure a single generate call."""
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    output = fn()
    elapsed = time.perf_counter() - t0
    peak_gb = mx.get_peak_memory() / 1e9
    return {
        "label": label,
        "time_s": round(elapsed, 3),
        "peak_mem_gb": round(peak_gb, 3),
        "output": output,
    }


def main() -> None:
    print(f"==> Loading model: {MODEL_ID}")
    t0 = time.perf_counter()
    # mlx_lm.load returns Union[Tuple[Module, Tok], Tuple[Module, Tok, dict]] —
    # default return_config=False gives 2-tuple, but mypy can't narrow the Union.
    model, tokenizer = load(MODEL_ID)  # type: ignore[misc]
    print(f"    Loaded in {time.perf_counter() - t0:.1f}s")

    head_dim, num_layers = _probe_arch(model)
    print(f"    head_dim={head_dim}, num_layers={num_layers}")

    # Build chat prompt via the model's chat template.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"\n==> Prompt ({len(prompt)} chars)")

    # --- Run 1: default mlx-lm KVCache (TQ OFF) ---
    print("\n==> RUN 1: TQ OFF (default cache)")
    run_off = _measure(
        "TQ_OFF",
        lambda: generate(model, tokenizer, prompt, max_tokens=MAX_TOKENS, verbose=False),
    )
    print(f"    time={run_off['time_s']}s  peak_mem={run_off['peak_mem_gb']} GB")
    print(f"    output:\n      {run_off['output']!r}")

    # --- Run 2: rachittshah TurboQuantKVCache (TQ ON, symmetric 3-bit proxy) ---
    print("\n==> RUN 2: TQ ON (TurboQuantKVCache bits=3, symmetric proxy for K4/V2)")
    tq_cache = [TurboQuantKVCache(bits=3, head_dim=head_dim) for _ in range(num_layers)]
    run_on = _measure(
        "TQ_ON_bits3",
        lambda: generate(
            model, tokenizer, prompt, max_tokens=MAX_TOKENS, prompt_cache=tq_cache, verbose=False
        ),
    )
    print(f"    time={run_on['time_s']}s  peak_mem={run_on['peak_mem_gb']} GB")
    print(f"    output:\n      {run_on['output']!r}")

    # --- Persist raw results ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "model_id": MODEL_ID,
        "max_tokens": MAX_TOKENS,
        "head_dim": head_dim,
        "num_layers": num_layers,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": USER_PROMPT,
        "runs": [run_off, run_on],
    }
    out_path = OUT_DIR / "raw-results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n==> Wrote {out_path}")


if __name__ == "__main__":
    main()
