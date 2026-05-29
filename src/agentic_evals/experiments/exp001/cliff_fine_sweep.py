"""Spike 0008 — fine sweep around buffer_ratio=0.5 on Qwen-2.5-7B long.

Spike 0006: buf=1024 / 1966 = 0.521 → PASS
v0 first run: buf=983 / 1966 = 0.500 → FAIL

The cliff sits somewhere in that 21-token (1%) gap. This spike pins it.

Sweep: buf ∈ {973, 983, 993, 1003, 1013, 1023} (ratios 0.495 → 0.520 in
1%-ish increments). Same prompt, same model, K3/V2 fixed, greedy.

Run with:
    uv run agentic-evals exp-001 cliff-fine-sweep
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mlx_lm import generate, load
from mlx_turboquant import make_turboquant_cache, patch_model  # type: ignore[import-untyped]

from agentic_evals.experiments.exp001.buffer_cliff import (
    USER_PROMPT,
    _build_long_prompt,
    _grade,
)
from agentic_evals.harness.runtime import measure, probe_arch

MODEL_ID = "mlx-community/Qwen2.5-7B-Instruct-4bit"
MAX_TOKENS = 200
BUFFER_VALUES: tuple[int, ...] = (973, 983, 993, 1003, 1013, 1023)

OUT_DIR = (
    Path(__file__).resolve().parents[4]
    / "docs"
    / "experiments"
    / "exp-001-mlx-kv-compression-toolcalling"
    / "spikes"
    / "0008-cliff-fine-sweep"
)


def main() -> None:
    print(f"==> Loading model: {MODEL_ID}")
    model, tokenizer = load(MODEL_ID)  # type: ignore[misc]
    head_dim, num_layers = probe_arch(model)
    print(f"    head_dim={head_dim}, num_layers={num_layers}")

    print("==> Patching model")
    model = patch_model(model)

    long_system = _build_long_prompt(target_filler_tokens=1700, tokenizer=tokenizer)
    messages = [
        {"role": "system", "content": long_system},
        {"role": "user", "content": USER_PROMPT},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_tokens = len(tokenizer.encode(prompt))
    print(f"==> Long prompt: {prompt_tokens} tokens")

    runs: list[dict[str, Any]] = []
    for buf in BUFFER_VALUES:
        ratio = buf / prompt_tokens
        cache = make_turboquant_cache(model, key_bits=3, value_bits=2, buffer_size=buf)

        def _run(p: str = prompt, c: list[Any] = cache) -> str:
            return generate(
                model, tokenizer, p, max_tokens=MAX_TOKENS, prompt_cache=c, verbose=False
            )

        run = measure(f"buf{buf}", _run)
        run["verdict"] = _grade(run["output"])
        run["buffer_size"] = buf
        run["buffer_ratio"] = round(ratio, 4)
        run["prompt_tokens"] = prompt_tokens
        preview = run["output"][:120].replace("\n", "\\n")
        print(
            f"  buf={buf:4d} (ratio={ratio:.4f}) :: verdict={run['verdict']:11s} "
            f"time={run['time_s']:.2f}s :: {preview!r}"
        )
        runs.append(run)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "model_id": MODEL_ID,
        "max_tokens": MAX_TOKENS,
        "prompt_tokens": prompt_tokens,
        "head_dim": head_dim,
        "num_layers": num_layers,
        "port": "yzamari/mlx-turboquant",
        "bits": "K3/V2 fixed",
        "runs": runs,
    }
    out_path = OUT_DIR / "raw-results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n==> Wrote {out_path}")


if __name__ == "__main__":
    main()
