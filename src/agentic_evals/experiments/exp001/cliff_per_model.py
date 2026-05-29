"""Spike 0009 — per-model cliff localization.

v0 data showed Llama-3.2-3B and Phi-4-mini both:
  - COLLAPSE at ratio=0.5 short prompt (smaller models hit cliff harder)
  - PASS at ratio=1.0 (no real compression)

This spike pins where in [0.5, 1.0] each cliff sits, on both short + long.

One model per process (the yzamari state-leak workaround) — pass the model
with ``--model`` (or set the ``OMB_MODEL`` env var directly):
    uv run agentic-evals exp-001 cliff-per-model \\
        --model mlx-community/Llama-3.2-3B-Instruct-4bit
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from mlx_lm import generate, load
from mlx_turboquant import make_turboquant_cache, patch_model  # type: ignore[import-untyped]

from agentic_evals.experiments.exp001.buffer_cliff import (
    SYSTEM_PROMPT_BASE,
    USER_PROMPT,
    _build_long_prompt,
    _grade,
)
from agentic_evals.harness.runtime import measure, probe_arch

MAX_TOKENS = 200
RATIOS: tuple[float, ...] = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.85)
DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"

BASE_OUT_DIR = (
    Path(__file__).resolve().parents[4]
    / "docs"
    / "experiments"
    / "exp-001-mlx-kv-compression-toolcalling"
    / "spikes"
    / "0009-per-model-cliff"
)


def _slug(model_id: str) -> str:
    """Filesystem-safe slug from an HF model id."""
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", model_id).strip("-").lower()


def _run_sweep_for_prompt(
    model: Any,
    tokenizer: Any,
    prompt: str,
    prompt_tokens: int,
    prompt_label: str,
) -> list[dict[str, Any]]:
    """Run the ratio sweep for one prompt regime. Includes baseline (no TQ)."""
    rows: list[dict[str, Any]] = []

    # Baseline first
    def _run_baseline(p: str = prompt) -> str:
        return generate(model, tokenizer, p, max_tokens=MAX_TOKENS, verbose=False)

    base = measure(f"{prompt_label}_baseline", _run_baseline)
    base.update(
        {
            "config": "baseline",
            "prompt_label": prompt_label,
            "prompt_tokens": prompt_tokens,
            "buffer_size": None,
            "ratio": None,
            "verdict": _grade(base["output"]),
        }
    )
    print(
        f"    {prompt_label} baseline                 verdict={base['verdict']:11s} "
        f"time={base['time_s']:.2f}s"
    )
    rows.append(base)

    # Then each ratio
    for ratio in RATIOS:
        buf = max(8, round(ratio * prompt_tokens))
        cache = make_turboquant_cache(model, key_bits=3, value_bits=2, buffer_size=buf)

        def _run_tq(p: str = prompt, c: list[Any] = cache) -> str:
            return generate(
                model, tokenizer, p, max_tokens=MAX_TOKENS, prompt_cache=c, verbose=False
            )

        row = measure(f"{prompt_label}_ratio{ratio:.2f}_buf{buf}", _run_tq)
        row.update(
            {
                "config": f"tq_K3V2_ratio{ratio:.2f}",
                "prompt_label": prompt_label,
                "prompt_tokens": prompt_tokens,
                "buffer_size": buf,
                "ratio": ratio,
                "verdict": _grade(row["output"]),
            }
        )
        print(
            f"    {prompt_label} ratio={ratio:.2f} buf={buf:4d} verdict={row['verdict']:11s} "
            f"time={row['time_s']:.2f}s"
        )
        rows.append(row)
    return rows


def main() -> None:
    model_id = os.environ.get("OMB_MODEL", "").strip() or DEFAULT_MODEL
    print(f"==> Spike 0009 — cliff localization for {model_id}")

    model, tokenizer = load(model_id)  # type: ignore[misc]
    head_dim, num_layers = probe_arch(model)
    print(f"    head_dim={head_dim}  num_layers={num_layers}")

    print("==> Patching model")
    model = patch_model(model)

    # SHORT prompt
    short_msgs = [
        {"role": "system", "content": SYSTEM_PROMPT_BASE},
        {"role": "user", "content": USER_PROMPT},
    ]
    prompt_short = tokenizer.apply_chat_template(
        short_msgs, tokenize=False, add_generation_prompt=True
    )
    short_tokens = len(tokenizer.encode(prompt_short))
    print(f"==> SHORT prompt: {short_tokens} tokens")

    # LONG prompt
    long_system = _build_long_prompt(target_filler_tokens=1700, tokenizer=tokenizer)
    long_msgs = [
        {"role": "system", "content": long_system},
        {"role": "user", "content": USER_PROMPT},
    ]
    prompt_long = tokenizer.apply_chat_template(
        long_msgs, tokenize=False, add_generation_prompt=True
    )
    long_tokens = len(tokenizer.encode(prompt_long))
    print(f"==> LONG prompt:  {long_tokens} tokens")

    rows: list[dict[str, Any]] = []
    print("\n-- SHORT sweep --")
    rows.extend(_run_sweep_for_prompt(model, tokenizer, prompt_short, short_tokens, "short"))
    print("\n-- LONG sweep --")
    rows.extend(_run_sweep_for_prompt(model, tokenizer, prompt_long, long_tokens, "long"))

    out_dir = BASE_OUT_DIR / _slug(model_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "raw-results.json"
    out_path.write_text(
        json.dumps(
            {
                "model_id": model_id,
                "max_tokens": MAX_TOKENS,
                "head_dim": head_dim,
                "num_layers": num_layers,
                "short_prompt_tokens": short_tokens,
                "long_prompt_tokens": long_tokens,
                "ratios_swept": list(RATIOS),
                "port": "yzamari/mlx-turboquant",
                "rows": rows,
            },
            indent=2,
        )
    )
    print(f"\n==> Wrote {out_path}")


if __name__ == "__main__":
    main()
