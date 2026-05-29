"""Spike 0006 — buffer_size cliff sweep (yzamari TurboQuant on Qwen-2.5-7B-Instruct).

Per spike 0005: yzamari K3/V2 with buffer=128 works perfectly on a 189-token
prompt; buffer=32 fails. The cliff sits in between. This spike maps it precisely
and tests whether the cliff is absolute (buffer in tokens) or relative
(buffer / total prompt length).

Two regimes:
  SHORT prompt (189 tokens, same as spike 0005): buffer ∈ {32, 64, 96, 128}
  LONG prompt (~2000 tokens): buffer ∈ {128, 256, 512, 1024, 2048}

Output: spike 0006's REPORT.md — the "money chart" for the blog.

Run with:
    uv run agentic-evals exp-001 buffer-cliff

NOTE: shared task defs (prompts, grading, long-prompt builder) now live in
`agentic_evals.harness.tasks`; runtime helpers in `agentic_evals.harness`. The
`_build_long_prompt` / `_grade` aliases below preserve back-compat for other
spike scripts that still import those names from this module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mlx_lm import generate, load
from mlx_turboquant import make_turboquant_cache, patch_model  # type: ignore[import-untyped]

from agentic_evals.harness.runtime import measure, probe_arch
from agentic_evals.harness.tasks import (
    SYSTEM_PROMPT_BASE,
    USER_PROMPT,
    build_long_system,
    grade,
)

# Back-compat aliases — other spikes import these names from this module.
_build_long_prompt = build_long_system
_grade = grade

# Declared so the re-exported task defs are explicit exports (mypy strict /
# implicit-reexport). These names are imported by sibling spike scripts.
__all__ = [
    "SYSTEM_PROMPT_BASE",
    "USER_PROMPT",
    "_build_long_prompt",
    "_grade",
    "main",
]

MODEL_ID = "mlx-community/Qwen2.5-7B-Instruct-4bit"
MAX_TOKENS = 200  # tool call should fit in ~30 tokens; pad for variance

OUT_DIR = (
    Path(__file__).resolve().parents[4]
    / "docs"
    / "experiments"
    / "exp-001-mlx-kv-compression-toolcalling"
    / "spikes"
    / "0006-buffer-cliff"
)


def main() -> None:
    print(f"==> Loading model: {MODEL_ID}")
    model, tokenizer = load(MODEL_ID)  # type: ignore[misc]
    head_dim, num_layers = probe_arch(model)
    print(f"    head_dim={head_dim}, num_layers={num_layers}")

    print("==> Patching model")
    model = patch_model(model)

    # SHORT prompt — same as spike 0005
    messages_short = [
        {"role": "system", "content": SYSTEM_PROMPT_BASE},
        {"role": "user", "content": USER_PROMPT},
    ]
    prompt_short = tokenizer.apply_chat_template(
        messages_short, tokenize=False, add_generation_prompt=True
    )
    short_token_count = len(tokenizer.encode(prompt_short))
    print(f"==> SHORT prompt: {short_token_count} tokens")

    # LONG prompt — aim for ~2000 tokens after chat template
    long_system = _build_long_prompt(target_filler_tokens=1700, tokenizer=tokenizer)
    messages_long = [
        {"role": "system", "content": long_system},
        {"role": "user", "content": USER_PROMPT},
    ]
    prompt_long = tokenizer.apply_chat_template(
        messages_long, tokenize=False, add_generation_prompt=True
    )
    long_token_count = len(tokenizer.encode(prompt_long))
    print(f"==> LONG prompt:  {long_token_count} tokens")

    configs: list[dict[str, Any]] = [
        # SHORT prompt sweep (refine the cliff between 32 and 128 from spike 0005)
        {"prompt_label": "short", "buffer_size": None, "label": "short_baseline"},
        {"prompt_label": "short", "buffer_size": 32, "label": "short_buf32"},
        {"prompt_label": "short", "buffer_size": 64, "label": "short_buf64"},
        {"prompt_label": "short", "buffer_size": 96, "label": "short_buf96"},
        {"prompt_label": "short", "buffer_size": 128, "label": "short_buf128"},
        # LONG prompt sweep
        {"prompt_label": "long", "buffer_size": None, "label": "long_baseline"},
        {"prompt_label": "long", "buffer_size": 128, "label": "long_buf128"},
        {"prompt_label": "long", "buffer_size": 256, "label": "long_buf256"},
        {"prompt_label": "long", "buffer_size": 512, "label": "long_buf512"},
        {"prompt_label": "long", "buffer_size": 1024, "label": "long_buf1024"},
    ]

    runs: list[dict[str, Any]] = []
    for cfg in configs:
        label = str(cfg["label"])
        prompt = prompt_short if cfg["prompt_label"] == "short" else prompt_long
        prompt_tokens = short_token_count if cfg["prompt_label"] == "short" else long_token_count
        print(f"\n==> RUN: {label}  (prompt={prompt_tokens} tokens, buffer={cfg['buffer_size']})")

        if cfg["buffer_size"] is None:

            def _run_baseline(p: str = prompt) -> str:
                return generate(model, tokenizer, p, max_tokens=MAX_TOKENS, verbose=False)

            run = measure(label, _run_baseline)
        else:
            cache = make_turboquant_cache(
                model, key_bits=3, value_bits=2, buffer_size=int(cfg["buffer_size"])
            )

            def _run(p: str = prompt, c: list[Any] = cache) -> str:
                return generate(
                    model, tokenizer, p, max_tokens=MAX_TOKENS, prompt_cache=c, verbose=False
                )

            run = measure(label, _run)

        verdict = _grade(run["output"])
        buf_ratio = (
            float(cfg["buffer_size"]) / prompt_tokens if cfg["buffer_size"] is not None else 1.0
        )
        print(f"    time={run['time_s']}s  verdict={verdict}  buf/total={buf_ratio:.3f}")
        preview = run["output"][:160].replace("\n", "\\n")
        print(f"    output[:160]: {preview!r}")
        run["config"] = cfg
        run["verdict"] = verdict
        run["prompt_tokens"] = prompt_tokens
        run["buffer_to_total_ratio"] = round(buf_ratio, 3)
        runs.append(run)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "model_id": MODEL_ID,
        "max_tokens": MAX_TOKENS,
        "head_dim": head_dim,
        "num_layers": num_layers,
        "short_prompt_tokens": short_token_count,
        "long_prompt_tokens": long_token_count,
        "port": "yzamari/mlx-turboquant",
        "bits": "K3/V2 fixed",
        "runs": runs,
    }
    out_path = OUT_DIR / "raw-results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n==> Wrote {out_path}")


if __name__ == "__main__":
    main()
