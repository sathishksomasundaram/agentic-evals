"""Spike 0007 — n=3 replication of spike 0006's long-context surprise.

Spike 0006 showed (n=1 each):
  long_baseline (no TQ, full FP): ❌ FAIL — model lost tool-calling under 1700-token filler dilution
  long_buf1024 (yzamari K3/V2 buf=1024): ✅ PASS — perfect tool call, 10% faster than baseline

We need to know if the surprise is real or a single-run artifact. n=3 each.
Sampling is greedy by default, so we expect output to be byte-identical across
runs of the same config — but timing will vary, and we want to catch any
non-determinism in the TQ Metal kernels.

Run with:
    uv run agentic-evals exp-001 replication-buf-cliff
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
N_REPLICATES = 3

OUT_DIR = (
    Path(__file__).resolve().parents[4]
    / "docs"
    / "experiments"
    / "exp-001-mlx-kv-compression-toolcalling"
    / "spikes"
    / "0007-buffer-cliff-replication"
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

    # Config A — baseline (no TQ) × N_REPLICATES
    print(f"\n==> Config A: long_baseline × {N_REPLICATES}")
    for i in range(N_REPLICATES):

        def _run_baseline(p: str = prompt) -> str:
            return generate(model, tokenizer, p, max_tokens=MAX_TOKENS, verbose=False)

        run = measure(f"baseline_r{i + 1}", _run_baseline)
        run["verdict"] = _grade(run["output"])
        preview = run["output"][:120].replace("\n", "\\n")
        print(f"  r{i + 1}: time={run['time_s']}s  verdict={run['verdict']}")
        print(f"       output[:120]: {preview!r}")
        runs.append({"config": "baseline", "replicate": i + 1, **run})

    # Config B — yzamari TQ K3/V2 buffer=1024 × N_REPLICATES
    print(f"\n==> Config B: long_tq_K3V2_buf1024 × {N_REPLICATES}")
    for i in range(N_REPLICATES):
        cache = make_turboquant_cache(model, key_bits=3, value_bits=2, buffer_size=1024)

        def _run_tq(p: str = prompt, c: list[Any] = cache) -> str:
            return generate(
                model, tokenizer, p, max_tokens=MAX_TOKENS, prompt_cache=c, verbose=False
            )

        run = measure(f"tq_buf1024_r{i + 1}", _run_tq)
        run["verdict"] = _grade(run["output"])
        preview = run["output"][:120].replace("\n", "\\n")
        print(f"  r{i + 1}: time={run['time_s']}s  verdict={run['verdict']}")
        print(f"       output[:120]: {preview!r}")
        runs.append({"config": "tq_buf1024", "replicate": i + 1, **run})

    # Summarize
    print("\n==> SUMMARY")
    for cfg in ("baseline", "tq_buf1024"):
        cfg_runs = [r for r in runs if r["config"] == cfg]
        verdicts = [r["verdict"] for r in cfg_runs]
        times = [r["time_s"] for r in cfg_runs]
        outputs = {r["output"] for r in cfg_runs}
        pass_rate = sum(1 for v in verdicts if v in ("PASS", "OTHER")) / len(verdicts)
        # OTHER includes the spike 0006 "perfect JSON but my grader said OTHER" false-negative.
        # For replication we count PASS+OTHER as "produced JSON" — we'll inspect manually.
        print(
            f"  {cfg}: verdicts={verdicts}  "
            f"mean_time={sum(times) / len(times):.3f}s  "
            f"unique_outputs={len(outputs)}  "
            f"pass-or-other_rate={pass_rate:.2f}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "model_id": MODEL_ID,
        "max_tokens": MAX_TOKENS,
        "n_replicates": N_REPLICATES,
        "prompt_tokens": prompt_tokens,
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
