"""v0 runner — multi-model leaderboard per ADR 0012.

For each model: 2 prompts × 3 TQ configs (baseline / ratio≈0.5 / ratio≈1.0)
= 6 cells per model = 36 cells total across the v0 list.

Each model is loaded once, all 6 cells run, then unloaded by going out of scope.
Outputs (under docs/experiments/exp-001-mlx-kv-compression-toolcalling/results/):
  raw-results.jsonl  — one JSON object per row
  leaderboard.csv    — flat table for downstream charting

One model per process (the yzamari state-leak workaround) — pass the model with
``--model`` (or set the ``OMB_V0_MODEL`` env var directly):
    uv run agentic-evals exp-001 v0-runner --model <hf-id>
"""

from __future__ import annotations

import csv
import gc
import json
import os
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm import generate, load
from mlx_turboquant import make_turboquant_cache, patch_model  # type: ignore[import-untyped]

from agentic_evals.experiments.exp001.buffer_cliff import (
    SYSTEM_PROMPT_BASE,
    USER_PROMPT,
    _build_long_prompt,
    _grade,
)
from agentic_evals.harness.runtime import measure, probe_arch

# Per ADR 0012 v0 model list (HF IDs verified 2026-05-27).
MODEL_IDS: tuple[str, ...] = (
    "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mlx-community/Qwen3-4B-Instruct-2507-6bit",
    "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit",
    "mlx-community/Phi-4-mini-instruct-4bit",
    "mlx-community/gemma-3-4b-it-4bit",
)

MAX_TOKENS = 200
OUT_DIR = (
    Path(__file__).resolve().parents[4]
    / "docs"
    / "experiments"
    / "exp-001-mlx-kv-compression-toolcalling"
    / "results"
)


def _build_prompts(tokenizer: Any) -> tuple[tuple[str, int], tuple[str, int]]:
    """Build (short, long) prompts and return their per-model token counts."""
    msgs_short = [
        {"role": "system", "content": SYSTEM_PROMPT_BASE},
        {"role": "user", "content": USER_PROMPT},
    ]
    prompt_short = tokenizer.apply_chat_template(
        msgs_short, tokenize=False, add_generation_prompt=True
    )

    long_system = _build_long_prompt(target_filler_tokens=1700, tokenizer=tokenizer)
    msgs_long = [
        {"role": "system", "content": long_system},
        {"role": "user", "content": USER_PROMPT},
    ]
    prompt_long = tokenizer.apply_chat_template(
        msgs_long, tokenize=False, add_generation_prompt=True
    )

    return (
        (prompt_short, len(tokenizer.encode(prompt_short))),
        (prompt_long, len(tokenizer.encode(prompt_long))),
    )


def _run_one_cell(
    model: Any,
    tokenizer: Any,
    prompt: str,
    prompt_tokens: int,
    config_label: str,
    tq_cache: list[Any] | None,
) -> dict[str, Any]:
    """One generation, captured + graded."""
    if tq_cache is None:

        def _run_baseline(p: str = prompt) -> str:
            return generate(model, tokenizer, p, max_tokens=MAX_TOKENS, verbose=False)

        measured = measure(config_label, _run_baseline)
    else:

        def _run_tq(p: str = prompt, c: list[Any] = tq_cache) -> str:
            return generate(
                model, tokenizer, p, max_tokens=MAX_TOKENS, prompt_cache=c, verbose=False
            )

        measured = measure(config_label, _run_tq)
    measured["verdict"] = _grade(measured["output"])
    measured["prompt_tokens"] = prompt_tokens
    return measured


def _run_model(model_id: str) -> list[dict[str, Any]]:
    """Load model, run all 6 cells, return list of structured rows."""
    print(f"\n{'=' * 78}\n==> {model_id}\n{'=' * 78}")
    t_load_start = time.perf_counter()
    model, tokenizer = load(model_id)  # type: ignore[misc]
    load_time = time.perf_counter() - t_load_start
    head_dim, num_layers = probe_arch(model)
    print(f"    loaded in {load_time:.1f}s · head_dim={head_dim} · num_layers={num_layers}")

    model = patch_model(model)

    (prompt_short, short_toks), (prompt_long, long_toks) = _build_prompts(tokenizer)
    print(f"    short prompt: {short_toks} tokens · long prompt: {long_toks} tokens")

    rows: list[dict[str, Any]] = []

    for prompt_label, prompt, prompt_tokens in (
        ("short", prompt_short, short_toks),
        ("long", prompt_long, long_toks),
    ):
        # Config A: baseline (no TQ)
        label_a = f"{prompt_label}_baseline"
        run = _run_one_cell(model, tokenizer, prompt, prompt_tokens, label_a, None)
        run.update({"model_id": model_id, "prompt_label": prompt_label, "config": "baseline"})
        rows.append(run)
        print(
            f"    {label_a:40s} verdict={run['verdict']:11s} "
            f"time={run['time_s']:.2f}s mem={run['peak_mem_gb']:.2f}GB"
        )

        # Config B: TQ K3/V2 buffer_ratio≈0.5
        buf_05 = max(8, round(0.5 * prompt_tokens))
        cache_05 = make_turboquant_cache(model, key_bits=3, value_bits=2, buffer_size=buf_05)
        label_b = f"{prompt_label}_tq_K3V2_ratio0.5_buf{buf_05}"
        run = _run_one_cell(model, tokenizer, prompt, prompt_tokens, label_b, cache_05)
        run.update(
            {
                "model_id": model_id,
                "prompt_label": prompt_label,
                "config": "tq_ratio_0.5",
                "buffer_size": buf_05,
            }
        )
        rows.append(run)
        print(
            f"    {label_b:40s} verdict={run['verdict']:11s} "
            f"time={run['time_s']:.2f}s mem={run['peak_mem_gb']:.2f}GB"
        )

        # Config C: TQ K3/V2 buffer_ratio≈1.0 (≈ no compression — control)
        buf_10 = prompt_tokens
        cache_10 = make_turboquant_cache(model, key_bits=3, value_bits=2, buffer_size=buf_10)
        label_c = f"{prompt_label}_tq_K3V2_ratio1.0_buf{buf_10}"
        run = _run_one_cell(model, tokenizer, prompt, prompt_tokens, label_c, cache_10)
        run.update(
            {
                "model_id": model_id,
                "prompt_label": prompt_label,
                "config": "tq_ratio_1.0",
                "buffer_size": buf_10,
            }
        )
        rows.append(run)
        print(
            f"    {label_c:40s} verdict={run['verdict']:11s} "
            f"time={run['time_s']:.2f}s mem={run['peak_mem_gb']:.2f}GB"
        )

    # Free unified memory between models
    del model, tokenizer
    mx.metal.clear_cache()
    gc.collect()
    return rows


def _rebuild_csv_from_jsonl(jsonl_path: Path, csv_path: Path) -> int:
    """CSV is materialized from JSONL at end of every run.

    JSONL is append-only across multiple invocations (per-process subprocess
    workaround for yzamari cross-model state leak). The CSV is always the
    latest authoritative flat view.
    """
    csv_fields = [
        "model_id",
        "prompt_label",
        "config",
        "buffer_size",
        "verdict",
        "time_s",
        "peak_mem_gb",
        "prompt_tokens",
        "label",
        "error",
        "run_timestamp",
    ]
    rows: list[dict[str, Any]] = []
    if jsonl_path.exists():
        with jsonl_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return len(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = OUT_DIR / "raw-results.jsonl"
    csv_path = OUT_DIR / "leaderboard.csv"

    # Per-process workaround: OMB_V0_MODEL env var overrides MODEL_IDS so each
    # subprocess invocation runs exactly one model. This defeats the yzamari
    # cross-model state leak documented in v0 REPORT finding #5.
    single = os.environ.get("OMB_V0_MODEL", "").strip()
    models_to_run: tuple[str, ...] = (single,) if single else MODEL_IDS
    print(f"==> Running {len(models_to_run)} model(s): {list(models_to_run)}")

    run_timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    all_rows: list[dict[str, Any]] = []
    for model_id in models_to_run:
        try:
            rows = _run_model(model_id)
            for r in rows:
                r["run_timestamp"] = run_timestamp
            all_rows.extend(rows)
            with jsonl_path.open("a") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
        except Exception as exc:  # noqa: BLE001 — v0 collects failures, doesn't propagate
            err_row: dict[str, Any] = {
                "model_id": model_id,
                "error": f"{type(exc).__name__}: {exc}",
                "run_timestamp": run_timestamp,
            }
            print(f"    !! FAILED on {model_id}: {err_row['error']}")
            with jsonl_path.open("a") as f:
                f.write(json.dumps(err_row) + "\n")
            all_rows.append(err_row)

    total_persisted = _rebuild_csv_from_jsonl(jsonl_path, csv_path)

    # Console summary — this invocation only
    print(f"\n{'=' * 78}\n==> v0 SUMMARY (this invocation)\n{'=' * 78}")
    print(f"  Rows this run: {len(all_rows)}")
    print(f"  Total rows in JSONL across all runs: {total_persisted}")
    print(f"  JSONL: {jsonl_path}")
    print(f"  CSV:   {csv_path}")
    print("\n  This run's verdict counts:")
    counts: dict[str, int] = {}
    for r in all_rows:
        v = r.get("verdict", "ERROR")
        counts[v] = counts.get(v, 0) + 1
    for v, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"    {v:12s} {n:3d}")


if __name__ == "__main__":
    main()
