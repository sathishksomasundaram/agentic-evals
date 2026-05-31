"""exp-003 — the article's *actual* model: Qwen3.6-35B-A3B coding viability.

exp-002 tested a *substitute* (`Qwen3-Coder-30B-A3B-Instruct-4bit`) because we
mistakenly believed the article's named model did not exist. It does:
`Qwen/Qwen3.6-35B-A3B`, with an MLX 4-bit build at
`mlx-community/Qwen3.6-35B-A3B-4bit`. This experiment re-runs the article's claim on
that real model, which differs from the substitute in two ways: it is a *thinking-
by-default* MoE (so the harness must read past a `<think>…</think>` block), and it is
multimodal (the pinned `mlx-lm` loads it as a text-only LM, dropping vision weights).

Runner instances live here; each exposes a ``main()`` dispatched by the CLI:

    uv run agentic-evals exp-003 <runner>

where ``<runner>`` is the module name with underscores written as hyphens.
"""
