"""exp-002 — MLX MoE coding viability on Apple Silicon.

Runner instances for the experiment documented under
``docs/experiments/exp-002-mlx-moe-coding-viability/``. Each module here has a
``main()`` entry point dispatched by the ``agentic-evals`` CLI:

    uv run agentic-evals exp-002 <runner>

where ``<runner>`` is the module name with underscores written as hyphens
(e.g. ``coding-viability`` → :mod:`.coding_viability`).
"""
