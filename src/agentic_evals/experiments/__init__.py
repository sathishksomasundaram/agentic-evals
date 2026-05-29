"""Per-experiment runner code.

Each experiment is a subpackage (e.g. :mod:`agentic_evals.experiments.exp001`)
holding its spike runners, multi-model runner, and chart builder. The generic,
reusable core lives in :mod:`agentic_evals.harness`; experiments import from it.

Runners are invoked through the ``agentic-evals`` console script — see
:mod:`agentic_evals.cli`:

    uv run agentic-evals exp-001 cliff-fine-sweep
"""
