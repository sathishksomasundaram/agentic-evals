"""agentic_evals.harness — generic, reusable evaluation primitives.

This is the part the open-model-benchmark platform and individual experiments
import. Experiments under `experiments/` (and the legacy `spikes/`) are *instances*
that orchestrate these primitives; the harness itself is task/experiment-agnostic
where possible.

Stable public API:

    from agentic_evals.harness import (
        # runtime — run a model under a config and measure it
        load_model, probe_arch, make_tq_cache, buffer_for_ratio, measure,
        # tasks — the tool-calling evaluation (prompts + grading)
        SYSTEM_PROMPT_BASE, USER_PROMPT, FILLER_PARAGRAPH, build_long_system, grade,
        # report — persist results
        append_jsonl, rebuild_csv_from_jsonl,
    )
"""

from agentic_evals.harness.report import append_jsonl, rebuild_csv_from_jsonl
from agentic_evals.harness.runtime import (
    buffer_for_ratio,
    load_model,
    make_tq_cache,
    measure,
    probe_arch,
)
from agentic_evals.harness.tasks import (
    FILLER_PARAGRAPH,
    SYSTEM_PROMPT_BASE,
    USER_PROMPT,
    build_long_system,
    grade,
)

__all__ = [
    "FILLER_PARAGRAPH",
    "SYSTEM_PROMPT_BASE",
    "USER_PROMPT",
    "append_jsonl",
    "buffer_for_ratio",
    "build_long_system",
    "grade",
    "load_model",
    "make_tq_cache",
    "measure",
    "probe_arch",
    "rebuild_csv_from_jsonl",
]
