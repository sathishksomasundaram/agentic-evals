"""``agentic-evals`` console entry point — dispatch per-experiment runners.

Usage:
    uv run agentic-evals <experiment> <runner> [--model HF_ID]
    uv run agentic-evals --list

Example:
    uv run agentic-evals exp-001 cliff-fine-sweep
    uv run agentic-evals exp-001 v0-runner --model mlx-community/Qwen2.5-7B-Instruct-4bit

The runner slug is the runner module's name with underscores written as hyphens
(``cliff_fine_sweep`` → ``cliff-fine-sweep``). Each runner module exposes a
``main() -> None`` entry point, which this CLI imports lazily and calls.

This thin dispatcher is also the substrate for the autonomous-experiment loop
(see docs/EXPERIMENT-METHODOLOGY.md): give an agent an experiment + a runner and
it can drive a spike to completion without bespoke per-runner scaffolding.
"""

from __future__ import annotations

import argparse
import importlib
import os
import pkgutil
import sys
from collections.abc import Sequence

# CLI experiment slug -> Python subpackage holding that experiment's runners.
_EXPERIMENTS: dict[str, str] = {
    "exp-001": "agentic_evals.experiments.exp001",
    "exp-002": "agentic_evals.experiments.exp002",
    "exp-003": "agentic_evals.experiments.exp003",
    "exp-005": "agentic_evals.experiments.exp005",
}


def _slug_to_module(runner: str) -> str:
    """Map a CLI runner slug (hyphens) to a Python module name (underscores)."""
    return runner.replace("-", "_")


def _module_to_slug(module: str) -> str:
    """Map a Python module name (underscores) to a CLI runner slug (hyphens)."""
    return module.replace("_", "-")


def _runners_for(experiment: str) -> list[str]:
    """Return the available runner slugs for an experiment, sorted."""
    pkg = importlib.import_module(_EXPERIMENTS[experiment])
    search_path = getattr(pkg, "__path__", [])
    return sorted(
        _module_to_slug(info.name)
        for info in pkgutil.iter_modules(search_path)
        if not info.name.startswith("_")
    )


def _print_listing() -> None:
    print("Available experiments and runners:\n")
    for experiment in sorted(_EXPERIMENTS):
        print(f"  {experiment}")
        for slug in _runners_for(experiment):
            print(f"    - {slug}")
    print("\nRun one with:  uv run agentic-evals <experiment> <runner>")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="agentic-evals",
        description="Dispatch reproducible per-experiment runners.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list available experiments and their runners, then exit",
    )
    parser.add_argument("experiment", nargs="?", help="experiment slug, e.g. exp-001")
    parser.add_argument("runner", nargs="?", help="runner slug, e.g. cliff-fine-sweep")
    parser.add_argument(
        "--model",
        default=None,
        help="HuggingFace model id; sets OMB_MODEL / OMB_V0_MODEL for runners "
        "that select a model per process",
    )
    args = parser.parse_args(argv)

    if args.list:
        _print_listing()
        return

    if not args.experiment or not args.runner:
        parser.error("an experiment and a runner are required (or use --list)")

    if args.experiment not in _EXPERIMENTS:
        parser.error(
            f"unknown experiment '{args.experiment}'. Known: {', '.join(sorted(_EXPERIMENTS))}"
        )

    available = _runners_for(args.experiment)
    if args.runner not in available:
        parser.error(
            f"unknown runner '{args.runner}' for {args.experiment}. "
            f"Available: {', '.join(available)}"
        )

    # A `--model` flag is the friendly front for the per-process model-selection
    # env vars the runners read (the yzamari state-leak workaround).
    if args.model:
        os.environ["OMB_MODEL"] = args.model
        os.environ["OMB_V0_MODEL"] = args.model

    module_path = f"{_EXPERIMENTS[args.experiment]}.{_slug_to_module(args.runner)}"
    module = importlib.import_module(module_path)
    runner_main = getattr(module, "main", None)
    if not callable(runner_main):
        parser.error(f"runner '{module_path}' has no callable main()")
    runner_main()


if __name__ == "__main__":
    main(sys.argv[1:])
