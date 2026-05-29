"""Runtime primitives: load a model, probe its architecture, build a compressed
KV cache, size a buffer by ratio, and measure a single generation.

All generic and reusable — no experiment-specific knowledge lives here.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

import mlx.core as mx
from mlx_lm import load as _mlx_load
from mlx_turboquant import (  # type: ignore[import-untyped]
    make_turboquant_cache,
    patch_model,
)


def load_model(model_id: str, *, patch_for_tq: bool = True) -> tuple[Any, Any]:
    """Load an MLX model + tokenizer.

    When `patch_for_tq` is True (default), install yzamari's TurboQuant-aware
    attention via `patch_model`. This is harmless for baseline (no-TQ) runs and
    required for TurboQuant caches to work, so we patch by default.
    """
    model, tokenizer = _mlx_load(model_id)  # type: ignore[misc]
    if patch_for_tq:
        model = patch_model(model)
    return model, tokenizer


def probe_arch(model: Any) -> tuple[int, int]:
    """Return (head_dim, num_layers) by best-effort inspection of a loaded model.

    Handles both top-level `.layers` and `.model.layers` arrangements (mlx-lm
    structure differs by model family).
    """
    candidates_layers = [
        getattr(model, "layers", None),
        getattr(getattr(model, "model", None), "layers", None),
    ]
    layers = next((c for c in candidates_layers if c is not None), None)
    num_layers = len(layers) if layers is not None else -1

    args = getattr(model, "args", None) or getattr(model, "config", None)
    head_dim = getattr(args, "head_dim", None) if args is not None else None
    if head_dim is None and args is not None:
        hidden = getattr(args, "hidden_size", None)
        heads = getattr(args, "num_attention_heads", None)
        if hidden and heads:
            head_dim = hidden // heads
    return int(head_dim or 128), int(num_layers or -1)


def make_tq_cache(
    model: Any, *, key_bits: int = 3, value_bits: int = 2, buffer_size: int = 128
) -> list[Any]:
    """Build a TurboQuant KV cache (yzamari) for `model`. Defaults to K3/V2."""
    cache: list[Any] = make_turboquant_cache(
        model, key_bits=key_bits, value_bits=value_bits, buffer_size=buffer_size
    )
    return cache


def buffer_for_ratio(ratio: float, prompt_tokens: int, *, min_buffer: int = 128) -> int:
    """Size a `buffer_size` from a ratio of prompt length.

    Codifies the v0 rule (RECOMMENDATIONS.md): `max(min_buffer, ceil(ratio * n))`.
    The empirically-safe default ratio for tool-calling on the models tested is
    ~0.55; below ~0.50 quality falls off a cliff.
    """
    return max(min_buffer, math.ceil(ratio * prompt_tokens))


def measure(label: str, fn: Callable[[], str]) -> dict[str, Any]:
    """Time + peak-GPU-memory-measure a single zero-arg callable returning a string."""
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    output = fn()
    elapsed = time.perf_counter() - t0
    peak_gb = mx.get_peak_memory() / 1e9
    return {
        "label": label,
        "time_s": round(elapsed, 3),
        "peak_mem_gb": round(peak_gb, 3),
        "output": output,
    }
