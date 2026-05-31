"""Thinking-aware text handling for the Qwen3.6 reasoning model.

The model emits a chain-of-thought wrapped in ``<think> … </think>`` before its
real answer. The answer's code fence is what we grade, and the thinking block can
itself contain draft code fences — so we must drop everything up to and including
the final ``</think>`` before extracting code. Kept as a tiny pure-Python module
(no model imports) so it is unit-testable without loading MLX.
"""

from __future__ import annotations

import re

_THINK_CLOSE = re.compile(r"</think\s*>", re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Return the answer after the model's reasoning block.

    If a ``</think>`` marker is present, return everything after the *last* one;
    otherwise return the text unchanged (the model may not have emitted thinking).
    """
    matches = list(_THINK_CLOSE.finditer(text))
    if not matches:
        return text
    return text[matches[-1].end() :]


def had_thinking(text: str) -> bool:
    """True if the text contains a closed ``</think>`` block."""
    return _THINK_CLOSE.search(text) is not None
