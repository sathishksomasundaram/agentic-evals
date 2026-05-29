"""Smoke test for the package.

Rule 4: one positive case + one negative case per unit. This is the scaffold
smoke test — verifies the package imports and pytest is wired correctly.
Replaces itself with real tests as actual code lands.
"""

import pytest


def test_package_imports_and_exposes_version() -> None:
    """Positive: the package imports cleanly and exposes a __version__ string."""
    import agentic_evals

    assert isinstance(agentic_evals.__version__, str)
    assert agentic_evals.__version__  # non-empty


def test_unknown_attribute_raises_attribute_error() -> None:
    """Negative: accessing an undefined attribute raises AttributeError, not None."""
    import agentic_evals

    with pytest.raises(AttributeError):
        _ = agentic_evals.this_attribute_does_not_exist
