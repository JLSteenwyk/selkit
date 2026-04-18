from __future__ import annotations

import pytest

from selkit import errors


def test_selkit_error_is_exception() -> None:
    assert issubclass(errors.SelkitError, Exception)


def test_input_error_inherits_from_selkit_error() -> None:
    assert issubclass(errors.SelkitInputError, errors.SelkitError)


def test_convergence_warning_is_warning() -> None:
    assert issubclass(errors.SelkitConvergenceWarning, Warning)


def test_input_error_preserves_message() -> None:
    with pytest.raises(errors.SelkitInputError, match="bad file"):
        raise errors.SelkitInputError("bad file")
