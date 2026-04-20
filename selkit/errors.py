from __future__ import annotations


class SelkitError(Exception):
    """Base class for all selkit exceptions."""


class SelkitInputError(SelkitError):
    """Malformed input files (alignment, tree, labels)."""


class SelkitConfigError(SelkitError):
    """Invalid configuration (bad model name, conflicting flags)."""


class SelkitEngineError(SelkitError):
    """Numerical failure inside the ML engine."""


class SelkitConvergenceWarning(Warning):
    """Multi-start optimization lnL values disagreed past tolerance."""
