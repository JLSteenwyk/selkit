from __future__ import annotations

from pathlib import Path

import pytest

_CORPUS_ROOT = Path(__file__).parent / "corpus"


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "paml_case" in metafunc.fixturenames:
        cases = [d for d in _CORPUS_ROOT.iterdir() if d.is_dir()]
        metafunc.parametrize("paml_case", cases, ids=[c.name for c in cases])
