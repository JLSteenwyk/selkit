from __future__ import annotations

import io

from selkit.progress.runner import ProgressReporter


def test_progress_reporter_marks_all_models_done() -> None:
    buf = io.StringIO()
    reporter = ProgressReporter(models=("M0", "M1a"), stream=buf)
    reporter("start", "M0")
    reporter("done", "M0")
    reporter("start", "M1a")
    reporter("done", "M1a")
    reporter.close()
    out = buf.getvalue()
    assert "M0" in out and "M1a" in out


def test_reporter_is_callable_matching_service_signature() -> None:
    reporter = ProgressReporter(models=("M0",), stream=io.StringIO())
    reporter("start", "M0")
    reporter("done", "M0")
    reporter.close()
