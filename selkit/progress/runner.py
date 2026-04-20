from __future__ import annotations

from typing import IO, Optional

from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn


class ProgressReporter:
    def __init__(self, models: tuple[str, ...], stream: Optional[IO[str]] = None) -> None:
        console = Console(file=stream, force_terminal=False) if stream else Console()
        self._progress = Progress(
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            TextColumn("{task.fields[status]}"),
            console=console,
            transient=False,
        )
        self._progress.start()
        self._tasks = {
            name: self._progress.add_task(name, total=1, status="queued")
            for name in models
        }

    def __call__(self, event: str, model: str) -> None:
        if model not in self._tasks:
            return
        tid = self._tasks[model]
        if event == "start":
            self._progress.update(tid, status="fitting", completed=0)
        elif event == "done":
            self._progress.update(tid, status="done", completed=1)

    def close(self) -> None:
        self._progress.stop()
