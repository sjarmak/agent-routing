"""Task sources.

No benchmark task content is vendored here. A task set is supplied by the
operator through a manifest, so the harness can point at a public suite, a
suite you are licensed to redistribute, or your own authored tasks without any
of them being baked into this repository.

The manifest shape deliberately matches the fields a CodeScaleBench suite
carries, so an existing suite file can be adapted with a projection rather
than a rewrite.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol


@dataclass(frozen=True)
class Task:
    """One unit of work an arm is measured on.

    `verifier` names the check that decides success. A task without one cannot
    contribute to cost per successful task, because success is undefined.
    """

    task_id: str
    prompt: str
    verifier: str
    work_type: str | None = None
    complexity: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("task_id", "prompt", "verifier"):
            if not getattr(self, field_name):
                raise ValueError(f"task is missing {field_name}")


class TaskSource(Protocol):
    def __iter__(self) -> Iterator[Task]: ...


@dataclass(frozen=True)
class ManifestTaskSource:
    """Reads tasks from a JSON manifest: {"tasks": [{...}, ...]}."""

    path: Path

    def __iter__(self) -> Iterator[Task]:
        payload = json.loads(Path(self.path).read_text(encoding="utf-8"))
        try:
            entries = payload["tasks"]
        except (KeyError, TypeError):
            raise ValueError(f"{self.path} has no top-level 'tasks' array") from None
        seen: set[str] = set()
        for entry in entries:
            task = Task(
                task_id=entry["task_id"],
                prompt=entry["prompt"],
                verifier=entry["verifier"],
                work_type=entry.get("work_type"),
                complexity=entry.get("complexity"),
            )
            if task.task_id in seen:
                raise ValueError(f"duplicate task_id {task.task_id!r} in {self.path}")
            seen.add(task.task_id)
            yield task


def pair_key(task: Task, seed: int, tool_env: str) -> str:
    """The identity two arms must share to be compared on this task."""
    return f"{task.task_id}|seed={seed}|env={tool_env}"
