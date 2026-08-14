"""Trial records and the append-only writer that preserves them.

A record is raw observation. Nothing in this module scores, ranks, or decides
whether a run is admissible; that happens later, from these records.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .arms import StepKind, Tier

SCHEMA_VERSION = "agent-routing-trial-v1"


@dataclass(frozen=True)
class InferenceCall:
    """One model invocation inside one task attempt.

    `cached_input_tokens` is the portion of `input_tokens` the provider reported
    as served from a prefix cache. Providers that do not report it leave it
    None, which keeps cache reuse unmeasured rather than silently zero.
    """

    step_kind: StepKind
    tier: Tier
    model: str
    input_tokens: int
    output_tokens: int
    ttft_ms: float
    latency_ms: float
    cached_input_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.input_tokens < 0 or self.output_tokens < 0:
            raise ValueError("token counts cannot be negative")
        if self.ttft_ms < 0 or self.latency_ms < 0:
            raise ValueError("latencies cannot be negative")
        if self.ttft_ms > self.latency_ms:
            raise ValueError("time to first token cannot exceed total call latency")
        if self.cached_input_tokens is not None and not (
            0 <= self.cached_input_tokens <= self.input_tokens
        ):
            raise ValueError("cached input tokens must fall within input tokens")


@dataclass(frozen=True)
class Attempt:
    """One end-to-end run of one task under one arm.

    A task may take several attempts before it succeeds. Every attempt costs
    money and time, including the ones that failed, which is why cost is
    aggregated over attempts and divided by successes.
    """

    attempt_index: int
    succeeded: bool
    wall_clock_ms: float
    calls: list[InferenceCall]
    tool_ms: float = 0.0
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if self.attempt_index < 1:
            raise ValueError("attempt_index is 1-based")
        if not self.calls:
            raise ValueError("an attempt with no inference calls is not an observation")
        if self.succeeded and self.failure_reason:
            raise ValueError("a successful attempt cannot carry a failure reason")


@dataclass(frozen=True)
class TrialRecord:
    """Every attempt at one task under one arm, with the identities to match it.

    `pair_key` is what makes two arms comparable: same task, same seed, same
    tool environment. Aggregation refuses to compare arms whose pair keys do
    not line up.
    """

    run_id: str
    arm: str
    task_id: str
    seed: int
    pair_key: str
    attempts: list[Attempt]
    endpoint: str
    recorded_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.attempts:
            raise ValueError("a trial with no attempts is not an observation")
        indices = [attempt.attempt_index for attempt in self.attempts]
        if indices != list(range(1, len(indices) + 1)):
            raise ValueError(f"attempt indices must be 1..n in order, got {indices}")
        terminal = self.attempts[-1]
        if any(attempt.succeeded for attempt in self.attempts[:-1]):
            raise ValueError("a trial cannot continue after a successful attempt")
        del terminal

    @property
    def succeeded(self) -> bool:
        return self.attempts[-1].succeeded

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"), sort_keys=True, default=_encode)


def _encode(value: object) -> str:
    if isinstance(value, (StepKind, Tier)):
        return value.value
    raise TypeError(f"cannot serialize {type(value).__name__}")


class TrialWriter:
    """Append-only JSONL writer for one run root.

    Refuses to write into a root that already holds a different run, so a
    second run cannot overwrite the evidence of the first.
    """

    def __init__(self, root: Path, run_id: str) -> None:
        self._root = Path(root)
        self._run_id = run_id
        self._path = self._root / "trials.jsonl"
        self._root.mkdir(parents=True, exist_ok=True)
        marker = self._root / "RUN_ID"
        if marker.exists():
            existing = marker.read_text().strip()
            if existing != run_id:
                raise FileExistsError(
                    f"{self._root} already holds run {existing!r}; "
                    f"choose a fresh root for run {run_id!r}"
                )
        else:
            marker.write_text(run_id + "\n")

    @property
    def path(self) -> Path:
        return self._path

    def append(self, record: TrialRecord) -> None:
        if record.run_id != self._run_id:
            raise ValueError(
                f"record belongs to run {record.run_id!r}, writer owns {self._run_id!r}"
            )
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(record.to_json() + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def read_trials(path: Path) -> list[dict]:
    """Read raw trial dicts. Deliberately returns dicts, not typed records.

    Analysis reads what was written, not what the current dataclasses happen to
    accept, so a schema change cannot silently reinterpret old evidence.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]
