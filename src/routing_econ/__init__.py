"""Measure whether routing agent steps across model tiers pays for itself."""

from .arms import ALL_LARGE, ALL_SMALL, ARMS, INVERTED, ROUTED, Arm, StepKind, Tier, arm
from .metrics import (
    ArmSummary,
    MissingPriceError,
    ModelPrice,
    cache_reuse_rate,
    inference_calls_per_task,
    model_time_share,
    percentile,
    summarize_arm,
    task_latencies_ms,
    token_totals,
    trial_cost,
    ttft_ms,
    unmatched_pairs,
)
from .records import Attempt, InferenceCall, TrialRecord, TrialWriter, read_trials
from .tasks import ManifestTaskSource, Task, TaskSource, pair_key

__all__ = [
    "ALL_LARGE",
    "ALL_SMALL",
    "ARMS",
    "INVERTED",
    "ROUTED",
    "Arm",
    "ArmSummary",
    "Attempt",
    "InferenceCall",
    "ManifestTaskSource",
    "MissingPriceError",
    "ModelPrice",
    "StepKind",
    "Task",
    "TaskSource",
    "Tier",
    "TrialRecord",
    "TrialWriter",
    "arm",
    "cache_reuse_rate",
    "inference_calls_per_task",
    "model_time_share",
    "pair_key",
    "percentile",
    "read_trials",
    "summarize_arm",
    "task_latencies_ms",
    "token_totals",
    "trial_cost",
    "ttft_ms",
    "unmatched_pairs",
]
